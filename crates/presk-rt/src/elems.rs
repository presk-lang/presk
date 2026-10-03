//! A tensor's elements, shared (plan 0.10.0 step 6, D96).
//!
//! The native host reads its values out of an environment, passes them to
//! calls, folds and kernels, and puts them in tuples and records. Until step
//! 6 a buffer owned its elements (`Buf::F32(Vec<f32>)`), so each of these
//! copied them: 6.2 GB a call of the course GPT's forward pass, 72.9 GB a
//! call of its gradient (D95). An [`Elems`] holds the elements behind a
//! reference count: a clone is a reference, and the elements are freed with
//! the last one.
//!
//! **Views.** An `Elems` is a range of an allocation's elements: a stacked
//! layer's slice is the range of its layer ([`Elems::view`]), as the C
//! program passes `w + l · chunk`. A stack can be its parts laid end to end
//! ([`Elems::stack`]): a scan's results, a slice set into a stack, a
//! gradient scattered into one. A range within one part is a range of that
//! part, so a fold that slices a stack its scan made, layer by layer,
//! copies nothing; the elements are laid out in one allocation only when
//! something reads a range across parts, once.
//!
//! **Exact.** Nothing writes elements another value can see. A kernel
//! writes only its output, a buffer allocated for the launch; the host
//! builds every other buffer anew or from parts of others. And a write
//! through [`DerefMut`] to elements another value can see copies them
//! first, so a value never changes under a reader.
//!
//! **Reused.** An allocation of 64 KB or more goes, when its last value
//! drops, to a free list by its length ([`FreeList`]), from which a kernel's
//! next output of that length is taken ([`Elems::fresh`]): every call of an
//! entry allocates the same lengths, so after the first no page is touched
//! for the first time, as the C program's arena reuses its memory. The list
//! keeps at most as many bytes as were ever held by live buffers at once,
//! so the process holds at most twice its peak. A buffer taken from it is
//! not zeroed: a kernel writes every element of its output before reading
//! any, as the C program's never-zeroed arena shows (C1); a debug build
//! fills it with NaN, or the least integer, so that the tests would see a
//! kernel that did not.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock};

/// The element types of a machine buffer, which a free list keeps.
pub trait Pooled: Copy + Default + Send + Sync + 'static {
    /// What a debug build fills a buffer handed out for writing with.
    const POISON: Self;
    fn free_list() -> &'static FreeList<Self>;
}

macro_rules! pooled {
    ($t:ty, $poison:expr) => {
        impl Pooled for $t {
            const POISON: $t = $poison;
            fn free_list() -> &'static FreeList<$t> {
                static LIST: FreeList<$t> = FreeList::new();
                &LIST
            }
        }
    };
}
pooled!(f64, f64::NAN);
pooled!(f32, f32::NAN);
pooled!(u16, 0x7fc1);
pooled!(i64, i64::MIN);

/// Allocations of at least this many bytes are kept for reuse.
const POOLED_BYTES: usize = 64 << 10;

/// Buffers of one element type kept for reuse, by length (plan 0.10.0 step
/// 6).
pub struct FreeList<T> {
    lists: Mutex<Lists<T>>,
}

struct Lists<T> {
    by_len: BTreeMap<usize, Vec<Vec<T>>>,
    /// Elements kept, held by live buffers, and the most ever held at once.
    kept: usize,
    live: usize,
    peak: usize,
}

impl<T> FreeList<T> {
    const fn new() -> Self {
        FreeList { lists: Mutex::new(Lists { by_len: BTreeMap::new(), kept: 0, live: 0, peak: 0 }) }
    }

    fn pooled(n: usize) -> bool {
        n * std::mem::size_of::<T>() >= POOLED_BYTES
    }

    fn lists(&self) -> std::sync::MutexGuard<'_, Lists<T>> {
        self.lists.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A kept buffer of `n` elements, if any.
    fn take(&self, n: usize) -> Option<Vec<T>> {
        let mut l = self.lists();
        let v = l.by_len.get_mut(&n)?.pop()?;
        l.kept -= n;
        Some(v)
    }

    /// A live buffer of `n` elements made.
    fn made(&self, n: usize) {
        let mut l = self.lists();
        l.live += n;
        l.peak = l.peak.max(l.live);
    }

    /// A live buffer dropped: kept, unless the list would hold more than the
    /// peak.
    fn dropped(&self, v: Vec<T>) {
        let n = v.len();
        let mut l = self.lists();
        l.live -= n;
        if l.kept + n <= l.peak {
            l.kept += n;
            l.by_len.entry(n).or_default().push(v);
            return;
        }
        drop(l);
        drop(v);
    }

    /// The elements kept, and the most ever held by live buffers at once.
    pub fn held(&self) -> (usize, usize) {
        let l = self.lists();
        (l.kept, l.peak)
    }
}

/// The elements of a buffer: a range of an allocation shared by its clones
/// and views.
pub struct Elems<T: Pooled> {
    repr: Arc<Repr<T>>,
    lo: usize,
    len: usize,
}

enum Repr<T: Pooled> {
    /// One allocation.
    Flat(Vec<T>),
    /// Parts laid end to end, each starting at its offset; laid out in one
    /// allocation when a range across parts is read.
    Parts { parts: Vec<Elems<T>>, starts: Vec<usize>, flat: OnceLock<Vec<T>> },
}

impl<T: Pooled> Drop for Repr<T> {
    fn drop(&mut self) {
        if let Repr::Flat(v) = self
            && FreeList::<T>::pooled(v.len())
        {
            T::free_list().dropped(std::mem::take(v));
        }
    }
}

impl<T: Pooled> Elems<T> {
    /// `n` elements to be written, a kernel's output: a buffer of that
    /// length taken from the free list, whatever it holds, or a new one of
    /// zeros; in a debug build, every element NaN or the least integer.
    pub fn fresh(n: usize) -> Elems<T> {
        let v = if FreeList::<T>::pooled(n) { T::free_list().take(n) } else { None };
        #[allow(unused_mut)]
        let mut v = v.unwrap_or_else(|| vec![T::default(); n]);
        #[cfg(debug_assertions)]
        v.fill(T::POISON);
        Elems::from(v)
    }

    pub fn as_slice(&self) -> &[T] {
        let all: &[T] = match &*self.repr {
            Repr::Flat(v) => v,
            Repr::Parts { parts, flat, .. } => flat.get_or_init(|| {
                let mut v = Vec::with_capacity(parts.iter().map(|p| p.len).sum());
                for p in parts {
                    v.extend_from_slice(p.as_slice());
                }
                v
            }),
        };
        &all[self.lo..self.lo + self.len]
    }

    /// Elements `lo..lo + len` of these, without a copy: a range of the same
    /// allocation, or of the part that holds the whole range.
    pub fn view(&self, lo: usize, len: usize) -> Elems<T> {
        assert!(lo + len <= self.len, "a view of {len} elements at {lo} of {}", self.len);
        let at = self.lo + lo;
        if let Repr::Parts { parts, starts, .. } = &*self.repr {
            // The last part starting at or before `at`; a range of it when
            // the whole range lies in it.
            let k = starts.partition_point(|s| *s <= at).saturating_sub(1);
            if let Some(p) = parts.get(k)
                && at + len <= starts[k] + p.len
            {
                return p.view(at - starts[k], len);
            }
        }
        Elems { repr: Arc::clone(&self.repr), lo: at, len }
    }

    /// The parts laid end to end, without a copy.
    pub fn stack(parts: Vec<Elems<T>>) -> Elems<T> {
        let mut starts = Vec::with_capacity(parts.len());
        let mut len = 0;
        for p in &parts {
            starts.push(len);
            len += p.len;
        }
        Elems { repr: Arc::new(Repr::Parts { parts, starts, flat: OnceLock::new() }), lo: 0, len }
    }
}

impl<T: Pooled> Elems<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether no other value shares these elements, so that dropping this
    /// one frees them.
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.repr) == 1
    }

    /// The allocation holding the elements, as an address, how many values
    /// share it, and the bytes it holds of its own (a stack's parts are
    /// their own allocations): what a profile counts the bytes a drop frees
    /// by.
    pub fn block(&self) -> (usize, usize, usize) {
        let own = match &*self.repr {
            Repr::Flat(v) => v.len(),
            Repr::Parts { flat, .. } => flat.get().map_or(0, |v| v.len()),
        };
        (Arc::as_ptr(&self.repr) as *const u8 as usize, Arc::strong_count(&self.repr), own * std::mem::size_of::<T>())
    }
}

impl<T: Pooled> Clone for Elems<T> {
    fn clone(&self) -> Self {
        Elems { repr: Arc::clone(&self.repr), lo: self.lo, len: self.len }
    }
}

impl<T: Pooled> Deref for Elems<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

/// A write copies the elements first when another value can see them, or
/// when they are a range of a stack or of a larger allocation.
impl<T: Pooled> DerefMut for Elems<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let whole = matches!(&*self.repr, Repr::Flat(v) if self.lo == 0 && self.len == v.len());
        if !(whole && self.is_unique()) {
            *self = Elems::from(self.as_slice().to_vec());
        }
        match Arc::get_mut(&mut self.repr) {
            Some(Repr::Flat(v)) => v,
            _ => unreachable!("elements just made one's own"),
        }
    }
}

impl<T: Pooled> From<Vec<T>> for Elems<T> {
    fn from(v: Vec<T>) -> Self {
        let len = v.len();
        if FreeList::<T>::pooled(len) {
            T::free_list().made(len);
        }
        Elems { repr: Arc::new(Repr::Flat(v)), lo: 0, len }
    }
}

impl<T: Pooled> FromIterator<T> for Elems<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        Elems::from(it.into_iter().collect::<Vec<T>>())
    }
}

impl<'a, T: Pooled> IntoIterator for &'a Elems<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Pooled + PartialEq> PartialEq for Elems<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Pooled + fmt::Debug> fmt::Debug for Elems<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}
