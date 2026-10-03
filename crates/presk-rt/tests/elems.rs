//! A tensor's elements shared, viewed, stacked and reused (plan 0.10.0 step
//! 6, D96): a clone, a view and a stack copy nothing, read the same
//! elements, a write never reaches another value, and an allocation dropped
//! is handed out again for its length.

use presk_rt::{Elems, Pooled};

fn addr<T: Pooled>(e: &Elems<T>) -> usize {
    e.block().0
}

#[test]
fn a_clone_and_a_view_share_the_elements() {
    let a = Elems::from((0..12).map(|x| x as f32).collect::<Vec<_>>());
    let b = a.clone();
    let v = a.view(4, 4);
    assert_eq!(addr(&a), addr(&b));
    assert_eq!(addr(&a), addr(&v));
    assert_eq!(&*v, &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(&*v.view(1, 2), &[5.0, 6.0]);
    assert!(!a.is_unique());
    drop((b, v));
    assert!(a.is_unique());
}

#[test]
fn a_range_within_a_part_is_that_part() {
    let parts: Vec<Elems<f64>> = (0..3).map(|l| Elems::from((0..4).map(|x| (10 * l + x) as f64).collect::<Vec<_>>())).collect();
    let s = Elems::stack(parts.clone());
    assert_eq!(s.len(), 12);
    for (l, p) in parts.iter().enumerate() {
        let v = s.view(4 * l, 4);
        assert_eq!(addr(&v), addr(p), "layer {l} is its part, not a copy");
        assert_eq!(&*v, &**p);
        assert_eq!(addr(&s.view(4 * l + 1, 2)), addr(p));
    }
    // Nothing read the whole: its parts were never laid out.
    assert_eq!(s.block().2, 0);
    // A range across two parts is read from the stack laid out once.
    assert_eq!(&*s.view(2, 4), &[2.0, 3.0, 10.0, 11.0]);
    assert_eq!(s.block().2, 12 * 8);
    assert_eq!(&*s, &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn a_stack_of_views_and_parts_stays_one_level_deep() {
    // A slice set into a stack L times over, as a backward fold writes each
    // layer's cotangent: the result's parts are the values written.
    let (n, chunk) = (4, 3);
    let mut s = Elems::from(vec![0i64; n * chunk]);
    let written: Vec<Elems<i64>> = (0..n).map(|l| Elems::from(vec![l as i64 + 1; chunk])).collect();
    for (i, w) in written.iter().enumerate() {
        s = Elems::stack((0..n).map(|k| if k == i { w.clone() } else { s.view(k * chunk, chunk) }).collect());
    }
    for (l, w) in written.iter().enumerate() {
        assert_eq!(addr(&s.view(l * chunk, chunk)), addr(w));
    }
    assert_eq!(&*s, &[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
}

#[test]
fn a_write_never_reaches_another_value() {
    // Shared: copied first.
    let a = Elems::from(vec![1.0f32, 2.0, 3.0]);
    let mut b = a.clone();
    b[0] = 9.0;
    assert_eq!(&*a, &[1.0, 2.0, 3.0]);
    assert_eq!(&*b, &[9.0, 2.0, 3.0]);
    assert_ne!(addr(&a), addr(&b));
    // Alone and whole: in place.
    let before = addr(&b);
    b[1] = 8.0;
    assert_eq!(addr(&b), before);
    // A view, even alone: copied, the rest of its allocation untouched.
    let c = Elems::from(vec![1u16, 2, 3, 4]);
    let mut v = c.view(1, 2);
    drop(c);
    v[0] = 7;
    assert_eq!(&*v, &[7, 3]);
    // A stack: laid out in an allocation of its own.
    let p = Elems::from(vec![5.0f64; 2]);
    let mut s = Elems::stack(vec![p.clone(), p.clone()]);
    s[3] = 0.0;
    assert_eq!(&*p, &[5.0, 5.0]);
    assert_eq!(&*s, &[5.0, 5.0, 5.0, 0.0]);
}

#[test]
fn a_dropped_allocation_is_handed_out_again() {
    // A length no other test uses: the free list is the process's.
    let n = 123_457;
    let a: Elems<f64> = Elems::fresh(n);
    let at = a.as_ptr();
    // A debug build hands out NaN, so that a kernel reading what it did not
    // write is seen.
    if cfg!(debug_assertions) {
        assert!(a.iter().all(|x| x.is_nan()));
    }
    let view = a.view(10, 10);
    drop(a);
    // Still held by a view: not free.
    let b: Elems<f64> = Elems::fresh(n);
    assert_ne!(b.as_ptr(), at);
    drop(view);
    let c: Elems<f64> = Elems::fresh(n);
    assert_eq!(c.as_ptr(), at, "the elements dropped last are handed out again");
    assert_eq!(c.len(), n);
    let (kept, peak) = f64::free_list().held();
    assert!(peak >= 2 * n && kept <= peak);
    drop((b, c));
}
