/* presk_rt.h: what the kernels and the host of Presk's C export need (plan
 * 0.9.0 §3.2-§3.5): the bf16 conversions and the saturating conversion to an
 * integer, inline; the transcendental functions (presk_libm.c); the
 * conversion of a buffer between machine types, the .psk.d reader and
 * writer, integer weights and their quantization, the arena, and the keys
 * `sample` draws from (presk_rt.c).
 *
 * Build with  cc -std=c11 -O2 -ffp-contract=off  and never with -ffast-math
 * or any of its parts: the kernels compute the bits of the native backend
 * only when every product is rounded before it is added (§3.5, rule 1).
 * GCC contracts `a + b*c` into a fused multiply-add in its GNU modes (its
 * default) and not under -std=c11; -ffp-contract=off holds in both. GCC
 * ignores the standard pragma below and warns about it under -Wall, so it is
 * given to Clang only. */
#ifndef PRESK_RT_H
#define PRESK_RT_H

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>

/* The team of a launch divided among p workers (plan 0.9.0 §3.6, presk D81):
 * p, the export's, at most the threads OpenMP would use, which
 * OMP_NUM_THREADS sets (the cores by default). A num_threads(p) clause
 * alone would override OMP_NUM_THREADS, and OMP_NUM_THREADS=1 would not
 * give one thread. The pieces are the team's, whatever its size, and the
 * bits do not depend on it (§3.5, rule 4). */
static inline int presk_team(int p)
{
    int m = omp_get_max_threads();
    return p < m ? p : m;
}
#endif

/* PRESK_CONST: a function of its arguments alone, which reads and writes no
 * memory but its own locals (GCC's and Clang's `const`). The conversions
 * below are. The SCoP mode's nests (plan 0.9.0 §3.7, presk D85) call them,
 * and a polyhedral tool that runs before inlining (Polly at its early
 * position) takes a call it knows nothing of for a side effect. */
#if defined(__GNUC__)
#define PRESK_CONST __attribute__((const))
#else
#define PRESK_CONST
#endif

/* bf16 is a storage type: a buffer holds the top sixteen bits of an f32, the
 * kernel computes in f32, and only loads and stores convert. */

/* The f32 a stored bf16 denotes. Exact: bf16 is a prefix of f32. */
static inline PRESK_CONST float presk_bf16_to_f32(uint16_t h)
{
    uint32_t w = (uint32_t)h << 16;
    float x;
    memcpy(&x, &w, sizeof x);
    return x;
}

/* An f32 rounded to the nearest bf16, ties to even; a NaN stays a NaN, made
 * quiet. The function of presk_rt::f32_to_bf16, which the kernel interpreter
 * and the Cranelift backend share. */
static inline PRESK_CONST uint16_t presk_f32_to_bf16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    if (x != x)
        return (uint16_t)((bits >> 16) | 0x0040u);
    uint32_t lsb = (bits >> 16) & 1u;
    return (uint16_t)((bits + 0x7fffu + lsb) >> 16);
}

/* A real converted to an integer, rounded toward zero, saturating, and 0 for
 * a NaN: Cranelift's fcvt_to_sint_sat, where C's own conversion is undefined
 * out of range. */
static inline PRESK_CONST int64_t presk_f64_to_i64(double x)
{
    if (x != x)
        return 0;
    if (x >= 9223372036854775808.0)
        return INT64_MAX;
    if (x < -9223372036854775808.0)
        return INT64_MIN;
    return (int64_t)x;
}

static inline PRESK_CONST int64_t presk_f32_to_i64(float x)
{
    return presk_f64_to_i64((double)x);
}

/* An f64 rounded to bf16 as the native backend rounds it: to f32, then to
 * bf16 (presk_rt::Buf::of). Two roundings, which a direct rounding to bf16
 * differs from when the first lands on a tie. */
static inline PRESK_CONST uint16_t presk_f64_to_bf16(double x)
{
    return presk_f32_to_bf16((float)x);
}

/* The transcendental functions the kernels call: musl's, from before musl
 * 1.1.23, which the Rust `libm` crate ports and the native backend calls
 * (§3.5, rule 2; D78). Their definitions are presk_libm.c's. */
double presk_exp(double x);
double presk_log(double x);
double presk_tanh(double x);
double presk_sin(double x);
double presk_cos(double x);
double presk_erf(double x);
float presk_expf(float x);
float presk_logf(float x);
float presk_tanhf(float x);
float presk_sinf(float x);
float presk_cosf(float x);
float presk_erff(float x);

/* ------------------------------------------------------------------------
 * The runtime (presk_rt.c)
 * ------------------------------------------------------------------------ */

/* The machine type of a buffer: the reals of a kernel, or indices. */
typedef enum { PRESK_F64, PRESK_F32, PRESK_BF16, PRESK_I64 } presk_dtype;

/* Bytes of one element. */
size_t presk_dtype_size(presk_dtype d);

/* n reals of type `from` at src, stored as type `to` at dst, as the native
 * backend converts a buffer at a kernel's edge (Buf::of(to, b.to_f64())):
 * through f64, where widening is exact, f64 to f32 rounds to nearest, and to
 * bf16 rounds to f32 and then to bf16. Indices are not converted. */
void presk_convert(void *dst, presk_dtype to, const void *src, presk_dtype from, size_t n);

/* n elements of type `to` at dst, each the real x rounded to it as the
 * native backend rounds it (Buf::of(to, [x; n])): a fill (zeros_like, a
 * dropout mask outside training) or a host scalar made a buffer (plan 0.9.0
 * step 3). */
void presk_fill(void *dst, presk_dtype to, double x, size_t n);

/* The last error of the reader, the writer or the arena, as text. */
const char *presk_error(void);

/* A leaf of a .psk.d sample: n elements of an idx[bound], or of a real
 * tensor when bound is 0. The C does not parse Presk types: the export
 * states the type as the file's header writes it, and its leaves in order
 * (§3.4). */
typedef struct {
    uint64_t bound;
    uint64_t n;
} presk_leaf;

/* A .psk.d file open for reading, its header checked. The functions below
 * return 0, or the code of presk-data's error, whose text presk_error()
 * gives: 9002 not a .psk.d file of version 1 or 2, 9003 truncated, 9004 an
 * index not below its bound, 9005 another type, or a length that is not the
 * layout's (a model of other dimensions). */
typedef struct {
    FILE *f;
    const presk_leaf *leaves;
    size_t nleaves;
    uint64_t count;     /* samples in the file */
    uint64_t next;      /* the sample presk_read reads */
    unsigned char *enc; /* version 2: each leaf's encoding, 0 f64, 1 f32,
                           2 bf16 (plan 0.17.0 step 7); NULL in version 1 */
} presk_reader;

/* Opens `path` and checks its header: magic, version 1 or 2, the sample
 * type `type` byte for byte, a version 2 file's encodings, one per leaf and
 * none of an index, and a length of exactly `count` samples of the leaves
 * in their encodings. */
int presk_open(presk_reader *r, const char *path, const char *type, const presk_leaf *leaves, size_t nleaves);

/* Reads the next sample: leaf i into dst[i], as dtypes[i]. A real leaf is
 * read in its encoding, f64 unless a version 2 file says f32 or bf16, each a
 * number of f64 exactly, and converted to dtypes[i] (presk_convert);
 * an index leaf is read into int64_t (PRESK_I64), each checked below its
 * bound. */
int presk_read(presk_reader *r, void *const *dst, const presk_dtype *dtypes);

void presk_close(presk_reader *r);

/* Opens a file of one sample, reads it and closes it: a checkpoint, or an
 * entry's inputs. */
int presk_load(const char *path, const char *type, const presk_leaf *leaves, size_t nleaves, void *const *dst, const presk_dtype *dtypes);

/* Writes `count` samples of the leaves as a .psk.d file of sample type
 * `type`, as presk-data writes one: leaf i of sample s from
 * src[s * nleaves + i], of type dtypes[i]. Reals are written as f64 (exact),
 * indices in the width their bound gives, each checked below it. */
int presk_write(const char *path, const char *type, const presk_leaf *leaves, size_t nleaves, uint64_t count, const void *const *src, const presk_dtype *dtypes);

/* The arena: one block, sized at export time from the recording's peak of
 * live bytes, from which every buffer is bump-allocated (§3.3). Nothing is
 * allocated from the system during a call of the entry. */
typedef struct {
    unsigned char *base;
    size_t size; /* bytes */
    size_t top;  /* bytes in use */
    size_t peak; /* the most bytes in use so far */
} presk_arena;

/* Every allocation is aligned to this many bytes, a cache line. */
#define PRESK_ALIGN 64

/* Allocates the block: 0, or 1 when the system has not the memory. */
int presk_arena_init(presk_arena *a, size_t size);
void presk_arena_free(presk_arena *a);

/* `bytes` bytes, aligned, or NULL when the arena has not them left. */
void *presk_try_alloc(presk_arena *a, size_t bytes);

/* The same, and the program stops with a message when the arena has not
 * them left, which is a fault of the export that sized it. */
void *presk_alloc(presk_arena *a, size_t bytes);

/* Where the arena is: what presk_release returns to. */
size_t presk_mark(const presk_arena *a);

/* Frees everything allocated after `mark`: a function's temporaries, when
 * it returns. */
void presk_release(presk_arena *a, size_t mark);

/* Frees everything allocated after `mark` but the `bytes` at `keep`, which
 * are moved to `mark` and returned: a function's result, kept when its
 * temporaries go. */
void *presk_release_keep(presk_arena *a, size_t mark, const void *keep, size_t bytes);

/* ------------------------------------------------------------------------
 * Integer weights (plan 0.9.0 step 7, presk D84)
 * ------------------------------------------------------------------------ */

/* A parameter the realization quantises (presk D44): integers, and one f32
 * scale per channel. The scale is constant along one axis, the one the
 * parameter is contracted over; split at that axis the tensor is
 * outer x k x inner, row-major, and element e has the scale of its channel,
 * s[(e / (k * inner)) * inner + e % inner]. Element e denotes f32(q_e) * s,
 * one f32 multiplication, which a kernel computes where it loads it, and
 * widens exactly when it computes in f64. The integers are made from the
 * checkpoint's reals each time they are read (presk_quantize): the scheme is
 * the realization's, the scales are data (presk question 57). */

/* int8: one integer per byte, in [-127, 127]. */
typedef struct {
    const int8_t *q;
    const float *s;
} presk_q8;

/* int4: two integers per byte, element 2i in the low nibble of byte i and
 * element 2i + 1 in the high one, each in two's complement, in [-7, 7]. */
typedef struct {
    const uint8_t *q;
    const float *s;
} presk_q4;

/* Integer e of an int4 buffer: its nibble, sign-extended. */
static inline int presk_int4(const uint8_t *q, int64_t e)
{
    unsigned b = q[e / 2];
    unsigned n = e % 2 ? b >> 4 : b & 0x0fu;
    return n >= 8 ? (int)n - 16 : (int)n;
}

/* How a channel's scale is chosen from its largest magnitude m (presk-rt's
 * QuantScheme): m / qmax rounded to f32 (absmax); or the smallest power of
 * two s with m / s <= qmax, then f32 (pow2), under which f32(q) * s is
 * exact. qmax is 2^(bits-1) - 1; an all-zero channel has the scale 0. */
typedef enum { PRESK_ABSMAX, PRESK_POW2 } presk_scheme;

/* The reals w of a tensor of outer x k x inner elements, quantised to
 * `bits` (8 or 4) as presk-rt's QTensor::quantize quantises the checkpoint's
 * f64 reals, bit for bit: per channel m = max |w| over its k elements (a NaN
 * passed over), its scale s by the scheme; then q = w / s rounded to the
 * nearest integer, ties to even, and clamped to [-qmax, qmax], or 0 when s
 * is 0 or w / s is a NaN. The integers go to q (outer * k * inner bytes for
 * int8, half as many rounded up for int4), the scales to s (outer * inner). */
void presk_quantize(void *q, float *s, const double *w, uint64_t outer, uint64_t k, uint64_t inner, int bits, presk_scheme scheme);

/* n elements of integer weights stored as type `to` at dst, as the native
 * backend converts them for a kernel that reads reals (Buf::of(to,
 * b.to_f64())): each f32(q_e) * s, then to `to`. q and s are a tensor of
 * `bits` whose channels are k x inner elements from q, and the elements are
 * from..from+n of it. */
void presk_dequantize(void *dst, presk_dtype to, const void *q, const float *s, int bits, uint64_t k, uint64_t inner, uint64_t from, size_t n);

/* ------------------------------------------------------------------------
 * Keys and their draws (plan 0.9.0 step 6, presk D83; plan 0.10.0 step 7b,
 * D99)
 * ------------------------------------------------------------------------ */

/* The generator of Presk's keys: rand_chacha 0.9's ChaCha20Rng, which the
 * native backend and the interpreter use (the reference's definition "The
 * generator"). Bernstein's ChaCha20, twenty rounds: words 0-3 the constant,
 * 4-11 the key, 12-13 a 64-bit block counter and 14-15 a 64-bit stream
 * number, 0; not RFC 7539's 32-bit counter and 96-bit nonce. As a rand_core
 * BlockRng it computes four blocks at a time, 64 words, and reads them in
 * order, a u64 as two words, the low one first. */
typedef struct {
    uint32_t key[8];
    uint64_t counter;     /* the block the next refill starts at */
    uint32_t results[64]; /* four blocks */
    unsigned index;       /* the next word of results; 64 when used up */
} presk_rng;

/* ChaCha20Rng::from_seed: the key is the 32 bytes, as little-endian words. */
void presk_rng_from_seed(presk_rng *r, const uint8_t seed[32]);

/* rand_core's SeedableRng::seed_from_u64, which rand_chacha keeps: eight
 * outputs of PCG32 from `state` fill the seed, each little-endian. */
void presk_rng_seed_u64(presk_rng *r, uint64_t state);

/* The next word, and the next two (BlockRng's next_u32 and next_u64). */
uint32_t presk_rng_next_u32(presk_rng *r);
uint64_t presk_rng_next_u64(presk_rng *r);

/* A real in [0, 1): the top 53 bits of next_u64 over 2^53 (presk-interp's
 * uniform01): the uniform draws of a key, in order. */
double presk_uniform01(presk_rng *r);

/* A key: 32 bytes (presk-interp's Key). Its keystream is the bytes of the
 * generator seeded with it, a word at a time, the low byte first. */
typedef struct {
    uint8_t b[32];
} presk_key;

/* The key a seed names (Key::from_seed, `presk run --seed`): the first 32
 * bytes of the keystream of seed_from_u64(seed). */
void presk_key_from_seed(presk_key *out, uint64_t seed);

/* Key l of split_key n k, for any n > l (Key::split): the bytes 32l to
 * 32l + 31 of k's keystream. */
void presk_key_split(const presk_key *k, uint64_t l, presk_key *out);

/* The child t of k (Key::child): split 0 of the key whose first eight bytes
 * are k's xored with t, the low byte first. A training step and a token of
 * a serve each take a child of the run's key. */
void presk_key_child(const presk_key *k, uint64_t t, presk_key *out);

/* `sample(key, logits)` (presk-kernel's host): `rows` rows of `vocab` logits
 * of type `from`, a token each into out. Per row, in f64: m the largest
 * logit, e_i = presk_exp(x_i - m), z their sum in order, u the key's uniform
 * draw of the row's rank (the rows in order, one draw each), and the token
 * the first i with u < e_0/z + ... + e_i/z, summed in order; the last if
 * none. */
void presk_sample(int64_t *out, const void *logits, presk_dtype from, size_t rows, size_t vocab, const presk_key *key);

#endif
