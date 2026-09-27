/* presk_rt.c: the runtime of Presk's C export (plan 0.9.0 §3.3, §3.4; presk
 * D78): the conversion of a buffer between machine types, the .psk.d reader
 * and writer, the quantization of integer weights (step 7, presk D84), the
 * arena, and the stream `sample` draws from with `sample` itself (step 6,
 * presk D83). Build it as the kernels are built (presk_rt.h).
 *
 * The .psk.d format is presk-data's (its module comment):
 *
 *   magic     "PSKD"                       4 bytes
 *   version   u16 = 1                      little-endian
 *   type_len  u32, then the type, UTF-8    the sample type in Presk syntax
 *   count     u64                          number of samples
 *   data      the samples, back to back: the leaves of the type in order,
 *             idx[N] as the smallest of u8/u16/u32/u64 holding N-1,
 *             reals as f64 little-endian
 *
 * The file is read and written byte by byte, so that it is little-endian
 * whatever the machine's order. The reader checks the header against what
 * the export states, the type and its leaves, and parses no Presk type. */

#include "presk_rt.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>

/* ------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------ */

static char presk_errbuf[1024];

const char *presk_error(void)
{
    return presk_errbuf;
}

/* Records a data error as presk-data prints one, "[E9002] ...", and returns
 * its code. */
static int fail(int code, const char *fmt, ...)
{
    va_list ap;
    int n = snprintf(presk_errbuf, sizeof presk_errbuf, "[E%d] ", code);
    va_start(ap, fmt);
    vsnprintf(presk_errbuf + n, sizeof presk_errbuf - (size_t)n, fmt, ap);
    va_end(ap);
    return code;
}

/* ------------------------------------------------------------------------
 * Machine types
 * ------------------------------------------------------------------------ */

size_t presk_dtype_size(presk_dtype d)
{
    switch (d) {
    case PRESK_F64:
    case PRESK_I64:
        return 8;
    case PRESK_F32:
        return 4;
    case PRESK_BF16:
        return 2;
    }
    return 0;
}

/* Element i of a real buffer, as f64: exact, as Buf::to_f64. */
static double load_real(const void *src, presk_dtype from, uint64_t i)
{
    switch (from) {
    case PRESK_F64:
        return ((const double *)src)[i];
    case PRESK_F32:
        return (double)((const float *)src)[i];
    case PRESK_BF16:
        return (double)presk_bf16_to_f32(((const uint16_t *)src)[i]);
    case PRESK_I64:
        break;
    }
    abort(); /* an index buffer is not a real one: a fault of the caller */
}

/* Element i of a real buffer set from an f64, rounded as Buf::of rounds. */
static void store_real(void *dst, presk_dtype to, uint64_t i, double x)
{
    switch (to) {
    case PRESK_F64:
        ((double *)dst)[i] = x;
        return;
    case PRESK_F32:
        ((float *)dst)[i] = (float)x;
        return;
    case PRESK_BF16:
        ((uint16_t *)dst)[i] = presk_f64_to_bf16(x);
        return;
    case PRESK_I64:
        break;
    }
    abort();
}

void presk_convert(void *dst, presk_dtype to, const void *src, presk_dtype from, size_t n)
{
    /* A buffer already of the kernel's type is lent as it is, NaNs and all
     * (kernel_input in presk-kernel). */
    if (to == from) {
        memmove(dst, src, n * presk_dtype_size(to));
        return;
    }
    for (size_t i = 0; i < n; i++)
        store_real(dst, to, i, load_real(src, from, i));
}

void presk_fill(void *dst, presk_dtype to, double x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        store_real(dst, to, i, x);
}

/* ------------------------------------------------------------------------
 * .psk.d files
 * ------------------------------------------------------------------------ */

/* Bytes of an idx[bound] element on disk: the smallest unsigned integer
 * holding bound - 1 (presk-data's idx_width). */
static unsigned idx_width(uint64_t bound)
{
    uint64_t m = bound - 1;
    if (m <= 0xffu)
        return 1;
    if (m <= 0xffffu)
        return 2;
    if (m <= 0xffffffffu)
        return 4;
    return 8;
}

static unsigned leaf_width(const presk_leaf *l)
{
    return l->bound ? idx_width(l->bound) : 8;
}

/* Bytes of one sample, or UINT64_MAX when that overflows. */
static uint64_t sample_bytes(const presk_leaf *leaves, size_t nleaves)
{
    uint64_t total = 0;
    for (size_t i = 0; i < nleaves; i++) {
        uint64_t w = leaf_width(&leaves[i]);
        if (leaves[i].n > (UINT64_MAX - 1 - total) / w)
            return UINT64_MAX;
        total += leaves[i].n * w;
    }
    return total;
}

static uint64_t get_le(const unsigned char *b, unsigned w)
{
    uint64_t x = 0;
    for (unsigned i = w; i-- > 0;)
        x = x << 8 | b[i];
    return x;
}

static void put_le(unsigned char *b, uint64_t x, unsigned w)
{
    for (unsigned i = 0; i < w; i++) {
        b[i] = (unsigned char)(x & 0xffu);
        x >>= 8;
    }
}

/* A leaf's type in the program: int64_t for an index, a real type for a
 * real. */
static int check_dtype(const presk_leaf *l, presk_dtype d, size_t i)
{
    if ((l->bound != 0) != (d == PRESK_I64))
        return fail(9005, "leaf %zu: %s, but the program holds it as %s", i, l->bound ? "indices" : "reals", d == PRESK_I64 ? "indices" : "reals");
    return 0;
}

/* Elements are read and written in pieces of this many bytes. */
#define CHUNK 32768

int presk_open(presk_reader *r, const char *path, const char *type, const presk_leaf *leaves, size_t nleaves)
{
    unsigned char b[8];
    char *found;
    size_t tlen = strlen(type);
    uint64_t flen, count, per, need, head;
    long end;
    FILE *f;

    r->f = NULL;
    r->leaves = leaves;
    r->nleaves = nleaves;
    r->count = 0;
    r->next = 0;
    f = fopen(path, "rb");
    if (!f)
        return fail(9002, "%s: %s", path, strerror(errno));
    if (fread(b, 1, 4, f) != 4) {
        fclose(f);
        return fail(9002, "%s: not a .psk.d file (too short)", path);
    }
    if (memcmp(b, "PSKD", 4) != 0) {
        fclose(f);
        return fail(9002, "%s: not a .psk.d file (bad magic)", path);
    }
    if (fread(b, 1, 2, f) != 2) {
        fclose(f);
        return fail(9003, "%s: truncated header", path);
    }
    if (get_le(b, 2) != 1) {
        fclose(f);
        return fail(9002, "%s: .psk.d version %" PRIu64 ", this reader reads version 1", path, get_le(b, 2));
    }
    if (fread(b, 1, 4, f) != 4) {
        fclose(f);
        return fail(9003, "%s: truncated header", path);
    }
    flen = get_le(b, 4);
    found = malloc(flen + 1);
    if (!found) {
        fclose(f);
        return fail(9003, "%s: a sample type of %" PRIu64 " bytes", path, flen);
    }
    if (fread(found, 1, flen, f) != flen) {
        free(found);
        fclose(f);
        return fail(9003, "%s: truncated header", path);
    }
    found[flen] = '\0';
    if (flen != tlen || memcmp(found, type, tlen) != 0) {
        fail(9005, "%s holds samples of `%s`, the program reads `%s`", path, found, type);
        free(found);
        fclose(f);
        return 9005;
    }
    free(found);
    if (fread(b, 1, 8, f) != 8) {
        fclose(f);
        return fail(9003, "%s: truncated header", path);
    }
    count = get_le(b, 8);
    head = 4 + 2 + 4 + tlen + 8;
    per = sample_bytes(leaves, nleaves);
    if (per == UINT64_MAX || (per > 0 && count > (UINT64_MAX - head) / per)) {
        fclose(f);
        return fail(9005, "%s: %" PRIu64 " samples of this type overflow a file", path, count);
    }
    need = head + count * per;
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0) {
        fclose(f);
        return fail(9003, "%s: %s", path, strerror(errno));
    }
    if ((uint64_t)end < need) {
        fclose(f);
        return fail(9003, "%s: truncated: %" PRIu64 " samples of %" PRIu64 " bytes announced, %" PRIu64 " bytes of data present", path, count, per, (uint64_t)end - head);
    }
    if ((uint64_t)end > need) {
        fclose(f);
        return fail(9005, "%s: %" PRIu64 " bytes of data where %" PRIu64 " samples of %" PRIu64 " bytes were expected: the file holds a sample of other dimensions", path, (uint64_t)end - head, count, per);
    }
    if (fseek(f, (long)head, SEEK_SET) != 0) {
        fclose(f);
        return fail(9003, "%s: %s", path, strerror(errno));
    }
    r->f = f;
    r->count = count;
    return 0;
}

int presk_read(presk_reader *r, void *const *dst, const presk_dtype *dtypes)
{
    unsigned char buf[CHUNK];
    int e;

    if (r->next >= r->count)
        return fail(9003, "sample %" PRIu64 " requested, %" PRIu64 " in the file", r->next, r->count);
    for (size_t i = 0; i < r->nleaves; i++) {
        const presk_leaf *l = &r->leaves[i];
        unsigned w = leaf_width(l);
        if ((e = check_dtype(l, dtypes[i], i)) != 0)
            return e;
        for (uint64_t k = 0; k < l->n;) {
            uint64_t left = l->n - k;
            size_t m = left < CHUNK / w ? (size_t)left : CHUNK / w;
            if (fread(buf, w, m, r->f) != m)
                return fail(9003, "truncated at sample %" PRIu64, r->next);
            for (size_t j = 0; j < m; j++) {
                uint64_t x = get_le(buf + j * w, w);
                if (l->bound) {
                    if (x >= l->bound)
                        return fail(9004, "sample %" PRIu64 ", leaf %zu, element %" PRIu64 ": index %" PRIu64 " is not below the bound %" PRIu64, r->next, i, k + j, x, l->bound);
                    ((int64_t *)dst[i])[k + j] = (int64_t)x;
                } else {
                    double d;
                    memcpy(&d, &x, sizeof d);
                    store_real(dst[i], dtypes[i], k + j, d);
                }
            }
            k += m;
        }
    }
    r->next++;
    return 0;
}

void presk_close(presk_reader *r)
{
    if (r->f)
        fclose(r->f);
    r->f = NULL;
}

int presk_load(const char *path, const char *type, const presk_leaf *leaves, size_t nleaves, void *const *dst, const presk_dtype *dtypes)
{
    presk_reader r;
    int e = presk_open(&r, path, type, leaves, nleaves);
    if (e)
        return e;
    if (r.count != 1) {
        presk_close(&r);
        return fail(9005, "%s: %" PRIu64 " samples, where one is read", path, r.count);
    }
    e = presk_read(&r, dst, dtypes);
    presk_close(&r);
    return e;
}

int presk_write(const char *path, const char *type, const presk_leaf *leaves, size_t nleaves, uint64_t count, const void *const *src, const presk_dtype *dtypes)
{
    unsigned char buf[CHUNK];
    size_t tlen = strlen(type);
    int e = 0;
    FILE *f;

    for (size_t i = 0; i < nleaves; i++)
        if ((e = check_dtype(&leaves[i], dtypes[i], i)) != 0)
            return e;
    if (tlen > 0xffffffffu)
        return fail(9005, "a sample type of %zu bytes", tlen);
    f = fopen(path, "wb");
    if (!f)
        return fail(9003, "%s: %s", path, strerror(errno));
    memcpy(buf, "PSKD", 4);
    put_le(buf + 4, 1, 2);
    put_le(buf + 6, tlen, 4);
    if (fwrite(buf, 1, 10, f) != 10 || fwrite(type, 1, tlen, f) != tlen)
        goto io;
    put_le(buf, count, 8);
    if (fwrite(buf, 1, 8, f) != 8)
        goto io;
    for (uint64_t s = 0; s < count; s++) {
        for (size_t i = 0; i < nleaves; i++) {
            const presk_leaf *l = &leaves[i];
            const void *from = src[s * nleaves + i];
            unsigned w = leaf_width(l);
            for (uint64_t k = 0; k < l->n;) {
                uint64_t left = l->n - k;
                size_t m = left < CHUNK / w ? (size_t)left : CHUNK / w;
                for (size_t j = 0; j < m; j++) {
                    uint64_t x;
                    if (l->bound) {
                        int64_t v = ((const int64_t *)from)[k + j];
                        if (v < 0 || (uint64_t)v >= l->bound) {
                            e = fail(9004, "sample %" PRIu64 ", leaf %zu, element %" PRIu64 ": index %" PRId64 " is %s", s, i, k + j, v, v < 0 ? "negative" : "not below the bound");
                            if (v >= 0)
                                snprintf(presk_errbuf + strlen(presk_errbuf), sizeof presk_errbuf - strlen(presk_errbuf), " %" PRIu64, l->bound);
                            goto out;
                        }
                        x = (uint64_t)v;
                    } else {
                        double d = load_real(from, dtypes[i], k + j);
                        memcpy(&x, &d, sizeof x);
                    }
                    put_le(buf + j * w, x, w);
                }
                if (fwrite(buf, w, m, f) != m)
                    goto io;
                k += m;
            }
        }
    }
    if (fclose(f) != 0) {
        f = NULL;
        goto io;
    }
    return 0;
io:
    e = fail(9003, "%s: %s", path, strerror(errno));
out:
    /* No partial file is left behind. */
    if (f)
        fclose(f);
    remove(path);
    return e;
}

/* ------------------------------------------------------------------------
 * Integer weights (plan 0.9.0 step 7, presk D84)
 * ------------------------------------------------------------------------ */

/* The smallest power of two s with m / s <= qmax, for m > 0: presk-rt's
 * loops, from 1. Each division by a power of two is exact until it
 * overflows, so the loops find the one answer from any start; m infinite
 * gives an infinite s. */
static double pow2_scale(double m, double qmax)
{
    double s = 1.0;
    while (m / (s / 2) <= qmax)
        s /= 2;
    while (m / s > qmax)
        s *= 2;
    return s;
}

/* x in [-qmax, qmax] rounded to the nearest integer, ties to even: Rust's
 * round_ties_even, whatever the rounding mode. |x| < 128, so its integer
 * part and the rest are exact. */
static int round_even(double x)
{
    double a = x < 0 ? -x : x;
    int i = (int)a;
    double f = a - (double)i;
    if (f > 0.5 || (f == 0.5 && (i & 1)))
        i++;
    return x < 0 ? -i : i;
}

void presk_quantize(void *q, float *s, const double *w, uint64_t outer, uint64_t k, uint64_t inner, int bits, presk_scheme scheme)
{
    const double qmax = (double)((1 << (bits - 1)) - 1);
    /* int4 nibbles are or-ed into their byte, in the order of the channels. */
    if (bits == 4)
        memset(q, 0, (size_t)((outer * k * inner + 1) / 2));
    for (uint64_t o = 0; o < outer; o++)
        for (uint64_t i = 0; i < inner; i++) {
            const uint64_t c = o * inner + i;
            double m = 0.0;
            for (uint64_t j = 0; j < k; j++) {
                double a = fabs(w[(o * k + j) * inner + i]);
                if (a > m) /* false for a NaN, which f64::max passes over */
                    m = a;
            }
            if (m == 0.0)
                s[c] = 0.0f; /* an all-zero channel stores zeros */
            else if (scheme == PRESK_POW2)
                s[c] = (float)pow2_scale(m, qmax);
            else
                s[c] = (float)(m / qmax);
            /* The f32 scale, widened: what the kernels multiply by. */
            const double sc = s[c];
            for (uint64_t j = 0; j < k; j++) {
                const uint64_t e = (o * k + j) * inner + i;
                int v = 0;
                if (sc != 0.0) {
                    double x = w[e] / sc;
                    /* Clamped before it is rounded, which gives what
                     * rounding first gives, since qmax is an integer; a
                     * NaN is 0, as `as i8` makes it. */
                    if (x == x)
                        v = round_even(x > qmax ? qmax : x < -qmax ? -qmax : x);
                }
                if (bits == 8)
                    ((int8_t *)q)[e] = (int8_t)v;
                else
                    ((uint8_t *)q)[e / 2] |= (uint8_t)(((unsigned)v & 0x0fu) << (e % 2 ? 4 : 0));
            }
        }
}

void presk_dequantize(void *dst, presk_dtype to, const void *q, const float *s, int bits, uint64_t k, uint64_t inner, uint64_t from, size_t n)
{
    for (size_t j = 0; j < n; j++) {
        const uint64_t e = from + j;
        const int v = bits == 8 ? ((const int8_t *)q)[e] : presk_int4(q, (int64_t)e);
        const float x = (float)v * s[(e / (k * inner)) * inner + e % inner];
        store_real(dst, to, j, (double)x);
    }
}

/* ------------------------------------------------------------------------
 * The arena
 * ------------------------------------------------------------------------ */

/* `bytes` rounded up to the alignment, or SIZE_MAX when that overflows. */
static size_t aligned(size_t bytes)
{
    if (bytes > SIZE_MAX - (PRESK_ALIGN - 1))
        return SIZE_MAX;
    return (bytes + (PRESK_ALIGN - 1)) / PRESK_ALIGN * PRESK_ALIGN;
}

int presk_arena_init(presk_arena *a, size_t size)
{
    size = aligned(size ? size : 1);
    a->base = size == SIZE_MAX ? NULL : aligned_alloc(PRESK_ALIGN, size);
    a->size = a->base ? size : 0;
    a->top = 0;
    a->peak = 0;
    if (!a->base) {
        snprintf(presk_errbuf, sizeof presk_errbuf, "an arena of %zu bytes: not that much memory", size);
        return 1;
    }
    return 0;
}

void presk_arena_free(presk_arena *a)
{
    free(a->base);
    a->base = NULL;
    a->size = a->top = 0;
}

void *presk_try_alloc(presk_arena *a, size_t bytes)
{
    size_t len = aligned(bytes);
    unsigned char *p;
    if (len > a->size - a->top) {
        snprintf(presk_errbuf, sizeof presk_errbuf, "the arena of %zu bytes has %zu left, and %zu were asked", a->size, a->size - a->top, bytes);
        return NULL;
    }
    p = a->base + a->top;
    a->top += len;
    if (a->top > a->peak)
        a->peak = a->top;
    return p;
}

void *presk_alloc(presk_arena *a, size_t bytes)
{
    void *p = presk_try_alloc(a, bytes);
    if (!p) {
        fprintf(stderr, "presk: %s: the export sized it too small\n", presk_errbuf);
        exit(EXIT_FAILURE);
    }
    return p;
}

size_t presk_mark(const presk_arena *a)
{
    return a->top;
}

void presk_release(presk_arena *a, size_t mark)
{
    a->top = mark;
}

void *presk_release_keep(presk_arena *a, size_t mark, const void *keep, size_t bytes)
{
    unsigned char *to = a->base + mark;
    memmove(to, keep, bytes);
    a->top = mark + aligned(bytes);
    return to;
}

/* ------------------------------------------------------------------------
 * The sampling stream (plan 0.9.0 step 6, presk D83)
 * ------------------------------------------------------------------------ */

#define PRESK_ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/* One quarter round. rand_chacha writes its rotations to the right, by 16,
 * 20, 24 and 25 bits: to the left by 16, 12, 8 and 7. */
#define PRESK_QR(a, b, c, d)          \
    do {                              \
        a += b;                       \
        d ^= a;                       \
        d = PRESK_ROTL(d, 16);        \
        c += d;                       \
        b ^= c;                       \
        b = PRESK_ROTL(b, 12);        \
        a += b;                       \
        d ^= a;                       \
        d = PRESK_ROTL(d, 8);         \
        c += d;                       \
        b ^= c;                       \
        b = PRESK_ROTL(b, 7);         \
    } while (0)

/* The block at `counter` into out[0..16]. */
static void chacha20_block(const uint32_t key[8], uint64_t counter, uint32_t out[16])
{
    uint32_t in[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, /* "expand 32-byte k" */
                       key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                       (uint32_t)counter, (uint32_t)(counter >> 32), 0u, 0u}; /* stream 0 */
    uint32_t x[16];
    memcpy(x, in, sizeof x);
    for (int i = 0; i < 10; i++) {
        PRESK_QR(x[0], x[4], x[8], x[12]);
        PRESK_QR(x[1], x[5], x[9], x[13]);
        PRESK_QR(x[2], x[6], x[10], x[14]);
        PRESK_QR(x[3], x[7], x[11], x[15]);
        PRESK_QR(x[0], x[5], x[10], x[15]);
        PRESK_QR(x[1], x[6], x[11], x[12]);
        PRESK_QR(x[2], x[7], x[8], x[13]);
        PRESK_QR(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; i++)
        out[i] = x[i] + in[i];
}

/* Four blocks from the counter, which moves on by four (ChaCha20Core's
 * generate, BlockRng's generate_and_set). */
static void refill(presk_rng *r)
{
    for (int b = 0; b < 4; b++)
        chacha20_block(r->key, r->counter + (uint64_t)b, r->results + 16 * b);
    r->counter += 4;
}

void presk_rng_from_seed(presk_rng *r, const uint8_t seed[32])
{
    for (int i = 0; i < 8; i++)
        r->key[i] = (uint32_t)seed[4 * i] | (uint32_t)seed[4 * i + 1] << 8 | (uint32_t)seed[4 * i + 2] << 16 | (uint32_t)seed[4 * i + 3] << 24;
    r->counter = 0;
    memset(r->results, 0, sizeof r->results);
    r->index = 64;
}

void presk_rng_seed_u64(presk_rng *r, uint64_t state)
{
    uint8_t seed[32];
    for (int i = 0; i < 8; i++) {
        /* PCG32: the state advanced first, then the XSH RR output. */
        state = state * UINT64_C(6364136223846793005) + UINT64_C(11634580027462260723);
        uint32_t xorshifted = (uint32_t)(((state >> 18) ^ state) >> 27);
        unsigned rot = (unsigned)(state >> 59);
        uint32_t x = (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
        for (int j = 0; j < 4; j++)
            seed[4 * i + j] = (uint8_t)(x >> (8 * j));
    }
    presk_rng_from_seed(r, seed);
}

uint32_t presk_rng_next_u32(presk_rng *r)
{
    if (r->index >= 64) {
        refill(r);
        r->index = 0;
    }
    return r->results[r->index++];
}

uint64_t presk_rng_next_u64(presk_rng *r)
{
    if (r->index < 63) {
        uint64_t lo = r->results[r->index], hi = r->results[r->index + 1];
        r->index += 2;
        return hi << 32 | lo;
    }
    if (r->index >= 64) {
        refill(r);
        r->index = 2;
        return (uint64_t)r->results[1] << 32 | r->results[0];
    }
    /* The last word of these blocks and the first of the next: never from
     * a fresh generator read by next_u64 alone, kept for next_u32. */
    uint64_t x = r->results[63];
    refill(r);
    r->index = 1;
    return (uint64_t)r->results[0] << 32 | x;
}

double presk_uniform01(presk_rng *r)
{
    return (double)(presk_rng_next_u64(r) >> 11) / 9007199254740992.0;
}

void presk_rng_init(presk_rng *r, uint64_t seed)
{
    presk_rng_seed_u64(r, seed ^ UINT64_C(0x5a5a5a5a));
}

void presk_sample(int64_t *out, const void *logits, presk_dtype from, size_t rows, size_t vocab, presk_rng *r)
{
    for (size_t p = 0; p < rows; p++) {
        uint64_t row = (uint64_t)(p * vocab);
        /* The largest, from -inf: a NaN is passed over, as f64::max passes
         * over it; which zero is kept does not matter, exp(-0) = exp(+0). */
        double m = -HUGE_VAL;
        for (size_t i = 0; i < vocab; i++) {
            double x = load_real(logits, from, row + i);
            if (x > m)
                m = x;
        }
        double z = 0.0;
        for (size_t i = 0; i < vocab; i++)
            z += presk_exp(load_real(logits, from, row + i) - m);
        double u = presk_uniform01(r);
        double acc = 0.0;
        int64_t pick = (int64_t)vocab - 1;
        for (size_t i = 0; i < vocab; i++) {
            acc += presk_exp(load_real(logits, from, row + i) - m) / z;
            if (u < acc) {
                pick = (int64_t)i;
                break;
            }
        }
        out[p] = pick;
    }
}
