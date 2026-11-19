/* FC1's C half (plan 0.18.2 step 3, presk D217): presk_rt.h's conversions
 * of binary16 and e8m15 over their inputs, block by block, each block's
 * outputs hashed (FNV-1a over 64-bit words), so that the program runs alike
 * natively and under an emulator and prints a few lines that
 * tests/codes.rs compares with presk_rt::codes' own.
 *
 *   fc1 PART PARTS STEP
 *
 * runs the blocks b with b % PARTS == PART, each on its inputs i with
 * i % STEP == 0, and prints "job block count hash" for each. The jobs:
 *   0  every f32 (by its bits) to binary16, 256 blocks of 2^24
 *   1  every f32 to e8m15, the same blocks
 *   2  every binary16 code to f32, one block
 *   3  every e8m15 code to f32, 16 blocks of 2^20
 *   4  binary16's midpoints from f64: between each two consecutive
 *      positive numbers, 0 and the least subnormal and the largest finite
 *      and the overflow threshold included, the midpoint and the doubles on
 *      either side of it, of both signs; one block
 *   5  e8m15's midpoints, 8 blocks of 2^20 codes
 *   6  2^32 random doubles to both formats, 128 blocks of 2^24 seeds, two
 *      doubles a seed, one of uniform bits and one of an exponent where the
 *      formats' numbers lie
 *   7  the overflow guard's test of every f32 stored in binary16 and in
 *      e8m15 (plan 0.18.2 step 4, presk D218), the flags the guarded
 *      conversions set, 256 blocks of 2^24
 * Build it with the export's flags, and -DPRESK_F16_INTEGER for binary16's
 * integer form where the compiler has _Float16. */

#include "presk_rt.h"

#include <stdlib.h>

static uint64_t hash_word(uint64_t h, uint64_t v)
{
    for (int k = 0; k < 8; k++) {
        h ^= (v >> (8 * k)) & 0xffu;
        h *= UINT64_C(0x100000001b3);
    }
    return h;
}

static uint64_t f32_bits(float x)
{
    uint32_t w;
    memcpy(&w, &x, sizeof w);
    return w;
}

/* The double of code c of the IEEE format of t trailing bits and exponent
 * bias `bias`, positive and finite; the overflow threshold for the code past
 * the largest finite. */
static double value(uint64_t c, unsigned t, int bias)
{
    uint64_t field = c >> t, frac = c & ((UINT64_C(1) << t) - 1);
    if (field == 0)
        return ldexp((double)frac, 1 - bias - (int)t);
    return ldexp((double)((UINT64_C(1) << t) | frac), (int)field - bias - (int)t);
}

static uint64_t mix(uint64_t z)
{
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double from_bits(uint64_t b)
{
    double x;
    memcpy(&x, &b, sizeof x);
    return x;
}

static uint64_t bits_of(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

/* One block of a job: its count of inputs and its hash. */
static uint64_t block(int job, uint64_t b, uint64_t step, uint64_t *count)
{
    uint64_t h = UINT64_C(0xcbf29ce484222325), n = 0;
    switch (job) {
    case 0:
    case 1:
        for (uint64_t i = b << 24; i < (b + 1) << 24; i += step) {
            uint32_t w = (uint32_t)i;
            float x;
            memcpy(&x, &w, sizeof x);
            h = hash_word(h, job == 0 ? presk_f32_to_f16(x) : presk_e8m15_code(presk_f32_to_e8m15(x)));
            n++;
        }
        break;
    case 2:
        for (uint64_t c = 0; c < 1u << 16; c++, n++)
            h = hash_word(h, f32_bits(presk_f16_to_f32((uint16_t)c)));
        break;
    case 3:
        for (uint64_t c = b << 20; c < (b + 1) << 20; c += step, n++)
            h = hash_word(h, f32_bits(presk_e8m15_to_f32(presk_e8m15_of((uint32_t)c))));
        break;
    case 4:
    case 5: {
        unsigned t = job == 4 ? 10 : 15;
        int bias = job == 4 ? 15 : 127;
        uint64_t top = job == 4 ? 0x7c00 : 0x7f8000, lo = job == 4 ? 0 : b << 20, hi = job == 4 ? top : (b + 1) << 20;
        if (hi > top)
            hi = top;
        for (uint64_t c = lo; c < hi; c += job == 4 ? 1 : step) {
            double m = (value(c, t, bias) + value(c + 1, t, bias)) / 2, below = from_bits(bits_of(m) - 1), above = from_bits(bits_of(m) + 1);
            double xs[6] = {m, below, above, -m, -below, -above};
            for (int k = 0; k < 6; k++, n++)
                h = hash_word(h, job == 4 ? presk_f64_to_f16(xs[k]) : presk_e8m15_code(presk_f64_to_e8m15(xs[k])));
        }
        break;
    }
    case 6:
        for (uint64_t i = b << 24; i < (b + 1) << 24; i += step) {
            uint64_t a = mix(i * UINT64_C(0x9e3779b97f4a7c15)), r = mix(i ^ UINT64_C(0x5851f42d4c957f2d)), e = (r >> 52) % 300;
            double xs[2] = {from_bits(a), from_bits((r & (UINT64_C(1) << 63)) | (1023 - 160 + e) << 52 | (a & ((UINT64_C(1) << 52) - 1)))};
            for (int k = 0; k < 2; k++, n++) {
                h = hash_word(h, presk_f64_to_f16(xs[k]));
                h = hash_word(h, presk_e8m15_code(presk_f64_to_e8m15(xs[k])));
            }
        }
        break;
    case 7:
        for (uint64_t i = b << 24; i < (b + 1) << 24; i += step, n++) {
            uint32_t w = (uint32_t)i, o16 = 0, o24 = 0;
            float x;
            memcpy(&x, &w, sizeof x);
            (void)presk_f32_to_f16_guarded(x, &o16);
            (void)presk_f32_to_e8m15_code_guarded(x, &o24);
            h = hash_word(h, o16 | o24 << 1);
        }
        break;
    }
    *count = n;
    return h;
}

/* Blocks per job. */
static const uint64_t BLOCKS[8] = {256, 256, 1, 16, 1, 8, 128, 256};

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s PART PARTS STEP\n", argv[0]);
        return 2;
    }
    uint64_t part = strtoull(argv[1], NULL, 10), parts = strtoull(argv[2], NULL, 10), step = strtoull(argv[3], NULL, 10), unit = 0;
    if (parts == 0 || step == 0)
        return 2;
    printf("form %s\n", PRESK_F16_CAST ? "cast" : "integer");
    for (int job = 0; job < 8; job++)
        for (uint64_t b = 0; b < BLOCKS[job]; b++, unit++) {
            if (unit % parts != part)
                continue;
            uint64_t n, h = block(job, b, step, &n);
            printf("%d %llu %llu %016llx\n", job, (unsigned long long)b, (unsigned long long)n, (unsigned long long)h);
        }
    return 0;
}
