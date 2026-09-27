/* presk_libm.c: the transcendental functions of Presk's C export (plan 0.9.0
 * §3.5, rule 2; presk D78).
 *
 * The native backend calls the Rust `libm` crate (0.2.16) for exp, log, tanh,
 * sin, cos and erf, and their f forms. The crate ports musl's math library as
 * it was before musl 1.1.23 replaced exp, expf, log and logf with ARM's
 * table-based routines. musl's functions are FreeBSD's msun, which comes
 * from Sun's fdlibm. This file holds those twelve functions and the helpers
 * they call, from musl 1.1.22 (tag v1.1.22, 2019-04-09), the last release
 * before that rewrite. The one exception is expm1f.c, taken at musl commit
 * 964104f9 (2021, released in 1.2.3), which fixes its overflow threshold, as
 * the crate has it. The test tests/libm.rs of the crate presk-export-c finds
 * each function bit-identical to the crate's.
 *
 * The edits made to musl's files, none of which changes an operation:
 * - The functions are named presk_exp, presk_expf, ..., as presk_rt.h
 *   declares them. Their helpers are static and named presk_...: scalbn is
 *   presk_scalbn, __sin presk_k_sin, __sindf presk_k_sindf, __rem_pio2
 *   presk_rem_pio2, erf.c's erfc1 presk_erfc1 and erff.c's presk_erfc1f.
 *   Nothing collides with the C library's libm, which a program may link.
 * - Each file's constants are moved from file scope into the function that
 *   reads them, since the files' constants share names (P1, ln2_hi, pp0,
 *   ...). erf.c's and erff.c's table is divided among erfc1, erfc2 and erf,
 *   which read disjoint parts of it.
 * - erfc and erfcf, which nothing here calls, are left out.
 * - `#include "libm.h"` gives way to the includes below. Of musl's internal
 *   libm.h, the macros these files use are copied as they are.
 * - M_PI_2, which <math.h> does not define in strict ISO C, is defined with
 *   musl's value when it is missing.
 * - Two edits for -Wall -Wextra -Werror under GCC 13 (statement C3), with no
 *   effect on a result. FORCE_EVAL reads back the volatile it writes, which
 *   GCC reports as set but not used. __rem_pio2_large's fq[] starts at zero,
 *   as the crate's does: GCC reports fq[0] as maybe used uninitialized,
 *   though the function writes fq[0..jz] before it reads them.
 *
 * Build it as the kernels are built, with -std=c11 -O2 -ffp-contract=off
 * and never -ffast-math (presk_rt.h): the functions round every operation
 * where the crate does only then.
 *
 * Licences. musl is under the MIT licence below, from its COPYRIGHT file.
 * Most of its math functions are FreeBSD's, each under the notice of Sun
 * Microsystems that heads it below, kept as musl keeps it; the file's
 * paragraph on them follows the licence (its paths shortened, since they
 * cannot be written inside a C comment).
 *
 * ----------------------------------------------------------------------
 * Copyright © 2005-2014 Rich Felker, et al.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ----------------------------------------------------------------------
 *
 * Much of the math library code (src/math and src/complex) is
 * Copyright © 1993,2004 Sun Microsystems or
 * Copyright © 2003-2011 David Schultz or
 * Copyright © 2003-2009 Steven G. Kargl or
 * Copyright © 2003-2009 Bruce D. Evans or
 * Copyright © 2008 Stephen L. Moshier
 * and labelled as such in comments in the individual source files. All
 * have been licensed under extremely permissive terms.
 */

#include "presk_rt.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* musl's and glibc's value */
#endif

/* ------------------------------------------------------------------------
 * musl 1.1.22, src/internal/libm.h: the macros the files below use
 * ------------------------------------------------------------------------ */

/* FORCE_EVAL reads __x back (edited, see above). */
#define FORCE_EVAL(x) do {                        \
	if (sizeof(x) == sizeof(float)) {         \
		volatile float __x;               \
		__x = (x);                        \
		(void)__x;                        \
	} else if (sizeof(x) == sizeof(double)) { \
		volatile double __x;              \
		__x = (x);                        \
		(void)__x;                        \
	} else {                                  \
		volatile long double __x;         \
		__x = (x);                        \
		(void)__x;                        \
	}                                         \
} while(0)

/* Get the more significant 32 bit int from a double.  */
#define GET_HIGH_WORD(hi,d)                       \
do {                                              \
  union {double f; uint64_t i;} __u;              \
  __u.f = (d);                                    \
  (hi) = __u.i >> 32;                             \
} while (0)

/* Set the less significant 32 bits of a double from an int.  */
#define SET_LOW_WORD(d,lo)                        \
do {                                              \
  union {double f; uint64_t i;} __u;              \
  __u.f = (d);                                    \
  __u.i &= 0xffffffff00000000ull;                 \
  __u.i |= (uint32_t)(lo);                        \
  (d) = __u.f;                                    \
} while (0)

/* Get a 32 bit int from a float.  */
#define GET_FLOAT_WORD(w,d)                       \
do {                                              \
  union {float f; uint32_t i;} __u;               \
  __u.f = (d);                                    \
  (w) = __u.i;                                    \
} while (0)

/* Set a float from a 32 bit int.  */
#define SET_FLOAT_WORD(d,w)                       \
do {                                              \
  union {float f; uint32_t i;} __u;               \
  __u.i = (w);                                    \
  (d) = __u.f;                                    \
} while (0)


/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/scalbn.c
 * ------------------------------------------------------------------------ */

static double presk_scalbn(double x, int n)
{
	union {double f; uint64_t i;} u;
	double_t y = x;

	if (n > 1023) {
		y *= 0x1p1023;
		n -= 1023;
		if (n > 1023) {
			y *= 0x1p1023;
			n -= 1023;
			if (n > 1023)
				n = 1023;
		}
	} else if (n < -1022) {
		/* make sure final n < -53 to avoid double
		   rounding in the subnormal range */
		y *= 0x1p-1022 * 0x1p53;
		n += 1022 - 53;
		if (n < -1022) {
			y *= 0x1p-1022 * 0x1p53;
			n += 1022 - 53;
			if (n < -1022)
				n = -1022;
		}
	}
	u.i = (uint64_t)(0x3ff+n)<<52;
	x = y * u.f;
	return x;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/scalbnf.c
 * ------------------------------------------------------------------------ */

static float presk_scalbnf(float x, int n)
{
	union {float f; uint32_t i;} u;
	float_t y = x;

	if (n > 127) {
		y *= 0x1p127f;
		n -= 127;
		if (n > 127) {
			y *= 0x1p127f;
			n -= 127;
			if (n > 127)
				n = 127;
		}
	} else if (n < -126) {
		y *= 0x1p-126f * 0x1p24f;
		n += 126 - 24;
		if (n < -126) {
			y *= 0x1p-126f * 0x1p24f;
			n += 126 - 24;
			if (n < -126)
				n = -126;
		}
	}
	u.i = (uint32_t)(0x7f+n)<<23;
	x = y * u.f;
	return x;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/expm1.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_expm1.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* expm1(x)
 * Returns exp(x)-1, the exponential of x minus 1.
 *
 * Method
 *   1. Argument reduction:
 *      Given x, find r and integer k such that
 *
 *               x = k*ln2 + r,  |r| <= 0.5*ln2 ~ 0.34658
 *
 *      Here a correction term c will be computed to compensate
 *      the error in r when rounded to a floating-point number.
 *
 *   2. Approximating expm1(r) by a special rational function on
 *      the interval [0,0.34658]:
 *      Since
 *          r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 - r^4/360 + ...
 *      we define R1(r*r) by
 *          r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 * R1(r*r)
 *      That is,
 *          R1(r**2) = 6/r *((exp(r)+1)/(exp(r)-1) - 2/r)
 *                   = 6/r * ( 1 + 2.0*(1/(exp(r)-1) - 1/r))
 *                   = 1 - r^2/60 + r^4/2520 - r^6/100800 + ...
 *      We use a special Remez algorithm on [0,0.347] to generate
 *      a polynomial of degree 5 in r*r to approximate R1. The
 *      maximum error of this polynomial approximation is bounded
 *      by 2**-61. In other words,
 *          R1(z) ~ 1.0 + Q1*z + Q2*z**2 + Q3*z**3 + Q4*z**4 + Q5*z**5
 *      where   Q1  =  -1.6666666666666567384E-2,
 *              Q2  =   3.9682539681370365873E-4,
 *              Q3  =  -9.9206344733435987357E-6,
 *              Q4  =   2.5051361420808517002E-7,
 *              Q5  =  -6.2843505682382617102E-9;
 *              z   =  r*r,
 *      with error bounded by
 *          |                  5           |     -61
 *          | 1.0+Q1*z+...+Q5*z   -  R1(z) | <= 2
 *          |                              |
 *
 *      expm1(r) = exp(r)-1 is then computed by the following
 *      specific way which minimize the accumulation rounding error:
 *                             2     3
 *                            r     r    [ 3 - (R1 + R1*r/2)  ]
 *            expm1(r) = r + --- + --- * [--------------------]
 *                            2     2    [ 6 - r*(3 - R1*r/2) ]
 *
 *      To compensate the error in the argument reduction, we use
 *              expm1(r+c) = expm1(r) + c + expm1(r)*c
 *                         ~ expm1(r) + c + r*c
 *      Thus c+r*c will be added in as the correction terms for
 *      expm1(r+c). Now rearrange the term to avoid optimization
 *      screw up:
 *                      (      2                                    2 )
 *                      ({  ( r    [ R1 -  (3 - R1*r/2) ]  )  }    r  )
 *       expm1(r+c)~r - ({r*(--- * [--------------------]-c)-c} - --- )
 *                      ({  ( 2    [ 6 - r*(3 - R1*r/2) ]  )  }    2  )
 *                      (                                             )
 *
 *                 = r - E
 *   3. Scale back to obtain expm1(x):
 *      From step 1, we have
 *         expm1(x) = either 2^k*[expm1(r)+1] - 1
 *                  = or     2^k*[expm1(r) + (1-2^-k)]
 *   4. Implementation notes:
 *      (A). To save one multiplication, we scale the coefficient Qi
 *           to Qi*2^i, and replace z by (x^2)/2.
 *      (B). To achieve maximum accuracy, we compute expm1(x) by
 *        (i)   if x < -56*ln2, return -1.0, (raise inexact if x!=inf)
 *        (ii)  if k=0, return r-E
 *        (iii) if k=-1, return 0.5*(r-E)-0.5
 *        (iv)  if k=1 if r < -0.25, return 2*((r+0.5)- E)
 *                     else          return  1.0+2.0*(r-E);
 *        (v)   if (k<-2||k>56) return 2^k(1-(E-r)) - 1 (or exp(x)-1)
 *        (vi)  if k <= 20, return 2^k((1-2^-k)-(E-r)), else
 *        (vii) return 2^k(1-((E+2^-k)-r))
 *
 * Special cases:
 *      expm1(INF) is INF, expm1(NaN) is NaN;
 *      expm1(-INF) is -1, and
 *      for finite argument, only expm1(0)=0 is exact.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Misc. info.
 *      For IEEE double
 *          if x >  7.09782712893383973096e+02 then expm1(x) overflow
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

static double presk_expm1(double x)
{
	static const double
	o_threshold = 7.09782712893383973096e+02, /* 0x40862E42, 0xFEFA39EF */
	ln2_hi      = 6.93147180369123816490e-01, /* 0x3fe62e42, 0xfee00000 */
	ln2_lo      = 1.90821492927058770002e-10, /* 0x3dea39ef, 0x35793c76 */
	invln2      = 1.44269504088896338700e+00, /* 0x3ff71547, 0x652b82fe */
	/* Scaled Q's: Qn_here = 2**n * Qn_above, for R(2*z) where z = hxs = x*x/2: */
	Q1 = -3.33333333333331316428e-02, /* BFA11111 111110F4 */
	Q2 =  1.58730158725481460165e-03, /* 3F5A01A0 19FE5585 */
	Q3 = -7.93650757867487942473e-05, /* BF14CE19 9EAADBB7 */
	Q4 =  4.00821782732936239552e-06, /* 3ED0CFCA 86E65239 */
	Q5 = -2.01099218183624371326e-07; /* BE8AFDB7 6E09C32D */

	double_t y,hi,lo,c,t,e,hxs,hfx,r1,twopk;
	union {double f; uint64_t i;} u = {x};
	uint32_t hx = u.i>>32 & 0x7fffffff;
	int k, sign = u.i>>63;

	/* filter out huge and non-finite argument */
	if (hx >= 0x4043687A) {  /* if |x|>=56*ln2 */
		if (isnan(x))
			return x;
		if (sign)
			return -1;
		if (x > o_threshold) {
			x *= 0x1p1023;
			return x;
		}
	}

	/* argument reduction */
	if (hx > 0x3fd62e42) {  /* if  |x| > 0.5 ln2 */
		if (hx < 0x3FF0A2B2) {  /* and |x| < 1.5 ln2 */
			if (!sign) {
				hi = x - ln2_hi;
				lo = ln2_lo;
				k =  1;
			} else {
				hi = x + ln2_hi;
				lo = -ln2_lo;
				k = -1;
			}
		} else {
			k  = invln2*x + (sign ? -0.5 : 0.5);
			t  = k;
			hi = x - t*ln2_hi;  /* t*ln2_hi is exact here */
			lo = t*ln2_lo;
		}
		x = hi-lo;
		c = (hi-x)-lo;
	} else if (hx < 0x3c900000) {  /* |x| < 2**-54, return x */
		if (hx < 0x00100000)
			FORCE_EVAL((float)x);
		return x;
	} else
		k = 0;

	/* x is now in primary range */
	hfx = 0.5*x;
	hxs = x*hfx;
	r1 = 1.0+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5))));
	t  = 3.0-r1*hfx;
	e  = hxs*((r1-t)/(6.0 - x*t));
	if (k == 0)   /* c is 0 */
		return x - (x*e-hxs);
	e  = x*(e-c) - c;
	e -= hxs;
	/* exp(x) ~ 2^k (x_reduced - e + 1) */
	if (k == -1)
		return 0.5*(x-e) - 0.5;
	if (k == 1) {
		if (x < -0.25)
			return -2.0*(e-(x+0.5));
		return 1.0+2.0*(x-e);
	}
	u.i = (uint64_t)(0x3ff + k)<<52;  /* 2^k */
	twopk = u.f;
	if (k < 0 || k > 56) {  /* suffice to return exp(x)-1 */
		y = x - e + 1.0;
		if (k == 1024)
			y = y*2.0*0x1p1023;
		else
			y = y*twopk;
		return y - 1.0;
	}
	u.i = (uint64_t)(0x3ff - k)<<52;  /* 2^-k */
	if (k < 20)
		y = (x-e+(1-u.f))*twopk;
	else
		y = (x-(e+u.f)+1)*twopk;
	return y;
}

/* -------------------------------------------------------------------------
 * musl commit 964104f9 (released in 1.2.3), src/math/expm1f.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_expm1f.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

static float presk_expm1f(float x)
{
	static const float
	ln2_hi      = 6.9313812256e-01, /* 0x3f317180 */
	ln2_lo      = 9.0580006145e-06, /* 0x3717f7d1 */
	invln2      = 1.4426950216e+00, /* 0x3fb8aa3b */
	/*
	 * Domain [-0.34568, 0.34568], range ~[-6.694e-10, 6.696e-10]:
	 * |6 / x * (1 + 2 * (1 / (exp(x) - 1) - 1 / x)) - q(x)| < 2**-30.04
	 * Scaled coefficients: Qn_here = 2**n * Qn_for_q (see s_expm1.c):
	 */
	Q1 = -3.3333212137e-2, /* -0x888868.0p-28 */
	Q2 =  1.5807170421e-3; /*  0xcf3010.0p-33 */

	float_t y,hi,lo,c,t,e,hxs,hfx,r1,twopk;
	union {float f; uint32_t i;} u = {x};
	uint32_t hx = u.i & 0x7fffffff;
	int k, sign = u.i >> 31;

	/* filter out huge and non-finite argument */
	if (hx >= 0x4195b844) {  /* if |x|>=27*ln2 */
		if (hx > 0x7f800000)  /* NaN */
			return x;
		if (sign)
			return -1;
		if (hx > 0x42b17217) { /* x > log(FLT_MAX) */
			x *= 0x1p127f;
			return x;
		}
	}

	/* argument reduction */
	if (hx > 0x3eb17218) {           /* if  |x| > 0.5 ln2 */
		if (hx < 0x3F851592) {       /* and |x| < 1.5 ln2 */
			if (!sign) {
				hi = x - ln2_hi;
				lo = ln2_lo;
				k =  1;
			} else {
				hi = x + ln2_hi;
				lo = -ln2_lo;
				k = -1;
			}
		} else {
			k  = invln2*x + (sign ? -0.5f : 0.5f);
			t  = k;
			hi = x - t*ln2_hi;      /* t*ln2_hi is exact here */
			lo = t*ln2_lo;
		}
		x = hi-lo;
		c = (hi-x)-lo;
	} else if (hx < 0x33000000) {  /* when |x|<2**-25, return x */
		if (hx < 0x00800000)
			FORCE_EVAL(x*x);
		return x;
	} else
		k = 0;

	/* x is now in primary range */
	hfx = 0.5f*x;
	hxs = x*hfx;
	r1 = 1.0f+hxs*(Q1+hxs*Q2);
	t  = 3.0f - r1*hfx;
	e  = hxs*((r1-t)/(6.0f - x*t));
	if (k == 0)  /* c is 0 */
		return x - (x*e-hxs);
	e  = x*(e-c) - c;
	e -= hxs;
	/* exp(x) ~ 2^k (x_reduced - e + 1) */
	if (k == -1)
		return 0.5f*(x-e) - 0.5f;
	if (k == 1) {
		if (x < -0.25f)
			return -2.0f*(e-(x+0.5f));
		return 1.0f + 2.0f*(x-e);
	}
	u.i = (0x7f+k)<<23;  /* 2^k */
	twopk = u.f;
	if (k < 0 || k > 56) {   /* suffice to return exp(x)-1 */
		y = x - e + 1.0f;
		if (k == 128)
			y = y*2.0f*0x1p127f;
		else
			y = y*twopk;
		return y - 1.0f;
	}
	u.i = (0x7f-k)<<23;  /* 2^-k */
	if (k < 23)
		y = (x-e+(1-u.f))*twopk;
	else
		y = (x-(e+u.f)+1)*twopk;
	return y;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__sin.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/k_sin.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* __sin( x, y, iy)
 * kernel sin function on ~[-pi/4, pi/4] (except on -0), pi/4 ~ 0.7854
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 * Input iy indicates whether y is 0. (if iy=0, y assume to be 0).
 *
 * Algorithm
 *      1. Since sin(-x) = -sin(x), we need only to consider positive x.
 *      2. Callers must return sin(-0) = -0 without calling here since our
 *         odd polynomial is not evaluated in a way that preserves -0.
 *         Callers may do the optimization sin(x) ~ x for tiny x.
 *      3. sin(x) is approximated by a polynomial of degree 13 on
 *         [0,pi/4]
 *                               3            13
 *              sin(x) ~ x + S1*x + ... + S6*x
 *         where
 *
 *      |sin(x)         2     4     6     8     10     12  |     -58
 *      |----- - (1+S1*x +S2*x +S3*x +S4*x +S5*x  +S6*x   )| <= 2
 *      |  x                                               |
 *
 *      4. sin(x+y) = sin(x) + sin'(x')*y
 *                  ~ sin(x) + (1-x*x/2)*y
 *         For better accuracy, let
 *                   3      2      2      2      2
 *              r = x *(S2+x *(S3+x *(S4+x *(S5+x *S6))))
 *         then                   3    2
 *              sin(x) = x + (S1*x + (x *(r-y/2)+y))
 */

static double presk_k_sin(double x, double y, int iy)
{
	static const double
	S1  = -1.66666666666666324348e-01, /* 0xBFC55555, 0x55555549 */
	S2  =  8.33333333332248946124e-03, /* 0x3F811111, 0x1110F8A6 */
	S3  = -1.98412698298579493134e-04, /* 0xBF2A01A0, 0x19C161D5 */
	S4  =  2.75573137070700676789e-06, /* 0x3EC71DE3, 0x57B1FE7D */
	S5  = -2.50507602534068634195e-08, /* 0xBE5AE5E6, 0x8A2B9CEB */
	S6  =  1.58969099521155010221e-10; /* 0x3DE5D93A, 0x5ACFD57C */

	double_t z,r,v,w;

	z = x*x;
	w = z*z;
	r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6);
	v = z*x;
	if (iy == 0)
		return x + v*(S1 + z*r);
	else
		return x - ((z*(0.5*y - v*r) - y) - v*S1);
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__cos.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/k_cos.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * __cos( x,  y )
 * kernel cos function on [-pi/4, pi/4], pi/4 ~ 0.785398164
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 *
 * Algorithm
 *      1. Since cos(-x) = cos(x), we need only to consider positive x.
 *      2. if x < 2^-27 (hx<0x3e400000 0), return 1 with inexact if x!=0.
 *      3. cos(x) is approximated by a polynomial of degree 14 on
 *         [0,pi/4]
 *                                       4            14
 *              cos(x) ~ 1 - x*x/2 + C1*x + ... + C6*x
 *         where the remez error is
 *
 *      |              2     4     6     8     10    12     14 |     -58
 *      |cos(x)-(1-.5*x +C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  )| <= 2
 *      |                                                      |
 *
 *                     4     6     8     10    12     14
 *      4. let r = C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  , then
 *             cos(x) ~ 1 - x*x/2 + r
 *         since cos(x+y) ~ cos(x) - sin(x)*y
 *                        ~ cos(x) - x*y,
 *         a correction term is necessary in cos(x) and hence
 *              cos(x+y) = 1 - (x*x/2 - (r - x*y))
 *         For better accuracy, rearrange to
 *              cos(x+y) ~ w + (tmp + (r-x*y))
 *         where w = 1 - x*x/2 and tmp is a tiny correction term
 *         (1 - x*x/2 == w + tmp exactly in infinite precision).
 *         The exactness of w + tmp in infinite precision depends on w
 *         and tmp having the same precision as x.  If they have extra
 *         precision due to compiler bugs, then the extra precision is
 *         only good provided it is retained in all terms of the final
 *         expression for cos().  Retention happens in all cases tested
 *         under FreeBSD, so don't pessimize things by forcibly clipping
 *         any extra precision in w.
 */

static double presk_k_cos(double x, double y)
{
	static const double
	C1  =  4.16666666666666019037e-02, /* 0x3FA55555, 0x5555554C */
	C2  = -1.38888888888741095749e-03, /* 0xBF56C16C, 0x16C15177 */
	C3  =  2.48015872894767294178e-05, /* 0x3EFA01A0, 0x19CB1590 */
	C4  = -2.75573143513906633035e-07, /* 0xBE927E4F, 0x809C52AD */
	C5  =  2.08757232129817482790e-09, /* 0x3E21EE9E, 0xBDB4B1C4 */
	C6  = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */

	double_t hz,z,r,w;

	z  = x*x;
	w  = z*z;
	r  = z*(C1+z*(C2+z*C3)) + w*w*(C4+z*(C5+z*C6));
	hz = 0.5*z;
	w  = 1.0-hz;
	return w + (((1.0-w)-hz) + (z*r-x*y));
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__sindf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/k_sinf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

static float presk_k_sindf(double x)
{
	/* |sin(x)/x - s(x)| < 2**-37.5 (~[-4.89e-12, 4.824e-12]). */
	static const double
	S1 = -0x15555554cbac77.0p-55, /* -0.166666666416265235595 */
	S2 =  0x111110896efbb2.0p-59, /*  0.0083333293858894631756 */
	S3 = -0x1a00f9e2cae774.0p-65, /* -0.000198393348360966317347 */
	S4 =  0x16cd878c3b46a7.0p-71; /*  0.0000027183114939898219064 */

	double_t r, s, w, z;

	/* Try to optimize for parallel evaluation as in __tandf.c. */
	z = x*x;
	w = z*z;
	r = S3 + z*S4;
	s = z*x;
	return (x + s*(S1 + z*S2)) + s*w*r;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__cosdf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/k_cosf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Debugged and optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

static float presk_k_cosdf(double x)
{
	/* |cos(x) - c(x)| < 2**-34.1 (~[-5.37e-11, 5.295e-11]). */
	static const double
	C0  = -0x1ffffffd0c5e81.0p-54, /* -0.499999997251031003120 */
	C1  =  0x155553e1053a42.0p-57, /*  0.0416666233237390631894 */
	C2  = -0x16c087e80f1e27.0p-62, /* -0.00138867637746099294692 */
	C3  =  0x199342e0ee5069.0p-68; /*  0.0000243904487962774090654 */

	double_t r, w, z;

	/* Try to optimize for parallel evaluation as in __tandf.c. */
	z = x*x;
	w = z*z;
	r = C2+z*C3;
	return ((1.0+z*C0) + w*C1) + (w*z)*r;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__rem_pio2_large.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/k_rem_pio2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * __rem_pio2_large(x,y,e0,nx,prec)
 * double x[],y[]; int e0,nx,prec;
 *
 * __rem_pio2_large return the last three digits of N with
 *              y = x - N*pi/2
 * so that |y| < pi/2.
 *
 * The method is to compute the integer (mod 8) and fraction parts of
 * (2/pi)*x without doing the full multiplication. In general we
 * skip the part of the product that are known to be a huge integer (
 * more accurately, = 0 mod 8 ). Thus the number of operations are
 * independent of the exponent of the input.
 *
 * (2/pi) is represented by an array of 24-bit integers in ipio2[].
 *
 * Input parameters:
 *      x[]     The input value (must be positive) is broken into nx
 *              pieces of 24-bit integers in double precision format.
 *              x[i] will be the i-th 24 bit of x. The scaled exponent
 *              of x[0] is given in input parameter e0 (i.e., x[0]*2^e0
 *              match x's up to 24 bits.
 *
 *              Example of breaking a double positive z into x[0]+x[1]+x[2]:
 *                      e0 = ilogb(z)-23
 *                      z  = scalbn(z,-e0)
 *              for i = 0,1,2
 *                      x[i] = floor(z)
 *                      z    = (z-x[i])*2**24
 *
 *
 *      y[]     ouput result in an array of double precision numbers.
 *              The dimension of y[] is:
 *                      24-bit  precision       1
 *                      53-bit  precision       2
 *                      64-bit  precision       2
 *                      113-bit precision       3
 *              The actual value is the sum of them. Thus for 113-bit
 *              precison, one may have to do something like:
 *
 *              long double t,w,r_head, r_tail;
 *              t = (long double)y[2] + (long double)y[1];
 *              w = (long double)y[0];
 *              r_head = t+w;
 *              r_tail = w - (r_head - t);
 *
 *      e0      The exponent of x[0]. Must be <= 16360 or you need to
 *              expand the ipio2 table.
 *
 *      nx      dimension of x[]
 *
 *      prec    an integer indicating the precision:
 *                      0       24  bits (single)
 *                      1       53  bits (double)
 *                      2       64  bits (extended)
 *                      3       113 bits (quad)
 *
 * External function:
 *      double scalbn(), floor();
 *
 *
 * Here is the description of some local variables:
 *
 *      jk      jk+1 is the initial number of terms of ipio2[] needed
 *              in the computation. The minimum and recommended value
 *              for jk is 3,4,4,6 for single, double, extended, and quad.
 *              jk+1 must be 2 larger than you might expect so that our
 *              recomputation test works. (Up to 24 bits in the integer
 *              part (the 24 bits of it that we compute) and 23 bits in
 *              the fraction part may be lost to cancelation before we
 *              recompute.)
 *
 *      jz      local integer variable indicating the number of
 *              terms of ipio2[] used.
 *
 *      jx      nx - 1
 *
 *      jv      index for pointing to the suitable ipio2[] for the
 *              computation. In general, we want
 *                      ( 2^e0*x[0] * ipio2[jv-1]*2^(-24jv) )/8
 *              is an integer. Thus
 *                      e0-3-24*jv >= 0 or (e0-3)/24 >= jv
 *              Hence jv = max(0,(e0-3)/24).
 *
 *      jp      jp+1 is the number of terms in PIo2[] needed, jp = jk.
 *
 *      q[]     double array with integral value, representing the
 *              24-bits chunk of the product of x and 2/pi.
 *
 *      q0      the corresponding exponent of q[0]. Note that the
 *              exponent for q[i] would be q0-24*i.
 *
 *      PIo2[]  double precision array, obtained by cutting pi/2
 *              into 24 bits chunks.
 *
 *      f[]     ipio2[] in floating point
 *
 *      iq[]    integer array by breaking up q[] in 24-bits chunk.
 *
 *      fq[]    final product of x*(2/pi) in fq[0],..,fq[jk]
 *
 *      ih      integer. If >0 it indicates q[] is >= 0.5, hence
 *              it also indicates the *sign* of the result.
 *
 */
/*
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

static int presk_rem_pio2_large(double *x, double *y, int e0, int nx, int prec)
{
	static const int init_jk[] = {3,4,4,6}; /* initial value for jk */
	/*
	 * Table of constants for 2/pi, 396 Hex digits (476 decimal) of 2/pi
	 *
	 *              integer array, contains the (24*i)-th to (24*i+23)-th
	 *              bit of 2/pi after binary point. The corresponding
	 *              floating value is
	 *
	 *                      ipio2[i] * 2^(-24(i+1)).
	 *
	 * NB: This table must have at least (e0-3)/24 + jk terms.
	 *     For quad precision (e0 <= 16360, jk = 6), this is 686.
	 */
	static const int32_t ipio2[] = {
	0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
	0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
	0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
	0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
	0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
	0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
	0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
	0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
	0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
	0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
	0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,

#if LDBL_MAX_EXP > 1024
	0x47C419, 0xC367CD, 0xDCE809, 0x2A8359, 0xC4768B, 0x961CA6,
	0xDDAF44, 0xD15719, 0x053EA5, 0xFF0705, 0x3F7E33, 0xE832C2,
	0xDE4F98, 0x327DBB, 0xC33D26, 0xEF6B1E, 0x5EF89F, 0x3A1F35,
	0xCAF27F, 0x1D87F1, 0x21907C, 0x7C246A, 0xFA6ED5, 0x772D30,
	0x433B15, 0xC614B5, 0x9D19C3, 0xC2C4AD, 0x414D2C, 0x5D000C,
	0x467D86, 0x2D71E3, 0x9AC69B, 0x006233, 0x7CD2B4, 0x97A7B4,
	0xD55537, 0xF63ED7, 0x1810A3, 0xFC764D, 0x2A9D64, 0xABD770,
	0xF87C63, 0x57B07A, 0xE71517, 0x5649C0, 0xD9D63B, 0x3884A7,
	0xCB2324, 0x778AD6, 0x23545A, 0xB91F00, 0x1B0AF1, 0xDFCE19,
	0xFF319F, 0x6A1E66, 0x615799, 0x47FBAC, 0xD87F7E, 0xB76522,
	0x89E832, 0x60BFE6, 0xCDC4EF, 0x09366C, 0xD43F5D, 0xD7DE16,
	0xDE3B58, 0x929BDE, 0x2822D2, 0xE88628, 0x4D58E2, 0x32CAC6,
	0x16E308, 0xCB7DE0, 0x50C017, 0xA71DF3, 0x5BE018, 0x34132E,
	0x621283, 0x014883, 0x5B8EF5, 0x7FB0AD, 0xF2E91E, 0x434A48,
	0xD36710, 0xD8DDAA, 0x425FAE, 0xCE616A, 0xA4280A, 0xB499D3,
	0xF2A606, 0x7F775C, 0x83C2A3, 0x883C61, 0x78738A, 0x5A8CAF,
	0xBDD76F, 0x63A62D, 0xCBBFF4, 0xEF818D, 0x67C126, 0x45CA55,
	0x36D9CA, 0xD2A828, 0x8D61C2, 0x77C912, 0x142604, 0x9B4612,
	0xC459C4, 0x44C5C8, 0x91B24D, 0xF31700, 0xAD43D4, 0xE54929,
	0x10D5FD, 0xFCBE00, 0xCC941E, 0xEECE70, 0xF53E13, 0x80F1EC,
	0xC3E7B3, 0x28F8C7, 0x940593, 0x3E71C1, 0xB3092E, 0xF3450B,
	0x9C1288, 0x7B20AB, 0x9FB52E, 0xC29247, 0x2F327B, 0x6D550C,
	0x90A772, 0x1FE76B, 0x96CB31, 0x4A1679, 0xE27941, 0x89DFF4,
	0x9794E8, 0x84E6E2, 0x973199, 0x6BED88, 0x365F5F, 0x0EFDBB,
	0xB49A48, 0x6CA467, 0x427271, 0x325D8D, 0xB8159F, 0x09E5BC,
	0x25318D, 0x3974F7, 0x1C0530, 0x010C0D, 0x68084B, 0x58EE2C,
	0x90AA47, 0x02E774, 0x24D6BD, 0xA67DF7, 0x72486E, 0xEF169F,
	0xA6948E, 0xF691B4, 0x5153D1, 0xF20ACF, 0x339820, 0x7E4BF5,
	0x6863B2, 0x5F3EDD, 0x035D40, 0x7F8985, 0x295255, 0xC06437,
	0x10D86D, 0x324832, 0x754C5B, 0xD4714E, 0x6E5445, 0xC1090B,
	0x69F52A, 0xD56614, 0x9D0727, 0x50045D, 0xDB3BB4, 0xC576EA,
	0x17F987, 0x7D6B49, 0xBA271D, 0x296996, 0xACCCC6, 0x5414AD,
	0x6AE290, 0x89D988, 0x50722C, 0xBEA404, 0x940777, 0x7030F3,
	0x27FC00, 0xA871EA, 0x49C266, 0x3DE064, 0x83DD97, 0x973FA3,
	0xFD9443, 0x8C860D, 0xDE4131, 0x9D3992, 0x8C70DD, 0xE7B717,
	0x3BDF08, 0x2B3715, 0xA0805C, 0x93805A, 0x921110, 0xD8E80F,
	0xAF806C, 0x4BFFDB, 0x0F9038, 0x761859, 0x15A562, 0xBBCB61,
	0xB989C7, 0xBD4010, 0x04F2D2, 0x277549, 0xF6B6EB, 0xBB22DB,
	0xAA140A, 0x2F2689, 0x768364, 0x333B09, 0x1A940E, 0xAA3A51,
	0xC2A31D, 0xAEEDAF, 0x12265C, 0x4DC26D, 0x9C7A2D, 0x9756C0,
	0x833F03, 0xF6F009, 0x8C402B, 0x99316D, 0x07B439, 0x15200C,
	0x5BC3D8, 0xC492F5, 0x4BADC6, 0xA5CA4E, 0xCD37A7, 0x36A9E6,
	0x9492AB, 0x6842DD, 0xDE6319, 0xEF8C76, 0x528B68, 0x37DBFC,
	0xABA1AE, 0x3115DF, 0xA1AE00, 0xDAFB0C, 0x664D64, 0xB705ED,
	0x306529, 0xBF5657, 0x3AFF47, 0xB9F96A, 0xF3BE75, 0xDF9328,
	0x3080AB, 0xF68C66, 0x15CB04, 0x0622FA, 0x1DE4D9, 0xA4B33D,
	0x8F1B57, 0x09CD36, 0xE9424E, 0xA4BE13, 0xB52333, 0x1AAAF0,
	0xA8654F, 0xA5C1D2, 0x0F3F0B, 0xCD785B, 0x76F923, 0x048B7B,
	0x721789, 0x53A6C6, 0xE26E6F, 0x00EBEF, 0x584A9B, 0xB7DAC4,
	0xBA66AA, 0xCFCF76, 0x1D02D1, 0x2DF1B1, 0xC1998C, 0x77ADC3,
	0xDA4886, 0xA05DF7, 0xF480C6, 0x2FF0AC, 0x9AECDD, 0xBC5C3F,
	0x6DDED0, 0x1FC790, 0xB6DB2A, 0x3A25A3, 0x9AAF00, 0x9353AD,
	0x0457B6, 0xB42D29, 0x7E804B, 0xA707DA, 0x0EAA76, 0xA1597B,
	0x2A1216, 0x2DB7DC, 0xFDE5FA, 0xFEDB89, 0xFDBE89, 0x6C76E4,
	0xFCA906, 0x70803E, 0x156E85, 0xFF87FD, 0x073E28, 0x336761,
	0x86182A, 0xEABD4D, 0xAFE7B3, 0x6E6D8F, 0x396795, 0x5BBF31,
	0x48D784, 0x16DF30, 0x432DC7, 0x356125, 0xCE70C9, 0xB8CB30,
	0xFD6CBF, 0xA200A4, 0xE46C05, 0xA0DD5A, 0x476F21, 0xD21262,
	0x845CB9, 0x496170, 0xE0566B, 0x015299, 0x375550, 0xB7D51E,
	0xC4F133, 0x5F6E13, 0xE4305D, 0xA92E85, 0xC3B21D, 0x3632A1,
	0xA4B708, 0xD4B1EA, 0x21F716, 0xE4698F, 0x77FF27, 0x80030C,
	0x2D408D, 0xA0CD4F, 0x99A520, 0xD3A2B3, 0x0A5D2F, 0x42F9B4,
	0xCBDA11, 0xD0BE7D, 0xC1DB9B, 0xBD17AB, 0x81A2CA, 0x5C6A08,
	0x17552E, 0x550027, 0xF0147F, 0x8607E1, 0x640B14, 0x8D4196,
	0xDEBE87, 0x2AFDDA, 0xB6256B, 0x34897B, 0xFEF305, 0x9EBFB9,
	0x4F6A68, 0xA82A4A, 0x5AC44F, 0xBCF82D, 0x985AD7, 0x95C7F4,
	0x8D4D0D, 0xA63A20, 0x5F57A4, 0xB13F14, 0x953880, 0x0120CC,
	0x86DD71, 0xB6DEC9, 0xF560BF, 0x11654D, 0x6B0701, 0xACB08C,
	0xD0C0B2, 0x485551, 0x0EFB1E, 0xC37295, 0x3B06A3, 0x3540C0,
	0x7BDC06, 0xCC45E0, 0xFA294E, 0xC8CAD6, 0x41F3E8, 0xDE647C,
	0xD8649B, 0x31BED9, 0xC397A4, 0xD45877, 0xC5E369, 0x13DAF0,
	0x3C3ABA, 0x461846, 0x5F7555, 0xF5BDD2, 0xC6926E, 0x5D2EAC,
	0xED440E, 0x423E1C, 0x87C461, 0xE9FD29, 0xF3D6E7, 0xCA7C22,
	0x35916F, 0xC5E008, 0x8DD7FF, 0xE26A6E, 0xC6FDB0, 0xC10893,
	0x745D7C, 0xB2AD6B, 0x9D6ECD, 0x7B723E, 0x6A11C6, 0xA9CFF7,
	0xDF7329, 0xBAC9B5, 0x5100B7, 0x0DB2E2, 0x24BA74, 0x607DE5,
	0x8AD874, 0x2C150D, 0x0C1881, 0x94667E, 0x162901, 0x767A9F,
	0xBEFDFD, 0xEF4556, 0x367ED9, 0x13D9EC, 0xB9BA8B, 0xFC97C4,
	0x27A831, 0xC36EF1, 0x36C594, 0x56A8D8, 0xB5A8B4, 0x0ECCCF,
	0x2D8912, 0x34576F, 0x89562C, 0xE3CE99, 0xB920D6, 0xAA5E6B,
	0x9C2A3E, 0xCC5F11, 0x4A0BFD, 0xFBF4E1, 0x6D3B8E, 0x2C86E2,
	0x84D4E9, 0xA9B4FC, 0xD1EEEF, 0xC9352E, 0x61392F, 0x442138,
	0xC8D91B, 0x0AFC81, 0x6A4AFB, 0xD81C2F, 0x84B453, 0x8C994E,
	0xCC2254, 0xDC552A, 0xD6C6C0, 0x96190B, 0xB8701A, 0x649569,
	0x605A26, 0xEE523F, 0x0F117F, 0x11B5F4, 0xF5CBFC, 0x2DBC34,
	0xEEBC34, 0xCC5DE8, 0x605EDD, 0x9B8E67, 0xEF3392, 0xB817C9,
	0x9B5861, 0xBC57E1, 0xC68351, 0x103ED8, 0x4871DD, 0xDD1C2D,
	0xA118AF, 0x462C21, 0xD7F359, 0x987AD9, 0xC0549E, 0xFA864F,
	0xFC0656, 0xAE79E5, 0x362289, 0x22AD38, 0xDC9367, 0xAAE855,
	0x382682, 0x9BE7CA, 0xA40D51, 0xB13399, 0x0ED7A9, 0x480569,
	0xF0B265, 0xA7887F, 0x974C88, 0x36D1F9, 0xB39221, 0x4A827B,
	0x21CF98, 0xDC9F40, 0x5547DC, 0x3A74E1, 0x42EB67, 0xDF9DFE,
	0x5FD45E, 0xA4677B, 0x7AACBA, 0xA2F655, 0x23882B, 0x55BA41,
	0x086E59, 0x862A21, 0x834739, 0xE6E389, 0xD49EE5, 0x40FB49,
	0xE956FF, 0xCA0F1C, 0x8A59C5, 0x2BFA94, 0xC5C1D3, 0xCFC50F,
	0xAE5ADB, 0x86C547, 0x624385, 0x3B8621, 0x94792C, 0x876110,
	0x7B4C2A, 0x1A2C80, 0x12BF43, 0x902688, 0x893C78, 0xE4C4A8,
	0x7BDBE5, 0xC23AC4, 0xEAF426, 0x8A67F7, 0xBF920D, 0x2BA365,
	0xB1933D, 0x0B7CBD, 0xDC51A4, 0x63DD27, 0xDDE169, 0x19949A,
	0x9529A8, 0x28CE68, 0xB4ED09, 0x209F44, 0xCA984E, 0x638270,
	0x237C7E, 0x32B90F, 0x8EF5A7, 0xE75614, 0x08F121, 0x2A9DB5,
	0x4D7E6F, 0x5119A5, 0xABF9B5, 0xD6DF82, 0x61DD96, 0x023616,
	0x9F3AC4, 0xA1A283, 0x6DED72, 0x7A8D39, 0xA9B882, 0x5C326B,
	0x5B2746, 0xED3400, 0x7700D2, 0x55F4FC, 0x4D5901, 0x8071E0,
#endif
	};
	static const double PIo2[] = {
	  1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
	  7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
	  5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
	  3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
	  1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
	  1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
	  2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
	  2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
	};

	int32_t jz,jx,jv,jp,jk,carry,n,iq[20],i,j,k,m,q0,ih;
	double z,fw,f[20],fq[20] = {0},q[20]; /* fq zeroed (edited, see the top) */

	/* initialize jk*/
	jk = init_jk[prec];
	jp = jk;

	/* determine jx,jv,q0, note that 3>q0 */
	jx = nx-1;
	jv = (e0-3)/24;  if(jv<0) jv=0;
	q0 = e0-24*(jv+1);

	/* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
	j = jv-jx; m = jx+jk;
	for (i=0; i<=m; i++,j++)
		f[i] = j<0 ? 0.0 : (double)ipio2[j];

	/* compute q[0],q[1],...q[jk] */
	for (i=0; i<=jk; i++) {
		for (j=0,fw=0.0; j<=jx; j++)
			fw += x[j]*f[jx+i-j];
		q[i] = fw;
	}

	jz = jk;
recompute:
	/* distill q[] into iq[] reversingly */
	for (i=0,j=jz,z=q[jz]; j>0; i++,j--) {
		fw    = (double)(int32_t)(0x1p-24*z);
		iq[i] = (int32_t)(z - 0x1p24*fw);
		z     = q[j-1]+fw;
	}

	/* compute n */
	z  = presk_scalbn(z,q0);       /* actual value of z */
	z -= 8.0*floor(z*0.125); /* trim off integer >= 8 */
	n  = (int32_t)z;
	z -= (double)n;
	ih = 0;
	if (q0 > 0) {  /* need iq[jz-1] to determine n */
		i  = iq[jz-1]>>(24-q0); n += i;
		iq[jz-1] -= i<<(24-q0);
		ih = iq[jz-1]>>(23-q0);
	}
	else if (q0 == 0) ih = iq[jz-1]>>23;
	else if (z >= 0.5) ih = 2;

	if (ih > 0) {  /* q > 0.5 */
		n += 1; carry = 0;
		for (i=0; i<jz; i++) {  /* compute 1-q */
			j = iq[i];
			if (carry == 0) {
				if (j != 0) {
					carry = 1;
					iq[i] = 0x1000000 - j;
				}
			} else
				iq[i] = 0xffffff - j;
		}
		if (q0 > 0) {  /* rare case: chance is 1 in 12 */
			switch(q0) {
			case 1:
				iq[jz-1] &= 0x7fffff; break;
			case 2:
				iq[jz-1] &= 0x3fffff; break;
			}
		}
		if (ih == 2) {
			z = 1.0 - z;
			if (carry != 0)
				z -= presk_scalbn(1.0,q0);
		}
	}

	/* check if recomputation is needed */
	if (z == 0.0) {
		j = 0;
		for (i=jz-1; i>=jk; i--) j |= iq[i];
		if (j == 0) {  /* need recomputation */
			for (k=1; iq[jk-k]==0; k++);  /* k = no. of terms needed */

			for (i=jz+1; i<=jz+k; i++) {  /* add q[jz+1] to q[jz+k] */
				f[jx+i] = (double)ipio2[jv+i];
				for (j=0,fw=0.0; j<=jx; j++)
					fw += x[j]*f[jx+i-j];
				q[i] = fw;
			}
			jz += k;
			goto recompute;
		}
	}

	/* chop off zero terms */
	if (z == 0.0) {
		jz -= 1;
		q0 -= 24;
		while (iq[jz] == 0) {
			jz--;
			q0 -= 24;
		}
	} else { /* break z into 24-bit if necessary */
		z = presk_scalbn(z,-q0);
		if (z >= 0x1p24) {
			fw = (double)(int32_t)(0x1p-24*z);
			iq[jz] = (int32_t)(z - 0x1p24*fw);
			jz += 1;
			q0 += 24;
			iq[jz] = (int32_t)fw;
		} else
			iq[jz] = (int32_t)z;
	}

	/* convert integer "bit" chunk to floating-point value */
	fw = presk_scalbn(1.0,q0);
	for (i=jz; i>=0; i--) {
		q[i] = fw*(double)iq[i];
		fw *= 0x1p-24;
	}

	/* compute PIo2[0,...,jp]*q[jz,...,0] */
	for(i=jz; i>=0; i--) {
		for (fw=0.0,k=0; k<=jp && k<=jz-i; k++)
			fw += PIo2[k]*q[i+k];
		fq[jz-i] = fw;
	}

	/* compress fq[] into y[] */
	switch(prec) {
	case 0:
		fw = 0.0;
		for (i=jz; i>=0; i--)
			fw += fq[i];
		y[0] = ih==0 ? fw : -fw;
		break;
	case 1:
	case 2:
		fw = 0.0;
		for (i=jz; i>=0; i--)
			fw += fq[i];
		// TODO: drop excess precision here once double_t is used
		fw = (double)fw;
		y[0] = ih==0 ? fw : -fw;
		fw = fq[0]-fw;
		for (i=1; i<=jz; i++)
			fw += fq[i];
		y[1] = ih==0 ? fw : -fw;
		break;
	case 3:  /* painful */
		for (i=jz; i>0; i--) {
			fw      = fq[i-1]+fq[i];
			fq[i]  += fq[i-1]-fw;
			fq[i-1] = fw;
		}
		for (i=jz; i>1; i--) {
			fw      = fq[i-1]+fq[i];
			fq[i]  += fq[i-1]-fw;
			fq[i-1] = fw;
		}
		for (fw=0.0,i=jz; i>=2; i--)
			fw += fq[i];
		if (ih==0) {
			y[0] =  fq[0]; y[1] =  fq[1]; y[2] =  fw;
		} else {
			y[0] = -fq[0]; y[1] = -fq[1]; y[2] = -fw;
		}
	}
	return n&7;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__rem_pio2.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * Optimized by Bruce D. Evans.
 */
/* __rem_pio2(x,y)
 *
 * return the remainder of x rem pi/2 in y[0]+y[1]
 * use __rem_pio2_large() for large x
 */

#if FLT_EVAL_METHOD==0 || FLT_EVAL_METHOD==1
#define EPS DBL_EPSILON
#elif FLT_EVAL_METHOD==2
#define EPS LDBL_EPSILON
#endif

/* caller must handle the case when reduction is not needed: |x| ~<= pi/4 */
static int presk_rem_pio2(double x, double *y)
{
	/*
	 * invpio2:  53 bits of 2/pi
	 * pio2_1:   first  33 bit of pi/2
	 * pio2_1t:  pi/2 - pio2_1
	 * pio2_2:   second 33 bit of pi/2
	 * pio2_2t:  pi/2 - (pio2_1+pio2_2)
	 * pio2_3:   third  33 bit of pi/2
	 * pio2_3t:  pi/2 - (pio2_1+pio2_2+pio2_3)
	 */
	static const double
	toint   = 1.5/EPS,
	invpio2 = 6.36619772367581382433e-01, /* 0x3FE45F30, 0x6DC9C883 */
	pio2_1  = 1.57079632673412561417e+00, /* 0x3FF921FB, 0x54400000 */
	pio2_1t = 6.07710050650619224932e-11, /* 0x3DD0B461, 0x1A626331 */
	pio2_2  = 6.07710050630396597660e-11, /* 0x3DD0B461, 0x1A600000 */
	pio2_2t = 2.02226624879595063154e-21, /* 0x3BA3198A, 0x2E037073 */
	pio2_3  = 2.02226624871116645580e-21, /* 0x3BA3198A, 0x2E000000 */
	pio2_3t = 8.47842766036889956997e-32; /* 0x397B839A, 0x252049C1 */

	union {double f; uint64_t i;} u = {x};
	double_t z,w,t,r,fn;
	double tx[3],ty[2];
	uint32_t ix;
	int sign, n, ex, ey, i;

	sign = u.i>>63;
	ix = u.i>>32 & 0x7fffffff;
	if (ix <= 0x400f6a7a) {  /* |x| ~<= 5pi/4 */
		if ((ix & 0xfffff) == 0x921fb)  /* |x| ~= pi/2 or 2pi/2 */
			goto medium;  /* cancellation -- use medium case */
		if (ix <= 0x4002d97c) {  /* |x| ~<= 3pi/4 */
			if (!sign) {
				z = x - pio2_1;  /* one round good to 85 bits */
				y[0] = z - pio2_1t;
				y[1] = (z-y[0]) - pio2_1t;
				return 1;
			} else {
				z = x + pio2_1;
				y[0] = z + pio2_1t;
				y[1] = (z-y[0]) + pio2_1t;
				return -1;
			}
		} else {
			if (!sign) {
				z = x - 2*pio2_1;
				y[0] = z - 2*pio2_1t;
				y[1] = (z-y[0]) - 2*pio2_1t;
				return 2;
			} else {
				z = x + 2*pio2_1;
				y[0] = z + 2*pio2_1t;
				y[1] = (z-y[0]) + 2*pio2_1t;
				return -2;
			}
		}
	}
	if (ix <= 0x401c463b) {  /* |x| ~<= 9pi/4 */
		if (ix <= 0x4015fdbc) {  /* |x| ~<= 7pi/4 */
			if (ix == 0x4012d97c)  /* |x| ~= 3pi/2 */
				goto medium;
			if (!sign) {
				z = x - 3*pio2_1;
				y[0] = z - 3*pio2_1t;
				y[1] = (z-y[0]) - 3*pio2_1t;
				return 3;
			} else {
				z = x + 3*pio2_1;
				y[0] = z + 3*pio2_1t;
				y[1] = (z-y[0]) + 3*pio2_1t;
				return -3;
			}
		} else {
			if (ix == 0x401921fb)  /* |x| ~= 4pi/2 */
				goto medium;
			if (!sign) {
				z = x - 4*pio2_1;
				y[0] = z - 4*pio2_1t;
				y[1] = (z-y[0]) - 4*pio2_1t;
				return 4;
			} else {
				z = x + 4*pio2_1;
				y[0] = z + 4*pio2_1t;
				y[1] = (z-y[0]) + 4*pio2_1t;
				return -4;
			}
		}
	}
	if (ix < 0x413921fb) {  /* |x| ~< 2^20*(pi/2), medium size */
medium:
		/* rint(x/(pi/2)), Assume round-to-nearest. */
		fn = (double_t)x*invpio2 + toint - toint;
		n = (int32_t)fn;
		r = x - fn*pio2_1;
		w = fn*pio2_1t;  /* 1st round, good to 85 bits */
		y[0] = r - w;
		u.f = y[0];
		ey = u.i>>52 & 0x7ff;
		ex = ix>>20;
		if (ex - ey > 16) { /* 2nd round, good to 118 bits */
			t = r;
			w = fn*pio2_2;
			r = t - w;
			w = fn*pio2_2t - ((t-r)-w);
			y[0] = r - w;
			u.f = y[0];
			ey = u.i>>52 & 0x7ff;
			if (ex - ey > 49) {  /* 3rd round, good to 151 bits, covers all cases */
				t = r;
				w = fn*pio2_3;
				r = t - w;
				w = fn*pio2_3t - ((t-r)-w);
				y[0] = r - w;
			}
		}
		y[1] = (r - y[0]) - w;
		return n;
	}
	/*
	 * all other (large) arguments
	 */
	if (ix >= 0x7ff00000) {  /* x is inf or NaN */
		y[0] = y[1] = x - x;
		return 0;
	}
	/* set z = scalbn(|x|,-ilogb(x)+23) */
	u.f = x;
	u.i &= (uint64_t)-1>>12;
	u.i |= (uint64_t)(0x3ff + 23)<<52;
	z = u.f;
	for (i=0; i < 2; i++) {
		tx[i] = (double)(int32_t)z;
		z     = (z-tx[i])*0x1p24;
	}
	tx[i] = z;
	/* skip zero terms, first term is non-zero */
	while (tx[i] == 0.0)
		i--;
	n = presk_rem_pio2_large(tx,ty,(int)(ix>>20)-(0x3ff+23),i+1,1);
	if (sign) {
		y[0] = -ty[0];
		y[1] = -ty[1];
		return -n;
	}
	y[0] = ty[0];
	y[1] = ty[1];
	return n;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/__rem_pio2f.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2f.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Debugged and optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* __rem_pio2f(x,y)
 *
 * return the remainder of x rem pi/2 in *y
 * use double precision for everything except passing x
 * use __rem_pio2_large() for large x
 */

#if FLT_EVAL_METHOD==0 || FLT_EVAL_METHOD==1
#define EPS DBL_EPSILON
#elif FLT_EVAL_METHOD==2
#define EPS LDBL_EPSILON
#endif

static int presk_rem_pio2f(float x, double *y)
{
	/*
	 * invpio2:  53 bits of 2/pi
	 * pio2_1:   first 25 bits of pi/2
	 * pio2_1t:  pi/2 - pio2_1
	 */
	static const double
	toint   = 1.5/EPS,
	invpio2 = 6.36619772367581382433e-01, /* 0x3FE45F30, 0x6DC9C883 */
	pio2_1  = 1.57079631090164184570e+00, /* 0x3FF921FB, 0x50000000 */
	pio2_1t = 1.58932547735281966916e-08; /* 0x3E5110b4, 0x611A6263 */

	union {float f; uint32_t i;} u = {x};
	double tx[1],ty[1];
	double_t fn;
	uint32_t ix;
	int n, sign, e0;

	ix = u.i & 0x7fffffff;
	/* 25+53 bit pi is good enough for medium size */
	if (ix < 0x4dc90fdb) {  /* |x| ~< 2^28*(pi/2), medium size */
		/* Use a specialized rint() to get fn.  Assume round-to-nearest. */
		fn = (double_t)x*invpio2 + toint - toint;
		n  = (int32_t)fn;
		*y = x - fn*pio2_1 - fn*pio2_1t;
		return n;
	}
	if(ix>=0x7f800000) {  /* x is inf or NaN */
		*y = x-x;
		return 0;
	}
	/* scale x into [2^23, 2^24-1] */
	sign = u.i>>31;
	e0 = (ix>>23) - (0x7f+23);  /* e0 = ilogb(|x|)-23, positive */
	u.i = ix - (e0<<23);
	tx[0] = u.f;
	n  =  presk_rem_pio2_large(tx,ty,e0,1,0);
	if (sign) {
		*y = -ty[0];
		return -n;
	}
	*y = ty[0];
	return n;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/exp.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_exp.c */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* exp(x)
 * Returns the exponential of x.
 *
 * Method
 *   1. Argument reduction:
 *      Reduce x to an r so that |r| <= 0.5*ln2 ~ 0.34658.
 *      Given x, find r and integer k such that
 *
 *               x = k*ln2 + r,  |r| <= 0.5*ln2.
 *
 *      Here r will be represented as r = hi-lo for better
 *      accuracy.
 *
 *   2. Approximation of exp(r) by a special rational function on
 *      the interval [0,0.34658]:
 *      Write
 *          R(r**2) = r*(exp(r)+1)/(exp(r)-1) = 2 + r*r/6 - r**4/360 + ...
 *      We use a special Remez algorithm on [0,0.34658] to generate
 *      a polynomial of degree 5 to approximate R. The maximum error
 *      of this polynomial approximation is bounded by 2**-59. In
 *      other words,
 *          R(z) ~ 2.0 + P1*z + P2*z**2 + P3*z**3 + P4*z**4 + P5*z**5
 *      (where z=r*r, and the values of P1 to P5 are listed below)
 *      and
 *          |                  5          |     -59
 *          | 2.0+P1*z+...+P5*z   -  R(z) | <= 2
 *          |                             |
 *      The computation of exp(r) thus becomes
 *                              2*r
 *              exp(r) = 1 + ----------
 *                            R(r) - r
 *                                 r*c(r)
 *                     = 1 + r + ----------- (for better accuracy)
 *                                2 - c(r)
 *      where
 *                              2       4             10
 *              c(r) = r - (P1*r  + P2*r  + ... + P5*r   ).
 *
 *   3. Scale back to obtain exp(x):
 *      From step 1, we have
 *         exp(x) = 2^k * exp(r)
 *
 * Special cases:
 *      exp(INF) is INF, exp(NaN) is NaN;
 *      exp(-INF) is 0, and
 *      for finite argument, only exp(0)=1 is exact.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Misc. info.
 *      For IEEE double
 *          if x >  709.782712893383973096 then exp(x) overflows
 *          if x < -745.133219101941108420 then exp(x) underflows
 */

double presk_exp(double x)
{
	static const double
	half[2] = {0.5,-0.5},
	ln2hi = 6.93147180369123816490e-01, /* 0x3fe62e42, 0xfee00000 */
	ln2lo = 1.90821492927058770002e-10, /* 0x3dea39ef, 0x35793c76 */
	invln2 = 1.44269504088896338700e+00, /* 0x3ff71547, 0x652b82fe */
	P1   =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
	P2   = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
	P3   =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
	P4   = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
	P5   =  4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */

	double_t hi, lo, c, xx, y;
	int k, sign;
	uint32_t hx;

	GET_HIGH_WORD(hx, x);
	sign = hx>>31;
	hx &= 0x7fffffff;  /* high word of |x| */

	/* special cases */
	if (hx >= 0x4086232b) {  /* if |x| >= 708.39... */
		if (isnan(x))
			return x;
		if (x > 709.782712893383973096) {
			/* overflow if x!=inf */
			x *= 0x1p1023;
			return x;
		}
		if (x < -708.39641853226410622) {
			/* underflow if x!=-inf */
			FORCE_EVAL((float)(-0x1p-149/x));
			if (x < -745.13321910194110842)
				return 0;
		}
	}

	/* argument reduction */
	if (hx > 0x3fd62e42) {  /* if |x| > 0.5 ln2 */
		if (hx >= 0x3ff0a2b2)  /* if |x| >= 1.5 ln2 */
			k = (int)(invln2*x + half[sign]);
		else
			k = 1 - sign - sign;
		hi = x - k*ln2hi;  /* k*ln2hi is exact here */
		lo = k*ln2lo;
		x = hi - lo;
	} else if (hx > 0x3e300000)  {  /* if |x| > 2**-28 */
		k = 0;
		hi = x;
		lo = 0;
	} else {
		/* inexact if x!=0 */
		FORCE_EVAL(0x1p1023 + x);
		return 1 + x;
	}

	/* x is now in primary range */
	xx = x*x;
	c = x - xx*(P1+xx*(P2+xx*(P3+xx*(P4+xx*P5))));
	y = 1 + (x*c/(2-c) - lo + hi);
	if (k == 0)
		return y;
	return presk_scalbn(y, k);
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/expf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_expf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

float presk_expf(float x)
{
	static const float
	half[2] = {0.5,-0.5},
	ln2hi   = 6.9314575195e-1f,  /* 0x3f317200 */
	ln2lo   = 1.4286067653e-6f,  /* 0x35bfbe8e */
	invln2  = 1.4426950216e+0f,  /* 0x3fb8aa3b */
	/*
	 * Domain [-0.34568, 0.34568], range ~[-4.278e-9, 4.447e-9]:
	 * |x*(exp(x)+1)/(exp(x)-1) - p(x)| < 2**-27.74
	 */
	P1 =  1.6666625440e-1f, /*  0xaaaa8f.0p-26 */
	P2 = -2.7667332906e-3f; /* -0xb55215.0p-32 */

	float_t hi, lo, c, xx, y;
	int k, sign;
	uint32_t hx;

	GET_FLOAT_WORD(hx, x);
	sign = hx >> 31;   /* sign bit of x */
	hx &= 0x7fffffff;  /* high word of |x| */

	/* special cases */
	if (hx >= 0x42aeac50) {  /* if |x| >= -87.33655f or NaN */
		if (hx > 0x7f800000) /* NaN */
			return x;
		if (hx >= 0x42b17218 && !sign) {  /* x >= 88.722839f */
			/* overflow */
			x *= 0x1p127f;
			return x;
		}
		if (sign) {
			/* underflow */
			FORCE_EVAL(-0x1p-149f/x);
			if (hx >= 0x42cff1b5)  /* x <= -103.972084f */
				return 0;
		}
	}

	/* argument reduction */
	if (hx > 0x3eb17218) {  /* if |x| > 0.5 ln2 */
		if (hx > 0x3f851592)  /* if |x| > 1.5 ln2 */
			k = invln2*x + half[sign];
		else
			k = 1 - sign - sign;
		hi = x - k*ln2hi;  /* k*ln2hi is exact here */
		lo = k*ln2lo;
		x = hi - lo;
	} else if (hx > 0x39000000) {  /* |x| > 2**-14 */
		k = 0;
		hi = x;
		lo = 0;
	} else {
		/* raise inexact */
		FORCE_EVAL(0x1p127f + x);
		return 1 + x;
	}

	/* x is now in primary range */
	xx = x*x;
	c = x - xx*(P1+xx*P2);
	y = 1 + (x*c/(2-c) - lo + hi);
	if (k == 0)
		return y;
	return presk_scalbnf(y, k);
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/log.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_log.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* log(x)
 * Return the logarithm of x
 *
 * Method :
 *   1. Argument Reduction: find k and f such that
 *                      x = 2^k * (1+f),
 *         where  sqrt(2)/2 < 1+f < sqrt(2) .
 *
 *   2. Approximation of log(1+f).
 *      Let s = f/(2+f) ; based on log(1+f) = log(1+s) - log(1-s)
 *               = 2s + 2/3 s**3 + 2/5 s**5 + .....,
 *               = 2s + s*R
 *      We use a special Remez algorithm on [0,0.1716] to generate
 *      a polynomial of degree 14 to approximate R The maximum error
 *      of this polynomial approximation is bounded by 2**-58.45. In
 *      other words,
 *                      2      4      6      8      10      12      14
 *          R(z) ~ Lg1*s +Lg2*s +Lg3*s +Lg4*s +Lg5*s  +Lg6*s  +Lg7*s
 *      (the values of Lg1 to Lg7 are listed in the program)
 *      and
 *          |      2          14          |     -58.45
 *          | Lg1*s +...+Lg7*s    -  R(z) | <= 2
 *          |                             |
 *      Note that 2s = f - s*f = f - hfsq + s*hfsq, where hfsq = f*f/2.
 *      In order to guarantee error in log below 1ulp, we compute log
 *      by
 *              log(1+f) = f - s*(f - R)        (if f is not too large)
 *              log(1+f) = f - (hfsq - s*(hfsq+R)).     (better accuracy)
 *
 *      3. Finally,  log(x) = k*ln2 + log(1+f).
 *                          = k*ln2_hi+(f-(hfsq-(s*(hfsq+R)+k*ln2_lo)))
 *         Here ln2 is split into two floating point number:
 *                      ln2_hi + ln2_lo,
 *         where n*ln2_hi is always exact for |n| < 2000.
 *
 * Special cases:
 *      log(x) is NaN with signal if x < 0 (including -INF) ;
 *      log(+INF) is +INF; log(0) is -INF with signal;
 *      log(NaN) is that NaN with no signal.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */

double presk_log(double x)
{
	static const double
	ln2_hi = 6.93147180369123816490e-01,  /* 3fe62e42 fee00000 */
	ln2_lo = 1.90821492927058770002e-10,  /* 3dea39ef 35793c76 */
	Lg1 = 6.666666666666735130e-01,  /* 3FE55555 55555593 */
	Lg2 = 3.999999999940941908e-01,  /* 3FD99999 9997FA04 */
	Lg3 = 2.857142874366239149e-01,  /* 3FD24924 94229359 */
	Lg4 = 2.222219843214978396e-01,  /* 3FCC71C5 1D8E78AF */
	Lg5 = 1.818357216161805012e-01,  /* 3FC74664 96CB03DE */
	Lg6 = 1.531383769920937332e-01,  /* 3FC39A09 D078C69F */
	Lg7 = 1.479819860511658591e-01;  /* 3FC2F112 DF3E5244 */

	union {double f; uint64_t i;} u = {x};
	double_t hfsq,f,s,z,R,w,t1,t2,dk;
	uint32_t hx;
	int k;

	hx = u.i>>32;
	k = 0;
	if (hx < 0x00100000 || hx>>31) {
		if (u.i<<1 == 0)
			return -1/(x*x);  /* log(+-0)=-inf */
		if (hx>>31)
			return (x-x)/0.0; /* log(-#) = NaN */
		/* subnormal number, scale x up */
		k -= 54;
		x *= 0x1p54;
		u.f = x;
		hx = u.i>>32;
	} else if (hx >= 0x7ff00000) {
		return x;
	} else if (hx == 0x3ff00000 && u.i<<32 == 0)
		return 0;

	/* reduce x into [sqrt(2)/2, sqrt(2)] */
	hx += 0x3ff00000 - 0x3fe6a09e;
	k += (int)(hx>>20) - 0x3ff;
	hx = (hx&0x000fffff) + 0x3fe6a09e;
	u.i = (uint64_t)hx<<32 | (u.i&0xffffffff);
	x = u.f;

	f = x - 1.0;
	hfsq = 0.5*f*f;
	s = f/(2.0+f);
	z = s*s;
	w = z*z;
	t1 = w*(Lg2+w*(Lg4+w*Lg6));
	t2 = z*(Lg1+w*(Lg3+w*(Lg5+w*Lg7)));
	R = t2 + t1;
	dk = k;
	return s*(hfsq+R) + dk*ln2_lo - hfsq + f + dk*ln2_hi;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/logf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/e_logf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

float presk_logf(float x)
{
	static const float
	ln2_hi = 6.9313812256e-01, /* 0x3f317180 */
	ln2_lo = 9.0580006145e-06, /* 0x3717f7d1 */
	/* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
	Lg1 = 0xaaaaaa.0p-24, /* 0.66666662693 */
	Lg2 = 0xccce13.0p-25, /* 0.40000972152 */
	Lg3 = 0x91e9ee.0p-25, /* 0.28498786688 */
	Lg4 = 0xf89e26.0p-26; /* 0.24279078841 */

	union {float f; uint32_t i;} u = {x};
	float_t hfsq,f,s,z,R,w,t1,t2,dk;
	uint32_t ix;
	int k;

	ix = u.i;
	k = 0;
	if (ix < 0x00800000 || ix>>31) {  /* x < 2**-126  */
		if (ix<<1 == 0)
			return -1/(x*x);  /* log(+-0)=-inf */
		if (ix>>31)
			return (x-x)/0.0f; /* log(-#) = NaN */
		/* subnormal number, scale up x */
		k -= 25;
		x *= 0x1p25f;
		u.f = x;
		ix = u.i;
	} else if (ix >= 0x7f800000) {
		return x;
	} else if (ix == 0x3f800000)
		return 0;

	/* reduce x into [sqrt(2)/2, sqrt(2)] */
	ix += 0x3f800000 - 0x3f3504f3;
	k += (int)(ix>>23) - 0x7f;
	ix = (ix&0x007fffff) + 0x3f3504f3;
	u.i = ix;
	x = u.f;

	f = x - 1.0f;
	s = f/(2.0f + f);
	z = s*s;
	w = z*z;
	t1= w*(Lg2+w*Lg4);
	t2= z*(Lg1+w*Lg3);
	R = t2 + t1;
	hfsq = 0.5f*f*f;
	dk = k;
	return s*(hfsq+R) + dk*ln2_lo - hfsq + f + dk*ln2_hi;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/tanh.c
 * ------------------------------------------------------------------------ */

/* tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
 *         = (exp(2*x) - 1)/(exp(2*x) - 1 + 2)
 *         = (1 - exp(-2*x))/(exp(-2*x) - 1 + 2)
 */
double presk_tanh(double x)
{
	union {double f; uint64_t i;} u = {.f = x};
	uint32_t w;
	int sign;
	double_t t;

	/* x = |x| */
	sign = u.i >> 63;
	u.i &= (uint64_t)-1/2;
	x = u.f;
	w = u.i >> 32;

	if (w > 0x3fe193ea) {
		/* |x| > log(3)/2 ~= 0.5493 or nan */
		if (w > 0x40340000) {
			/* |x| > 20 or nan */
			/* note: this branch avoids raising overflow */
			t = 1 - 0/x;
		} else {
			t = presk_expm1(2*x);
			t = 1 - 2/(t+2);
		}
	} else if (w > 0x3fd058ae) {
		/* |x| > log(5/3)/2 ~= 0.2554 */
		t = presk_expm1(2*x);
		t = t/(t+2);
	} else if (w >= 0x00100000) {
		/* |x| >= 0x1p-1022, up to 2ulp error in [0.1,0.2554] */
		t = presk_expm1(-2*x);
		t = -t/(t+2);
	} else {
		/* |x| is subnormal */
		/* note: the branch above would not raise underflow in [0x1p-1023,0x1p-1022) */
		FORCE_EVAL((float)x);
		t = x;
	}
	return sign ? -t : t;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/tanhf.c
 * ------------------------------------------------------------------------ */

float presk_tanhf(float x)
{
	union {float f; uint32_t i;} u = {.f = x};
	uint32_t w;
	int sign;
	float t;

	/* x = |x| */
	sign = u.i >> 31;
	u.i &= 0x7fffffff;
	x = u.f;
	w = u.i;

	if (w > 0x3f0c9f54) {
		/* |x| > log(3)/2 ~= 0.5493 or nan */
		if (w > 0x41200000) {
			/* |x| > 10 */
			t = 1 + 0/x;
		} else {
			t = presk_expm1f(2*x);
			t = 1 - 2/(t+2);
		}
	} else if (w > 0x3e82c578) {
		/* |x| > log(5/3)/2 ~= 0.2554 */
		t = presk_expm1f(2*x);
		t = t/(t+2);
	} else if (w >= 0x00800000) {
		/* |x| >= 0x1p-126 */
		t = presk_expm1f(-2*x);
		t = -t/(t+2);
	} else {
		/* |x| is subnormal */
		FORCE_EVAL(x*x);
		t = x;
	}
	return sign ? -t : t;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/sin.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_sin.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* sin(x)
 * Return sine function of x.
 *
 * kernel function:
 *      __sin            ... sine function on [-pi/4,pi/4]
 *      __cos            ... cose function on [-pi/4,pi/4]
 *      __rem_pio2       ... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *      in [-pi/4 , +pi/4], and let n = k mod 4.
 *      We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *          0          S           C             T
 *          1          C          -S            -1/T
 *          2         -S          -C             T
 *          3         -C           S            -1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *      TRIG(x) returns trig(x) nearly rounded
 */

double presk_sin(double x)
{
	double y[2];
	uint32_t ix;
	unsigned n;

	/* High word of x. */
	GET_HIGH_WORD(ix, x);
	ix &= 0x7fffffff;

	/* |x| ~< pi/4 */
	if (ix <= 0x3fe921fb) {
		if (ix < 0x3e500000) {  /* |x| < 2**-26 */
			/* raise inexact if x != 0 and underflow if subnormal*/
			FORCE_EVAL(ix < 0x00100000 ? x/0x1p120f : x+0x1p120f);
			return x;
		}
		return presk_k_sin(x, 0.0, 0);
	}

	/* sin(Inf or NaN) is NaN */
	if (ix >= 0x7ff00000)
		return x - x;

	/* argument reduction needed */
	n = presk_rem_pio2(x, y);
	switch (n&3) {
	case 0: return  presk_k_sin(y[0], y[1], 1);
	case 1: return  presk_k_cos(y[0], y[1]);
	case 2: return -presk_k_sin(y[0], y[1], 1);
	default:
		return -presk_k_cos(y[0], y[1]);
	}
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/sinf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_sinf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

float presk_sinf(float x)
{
	/* Small multiples of pi/2 rounded to double precision. */
	static const double
	s1pio2 = 1*M_PI_2, /* 0x3FF921FB, 0x54442D18 */
	s2pio2 = 2*M_PI_2, /* 0x400921FB, 0x54442D18 */
	s3pio2 = 3*M_PI_2, /* 0x4012D97C, 0x7F3321D2 */
	s4pio2 = 4*M_PI_2; /* 0x401921FB, 0x54442D18 */

	double y;
	uint32_t ix;
	int n, sign;

	GET_FLOAT_WORD(ix, x);
	sign = ix >> 31;
	ix &= 0x7fffffff;

	if (ix <= 0x3f490fda) {  /* |x| ~<= pi/4 */
		if (ix < 0x39800000) {  /* |x| < 2**-12 */
			/* raise inexact if x!=0 and underflow if subnormal */
			FORCE_EVAL(ix < 0x00800000 ? x/0x1p120f : x+0x1p120f);
			return x;
		}
		return presk_k_sindf(x);
	}
	if (ix <= 0x407b53d1) {  /* |x| ~<= 5*pi/4 */
		if (ix <= 0x4016cbe3) {  /* |x| ~<= 3pi/4 */
			if (sign)
				return -presk_k_cosdf(x + s1pio2);
			else
				return presk_k_cosdf(x - s1pio2);
		}
		return presk_k_sindf(sign ? -(x + s2pio2) : -(x - s2pio2));
	}
	if (ix <= 0x40e231d5) {  /* |x| ~<= 9*pi/4 */
		if (ix <= 0x40afeddf) {  /* |x| ~<= 7*pi/4 */
			if (sign)
				return presk_k_cosdf(x + s3pio2);
			else
				return -presk_k_cosdf(x - s3pio2);
		}
		return presk_k_sindf(sign ? x + s4pio2 : x - s4pio2);
	}

	/* sin(Inf or NaN) is NaN */
	if (ix >= 0x7f800000)
		return x - x;

	/* general argument reduction needed */
	n = presk_rem_pio2f(x, &y);
	switch (n&3) {
	case 0: return  presk_k_sindf(y);
	case 1: return  presk_k_cosdf(y);
	case 2: return  presk_k_sindf(-y);
	default:
		return -presk_k_cosdf(y);
	}
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/cos.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_cos.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* cos(x)
 * Return cosine function of x.
 *
 * kernel function:
 *      __sin           ... sine function on [-pi/4,pi/4]
 *      __cos           ... cosine function on [-pi/4,pi/4]
 *      __rem_pio2      ... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *      in [-pi/4 , +pi/4], and let n = k mod 4.
 *      We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *          0          S           C             T
 *          1          C          -S            -1/T
 *          2         -S          -C             T
 *          3         -C           S            -1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *      TRIG(x) returns trig(x) nearly rounded
 */

double presk_cos(double x)
{
	double y[2];
	uint32_t ix;
	unsigned n;

	GET_HIGH_WORD(ix, x);
	ix &= 0x7fffffff;

	/* |x| ~< pi/4 */
	if (ix <= 0x3fe921fb) {
		if (ix < 0x3e46a09e) {  /* |x| < 2**-27 * sqrt(2) */
			/* raise inexact if x!=0 */
			FORCE_EVAL(x + 0x1p120f);
			return 1.0;
		}
		return presk_k_cos(x, 0);
	}

	/* cos(Inf or NaN) is NaN */
	if (ix >= 0x7ff00000)
		return x-x;

	/* argument reduction */
	n = presk_rem_pio2(x, y);
	switch (n&3) {
	case 0: return  presk_k_cos(y[0], y[1]);
	case 1: return -presk_k_sin(y[0], y[1], 1);
	case 2: return -presk_k_cos(y[0], y[1]);
	default:
		return  presk_k_sin(y[0], y[1], 1);
	}
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/cosf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_cosf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

float presk_cosf(float x)
{
	/* Small multiples of pi/2 rounded to double precision. */
	static const double
	c1pio2 = 1*M_PI_2, /* 0x3FF921FB, 0x54442D18 */
	c2pio2 = 2*M_PI_2, /* 0x400921FB, 0x54442D18 */
	c3pio2 = 3*M_PI_2, /* 0x4012D97C, 0x7F3321D2 */
	c4pio2 = 4*M_PI_2; /* 0x401921FB, 0x54442D18 */

	double y;
	uint32_t ix;
	unsigned n, sign;

	GET_FLOAT_WORD(ix, x);
	sign = ix >> 31;
	ix &= 0x7fffffff;

	if (ix <= 0x3f490fda) {  /* |x| ~<= pi/4 */
		if (ix < 0x39800000) {  /* |x| < 2**-12 */
			/* raise inexact if x != 0 */
			FORCE_EVAL(x + 0x1p120f);
			return 1.0f;
		}
		return presk_k_cosdf(x);
	}
	if (ix <= 0x407b53d1) {  /* |x| ~<= 5*pi/4 */
		if (ix > 0x4016cbe3)  /* |x|  ~> 3*pi/4 */
			return -presk_k_cosdf(sign ? x+c2pio2 : x-c2pio2);
		else {
			if (sign)
				return presk_k_sindf(x + c1pio2);
			else
				return presk_k_sindf(c1pio2 - x);
		}
	}
	if (ix <= 0x40e231d5) {  /* |x| ~<= 9*pi/4 */
		if (ix > 0x40afeddf)  /* |x| ~> 7*pi/4 */
			return presk_k_cosdf(sign ? x+c4pio2 : x-c4pio2);
		else {
			if (sign)
				return presk_k_sindf(-x - c3pio2);
			else
				return presk_k_sindf(x - c3pio2);
		}
	}

	/* cos(Inf or NaN) is NaN */
	if (ix >= 0x7f800000)
		return x-x;

	/* general argument reduction needed */
	n = presk_rem_pio2f(x,&y);
	switch (n&3) {
	case 0: return  presk_k_cosdf(y);
	case 1: return  presk_k_sindf(-y);
	case 2: return -presk_k_cosdf(y);
	default:
		return  presk_k_sindf(y);
	}
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/erf.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_erf.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* double erf(double x)
 * double erfc(double x)
 *                           x
 *                    2      |\
 *     erf(x)  =  ---------  | exp(-t*t)dt
 *                 sqrt(pi) \|
 *                           0
 *
 *     erfc(x) =  1-erf(x)
 *  Note that
 *              erf(-x) = -erf(x)
 *              erfc(-x) = 2 - erfc(x)
 *
 * Method:
 *      1. For |x| in [0, 0.84375]
 *          erf(x)  = x + x*R(x^2)
 *          erfc(x) = 1 - erf(x)           if x in [-.84375,0.25]
 *                  = 0.5 + ((0.5-x)-x*R)  if x in [0.25,0.84375]
 *         where R = P/Q where P is an odd poly of degree 8 and
 *         Q is an odd poly of degree 10.
 *                                               -57.90
 *                      | R - (erf(x)-x)/x | <= 2
 *
 *
 *         Remark. The formula is derived by noting
 *          erf(x) = (2/sqrt(pi))*(x - x^3/3 + x^5/10 - x^7/42 + ....)
 *         and that
 *          2/sqrt(pi) = 1.128379167095512573896158903121545171688
 *         is close to one. The interval is chosen because the fix
 *         point of erf(x) is near 0.6174 (i.e., erf(x)=x when x is
 *         near 0.6174), and by some experiment, 0.84375 is chosen to
 *         guarantee the error is less than one ulp for erf.
 *
 *      2. For |x| in [0.84375,1.25], let s = |x| - 1, and
 *         c = 0.84506291151 rounded to single (24 bits)
 *              erf(x)  = sign(x) * (c  + P1(s)/Q1(s))
 *              erfc(x) = (1-c)  - P1(s)/Q1(s) if x > 0
 *                        1+(c+P1(s)/Q1(s))    if x < 0
 *              |P1/Q1 - (erf(|x|)-c)| <= 2**-59.06
 *         Remark: here we use the taylor series expansion at x=1.
 *              erf(1+s) = erf(1) + s*Poly(s)
 *                       = 0.845.. + P1(s)/Q1(s)
 *         That is, we use rational approximation to approximate
 *                      erf(1+s) - (c = (single)0.84506291151)
 *         Note that |P1/Q1|< 0.078 for x in [0.84375,1.25]
 *         where
 *              P1(s) = degree 6 poly in s
 *              Q1(s) = degree 6 poly in s
 *
 *      3. For x in [1.25,1/0.35(~2.857143)],
 *              erfc(x) = (1/x)*exp(-x*x-0.5625+R1/S1)
 *              erf(x)  = 1 - erfc(x)
 *         where
 *              R1(z) = degree 7 poly in z, (z=1/x^2)
 *              S1(z) = degree 8 poly in z
 *
 *      4. For x in [1/0.35,28]
 *              erfc(x) = (1/x)*exp(-x*x-0.5625+R2/S2) if x > 0
 *                      = 2.0 - (1/x)*exp(-x*x-0.5625+R2/S2) if -6<x<0
 *                      = 2.0 - tiny            (if x <= -6)
 *              erf(x)  = sign(x)*(1.0 - erfc(x)) if x < 6, else
 *              erf(x)  = sign(x)*(1.0 - tiny)
 *         where
 *              R2(z) = degree 6 poly in z, (z=1/x^2)
 *              S2(z) = degree 7 poly in z
 *
 *      Note1:
 *         To compute exp(-x*x-0.5625+R/S), let s be a single
 *         precision number and s := x; then
 *              -x*x = -s*s + (s-x)*(s+x)
 *              exp(-x*x-0.5626+R/S) =
 *                      exp(-s*s-0.5625)*exp((s-x)*(s+x)+R/S);
 *      Note2:
 *         Here 4 and 5 make use of the asymptotic series
 *                        exp(-x*x)
 *              erfc(x) ~ ---------- * ( 1 + Poly(1/x^2) )
 *                        x*sqrt(pi)
 *         We use rational approximation to approximate
 *              g(s)=f(1/x^2) = log(erfc(x)*x) - x*x + 0.5625
 *         Here is the error bound for R1/S1 and R2/S2
 *              |R1/S1 - f(x)|  < 2**(-62.57)
 *              |R2/S2 - f(x)|  < 2**(-61.52)
 *
 *      5. For inf > x >= 28
 *              erf(x)  = sign(x) *(1 - tiny)  (raise inexact)
 *              erfc(x) = tiny*tiny (raise underflow) if x > 0
 *                      = 2 - tiny if x<0
 *
 *      7. Special case:
 *              erf(0)  = 0, erf(inf)  = 1, erf(-inf) = -1,
 *              erfc(0) = 1, erfc(inf) = 0, erfc(-inf) = 2,
 *              erfc/erf(NaN) is NaN
 */

static double presk_erfc1(double x)
{
	static const double
	erx  = 8.45062911510467529297e-01, /* 0x3FEB0AC1, 0x60000000 */
	/*
	 * Coefficients for approximation to  erf  in [0.84375,1.25]
	 */
	pa0  = -2.36211856075265944077e-03, /* 0xBF6359B8, 0xBEF77538 */
	pa1  =  4.14856118683748331666e-01, /* 0x3FDA8D00, 0xAD92B34D */
	pa2  = -3.72207876035701323847e-01, /* 0xBFD7D240, 0xFBB8C3F1 */
	pa3  =  3.18346619901161753674e-01, /* 0x3FD45FCA, 0x805120E4 */
	pa4  = -1.10894694282396677476e-01, /* 0xBFBC6398, 0x3D3E28EC */
	pa5  =  3.54783043256182359371e-02, /* 0x3FA22A36, 0x599795EB */
	pa6  = -2.16637559486879084300e-03, /* 0xBF61BF38, 0x0A96073F */
	qa1  =  1.06420880400844228286e-01, /* 0x3FBB3E66, 0x18EEE323 */
	qa2  =  5.40397917702171048937e-01, /* 0x3FE14AF0, 0x92EB6F33 */
	qa3  =  7.18286544141962662868e-02, /* 0x3FB2635C, 0xD99FE9A7 */
	qa4  =  1.26171219808761642112e-01, /* 0x3FC02660, 0xE763351F */
	qa5  =  1.36370839120290507362e-02, /* 0x3F8BEDC2, 0x6B51DD1C */
	qa6  =  1.19844998467991074170e-02; /* 0x3F888B54, 0x5735151D */

	double_t s,P,Q;

	s = fabs(x) - 1;
	P = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))));
	Q = 1+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
	return 1 - erx - P/Q;
}

static double presk_erfc2(uint32_t ix, double x)
{
	static const double
	/*
	 * Coefficients for approximation to  erfc in [1.25,1/0.35]
	 */
	ra0  = -9.86494403484714822705e-03, /* 0xBF843412, 0x600D6435 */
	ra1  = -6.93858572707181764372e-01, /* 0xBFE63416, 0xE4BA7360 */
	ra2  = -1.05586262253232909814e+01, /* 0xC0251E04, 0x41B0E726 */
	ra3  = -6.23753324503260060396e+01, /* 0xC04F300A, 0xE4CBA38D */
	ra4  = -1.62396669462573470355e+02, /* 0xC0644CB1, 0x84282266 */
	ra5  = -1.84605092906711035994e+02, /* 0xC067135C, 0xEBCCABB2 */
	ra6  = -8.12874355063065934246e+01, /* 0xC0545265, 0x57E4D2F2 */
	ra7  = -9.81432934416914548592e+00, /* 0xC023A0EF, 0xC69AC25C */
	sa1  =  1.96512716674392571292e+01, /* 0x4033A6B9, 0xBD707687 */
	sa2  =  1.37657754143519042600e+02, /* 0x4061350C, 0x526AE721 */
	sa3  =  4.34565877475229228821e+02, /* 0x407B290D, 0xD58A1A71 */
	sa4  =  6.45387271733267880336e+02, /* 0x40842B19, 0x21EC2868 */
	sa5  =  4.29008140027567833386e+02, /* 0x407AD021, 0x57700314 */
	sa6  =  1.08635005541779435134e+02, /* 0x405B28A3, 0xEE48AE2C */
	sa7  =  6.57024977031928170135e+00, /* 0x401A47EF, 0x8E484A93 */
	sa8  = -6.04244152148580987438e-02, /* 0xBFAEEFF2, 0xEE749A62 */
	/*
	 * Coefficients for approximation to  erfc in [1/.35,28]
	 */
	rb0  = -9.86494292470009928597e-03, /* 0xBF843412, 0x39E86F4A */
	rb1  = -7.99283237680523006574e-01, /* 0xBFE993BA, 0x70C285DE */
	rb2  = -1.77579549177547519889e+01, /* 0xC031C209, 0x555F995A */
	rb3  = -1.60636384855821916062e+02, /* 0xC064145D, 0x43C5ED98 */
	rb4  = -6.37566443368389627722e+02, /* 0xC083EC88, 0x1375F228 */
	rb5  = -1.02509513161107724954e+03, /* 0xC0900461, 0x6A2E5992 */
	rb6  = -4.83519191608651397019e+02, /* 0xC07E384E, 0x9BDC383F */
	sb1  =  3.03380607434824582924e+01, /* 0x403E568B, 0x261D5190 */
	sb2  =  3.25792512996573918826e+02, /* 0x40745CAE, 0x221B9F0A */
	sb3  =  1.53672958608443695994e+03, /* 0x409802EB, 0x189D5118 */
	sb4  =  3.19985821950859553908e+03, /* 0x40A8FFB7, 0x688C246A */
	sb5  =  2.55305040643316442583e+03, /* 0x40A3F219, 0xCEDF3BE6 */
	sb6  =  4.74528541206955367215e+02, /* 0x407DA874, 0xE79FE763 */
	sb7  = -2.24409524465858183362e+01; /* 0xC03670E2, 0x42712D62 */

	double_t s,R,S;
	double z;

	if (ix < 0x3ff40000)  /* |x| < 1.25 */
		return presk_erfc1(x);

	x = fabs(x);
	s = 1/(x*x);
	if (ix < 0x4006db6d) {  /* |x| < 1/.35 ~ 2.85714 */
		R = ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(
		     ra5+s*(ra6+s*ra7))))));
		S = 1.0+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(
		     sa5+s*(sa6+s*(sa7+s*sa8)))))));
	} else {                /* |x| > 1/.35 */
		R = rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(
		     rb5+s*rb6)))));
		S = 1.0+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(
		     sb5+s*(sb6+s*sb7))))));
	}
	z = x;
	SET_LOW_WORD(z,0);
	return presk_exp(-z*z-0.5625)*presk_exp((z-x)*(z+x)+R/S)/x;
}

double presk_erf(double x)
{
	static const double
	/*
	 * Coefficients for approximation to  erf on [0,0.84375]
	 */
	efx8 =  1.02703333676410069053e+00, /* 0x3FF06EBA, 0x8214DB69 */
	pp0  =  1.28379167095512558561e-01, /* 0x3FC06EBA, 0x8214DB68 */
	pp1  = -3.25042107247001499370e-01, /* 0xBFD4CD7D, 0x691CB913 */
	pp2  = -2.84817495755985104766e-02, /* 0xBF9D2A51, 0xDBD7194F */
	pp3  = -5.77027029648944159157e-03, /* 0xBF77A291, 0x236668E4 */
	pp4  = -2.37630166566501626084e-05, /* 0xBEF8EAD6, 0x120016AC */
	qq1  =  3.97917223959155352819e-01, /* 0x3FD97779, 0xCDDADC09 */
	qq2  =  6.50222499887672944485e-02, /* 0x3FB0A54C, 0x5536CEBA */
	qq3  =  5.08130628187576562776e-03, /* 0x3F74D022, 0xC4D36B0F */
	qq4  =  1.32494738004321644526e-04, /* 0x3F215DC9, 0x221C1A10 */
	qq5  = -3.96022827877536812320e-06; /* 0xBED09C43, 0x42A26120 */

	double r,s,z,y;
	uint32_t ix;
	int sign;

	GET_HIGH_WORD(ix, x);
	sign = ix>>31;
	ix &= 0x7fffffff;
	if (ix >= 0x7ff00000) {
		/* erf(nan)=nan, erf(+-inf)=+-1 */
		return 1-2*sign + 1/x;
	}
	if (ix < 0x3feb0000) {  /* |x| < 0.84375 */
		if (ix < 0x3e300000) {  /* |x| < 2**-28 */
			/* avoid underflow */
			return 0.125*(8*x + efx8*x);
		}
		z = x*x;
		r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)));
		s = 1.0+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))));
		y = r/s;
		return x + x*y;
	}
	if (ix < 0x40180000)  /* 0.84375 <= |x| < 6 */
		y = 1 - presk_erfc2(ix,x);
	else
		y = 1 - 0x1p-1022;
	return sign ? -y : y;
}

/* -------------------------------------------------------------------------
 * musl 1.1.22, src/math/erff.c
 * ------------------------------------------------------------------------ */

/* origin: FreeBSD /usr/src/lib/msun/src/s_erff.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

static float presk_erfc1f(float x)
{
	static const float
	erx  =  8.4506291151e-01, /* 0x3f58560b */
	/*
	 * Coefficients for approximation to  erf  in [0.84375,1.25]
	 */
	pa0  = -2.3621185683e-03, /* 0xbb1acdc6 */
	pa1  =  4.1485610604e-01, /* 0x3ed46805 */
	pa2  = -3.7220788002e-01, /* 0xbebe9208 */
	pa3  =  3.1834661961e-01, /* 0x3ea2fe54 */
	pa4  = -1.1089469492e-01, /* 0xbde31cc2 */
	pa5  =  3.5478305072e-02, /* 0x3d1151b3 */
	pa6  = -2.1663755178e-03, /* 0xbb0df9c0 */
	qa1  =  1.0642088205e-01, /* 0x3dd9f331 */
	qa2  =  5.4039794207e-01, /* 0x3f0a5785 */
	qa3  =  7.1828655899e-02, /* 0x3d931ae7 */
	qa4  =  1.2617121637e-01, /* 0x3e013307 */
	qa5  =  1.3637083583e-02, /* 0x3c5f6e13 */
	qa6  =  1.1984500103e-02; /* 0x3c445aa3 */

	float_t s,P,Q;

	s = fabsf(x) - 1;
	P = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))));
	Q = 1+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
	return 1 - erx - P/Q;
}

static float presk_erfc2f(uint32_t ix, float x)
{
	static const float
	/*
	 * Coefficients for approximation to  erfc in [1.25,1/0.35]
	 */
	ra0  = -9.8649440333e-03, /* 0xbc21a093 */
	ra1  = -6.9385856390e-01, /* 0xbf31a0b7 */
	ra2  = -1.0558626175e+01, /* 0xc128f022 */
	ra3  = -6.2375331879e+01, /* 0xc2798057 */
	ra4  = -1.6239666748e+02, /* 0xc322658c */
	ra5  = -1.8460508728e+02, /* 0xc3389ae7 */
	ra6  = -8.1287437439e+01, /* 0xc2a2932b */
	ra7  = -9.8143291473e+00, /* 0xc11d077e */
	sa1  =  1.9651271820e+01, /* 0x419d35ce */
	sa2  =  1.3765776062e+02, /* 0x4309a863 */
	sa3  =  4.3456588745e+02, /* 0x43d9486f */
	sa4  =  6.4538726807e+02, /* 0x442158c9 */
	sa5  =  4.2900814819e+02, /* 0x43d6810b */
	sa6  =  1.0863500214e+02, /* 0x42d9451f */
	sa7  =  6.5702495575e+00, /* 0x40d23f7c */
	sa8  = -6.0424413532e-02, /* 0xbd777f97 */
	/*
	 * Coefficients for approximation to  erfc in [1/.35,28]
	 */
	rb0  = -9.8649431020e-03, /* 0xbc21a092 */
	rb1  = -7.9928326607e-01, /* 0xbf4c9dd4 */
	rb2  = -1.7757955551e+01, /* 0xc18e104b */
	rb3  = -1.6063638306e+02, /* 0xc320a2ea */
	rb4  = -6.3756646729e+02, /* 0xc41f6441 */
	rb5  = -1.0250950928e+03, /* 0xc480230b */
	rb6  = -4.8351919556e+02, /* 0xc3f1c275 */
	sb1  =  3.0338060379e+01, /* 0x41f2b459 */
	sb2  =  3.2579251099e+02, /* 0x43a2e571 */
	sb3  =  1.5367296143e+03, /* 0x44c01759 */
	sb4  =  3.1998581543e+03, /* 0x4547fdbb */
	sb5  =  2.5530502930e+03, /* 0x451f90ce */
	sb6  =  4.7452853394e+02, /* 0x43ed43a7 */
	sb7  = -2.2440952301e+01; /* 0xc1b38712 */

	float_t s,R,S;
	float z;

	if (ix < 0x3fa00000)  /* |x| < 1.25 */
		return presk_erfc1f(x);

	x = fabsf(x);
	s = 1/(x*x);
	if (ix < 0x4036db6d) {   /* |x| < 1/0.35 */
		R = ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(
		     ra5+s*(ra6+s*ra7))))));
		S = 1.0f+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(
		     sa5+s*(sa6+s*(sa7+s*sa8)))))));
	} else {                 /* |x| >= 1/0.35 */
		R = rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(
		     rb5+s*rb6)))));
		S = 1.0f+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(
		     sb5+s*(sb6+s*sb7))))));
	}
	GET_FLOAT_WORD(ix, x);
	SET_FLOAT_WORD(z, ix&0xffffe000);
	return presk_expf(-z*z - 0.5625f) * presk_expf((z-x)*(z+x) + R/S)/x;
}

float presk_erff(float x)
{
	static const float
	/*
	 * Coefficients for approximation to  erf on [0,0.84375]
	 */
	efx8 =  1.0270333290e+00, /* 0x3f8375d4 */
	pp0  =  1.2837916613e-01, /* 0x3e0375d4 */
	pp1  = -3.2504209876e-01, /* 0xbea66beb */
	pp2  = -2.8481749818e-02, /* 0xbce9528f */
	pp3  = -5.7702702470e-03, /* 0xbbbd1489 */
	pp4  = -2.3763017452e-05, /* 0xb7c756b1 */
	qq1  =  3.9791721106e-01, /* 0x3ecbbbce */
	qq2  =  6.5022252500e-02, /* 0x3d852a63 */
	qq3  =  5.0813062117e-03, /* 0x3ba68116 */
	qq4  =  1.3249473704e-04, /* 0x390aee49 */
	qq5  = -3.9602282413e-06; /* 0xb684e21a */

	float r,s,z,y;
	uint32_t ix;
	int sign;

	GET_FLOAT_WORD(ix, x);
	sign = ix>>31;
	ix &= 0x7fffffff;
	if (ix >= 0x7f800000) {
		/* erf(nan)=nan, erf(+-inf)=+-1 */
		return 1-2*sign + 1/x;
	}
	if (ix < 0x3f580000) {  /* |x| < 0.84375 */
		if (ix < 0x31800000) {  /* |x| < 2**-28 */
			/*avoid underflow */
			return 0.125f*(8*x + efx8*x);
		}
		z = x*x;
		r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)));
		s = 1+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))));
		y = r/s;
		return x + x*y;
	}
	if (ix < 0x40c00000)  /* |x| < 6 */
		y = 1 - presk_erfc2f(ix,x);
	else
		y = 1 - 0x1p-120f;
	return sign ? -y : y;
}
