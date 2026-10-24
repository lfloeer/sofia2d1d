/*
 * Inner loops of the 2D-1D wavelet transform.
 *
 * They live in C to be able to promise the compiler that the arrays do not
 * overlap, which is what lets it vectorise them. All convolutions accumulate
 * in single precision in order of increasing kernel index, exactly like the
 * generic code paths in pyudwt.pyx, so results do not depend on the code path.
 */
#ifndef SOFIA2D1D_KERNELS_H
#define SOFIA2D1D_KERNELS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* OpenMP is optional: without it, everything runs on a single thread. */
#ifdef _OPENMP
#include <omp.h>
#define S2_HAVE_OPENMP 1
#else
#define S2_HAVE_OPENMP 0
static inline int omp_get_max_threads(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

#if defined(_MSC_VER)
#define S2_RESTRICT __restrict
#else
#define S2_RESTRICT __restrict__
#endif

/*
 * Portable builds get an AVX2 version of each kernel as well, selected at
 * load time on CPUs that support it (needs ifunc support, i.e. glibc; not musl).
 * Floating point contraction is disabled at compile time, so both versions
 * give identical results.
 */
#if defined(__x86_64__) && defined(__ELF__) && defined(__GLIBC__) && !defined(__AVX2__) \
    && !defined(S2_NO_DISPATCH) && defined(__has_attribute)
#if __has_attribute(target_clones)
#define S2_KERNEL __attribute__((target_clones("default", "avx2"))) static
#endif
#endif
#ifndef S2_KERNEL
#define S2_KERNEL static inline
#endif

/* out = sum_j k[j] * r_j */
S2_KERNEL void s2_conv5(const float* S2_RESTRICT r0, const float* S2_RESTRICT r1,
                            const float* S2_RESTRICT r2, const float* S2_RESTRICT r3,
                            const float* S2_RESTRICT r4, float* S2_RESTRICT out,
                            ptrdiff_t n, const float* k)
{
    const float k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4];
    for (ptrdiff_t i = 0; i < n; i++) {
        float s = 0.0f + r0[i] * k0;
        s += r1[i] * k1;
        s += r2[i] * k2;
        s += r3[i] * k3;
        s += r4[i] * k4;
        out[i] = s;
    }
}

/* smooth = sum_j k[j] * r_j, detail = centre - smooth */
S2_KERNEL void s2_conv5_detail(const float* S2_RESTRICT r0, const float* S2_RESTRICT r1,
                                   const float* S2_RESTRICT r2, const float* S2_RESTRICT r3,
                                   const float* S2_RESTRICT r4, const float* S2_RESTRICT centre,
                                   float* S2_RESTRICT smooth, float* S2_RESTRICT detail,
                                   ptrdiff_t n, const float* k)
{
    const float k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4];
    for (ptrdiff_t i = 0; i < n; i++) {
        float s = 0.0f + r0[i] * k0;
        s += r1[i] * k1;
        s += r2[i] * k2;
        s += r3[i] * k3;
        s += r4[i] * k4;
        smooth[i] = s;
        detail[i] = centre[i] - s;
    }
}

/* smooth = sum_j k[j] * r_j, a = a - smooth */
S2_KERNEL void s2_conv5_subtract(const float* S2_RESTRICT r0, const float* S2_RESTRICT r1,
                                     const float* S2_RESTRICT r2, const float* S2_RESTRICT r3,
                                     const float* S2_RESTRICT r4, float* S2_RESTRICT smooth,
                                     float* S2_RESTRICT a, ptrdiff_t n, const float* k)
{
    const float k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4];
    for (ptrdiff_t i = 0; i < n; i++) {
        float s = 0.0f + r0[i] * k0;
        s += r1[i] * k1;
        s += r2[i] * k2;
        s += r3[i] * k3;
        s += r4[i] * k4;
        smooth[i] = s;
        a[i] = a[i] - s;
    }
}

/* detail = a - b */
S2_KERNEL void s2_difference(const float* S2_RESTRICT a, const float* S2_RESTRICT b,
                                 float* S2_RESTRICT detail, ptrdiff_t n)
{
    for (ptrdiff_t i = 0; i < n; i++)
        detail[i] = a[i] - b[i];
}

/* r += w */
S2_KERNEL void s2_add(float* S2_RESTRICT r, const float* S2_RESTRICT w, ptrdiff_t n)
{
    for (ptrdiff_t i = 0; i < n; i++)
        r[i] = r[i] + w[i];
}

/* r += w where |w| > t */
S2_KERNEL void s2_add_hard(float* S2_RESTRICT r, const float* S2_RESTRICT w, ptrdiff_t n, float t)
{
    for (ptrdiff_t i = 0; i < n; i++)
        r[i] = r[i] + (fabsf(w[i]) > t ? w[i] : 0.0f);
}

/* r += sign(w) (|w| - t) where |w| > t */
S2_KERNEL void s2_add_soft(float* S2_RESTRICT r, const float* S2_RESTRICT w, ptrdiff_t n, float t)
{
    for (ptrdiff_t i = 0; i < n; i++) {
        const float v = w[i];
        r[i] = r[i] + (v > t ? v - t : (v < -t ? v + t : 0.0f));
    }
}

/*
 * Hard thresholding with a multi-resolution support stored as bits (little
 * bit order): r += w where the support is set or |w| > t, and the support is
 * extended by the coefficients above the threshold unless it is fixed.
 */
S2_KERNEL void s2_add_mrs(float* S2_RESTRICT r, const float* S2_RESTRICT w,
                              unsigned char* S2_RESTRICT m, ptrdiff_t n, float t, int fixed)
{
    for (ptrdiff_t g = 0; 8 * g < n; g++) {
        const ptrdiff_t base = 8 * g;
        const ptrdiff_t len = n - base < 8 ? n - base : 8;
        const unsigned int old = m[g];
        unsigned int bits = old;
        if (!fixed) {
            for (ptrdiff_t j = 0; j < len; j++)
                bits |= (unsigned int)(fabsf(w[base + j]) > t) << j;
            m[g] = (unsigned char)bits;
        }
        if (bits == 0)
            continue;
        for (ptrdiff_t j = 0; j < len; j++)
            r[base + j] = r[base + j] + (((bits >> j) & 1u) ? w[base + j] : 0.0f);
    }
}

/*
 * Count the |w| < lo and append the |w| in [lo, hi] to out, which must have
 * room for n values. Returns the number of appended values.
 */
S2_KERNEL ptrdiff_t s2_collect(const float* S2_RESTRICT w, ptrdiff_t n, float lo, float hi,
                                   float* S2_RESTRICT out, ptrdiff_t* below)
{
    ptrdiff_t nbelow = 0, ninside = 0, c = 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        const float a = fabsf(w[i]);
        nbelow += a < lo;
        ninside += (a >= lo) & (a <= hi);
    }
    *below += nbelow;
    if (ninside == 0)
        return 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        const float a = fabsf(w[i]);
        out[c] = a;
        c += (a >= lo) & (a <= hi);
    }
    return c;
}

/*
 * Histogram of |w| over the top bits of its IEEE 754 representation (whose
 * order is the order of the values for non-negative floats).
 */
S2_KERNEL void s2_hist(const float* S2_RESTRICT w, ptrdiff_t n, unsigned long long* S2_RESTRICT hist, int shift)
{
    for (ptrdiff_t i = 0; i < n; i++) {
        const float a = fabsf(w[i]);
        uint32_t bits;
        memcpy(&bits, &a, sizeof(bits));
        hist[bits >> shift]++;
    }
}

#endif
