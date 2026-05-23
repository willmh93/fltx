/**
 * fltx/detail/f256_expansion.h - Quad-double normalization and expansion helpers for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_DETAIL_EXPANSION_INCLUDED
#define F256_DETAIL_EXPANSION_INCLUDED
#include "fltx/f256_type.h"

namespace bl {

namespace detail::_f256 // primitives and kernels
{
    // error-free transforms
    BL_FORCE_INLINE constexpr void three_sum(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        two_sum_precise(a, b, t1, t2);
        two_sum_precise(c, t1, a, t3);
        two_sum_precise(t2, t3, b, c);
    }

    BL_FORCE_INLINE constexpr void three_sum2(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        two_sum_precise(a, b, t1, t2);
        two_sum_precise(c, t1, a, t3);
        b = t2 + t3;
    }

    // renormalization
    BL_FORCE_INLINE constexpr f256_s renorm(double c0, double c1, double c2, double c3) noexcept
    {
        double s0{}, s1{}, s2 = 0.0, s3 = 0.0;

        quick_two_sum_precise(c2, c3, s0, c3);
        quick_two_sum_precise(c1, s0, s0, c2);
        quick_two_sum_precise(c0, s0, c0, c1);

        s0 = c0;
        s1 = c1;

        if (s1 != 0.0)
        {
            quick_two_sum_precise(s1, c2, s1, s2);
            if (s2 != 0.0)
                quick_two_sum_precise(s2, c3, s2, s3);
            else
                quick_two_sum_precise(s1, c3, s1, s2);
        }
        else
        {
            quick_two_sum_precise(s0, c2, s0, s1);
            if (s1 != 0.0)
                quick_two_sum_precise(s1, c3, s1, s2);
            else
                quick_two_sum_precise(s0, c3, s0, s1);
        }

        return { s0, s1, s2, s3 };
    }

    // Error-free residuals here are association-sensitive, fast-math can fold away low limbs
    BL_PUSH_PRECISE
    BL_FORCE_INLINE constexpr f256_s renorm4(double c0, double c1, double c2, double c3) noexcept
    {
        double s, e;
        s = c2 + c3,  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        double s0 = c0, s1 = c1, s2 = 0.0, s3 = 0.0;

        if (c2 != 0.0)
        {
            if (s1 != 0.0) {
                s = s1 + c2; e = c2 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c2; e = c2 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0) {
                s = s2 + c3; e = c3 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c3; e = c3 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c3; e = c3 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }

    BL_FORCE_INLINE constexpr f256_s renorm5(double c0, double c1, double c2, double c3, double c4) noexcept
    {
        double s, e;

        s = c3 + c4;  e = c4 - (s - c3);  c3 = s;  c4 = e;
        s = c2 + c3;  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        double s0 = c0, s1 = c1, s2 = 0.0, s3 = 0.0;
        if (c2 != 0.0)
        {
            if (s1 != 0.0) {
                s = s1 + c2;
                e = c2 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c2;
                e = c2 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0) {
                s = s2 + c3; e = c3 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c3; e = c3 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c3; e = c3 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c4 != 0.0)
        {
            if (s3 != 0.0) {
                s3 += c4;
            }
            else if (s2 != 0.0) {
                s = s2 + c4; e = c4 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c4; e = c4 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c4; e = c4 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }
    BL_POP_PRECISE

    BL_FORCE_INLINE constexpr f256_s canonicalize_math_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<8>(value.x3);
        return value;
    }

    #if defined(FLTX_CONSTEXPR_PARITY)
        #define F256_CANONICALIZE_MATH_RESULT(value) ::bl::detail::_f256::canonicalize_math_result(value)
    #else
        #define F256_CANONICALIZE_MATH_RESULT(value) (value)
    #endif

    // Shewchuk-style expansion sum, expansions sorted by increasing magnitude (small -> large)
    // expansion arithmetic
    BL_FORCE_INLINE constexpr int fast_expansion_sum_zeroelim(int elen, const double* e, int flen, const double* f, double* h) noexcept
    {
        int eindex = 0;
        int findex = 0;
        int hindex = 0;

        if (elen == 0) {
            for (int i = 0; i < flen; ++i) if (f[i] != 0.0) h[hindex++] = f[i];
            return hindex;
        }
        if (flen == 0) {
            for (int i = 0; i < elen; ++i) if (e[i] != 0.0) h[hindex++] = e[i];
            return hindex;
        }

        double Q{};
        double Qnew{};
        double hh{};

        double enow = e[eindex];
        double fnow = f[findex];

        if (detail::fp::absd(enow) < detail::fp::absd(fnow)) { Q = enow; ++eindex; enow = (eindex < elen) ? e[eindex] : 0.0; }
        else { Q = fnow; ++findex; fnow = (findex < flen) ? f[findex] : 0.0; }

        while (eindex < elen && findex < flen) {
            if (detail::fp::absd(enow) < detail::fp::absd(fnow)) {
                detail::fp::two_sum_precise(Q, enow, Qnew, hh);
                ++eindex;
                enow = (eindex < elen) ? e[eindex] : 0.0;
            }
            else {
                detail::fp::two_sum_precise(Q, fnow, Qnew, hh);
                ++findex;
                fnow = (findex < flen) ? f[findex] : 0.0;
            }

            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (eindex < elen) {
            detail::fp::two_sum_precise(Q, e[eindex], Qnew, hh);
            ++eindex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (findex < flen) {
            detail::fp::two_sum_precise(Q, f[findex], Qnew, hh);
            ++findex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }

    BL_FORCE_INLINE constexpr int scale_expansion_zeroelim(int elen, const double* e, double b, double* h) noexcept
    {
        int hindex = 0;
        if (elen == 0 || b == 0.0) return 0;

        double Q{}, sum{}, hh{};
        double product1{}, product0{};

        detail::fp::two_prod_precise(e[0], b, product1, product0);
        Q = product1;
        if (product0 != 0.0) h[hindex++] = product0;

        for (int i = 1; i < elen; ++i) {
            detail::fp::two_prod_precise(e[i], b, product1, product0);

            detail::fp::two_sum_precise(Q, product0, sum, hh);
            if (hh != 0.0) h[hindex++] = hh;

            detail::fp::quick_two_sum_precise(product1, sum, Q, hh);
            if (hh != 0.0) h[hindex++] = hh;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }

    BL_FORCE_INLINE constexpr int compress_expansion_zeroelim(int elen, const double* e, double* h) noexcept
    {
        double g[40]{};

        if (elen <= 0) return 0;

        double Q = e[elen - 1];
        for (int i = elen - 2; i >= 0; --i) {
            double Qnew{}, q{};
            detail::fp::two_sum_precise(Q, e[i], Qnew, q);
            Q = Qnew;
            g[i + 1] = q;
        }
        g[0] = Q;

        int hindex = 0;
        Q = g[0];
        for (int i = 1; i < elen; ++i) {
            double Qnew{}, q{};
            detail::fp::two_sum_precise(Q, g[i], Qnew, q);
            if (q != 0.0) h[hindex++] = q;
            Q = Qnew;
        }
        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }

} // namespace detail::_f256

} // namespace bl

#endif
