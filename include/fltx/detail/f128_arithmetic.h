/**
 * fltx/detail/f128_arithmetic.h - Low-level double-double arithmetic helpers for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_ARITHMETIC_INCLUDED
#define F128_DETAIL_ARITHMETIC_INCLUDED
#include "fltx/f128_classification.h"
#include "fltx/f128_conversions.h"
#include "fltx/f128_limits.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    // public arithmetic special cases
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s quiet_nan() noexcept
    {
        return { std::numeric_limits<double>::quiet_NaN(), 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s signed_infinity(bool negative) noexcept
    {
        const double inf = std::numeric_limits<double>::infinity();
        return { negative ? -inf : inf, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s signed_zero(bool negative) noexcept
    {
        return { negative ? -0.0 : 0.0, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s add_special(const f128_s& a, const f128_s& b) noexcept
    {
        if (detail::fp::isnan(a.hi) || detail::fp::isnan(b.hi))
            return quiet_nan();

        const bool a_inf = isinf(a.hi);
        const bool b_inf = isinf(b.hi);
        if (a_inf && b_inf && signbit(a.hi) != signbit(b.hi))
            return quiet_nan();
        if (a_inf)
            return signed_infinity(signbit(a.hi));
        return signed_infinity(signbit(b.hi));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_special(const f128_s& a, const f128_s& b) noexcept
    {
        return add_special(a, f128_s{ -b.hi, -b.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_special(const f128_s& a, const f128_s& b) noexcept
    {
        if (detail::fp::isnan(a.hi) || detail::fp::isnan(b.hi))
            return quiet_nan();

        const bool a_inf = isinf(a.hi);
        const bool b_inf = isinf(b.hi);
        if ((a_inf && b.hi == 0.0) || (b_inf && a.hi == 0.0))
            return quiet_nan();

        return signed_infinity(bl::signbit(a) != bl::signbit(b));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_special(const f128_s& a, const f128_s& b) noexcept
    {
        if (detail::fp::isnan(a.hi) || detail::fp::isnan(b.hi))
            return quiet_nan();

        const bool a_zero = a.hi == 0.0;
        const bool b_zero = b.hi == 0.0;
        const bool negative = bl::signbit(a) != bl::signbit(b);
        if (b_zero)
            return a_zero ? quiet_nan() : signed_infinity(negative);

        const bool a_inf = isinf(a.hi);
        const bool b_inf = isinf(b.hi);
        if (a_inf && b_inf)
            return quiet_nan();
        if (a_inf)
            return signed_infinity(negative);
        return signed_zero(negative);
    }

    // residual helpers
    BL_FORCE_INLINE constexpr f128_s sub_mul_scalar_exact(const f128_s& r, const f128_s& b, double q) noexcept
    {
        double p{}, e{};
        two_prod_precise(b.hi, q, p, e);
        e += b.lo * q;

        double s{}, t{};
        detail::fp::two_diff_precise(r.hi, p, s, t);
        t += r.lo - e;

        return renorm(s, t);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double product_split_high(double value) noexcept
    {
        constexpr double split = 134217729.0;
        constexpr int split_shift = 27;

        const double scaled = split * value;
        if (detail::fp::isinf(scaled))
        {
            return detail::fp::ldexp(
                value - (value - detail::fp::ldexp(value, -split_shift)),
                split_shift);
        }

        return scaled - (scaled - value);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_compensated_inline(const f128_s& a, const f128_s& b) noexcept
    {
        const double q0 = a.hi / b.hi;
        if (detail::fp::isinf_or_nan(q0)) [[unlikely]]
            return { q0, 0.0 };
        if (q0 == 0.0 && a.hi == 0.0 && a.lo == 0.0) [[unlikely]]
            return signed_zero(bl::signbit(a) != bl::signbit(b));

        double p0{}, e0{};
        two_prod_precise(q0, b.hi, p0, e0);

        const double q1 = (((a.hi - p0) - e0) + a.lo - (q0 * b.lo)) / b.hi;
        const double hi = q0 + q1;
        return { hi, (q0 - hi) + q1 };
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s div_f128_double_runtime(const f128_s& a, double b) noexcept
    {
        return div_compensated_inline(a, f128_s{ b, 0.0 });
    }

    // core arithmetic
    BL_PUSH_PRECISE;
    BL_FORCE_INLINE constexpr void mul_expansion_inline(const f128_s& a, const f128_s& b, double& p, double& e) noexcept
    {
        two_prod_precise(a.hi, b.hi, p, e);

        e += a.hi * b.lo + a.lo * b.hi;
        e += a.lo * b.lo;
    }
    BL_POP_PRECISE;

    // fused expressions
    BL_PUSH_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s add_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double s1{}, s2{};
        two_sum_precise(a.hi, b.hi, s1, s2);

        double t1{}, t2{};
        two_sum_precise(a.lo, b.lo, t1, t2);

        s2 += t1;
        detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        s2 += t2;
        detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        return { s1, s2 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double s1{}, s2{};
        detail::fp::two_diff_precise(a.hi, b.hi, s1, s2);

        double t1{}, t2{};
        detail::fp::two_diff_precise(a.lo, b.lo, t1, t2);

        s2 += t1;
        detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        s2 += t2;
        detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        return { s1, s2 };
    }
    BL_POP_PRECISE;

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double p{}, e{};
        two_prod_precise(a.hi, b.hi, p, e);
        e += a.hi * b.lo + a.lo * b.hi;
        detail::fp::quick_two_sum_precise(p, e, p, e);
        return { p, e };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_dd_inline(const f128_s& a, const f128_s& b) noexcept
    {
        return mul_inline(a, b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sqr_dd_inline(const f128_s& a) noexcept
    {
        double p{}, e{};
        two_prod_precise(a.hi, a.hi, p, e);
        e += (a.hi + a.hi) * a.lo;
        detail::fp::quick_two_sum_precise(p, e, p, e);
        return { p, e };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_inline(const f128_s& a, const f128_s& b) noexcept
    {
        return div_compensated_inline(a, b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s add_double_inline(const f128_s& a, double b) noexcept
    {
        double s{}, e{};
        two_sum_precise(a.hi, b, s, e);
        e += a.lo;
        return renorm(s, e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s add_double_inline(double a, const f128_s& b) noexcept
    {
        return add_double_inline(b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_double_inline(const f128_s& a, double b) noexcept
    {
        return add_double_inline(a, -b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_double_inline(double a, const f128_s& b) noexcept
    {
        return add_double_inline(-b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_double_inline(const f128_s& a, double b) noexcept
    {
        double p{}, e{};
        two_prod_precise(a.hi, b, p, e);

        e += a.lo * b;
        return renorm(p, e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_double_inline(double a, const f128_s& b) noexcept
    {
        return mul_double_inline(b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_pwr2_inline(const f128_s& a, double b) noexcept
    {
        return { a.hi * b, a.lo * b };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_double_inline(const f128_s& a, double b) noexcept
    {
        if (bl::use_constexpr_math())
        {
            if (detail::fp::isnan(a.hi) || detail::fp::isnan(b)) [[unlikely]]
                return std::numeric_limits<f128_s>::quiet_NaN();

            if (isinf(b))
            {
                if (isinf(a.hi))
                    return std::numeric_limits<f128_s>::quiet_NaN();

                const bool neg = signbit(a.hi) ^ signbit(b);
                return f128_s{ neg ? -0.0 : 0.0, 0.0 };
            }

            if (b == 0.0) [[unlikely]]
            {
                if (a.hi == 0.0 && a.lo == 0.0)
                    return std::numeric_limits<f128_s>::quiet_NaN();

                const bool neg = signbit(a.hi) ^ signbit(b);
                return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
            }

            if (isinf(a.hi)) [[unlikely]]
            {
                const bool neg = signbit(a.hi) ^ signbit(b);
                return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
            }
        }

        return div_compensated_inline(a, f128_s{ b, 0.0 });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_double_inline(double a, const f128_s& b) noexcept
    {
        return div_compensated_inline(f128_s{ a, 0.0 }, b);
    }

    BL_PUSH_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_add_inline(const f128_s& a, const f128_s& b, const f128_s& c) noexcept
    {
        double p{}, e{};
        mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        two_sum_precise(p, c.hi, s, t);
        t += e + c.lo;
        return renorm(s, t);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_sub_inline(const f128_s& a, const f128_s& b, const f128_s& c) noexcept
    {
        double p{}, e{};
        mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        two_sum_precise(p, -c.hi, s, t);
        t += e - c.lo;
        return renorm(s, t);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_mul_inline(const f128_s& c, const f128_s& a, const f128_s& b) noexcept
    {
        double p{}, e{};
        mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        two_sum_precise(c.hi, -p, s, t);
        t += c.lo - e;
        return renorm(s, t);
    }

    BL_FORCE_INLINE constexpr void mul_add_pair_same_rhs_inline(
        const f128_s& a0,
        const f128_s& a1,
        const f128_s& b,
        const f128_s& c0,
        const f128_s& c1,
        f128_s& out0,
        f128_s& out1) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};

        #if BL_F128_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f128_runtime_product_pair_simd_enabled())
        {
            simd::f64x2 p{}, e{};
            const simd::f64x2 ah = simd::f64x2_set(a0.hi, a1.hi);
            const simd::f64x2 bh = simd::f64x2_splat(b.hi);
            const simd::f64x2 al = simd::f64x2_set(a0.lo, a1.lo);
            const simd::f64x2 bl = simd::f64x2_splat(b.lo);
            simd::f64x2_two_prod_precise(ah, bh, p, e);
            e = simd::f64x2_add(e, simd::f64x2_mul(ah, bl));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bh));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bl));
            simd::f64x2_store(p, p0, p1);
            simd::f64x2_store(e, e0, e1);
        }
        else
        #endif
        {
            mul_expansion_inline(a0, b, p0, e0);
            mul_expansion_inline(a1, b, p1, e1);
        }

        double s0{}, t0{};
        two_sum_precise(p0, c0.hi, s0, t0);
        t0 += e0 + c0.lo;
        out0 = renorm(s0, t0);

        double s1{}, t1{};
        two_sum_precise(p1, c1.hi, s1, t1);
        t1 += e1 + c1.lo;
        out1 = renorm(s1, t1);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sum_products_inline(const f128_s& a, const f128_s& b, const f128_s& c, const f128_s& d) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};

        #if BL_F128_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f128_runtime_product_pair_simd_enabled())
        {
            simd::f64x2 p{}, e{};
            const simd::f64x2 ah = simd::f64x2_set(a.hi, c.hi);
            const simd::f64x2 bh = simd::f64x2_set(b.hi, d.hi);
            const simd::f64x2 al = simd::f64x2_set(a.lo, c.lo);
            const simd::f64x2 bl = simd::f64x2_set(b.lo, d.lo);
            simd::f64x2_two_prod_precise(ah, bh, p, e);
            e = simd::f64x2_add(e, simd::f64x2_mul(ah, bl));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bh));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bl));
            simd::f64x2_store(p, p0, p1);
            simd::f64x2_store(e, e0, e1);
        }
        else
        #endif
        {
            mul_expansion_inline(a, b, p0, e0);
            mul_expansion_inline(c, d, p1, e1);
        }

        double s{}, t{};
        two_sum_precise(p0, p1, s, t);
        t += e0 + e1;
        return renorm(s, t);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s diff_products_inline(const f128_s& a, const f128_s& b, const f128_s& c, const f128_s& d) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};

        #if BL_F128_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f128_runtime_product_pair_simd_enabled())
        {
            simd::f64x2 p{}, e{};
            const simd::f64x2 ah = simd::f64x2_set(a.hi, c.hi);
            const simd::f64x2 bh = simd::f64x2_set(b.hi, d.hi);
            const simd::f64x2 al = simd::f64x2_set(a.lo, c.lo);
            const simd::f64x2 bl = simd::f64x2_set(b.lo, d.lo);
            simd::f64x2_two_prod_precise(ah, bh, p, e);
            e = simd::f64x2_add(e, simd::f64x2_mul(ah, bl));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bh));
            e = simd::f64x2_add(e, simd::f64x2_mul(al, bl));
            simd::f64x2_store(p, p0, p1);
            simd::f64x2_store(e, e0, e1);
        }
        else
        #endif
        {
            mul_expansion_inline(a, b, p0, e0);
            mul_expansion_inline(c, d, p1, e1);
        }

        double s{}, t{};
        two_sum_precise(p0, -p1, s, t);
        t += e0 - e1;
        return renorm(s, t);
    }
    BL_POP_PRECISE;

} // namespace detail::_f128

// reciprocal helpers
[[nodiscard]] BL_FORCE_INLINE constexpr f128 recip(f128_s b) noexcept
{
    constexpr f128_s one = f128_s{ 1.0 };
    f128_s y = f128_s{ 1.0 / b.hi };
    f128_s e = one - b * y;

    y += y * e;
    e = one - b * y;
    y += y * e;

    return y;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 inv(const f128_s& a) { return recip(a); }

} // namespace bl

#endif
