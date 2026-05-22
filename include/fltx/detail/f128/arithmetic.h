/**
 * fltx/detail/f128/arithmetic.h - Low-level double-double arithmetic helpers for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_ARITHMETIC_INCLUDED
#define FLTX_F128_DETAIL_ARITHMETIC_INCLUDED
#include "fltx/f128/conversions.h"
#include "fltx/f128/stl.h"

namespace bl {

namespace detail::_f128
{
    // residual helpers
    BL_FORCE_INLINE constexpr f128_s sub_mul_scalar_exact(const f128_s& r, const f128_s& b, double q) noexcept
    {
        double p{}, e{};
        two_prod_precise(b.hi, q, p, e);
        e += b.lo * q;

        double s{}, t{};
        two_sum_precise(r.hi, -p, s, t);
        t += r.lo - e;

        return renorm(s, t);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s div_f128_double_runtime(const f128_s& a, double b) noexcept
    {
        const double q0 = a.hi / b;
        const f128_s r = sub_mul_scalar_exact(a, f128_s{ b, 0.0 }, q0);
        const double q1 = r.hi / b;
        return renorm(q0, q1);
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
        // accurate sum of the high parts
        double s{}, e{};
        two_sum_precise(a.hi, b.hi, s, e);

        // fold low parts into the error
        double t = a.lo + b.lo;
        e += t;

        // renormalize
        return renorm(s, e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double s{}, e{};
        two_sum_precise(a.hi, -b.hi, s, e);

        double t = a.lo - b.lo;
        e += t;

        return renorm(s, e);
    }
    BL_POP_PRECISE;

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double p, e;
        two_prod_precise(a.hi, b.hi, p, e);

        e += a.hi * b.lo + a.lo * b.hi;
        e += a.lo * b.lo;
        return renorm(p, e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_inline(const f128_s& a, const f128_s& b) noexcept
    {
        if (b.lo == 0.0)
            return a / b.hi;

        const double inv_b0 = 1.0 / b.hi;

        const double q0 = a.hi * inv_b0;
        f128_s r = sub_mul_scalar_exact(a, b, q0);

        const double q1 = r.hi * inv_b0;

        return renorm(q0, q1);
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

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_double_inline(const f128_s& a, double b) noexcept
    {
        if (bl::use_constexpr_math())
        {
            if (isnan(a.hi) || isnan(b)) [[unlikely]]
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

        if (!bl::use_constexpr_math())
            return div_f128_double_runtime(a, b);

        const double q0 = a.hi / b;
        const f128_s r = sub_mul_scalar_exact(a, f128_s{ b, 0.0 }, q0);
        const double q1 = r.hi / b;

        return renorm(q0, q1);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_double_inline(double a, const f128_s& b) noexcept
    {
        const double q0 = a / b.hi;
        const f128_s r = sub_mul_scalar_exact(f128_s{ a, 0.0 }, b, q0);
        const double q1 = r.hi / b.hi;
        return renorm(q0, q1);
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
