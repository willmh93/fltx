/**
 * f256_expressions.h - f256 fused operations and expression templates.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_EXPRESSIONS_INCLUDED
#define F256_EXPRESSIONS_INCLUDED

#ifndef F256_INCLUDED
#include "f256.h"
#endif

#include <array>

namespace bl {

namespace detail::_f256
{
    struct f256_raw5 { double x0, x1, x2, x3, x4; };
    struct pow2_scale_info { bool valid; bool negative; int exponent; };

    [[nodiscard]] BL_FORCE_INLINE constexpr pow2_scale_info exact_pow2_scale_info(double value) noexcept
    {
        constexpr std::uint64_t sign_mask = 0x8000000000000000ull;
        constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
        constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;

        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        const std::uint64_t abs_bits = bits & ~sign_mask;
        const bool negative = (bits & sign_mask) != 0;

        if (abs_bits == 0 || abs_bits >= exponent_mask)
            return { false, negative, 0 };

        const std::uint32_t exponent_bits = static_cast<std::uint32_t>((abs_bits & exponent_mask) >> 52);
        const std::uint64_t fraction = abs_bits & fraction_mask;

        if (exponent_bits != 0)
            return { fraction == 0, negative, static_cast<int>(exponent_bits) - 1023 };

        if ((fraction & (fraction - 1)) != 0)
            return { false, negative, 0 };

        return { true, negative, bl::detail::fp::highest_bit_index_constexpr(fraction) - 1074 };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr bool finite_scaled_limb_is_safe(double value, double scaled) noexcept
    {
        if (value == 0.0 || !bl::detail::fp::isfinite(value))
            return true;

        if (!bl::detail::fp::isfinite(scaled) || scaled == 0.0)
            return false;

        constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
        const std::uint64_t scaled_abs = std::bit_cast<std::uint64_t>(scaled) & 0x7fffffffffffffffull;
        return (scaled_abs & exponent_mask) != 0;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_raw5 sqr_raw5_inline(const f256_s& a) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p34{}, q34{};
            f256_simd2d p67{}, q67{};

            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x0), f256_simd_set(a.x0, a.x1), p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(a.x2, a.x1), p34, q34);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(a.x3, a.x2), p67, q67);

            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p34, p3, p4);
            f256_simd_store(q34, q3, q4);
            f256_simd_store(p67, p6, p7);
            f256_simd_store(q67, q6, q7);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, a.x0, p0, q0);
            two_prod_precise(a.x0, a.x1, p1, q1);
            two_prod_precise(a.x0, a.x2, p3, q3);
            two_prod_precise(a.x1, a.x1, p4, q4);
            two_prod_precise(a.x0, a.x3, p6, q6);
            two_prod_precise(a.x1, a.x2, p7, q7);
        }
        p2 = p1;
        q2 = q1;
        p5 = p3;
        q5 = q3;

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        p8 = p7;
        q8 = q7;
        p9 = p6;
        q9 = q6;

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * a.x3 + a.x2 * a.x2 + a.x3 * a.x1 + q6 + q7 + q8 + q9 + s2;

        return { p0, p1, s0, t0, t1 };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_raw5 mul_raw5_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (&a == &b)
            return sqr_raw5_inline(a);

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            f256_simd2d p45{}, q45{};
            f256_simd2d p67{}, q67{};
            f256_simd2d p89{}, q89{};

            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x0), f256_simd_set(b.x0, b.x1), p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x0), f256_simd_set(b.x0, b.x2), p23, q23);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x2), f256_simd_set(b.x1, b.x0), p45, q45);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(b.x3, b.x2), p67, q67);
            f256_simd_two_prod_precise(f256_simd_set(a.x2, a.x3), f256_simd_set(b.x1, b.x0), p89, q89);

            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, q3);
            f256_simd_store(p45, p4, p5);
            f256_simd_store(q45, q4, q5);
            f256_simd_store(p67, p6, p7);
            f256_simd_store(q67, q6, q7);
            f256_simd_store(p89, p8, p9);
            f256_simd_store(q89, q8, q9);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, b.x0, p0, q0);
            two_prod_precise(a.x0, b.x1, p1, q1);
            two_prod_precise(a.x1, b.x0, p2, q2);
            two_prod_precise(a.x0, b.x2, p3, q3);
            two_prod_precise(a.x1, b.x1, p4, q4);
            two_prod_precise(a.x2, b.x0, p5, q5);
            two_prod_precise(a.x0, b.x3, p6, q6);
            two_prod_precise(a.x1, b.x2, p7, q7);
            two_prod_precise(a.x2, b.x1, p8, q8);
            two_prod_precise(a.x3, b.x0, p9, q9);
        }

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + s2;

        return { p0, p1, s0, t0, t1 };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_raw5 neg_raw5(f256_raw5 v) noexcept
    {
        return { -v.x0, -v.x1, -v.x2, -v.x3, -v.x4 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_inline(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s scale_unchecked_inline(const f256_s& a, double scalar) noexcept
    {
        return { a.x0 * scalar, a.x1 * scalar, a.x2 * scalar, a.x3 * scalar };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s scale_pow2_inline(const f256_s& a, int exponent, bool negative) noexcept
    {
        #if defined(BL_FAST_MATH)
        if (!bl::use_constexpr_math())
        {
            const double scale = bl::detail::fp::scalbn_constexpr2(negative ? -1.0 : 1.0, exponent);
            return scale_unchecked_inline(a, scale);
        }
        #endif

        const double x0 = bl::detail::fp::scalbn_constexpr2(a.x0, exponent);
        const double x1 = bl::detail::fp::scalbn_constexpr2(a.x1, exponent);
        const double x2 = bl::detail::fp::scalbn_constexpr2(a.x2, exponent);
        const double x3 = bl::detail::fp::scalbn_constexpr2(a.x3, exponent);

        #if !defined(BL_FAST_MATH)
        if (!finite_scaled_limb_is_safe(a.x0, x0) ||
            !finite_scaled_limb_is_safe(a.x1, x1) ||
            !finite_scaled_limb_is_safe(a.x2, x2) ||
            !finite_scaled_limb_is_safe(a.x3, x3))
        {
            const double scalar = negative ? -bl::detail::fp::scalbn_constexpr2(1.0, exponent) : bl::detail::fp::scalbn_constexpr2(1.0, exponent);
            return mul_double_inline(a, scalar);
        }
        #endif

        if (negative)
            return { -x0, -x1, -x2, -x3 };

        return { x0, x1, x2, x3 };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s scale_pow2_or_checked_inline(const f256_s& a,double scalar,pow2_scale_info scale) noexcept
    {
        #if defined(BL_FAST_MATH)
        if (!bl::use_constexpr_math())
            return scale_unchecked_inline(a, scalar);
        #endif

        return scale_pow2_inline(a, scale.exponent, scale.negative);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_pow2_or_double_inline(const f256_s& a, double b) noexcept
    {
        const pow2_scale_info scale = exact_pow2_scale_info(b);
        if (scale.valid)
            return scale_pow2_or_checked_inline(a, b, scale);

        return mul_double_inline(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_raw5_value_inline(f256_raw5 p, const f256_s& v) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        two_sum_precise(p.x0, v.x0, s0, e0);
        two_sum_precise(p.x1, v.x1, s1, e1);
        two_sum_precise(p.x2, v.x2, s2, e2);
        two_sum_precise(p.x3, v.x3, s3, e3);

        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3 + p.x4;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_raw5_raw5_inline(f256_raw5 a, f256_raw5 b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        two_sum_precise(a.x0, b.x0, s0, e0);
        two_sum_precise(a.x1, b.x1, s1, e1);
        two_sum_precise(a.x2, b.x2, s2, e2);
        two_sum_precise(a.x3, b.x3, s3, e3);

        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3 + a.x4 + b.x4;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_raw5_raw5_value_inline(f256_raw5 a, f256_raw5 b, const f256_s& v) noexcept
    {
        return add_inline(add_raw5_raw5_inline(a, b), v);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return add_raw5_value_inline(mul_raw5_inline(a, b), c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return add_raw5_value_inline(mul_raw5_inline(a, b), -c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_mul_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return add_raw5_value_inline(neg_raw5(mul_raw5_inline(b, c)), a);
    }
    
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return add_raw5_raw5_inline(mul_raw5_inline(a, b), mul_raw5_inline(c, d));
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return add_raw5_raw5_inline(mul_raw5_inline(a, b), neg_raw5(mul_raw5_inline(c, d)));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_add_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return add_raw5_raw5_value_inline(mul_raw5_inline(a, b), mul_raw5_inline(c, d), e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_sub_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return add_raw5_raw5_value_inline(mul_raw5_inline(a, b), mul_raw5_inline(c, d), -e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_add_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return add_raw5_raw5_value_inline(mul_raw5_inline(a, b), neg_raw5(mul_raw5_inline(c, d)), e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_sub_inline(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return add_raw5_raw5_value_inline(mul_raw5_inline(a, b), neg_raw5(mul_raw5_inline(c, d)), -e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return add_inline(add_inline(a, b), c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_add_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return add_inline(sub_inline(a, b), c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_sub_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return sub_inline(add_inline(a, b), c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_sub_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return sub_inline(sub_inline(a, b), c);
    }

    template<int Scale>
    [[nodiscard]] BL_FORCE_INLINE constexpr double scale_limb_inline(double value) noexcept
    {
        if constexpr (Scale == 1)
            return value;
        else if constexpr (Scale == -1)
            return -value;
        else
            return value * static_cast<double>(Scale);
    }

    template<int AScale, int BScale>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scaled_inline(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        two_sum_precise(scale_limb_inline<AScale>(a.x0), scale_limb_inline<BScale>(b.x0), s0, e0);
        two_sum_precise(scale_limb_inline<AScale>(a.x1), scale_limb_inline<BScale>(b.x1), s1, e1);
        two_sum_precise(scale_limb_inline<AScale>(a.x2), scale_limb_inline<BScale>(b.x2), s2, e2);
        two_sum_precise(scale_limb_inline<AScale>(a.x3), scale_limb_inline<BScale>(b.x3), s3, e3);

        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_raw5 mul_double_raw5_inline(const f256_s& a, double b) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            const f256_simd2d bv = f256_simd_splat(b);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), bv, p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x2, a.x3), bv, p23, q23);
            double ignored{};
            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, ignored);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, b, p0, q0);
            two_prod_precise(a.x1, b, p1, q1);
            two_prod_precise(a.x2, b, p2, q2);
            p3 = a.x3 * b;
        }

        s0 = p0;
        two_sum_precise(q0, p1, s1, s2);
        three_sum(s2, q1, p2);
        three_sum2(q1, q2, p3);
        s3 = q1;
        s4 = q2 + p2;

        return { s0, s1, s2, s3, s4 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_mul_double_inline(const f256_s& addend, const f256_s& value, double scalar) noexcept
    {
        const pow2_scale_info scale = exact_pow2_scale_info(scalar);
        if (scale.valid)
            return add_inline(addend, scale_pow2_or_checked_inline(value, scalar, scale));

        return add_raw5_value_inline(mul_double_raw5_inline(value, scalar), addend);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_double_inline(const f256_s& minuend, const f256_s& value, double scalar) noexcept
    {
        const pow2_scale_info scale = exact_pow2_scale_info(scalar);
        if (scale.valid)
            return sub_inline(minuend, scale_pow2_or_checked_inline(value, scalar, scale));

        return add_raw5_value_inline(neg_raw5(mul_double_raw5_inline(value, scalar)), minuend);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_sub_inline(const f256_s& value, double scalar, const f256_s& subtrahend) noexcept
    {
        const pow2_scale_info scale = exact_pow2_scale_info(scalar);
        if (scale.valid)
            return sub_inline(scale_pow2_or_checked_inline(value, scalar, scale), subtrahend);

        return add_raw5_value_inline(mul_double_raw5_inline(value, scalar), -subtrahend);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_add_double_inline(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept
    {
        return div_inline(numerator, add_double_inline(base_denominator, scalar));
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_sub_inline(const f256_s& numerator, double scalar, const f256_s& base_denominator) noexcept
    {
        return div_inline(numerator, sub_double_inline(scalar, base_denominator));
    }

}

namespace detail::_f256_expr
{
    template<class T> using clean_t = std::remove_cv_t<std::remove_reference_t<T>>;

    struct leaf_expr
    {
        f256_s value;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ value }; }
    };

    template<class L>
    struct mul_double_expr
    {
        L left;
        double right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L>
    struct add_double_expr
    {
        L left;
        double right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class R>
    struct double_sub_expr
    {
        double left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L, class R>
    struct add_expr
    {
        L left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L, class R>
    struct sub_expr
    {
        L left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L, class R>
    struct mul_expr
    {
        L left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L, class R>
    struct div_expr
    {
        L left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class L>
    struct div_double_expr
    {
        L left;
        double right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };

    template<class R>
    struct double_div_expr
    {
        double left;
        R right;

        [[nodiscard]] BL_FORCE_INLINE constexpr operator f256() const noexcept { return f256{ eval_to_f256_s(*this) }; }
    };
                                
    template<>                  struct is_expr<leaf_expr>          : std::true_type {};
    template<class L>           struct is_expr<mul_double_expr<L>> : std::true_type {};
    template<class L>           struct is_expr<add_double_expr<L>> : std::true_type {};
    template<class R>           struct is_expr<double_sub_expr<R>> : std::true_type {};
    template<class L, class R>  struct is_expr<add_expr<L, R>>     : std::true_type {};
    template<class L, class R>  struct is_expr<sub_expr<L, R>>     : std::true_type {};
    template<class L, class R>  struct is_expr<mul_expr<L, R>>     : std::true_type {};
    template<class L, class R>  struct is_expr<div_expr<L, R>>     : std::true_type {};
    template<class L>           struct is_expr<div_double_expr<L>> : std::true_type {};
    template<class R>           struct is_expr<double_div_expr<R>> : std::true_type {};
                                
    template<class T>           inline constexpr bool is_f256_value_v = std::is_same_v<clean_t<T>, f256>;
    template<class T>           inline constexpr bool is_leaf_v       = std::is_same_v<clean_t<T>, leaf_expr>;
    template<class T>           inline constexpr bool is_expr_v       = is_expr<clean_t<T>>::value;
    template<class T>           inline constexpr bool is_operand_v    = is_expr_v<T> || is_f256_value_v<T>;
                                
    template<class T>           inline constexpr bool is_add_v = false;
    template<class T>           inline constexpr bool is_sub_v = false;
    template<class T>           inline constexpr bool is_mul_v = false;
    template<class T>           inline constexpr bool is_div_v = false;
    template<class T>           inline constexpr bool is_mul_double_v = false;
    template<class T>           inline constexpr bool is_div_double_v = false;
    template<class T>           inline constexpr bool is_add_double_v = false;
    template<class T>           inline constexpr bool is_double_div_v = false;
    template<class T>           inline constexpr bool is_double_sub_v = false;
    template<class T>           inline constexpr bool is_leaf_product_v = false;
    template<class T>           inline constexpr bool is_leaf_add_v = false;
    template<class T>           inline constexpr bool is_leaf_sub_v = false;
    template<class T>           inline constexpr bool is_add_product_product_v = false;
    template<class T>           inline constexpr bool is_sub_product_product_v = false;
    template<class T>           inline constexpr bool is_leaf_add_add_leaf_v = false;
    template<class T>           inline constexpr bool is_leaf_add_sub_leaf_v = false;
    template<class T>           inline constexpr bool is_leaf_sub_add_leaf_v = false;
    template<class T>           inline constexpr bool is_leaf_sub_sub_leaf_v = false;
    template<class T>           inline constexpr bool is_product_add_leaf_v = false;
    template<class T>           inline constexpr bool is_product_sub_leaf_v = false;
    template<class T>           inline constexpr bool is_add_product_product_add_product_v = false;
    template<class T>           inline constexpr bool is_add_product_product_add_add_product_product_v = false;
    template<class T>           inline constexpr bool is_mul_double_add_mul_double_v = false;
    template<class T>           inline constexpr bool is_mul_double_add_mul_double_add_leaf_v = false;
                                
    template<class L>           inline constexpr bool is_mul_double_v<mul_double_expr<L>> = true;
    template<class L>           inline constexpr bool is_add_double_v<add_double_expr<L>> = true;
    template<class R>           inline constexpr bool is_double_sub_v<double_sub_expr<R>> = true;
    template<class L, class R>  inline constexpr bool is_add_v<add_expr<L, R>> = true;
    template<class L, class R>  inline constexpr bool is_sub_v<sub_expr<L, R>> = true;
    template<class L, class R>  inline constexpr bool is_mul_v<mul_expr<L, R>> = true;
    template<class L, class R>  inline constexpr bool is_div_v<div_expr<L, R>> = true;
    template<class L>           inline constexpr bool is_div_double_v<div_double_expr<L>> = true;
    template<class R>           inline constexpr bool is_double_div_v<double_div_expr<R>> = true;
    template<class L, class R>  inline constexpr bool is_leaf_product_v<mul_expr<L, R>> = is_leaf_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_add_v<add_expr<L, R>> = is_leaf_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_sub_v<sub_expr<L, R>> = is_leaf_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_add_product_product_v<add_expr<L, R>> = is_leaf_product_v<L> && is_leaf_product_v<R>;
    template<class L, class R>  inline constexpr bool is_sub_product_product_v<sub_expr<L, R>> = is_leaf_product_v<L> && is_leaf_product_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_add_add_leaf_v<add_expr<L, R>> = is_leaf_add_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_add_sub_leaf_v<sub_expr<L, R>> = is_leaf_add_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_sub_add_leaf_v<add_expr<L, R>> = is_leaf_sub_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_leaf_sub_sub_leaf_v<sub_expr<L, R>> = is_leaf_sub_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_product_add_leaf_v<add_expr<L, R>> = is_leaf_product_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_product_sub_leaf_v<sub_expr<L, R>> = is_leaf_product_v<L> && is_leaf_v<R>;
    template<class L, class R>  inline constexpr bool is_add_product_product_add_product_v<add_expr<L, R>> = is_add_product_product_v<L> && is_leaf_product_v<R>;
    template<class L, class R>  inline constexpr bool is_add_product_product_add_add_product_product_v<add_expr<L, R>> = is_add_product_product_v<L> && is_add_product_product_v<R>;
    template<class L, class R>  inline constexpr bool is_mul_double_add_mul_double_v<add_expr<L, R>> = is_mul_double_v<L> && is_mul_double_v<R>;
    template<class L, class R>  inline constexpr bool is_mul_double_add_mul_double_add_leaf_v<add_expr<L, R>> = is_mul_double_add_mul_double_v<L> && is_leaf_v<R>;

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto as_expr(T&& value) noexcept
    {
        if constexpr (is_expr_v<T>)
            return std::forward<T>(value);
        else
            return leaf_expr{ static_cast<const f256_s&>(value) };
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr const f256_s& leaf_value(const leaf_expr& expr) noexcept
    {
        return expr.value;
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr bool same_f256_value(const f256_s& a, const f256_s& b) noexcept
    {
        return std::bit_cast<std::uint64_t>(a.x0) == std::bit_cast<std::uint64_t>(b.x0) &&
               std::bit_cast<std::uint64_t>(a.x1) == std::bit_cast<std::uint64_t>(b.x1) &&
               std::bit_cast<std::uint64_t>(a.x2) == std::bit_cast<std::uint64_t>(b.x2) &&
               std::bit_cast<std::uint64_t>(a.x3) == std::bit_cast<std::uint64_t>(b.x3);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr bool f256_value_less(const f256_s& a, const f256_s& b) noexcept
    {
        const std::uint64_t a0 = std::bit_cast<std::uint64_t>(a.x0);
        const std::uint64_t b0 = std::bit_cast<std::uint64_t>(b.x0);
        if (a0 != b0)
            return a0 < b0;

        const std::uint64_t a1 = std::bit_cast<std::uint64_t>(a.x1);
        const std::uint64_t b1 = std::bit_cast<std::uint64_t>(b.x1);
        if (a1 != b1)
            return a1 < b1;

        const std::uint64_t a2 = std::bit_cast<std::uint64_t>(a.x2);
        const std::uint64_t b2 = std::bit_cast<std::uint64_t>(b.x2);
        if (a2 != b2)
            return a2 < b2;

        return std::bit_cast<std::uint64_t>(a.x3) < std::bit_cast<std::uint64_t>(b.x3);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr bool same_leaf_value(const leaf_expr& a, const leaf_expr& b) noexcept
    {
        return same_f256_value(a.value, b.value);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(a, b);

        return detail::_f256_runtime::add(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(a, b);

        return detail::_f256_runtime::sub(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_inline(a, b);

        return detail::_f256_runtime::mul(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(a, b);

        return detail::_f256_runtime::div(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_eval(const f256_s& a, double b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_double_inline(a, b);

        return detail::_f256_runtime::add_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_eval(const f256_s& a, double b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_double_inline(a, b);

        return detail::_f256_runtime::sub_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_eval(double a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_double_inline(a, b);

        return detail::_f256_runtime::sub_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_eval(const f256_s& a, double b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_double_inline(a, b);

        return detail::_f256_runtime::div_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_eval(double a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_double_inline(a, b);

        return detail::_f256_runtime::div_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_eval(const f256_s& a) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sqr_inline(a);

        return detail::_f256_runtime::sqr(a);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_pow2_or_double_eval(const f256_s& a, double b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_pow2_or_double_inline(a, b);

        return detail::_f256_runtime::mul_pow2_or_double(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_add_inline(a, b, c);

        return detail::_f256_runtime::mul_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_sub_inline(a, b, c);

        return detail::_f256_runtime::mul_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::value_sub_mul_inline(a, b, c);

        return detail::_f256_runtime::value_sub_mul(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_add_mul_inline(a, b, c, d);

        return detail::_f256_runtime::mul_add_mul(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_sub_mul_inline(a, b, c, d);

        return detail::_f256_runtime::mul_sub_mul(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_add_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_add_mul_add_inline(a, b, c, d, e);

        return detail::_f256_runtime::mul_add_mul_add(a, b, c, d, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_add_mul_sub_inline(a, b, c, d, e);

        return detail::_f256_runtime::mul_add_mul_sub(a, b, c, d, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_add_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_sub_mul_add_inline(a, b, c, d, e);

        return detail::_f256_runtime::mul_sub_mul_add(a, b, c, d, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_sub_mul_sub_inline(a, b, c, d, e);

        return detail::_f256_runtime::mul_sub_mul_sub(a, b, c, d, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_add_add_inline(a, b, c);

        return detail::_f256_runtime::add_add_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_sub_add_inline(a, b, c);

        return detail::_f256_runtime::add_sub_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_add_sub_inline(a, b, c);

        return detail::_f256_runtime::add_add_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_sub_sub_inline(a, b, c);

        return detail::_f256_runtime::add_sub_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scaled_2_1_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_scaled_inline<2, 1>(a, b);

        return detail::_f256_runtime::add_scaled_2_1(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scaled_1_2_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_scaled_inline<1, 2>(a, b);

        return detail::_f256_runtime::add_scaled_1_2(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scaled_2_neg1_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_scaled_inline<2, -1>(a, b);

        return detail::_f256_runtime::add_scaled_2_neg1(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scaled_1_neg2_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_scaled_inline<1, -2>(a, b);

        return detail::_f256_runtime::add_scaled_1_neg2(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_mul_double_eval(const f256_s& addend, const f256_s& value, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_mul_double_inline(addend, value, scalar);

        return detail::_f256_runtime::add_mul_double(addend, value, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_double_eval(const f256_s& minuend, const f256_s& value, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_mul_double_inline(minuend, value, scalar);

        return detail::_f256_runtime::sub_mul_double(minuend, value, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_sub_eval(const f256_s& value, double scalar, const f256_s& subtrahend) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::mul_double_sub_inline(value, scalar, subtrahend);

        return detail::_f256_runtime::mul_double_sub(value, scalar, subtrahend);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_add_double_eval(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_add_double_inline(numerator, base_denominator, scalar);

        return detail::_f256_runtime::div_add_double(numerator, base_denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_sub_eval(const f256_s& numerator, double scalar, const f256_s& base_denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_double_sub_inline(numerator, scalar, base_denominator);

        return detail::_f256_runtime::div_double_sub(numerator, scalar, base_denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b);

        return detail::_f256_runtime::sqr_add(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b);

        return detail::_f256_runtime::sqr_sub(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_sqr_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_value_inline(detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), a);

        return detail::_f256_runtime::value_sub_sqr(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_add_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::mul_add_inline(a, b, c), d);

        return detail::_f256_runtime::mul_add_add(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::mul_add_inline(a, b, c), d);

        return detail::_f256_runtime::mul_add_sub(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_add_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::mul_sub_inline(a, b, c), d);

        return detail::_f256_runtime::mul_sub_add(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::mul_sub_inline(a, b, c), d);

        return detail::_f256_runtime::mul_sub_sub(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b), c);

        return detail::_f256_runtime::sqr_add_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b), c);

        return detail::_f256_runtime::sqr_add_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b), c);

        return detail::_f256_runtime::sqr_sub_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b), c);

        return detail::_f256_runtime::sqr_sub_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_sqr_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b));

        return detail::_f256_runtime::sqr_add_sqr(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_sqr_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)));

        return detail::_f256_runtime::sqr_sub_sqr(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_twice_eval(const f256_s& a) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto square = detail::_f256::sqr_raw5_inline(a);
            return detail::_f256::add_raw5_raw5_inline(square, square);
        }

        return detail::_f256_runtime::sqr_twice(a);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_twice_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto product = detail::_f256::mul_raw5_inline(a, b);
            return detail::_f256::add_raw5_raw5_inline(product, product);
        }

        return detail::_f256_runtime::mul_twice(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_twice_add_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto square = detail::_f256::sqr_raw5_inline(a);
            return detail::_f256::add_raw5_raw5_value_inline(square, square, b);
        }

        return detail::_f256_runtime::sqr_twice_add(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_twice_sub_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto square = detail::_f256::sqr_raw5_inline(a);
            return detail::_f256::add_raw5_raw5_value_inline(square, square, -b);
        }

        return detail::_f256_runtime::sqr_twice_sub(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_sqr_twice_eval(const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto neg_square = detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b));
            return detail::_f256::add_raw5_raw5_value_inline(neg_square, neg_square, a);
        }

        return detail::_f256_runtime::value_sub_sqr_twice(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_twice_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto product = detail::_f256::mul_raw5_inline(a, b);
            return detail::_f256::add_raw5_raw5_value_inline(product, product, c);
        }

        return detail::_f256_runtime::mul_twice_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_twice_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto product = detail::_f256::mul_raw5_inline(a, b);
            return detail::_f256::add_raw5_raw5_value_inline(product, product, -c);
        }

        return detail::_f256_runtime::mul_twice_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_mul_twice_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
        {
            const auto neg_product = detail::_f256::neg_raw5(detail::_f256::mul_raw5_inline(b, c));
            return detail::_f256::add_raw5_raw5_value_inline(neg_product, neg_product, a);
        }

        return detail::_f256_runtime::value_sub_mul_twice(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_sqr_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b), c);

        return detail::_f256_runtime::sqr_add_sqr_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_add_sqr_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b), -c);

        return detail::_f256_runtime::sqr_add_sqr_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_sqr_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), c);

        return detail::_f256_runtime::sqr_sub_sqr_add(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_sub_sqr_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), -c);

        return detail::_f256_runtime::sqr_sub_sqr_sub(a, b, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_add_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::mul_inline(e, f));

        return detail::_f256_runtime::mul_add_mul_add_mul(a, b, c, d, e, f);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_add_mul_add_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f, const f256_s& g, const f256_s& h) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::mul_add_mul_inline(e, f, g, h));

        return detail::_f256_runtime::mul_add_mul_add_mul_add_mul(a, b, c, d, e, f, g, h);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_add_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_inline(detail::_f256::add_add_add_inline(a, b, c), d);

        return detail::_f256_runtime::add_add_add_add(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::add_add_add_inline(a, b, c), d);

        return detail::_f256_runtime::add_add_add_sub(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_sub_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::add_add_sub_inline(a, b, c), d);

        return detail::_f256_runtime::add_add_sub_sub(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_sub_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::sub_inline(detail::_f256::add_sub_sub_inline(a, b, c), d);

        return detail::_f256_runtime::add_sub_sub_sub(a, b, c, d);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_add_mul_double_eval(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_inline(detail::_f256::mul_double_raw5_inline(a, a_scalar), detail::_f256::mul_double_raw5_inline(b, b_scalar));

        return detail::_f256_runtime::mul_double_add_mul_double(a, a_scalar, b, b_scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_add_mul_double_add_eval(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar, const f256_s& c) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::mul_double_raw5_inline(a, a_scalar), detail::_f256::mul_double_raw5_inline(b, b_scalar), c);

        return detail::_f256_runtime::mul_double_add_mul_double_add(a, a_scalar, b, b_scalar, c);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_add_eval(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(numerator, detail::_f256::add_inline(a, b));

        return detail::_f256_runtime::div_add(numerator, a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_sub_eval(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(numerator, detail::_f256::sub_inline(a, b));

        return detail::_f256_runtime::div_sub(numerator, a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_add_inline(a, b, c), denominator);

        return detail::_f256_runtime::mul_add_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_sub_inline(a, b, c), denominator);

        return detail::_f256_runtime::mul_sub_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_mul_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::value_sub_mul_inline(a, b, c), denominator);

        return detail::_f256_runtime::value_sub_mul_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), denominator);

        return detail::_f256_runtime::mul_add_mul_div(a, b, c, d, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_sub_mul_inline(a, b, c, d), denominator);

        return detail::_f256_runtime::mul_sub_mul_div(a, b, c, d, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_add_add_inline(a, b, c), denominator);

        return detail::_f256_runtime::add_add_add_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_add_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_sub_add_inline(a, b, c), denominator);

        return detail::_f256_runtime::add_sub_add_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_sub_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_add_sub_inline(a, b, c), denominator);

        return detail::_f256_runtime::add_add_sub_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_sub_div_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_sub_sub_inline(a, b, c), denominator);

        return detail::_f256_runtime::add_sub_sub_div(a, b, c, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_mul_double_div_eval(const f256_s& addend, const f256_s& value, double scalar, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_mul_double_inline(addend, value, scalar), denominator);

        return detail::_f256_runtime::add_mul_double_div(addend, value, scalar, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_double_div_eval(const f256_s& minuend, const f256_s& value, double scalar, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::sub_mul_double_inline(minuend, value, scalar), denominator);

        return detail::_f256_runtime::sub_mul_double_div(minuend, value, scalar, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_sub_div_eval(const f256_s& value, double scalar, const f256_s& subtrahend, const f256_s& denominator) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_double_sub_inline(value, scalar, subtrahend), denominator);

        return detail::_f256_runtime::mul_double_sub_div(value, scalar, subtrahend, denominator);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::mul_add_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::mul_sub_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s value_sub_mul_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::value_sub_mul_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::value_sub_mul_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_add_mul_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::mul_add_mul_div_add_double(a, b, c, d, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_sub_mul_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_sub_mul_inline(a, b, c, d), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::mul_sub_mul_div_add_double(a, b, c, d, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_add_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_add_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::add_add_add_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_add_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_sub_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::add_sub_add_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_add_sub_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_add_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::add_add_sub_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_sub_sub_div_add_double_eval(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_sub_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));

        return detail::_f256_runtime::add_sub_sub_div_add_double(a, b, c, denominator, scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_mul_double_div_add_double_eval(const f256_s& addend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::add_mul_double_inline(addend, value, value_scalar), detail::_f256::add_double_inline(denominator, denominator_scalar));

        return detail::_f256_runtime::add_mul_double_div_add_double(addend, value, value_scalar, denominator, denominator_scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_double_div_add_double_eval(const f256_s& minuend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::sub_mul_double_inline(minuend, value, value_scalar), detail::_f256::add_double_inline(denominator, denominator_scalar));

        return detail::_f256_runtime::sub_mul_double_div_add_double(minuend, value, value_scalar, denominator, denominator_scalar);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_sub_div_add_double_eval(const f256_s& value, double value_scalar, const f256_s& subtrahend, const f256_s& denominator, double denominator_scalar) noexcept
    {
        if (bl::is_constant_evaluated())
            return detail::_f256::div_inline(detail::_f256::mul_double_sub_inline(value, value_scalar, subtrahend), detail::_f256::add_double_inline(denominator, denominator_scalar));

        return detail::_f256_runtime::mul_double_sub_div_add_double(value, value_scalar, subtrahend, denominator, denominator_scalar);
    }

    enum class normalized_linear_atom_kind { value, product, square };

    struct normalized_linear_atom
    {
        normalized_linear_atom_kind kind{};
        const f256_s* left{};
        const f256_s* right{};
    };

    struct normalized_linear_term
    {
        normalized_linear_atom atom{};
        int coefficient{};
    };

    template<std::size_t MaxTerms>
    struct normalized_linear_form
    {
        std::array<normalized_linear_term, MaxTerms> terms{};
        std::size_t count{};
        bool valid = true;
    };

    struct normalized_linear_eval_result
    {
        bool matched{};
        f256_s value{};
    };

    struct normalized_scalar_coefficient
    {
        bool valid{};
        int value{};
    };

    inline constexpr std::size_t normalized_linear_max_terms = 8;

    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_scalar_coefficient normalized_integer_scalar(double value) noexcept
    {
        if (value == 1.0)
            return { true, 1 };
        if (value == -1.0)
            return { true, -1 };
        if (value == 2.0)
            return { true, 2 };
        if (value == -2.0)
            return { true, -2 };

        return {};
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_atom make_value_atom(const leaf_expr& leaf) noexcept
    {
        return { normalized_linear_atom_kind::value, &leaf_value(leaf), nullptr };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_atom make_product_atom(const leaf_expr& left_leaf, const leaf_expr& right_leaf) noexcept
    {
        const f256_s& left = leaf_value(left_leaf);
        const f256_s& right = leaf_value(right_leaf);

        if (same_f256_value(left, right))
            return { normalized_linear_atom_kind::square, &left, nullptr };
        if (f256_value_less(right, left))
            return { normalized_linear_atom_kind::product, &right, &left };

        return { normalized_linear_atom_kind::product, &left, &right };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool same_normalized_atom(const normalized_linear_atom& a, const normalized_linear_atom& b) noexcept
    {
        if (a.kind != b.kind || a.left == nullptr || b.left == nullptr)
            return false;
        if (!same_f256_value(*a.left, *b.left))
            return false;

        if (a.kind != normalized_linear_atom_kind::product)
            return true;

        return a.right != nullptr && b.right != nullptr && same_f256_value(*a.right, *b.right);
    }

    template<std::size_t MaxTerms>
    BL_FORCE_INLINE constexpr void add_normalized_term(normalized_linear_form<MaxTerms>& form, const normalized_linear_atom& atom, int coefficient) noexcept
    {
        if (!form.valid || coefficient == 0)
            return;

        for (std::size_t i = 0; i < form.count; ++i)
        {
            if (!same_normalized_atom(form.terms[i].atom, atom))
                continue;

            form.terms[i].coefficient += coefficient;
            if (form.terms[i].coefficient == 0)
                form.terms[i] = form.terms[--form.count];
            return;
        }

        if (form.count == MaxTerms)
        {
            form.valid = false;
            return;
        }

        form.terms[form.count++] = { atom, coefficient };
    }

    template<std::size_t MaxTerms, class Expr>
    BL_FORCE_INLINE constexpr void collect_normalized_linear_form(normalized_linear_form<MaxTerms>& form, const Expr& expr, int coefficient) noexcept
    {
        if (!form.valid || coefficient == 0)
            return;

        using ExprType = clean_t<Expr>;

        if constexpr (is_leaf_v<ExprType>)
        {
            add_normalized_term(form, make_value_atom(expr), coefficient);
        }
        else if constexpr (is_add_v<ExprType>)
        {
            collect_normalized_linear_form(form, expr.left, coefficient);
            collect_normalized_linear_form(form, expr.right, coefficient);
        }
        else if constexpr (is_sub_v<ExprType>)
        {
            collect_normalized_linear_form(form, expr.left, coefficient);
            collect_normalized_linear_form(form, expr.right, -coefficient);
        }
        else if constexpr (is_leaf_product_v<ExprType>)
        {
            add_normalized_term(form, make_product_atom(expr.left, expr.right), coefficient);
        }
        else if constexpr (is_mul_double_v<ExprType>)
        {
            const normalized_scalar_coefficient scalar = normalized_integer_scalar(expr.right);
            if (!scalar.valid)
            {
                form.valid = false;
                return;
            }

            collect_normalized_linear_form(form, expr.left, coefficient * scalar.value);
        }
        else
        {
            form.valid = false;
        }
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool normalized_atom_is_square(const normalized_linear_atom& atom) noexcept
    {
        return atom.kind == normalized_linear_atom_kind::square;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool normalized_atom_is_product_like(const normalized_linear_atom& atom) noexcept
    {
        return atom.kind == normalized_linear_atom_kind::product || atom.kind == normalized_linear_atom_kind::square;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr const f256_s& normalized_atom_left(const normalized_linear_atom& atom) noexcept
    {
        return *atom.left;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr const f256_s& normalized_atom_right(const normalized_linear_atom& atom) noexcept
    {
        return atom.kind == normalized_linear_atom_kind::square ? *atom.left : *atom.right;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool normalized_atoms_are_squares(const normalized_linear_atom& a, const normalized_linear_atom& b) noexcept
    {
        return normalized_atom_is_square(a) && normalized_atom_is_square(b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_normalized_product_atom(const normalized_linear_atom& atom) noexcept
    {
        if (atom.kind == normalized_linear_atom_kind::square)
            return sqr_eval(normalized_atom_left(atom));

        return mul_eval(normalized_atom_left(atom), normalized_atom_right(atom));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_normalized_product_twice(const normalized_linear_atom& atom) noexcept
    {
        if (atom.kind == normalized_linear_atom_kind::square)
            return sqr_twice_eval(normalized_atom_left(atom));

        return mul_twice_eval(normalized_atom_left(atom), normalized_atom_right(atom));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_product_value(
        const normalized_linear_term& product_term, const normalized_linear_term& value_term) noexcept
    {
        const normalized_linear_atom& product = product_term.atom;
        const f256_s& value = normalized_atom_left(value_term.atom);

        if (product_term.coefficient == 1 && value_term.coefficient == 1)
        {
            if (normalized_atom_is_square(product))
                return { true, sqr_add_eval(normalized_atom_left(product), value) };
            return { true, mul_add_eval(normalized_atom_left(product), normalized_atom_right(product), value) };
        }
        if (product_term.coefficient == 1 && value_term.coefficient == -1)
        {
            if (normalized_atom_is_square(product))
                return { true, sqr_sub_eval(normalized_atom_left(product), value) };
            return { true, mul_sub_eval(normalized_atom_left(product), normalized_atom_right(product), value) };
        }
        if (product_term.coefficient == -1 && value_term.coefficient == 1)
        {
            if (normalized_atom_is_square(product))
                return { true, value_sub_sqr_eval(value, normalized_atom_left(product)) };
            return { true, value_sub_mul_eval(value, normalized_atom_left(product), normalized_atom_right(product)) };
        }
        if (product_term.coefficient == 2 && value_term.coefficient == 1)
        {
            if (normalized_atom_is_square(product))
                return { true, sqr_twice_add_eval(normalized_atom_left(product), value) };
            return { true, mul_twice_add_eval(normalized_atom_left(product), normalized_atom_right(product), value) };
        }
        if (product_term.coefficient == 2 && value_term.coefficient == -1)
        {
            if (normalized_atom_is_square(product))
                return { true, sqr_twice_sub_eval(normalized_atom_left(product), value) };
            return { true, mul_twice_sub_eval(normalized_atom_left(product), normalized_atom_right(product), value) };
        }
        if (product_term.coefficient == -2 && value_term.coefficient == 1)
        {
            if (normalized_atom_is_square(product))
                return { true, value_sub_sqr_twice_eval(value, normalized_atom_left(product)) };
            return { true, value_sub_mul_twice_eval(value, normalized_atom_left(product), normalized_atom_right(product)) };
        }

        return {};
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_product_pair(
        const normalized_linear_term& first, const normalized_linear_term& second, const normalized_linear_term* value_term) noexcept
    {
        const normalized_linear_term* positive = &first;
        const normalized_linear_term* other = &second;
        bool subtract = false;

        if (first.coefficient == 1 && second.coefficient == 1)
        {
            subtract = false;
        }
        else if (first.coefficient == 1 && second.coefficient == -1)
        {
            subtract = true;
        }
        else if (first.coefficient == -1 && second.coefficient == 1)
        {
            positive = &second;
            other = &first;
            subtract = true;
        }
        else
        {
            return {};
        }

        const normalized_linear_atom& lhs = positive->atom;
        const normalized_linear_atom& rhs = other->atom;

        if (value_term == nullptr)
        {
            if (!subtract)
            {
                if (normalized_atoms_are_squares(lhs, rhs))
                    return { true, sqr_add_sqr_eval(normalized_atom_left(lhs), normalized_atom_left(rhs)) };
                return { true, mul_add_mul_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs)) };
            }

            if (normalized_atoms_are_squares(lhs, rhs))
                return { true, sqr_sub_sqr_eval(normalized_atom_left(lhs), normalized_atom_left(rhs)) };
            return { true, mul_sub_mul_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs)) };
        }

        const f256_s& value = normalized_atom_left(value_term->atom);

        if (!subtract)
        {
            if (value_term->coefficient == 1)
            {
                if (normalized_atoms_are_squares(lhs, rhs))
                    return { true, sqr_add_sqr_add_eval(normalized_atom_left(lhs), normalized_atom_left(rhs), value) };
                return { true, mul_add_mul_add_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs), value) };
            }
            if (value_term->coefficient == -1)
            {
                if (normalized_atoms_are_squares(lhs, rhs))
                    return { true, sqr_add_sqr_sub_eval(normalized_atom_left(lhs), normalized_atom_left(rhs), value) };
                return { true, mul_add_mul_sub_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs), value) };
            }

            return {};
        }

        if (value_term->coefficient == 1)
        {
            if (normalized_atoms_are_squares(lhs, rhs))
                return { true, sqr_sub_sqr_add_eval(normalized_atom_left(lhs), normalized_atom_left(rhs), value) };
            return { true, mul_sub_mul_add_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs), value) };
        }
        if (value_term->coefficient == -1)
        {
            if (normalized_atoms_are_squares(lhs, rhs))
                return { true, sqr_sub_sqr_sub_eval(normalized_atom_left(lhs), normalized_atom_left(rhs), value) };
            return { true, mul_sub_mul_sub_eval(normalized_atom_left(lhs), normalized_atom_right(lhs), normalized_atom_left(rhs), normalized_atom_right(rhs), value) };
        }

        return {};
    }

    template<class LeftProduct, class RightProduct, class ValueLeaf>
    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_product_pair_value_expr(
        const LeftProduct& left_product, int left_coefficient,
        const RightProduct& right_product, int right_coefficient,
        const ValueLeaf& value_leaf, int value_coefficient) noexcept
    {
        normalized_linear_term left{ make_product_atom(left_product.left, left_product.right), left_coefficient };
        normalized_linear_term right{ make_product_atom(right_product.left, right_product.right), right_coefficient };
        normalized_linear_term value{ make_value_atom(value_leaf), value_coefficient };

        if (same_normalized_atom(left.atom, right.atom))
        {
            left.coefficient += right.coefficient;
            return eval_normalized_product_value(left, value);
        }

        return eval_normalized_product_pair(left, right, &value);
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_product_pair_expr(
        const LeftProduct& left_product, int left_coefficient,
        const RightProduct& right_product, int right_coefficient) noexcept
    {
        normalized_linear_term left{ make_product_atom(left_product.left, left_product.right), left_coefficient };
        normalized_linear_term right{ make_product_atom(right_product.left, right_product.right), right_coefficient };

        if (same_normalized_atom(left.atom, right.atom))
        {
            left.coefficient += right.coefficient;
            if (left.coefficient == 1)
                return { true, eval_normalized_product_atom(left.atom) };
            if (left.coefficient == 2)
                return { true, eval_normalized_product_twice(left.atom) };

            return {};
        }

        return eval_normalized_product_pair(left, right, nullptr);
    }

    template<std::size_t MaxTerms>
    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_linear_form(const normalized_linear_form<MaxTerms>& form) noexcept
    {
        if (!form.valid)
            return {};

        std::array<std::size_t, MaxTerms> product_indices{};
        std::array<std::size_t, MaxTerms> value_indices{};
        std::size_t product_count{};
        std::size_t value_count{};

        for (std::size_t i = 0; i < form.count; ++i)
        {
            const normalized_linear_term& term = form.terms[i];

            if (term.atom.kind == normalized_linear_atom_kind::value)
                value_indices[value_count++] = i;
            else if (normalized_atom_is_product_like(term.atom))
                product_indices[product_count++] = i;
            else
                return {};
        }

        if (product_count == 1 && value_count == 0)
        {
            const normalized_linear_term& product = form.terms[product_indices[0]];
            if (product.coefficient == 1)
                return { true, eval_normalized_product_atom(product.atom) };
            if (product.coefficient == 2)
                return { true, eval_normalized_product_twice(product.atom) };
        }
        else if (product_count == 1 && value_count == 1)
        {
            return eval_normalized_product_value(form.terms[product_indices[0]], form.terms[value_indices[0]]);
        }
        else if (product_count == 2 && value_count == 0)
        {
            return eval_normalized_product_pair(form.terms[product_indices[0]], form.terms[product_indices[1]], nullptr);
        }
        else if (product_count == 2 && value_count == 1)
        {
            return eval_normalized_product_pair(form.terms[product_indices[0]], form.terms[product_indices[1]], &form.terms[value_indices[0]]);
        }

        return {};
    }

    template<class Expr>
    [[nodiscard]] BL_FORCE_INLINE constexpr normalized_linear_eval_result eval_normalized_linear_expr(const Expr& expr) noexcept
    {
        using ExprType = clean_t<Expr>;

        if constexpr (is_add_v<ExprType>)
        {
            using LeftType = clean_t<decltype(expr.left)>;
            using RightType = clean_t<decltype(expr.right)>;

            if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
                return eval_normalized_product_pair_value_expr(expr.left.left, 1, expr.left.right, 1, expr.right, 1);
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
                return eval_normalized_product_pair_value_expr(expr.left.left, 1, expr.left.right, -1, expr.right, 1);
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_product_v<RightType>)
                return eval_normalized_product_pair_expr(expr.left, 1, expr.right, 1);
        }
        else if constexpr (is_sub_v<ExprType>)
        {
            using LeftType = clean_t<decltype(expr.left)>;
            using RightType = clean_t<decltype(expr.right)>;

            if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
                return eval_normalized_product_pair_value_expr(expr.left.left, 1, expr.left.right, 1, expr.right, -1);
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
                return eval_normalized_product_pair_value_expr(expr.left.left, 1, expr.left.right, -1, expr.right, -1);
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_product_v<RightType>)
                return eval_normalized_product_pair_expr(expr.left, 1, expr.right, -1);
        }

        normalized_linear_form<normalized_linear_max_terms> form{};
        collect_normalized_linear_form(form, expr, 1);
        return eval_normalized_linear_form(form);
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr bool leaf_product_is_square(const Product& product) noexcept
    {
        return same_leaf_value(product.left, product.right);
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product(const Product& product) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_eval(leaf_value(product.left));

        return mul_eval(leaf_value(product.left), leaf_value(product.right));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_value(const Product& product, const leaf_expr& value) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_add_eval(leaf_value(product.left), leaf_value(value));

        return mul_add_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(value));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_value(const Product& product, const leaf_expr& value) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_sub_eval(leaf_value(product.left), leaf_value(value));

        return mul_sub_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(value));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_value_sub_leaf_product(const leaf_expr& value, const Product& product) noexcept
    {
        if (leaf_product_is_square(product))
            return value_sub_sqr_eval(leaf_value(value), leaf_value(product.left));

        return value_sub_mul_eval(leaf_value(value), leaf_value(product.left), leaf_value(product.right));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_leaf_product(const LeftProduct& left, const RightProduct& right) noexcept
    {
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_add_sqr_eval(leaf_value(left.left), leaf_value(right.left));

        return mul_add_mul_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_leaf_product(const LeftProduct& left, const RightProduct& right) noexcept
    {
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_sub_sqr_eval(leaf_value(left.left), leaf_value(right.left));

        return mul_sub_mul_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr bool leaf_products_are_same(const LeftProduct& left, const RightProduct& right) noexcept
    {
        return (same_leaf_value(left.left, right.left) && same_leaf_value(left.right, right.right)) ||
               (same_leaf_value(left.left, right.right) && same_leaf_value(left.right, right.left));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_twice_add_value(const Product& product, const leaf_expr& value) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_twice_add_eval(leaf_value(product.left), leaf_value(value));

        return mul_twice_add_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(value));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_twice_sub_value(const Product& product, const leaf_expr& value) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_twice_sub_eval(leaf_value(product.left), leaf_value(value));

        return mul_twice_sub_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(value));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_value_sub_leaf_product_twice(const leaf_expr& value, const Product& product) noexcept
    {
        if (leaf_product_is_square(product))
            return value_sub_sqr_twice_eval(leaf_value(value), leaf_value(product.left));

        return value_sub_mul_twice_eval(leaf_value(value), leaf_value(product.left), leaf_value(product.right));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_leaf_product_add_value(const LeftProduct& left, const RightProduct& right, const leaf_expr& value) noexcept
    {
        if (leaf_products_are_same(left, right))
            return eval_leaf_product_twice_add_value(left, value);
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_add_sqr_add_eval(leaf_value(left.left), leaf_value(right.left), leaf_value(value));

        return mul_add_mul_add_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right),
            leaf_value(value));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_leaf_product_sub_value(const LeftProduct& left, const RightProduct& right, const leaf_expr& value) noexcept
    {
        if (leaf_products_are_same(left, right))
            return eval_leaf_product_twice_sub_value(left, value);
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_add_sqr_sub_eval(leaf_value(left.left), leaf_value(right.left), leaf_value(value));

        return mul_add_mul_sub_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right),
            leaf_value(value));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_leaf_product_add_value(const LeftProduct& left, const RightProduct& right, const leaf_expr& value) noexcept
    {
        if (leaf_products_are_same(left, right))
            return leaf_value(value);
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_sub_sqr_add_eval(leaf_value(left.left), leaf_value(right.left), leaf_value(value));

        return mul_sub_mul_add_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right),
            leaf_value(value));
    }

    template<class LeftProduct, class RightProduct>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_leaf_product_sub_value(const LeftProduct& left, const RightProduct& right, const leaf_expr& value) noexcept
    {
        if (leaf_product_is_square(left) && leaf_product_is_square(right))
            return sqr_sub_sqr_sub_eval(leaf_value(left.left), leaf_value(right.left), leaf_value(value));

        return mul_sub_mul_sub_eval(
            leaf_value(left.left), leaf_value(left.right),
            leaf_value(right.left), leaf_value(right.right),
            leaf_value(value));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_value_add_value(const Product& product, const leaf_expr& first, const leaf_expr& second) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_add_add_eval(leaf_value(product.left), leaf_value(first), leaf_value(second));

        return mul_add_add_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(first), leaf_value(second));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_add_value_sub_value(const Product& product, const leaf_expr& addend, const leaf_expr& subtrahend) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_add_sub_eval(leaf_value(product.left), leaf_value(addend), leaf_value(subtrahend));

        return mul_add_sub_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(addend), leaf_value(subtrahend));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_value_add_value(const Product& product, const leaf_expr& subtrahend, const leaf_expr& addend) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_sub_add_eval(leaf_value(product.left), leaf_value(subtrahend), leaf_value(addend));

        return mul_sub_add_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(subtrahend), leaf_value(addend));
    }

    template<class Product>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_leaf_product_sub_value_sub_value(const Product& product, const leaf_expr& first, const leaf_expr& second) noexcept
    {
        if (leaf_product_is_square(product))
            return sqr_sub_sub_eval(leaf_value(product.left), leaf_value(first), leaf_value(second));

        return mul_sub_sub_eval(leaf_value(product.left), leaf_value(product.right), leaf_value(first), leaf_value(second));
    }

    template<class Numerator>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_div_with_denominator(const Numerator& numerator, const f256_s& denominator) noexcept
    {
        using NumeratorType = clean_t<Numerator>;

        if constexpr (is_product_add_leaf_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left))
                return div_eval(sqr_add_eval(leaf_value(numerator.left.left), leaf_value(numerator.right)), denominator);

            return mul_add_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_product_sub_leaf_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left))
                return div_eval(sqr_sub_eval(leaf_value(numerator.left.left), leaf_value(numerator.right)), denominator);

            return mul_sub_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_leaf_product_v<NumeratorType>)
        {
            return div_eval(eval_leaf_product(numerator), denominator);
        }
        else if constexpr (is_add_product_product_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left) && leaf_product_is_square(numerator.right))
                return div_eval(sqr_add_sqr_eval(leaf_value(numerator.left.left), leaf_value(numerator.right.left)), denominator);

            return mul_add_mul_div_eval(
                leaf_value(numerator.left.left), leaf_value(numerator.left.right),
                leaf_value(numerator.right.left), leaf_value(numerator.right.right),
                denominator);
        }
        else if constexpr (is_sub_product_product_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left) && leaf_product_is_square(numerator.right))
                return div_eval(sqr_sub_sqr_eval(leaf_value(numerator.left.left), leaf_value(numerator.right.left)), denominator);

            return mul_sub_mul_div_eval(
                leaf_value(numerator.left.left), leaf_value(numerator.left.right),
                leaf_value(numerator.right.left), leaf_value(numerator.right.right),
                denominator);
        }
        else if constexpr (is_leaf_add_add_leaf_v<NumeratorType>)
        {
            return add_add_add_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_leaf_sub_add_leaf_v<NumeratorType>)
        {
            return add_sub_add_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_leaf_add_sub_leaf_v<NumeratorType>)
        {
            return add_add_sub_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_leaf_sub_sub_leaf_v<NumeratorType>)
        {
            return add_sub_sub_div_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator);
        }
        else if constexpr (is_add_v<NumeratorType>)
        {
            using LeftType = clean_t<decltype(numerator.left)>;
            using RightType = clean_t<decltype(numerator.right)>;

            if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_mul_double_div_eval(leaf_value(numerator.right), eval_to_f256_s(numerator.left.left), numerator.left.right, denominator);
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return add_mul_double_div_eval(leaf_value(numerator.left), eval_to_f256_s(numerator.right.left), numerator.right.right, denominator);
            }
            else
            {
                return div_eval(eval_to_f256_s(numerator), denominator);
            }
        }
        else if constexpr (is_sub_v<NumeratorType>)
        {
            using LeftType = clean_t<decltype(numerator.left)>;
            using RightType = clean_t<decltype(numerator.right)>;

            if constexpr (is_leaf_v<LeftType> && is_leaf_product_v<RightType>)
            {
                if (leaf_product_is_square(numerator.right))
                    return div_eval(value_sub_sqr_eval(leaf_value(numerator.left), leaf_value(numerator.right.left)), denominator);

                return value_sub_mul_div_eval(leaf_value(numerator.left), leaf_value(numerator.right.left), leaf_value(numerator.right.right), denominator);
            }
            else if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_double_sub_div_eval(eval_to_f256_s(numerator.left.left), numerator.left.right, leaf_value(numerator.right), denominator);
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return sub_mul_double_div_eval(leaf_value(numerator.left), eval_to_f256_s(numerator.right.left), numerator.right.right, denominator);
            }
            else
            {
                return div_eval(eval_to_f256_s(numerator), denominator);
            }
        }
        else
        {
            return div_eval(eval_to_f256_s(numerator), denominator);
        }
    }

    template<class Numerator>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_div_with_add_double_denominator(const Numerator& numerator, const f256_s& denominator, double scalar) noexcept
    {
        using NumeratorType = clean_t<Numerator>;

        if constexpr (is_product_add_leaf_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left))
                return div_add_double_eval(sqr_add_eval(leaf_value(numerator.left.left), leaf_value(numerator.right)), denominator, scalar);

            return mul_add_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_product_sub_leaf_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left))
                return div_add_double_eval(sqr_sub_eval(leaf_value(numerator.left.left), leaf_value(numerator.right)), denominator, scalar);

            return mul_sub_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_add_product_product_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left) && leaf_product_is_square(numerator.right))
                return div_add_double_eval(sqr_add_sqr_eval(leaf_value(numerator.left.left), leaf_value(numerator.right.left)), denominator, scalar);

            return mul_add_mul_div_add_double_eval(
                leaf_value(numerator.left.left), leaf_value(numerator.left.right),
                leaf_value(numerator.right.left), leaf_value(numerator.right.right),
                denominator, scalar);
        }
        else if constexpr (is_sub_product_product_v<NumeratorType>)
        {
            if (leaf_product_is_square(numerator.left) && leaf_product_is_square(numerator.right))
                return div_add_double_eval(sqr_sub_sqr_eval(leaf_value(numerator.left.left), leaf_value(numerator.right.left)), denominator, scalar);

            return mul_sub_mul_div_add_double_eval(
                leaf_value(numerator.left.left), leaf_value(numerator.left.right),
                leaf_value(numerator.right.left), leaf_value(numerator.right.right),
                denominator, scalar);
        }
        else if constexpr (is_leaf_add_add_leaf_v<NumeratorType>)
        {
            return add_add_add_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_leaf_sub_add_leaf_v<NumeratorType>)
        {
            return add_sub_add_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_leaf_add_sub_leaf_v<NumeratorType>)
        {
            return add_add_sub_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_leaf_sub_sub_leaf_v<NumeratorType>)
        {
            return add_sub_sub_div_add_double_eval(leaf_value(numerator.left.left), leaf_value(numerator.left.right), leaf_value(numerator.right), denominator, scalar);
        }
        else if constexpr (is_add_v<NumeratorType>)
        {
            using LeftType = clean_t<decltype(numerator.left)>;
            using RightType = clean_t<decltype(numerator.right)>;

            if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_mul_double_div_add_double_eval(leaf_value(numerator.right), eval_to_f256_s(numerator.left.left), numerator.left.right, denominator, scalar);
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return add_mul_double_div_add_double_eval(leaf_value(numerator.left), eval_to_f256_s(numerator.right.left), numerator.right.right, denominator, scalar);
            }
            else
            {
                return div_add_double_eval(eval_to_f256_s(numerator), denominator, scalar);
            }
        }
        else if constexpr (is_sub_v<NumeratorType>)
        {
            using LeftType = clean_t<decltype(numerator.left)>;
            using RightType = clean_t<decltype(numerator.right)>;

            if constexpr (is_leaf_v<LeftType> && is_leaf_product_v<RightType>)
            {
                if (leaf_product_is_square(numerator.right))
                    return div_add_double_eval(value_sub_sqr_eval(leaf_value(numerator.left), leaf_value(numerator.right.left)), denominator, scalar);

                return value_sub_mul_div_add_double_eval(leaf_value(numerator.left), leaf_value(numerator.right.left), leaf_value(numerator.right.right), denominator, scalar);
            }
            else if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_double_sub_div_add_double_eval(eval_to_f256_s(numerator.left.left), numerator.left.right, leaf_value(numerator.right), denominator, scalar);
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return sub_mul_double_div_add_double_eval(leaf_value(numerator.left), eval_to_f256_s(numerator.right.left), numerator.right.right, denominator, scalar);
            }
            else
            {
                return div_add_double_eval(eval_to_f256_s(numerator), denominator, scalar);
            }
        }
        else
        {
            return div_add_double_eval(eval_to_f256_s(numerator), denominator, scalar);
        }
    }

    template<class Expr> [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_eager(const Expr& expr) noexcept;
    template<class Expr> [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_eager(const Expr& expr) noexcept
    {
        using ExprType = clean_t<Expr>;

        if constexpr (is_leaf_v<ExprType>)
        {
            return leaf_value(expr);
        }
        else if constexpr (is_mul_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            if (&left == &right)
                return sqr_eval(left);

            return mul_eval(left, right);
        }
        else if constexpr (is_mul_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return mul_pow2_or_double_eval(left, expr.right);
        }
        else if constexpr (is_add_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return add_double_eval(left, expr.right);
        }
        else if constexpr (is_double_sub_v<ExprType>)
        {
            const f256_s right = eval_eager(expr.right);

            return sub_double_eval(expr.left, right);
        }
        else if constexpr (is_div_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return div_double_eval(left, expr.right);
        }
        else if constexpr (is_double_div_v<ExprType>)
        {
            const f256_s right = eval_eager(expr.right);

            return div_double_eval(expr.left, right);
        }
        else if constexpr (is_add_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return add_eval(left, right);
        }
        else if constexpr (is_div_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return div_eval(left, right);
        }
        else
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return sub_eval(left, right);
        }
    }
    template<class Expr> [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_to_f256_s(const Expr& expr) noexcept
    {
        using ExprType = clean_t<Expr>;

        if constexpr (is_leaf_v<ExprType>)
        {
            return leaf_value(expr);
        }
        else if constexpr (is_mul_v<ExprType>)
        {
            if constexpr (is_leaf_product_v<ExprType>)
            {
                return eval_leaf_product(expr);
            }
            else
            {
                return eval_eager(expr);
            }
        }
        else if constexpr (is_mul_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return mul_pow2_or_double_eval(left, expr.right);
        }
        else if constexpr (is_add_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return add_double_eval(left, expr.right);
        }
        else if constexpr (is_double_sub_v<ExprType>)
        {
            const f256_s right = eval_to_f256_s(expr.right);

            return sub_double_eval(expr.left, right);
        }
        else if constexpr (is_div_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return div_double_eval(left, expr.right);
        }
        else if constexpr (is_double_div_v<ExprType>)
        {
            const f256_s right = eval_to_f256_s(expr.right);

            return div_double_eval(expr.left, right);
        }
        else if constexpr (is_div_v<ExprType>)
        {
            using RightType = clean_t<decltype(expr.right)>;

            if constexpr (is_add_double_v<RightType>)
            {
                const f256_s base_denominator = eval_to_f256_s(expr.right.left);

                return eval_div_with_add_double_denominator(expr.left, base_denominator, expr.right.right);
            }
            else if constexpr (is_double_sub_v<RightType>)
            {
                const f256_s left_value = eval_to_f256_s(expr.left);
                const f256_s base_denominator = eval_to_f256_s(expr.right.right);

                return div_double_sub_eval(left_value, expr.right.left, base_denominator);
            }
            else if constexpr (is_leaf_add_v<RightType>)
            {
                const f256_s left_value = eval_to_f256_s(expr.left);

                return div_add_eval(left_value, leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_sub_v<RightType>)
            {
                const f256_s left_value = eval_to_f256_s(expr.left);

                return div_sub_eval(left_value, leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else
            {
                const f256_s right_value = eval_to_f256_s(expr.right);

                return eval_div_with_denominator(expr.left, right_value);
            }
        }
        else if constexpr (is_add_v<ExprType>)
        {
            using LeftType = clean_t<decltype(expr.left)>;
            using RightType = clean_t<decltype(expr.right)>;

            if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_add_leaf_product_add_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_leaf_product_add_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_add_product_product_v<RightType>)
            {
                return eval_leaf_product_add_leaf_product_add_value(expr.right.left, expr.right.right, expr.left);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_sub_product_product_v<RightType>)
            {
                return eval_leaf_product_sub_leaf_product_add_value(expr.right.left, expr.right.right, expr.left);
            }

            const normalized_linear_eval_result normalized = eval_normalized_linear_expr(expr);
            if (normalized.matched)
                return normalized.value;

            if constexpr (is_leaf_add_add_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_add_add_add_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_add_v<LeftType> && is_leaf_add_v<RightType>)
            {
                return add_add_add_add_eval(
                    leaf_value(expr.left.left), leaf_value(expr.left.right),
                    leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_add_v<LeftType> && is_leaf_v<RightType>)
            {
                if (same_leaf_value(expr.left.left, expr.left.right))
                    return add_scaled_2_1_eval(leaf_value(expr.left.left), leaf_value(expr.right));

                return add_add_add_eval(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_sub_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_sub_add_eval(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_add_v<RightType>)
            {
                if (same_leaf_value(expr.right.left, expr.right.right))
                    return add_scaled_1_2_eval(leaf_value(expr.left), leaf_value(expr.right.left));

                return add_add_add_eval(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_sub_v<RightType>)
            {
                return add_sub_add_eval(leaf_value(expr.left), leaf_value(expr.right.right), leaf_value(expr.right.left));
            }
            else if constexpr (is_product_add_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_add_value_add_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_product_sub_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_value_add_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_add_v<RightType>)
            {
                return eval_leaf_product_add_value_add_value(expr.left, expr.right.left, expr.right.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_sub_v<RightType>)
            {
                return eval_leaf_product_add_value_sub_value(expr.left, expr.right.left, expr.right.right);
            }
            else if constexpr (is_leaf_v<LeftType> && is_product_add_leaf_v<RightType>)
            {
                return eval_leaf_product_add_value_add_value(expr.right.left, expr.right.right, expr.left);
            }
            else if constexpr (is_leaf_v<LeftType> && is_product_sub_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_value_add_value(expr.right.left, expr.right.right, expr.left);
            }
            else if constexpr (is_mul_double_add_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_double_add_mul_double_add_eval(
                    eval_to_f256_s(expr.left.left.left), expr.left.left.right,
                    eval_to_f256_s(expr.left.right.left), expr.left.right.right,
                    leaf_value(expr.right));
            }
            else if constexpr (is_mul_double_v<LeftType> && is_mul_double_v<RightType>)
            {
                return mul_double_add_mul_double_eval(
                    eval_to_f256_s(expr.left.left), expr.left.right,
                    eval_to_f256_s(expr.right.left), expr.right.right);
            }
            else if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_mul_double_eval(leaf_value(expr.right), eval_to_f256_s(expr.left.left), expr.left.right);
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return add_mul_double_eval(leaf_value(expr.left), eval_to_f256_s(expr.right.left), expr.right.right);
            }
            else if constexpr (is_add_product_product_add_add_product_product_v<ExprType>)
            {
                return mul_add_mul_add_mul_add_mul_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right.left.left), leaf_value(expr.right.left.right),
                    leaf_value(expr.right.right.left), leaf_value(expr.right.right.right));
            }
            else if constexpr (is_add_product_product_add_product_v<ExprType>)
            {
                return mul_add_mul_add_mul_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_add_product_product_v<RightType>)
            {
                return mul_add_mul_add_mul_eval(
                    leaf_value(expr.right.left.left), leaf_value(expr.right.left.right),
                    leaf_value(expr.right.right.left), leaf_value(expr.right.right.right),
                    leaf_value(expr.left.left), leaf_value(expr.left.right));
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_product_v<RightType>)
            {
                return eval_leaf_product_add_leaf_product(expr.left, expr.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_add_value(expr.left, expr.right);
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_product_v<RightType>)
            {
                return eval_leaf_product_add_value(expr.right, expr.left);
            }
            else if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_add_mul_add_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_sub_mul_add_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else
            {
                return eval_eager(expr);
            }
        }
        else
        {
            using LeftType = clean_t<decltype(expr.left)>;
            using RightType = clean_t<decltype(expr.right)>;

            if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_add_leaf_product_sub_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_leaf_product_sub_value(expr.left.left, expr.left.right, expr.right);
            }

            const normalized_linear_eval_result normalized = eval_normalized_linear_expr(expr);
            if (normalized.matched)
                return normalized.value;

            if constexpr (is_leaf_add_add_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_add_add_sub_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_add_sub_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_add_sub_sub_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_sub_sub_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_sub_sub_sub_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_add_v<LeftType> && is_leaf_add_v<RightType>)
            {
                return add_add_sub_sub_eval(
                    leaf_value(expr.left.left), leaf_value(expr.left.right),
                    leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_add_v<LeftType> && is_leaf_v<RightType>)
            {
                if (same_leaf_value(expr.left.left, expr.left.right))
                    return add_scaled_2_neg1_eval(leaf_value(expr.left.left), leaf_value(expr.right));

                return add_add_sub_eval(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_sub_v<LeftType> && is_leaf_v<RightType>)
            {
                return add_sub_sub_eval(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_add_v<RightType>)
            {
                if (same_leaf_value(expr.right.left, expr.right.right))
                    return add_scaled_1_neg2_eval(leaf_value(expr.left), leaf_value(expr.right.left));

                return add_sub_sub_eval(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_sub_v<RightType>)
            {
                return add_sub_add_eval(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_product_add_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_add_value_sub_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_product_sub_leaf_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_value_sub_value(expr.left.left, expr.left.right, expr.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_add_v<RightType>)
            {
                return eval_leaf_product_sub_value_sub_value(expr.left, expr.right.left, expr.right.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_sub_v<RightType>)
            {
                return eval_leaf_product_sub_value_add_value(expr.left, expr.right.left, expr.right.right);
            }
            else if constexpr (is_mul_double_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_double_sub_eval(eval_to_f256_s(expr.left.left), expr.left.right, leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<LeftType> && is_mul_double_v<RightType>)
            {
                return sub_mul_double_eval(leaf_value(expr.left), eval_to_f256_s(expr.right.left), expr.right.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_product_v<RightType>)
            {
                return eval_leaf_product_sub_leaf_product(expr.left, expr.right);
            }
            else if constexpr (is_leaf_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return eval_leaf_product_sub_value(expr.left, expr.right);
            }
            else if constexpr (is_leaf_v<LeftType> && is_leaf_product_v<RightType>)
            {
                return eval_value_sub_leaf_product(expr.left, expr.right);
            }
            else if constexpr (is_add_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_add_mul_sub_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else if constexpr (is_sub_product_product_v<LeftType> && is_leaf_v<RightType>)
            {
                return mul_sub_mul_sub_eval(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else
            {
                return eval_eager(expr);
            }
        }
    }

    template<class L, class R, std::enable_if_t<is_expr_v<L> && is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(L&& left, R&& right) noexcept
    {
        return add_expr<decltype(as_expr(std::forward<L>(left))), decltype(as_expr(std::forward<R>(right)))>{
            as_expr(std::forward<L>(left)),
            as_expr(std::forward<R>(right))
        };
    }

    template<class L, class R, std::enable_if_t<is_expr_v<L> && is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(L&& left, R&& right) noexcept
    {
        return sub_expr<decltype(as_expr(std::forward<L>(left))), decltype(as_expr(std::forward<R>(right)))>{
            as_expr(std::forward<L>(left)),
            as_expr(std::forward<R>(right))
        };
    }

    template<class L, class R, std::enable_if_t<is_expr_v<L> && is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, R&& right) noexcept
    {
        return mul_expr<decltype(as_expr(std::forward<L>(left))), decltype(as_expr(std::forward<R>(right)))>{
            as_expr(std::forward<L>(left)),
            as_expr(std::forward<R>(right))
        };
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, double right) noexcept
    {
        return mul_double_expr<decltype(as_expr(std::forward<L>(left)))>{ as_expr(std::forward<L>(left)), right };
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(double left, R&& right) noexcept
    {
        return mul_double_expr<decltype(as_expr(std::forward<R>(right)))>{ as_expr(std::forward<R>(right)), left };
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, float right) noexcept
    {
        return std::forward<L>(left) * static_cast<double>(right);
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(float left, R&& right) noexcept
    {
        return static_cast<double>(left) * std::forward<R>(right);
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(L&& left, double right) noexcept
    {
        return add_double_expr<decltype(as_expr(std::forward<L>(left)))>{ as_expr(std::forward<L>(left)), right };
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(double left, R&& right) noexcept
    {
        return add_double_expr<decltype(as_expr(std::forward<R>(right)))>{ as_expr(std::forward<R>(right)), left };
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(L&& left, double right) noexcept
    {
        return add_double_expr<decltype(as_expr(std::forward<L>(left)))>{ as_expr(std::forward<L>(left)), -right };
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(double left, R&& right) noexcept
    {
        return double_sub_expr<decltype(as_expr(std::forward<R>(right)))>{ left, as_expr(std::forward<R>(right)) };
    }

    template<class L, class R, std::enable_if_t<is_expr_v<L> && is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, R&& right) noexcept
    {
        return div_expr<decltype(as_expr(std::forward<L>(left))), decltype(as_expr(std::forward<R>(right)))>{
            as_expr(std::forward<L>(left)),
            as_expr(std::forward<R>(right))
        };
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, double right) noexcept
    {
        return div_double_expr<decltype(as_expr(std::forward<L>(left)))>{ as_expr(std::forward<L>(left)), right };
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(double left, R&& right) noexcept
    {
        return double_div_expr<decltype(as_expr(std::forward<R>(right)))>{ left, as_expr(std::forward<R>(right)) };
    }

    template<class L, std::enable_if_t<is_expr_v<L>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, float right) noexcept
    {
        return std::forward<L>(left) / static_cast<double>(right);
    }

    template<class R, std::enable_if_t<is_expr_v<R>, int> = 0>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(float left, R&& right) noexcept
    {
        return static_cast<double>(left) / std::forward<R>(right);
    }
}

template<class L, class R, std::enable_if_t<detail::_f256_expr::is_operand_v<L> && detail::_f256_expr::is_operand_v<R> && (detail::_f256_expr::is_f256_value_v<L> || detail::_f256_expr::is_f256_value_v<R>), int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(L&& left, R&& right) noexcept
{
    return detail::_f256_expr::add_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left))), decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, class R, std::enable_if_t<detail::_f256_expr::is_operand_v<L> && detail::_f256_expr::is_operand_v<R> && (detail::_f256_expr::is_f256_value_v<L> || detail::_f256_expr::is_f256_value_v<R>), int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(L&& left, R&& right) noexcept
{
    return detail::_f256_expr::sub_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left))), decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(L&& left, double right) noexcept
{
    return detail::_f256_expr::add_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        right
    };
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(double left, R&& right) noexcept
{
    return detail::_f256_expr::add_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<R>(right)),
        left
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(L&& left, double right) noexcept
{
    return detail::_f256_expr::add_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        -right
    };
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(double left, R&& right) noexcept
{
    return detail::_f256_expr::double_sub_expr<decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        left,
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(L&& left, float right) noexcept
{
    return std::forward<L>(left) + static_cast<double>(right);
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator+(float left, R&& right) noexcept
{
    return static_cast<double>(left) + std::forward<R>(right);
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(L&& left, float right) noexcept
{
    return std::forward<L>(left) - static_cast<double>(right);
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator-(float left, R&& right) noexcept
{
    return static_cast<double>(left) - std::forward<R>(right);
}

template<class L, class R, std::enable_if_t<detail::_f256_expr::is_operand_v<L> && detail::_f256_expr::is_operand_v<R> && (detail::_f256_expr::is_f256_value_v<L> || detail::_f256_expr::is_f256_value_v<R>), int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, R&& right) noexcept
{
    return detail::_f256_expr::mul_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left))), decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, double right) noexcept
{
    return detail::_f256_expr::mul_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        right
    };
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(double left, R&& right) noexcept
{
    return detail::_f256_expr::mul_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<R>(right)),
        left
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(L&& left, float right) noexcept
{
    return std::forward<L>(left) * static_cast<double>(right);
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator*(float left, R&& right) noexcept
{
    return static_cast<double>(left) * std::forward<R>(right);
}

template<class L, class R, std::enable_if_t<detail::_f256_expr::is_operand_v<L> && detail::_f256_expr::is_operand_v<R> && (detail::_f256_expr::is_f256_value_v<L> || detail::_f256_expr::is_f256_value_v<R>), int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, R&& right) noexcept
{
    return detail::_f256_expr::div_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left))), decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, double right) noexcept
{
    return detail::_f256_expr::div_double_expr<decltype(detail::_f256_expr::as_expr(std::forward<L>(left)))>{
        detail::_f256_expr::as_expr(std::forward<L>(left)),
        right
    };
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(double left, R&& right) noexcept
{
    return detail::_f256_expr::double_div_expr<decltype(detail::_f256_expr::as_expr(std::forward<R>(right)))>{
        left,
        detail::_f256_expr::as_expr(std::forward<R>(right))
    };
}

template<class L, std::enable_if_t<detail::_f256_expr::is_f256_value_v<L>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(L&& left, float right) noexcept
{
    return std::forward<L>(left) / static_cast<double>(right);
}

template<class R, std::enable_if_t<detail::_f256_expr::is_f256_value_v<R>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr auto operator/(float left, R&& right) noexcept
{
    return static_cast<double>(left) / std::forward<R>(right);
}

} // namespace bl

#endif
