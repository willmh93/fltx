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
    [[nodiscard]] BL_FORCE_INLINE constexpr bool same_leaf_value(const leaf_expr& a, const leaf_expr& b) noexcept
    {
        return std::bit_cast<std::uint64_t>(a.value.x0) == std::bit_cast<std::uint64_t>(b.value.x0) &&
               std::bit_cast<std::uint64_t>(a.value.x1) == std::bit_cast<std::uint64_t>(b.value.x1) &&
               std::bit_cast<std::uint64_t>(a.value.x2) == std::bit_cast<std::uint64_t>(b.value.x2) &&
               std::bit_cast<std::uint64_t>(a.value.x3) == std::bit_cast<std::uint64_t>(b.value.x3);
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
                return detail::_f256::sqr_inline(left);

            return detail::_f256::mul_inline(left, right);
        }
        else if constexpr (is_mul_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return detail::_f256::mul_pow2_or_double_inline(left, expr.right);
        }
        else if constexpr (is_add_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return detail::_f256::add_double_inline(left, expr.right);
        }
        else if constexpr (is_double_sub_v<ExprType>)
        {
            const f256_s right = eval_eager(expr.right);

            return detail::_f256::sub_double_inline(expr.left, right);
        }
        else if constexpr (is_div_double_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);

            return detail::_f256::div_double_inline(left, expr.right);
        }
        else if constexpr (is_double_div_v<ExprType>)
        {
            const f256_s right = eval_eager(expr.right);

            return detail::_f256::div_double_inline(expr.left, right);
        }
        else if constexpr (is_add_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return detail::_f256::add_inline(left, right);
        }
        else if constexpr (is_div_v<ExprType>)
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return detail::_f256::div_inline(left, right);
        }
        else
        {
            const f256_s left = eval_eager(expr.left);
            const f256_s right = eval_eager(expr.right);

            return detail::_f256::sub_inline(left, right);
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
                if (same_leaf_value(expr.left, expr.right))
                    return detail::_f256::sqr_inline(leaf_value(expr.left));

                return detail::_f256::mul_inline(leaf_value(expr.left), leaf_value(expr.right));
            }
            else
            {
                return eval_eager(expr);
            }
        }
        else if constexpr (is_mul_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return detail::_f256::mul_pow2_or_double_inline(left, expr.right);
        }
        else if constexpr (is_add_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return detail::_f256::add_double_inline(left, expr.right);
        }
        else if constexpr (is_double_sub_v<ExprType>)
        {
            const f256_s right = eval_to_f256_s(expr.right);

            return detail::_f256::sub_double_inline(expr.left, right);
        }
        else if constexpr (is_div_double_v<ExprType>)
        {
            const f256_s left = eval_to_f256_s(expr.left);

            return detail::_f256::div_double_inline(left, expr.right);
        }
        else if constexpr (is_double_div_v<ExprType>)
        {
            const f256_s right = eval_to_f256_s(expr.right);

            return detail::_f256::div_double_inline(expr.left, right);
        }
        else if constexpr (is_div_v<ExprType>)
        {
            using RightType = clean_t<decltype(expr.right)>;
            const f256_s left_value = eval_to_f256_s(expr.left);

            if constexpr (is_add_double_v<RightType>)
            {
                const f256_s base_denominator = eval_to_f256_s(expr.right.left);

                return detail::_f256::div_add_double_inline(left_value, base_denominator, expr.right.right);
            }
            else if constexpr (is_double_sub_v<RightType>)
            {
                const f256_s base_denominator = eval_to_f256_s(expr.right.right);

                return detail::_f256::div_double_sub_inline(left_value, expr.right.left, base_denominator);
            }
            else
            {
                const f256_s right_value = eval_to_f256_s(expr.right);

                return detail::_f256::div_inline(left_value, right_value);
            }
        }
        else if constexpr (is_add_v<ExprType>)
        {
            if constexpr (is_leaf_add_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                if (same_leaf_value(expr.left.left, expr.left.right))
                    return detail::_f256::add_scaled_inline<2, 1>(leaf_value(expr.left.left), leaf_value(expr.right));

                return detail::_f256::add_add_add_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_sub_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_sub_add_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_add_v<clean_t<decltype(expr.right)>>)
            {
                if (same_leaf_value(expr.right.left, expr.right.right))
                    return detail::_f256::add_scaled_inline<1, 2>(leaf_value(expr.left), leaf_value(expr.right.left));

                return detail::_f256::add_add_add_inline(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_sub_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_sub_add_inline(leaf_value(expr.left), leaf_value(expr.right.right), leaf_value(expr.right.left));
            }
            else if constexpr (is_mul_double_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_mul_double_inline(leaf_value(expr.right), eval_to_f256_s(expr.left.left), expr.left.right);
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_mul_double_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_mul_double_inline(leaf_value(expr.left), eval_to_f256_s(expr.right.left), expr.right.right);
            }
            else if constexpr (is_leaf_product_v<clean_t<decltype(expr.left)>> && is_leaf_product_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_add_mul_inline(
                    leaf_value(expr.left.left), leaf_value(expr.left.right),
                    leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_add_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_product_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_add_inline(leaf_value(expr.right.left), leaf_value(expr.right.right), leaf_value(expr.left));
            }
            else if constexpr (is_add_product_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_add_mul_add_inline(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else if constexpr (is_sub_product_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_sub_mul_add_inline(
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
            if constexpr (is_leaf_add_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                if (same_leaf_value(expr.left.left, expr.left.right))
                    return detail::_f256::add_scaled_inline<2, -1>(leaf_value(expr.left.left), leaf_value(expr.right));

                return detail::_f256::add_add_sub_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_sub_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_sub_sub_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_add_v<clean_t<decltype(expr.right)>>)
            {
                if (same_leaf_value(expr.right.left, expr.right.right))
                    return detail::_f256::add_scaled_inline<1, -2>(leaf_value(expr.left), leaf_value(expr.right.left));

                return detail::_f256::add_sub_sub_inline(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_sub_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::add_sub_add_inline(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_mul_double_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_double_sub_inline(eval_to_f256_s(expr.left.left), expr.left.right, leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_mul_double_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::sub_mul_double_inline(leaf_value(expr.left), eval_to_f256_s(expr.right.left), expr.right.right);
            }
            else if constexpr (is_leaf_product_v<clean_t<decltype(expr.left)>> && is_leaf_product_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_sub_mul_inline(
                    leaf_value(expr.left.left), leaf_value(expr.left.right),
                    leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_leaf_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_sub_inline(leaf_value(expr.left.left), leaf_value(expr.left.right), leaf_value(expr.right));
            }
            else if constexpr (is_leaf_v<clean_t<decltype(expr.left)>> && is_leaf_product_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::value_sub_mul_inline(leaf_value(expr.left), leaf_value(expr.right.left), leaf_value(expr.right.right));
            }
            else if constexpr (is_add_product_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_add_mul_sub_inline(
                    leaf_value(expr.left.left.left), leaf_value(expr.left.left.right),
                    leaf_value(expr.left.right.left), leaf_value(expr.left.right.right),
                    leaf_value(expr.right));
            }
            else if constexpr (is_sub_product_product_v<clean_t<decltype(expr.left)>> && is_leaf_v<clean_t<decltype(expr.right)>>)
            {
                return detail::_f256::mul_sub_mul_sub_inline(
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
