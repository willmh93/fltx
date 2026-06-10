/**
 * fltx/detail/f256_arithmetic.h - Low-level quad-double arithmetic helpers for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_DETAIL_ARITHMETIC_INCLUDED
#define F256_DETAIL_ARITHMETIC_INCLUDED
#include "fltx/detail/f256_expansion.h"
#include "fltx/f256_classification.h"
#include "fltx/f256_limits.h"

namespace bl {

namespace detail::_f256_runtime
{
    // core operations
    [[nodiscard]] BL_NO_INLINE f256_s add(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div(const f256_s& a, const f256_s& b) noexcept;

    // double-double operations
    [[nodiscard]] BL_NO_INLINE f256_s add_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_dd(detail::_f256::dd_scalar a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_dd(detail::_f256::dd_scalar a, const f256_s& b) noexcept;

    // double operations
    [[nodiscard]] BL_NO_INLINE f256_s add_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_double(double a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_double(double a, const f256_s& b) noexcept;

    // fused operations
    [[nodiscard]] BL_NO_INLINE f256_s sqr(const f256_s& a) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_pow2_or_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s value_sub_mul(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_mul_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_mul_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_add_mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f, const f256_s& g, const f256_s& h) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_add_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_add_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_scaled_2_1(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_scaled_1_2(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_scaled_2_neg1(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_scaled_1_neg2(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_mul_double(const f256_s& addend, const f256_s& value, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_mul_double(const f256_s& minuend, const f256_s& value, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double_sub(const f256_s& value, double scalar, const f256_s& subtrahend) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double_add_mul_double(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double_add_mul_double_add(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar, const f256_s& c) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_add(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_sub(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_add_double(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_double_sub(const f256_s& numerator, double scalar, const f256_s& base_denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s value_sub_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_mul_double_div(const f256_s& addend, const f256_s& value, double scalar, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_mul_double_div(const f256_s& minuend, const f256_s& value, double scalar, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double_sub_div(const f256_s& value, double scalar, const f256_s& subtrahend, const f256_s& denominator) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s value_sub_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_add_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_sub_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_add_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_sub_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s add_mul_double_div_add_double(const f256_s& addend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_mul_double_div_add_double(const f256_s& minuend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double_sub_div_add_double(const f256_s& value, double value_scalar, const f256_s& subtrahend, const f256_s& denominator, double denominator_scalar) noexcept;

    // runtime math
    [[nodiscard]] BL_NO_INLINE f256_s floor(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s ceil(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s trunc(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s pow10_256(int k);

} // namespace detail::_f256_runtime


namespace detail::_f256 // primitives and kernels
{
    // public arithmetic special cases
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s quiet_nan() noexcept
    {
        return { std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s signed_infinity(bool negative) noexcept
    {
        const double inf = std::numeric_limits<double>::infinity();
        return { negative ? -inf : inf, 0.0, 0.0, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s signed_zero(bool negative) noexcept
    {
        return { std::bit_cast<double>(negative ? 0x8000000000000000ull : 0ull), 0.0, 0.0, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_special(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isnan(a.x0) || detail::fp::isnan(b.x0))
            return quiet_nan();

        const bool a_inf = isinf(a.x0);
        const bool b_inf = isinf(b.x0);
        if (a_inf && b_inf && signbit(a.x0) != signbit(b.x0))
            return quiet_nan();
        if (a_inf)
            return signed_infinity(signbit(a.x0));
        return signed_infinity(signbit(b.x0));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_special(const f256_s& a, const f256_s& b) noexcept
    {
        return add_special(a, f256_s{ -b.x0, -b.x1, -b.x2, -b.x3 });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_special(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isnan(a.x0) || detail::fp::isnan(b.x0))
            return quiet_nan();

        const bool a_inf = isinf(a.x0);
        const bool b_inf = isinf(b.x0);
        if ((a_inf && b.x0 == 0.0) || (b_inf && a.x0 == 0.0))
            return quiet_nan();

        return signed_infinity(bl::signbit(a) != bl::signbit(b));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_special(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isnan(a.x0) || detail::fp::isnan(b.x0))
            return quiet_nan();

        const bool a_zero = a.x0 == 0.0;
        const bool b_zero = b.x0 == 0.0;
        const bool negative = bl::signbit(a) != bl::signbit(b);
        if (b_zero)
            return a_zero ? quiet_nan() : signed_infinity(negative);

        const bool a_inf = isinf(a.x0);
        const bool b_inf = isinf(b.x0);
        if (a_inf && b_inf)
            return quiet_nan();
        if (a_inf)
            return signed_infinity(negative);
        return signed_zero(negative);
    }

    // scalar accumulation
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_scalar_precise(const f256_s& a, double b) noexcept
    {
        double s0{}, e0{}; two_sum_precise(a.x0, b, s0, e0);
        double s1{}, e1{}; two_sum_precise(a.x1, e0, s1, e1);
        double s2{}, e2{}; two_sum_precise(a.x2, e1, s2, e2);
        double s3{}, e3{}; two_sum_precise(a.x3, e2, s3, e3);

        return renorm5(s0, s1, s2, s3, e3);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s from_expansion_fast(const double* h, int n) noexcept
    {
        if (n <= 0) return {};

        double comp[40]{};
        const int m = compress_expansion_zeroelim(n, h, comp);

        f256_s sum{};
        for (int i = 0; i < m; ++i)
            sum = add_scalar_precise(sum, comp[i]);

        return sum;
    }

    // core arithmetic
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

		#if BL_F256_ENABLE_SIMD
        if (f256_runtime_addsub_simd_enabled())
        {
            const simd::f64x2 a01 = simd::f64x2_set(a.x0, a.x1);
            const simd::f64x2 b01 = simd::f64x2_set(b.x0, b.x1);
            const simd::f64x2 a23 = simd::f64x2_set(a.x2, a.x3);
            const simd::f64x2 b23 = simd::f64x2_set(b.x2, b.x3);
            simd::f64x2 s01{}, e01{}, s23{}, e23{};
            simd::f64x2_two_sum(a01, b01, s01, e01);
            simd::f64x2_two_sum(a23, b23, s23, e23);
            simd::f64x2_store(s01, s0, s1);
            simd::f64x2_store(e01, e0, e1);
            simd::f64x2_store(s23, s2, s3);
            simd::f64x2_store(e23, e2, e3);
        }
        else
		#endif
        {
            two_sum_precise(a.x0, b.x0, s0, e0);
            two_sum_precise(a.x1, b.x1, s1, e1);
            two_sum_precise(a.x2, b.x2, s2, e2);
            two_sum_precise(a.x3, b.x3, s3, e3);
        }
        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        #if BL_F256_ENABLE_SIMD
        if (f256_runtime_addsub_simd_enabled())
        {
            const simd::f64x2 a01 = simd::f64x2_set(a.x0, a.x1);
            const simd::f64x2 b01 = simd::f64x2_set(-b.x0, -b.x1);
            const simd::f64x2 a23 = simd::f64x2_set(a.x2, a.x3);
            const simd::f64x2 b23 = simd::f64x2_set(-b.x2, -b.x3);
            simd::f64x2 s01{}, e01{}, s23{}, e23{};
            simd::f64x2_two_sum(a01, b01, s01, e01);
            simd::f64x2_two_sum(a23, b23, s23, e23);
            simd::f64x2_store(s01, s0, s1);
            simd::f64x2_store(e01, e0, e1);
            simd::f64x2_store(s23, s2, s3);
            simd::f64x2_store(e23, e2, e3);
        }
        else
		#endif
        {
            two_sum_precise(a.x0, -b.x0, s0, e0);
            two_sum_precise(a.x1, -b.x1, s1, e1);
            two_sum_precise(a.x2, -b.x2, s2, e2);
            two_sum_precise(a.x3, -b.x3, s3, e3);
        }
        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_SSE2 || BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f256_runtime_product_simd_enabled())
        {
            simd::f64x2 p01{}, q01{};
            simd::f64x2 p23{}, q23{};
            simd::f64x2 p45{}, q45{};
            simd::f64x2 p67{}, q67{};
            simd::f64x2 p89{}, q89{};

            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x0), simd::f64x2_set(b.x0, b.x1), p01, q01);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x1, a.x0), simd::f64x2_set(b.x0, b.x2), p23, q23);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x1, a.x2), simd::f64x2_set(b.x1, b.x0), p45, q45);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x1), simd::f64x2_set(b.x3, b.x2), p67, q67);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x2, a.x3), simd::f64x2_set(b.x1, b.x0), p89, q89);

            simd::f64x2_store(p01, p0, p1);
            simd::f64x2_store(q01, q0, q1);
            simd::f64x2_store(p23, p2, p3);
            simd::f64x2_store(q23, q2, q3);
            simd::f64x2_store(p45, p4, p5);
            simd::f64x2_store(q45, q4, q5);
            simd::f64x2_store(p67, p6, p7);
            simd::f64x2_store(q67, q6, q7);
            simd::f64x2_store(p89, p8, p9);
            simd::f64x2_store(q89, q8, q9);
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

        return renorm5(p0, p1, s0, t0, t1);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        //if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
        //    return a / b.x0;

        const double inv_b0 = 1.0 / b.x0;

        const double q0 = a.x0 * inv_b0;
        if (detail::fp::isinf_or_nan(q0)) [[unlikely]]
            return f256_s{ q0, 0.0, 0.0, 0.0 };
        if (q0 == 0.0 && bl::iszero(a)) [[unlikely]]
            return signed_zero(bl::signbit(a) != bl::signbit(b));

        f256_s r = sub_mul_scalar_exact(a, b, q0);

        const double q1 = r.x0 * inv_b0;
        r = sub_mul_scalar_exact(r, b, q1);

        const double q2 = r.x0 * inv_b0;
        r = sub_mul_scalar_fast(r, b, q2);

        const double q3 = r.x0 * inv_b0;
        r = sub_mul_scalar_fast(r, b, q3);

        const double q4 = r.x0 * inv_b0;

        return renorm5(q0, q1, q2, q3, q4);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_inline(const f256_s& a) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p01{}, q01{};
            simd::f64x2 p34{}, q34{};
            simd::f64x2 p67{}, q67{};

            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x0), simd::f64x2_set(a.x0, a.x1), p01, q01);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x1), simd::f64x2_set(a.x2, a.x1), p34, q34);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x1), simd::f64x2_set(a.x3, a.x2), p67, q67);

            simd::f64x2_store(p01, p0, p1);
            simd::f64x2_store(q01, q0, q1);
            simd::f64x2_store(p34, p3, p4);
            simd::f64x2_store(q34, q3, q4);
            simd::f64x2_store(p67, p6, p7);
            simd::f64x2_store(q67, q6, q7);
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

return renorm5(p0, p1, s0, t0, t1);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_inline(const f256_s& a, double b) noexcept
    {
        double c0{}, c1{}, c2{}, c3{}, e{};

        two_sum_precise(a.x0, b, c0, e);
        if (e == 0.0) return renorm4(c0, a.x1, a.x2, a.x3);

        two_sum_precise(a.x1, e, c1, e);
        if (e == 0.0) return renorm4(c0, c1, a.x2, a.x3);

        two_sum_precise(a.x2, e, c2, e);
        if (e == 0.0) return renorm4(c0, c1, c2, a.x3);

        two_sum_precise(a.x3, e, c3, e);
        if (e == 0.0) return renorm4(c0, c1, c2, c3);

        return renorm5(c0, c1, c2, c3, e);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_inline(double a, const f256_s& b) noexcept
    {
        return add_double_inline(b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_inline(const f256_s& a, double b) noexcept
    {
        return add_double_inline(a, -b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_inline(double a, const f256_s& b) noexcept
    {
        return add_double_inline(-b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_inline(const f256_s& a, double b) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p01{}, q01{};
            simd::f64x2 p23{}, q23{};
            const simd::f64x2 bv = simd::f64x2_splat(b);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x0, a.x1), bv, p01, q01);
            simd::f64x2_two_prod_precise(simd::f64x2_set(a.x2, a.x3), bv, p23, q23);
            double ignored{};
            simd::f64x2_store(p01, p0, p1);
            simd::f64x2_store(q01, q0, q1);
            simd::f64x2_store(p23, p2, p3);
            simd::f64x2_store(q23, q2, ignored);
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

        return renorm5(s0, s1, s2, s3, s4);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_inline(double a, const f256_s& b) noexcept
    {
        return mul_double_inline(b, a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_inline(const f256_s& a, double b) noexcept
    {
        using namespace detail::_f256;

        if (bl::detail::use_constexpr_math())
        {
            if (detail::fp::isnan(a.x0) || detail::fp::isnan(b))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (isinf(b))
            {
                if (isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit(a.x0) ^ signbit(b);
                return signed_zero(neg);
            }

            if (b == 0.0)
            {
                if (iszero(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit(a.x0) ^ signbit(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (isinf(a))
            {
                const bool neg = signbit(a.x0) ^ signbit(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        const double inv_b = 1.0 / b;
        const f256_s divisor{ b, 0.0, 0.0, 0.0 };

        const double q0 = a.x0 * inv_b;
        if (detail::fp::isinf_or_nan(q0)) [[unlikely]]
            return f256_s{ q0, 0.0, 0.0, 0.0 };
        if (q0 == 0.0 && bl::iszero(a)) [[unlikely]]
            return signed_zero(bl::signbit(a) != signbit(b));

        f256_s r = sub_mul_scalar_exact(a, divisor, q0);

        const double q1 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q1);
        const double q2 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q2);
        const double q3 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q3);
        const double q4 = r.x0 * inv_b;

        return renorm5(q0, q1, q2, q3, q4);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_inline(double a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (bl::detail::use_constexpr_math())
        {
            if (detail::fp::isnan(a) || detail::fp::isnan(b.x0))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (isinf(b))
            {
                if (isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit(a) ^ signbit(b.x0);
                return signed_zero(neg);
            }

            if (iszero(b))
            {
                if (a == 0.0)
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit(a) ^ signbit(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (isinf(a))
            {
                const bool neg = signbit(a) ^ signbit(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return div_double_inline(f256_s{ a, 0.0, 0.0, 0.0 }, b.x0);

        const double inv_b0 = 1.0 / b.x0;
        const double q0     = a * inv_b0;
        if (detail::fp::isinf_or_nan(q0)) [[unlikely]]
            return f256_s{ q0, 0.0, 0.0, 0.0 };
        if (q0 == 0.0 && a == 0.0) [[unlikely]]
            return signed_zero(signbit(a) != bl::signbit(b));

        double p0{}, p1{}, p2{}, p3{};
        double e0{}, e1{}, e2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        two_prod_precise(b.x0, q0, p0, e0);
        two_prod_precise(b.x1, q0, p1, e1);
        two_prod_precise(b.x2, q0, p2, e2);
        p3 = b.x3 * q0;

        s0 = p0;
        two_sum_precise(e0, p1, s1, s2);
        three_sum(s2, e1, p2);
        three_sum2(e1, e2, p3);
        s3 = e1;
        s4 = e2 + p2;

        double c0{}, t0{};
        two_sum_precise(a, -s0, c0, t0);

        double c1 = -s1;
        double c2 = -s2;
        double c3 = -s3;
        double t1 = 0.0;
        double t2 = 0.0;

        two_sum_precise(c1, t0, c1, t0);
        three_sum(c2, t0, t1);
        three_sum2(c3, t0, t2);
        t0 += t1 - s4;

        f256_s r = renorm5(c0, c1, c2, c3, t0);

        const double q1 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q1);
        const double q2 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q2);
        const double q3 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q3);
        const double q4 = r.x0 * inv_b0;

        return renorm5(q0, q1, q2, q3, q4);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_checked_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b.x0)) [[unlikely]]
            return add_special(a, b);

        const f256_s out = add_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_checked_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b.x0)) [[unlikely]]
            return sub_special(a, b);

        const f256_s out = sub_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_checked_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b.x0)) [[unlikely]]
            return mul_special(a, b);
        if (bl::iszero(a) || bl::iszero(b)) [[unlikely]]
            return signed_zero(bl::signbit(a) != bl::signbit(b));

        const f256_s out = mul_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a) != bl::signbit(b));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_checked_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(b.x0)) [[unlikely]]
            return div_special(a, b);

        return div_inline(a, b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_checked_inline(const f256_s& a, double b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b)) [[unlikely]]
            return add_special(a, f256_s{ b, 0.0, 0.0, 0.0 });

        const f256_s out = add_double_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_checked_inline(const f256_s& a, double b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b)) [[unlikely]]
            return sub_special(a, f256_s{ b, 0.0, 0.0, 0.0 });

        const f256_s out = sub_double_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_checked_inline(double a, const f256_s& b) noexcept
    {
        if (detail::fp::isinf_or_nan(a) || detail::fp::isinf_or_nan(b.x0)) [[unlikely]]
            return sub_special(f256_s{ a, 0.0, 0.0, 0.0 }, b);

        const f256_s out = sub_double_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(signbit(a));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_checked_inline(const f256_s& a, double b) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0) || detail::fp::isinf_or_nan(b)) [[unlikely]]
            return mul_special(a, f256_s{ b, 0.0, 0.0, 0.0 });
        if (bl::iszero(a) || b == 0.0) [[unlikely]]
            return signed_zero(bl::signbit(a) != signbit(b));

        const f256_s out = mul_double_inline(a, b);
        if (detail::fp::isinf_or_nan(out.x0)) [[unlikely]]
            return signed_infinity(bl::signbit(a) != signbit(b));
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_checked_inline(const f256_s& a, double b) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(b)) [[unlikely]]
            return div_special(a, f256_s{ b, 0.0, 0.0, 0.0 });

        return div_double_inline(a, b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_checked_inline(double a, const f256_s& b) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(b.x0)) [[unlikely]]
            return div_special(f256_s{ a, 0.0, 0.0, 0.0 }, b);

        return div_double_inline(a, b);
    }

    // residual helpers
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_fast(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};
        double p2{}, e2{};
        double p3{}, e3{};

        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p01{}, e01{};
            simd::f64x2 p23{}, e23{};
            const simd::f64x2 qv = simd::f64x2_splat(q);
            simd::f64x2_two_prod_precise(simd::f64x2_set(b.x0, b.x1), qv, p01, e01);
            simd::f64x2_two_prod_precise(simd::f64x2_set(b.x2, b.x3), qv, p23, e23);
            simd::f64x2_store(p01, p0, p1);
            simd::f64x2_store(e01, e0, e1);
            simd::f64x2_store(p23, p2, p3);
            simd::f64x2_store(e23, e2, e3);
        }
        else
        #endif
        {
            two_prod_precise(b.x0, q, p0, e0);
            two_prod_precise(b.x1, q, p1, e1);
            two_prod_precise(b.x2, q, p2, e2);
            two_prod_precise(b.x3, q, p3, e3);
        }

        double s0 = r.x0-p0; double v0 = s0-r.x0; double u0 = s0-v0; double w0 = r.x0-u0;  u0 = -p0 - v0;
        double s1 = r.x1-p1; double v1 = s1-r.x1; double u1 = s1-v1; double w1 = r.x1-u1;  u1 = -p1 - v1;
        double s2 = r.x2-p2; double v2 = s2-r.x2; double u2 = s2-v2; double w2 = r.x2-u2;  u2 = -p2 - v2;
        double s3 = r.x3-p3; double v3 = s3-r.x3; double u3 = s3-v3; double w3 = r.x3-u3;  u3 = -p3 - v3;

        double t0 = w0 + u0;
        double t1 = w1 + u1;
        double t2 = w2 + u2;
        double t3 = w3 + u3;

        double tail0 = t0 - e0; two_sum_precise(s1, tail0, s1, t0);
        double tail1 = t1 - e1; three_sum(s2, t0, tail1);
        double tail2 = t2 - e2; three_sum2(s3, t0, tail2);

        t0 = t0 + tail1 + t3 - e3;

        return renorm5(s0, s1, s2, s3, t0);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_exact(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p01{}, q01{};
            simd::f64x2 p23{}, q23{};
            const simd::f64x2 qv = simd::f64x2_splat(q);
            simd::f64x2_two_prod_precise(simd::f64x2_set(b.x0, b.x1), qv, p01, q01);
            simd::f64x2_two_prod_precise(simd::f64x2_set(b.x2, b.x3), qv, p23, q23);
            double ignored{};
            simd::f64x2_store(p01, p0, p1);
            simd::f64x2_store(q01, q0, q1);
            simd::f64x2_store(p23, p2, p3);
            simd::f64x2_store(q23, q2, ignored);
        }
        else
        #endif
        {
            two_prod_precise(b.x0, q, p0, q0);
            two_prod_precise(b.x1, q, p1, q1);
            two_prod_precise(b.x2, q, p2, q2);
            p3 = b.x3 * q;
        }

        s0 = p0;
        two_sum_precise(q0, p1, s1, s2);
        three_sum(s2, q1, p2);
        three_sum2(q1, q2, p3);
        s3 = q1;
        s4 = q2 + p2;

        double c0{}, e0{};
        double c1{}, e1{};
        double c2{}, e2{};
        double c3{}, e3{};

        two_sum_precise(r.x0, -s0, c0, e0);
        two_sum_precise(r.x1, -s1, c1, e1);
        two_sum_precise(r.x2, -s2, c2, e2);
        two_sum_precise(r.x3, -s3, c3, e3);

        two_sum_precise(c1, e0, c1, e0);
        three_sum(c2, e0, e1);
        three_sum2(c3, e0, e2);

        e0 += e1 + e3 - s4;

        return renorm5(c0, c1, c2, c3, e0);
    }

    // double-double dispatch
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_dd(const f256_s& a, dd_scalar b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            add_checked_inline(a, f256_s{ b.hi, b.lo, 0.0, 0.0 }),
            detail::_f256_runtime::add_dd(a, b)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_dd(const f256_s& a, dd_scalar b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_checked_inline(a, f256_s{ b.hi, b.lo, 0.0, 0.0 }),
            detail::_f256_runtime::sub_dd(a, b)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_dd(dd_scalar a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_checked_inline(f256_s{ a.hi, a.lo, 0.0, 0.0 }, b),
            detail::_f256_runtime::sub_dd(a, b)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_dd(const f256_s& a, dd_scalar b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_checked_inline(a, f256_s{ b.hi, b.lo, 0.0, 0.0 }),
            detail::_f256_runtime::mul_dd(a, b)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_dd(const f256_s& a, dd_scalar b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            div_checked_inline(a, f256_s{ b.hi, b.lo, 0.0, 0.0 }),
            detail::_f256_runtime::div_dd(a, b)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_dd(dd_scalar a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            div_checked_inline(f256_s{ a.hi, a.lo, 0.0, 0.0 }, b),
            detail::_f256_runtime::div_dd(a, b)
        );
    }

} // namespace detail::_f256

// reciprocal helpers
[[nodiscard]] BL_FORCE_INLINE constexpr f256 recip(f256_s b) noexcept
{
    using namespace detail::_f256;

    constexpr f256_s one = f256_s{ 1.0 };

    const double inv_b0 = 1.0 / b.x0;
    const double q0     = inv_b0;
    f256_s r = sub_mul_scalar_exact(one, b, q0);

    const double q1 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q1);
    const double q2 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q2);
    const double q3 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q3);
    const double q4 = r.x0 * inv_b0;

    return renorm5(q0, q1, q2, q3, q4);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 inv(const f256_s& a) { return recip(a); }

} // namespace bl

#endif
