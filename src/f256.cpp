/**
 * fltx/f256.cpp - Hot f256 runtime helpers and optimized fused expression bodies.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256/expressions.h"

namespace bl::detail::_f256_runtime
{
    namespace
    {
        [[nodiscard]] BL_FORCE_INLINE f256_s div_add_double_impl(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept
        {
            double head{}, carry{};
            detail::fp::two_sum_precise(base_denominator.x0, scalar, head, carry);

            const double head_abs = head < 0.0 ? -head : head;
            const double base_abs = base_denominator.x0 < 0.0 ? -base_denominator.x0 : base_denominator.x0;
            if (carry == 0.0 && head_abs >= base_abs)
                return detail::_f256::div_inline(numerator, f256_s{ head, base_denominator.x1, base_denominator.x2, base_denominator.x3 });

            return detail::_f256::div_inline(numerator, detail::_f256::add_double_inline(base_denominator, scalar));
        }

        // Exact residual for long division by a double-double denominator.
        BL_PUSH_PRECISE
        [[nodiscard]] BL_FORCE_INLINE f256_s sub_mul_scalar_exact_dd(const f256_s& r, detail::_f256::dd_scalar b, double q) noexcept
        {
            using namespace detail::_f256;

            double p0{}, p1{}, p2 = 0.0, p3 = 0.0;
            double q0{}, q1{}, q2 = 0.0;
            double s0{}, s1{}, s2{}, s3{}, s4{};

            two_prod_precise(b.hi, q, p0, q0);
            two_prod_precise(b.lo, q, p1, q1);

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
        BL_POP_PRECISE

        [[nodiscard]] BL_FORCE_INLINE f256_s add_dd_impl(const f256_s& a, detail::_f256::dd_scalar b) noexcept
        {
            using namespace detail::_f256;

            if (b.lo == 0.0)
                return add_double_inline(a, b.hi);

            double s0{}, e0{};
            double s1{}, e1{};
            double s2 = a.x2, e2 = 0.0;
            double s3 = a.x3, e3 = 0.0;

            two_sum_precise(a.x0, b.hi, s0, e0);
            two_sum_precise(a.x1, b.lo, s1, e1);
            two_sum_precise(s1, e0, s1, e0);
            three_sum(s2, e0, e1);
            three_sum2(s3, e0, e2);

            e0 += e1 + e3;

            if (e0 == 0.0)
                return renorm4(s0, s1, s2, s3);

            return renorm5(s0, s1, s2, s3, e0);
        }

        [[nodiscard]] BL_FORCE_INLINE f256_s sub_dd_impl(const f256_s& a, detail::_f256::dd_scalar b) noexcept
        {
            using namespace detail::_f256;

            if (b.lo == 0.0)
                return sub_double_inline(a, b.hi);

            b.hi = -b.hi;
            b.lo = -b.lo;
            return add_dd_impl(a, b);
        }

        [[nodiscard]] BL_FORCE_INLINE f256_s sub_dd_impl(detail::_f256::dd_scalar a, const f256_s& b) noexcept
        {
            using namespace detail::_f256;

            if (a.lo == 0.0)
                return sub_double_inline(a.hi, b);

            double s0{}, e0{};
            double s1{}, e1{};
            double s2{}, e2{};
            double s3{}, e3{};

            two_sum_precise(a.hi, -b.x0, s0, e0);
            two_sum_precise(a.lo, -b.x1, s1, e1);
            two_sum_precise(0.0, -b.x2, s2, e2);
            two_sum_precise(0.0, -b.x3, s3, e3);
            two_sum_precise(s1, e0, s1, e0);
            three_sum(s2, e0, e1);
            three_sum2(s3, e0, e2);

            e0 += e1 + e3;

            if (e0 == 0.0)
                return renorm4(s0, s1, s2, s3);

            return renorm5(s0, s1, s2, s3, e0);
        }

        [[nodiscard]] BL_FORCE_INLINE f256_s mul_dd_impl(const f256_s& a, detail::_f256::dd_scalar b) noexcept
        {
            using namespace detail::_f256;

            if (b.lo == 0.0)
                return mul_double_inline(a, b.hi);
            if (!isfinite(a.x0) || !isfinite(b.hi) || !isfinite(b.lo))
                return mul_inline(a, f256_s{ b.hi, b.lo });

            double p0{}, p1{}, p2{}, p3 = 0.0, p4{}, p5{};
            double q0{}, q1{}, q2{}, q3 = 0.0, q4{}, q5{};
            double p6 = 0.0, p7 = 0.0, p8{}, p9{};
            double q6 = 0.0, q7 = 0.0, q8{}, q9{};
            double r0{}, r1{};
            double t0{}, t1{};
            double s0{}, s1{}, s2{};

            two_prod_precise(a.x0, b.hi, p0, q0);
            two_prod_precise(a.x0, b.lo, p1, q1);
            two_prod_precise(a.x1, b.hi, p2, q2);
            two_prod_precise(a.x1, b.lo, p4, q4);
            two_prod_precise(a.x2, b.hi, p5, q5);
            two_prod_precise(a.x2, b.lo, p8, q8);
            two_prod_precise(a.x3, b.hi, p9, q9);

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
            t1 += a.x3 * b.lo + q6 + q7 + q8 + q9 + s2;

            return renorm5(p0, p1, s0, t0, t1);
        }

        [[nodiscard]] BL_FORCE_INLINE f256_s div_dd_impl(const f256_s& a, detail::_f256::dd_scalar b) noexcept
        {
            using namespace detail::_f256;

            if (b.lo == 0.0)
                return div_double_inline(a, b.hi);
            if (b.hi == 0.0 || !isfinite(b.hi) || !isfinite(b.lo))
                return div_inline(a, f256_s{ b.hi, b.lo });

            const double inv_b0 = 1.0 / b.hi;

            const double q0 = a.x0 * inv_b0;
            f256_s r = sub_mul_scalar_exact_dd(a, b, q0);

            const double q1 = r.x0 * inv_b0;
            r = sub_mul_scalar_exact_dd(r, b, q1);

            const double q2 = r.x0 * inv_b0;
            r = sub_mul_scalar_exact_dd(r, b, q2);

            const double q3 = r.x0 * inv_b0;
            r = sub_mul_scalar_exact_dd(r, b, q3);

            const double q4 = r.x0 * inv_b0;

            return renorm5(q0, q1, q2, q3, q4);
        }

        [[nodiscard]] BL_FORCE_INLINE f256_s div_dd_impl(detail::_f256::dd_scalar a, const f256_s& b) noexcept
        {
            using namespace detail::_f256;

            if (a.lo == 0.0)
                return div_double_inline(a.hi, b);

            return div_inline(f256_s{ a.hi, a.lo }, b);
        }

    } // namespace

    f256_s floor(const f256_s& a)
    {
        return detail::_f256_constexpr::floor(a);
    }

    f256_s ceil(const f256_s& a)
    {
        return detail::_f256_constexpr::ceil(a);
    }

    f256_s trunc(const f256_s& a)
    {
        return detail::_f256_constexpr::trunc(a);
    }

    f256_s pow10_256(int k)
    {
        return detail::_f256_constexpr::pow10_256(k);
    }

    f256_s to_f256(uint64_t u) noexcept
    {
        return detail::_f256_constexpr::to_f256(u);
    }

    f256_s to_f256(int64_t v) noexcept
    {
        return detail::_f256_constexpr::to_f256(v);
    }

    f256_s& assign(f256_s& out, uint64_t u) noexcept
    {
        return detail::_f256_constexpr::assign(out, u);
    }

    f256_s& assign(f256_s& out, int64_t v) noexcept
    {
        return detail::_f256_constexpr::assign(out, v);
    }

    f256_s add(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_inline(a, b);
    }

    f256_s sub(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::sub_inline(a, b);
    }

    f256_s mul(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::mul_inline(a, b);
    }

    f256_s div(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::div_inline(a, b);
    }

    f256_s add_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept
    {
        return add_dd_impl(a, b);
    }

    f256_s sub_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept
    {
        return sub_dd_impl(a, b);
    }

    f256_s sub_dd(detail::_f256::dd_scalar a, const f256_s& b) noexcept
    {
        return sub_dd_impl(a, b);
    }

    f256_s mul_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept
    {
        return mul_dd_impl(a, b);
    }

    f256_s div_dd(const f256_s& a, detail::_f256::dd_scalar b) noexcept
    {
        return div_dd_impl(a, b);
    }

    f256_s div_dd(detail::_f256::dd_scalar a, const f256_s& b) noexcept
    {
        return div_dd_impl(a, b);
    }

    f256_s add_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::add_double_inline(a, b);
    }

    f256_s sub_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::sub_double_inline(a, b);
    }

    f256_s sub_double(double a, const f256_s& b) noexcept
    {
        return detail::_f256::sub_double_inline(a, b);
    }

    f256_s mul_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::mul_double_inline(a, b);
    }

    f256_s div_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::div_double_inline(a, b);
    }

    f256_s div_double(double a, const f256_s& b) noexcept
    {
        return detail::_f256::div_double_inline(a, b);
    }

    f256_s sqr(const f256_s& a) noexcept
    {
        return detail::_f256::sqr_inline(a);
    }

    f256_s mul_pow2_or_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::mul_pow2_or_double_inline(a, b);
    }

    f256_s mul_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::mul_add_inline(a, b, c);
    }

    f256_s mul_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::mul_sub_inline(a, b, c);
    }

    f256_s value_sub_mul(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::value_sub_mul_inline(a, b, c);
    }

    f256_s mul_add_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::mul_add_inline(a, b, c), d);
    }

    f256_s mul_add_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::mul_add_inline(a, b, c), d);
    }

    f256_s mul_sub_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::mul_sub_inline(a, b, c), d);
    }

    f256_s mul_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::mul_sub_inline(a, b, c), d);
    }

    f256_s mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::mul_add_mul_inline(a, b, c, d);
    }

    f256_s mul_sub_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::mul_sub_mul_inline(a, b, c, d);
    }

    f256_s mul_add_mul_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return detail::_f256::mul_add_mul_add_inline(a, b, c, d, e);
    }

    f256_s mul_add_mul_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return detail::_f256::mul_add_mul_sub_inline(a, b, c, d, e);
    }

    f256_s mul_sub_mul_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return detail::_f256::mul_sub_mul_add_inline(a, b, c, d, e);
    }

    f256_s mul_sub_mul_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e) noexcept
    {
        return detail::_f256::mul_sub_mul_sub_inline(a, b, c, d, e);
    }

    f256_s mul_add_mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::mul_inline(e, f));
    }

    f256_s mul_add_mul_add_mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& e, const f256_s& f, const f256_s& g, const f256_s& h) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::mul_add_mul_inline(e, f, g, h));
    }

    f256_s add_add_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_add_add_inline(a, b, c);
    }

    f256_s add_sub_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_sub_add_inline(a, b, c);
    }

    f256_s add_add_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_add_sub_inline(a, b, c);
    }

    f256_s add_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_sub_sub_inline(a, b, c);
    }

    f256_s add_add_add_add(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::add_add_add_inline(a, b, c), d);
    }

    f256_s add_add_add_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::add_add_add_inline(a, b, c), d);
    }

    f256_s add_add_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::add_add_sub_inline(a, b, c), d);
    }

    f256_s add_sub_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::add_sub_sub_inline(a, b, c), d);
    }

    f256_s add_scaled_2_1(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_scaled_inline<2, 1>(a, b);
    }

    f256_s add_scaled_1_2(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_scaled_inline<1, 2>(a, b);
    }

    f256_s add_scaled_2_neg1(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_scaled_inline<2, -1>(a, b);
    }

    f256_s add_scaled_1_neg2(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_scaled_inline<1, -2>(a, b);
    }

    f256_s add_mul_double(const f256_s& addend, const f256_s& value, double scalar) noexcept
    {
        return detail::_f256::add_mul_double_inline(addend, value, scalar);
    }

    f256_s sub_mul_double(const f256_s& minuend, const f256_s& value, double scalar) noexcept
    {
        return detail::_f256::sub_mul_double_inline(minuend, value, scalar);
    }

    f256_s mul_double_sub(const f256_s& value, double scalar, const f256_s& subtrahend) noexcept
    {
        return detail::_f256::mul_double_sub_inline(value, scalar, subtrahend);
    }

    f256_s mul_double_add_mul_double(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar) noexcept
    {
        return detail::_f256::add_raw5_raw5_inline(
            detail::_f256::mul_double_raw5_inline(a, a_scalar),
            detail::_f256::mul_double_raw5_inline(b, b_scalar));
    }

    f256_s mul_double_add_mul_double_add(const f256_s& a, double a_scalar, const f256_s& b, double b_scalar, const f256_s& c) noexcept
    {
        return detail::_f256::add_raw5_raw5_value_inline(
            detail::_f256::mul_double_raw5_inline(a, a_scalar),
            detail::_f256::mul_double_raw5_inline(b, b_scalar),
            c);
    }

    f256_s div_add(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::div_inline(numerator, detail::_f256::add_inline(a, b));
    }

    f256_s div_sub(const f256_s& numerator, const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::div_inline(numerator, detail::_f256::sub_inline(a, b));
    }

    f256_s div_add_double(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept
    {
        return div_add_double_impl(numerator, base_denominator, scalar);
    }

    f256_s div_double_sub(const f256_s& numerator, double scalar, const f256_s& base_denominator) noexcept
    {
        return detail::_f256::div_double_sub_inline(numerator, scalar, base_denominator);
    }

    f256_s mul_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_add_inline(a, b, c), denominator);
    }

    f256_s mul_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_sub_inline(a, b, c), denominator);
    }

    f256_s value_sub_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::value_sub_mul_inline(a, b, c), denominator);
    }

    f256_s mul_add_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), denominator);
    }

    f256_s mul_sub_mul_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_sub_mul_inline(a, b, c, d), denominator);
    }

    f256_s add_add_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_add_add_inline(a, b, c), denominator);
    }

    f256_s add_sub_add_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_sub_add_inline(a, b, c), denominator);
    }

    f256_s add_add_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_add_sub_inline(a, b, c), denominator);
    }

    f256_s add_sub_sub_div(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_sub_sub_inline(a, b, c), denominator);
    }

    f256_s add_mul_double_div(const f256_s& addend, const f256_s& value, double scalar, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_mul_double_inline(addend, value, scalar), denominator);
    }

    f256_s sub_mul_double_div(const f256_s& minuend, const f256_s& value, double scalar, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::sub_mul_double_inline(minuend, value, scalar), denominator);
    }

    f256_s mul_double_sub_div(const f256_s& value, double scalar, const f256_s& subtrahend, const f256_s& denominator) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_double_sub_inline(value, scalar, subtrahend), denominator);
    }

    f256_s mul_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s mul_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s value_sub_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::value_sub_mul_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s mul_add_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_add_mul_inline(a, b, c, d), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s mul_sub_mul_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_sub_mul_inline(a, b, c, d), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s add_add_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_add_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s add_sub_add_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_sub_add_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s add_add_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_add_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s add_sub_sub_div_add_double(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& denominator, double scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_sub_sub_inline(a, b, c), detail::_f256::add_double_inline(denominator, scalar));
    }

    f256_s add_mul_double_div_add_double(const f256_s& addend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::add_mul_double_inline(addend, value, value_scalar), detail::_f256::add_double_inline(denominator, denominator_scalar));
    }

    f256_s sub_mul_double_div_add_double(const f256_s& minuend, const f256_s& value, double value_scalar, const f256_s& denominator, double denominator_scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::sub_mul_double_inline(minuend, value, value_scalar), detail::_f256::add_double_inline(denominator, denominator_scalar));
    }

    f256_s mul_double_sub_div_add_double(const f256_s& value, double value_scalar, const f256_s& subtrahend, const f256_s& denominator, double denominator_scalar) noexcept
    {
        return detail::_f256::div_inline(detail::_f256::mul_double_sub_inline(value, value_scalar, subtrahend), detail::_f256::add_double_inline(denominator, denominator_scalar));
    }

} // namespace bl::detail::_f256_runtime
