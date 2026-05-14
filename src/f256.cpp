#include "f256.h"

namespace bl::detail::_f256_runtime
{
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

    // Core arithmetic bodies

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

    // Fused expression bodies

    f256_s sqr(const f256_s& a) noexcept
    {
        return detail::_f256::sqr_inline(a);
    }

    f256_s mul_pow2_or_double(const f256_s& a, double b) noexcept
    {
        return detail::_f256::mul_pow2_or_double_inline(a, b);
    }

    f256_s sqr_add(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b);
    }

    f256_s sqr_sub(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b);
    }

    f256_s value_sub_sqr(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_raw5_value_inline(detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), a);
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

    f256_s sqr_add_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b), c);
    }

    f256_s sqr_add_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), b), c);
    }

    f256_s sqr_sub_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b), c);
    }

    f256_s sqr_sub_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::sub_inline(detail::_f256::add_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), -b), c);
    }

    f256_s mul_add_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::mul_add_mul_inline(a, b, c, d);
    }

    f256_s mul_sub_mul(const f256_s& a, const f256_s& b, const f256_s& c, const f256_s& d) noexcept
    {
        return detail::_f256::mul_sub_mul_inline(a, b, c, d);
    }

    f256_s sqr_add_sqr(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_raw5_raw5_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b));
    }

    f256_s sqr_sub_sqr(const f256_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_raw5_raw5_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)));
    }

    f256_s sqr_twice(const f256_s& a) noexcept
    {
        const auto square = detail::_f256::sqr_raw5_inline(a);
        return detail::_f256::add_raw5_raw5_inline(square, square);
    }

    f256_s mul_twice(const f256_s& a, const f256_s& b) noexcept
    {
        const auto product = detail::_f256::mul_raw5_inline(a, b);
        return detail::_f256::add_raw5_raw5_inline(product, product);
    }

    f256_s sqr_twice_add(const f256_s& a, const f256_s& b) noexcept
    {
        const auto square = detail::_f256::sqr_raw5_inline(a);
        return detail::_f256::add_raw5_raw5_value_inline(square, square, b);
    }

    f256_s sqr_twice_sub(const f256_s& a, const f256_s& b) noexcept
    {
        const auto square = detail::_f256::sqr_raw5_inline(a);
        return detail::_f256::add_raw5_raw5_value_inline(square, square, -b);
    }

    f256_s value_sub_sqr_twice(const f256_s& a, const f256_s& b) noexcept
    {
        const auto neg_square = detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b));
        return detail::_f256::add_raw5_raw5_value_inline(neg_square, neg_square, a);
    }

    f256_s mul_twice_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        const auto product = detail::_f256::mul_raw5_inline(a, b);
        return detail::_f256::add_raw5_raw5_value_inline(product, product, c);
    }

    f256_s mul_twice_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        const auto product = detail::_f256::mul_raw5_inline(a, b);
        return detail::_f256::add_raw5_raw5_value_inline(product, product, -c);
    }

    f256_s value_sub_mul_twice(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        const auto neg_product = detail::_f256::neg_raw5(detail::_f256::mul_raw5_inline(b, c));
        return detail::_f256::add_raw5_raw5_value_inline(neg_product, neg_product, a);
    }

    f256_s sqr_add_sqr_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b), c);
    }

    f256_s sqr_add_sqr_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::sqr_raw5_inline(b), -c);
    }

    f256_s sqr_sub_sqr_add(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), c);
    }

    f256_s sqr_sub_sqr_sub(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::add_raw5_raw5_value_inline(detail::_f256::sqr_raw5_inline(a), detail::_f256::neg_raw5(detail::_f256::sqr_raw5_inline(b)), -c);
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
        double head{}, carry{};
        detail::fp::two_sum_precise(base_denominator.x0, scalar, head, carry);

        const double head_abs = head < 0.0 ? -head : head;
        const double base_abs = base_denominator.x0 < 0.0 ? -base_denominator.x0 : base_denominator.x0;
        if (carry == 0.0 && head_abs >= base_abs)
            return detail::_f256::div_inline(numerator, f256_s{ head, base_denominator.x1, base_denominator.x2, base_denominator.x3 });

        return detail::_f256::div_inline(numerator, detail::_f256::add_double_inline(base_denominator, scalar));
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
}
