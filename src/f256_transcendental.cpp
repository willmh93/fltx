/**
 * fltx/f256_transcendental.cpp - Runtime f256 transcendental math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256_math_transcendental.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::mul_add_horner_step_inline(a, b, c);
    }

    BL_NO_INLINE f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f256_s{};
            right_out = f256_s{};
            return;
        }

        f256_s left  = left_coeffs[0];
        f256_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            left = detail::_f256::mul_add_inline(left, x, left_coeffs[i]);
            right = detail::_f256::mul_add_inline(right, x, right_coeffs[i]);
        }

        left_out = left;
        right_out = right;
    }

    BL_NO_INLINE f256_s cheb_eval(const f256_s& x, const f256_s* coeffs, std::size_t count, double shift) noexcept
    {
        if (count == 0)
            return {};

        const f256_s t = detail::_f256::sub_inline(
            detail::_f256::mul_double_inline(x, 2.0),
            f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (std::size_t i = count - 1; i >= 1; --i)
        {
            const f256_s b0 = detail::_f256::add_inline(
                detail::_f256::mul_double_sub_inline(detail::_f256::mul_inline(t, b1), 2.0, b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return detail::_f256::add_inline(detail::_f256::mul_sub_inline(t, b1, b2), coeffs[0]);
    }

    BL_NO_INLINE f256_s log1p_series_reduced(const f256_s& x) noexcept
    {
        const f256_s z = detail::_f256::div_add_double_inline(x, x, 2.0);
        const f256_s z2 = detail::_f256::sqr_inline(z);

        f256_s term = z;
        f256_s sum  = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term = detail::_f256::mul_inline(term, z2);
            const f256_s add = detail::_f256::div_double_inline(term, static_cast<double>(k));
            sum = detail::_f256::add_inline(sum, add);

            const f256_s asum  = detail::_f256::mag(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (detail::_f256::mag(add) <= detail::_f256::mul_inline(f256_s::eps(), scale))
                break;
        }

        return detail::_f256::add_inline(sum, sum);
    }

    BL_NO_INLINE f256_s exp(const f256_s& x) { return detail::_f256_impl::exp(x); }
    BL_NO_INLINE f256_s exp2(const f256_s& x) { return detail::_f256_impl::exp2(x); }
    BL_NO_INLINE f256_s expm1(const f256_s& x) { return detail::_f256_impl::expm1(x); }

    BL_NO_INLINE f256_s log(const f256_s& a) { return detail::_f256_impl::log(a); }
    BL_NO_INLINE f256_s log2(const f256_s& a) { return detail::_f256_impl::log2(a); }
    BL_NO_INLINE f256_s log10(const f256_s& a) { return detail::_f256_impl::log10(a); }
    BL_NO_INLINE f256_s log1p(const f256_s& x) { return detail::_f256_impl::log1p(x); }

    BL_NO_INLINE f256_s cbrt(const f256_s& x) { return detail::_f256_impl::cbrt(x); }
    BL_NO_INLINE f256_s sinh(const f256_s& x) { return detail::_f256_impl::sinh(x); }
    BL_NO_INLINE f256_s cosh(const f256_s& x) { return detail::_f256_impl::cosh(x); }
    BL_NO_INLINE f256_s tanh(const f256_s& x) { return detail::_f256_impl::tanh(x); }
    BL_NO_INLINE f256_s asinh(const f256_s& x) { return detail::_f256_impl::asinh(x); }
    BL_NO_INLINE f256_s acosh(const f256_s& x) { return detail::_f256_impl::acosh(x); }
    BL_NO_INLINE f256_s atanh(const f256_s& x) { return detail::_f256_impl::atanh(x); }

    BL_NO_INLINE f256_s pow10_256(int k) { return detail::_f256_impl::pow10_256(k); }

    namespace
    {
        using namespace detail::_f256;

        BL_NO_INLINE f256_s pow_positive_eighth_fraction_runtime(const f256_s& x, int numerator)
        {
            const f256_s r2 = detail::_f256_runtime::sqrt(x);
            if (numerator == 4)
                return r2;

            const f256_s r4 = detail::_f256_runtime::sqrt(r2);
            if (numerator == 2)
                return r4;

            f256_s out{ 1.0 };
            if ((numerator & 4) != 0)
                out = mul_inline(out, r2);
            if ((numerator & 2) != 0)
                out = mul_inline(out, r4);
            if ((numerator & 1) != 0)
            {
                const f256_s r8 = polish_eighth_root(x, detail::_f256_runtime::sqrt(r4));
                if (numerator == 1)
                    return r8;
                out = mul_inline(out, r8);
            }
            return out;
        }

        BL_NO_INLINE f256_s pow_dyadic_eighth_runtime(const f256_s& x, int64_t n)
        {
            if (n == 0)
                return f256_s{ 1.0 };

            const bool neg = n < 0;
            const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
            const uint64_t whole     = magnitude / 8u;
            const int rem = static_cast<int>(magnitude & 7u);

            f256_s result = (whole == 0u) ? f256_s{ 1.0 } : powi(x, static_cast<int64_t>(whole));
            if (rem != 0)
                result = mul_inline(result, pow_positive_eighth_fraction_runtime(x, rem));
            if (neg)
                result = recip(result);

            return result;
        }

        BL_FORCE_INLINE f256_s exp_for_pow_runtime(const f256_s& x) noexcept
        {
            return detail::_f256_runtime::exp(x);
        }

    } // namespace

    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y)
    {
        using namespace detail::_f256;

        if (iszero(y))
            return f256_s{ 1.0 };

        if (isnan(x) || isnan(y))
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s yi = detail::_f256_runtime::trunc(y);
        const bool y_is_int = (yi == y);

        int64_t yi64{};
        if (y_is_int && try_get_int64(yi, yi64))
            return powi(x, yi64);

        int64_t dyadic_exponent{};
        if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
            return F256_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_runtime(x, dyadic_exponent));

        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
        {
            if (!y_is_int)
                return std::numeric_limits<f256_s>::quiet_NaN();

            const f256_s magnitude = exp_for_pow_runtime(mul_inline(y, detail::_f256_runtime::log(-x)));
            return is_odd_integer(yi) ? -magnitude : magnitude;
        }

        return F256_CANONICALIZE_MATH_RESULT(exp_for_pow_runtime(mul_inline(y, detail::_f256_runtime::log(x))));
    }

    BL_NO_INLINE f256_s pow(const f256_s& x, double y)
    {
        using namespace detail::_f256;

        if (y == 0.0)
            return f256_s{ 1.0 };

        if (isnan(x) || isnan(y))
            return std::numeric_limits<f256_s>::quiet_NaN();

        if (y == 1.0) return x;
        if (y == 2.0) return F256_CANONICALIZE_MATH_RESULT(sqr_inline(x));
        if (y == -1.0) return F256_CANONICALIZE_MATH_RESULT(f256_s{ 1.0 } / x);
        if (y == 0.5) return F256_CANONICALIZE_MATH_RESULT(detail::_f256_runtime::sqrt(x));

        const double yi = std::trunc(y);
        const bool y_is_int = (yi == y);

        if (y_is_int && absd(yi) < 0x1p63)
            return powi(x, static_cast<int64_t>(yi));

        int64_t dyadic_exponent{};
        if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
            return F256_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_runtime(x, dyadic_exponent));

        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
        {
            if (!y_is_int)
                return std::numeric_limits<f256_s>::quiet_NaN();

            const f256_s magnitude = exp_for_pow_runtime(mul_double_inline(detail::_f256_runtime::log(-x), y));
            const bool y_is_odd =
                (absd(yi) < 0x1p53) &&
                ((static_cast<int64_t>(yi) & 1ll) != 0);

            return F256_CANONICALIZE_MATH_RESULT(y_is_odd ? -magnitude : magnitude);
        }

        return F256_CANONICALIZE_MATH_RESULT(exp_for_pow_runtime(mul_double_inline(detail::_f256_runtime::log(x), y)));
    }

    BL_NO_INLINE bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out) { return detail::_f256_impl::sincos(x, s_out, c_out); }
    BL_NO_INLINE f256_s sin(const f256_s& x) { return detail::_f256_impl::sin(x); }
    BL_NO_INLINE f256_s cos(const f256_s& x) { return detail::_f256_impl::cos(x); }
    BL_NO_INLINE f256_s tan(const f256_s& x) { return detail::_f256_impl::tan(x); }
    BL_NO_INLINE f256_s atan(const f256_s& x) { return detail::_f256_impl::atan(x); }
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x) { return detail::_f256_impl::atan2(y, x); }
    BL_NO_INLINE f256_s asin(const f256_s& x) { return detail::_f256_impl::asin(x); }
    BL_NO_INLINE f256_s acos(const f256_s& x) { return detail::_f256_impl::acos(x); }

    namespace
    {
        using namespace detail::_f256;

        BL_NO_INLINE f256_s lgamma1p_series_runtime(const f256_s& y) noexcept
        {
            constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));
            const f256_s p = detail::_f256_runtime::horner_reverse(
                lgamma1p_coeff,
                static_cast<std::size_t>(count),
                y);

            return mul_inline(y, mul_add_inline(y, p, -std::numbers::egamma_v<f256_s>));
        }

        BL_NO_INLINE bool try_lgamma_near_one_or_two_runtime(const f256_s& x, f256_s& out) noexcept
        {
            const f256_s y1 = sub_double_inline(x, 1.0);
            if (mag(y1) <= f256_s{ 0.25 })
            {
                out = lgamma1p_series_runtime(y1);
                return true;
            }

            const f256_s y2 = sub_double_inline(x, 2.0);
            if (mag(y2) <= f256_s{ 0.25 })
            {
                out = add_inline(detail::_f256_runtime::log1p_series_reduced(y2), lgamma1p_series_runtime(y2));
                return true;
            }

            return false;
        }

        BL_NO_INLINE bool try_lgamma_short_recurrence_runtime(const f256_s& x, f256_s& out) noexcept
        {
            if (!(x > f256_s{ 0.0 }) || !(x < f256_s{ 32.0 }))
                return false;

            f256_s z = x;
            f256_s product{ 1.0 };
            bool shifted_up = false;

            while (z < f256_s{ 1.0 })
            {
                product = mul_inline(product, z);
                z = add_double_inline(z, 1.0);
                shifted_up = true;
            }
            while (z > f256_s{ 2.25 })
            {
                z = sub_double_inline(z, 1.0);
                product = mul_inline(product, z);
            }

            f256_s near_value{};
            if (!try_lgamma_near_one_or_two_runtime(z, near_value))
                return false;

            const f256_s log_product = detail::_f256_runtime::log(product);
            out = shifted_up ? sub_inline(near_value, log_product) : add_inline(near_value, log_product);
            return true;
        }

        BL_NO_INLINE void positive_recurrence_product_runtime(
            const f256_s& x,
            const f256_s& asymptotic_min,
            f256_s& z,
            f256_s& product,
            int& product_exp2) noexcept
        {
            z = x;
            product = f256_s{ 1.0 };
            product_exp2 = 0;

            while (z < asymptotic_min)
            {
                product = mul_inline(product, z);

                const double hi = product.x0;
                if (hi != 0.0)
                {
                    const int e = frexp_exponent(hi);
                    if (e > 512 || e < -512)
                    {
                        product = detail::_f256_runtime::ldexp(product, -e);
                        product_exp2 += e;
                    }
                }

                z = add_double_inline(z, 1.0);
            }
        }

        BL_NO_INLINE f256_s lgamma_stirling_asymptotic_runtime(const f256_s& z) noexcept
        {
            const f256_s inv    = f256_s{ 1.0 } / z;
            const f256_s inv2   = sqr_eval(inv);
            const f256_s series = mul_eval(inv, detail::_f256_runtime::horner_reverse(
                lgamma_stirling_coeffs,
                sizeof(lgamma_stirling_coeffs) / sizeof(lgamma_stirling_coeffs[0]),
                inv2));

            return add_eval(
                add_eval(mul_sub_eval(sub_double_eval(z, 0.5), detail::_f256_runtime::log(z), z), half_log_two_pi),
                series);
        }

        BL_NO_INLINE f256_s lgamma_positive_recurrence_runtime(const f256_s& x) noexcept
        {
            f256_s near_value{};
            if (try_lgamma_near_one_or_two_runtime(x, near_value))
                return near_value;
            if (try_lgamma_short_recurrence_runtime(x, near_value))
                return near_value;

            constexpr f256_s asymptotic_min = f256_s{ 128.0 };

            f256_s z{};
            f256_s product{};
            int product_exp2 = 0;
            positive_recurrence_product_runtime(x, asymptotic_min, z, product, product_exp2);

            return sub_mul_double_eval(
                sub_eval(lgamma_stirling_asymptotic_runtime(z), detail::_f256_runtime::log(product)),
                std::numbers::ln2_v<f256_s>,
                static_cast<double>(product_exp2));
        }

    } // namespace

    BL_NO_INLINE f256_s erf(const f256_s& x) { return detail::_f256_impl::erf(x); }
    BL_NO_INLINE f256_s erfc(const f256_s& x) { return detail::_f256_impl::erfc(x); }

    BL_NO_INLINE f256_s lgamma(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))
            return x;
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<f256_s>::quiet_NaN()
                : std::numeric_limits<f256_s>::infinity();

        if (x > f256_s{ 0.0 })
            return F256_CANONICALIZE_MATH_RESULT(lgamma_positive_recurrence_runtime(x));

        const f256_s xi = detail::_f256_runtime::trunc(x);
        if (xi == x)
            return std::numeric_limits<f256_s>::infinity();

        const f256_s sinpix = detail::_f256_runtime::sin(mul_inline(std::numbers::pi_v<f256_s>, x));
        if (iszero(sinpix))
            return std::numeric_limits<f256_s>::infinity();

        const f256_s out =
            mul_double_eval(half_log_pi, 2.0)
            - detail::_f256_runtime::log(mag(sinpix))
            - lgamma_positive_recurrence_runtime(f256_s{ 1.0 } - x);

        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    BL_NO_INLINE f256_s tgamma(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))
            return x;
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<f256_s>::quiet_NaN()
                : std::numeric_limits<f256_s>::infinity();

        if (x > f256_s{ 0.0 })
            return F256_CANONICALIZE_MATH_RESULT(detail::_f256_runtime::exp(lgamma_positive_recurrence_runtime(x)));

        const f256_s xi = detail::_f256_runtime::trunc(x);
        if (xi == x)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s sinpix = detail::_f256_runtime::sin(mul_inline(std::numbers::pi_v<f256_s>, x));
        if (iszero(sinpix))
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s log_abs = sub_eval(
            sub_eval(mul_double_eval(half_log_pi, 2.0), detail::_f256_runtime::log(mag(sinpix))),
            lgamma_positive_recurrence_runtime(sub_double_inline(1.0, x)));
        f256_s out = detail::_f256_runtime::exp(log_abs);
        if (signbit(sinpix))
            out = -out;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

} // namespace bl::detail::_f256_runtime
