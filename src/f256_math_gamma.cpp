/**
 * fltx/f256_math_gamma.cpp - Runtime f256 gamma helpers and functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256_math_gamma.h"

namespace bl::detail::_f256_runtime
{
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
