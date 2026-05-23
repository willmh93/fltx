/**
 * fltx/detail/f128_math_gamma.h - gamma implementation details.
 *
 * f128 gamma/lgamma recurrences, local series, and asymptotic paths.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_GAMMA_INCLUDED
#define FLTX_F128_DETAIL_GAMMA_INCLUDED
#include "fltx/detail/f128_math_basic.h"
#include "fltx/detail/f128_math_trig.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    // near-one series
    BL_MSVC_NOINLINE constexpr f128_s lgamma1p_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        const f128_s p = horner_reverse(lgamma1p_coeff, static_cast<std::size_t>(count), y);

        return mul_inline(y, mul_add_inline(y, p, -std::numbers::egamma_v<f128_s>));
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma1p5_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p5_coeff) / sizeof(lgamma1p5_coeff[0]));

        const f128_s p = horner_reverse(lgamma1p5_coeff, static_cast<std::size_t>(count), y);

        const f128_s constant = sub_inline(half_log_two_pi, mul_inline(f128_s{ 1.5 }, std::numbers::ln2_v<f128_s>));
        const f128_s linear   = sub_inline(sub_inline(f128_s{ 2.0 }, std::numbers::egamma_v<f128_s>), mul_inline(f128_s{ 2.0 }, std::numbers::ln2_v<f128_s>));
        return mul_add_inline(y, mul_add_inline(y, p, linear), constant);
    }

    BL_MSVC_NOINLINE constexpr bool try_lgamma_near_one_or_two(const f128_s& x, f128_s& out) noexcept
    {
        const f128_s y1 = sub_inline(x, f128_s{ 1.0 });
        if (mag(y1) <= f128_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f128_s y15 = sub_inline(x, f128_s{ 1.5 });
        if (mag(y15) <= f128_s{ 0.25 })
        {
            out = lgamma1p5_series(y15);
            return true;
        }

        const f128_s y2 = sub_inline(x, f128_s{ 2.0 });
        if (mag(y2) <= f128_s{ 0.25 })
        {
            out = add_inline(f128_log1p_series_reduced(y2), lgamma1p_series(y2));
            return true;
        }

        return false;
    }

    // asymptotic helpers
    BL_MSVC_NOINLINE constexpr f128_s lgamma_stirling_asymptotic(const f128_s& z) noexcept
    {
        const f128_s inv    = div_inline(f128_s{ 1.0 }, z);
        const f128_s inv2   = mul_inline(inv, inv);
        const f128_s series = mul_inline(inv, horner_reverse(
            lgamma_stirling_coeffs,
            sizeof(lgamma_stirling_coeffs) / sizeof(lgamma_stirling_coeffs[0]),
            inv2));

        return add_inline(add_inline(sub_inline(mul_inline(sub_inline(z, f128_s{ 0.5 }), detail::_f128_impl::log(z)), z), half_log_two_pi), series);
    }

    BL_MSVC_NOINLINE constexpr void positive_recurrence_product(const f128_s& x, const f128_s& asymptotic_min, f128_s& z, f128_s& product, int& product_scale2) noexcept
    {
        z = x;
        product = f128_s{ 1.0 };
        product_scale2 = 0;

        while (z < asymptotic_min)
        {
            product = mul_inline(product, z);

            const double hi = product.hi;
            if (hi != 0.0)
            {
                const int exponent = frexp_exponent(hi);
                if (exponent > 512 || exponent < -512)
                {
                    product = detail::_f128_impl::ldexp(product, -exponent);
                    product_scale2 += exponent;
                }
            }

            z = add_inline(z, f128_s{ 1.0 });
        }
    }

    // positive range helpers
    BL_MSVC_NOINLINE constexpr f128_s lgamma_positive_low_range(const f128_s& x) noexcept
    {
        f128_s y = x;
        f128_s product{ 1.0 };
        bool shifted_up = false;

        if (y < f128_s{ 0.75 })
        {
            shifted_up = true;
            do
            {
                product = mul_inline(product, y);
                y = add_inline(y, f128_s{ 1.0 });
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y = sub_inline(y, f128_s{ 1.0 });
                product = mul_inline(product, y);
            }
        }

        f128_s local{};
        try_lgamma_near_one_or_two(y, local);

        if (product == f128_s{ 1.0 })
            return local;

        const f128_s correction = detail::_f128_impl::log(product);
        return shifted_up ? sub_inline(local, correction) : add_inline(local, correction);
    }

    BL_MSVC_NOINLINE constexpr f128_s gamma_positive_low_range(const f128_s& x) noexcept
    {
        f128_s y = x;
        f128_s product{ 1.0 };
        bool shifted_up = false;

        if (y < f128_s{ 0.75 })
        {
            shifted_up = true;
            do
            {
                product = mul_inline(product, y);
                y = add_inline(y, f128_s{ 1.0 });
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y = sub_inline(y, f128_s{ 1.0 });
                product = mul_inline(product, y);
            }
        }

        f128_s local_lgamma{};
        try_lgamma_near_one_or_two(y, local_lgamma);
        const f128_s local_gamma = detail::_f128_impl::exp(local_lgamma);
        return shifted_up ? div_inline(local_gamma, product) : mul_inline(local_gamma, product);
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma_positive_recurrence(const f128_s& x) noexcept
    {
        f128_s near_value{};
        if (try_lgamma_near_one_or_two(x, near_value))
            return near_value;

        if (x <= f128_s{ 16.0 })
            return lgamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z{};
        f128_s product{};
        int product_scale2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_scale2);

        return sub_inline(
            sub_inline(lgamma_stirling_asymptotic(z), detail::_f128_impl::log(product)),
            mul_inline(f128_s{ static_cast<double>(product_scale2) }, std::numbers::ln2_v<f128_s>));
    }

    BL_MSVC_NOINLINE constexpr f128_s gamma_positive_recurrence(const f128_s& x) noexcept
    {
        f128_s near_lgamma{};
        if (try_lgamma_near_one_or_two(x, near_lgamma))
            return detail::_f128_impl::exp(near_lgamma);

        if (x <= f128_s{ 16.0 })
            return gamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z{};
        f128_s product{};
        int product_scale2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_scale2);

        f128_s out = div_inline(detail::_f128_impl::exp(lgamma_stirling_asymptotic(z)), product);
        if (product_scale2 != 0)
            out = detail::_f128_impl::ldexp(out, -product_scale2);

        return out;
    }

} // namespace detail::_f128

// gamma functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::lgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(lgamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = detail::_f128_impl::sin(mul_inline(std::numbers::pi_v<f128_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        sub_inline(
            sub_inline(detail::_f128_impl::log(std::numbers::pi_v<f128_s>), detail::_f128_impl::log(detail::_f128::mag(sinpix))),
            lgamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x)));

    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::tgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(gamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = detail::_f128_impl::sin(mul_inline(std::numbers::pi_v<f128_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = div_inline(std::numbers::pi_v<f128_s>, mul_inline(sinpix, gamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x))));
    return F128_CANONICALIZE_MATH_RESULT(out);
}

} // namespace bl

#endif
