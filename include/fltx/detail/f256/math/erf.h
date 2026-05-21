/**
 * fltx/detail/f256/math/erf.h - erf implementation details.
 *
 * f256 erf/erfc Chebyshev, series, and continued-fraction implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_ERF_IMPL_INCLUDED
#define FLTX_F256_DETAIL_ERF_IMPL_INCLUDED
#include "fltx/detail/f256/math/exp_log.h"

namespace bl {

namespace detail::_f256
{
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_cheb_eval(const f256_s& x, const f256_s* coeffs, double shift)
    {
        if (!bl::use_constexpr_math())
            return detail::_f256_runtime::cheb_eval(x, coeffs, f256_erf_cheb_coeff_count, shift);

        const f256_s t = sub_inline(mul_double_inline(x, 2.0), f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (int i = static_cast<int>(f256_erf_cheb_coeff_count) - 1; i >= 1; --i)
        {
            const f256_s b0 = add_inline(
                mul_double_sub_inline(mul_inline(t, b1), 2.0, b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return add_inline(mul_sub_inline(t, b1, b2), coeffs[0]);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_cheb(const f256_s& x)
    {
        if (x < f256_s{ 1.0 })
            return erf_cheb_eval(x, f256_erf_cheb_0_1, 1.0);
        if (x < f256_s{ 2.0 })
            return erf_cheb_eval(x, f256_erf_cheb_1_2, 3.0);
        return erf_cheb_eval(x, f256_erf_cheb_2_3, 5.0);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_series(const f256_s& x)
    {
        const f256_s xx = mul_inline(x, x);
        f256_s power = x;
        f256_s sum   = x;

        for (int n = 1; n < 512; ++n)
        {
            power = mul_inline(
                power,
                div_inline(-xx, f256_s{ static_cast<double>(n) }));

            const f256_s term = div_inline(
                power,
                f256_s{ static_cast<double>(2 * n + 1) });

            sum = add_inline(sum, term);
            if (abs(term) < f256_s::eps())
                break;
        }

        return mul_inline(
            mul_inline(f256_s{ 2.0 }, std::numbers::inv_sqrtpi_v<f256_s>),
            sum);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cheb_3_4(const f256_s& x)
    {
        return erf_cheb_eval(x, f256_erfc_cheb_3_4, 7.0);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cf(const f256_s& x)
    {
        const f256_s z = sqr_inline(x);
        constexpr f256_s a = f256_s{ 0.5 };
        constexpr f256_s tiny = f256_s{ 1.0e-300, 0.0, 0.0, 0.0 };

        f256_s b = add_double_inline(z, 0.5);
        f256_s c = f256_s{ 1.0 } / tiny;
        f256_s d = f256_s{ 1.0 } / b;
        f256_s h = d;

        for (int i = 1; i <= 160; ++i)
        {
            const f256_s ii = f256_s{ static_cast<double>(i) };
            const f256_s an = -mul_inline(ii, sub_inline(ii, a));

            b = add_double_inline(b, 2.0);

            d = mul_add_inline(an, d, b);
            if (abs(d) < tiny)
                d = tiny;

            c = add_inline(b, div_inline(an, c));
            if (abs(c) < tiny)
                c = tiny;

            d = f256_s{ 1.0 } / d;
            const f256_s delta = mul_inline(d, c);
            h = mul_inline(h, delta);

            if (abs(sub_double_inline(delta, 1.0)) <= mul_double_inline(f256_s::eps(), 64.0))
                break;
        }

        return mul_inline(
            mul_inline(_exp(-z), x),
            mul_inline(std::numbers::inv_sqrtpi_v<f256_s>, h));
    }

} // namespace detail::_f256

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::erf(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s out{};

    if (ax < f256_s{ 1.0 })
    {
        out = erf_positive_series(ax);
    }
    else if (ax < f256_s{ 3.0 })
    {
        out = erf_positive_cheb(ax);
    }
    else if (ax < f256_s{ 4.0 })
    {
        out = f256_s{ 1.0 } - erfc_positive_cheb_3_4(ax);
    }
    else
    {
        out = f256_s{ 1.0 } - erfc_positive_cf(ax);
    }

    if (neg)
        out = -out;

    return canonicalize_math_result(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::erfc(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return f256_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f256_s{ 2.0 } : f256_s{ 0.0 };

    if (signbit(x))
        return canonicalize_math_result(add_double_inline(detail::_f256_constexpr::erf(-x), 1.0));

    if (x < f256_s{ 1.0 })
        return canonicalize_math_result(sub_double_inline(1.0, erf_positive_series(x)));

    if (x < f256_s{ 3.0 })
        return canonicalize_math_result(sub_double_inline(1.0, erf_positive_cheb(x)));

    if (x < f256_s{ 4.0 })
        return canonicalize_math_result(erfc_positive_cheb_3_4(x));

    if (x > f256_s{ 40.0 })
        return f256_s{ 0.0 };

    return canonicalize_math_result(erfc_positive_cf(x));
}

} // namespace bl

#endif
