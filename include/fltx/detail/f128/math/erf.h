/**
 * fltx/detail/f128/math/erf.h - erf implementation details.
 *
 * f128 erf/erfc series and continued-fraction implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_ERF_INCLUDED
#define FLTX_F128_DETAIL_ERF_INCLUDED
#include "fltx/detail/f128/math/exp_log.h"

namespace bl {

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr f128_s erf_positive_series(const f128_s& x)
    {
        const f128_s xx = mul_inline(x, x);
        f128_s power = x;
        f128_s sum   = x;

        for (int n = 1; n < 256; ++n)
        {
            power = mul_inline(power, div_inline(-xx, f128_s{ static_cast<double>(n) }));
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * n + 1) });
            sum = add_inline(sum, term);
            if (abs(term) < f128_s::eps())
                break;
        }

        return canonicalize_math_result(mul_inline(mul_inline(f128_s{ 2.0 }, std::numbers::inv_sqrtpi_v<f128_s>), sum));
    }

    BL_FORCE_INLINE constexpr f128_s erfc_positive_cf(const f128_s& x)
    {
        const f128_s z = mul_inline(x, x);
        constexpr f128_s a = f128_s{ 0.5 };
        constexpr f128_s tiny = f128_s{ 1.0e-300 };

        f128_s b = sub_inline(add_inline(z, f128_s{ 1.0 }), a);
        f128_s c = div_inline(f128_s{ 1.0 }, tiny);
        f128_s d = div_inline(f128_s{ 1.0 }, b);
        f128_s h = d;

        for (int i = 1; i <= 96; ++i)
        {
            const f128_s ii = f128_s{ static_cast<double>(i) };
            const f128_s an = -mul_inline(ii, sub_inline(ii, a));

            b = add_inline(b, f128_s{ 2.0 });

            d = mul_add_inline(an, d, b);
            if (abs(d) < tiny)
                d = tiny;

            c = add_inline(b, div_inline(an, c));
            if (abs(c) < tiny)
                c = tiny;

            d = div_inline(f128_s{ 1.0 }, d);
            const f128_s delta = mul_inline(d, c);
            h = mul_inline(h, delta);

            if (abs(sub_inline(delta, f128_s{ 1.0 })) <= mul_inline(f128_s{ 32.0 }, f128_s::eps()))
                break;
        }

        const f128_s out = mul_inline(mul_inline(mul_inline(detail::_f128_constexpr::exp(-z), x), std::numbers::inv_sqrtpi_v<f128_s>), h);
        return canonicalize_math_result(out);
    }

} // namespace detail::_f128

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::erf(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s out = ax < f128_s{ 2.0 }
        ? erf_positive_series(ax)
        : (ax > f128_s{ 27.0 } ? f128_s{ 1.0 } : sub_inline(f128_s{ 1.0 }, erfc_positive_cf(ax)));

    if (neg)
        out = -out;

    return canonicalize_math_result(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::erfc(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return f128_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f128_s{ 2.0 } : f128_s{ 0.0 };

    if (signbit(x))
    {
        const f128_s ax = -x;
        if (ax < f128_s{ 2.0 })
            return canonicalize_math_result(add_inline(f128_s{ 1.0 }, erf_positive_series(ax)));
        if (ax > f128_s{ 27.0 })
            return f128_s{ 2.0 };
        return canonicalize_math_result(sub_inline(f128_s{ 2.0 }, erfc_positive_cf(ax)));
    }

    // use the existing high-quality erf series throughout the region where it is stable
    if (x < f128_s{ 2.0 })
        return canonicalize_math_result(sub_inline(f128_s{ 1.0 }, erf_positive_series(x)));

    if (x > f128_s{ 27.0 })
        return f128_s{ 0.0 };

    return canonicalize_math_result(erfc_positive_cf(x));
}

} // namespace bl

#endif
