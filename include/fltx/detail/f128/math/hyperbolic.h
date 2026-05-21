/**
 * fltx/detail/f128/math/hyperbolic.h - Hyperbolic implementation details.
 *
 * f128 sinh/cosh/tanh and inverse hyperbolic implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_HYPERBOLIC_INCLUDED
#define FLTX_F128_DETAIL_HYPERBOLIC_INCLUDED
#include "fltx/detail/f128/math/exp_log.h"

namespace bl {

namespace detail::_f128
{
    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum   = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (abs(term) <= f128_s::eps())
                break;
        }

        return sum;
    }

    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series_runtime(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum   = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (abs(term) <= f128_s::eps())
                break;
        }

        return sum;
    }

} // namespace detail::_f128

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::sinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax <= f128_s{ 0.5 })
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s term = x;
        f128_s sum  = x;

        for (int n = 1; n <= 40; ++n)
        {
            const double denom = static_cast<double>((n * 2) * (n * 2 + 1));
            term = div_inline(mul_inline(term, x2), f128_s{ denom });
            sum = add_inline(sum, term);

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return canonicalize_math_result(sum);
    }

    const f128_s ex = detail::_f128_constexpr::exp(ax);
    f128_s out = mul_inline(sub_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 });
    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::cosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = abs(x);
    const f128_s ex = detail::_f128_constexpr::exp(ax);
    return canonicalize_math_result(mul_inline(add_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 }));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::tanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = abs(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = detail::_f128_constexpr::expm1(add_inline(ax, ax));
    f128_s out = div_inline(em1, add_inline(em1, f128_s{ 2.0 }));
    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::asinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = abs(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_constexpr::log(ax), std::numbers::ln2_v<f128_s>);
    else
        out = detail::_f128_constexpr::log(add_inline(ax, detail::_f128_constexpr::sqrt(add_inline(mul_inline(ax, ax), f128_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::acosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x < f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (x == f128_s{ 1.0 })
        return f128_s{ 0.0 };
    if (isinf(x))
        return x;

    f128_s out{};
    if (x > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_constexpr::log(x), std::numbers::ln2_v<f128_s>);
    else
        out = detail::_f128_constexpr::log(add_inline(x, detail::_f128_constexpr::sqrt(mul_inline(sub_inline(x, f128_s{ 1.0 }), add_inline(x, f128_s{ 1.0 })))));

    return canonicalize_math_result(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::atanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax > f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (ax == f128_s{ 1.0 })
        return signbit(x)
            ? f128_s{ -std::numeric_limits<double>::infinity(), 0.0 }
            : f128_s{  std::numeric_limits<double>::infinity(), 0.0 };

    if (ax <= f128_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
            return canonicalize_math_result(atanh_small_series(x));

        return canonicalize_math_result(atanh_small_series_runtime(x));
    }

    const f128_s out = mul_inline(detail::_f128_constexpr::log(div_inline(add_inline(f128_s{ 1.0 }, x), sub_inline(f128_s{ 1.0 }, x))), f128_s{ 0.5 });
    return canonicalize_math_result(out);
}

} // namespace bl

#endif
