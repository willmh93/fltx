/**
 * fltx/detail/f256_math_hyperbolic.h - Hyperbolic implementation details.
 *
 * f256 sinh/cosh/tanh and inverse hyperbolic implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_HYPERBOLIC_INCLUDED
#define FLTX_F256_DETAIL_HYPERBOLIC_INCLUDED
#include "fltx/detail/f256_math_exp_log.h"

namespace bl {

namespace detail::_f256 // primitives and kernels
{
    // atanh series
    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series(const f256_s& x)
    {
        const f256_s x2 = sqr_inline(x);
        f256_s sum   = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f256_s term = div_double_inline(power, static_cast<double>(2 * k + 1));
            sum = add_inline(sum, term);

            if (mag(term) <= f256_s::eps())
                break;
        }

        return sum;
    }

    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series_runtime(const f256_s& x)
    {
        const f256_s x2 = sqr_inline(x);
        f256_s sum   = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f256_s term = div_double_inline(power, static_cast<double>(2 * k + 1));
            sum = add_inline(sum, term);

            if (mag(term) <= f256_s::eps())
                break;
        }

        return sum;
    }

} // namespace detail::_f256

// sinh/cosh/tanh
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sinh(const f256_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (!bl::use_constexpr_math())
    {
        const f256_s em1 = _expm1(ax);
        f256_s out = div_inline(
            mul_add_inline(em1, em1, mul_double_inline(em1, 2.0)),
            mul_double_inline(add_scalar_precise(em1, 1.0), 2.0));
        if (signbit(x))
            out = -out;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    if (ax <= f256_s{ 0.5 })
    {
        const f256_s x2 = mul_inline(x, x);
        f256_s term = x;
        f256_s sum  = x;

        for (int n = 1; n <= 256; ++n)
        {
            term = div_double_inline(
                mul_inline(term, x2),
                static_cast<double>((2 * n) * (2 * n + 1)));
            sum = add_inline(sum, term);

            const f256_s asum  = detail::_f256::mag(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (detail::_f256::mag(term) <= mul_inline(f256_s::eps(), scale))
                break;
        }

        return F256_CANONICALIZE_MATH_RESULT(sum);
    }

    const f256_s ex     = _exp(ax);
    const f256_s inv_ex = recip(ex);
    f256_s out = mul_double_inline(sub_inline(ex, inv_ex), 0.5);
    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::cosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s ax     = detail::_f256::mag(x);
    const f256_s ex     = _exp(ax);
    const f256_s inv_ex = recip(ex);
    return F256_CANONICALIZE_MATH_RESULT(
        mul_double_inline(add_inline(ex, inv_ex), 0.5));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::tanh(const f256_s& x)
{
    using namespace detail::_f256;
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s ax = detail::_f256::mag(x);
    if (ax > f256_s{ 20.0 })
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s em1   = _expm1(mul_double_inline(ax, 2.0));
    const f256_s denom = add_scalar_precise(em1, 2.0);

    f256_s out = div_inline(em1, denom);

    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

// inverse hyperbolic functions
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::asinh(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    f256_s out{};
    if (ax > f256_s{ 0x1p500 })
        out = add_inline(detail::_f256_impl::log(ax), std::numbers::ln2_v<f256_s>);
    else
        out = detail::_f256_impl::log(add_inline(ax, detail::_f256_impl::sqrt(add_raw5_value_inline(sqr_raw5_inline(ax), f256_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::acosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x < f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (x == f256_s{ 1.0 })
        return f256_s{ 0.0 };
    if (isinf(x))
        return x;

    f256_s out{};
    if (x > f256_s{ 0x1p500 })
        out = add_inline(detail::_f256_impl::log(x), std::numbers::ln2_v<f256_s>);
    else
        out = detail::_f256_impl::log(add_inline(
            x,
            detail::_f256_impl::sqrt(mul_inline(sub_double_inline(x, 1.0), add_double_inline(x, 1.0)))));

    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::atanh(const f256_s& x)
{
    if (isnan(x) || iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (ax > f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (ax == f256_s{ 1.0 })
        return signbit(x)
        ? f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }
        : f256_s{ std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };

    if (ax <= f256_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
        {
            return F256_CANONICALIZE_MATH_RESULT(atanh_small_series(x));
        }

        return F256_CANONICALIZE_MATH_RESULT(atanh_small_series_runtime(x));
    }

    const f256_s out = mul_double_inline(
        detail::_f256_impl::log(div_inline(add_double_inline(x, 1.0), sub_double_inline(1.0, x))),
        0.5);
    return F256_CANONICALIZE_MATH_RESULT(out);
}

} // namespace bl

#endif
