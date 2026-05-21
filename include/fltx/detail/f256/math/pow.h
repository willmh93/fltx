/**
 * fltx/detail/f256/math/pow.h - pow implementation details.
 *
 * f256 power helpers and constexpr pow overloads.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_POW_INCLUDED
#define FLTX_F256_DETAIL_POW_INCLUDED
#include "fltx/detail/f256/math/exp_log.h"

namespace bl {

namespace detail::_f256
{
    BL_FORCE_INLINE constexpr f256_s powi(f256_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

    BL_FORCE_INLINE constexpr f256_s polish_eighth_root(const f256_s& x, const f256_s& y)
    {
        if (iszero(y))
            return y;

        const f256_s y2 = sqr_inline(y);
        const f256_s y4 = sqr_inline(y2);
        const f256_s y7 = mul_inline(mul_inline(y4, y2), y);
        const f256_s y8 = sqr_inline(y4);
        const f256_s correction = div_double_inline(div_inline(sub_inline(x, y8), y7), 8.0);

        return add_inline(y, correction);
    }

    BL_FORCE_INLINE constexpr f256_s pow_positive_eighth_fraction(const f256_s& x, int numerator)
    {
        const f256_s r2 = detail::_f256_constexpr::sqrt(x);
        if (numerator == 4)
            return r2;

        const f256_s r4 = detail::_f256_constexpr::sqrt(r2);
        if (numerator == 2)
            return r4;

        f256_s out{ 1.0 };
        if ((numerator & 4) != 0)
            out = mul_inline(out, r2);
        if ((numerator & 2) != 0)
            out = mul_inline(out, r4);
        if ((numerator & 1) != 0)
        {
            const f256_s r8 = polish_eighth_root(x, detail::_f256_constexpr::sqrt(r4));
            if (numerator == 1)
                return r8;
            out = mul_inline(out, r8);
        }
        return out;
    }

    BL_FORCE_INLINE constexpr bool pow_dyadic_eighth_exponent_in_range(int64_t n) noexcept
    {
        if (n == std::numeric_limits<int64_t>::min())
            return false;

        const bool neg = n < 0;
        const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
        return magnitude <= 1024;
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f256_s& x, const f256_s& y, int64_t& n)
    {
        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
            return false;

        if (!try_get_int64(mul_double_inline(y, 8.0), n))
            return false;

        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f256_s& x, double y, int64_t& n) noexcept
    {
        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
            return false;

        const double scaled = y * 8.0;
        if (!isfinite(scaled) || absd(scaled) >= 0x1p63)
            return false;

        const double rounded = trunc(scaled);
        if (rounded != scaled)
            return false;

        n = static_cast<int64_t>(rounded);
        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_NO_INLINE constexpr f256_s pow_dyadic_eighth_unchecked(const f256_s& x, int64_t n)
    {
        if (n == 0)
            return f256_s{ 1.0 };

        const bool neg = n < 0;
        const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
        const uint64_t whole     = magnitude / 8u;
        const int rem = static_cast<int>(magnitude & 7u);

        f256_s result = (whole == 0u) ? f256_s{ 1.0 } : powi(x, static_cast<int64_t>(whole));
        if (rem != 0)
            result = mul_inline(result, pow_positive_eighth_fraction(x, rem));
        if (neg)
            result = recip(result);

        return result;
    }

} // namespace detail::_f256

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::pow(const f256_s& x, const f256_s& y)
{
    if (iszero(y))
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s yi = detail::_f256_constexpr::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && try_get_int64(yi, yi64))
        return powi(x, yi64);

    int64_t dyadic_exponent{};
    if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
        return canonicalize_math_result(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = exp_for_pow(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return canonicalize_math_result(exp_for_pow(mul_inline(y, _log(x))));
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::pow(const f256_s& x, double y)
{
    if (y == 0.0)
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (y == 1.0) return x;
    if (y == 2.0) return canonicalize_math_result(sqr_inline(x));
    if (y == -1.0) return canonicalize_math_result(f256_s{ 1.0 } / x);
    if (y == 0.5) return canonicalize_math_result(detail::_f256_constexpr::sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? detail::fp::ceil(y)
            : detail::fp::floor(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && absd(yi) < 0x1p63)
        return powi(x, static_cast<int64_t>(yi));

    int64_t dyadic_exponent{};
    if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
        return canonicalize_math_result(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = exp_for_pow(mul_double_inline(_log(-x), y));
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return canonicalize_math_result(exp_for_pow(mul_double_inline(_log(x), y)));
}

} // namespace bl

#endif
