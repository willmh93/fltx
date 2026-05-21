/**
 * fltx/detail/f128/math/pow.h - pow implementation details.
 *
 * f128 power helpers and constexpr pow overloads, layered over the shared sqrt/log/exp machinery.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_POW_IMPL_INCLUDED
#define FLTX_F128_DETAIL_POW_IMPL_INCLUDED
#include "fltx/detail/f128/math/exp_log.h"

namespace bl {

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr bool is_odd_integer(const f128_s& x) noexcept
    {
        int64_t value{};
        if (f128_try_get_int64(x, value))
            return (value & 1ll) != 0;

        if (x.lo != 0.0 || !isfinite(x.hi))
            return false;

        return double_integer_is_odd(x.hi);
    }

    BL_FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

} // namespace detail::_f128

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s yi = detail::_f128_constexpr::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && f128_try_get_int64(yi, yi64))
        return powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return canonicalize_math_result(_exp(mul_inline(y, _log(x))));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::pow(const f128_s& x, double y)
{
    if (y == 0.0)
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (y == 1.0)  return x;
    if (y == 2.0)  return canonicalize_math_result(x * x);
    if (y == -1.0) return canonicalize_math_result(f128_s{ 1.0 } / x);
    if (y == 0.5)  return canonicalize_math_result(detail::_f128_constexpr::sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? ceil_constexpr(y)
            : floor_constexpr(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && absd(yi) < 0x1p63)
        return powi(x, static_cast<int64_t>(yi));

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(_log(-x) * y);
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return canonicalize_math_result(_exp(_log(x) * y));
}

} // namespace bl

#endif
