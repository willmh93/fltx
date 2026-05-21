/**
 * fltx/f128/rounding.h - core f128 rounding wrappers and pow10 support.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_CORE_ROUNDING_INCLUDED
#define FLTX_F128_CORE_ROUNDING_INCLUDED
#include "fltx/f128/arithmetic.h"
#include "fltx/f128/classification.h"

namespace bl {

namespace detail::_f128_runtime
{
    [[nodiscard]] BL_NO_INLINE f128_s floor(const f128_s& a);
    [[nodiscard]] BL_NO_INLINE f128_s ceil(const f128_s& a);
    [[nodiscard]] BL_NO_INLINE f128_s trunc(const f128_s& a);
    [[nodiscard]] BL_NO_INLINE f128_s pow10_128(int k);

} // namespace detail::_f128_runtime

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::floor(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128::floor_constexpr(a.lo));
    }

    f128_s r{ detail::_f128::floor_constexpr(a.hi), 0.0 };
    if (r > a)
        r -= 1.0;
    return r;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::ceil(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128::ceil_constexpr(a.lo));
    }

    f128_s r{ detail::_f128::ceil_constexpr(a.hi), 0.0 };
    if (r < a)
        r += 1.0;
    return r;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::trunc(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    return (a.hi < 0.0) ? detail::_f128_constexpr::ceil(a) : detail::_f128_constexpr::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::pow10_128(int k)
{
    if (k == 0) [[unlikely]]
        return f128_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    // fast small-exponent path
    if (n <= 16) {
        f128_s r = f128_s{ 1.0 };
        const f128_s ten = f128_s{ 10.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f128_s{ 1.0 } / r);
    }

    f128_s r = f128_s{ 1.0 };
    f128_s base = f128_s{ 10.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f128_s{ 1.0 } / r);
}


[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 floor(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::floor(a),
        detail::_f128_runtime::floor(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 ceil(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::ceil(a),
        detail::_f128_runtime::ceil(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 trunc(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::trunc(a),
        detail::_f128_runtime::trunc(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow10_128(int k)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::pow10_128(k),
        detail::_f128_runtime::pow10_128(k)
    );
}

} // namespace bl

#endif
