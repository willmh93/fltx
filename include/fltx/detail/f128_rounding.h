/**
 * fltx/detail/f128_rounding.h - f128 rounding implementation.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_F128_ROUNDING_INCLUDED
#define FLTX_DETAIL_F128_ROUNDING_INCLUDED
#include "fltx/f128_arithmetic.h"
#include "fltx/f128_classification.h"

namespace bl {

namespace detail::_f128_runtime
{
    [[nodiscard]] BL_NO_INLINE f128_s trunc(const f128_s& a);
    [[nodiscard]] BL_NO_INLINE f128_s pow10_128(int k);

} // namespace detail::_f128_runtime

namespace detail::_f128_impl
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double floor_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::_f128::floor(x),
            std::floor(x)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double ceil_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::_f128::ceil(x),
            std::ceil(x)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s floor(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s ceil(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s trunc(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s pow10_128(int k);

} // namespace detail::_f128_impl

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::floor(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128_impl::floor_limb(a.lo));
    }

    double hi = detail::_f128_impl::floor_limb(a.hi);
    if (hi == a.hi && a.lo < 0.0)
        hi -= 1.0;
    return f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::ceil(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128_impl::ceil_limb(a.lo));
    }

    double hi = detail::_f128_impl::ceil_limb(a.hi);
    if (hi == a.hi && a.lo > 0.0)
        hi += 1.0;
    return f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::trunc(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    return (a.hi < 0.0) ? detail::_f128_impl::ceil(a) : detail::_f128_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::pow10_128(int k)
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

} // namespace bl

#endif
