/**
 * fltx/detail/f256_rounding.h - f256 rounding implementation.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_F256_ROUNDING_INCLUDED
#define FLTX_DETAIL_F256_ROUNDING_INCLUDED
#include "fltx/f256_arithmetic.h"
#include "fltx/f256_classification.h"

namespace bl {

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_NO_INLINE f256_s floor(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s ceil(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s trunc(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s pow10_256(int k);

} // namespace detail::_f256_runtime

namespace detail::_f256_impl
{
    [[nodiscard]] inline constexpr f256_s floor(const f256_s& a);
    [[nodiscard]] inline constexpr f256_s ceil(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s trunc(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s pow10_256(int k);

} // namespace detail::_f256_impl

[[nodiscard]] inline constexpr f256_s detail::_f256_impl::floor(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_x0_threshold = detail::fp::double_integer_threshold;

    if (absd(a.x0) >= integer_x0_threshold)
    {
        if (a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256_s{ a.x0, 0.0, 0.0, 0.0 };

        return f256_s{ a.x0, 0.0, 0.0, 0.0 } + detail::_f256_impl::floor(f256_s{ a.x1, a.x2, a.x3, 0.0 });
    }

    f256_s r{ detail::fp::floor(a.x0), 0.0, 0.0, 0.0 };
    if (r > a)
        r -= 1.0;
    return r;
}

[[nodiscard]] inline constexpr f256_s detail::_f256_impl::ceil(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_x0_threshold = detail::fp::double_integer_threshold;

    if (absd(a.x0) >= integer_x0_threshold)
    {
        if (a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256_s{ a.x0, 0.0, 0.0, 0.0 };

        return f256_s{ a.x0, 0.0, 0.0, 0.0 } + detail::_f256_impl::ceil(f256_s{ a.x1, a.x2, a.x3, 0.0 });
    }

    f256_s r{ detail::fp::ceil(a.x0), 0.0, 0.0, 0.0 };
    if (r < a)
        r += 1.0;
    return r;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::trunc(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    return (a.x0 < 0.0) ? detail::_f256_impl::ceil(a) : detail::_f256_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::pow10_256(int k)
{
    if (k == 0) [[unlikely]]
        return f256_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    if (n <= 16) {
        f256_s r = f256_s{ 1.0 };
        const f256_s ten = f256_s{ 10.0, 0.0, 0.0, 0.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f256_s{ 1.0 } / r);
    }

    f256_s r = f256_s{ 1.0 };
    f256_s base = f256_s{ 10.0, 0.0, 0.0, 0.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f256_s{ 1.0 } / r);
}

} // namespace bl

#endif
