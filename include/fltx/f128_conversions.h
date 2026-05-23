/**
 * fltx/f128_conversions.h - Public f128 scalar conversion helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_CONVERSIONS_INCLUDED
#define F128_CONVERSIONS_INCLUDED
#include "fltx/detail/f128_conversions.h"

namespace bl {

BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::assign(*this, u),
        detail::_f128_runtime::assign(*this, u)
    );
}

BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::assign(*this, v),
        detail::_f128_runtime::assign(*this, v)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(double x) noexcept
{
    return f128_s{ x, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(float x) noexcept
{
    return f128_s{ (double)x, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(int32_t v) noexcept
{
    return f128_s{ (double)v, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(uint32_t v) noexcept
{
    return f128_s{ (double)v, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::to_f128(u),
        detail::_f128_runtime::to_f128(u)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::to_f128(v),
        detail::_f128_runtime::to_f128(v)
    );
}

} // namespace bl

#endif
