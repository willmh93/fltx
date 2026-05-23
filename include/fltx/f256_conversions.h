/**
 * fltx/f256_conversions.h - Public f256 scalar conversion helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_CONVERSIONS_INCLUDED
#define FLTX_F256_CONVERSIONS_INCLUDED
#include "fltx/detail/f256_conversions.h"

namespace bl {

BL_FORCE_INLINE constexpr f256_s& f256_s::operator=(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::assign(*this, u),
        detail::_f256_runtime::assign(*this, u)
    );
}

BL_FORCE_INLINE constexpr f256_s& f256_s::operator=(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::assign(*this, v),
        detail::_f256_runtime::assign(*this, v)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(double x) noexcept
{
    return f256_s{ x, 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(float x) noexcept
{
    return f256_s{ (double)x, 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(int32_t v) noexcept
{
    return f256_s{ (double)v, 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(uint32_t v) noexcept
{
    return f256_s{ (double)v, 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::to_f256(u),
        detail::_f256_runtime::to_f256(u)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::to_f256(v),
        detail::_f256_runtime::to_f256(v)
    );
}

} // namespace bl

#endif
