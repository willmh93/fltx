/**
 * fltx/f256_rounding.h - Public f256 rounding wrappers and pow10 support.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_ROUNDING_INCLUDED
#define FLTX_F256_ROUNDING_INCLUDED
#include "fltx/detail/f256_rounding.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256 floor(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::floor(a),
        detail::_f256_runtime::floor(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 ceil(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::ceil(a),
        detail::_f256_runtime::ceil(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 trunc(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::trunc(a),
        detail::_f256_runtime::trunc(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 pow10_256(int k)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::pow10_256(k),
        detail::_f256_runtime::pow10_256(k)
    );
}

} // namespace bl

#endif
