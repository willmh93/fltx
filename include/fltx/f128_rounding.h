/**
 * fltx/f128_rounding.h - Public f128 rounding wrappers and pow10 support.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_ROUNDING_INCLUDED
#define FLTX_F128_ROUNDING_INCLUDED
#include "fltx/detail/f128_rounding.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128 floor(const f128_s& a)
{
    return detail::_f128_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 ceil(const f128_s& a)
{
    return detail::_f128_impl::ceil(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 trunc(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::trunc(a),
        detail::_f128_runtime::trunc(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 pow10_128(int k)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_impl::pow10_128(k),
        detail::_f128_runtime::pow10_128(k)
    );
}

} // namespace bl

#endif
