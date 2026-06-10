/**
 * fltx/f32_math.h - constexpr <cmath>-style functions for f32.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F32_MATH_INCLUDED
#define F32_MATH_INCLUDED

#include "fltx/detail/f32_math_basic.h"
#include "fltx/detail/f32_math_transcendental.h"
#include "fltx/traits.h"

namespace bl
{
    template<fltx_f32 T>
    [[nodiscard]] BL_FORCE_INLINE constexpr std::remove_cv_t<T> pow10(int exponent) noexcept
    {
        return static_cast<std::remove_cv_t<T>>(detail::_f32_impl::pow10(exponent));
    }

} // namespace bl

#endif
