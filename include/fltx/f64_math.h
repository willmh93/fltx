/**
 * fltx/f64_math.h - constexpr <cmath>-style functions for f64.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_MATH_INCLUDED
#define F64_MATH_INCLUDED

#include "fltx/detail/f64_math_basic.h"
#include "fltx/detail/f64_math_transcendental.h"
#include "fltx/traits.h"

namespace bl
{
    template<fltx_f64 T>
    [[nodiscard]] BL_FORCE_INLINE constexpr std::remove_cv_t<T> pow10(int exponent) noexcept
    {
        return static_cast<std::remove_cv_t<T>>(detail::_f64_impl::pow10(exponent));
    }

} // namespace bl

#endif
