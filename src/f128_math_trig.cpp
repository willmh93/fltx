/**
 * fltx/f128_math_trig.cpp - Runtime f128 trigonometric functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128_math_trig.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out) { return detail::_f128_impl::sincos(x, s_out, c_out); }
    BL_NO_INLINE f128_s sin(const f128_s& x) { return detail::_f128_impl::sin(x); }
    BL_NO_INLINE f128_s cos(const f128_s& x) { return detail::_f128_impl::cos(x); }
    BL_NO_INLINE f128_s tan(const f128_s& x) { return detail::_f128_impl::tan(x); }
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x) { return detail::_f128_impl::atan2(y, x); }

} // namespace bl::detail::_f128_runtime
