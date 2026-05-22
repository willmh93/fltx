/**
 * fltx/f128_math_erf_gamma.cpp - Runtime f128 error and gamma functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128/math/erf.h"
#include "fltx/detail/f128/math/gamma.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s erf(const f128_s& x) { return detail::_f128_constexpr::erf(x); }
    BL_NO_INLINE f128_s erfc(const f128_s& x) { return detail::_f128_constexpr::erfc(x); }
    BL_NO_INLINE f128_s lgamma(const f128_s& x) { return detail::_f128_constexpr::lgamma(x); }
    BL_NO_INLINE f128_s tgamma(const f128_s& x) { return detail::_f128_constexpr::tgamma(x); }

} // namespace bl::detail::_f128_runtime
