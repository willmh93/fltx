/**
 * fltx/f128_math_erf_gamma.cpp - Runtime f128 error and gamma functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128_math_erf.h"
#include "fltx/detail/f128_math_gamma.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s erf(const f128_s& x) { return detail::_f128_impl::erf(x); }
    BL_NO_INLINE f128_s erfc(const f128_s& x) { return detail::_f128_impl::erfc(x); }
    BL_NO_INLINE f128_s lgamma(const f128_s& x) { return detail::_f128_impl::lgamma(x); }
    BL_NO_INLINE f128_s tgamma(const f128_s& x) { return detail::_f128_impl::tgamma(x); }

} // namespace bl::detail::_f128_runtime
