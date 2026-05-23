/**
 * fltx/f128_math_exp_log_pow.cpp - Runtime f128 exponential, logarithm, and power functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128_math_pow.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s exp(const f128_s& x) { return detail::_f128_impl::exp(x); }
    BL_NO_INLINE f128_s exp2(const f128_s& x) { return detail::_f128_impl::exp2(x); }
    BL_NO_INLINE f128_s expm1(const f128_s& x) { return detail::_f128_impl::expm1(x); }

    BL_NO_INLINE f128_s log(const f128_s& a) { return detail::_f128_impl::log(a); }
    BL_NO_INLINE f128_s log2(const f128_s& a) { return detail::_f128_impl::log2(a); }
    BL_NO_INLINE f128_s log10(const f128_s& x) { return detail::_f128_impl::log10(x); }
    BL_NO_INLINE f128_s log1p(const f128_s& x) { return detail::_f128_impl::log1p(x); }

    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y) { return detail::_f128_impl::pow(x, y); }
    BL_NO_INLINE f128_s pow(const f128_s& x, double y) { return detail::_f128_impl::pow(x, y); }

} // namespace bl::detail::_f128_runtime
