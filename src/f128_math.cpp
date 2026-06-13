/**
 * fltx/f128_math.cpp - Core f128 runtime helpers and common math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128_math_basic.h"

namespace bl::detail::_f128_runtime
{
    // roots
    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y)
    {
        return detail::_f128_impl::hypot(x, y);
    }

    // rounding and decimals
    BL_NO_INLINE f128_s round(const f128_s& a)
    {
        return detail::_f128_impl::round_runtime(a);
    }

    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec) 
    { 
        return detail::_f128_impl::round_to_decimals(v, prec);
    }

    BL_NO_INLINE f128_s round_to_significant_figures(f128_s v, int figures)
    {
        return detail::_f128_impl::round_to_significant_figures(v, figures);
    }

    BL_NO_INLINE f128_s nearbyint_slow(const f128_s& a)
    {
        return detail::_f128::nearbyint_generic(a);
    }

    BL_NO_INLINE f128_s nearbyint(const f128_s& a)
    {
        return detail::_f128_impl::nearbyint_runtime(a);
    }

    BL_NO_INLINE f128_s rint(const f128_s& x)
    {
        return detail::_f128_impl::rint(x);
    }

    BL_NO_INLINE long lround(const f128_s& x)
    {
        return detail::_f128_impl::lround(x);
    }

    BL_NO_INLINE long long llround(const f128_s& x)
    {
        return detail::_f128_impl::llround(x);
    }

    BL_NO_INLINE long lrint(const f128_s& x)
    {
        return detail::_f128_impl::lrint(x);
    }

    BL_NO_INLINE long long llrint(const f128_s& x)
    {
        return detail::_f128_impl::llrint(x);
    }

    // remainders
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y)
    {
        return detail::_f128_impl::fmod(x, y);
    }

    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y)
    {
        return detail::_f128_impl::remainder(x, y);
    }

    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo) 
    { 
        return detail::_f128_impl::remquo(x, y, quo); 
    }

    // fractional decomposition
    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept 
    { 
        return detail::_f128_impl::modf(x, iptr);
    }

    // decomposition and scaling
    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e)
    {
        return detail::_f128_impl::ldexp(x, e);
    }

} // namespace bl::detail::_f128_runtime
