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
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y) 
    { 
        return detail::_f128_impl::fmod(x, y);
    }

    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec) 
    { 
        return detail::_f128_impl::round_to_decimals(v, prec);
    }

    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y) 
    { 
        return detail::_f128_impl::remainder(x, y); 
    }

    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e) 
    {
        return detail::_f128_impl::ldexp(x, e);
    }

    BL_NO_INLINE long lround(const f128_s& x)
    {
        long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f128_impl::lround(x);
    }

    BL_NO_INLINE long long llround(const f128_s& x)
    {
        long long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f128_impl::llround(x);
    }

    BL_NO_INLINE long lrint(const f128_s& x)
    {
        long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f128_impl::lrint(x);
    }

    BL_NO_INLINE long long llrint(const f128_s& x)
    {
        long long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f128_impl::llrint(x);
    }

    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y) 
    { 
        return detail::_f128_impl::hypot(x, y); 
    }

    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo) 
    { 
        return detail::_f128_impl::remquo(x, y, quo); 
    }

    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept 
    { 
        return detail::_f128_impl::modf(x, iptr);
    }

    BL_NO_INLINE f128_s nextafter(const f128_s& from, const f128_s& to) noexcept 
    { 
        return detail::_f128_impl::nextafter(from, to); 
    }

} // namespace bl::detail::_f128_runtime
