/**
 * fltx/f256_math.cpp - Core f256 runtime helpers and common math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256_math_basic.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s fmod(const f256_s& x, const f256_s& y)
    {
        return detail::_f256_impl::fmod(x, y); 
    }

    BL_NO_INLINE f256_s round(const f256_s& a) 
    {
        return detail::_f256_impl::round(a);
    }

    BL_NO_INLINE f256_s round_to_decimals(f256_s v, int prec)
    {
        return detail::_f256_impl::round_to_decimals(v, prec);
    }

    BL_NO_INLINE f256_s sqrt(const f256_s& a)
    {
        return detail::_f256_impl::sqrt(a);
    }

    BL_NO_INLINE f256_s nearbyint(const f256_s& a)
    {
        return detail::_f256_impl::nearbyint(a);
    }

    BL_NO_INLINE long lround(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
    }

    BL_NO_INLINE long long llround(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
    }

    BL_NO_INLINE long lrint(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_impl::nearbyint(x));
    }

    BL_NO_INLINE long long llrint(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_impl::nearbyint(x));
    }

    BL_NO_INLINE f256_s ldexp(const f256_s& a, int e)
    { 
        return detail::_f256_impl::ldexp(a, e); 
    }

    BL_NO_INLINE f256_s hypot(const f256_s& x, const f256_s& y)
    { 
        return detail::_f256_impl::hypot(x, y); 
    }

    BL_NO_INLINE f256_s remquo(const f256_s& x, const f256_s& y, int* quo)
    { 
        return detail::_f256_impl::remquo(x, y, quo); 
    }

    BL_NO_INLINE f256_s remainder(const f256_s& x, const f256_s& y)
    { 
        return detail::_f256_impl::remainder(x, y); 
    }

    BL_NO_INLINE f256_s frexp(const f256_s& x, int* exp) noexcept
    { 
        return detail::_f256_impl::frexp(x, exp); 
    }

    BL_NO_INLINE f256_s modf(const f256_s& x, f256_s* iptr) noexcept
    { 
        return detail::_f256_impl::modf(x, iptr); 
    }

    BL_NO_INLINE int ilogb(const f256_s& x) noexcept 
    { 
        return detail::_f256_impl::ilogb(x);
    }

    BL_NO_INLINE f256_s logb(const f256_s& x) noexcept 
    { 
        return detail::_f256_impl::logb(x); 
    }

    BL_NO_INLINE f256_s scalbn(const f256_s& x, int e) noexcept 
    { 
        return detail::_f256_impl::scalbn(x, e); 
    }

    BL_NO_INLINE f256_s scalbln(const f256_s& x, long e) noexcept
    { 
        return detail::_f256_impl::scalbln(x, e); 
    }

    BL_NO_INLINE f256_s nextafter(const f256_s& from, const f256_s& to) noexcept 
    {
        return detail::_f256_impl::nextafter(from, to); 
    }

    BL_NO_INLINE f256_s nexttoward(const f256_s& from, long double to) noexcept
    { 
        return detail::_f256_impl::nexttoward(from, to);
    }

    BL_NO_INLINE f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept 
    {
        return detail::_f256_impl::nexttoward(from, to); 
    }

} // namespace bl::detail::_f256_runtime
