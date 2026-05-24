/**
 * fltx/f32_math_basic.h - constexpr <cmath>-style basic math functions for f32.
 *
 * f32 rounding, decomposition, remainder, min/max, and adjacent-value helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F32_MATH_BASIC_INCLUDED
#define F32_MATH_BASIC_INCLUDED

#include "fltx/f32_classification.h"
#include "fltx/f64_math_basic.h"


namespace bl {

namespace detail::_f32_impl
{
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::trunc;
    using detail::fp::copysign;
    using detail::fp::fabs;
    using detail::fp::isfinite;
    using detail::fp::isinf;
    using detail::fp::isnan;
    using detail::fp::nearbyint;
    using detail::fp::nextafter;
    using detail::fp::round_half_away_zero;
    using detail::fp::signbit;
    using detail::fp::to_signed_integer_or_zero;

    BL_FORCE_INLINE constexpr int normalize_remquo_bits(int q) noexcept
    {
        const int magnitude = q < 0 ? -q : q;
        const int low_bits  = magnitude & 0x7;
        if (low_bits == 0)
            return 0;
        return q < 0 ? -low_bits : low_bits;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float remquo(float x, float y, int* quo) noexcept
    {
        int q = 0;
        const double r = bl::remquo(static_cast<double>(x), static_cast<double>(y), &q);
        if (quo)
            *quo = normalize_remquo_bits(q);
        return static_cast<float>(r);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float fmin(float a, float b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a < b) return a;
        if (b < a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? a : b;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float fmax(float a, float b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a > b) return a;
        if (b > a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? b : a;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float frexp(float x, int* exp) noexcept
    {
        int e = 0;
        const double m = bl::frexp(static_cast<double>(x), &e);
        if (exp)
            *exp = e;
        return static_cast<float>(m);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float modf(float x, float* iptr) noexcept
    {
        double integral = 0.0;
        const double fractional = bl::modf(static_cast<double>(x), &integral);
        if (iptr)
            *iptr = static_cast<float>(integral);
        return static_cast<float>(fractional);
    }

} // namespace detail::_f32_impl

[[nodiscard]] BL_FORCE_INLINE constexpr float floor(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_impl::floor(static_cast<double>(x))),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float ceil(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_impl::ceil(static_cast<double>(x))),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float trunc(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_impl::trunc(static_cast<double>(x))),
        std::trunc(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float round(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::round_half_away_zero(x),
        std::round(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float nearbyint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::nearbyint(x),
        std::nearbyint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float rint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::nearbyint(x),
        std::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::to_signed_integer_or_zero<long>(detail::_f32_impl::round_half_away_zero(x)),
        std::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::to_signed_integer_or_zero<long long>(detail::_f32_impl::round_half_away_zero(x)),
        std::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::to_signed_integer_or_zero<long>(detail::_f32_impl::nearbyint(x)),
        std::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::to_signed_integer_or_zero<long long>(detail::_f32_impl::nearbyint(x)),
        std::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fmod(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::fmod(static_cast<double>(x), static_cast<double>(y))),
        std::fmod(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float remainder(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::remainder(static_cast<double>(x), static_cast<double>(y))),
        std::remainder(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float remquo(float x, float y, int* quo) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::remquo(x, y, quo),
        std::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fma(float x, float y, float z) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::fma(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z))),
        std::fma(x, y, z)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fmin(float a, float b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::fmin(a, b),
        std::fmin(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fmax(float a, float b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::fmax(a, b),
        std::fmax(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fdim(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        (x > y) ? (x - y) : 0.0f,
        std::fdim(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float copysign(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::copysign(x, y),
        std::copysign(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float ldexp(float x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::ldexp(static_cast<double>(x), e)),
        std::ldexp(x, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float scalbn(float x, int e) noexcept
{
    return ldexp(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float scalbln(float x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float frexp(float x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::frexp(x, exp),
        std::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float modf(float x, float* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::modf(x, iptr),
        std::modf(x, iptr)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        bl::ilogb(static_cast<double>(x)),
        std::ilogb(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float logb(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::logb(static_cast<double>(x))),
        std::logb(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float nextafter(float from, float to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::nextafter(from, to),
        std::nextafter(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float nexttoward(float from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        nextafter(from, static_cast<float>(to)),
        std::nexttoward(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float nexttoward(float from, float to) noexcept
{
    return nextafter(from, to);
}

} // namespace bl

#endif
