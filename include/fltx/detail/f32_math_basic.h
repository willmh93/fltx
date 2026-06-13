/**
 * fltx/detail/f32_math_basic.h - constexpr <cmath>-style basic math helpers for f32.
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
#include "fltx/detail/f64_math_basic.h"
#include "fltx/round_options.h"


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

    using detail::_native_float_decimal::exact_dyadic_to_float;
    using detail::_native_float_decimal::f32_decimal_round_traits;

    inline constexpr int pow10_f32_min_exponent = -45;
    inline constexpr int pow10_f32_max_exponent = 38;
    inline constexpr std::uint32_t pow10_f32_bits[] =
    {
        0x00000001u, 0x00000007u, 0x00000047u, 0x000002cau,
        0x00001be0u, 0x000116c2u, 0x000ae398u, 0x006ce3eeu,
        0x02081ceau, 0x03aa2425u, 0x0554ad2eu, 0x0704ec3du,
        0x08a6274cu, 0x0a4fb11fu, 0x0c01ceb3u, 0x0da24260u,
        0x0f4ad2f8u, 0x10fd87b6u, 0x129e74d2u, 0x14461206u,
        0x15f79688u, 0x179abe15u, 0x19416d9au, 0x1af1c901u,
        0x1c971da0u, 0x1e3ce508u, 0x1fec1e4au, 0x219392efu,
        0x233877aau, 0x24e69595u, 0x26901d7du, 0x283424dcu,
        0x29e12e13u, 0x2b8cbcccu, 0x2d2febffu, 0x2edbe6ffu,
        0x3089705fu, 0x322bcc77u, 0x33d6bf95u, 0x358637bdu,
        0x3727c5acu, 0x38d1b717u, 0x3a83126fu, 0x3c23d70au,
        0x3dcccccdu, 0x3f800000u, 0x41200000u, 0x42c80000u,
        0x447a0000u, 0x461c4000u, 0x47c35000u, 0x49742400u,
        0x4b189680u, 0x4cbebc20u, 0x4e6e6b28u, 0x501502f9u,
        0x51ba43b7u, 0x5368d4a5u, 0x551184e7u, 0x56b5e621u,
        0x58635fa9u, 0x5a0e1bcau, 0x5bb1a2bcu, 0x5d5e0b6bu,
        0x5f0ac723u, 0x60ad78ecu, 0x6258d727u, 0x64078678u,
        0x65a96816u, 0x6753c21cu, 0x69045951u, 0x6aa56fa6u,
        0x6c4ecb8fu, 0x6e013f39u, 0x6fa18f08u, 0x7149f2cau,
        0x72fc6f7cu, 0x749dc5aeu, 0x76453719u, 0x77f684dfu,
        0x799a130cu, 0x7b4097ceu, 0x7cf0bdc2u, 0x7e967699u
    };

    [[nodiscard]] BL_FORCE_INLINE constexpr float pow10(int exponent) noexcept
    {
        if (exponent > pow10_f32_max_exponent)
            return std::numeric_limits<float>::infinity();
        if (exponent < pow10_f32_min_exponent)
            return 0.0f;

        return std::bit_cast<float>(pow10_f32_bits[exponent - pow10_f32_min_exponent]);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float round_to_decimals(float v, int prec) noexcept
    {
        return detail::_f64_impl::round_to_decimals_native<f32_decimal_round_traits>(v, prec);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float round_to_significant_figures(float v, int figures) noexcept
    {
        return detail::_f64_impl::round_to_significant_figures_native<f32_decimal_round_traits>(v, figures);
    }

    BL_FORCE_INLINE constexpr int normalize_remquo_bits(int q) noexcept
    {
        const int magnitude = q < 0 ? -q : q;
        const int bits = detail::fp::remquo_low_quotient_bits(
            static_cast<std::uint64_t>(magnitude),
            q < 0,
            0x7u);
        if (bits == 0)
            return 0;
        return bits;
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

[[nodiscard]] BL_FORCE_INLINE constexpr float round_to_decimals(float x, int prec) noexcept
{
    return detail::_f32_impl::round_to_decimals(x, prec);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float round_to_precision(float x, int figures) noexcept
{
    return detail::_f32_impl::round_to_significant_figures(x, figures);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float round_to(float x, int precision, round_format format) noexcept
{
    return format == round_format::decimals
        ? round_to_decimals(x, precision)
        : round_to_precision(x, precision);
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
