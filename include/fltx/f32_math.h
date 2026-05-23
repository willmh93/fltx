/**
 * fltx/f32_math.h - constexpr <cmath>-style functions for f32.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F32_MATH_INCLUDED
#define F32_MATH_INCLUDED

#include "fltx/f64_math.h"

namespace bl {

namespace detail::_f32_constexpr
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

    BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
    {
        return x == 0.0f;
    }

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

} // namespace detail::_f32_constexpr

namespace detail::_f32_runtime
{
    [[nodiscard]] BL_FORCE_INLINE float erfc(float x) noexcept
    {
#if defined(__MINGW32__)
        return static_cast<float>(std::erfc(static_cast<double>(x)));
#else
        return std::erfc(x);
#endif
    }

} // namespace detail::_f32_runtime


[[nodiscard]] BL_FORCE_INLINE constexpr float abs(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::fabs(x),
        std::abs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fabs(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::fabs(x),
        std::fabs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::signbit(x),
        std::signbit(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::isnan(x),
        std::isnan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::isinf(x),
        std::isinf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::isfinite(x),
        std::isfinite(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
{
    return detail::_f32_constexpr::iszero(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float floor(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_constexpr::floor(static_cast<double>(x))),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float ceil(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_constexpr::ceil(static_cast<double>(x))),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float trunc(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(detail::_f32_constexpr::trunc(static_cast<double>(x))),
        std::trunc(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float round(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::round_half_away_zero(x),
        std::round(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float nearbyint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::nearbyint(x),
        std::nearbyint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float rint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::nearbyint(x),
        std::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::to_signed_integer_or_zero<long>(detail::_f32_constexpr::round_half_away_zero(x)),
        std::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::to_signed_integer_or_zero<long long>(detail::_f32_constexpr::round_half_away_zero(x)),
        std::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::to_signed_integer_or_zero<long>(detail::_f32_constexpr::nearbyint(x)),
        std::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::to_signed_integer_or_zero<long long>(detail::_f32_constexpr::nearbyint(x)),
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
        detail::_f32_constexpr::remquo(x, y, quo),
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
        detail::_f32_constexpr::fmin(a, b),
        std::fmin(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fmax(float a, float b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::fmax(a, b),
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
        detail::_f32_constexpr::copysign(x, y),
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
        detail::_f32_constexpr::frexp(x, exp),
        std::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float modf(float x, float* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_constexpr::modf(x, iptr),
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
        detail::_f32_constexpr::nextafter(from, to),
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

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(float x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<float>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(float x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(float a, float b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(float a, float b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(float a, float b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(float a, float b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(float a, float b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(float a, float b) noexcept
{
    return !isunordered(a, b) && a != b;
}


// exp / log

[[nodiscard]] BL_FORCE_INLINE constexpr float exp(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::exp(static_cast<double>(x))),
        std::exp(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float exp2(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::exp2(static_cast<double>(x))),
        std::exp2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float expm1(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::expm1(static_cast<double>(x))),
        std::expm1(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        bl::log(static_cast<double>(x)),
        std::log(static_cast<double>(x))
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float log(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::log(static_cast<double>(x))),
        std::log(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float log2(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::log2(static_cast<double>(x))),
        std::log2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float log10(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::log10(static_cast<double>(x))),
        std::log10(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float log1p(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::log1p(static_cast<double>(x))),
        std::log1p(x)
    );
}


// roots

[[nodiscard]] BL_FORCE_INLINE constexpr float sqrt(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::sqrt(static_cast<double>(x))),
        std::sqrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float cbrt(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::cbrt(static_cast<double>(x))),
        std::cbrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float hypot(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::hypot(static_cast<double>(x), static_cast<double>(y))),
        std::hypot(x, y)
    );
}


// pow

[[nodiscard]] BL_FORCE_INLINE constexpr float pow(float x, float y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::pow(static_cast<double>(x), static_cast<double>(y))),
        std::pow(x, y)
    );
}


// trig

[[nodiscard]] BL_FORCE_INLINE constexpr float sin(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::sin(static_cast<double>(x))),
        std::sin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float cos(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::cos(static_cast<double>(x))),
        std::cos(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(float x, float& s_out, float& c_out) noexcept
{
    s_out = bl::sin(x);
    c_out = bl::cos(x);
    return bl::isfinite(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float tan(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::tan(static_cast<double>(x))),
        std::tan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float atan(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::atan(static_cast<double>(x))),
        std::atan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float atan2(float y, float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::atan2(static_cast<double>(y), static_cast<double>(x))),
        std::atan2(y, x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float asin(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::asin(static_cast<double>(x))),
        std::asin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float acos(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::acos(static_cast<double>(x))),
        std::acos(x)
    );
}

template<class Vec> requires detail::fp::sincos_vector_assignable<Vec, float>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(float x, Vec& out)
{
    float s_out{};
    float c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    detail::fp::assign_sincos_vector(out, s_out, c_out);
    return ok;
}

template<class Value> requires std::same_as<std::remove_cvref_t<Value>, float>
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<float> sincos(float x)
{
    float s_out{};
    float c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    return detail::fp::make_sincos_result(s_out, c_out, ok);
}


// hyperbolic

[[nodiscard]] BL_FORCE_INLINE constexpr float sinh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::sinh(static_cast<double>(x))),
        std::sinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float cosh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::cosh(static_cast<double>(x))),
        std::cosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float tanh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::tanh(static_cast<double>(x))),
        std::tanh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float asinh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::asinh(static_cast<double>(x))),
        std::asinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float acosh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::acosh(static_cast<double>(x))),
        std::acosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float atanh(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::atanh(static_cast<double>(x))),
        std::atanh(x)
    );
}


// erf / erfc

[[nodiscard]] BL_FORCE_INLINE constexpr float erf(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::erf(static_cast<double>(x))),
        std::erf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float erfc(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::erfc(static_cast<double>(x))),
        detail::_f32_runtime::erfc(x)
    );
}


// gamma

[[nodiscard]] BL_FORCE_INLINE constexpr float lgamma(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::lgamma(static_cast<double>(x))),
        std::lgamma(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float tgamma(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        static_cast<float>(bl::tgamma(static_cast<double>(x))),
        std::tgamma(x)
    );
}


} // namespace bl

#endif
