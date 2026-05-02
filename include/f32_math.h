
/**
 * f32_math.h — f32 (float) constexpr <cmath> style API
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */


#ifndef F32_INCLUDED
#define F32_INCLUDED

#include "f64_math.h"
#include "fltx_math_utils.h"

namespace bl {

using f32 = float;

namespace detail::_f32
{
    using detail::fp::nearbyint_ties_even;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::trunc_constexpr;

    BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
    {
        return x == 0.0f;
    }
    BL_FORCE_INLINE constexpr bool signbit_constexpr(float x) noexcept
    {
        return (std::bit_cast<std::uint32_t>(x) & 0x80000000u) != 0u;
    }
    BL_FORCE_INLINE constexpr bool isnan(float x) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
        return (bits & 0x7fffffffu) > 0x7f800000u;
    }
    BL_FORCE_INLINE constexpr bool isinf(float x) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
        return (bits & 0x7fffffffu) == 0x7f800000u;
    }
    BL_FORCE_INLINE constexpr bool isfinite(float x) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
        return (bits & 0x7f800000u) != 0x7f800000u;
    }

    BL_FORCE_INLINE constexpr float fabs_constexpr(float x) noexcept
    {
        return std::bit_cast<float>(std::bit_cast<std::uint32_t>(x) & 0x7fffffffu);
    }
    BL_FORCE_INLINE constexpr float nearbyint_constexpr(float x) noexcept
    {
        const double y = nearbyint_ties_even(static_cast<double>(x));
        const float out = static_cast<float>(y);
        if (out == 0.0f)
            return signbit_constexpr(x) ? -0.0f : 0.0f;
        return out;
    }
    BL_FORCE_INLINE constexpr float round_half_away_zero(float x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        constexpr float integer_threshold = 8388608.0f;
        const float ax = fabs_constexpr(x);
        if (ax >= integer_threshold)
            return x;

        if (signbit_constexpr(x))
        {
            const float y = static_cast<float>(-floor_constexpr(static_cast<double>(-x) + 0.5));
            return iszero(y) ? -0.0f : y;
        }

        return static_cast<float>(floor_constexpr(static_cast<double>(x) + 0.5));
    }
    BL_FORCE_INLINE constexpr float nextafter_float_constexpr(float from, float to) noexcept
    {
        if (isnan(from) || isnan(to))
            return std::numeric_limits<float>::quiet_NaN();

        if (from == to)
            return to;

        if (from == 0.0f)
            return signbit_constexpr(to)
                ? -std::numeric_limits<float>::denorm_min()
                : std::numeric_limits<float>::denorm_min();

        std::uint32_t bits = std::bit_cast<std::uint32_t>(from);
        if ((from > 0.0f) == (from < to))
            ++bits;
        else
            --bits;

        return std::bit_cast<float>(bits);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(float x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);

        if (isnan(x) || isinf(x))
            return 0;

        const double dx = static_cast<double>(x);
        constexpr double lo = static_cast<double>(std::numeric_limits<SignedInt>::lowest());
        constexpr double hi = static_cast<double>(std::numeric_limits<SignedInt>::max());
        if (dx < lo || dx > hi)
            return 0;

        return static_cast<SignedInt>(x);
    }
    BL_FORCE_INLINE constexpr int normalize_remquo_bits(int q) noexcept
    {
        const int magnitude = q < 0 ? -q : q;
        const int low_bits = magnitude & 0x7;
        if (low_bits == 0)
            return 0;
        return q < 0 ? -low_bits : low_bits;
    }
}

[[nodiscard]] BL_FORCE_INLINE constexpr float abs(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::abs(x);
    return detail::_f32::fabs_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr float fabs(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fabs(x);
    return detail::_f32::fabs_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::signbit(x);
    return detail::_f32::signbit_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isnan(x);
    return detail::_f32::isnan(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isinf(x);
    return detail::_f32::isinf(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isfinite(x);
    return detail::_f32::isfinite(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
{
    return detail::_f32::iszero(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float floor(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::floor(x);
    return static_cast<float>(detail::_f32::floor_constexpr(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float ceil(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::ceil(x);
    return static_cast<float>(detail::_f32::ceil_constexpr(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float trunc(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::trunc(x);
    return static_cast<float>(detail::_f32::trunc_constexpr(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float round(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::round(x);
    return detail::_f32::round_half_away_zero(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr float nearbyint(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nearbyint(x);
    return detail::_f32::nearbyint_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr float rint(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::rint(x);
    return nearbyint(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lround(x);
    return detail::_f32::to_signed_integer_or_zero<long>(round(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::llround(x);
    return detail::_f32::to_signed_integer_or_zero<long long>(round(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lrint(x);
    return detail::_f32::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::llrint(x);
    return detail::_f32::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fmod(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fmod(x, y);
    return static_cast<float>(bl::fmod(static_cast<double>(x), static_cast<double>(y)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float remainder(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::remainder(x, y);
    return static_cast<float>(bl::remainder(static_cast<double>(x), static_cast<double>(y)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float remquo(float x, float y, int* quo) noexcept
{
    if (!bl::use_constexpr_math())
        return std::remquo(x, y, quo);

    int q = 0;
    const double r = bl::remquo(static_cast<double>(x), static_cast<double>(y), &q);
    if (quo)
        *quo = detail::_f32::normalize_remquo_bits(q);
    return static_cast<float>(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fma(float x, float y, float z) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fma(x, y, z);
    return static_cast<float>(bl::fma(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)));
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
[[nodiscard]] BL_FORCE_INLINE constexpr float fdim(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fdim(x, y);
    return (x > y) ? (x - y) : 0.0f;
}
[[nodiscard]] BL_FORCE_INLINE constexpr float copysign(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::copysign(x, y);

    const std::uint32_t xb = std::bit_cast<std::uint32_t>(x) & 0x7fffffffu;
    const std::uint32_t yb = std::bit_cast<std::uint32_t>(y) & 0x80000000u;
    return std::bit_cast<float>(xb | yb);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float ldexp(float x, int e) noexcept
{
    if (!bl::use_constexpr_math())
        return std::ldexp(x, e);
    return static_cast<float>(bl::ldexp(static_cast<double>(x), e));
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
    if (!bl::use_constexpr_math())
        return std::frexp(x, exp);

    int e = 0;
    const double m = bl::frexp(static_cast<double>(x), &e);
    if (exp)
        *exp = e;
    return static_cast<float>(m);
}
[[nodiscard]] BL_FORCE_INLINE constexpr float modf(float x, float* iptr) noexcept
{
    if (!bl::use_constexpr_math())
        return std::modf(x, iptr);

    double integral = 0.0;
    const double fractional = bl::modf(static_cast<double>(x), &integral);
    if (iptr)
        *iptr = static_cast<float>(integral);
    return static_cast<float>(fractional);
}
[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::ilogb(x);
    return bl::ilogb(static_cast<double>(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float logb(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::logb(x);
    return static_cast<float>(bl::logb(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float nextafter(float from, float to) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nextafter(from, to);
    return detail::_f32::nextafter_float_constexpr(from, to);
}
[[nodiscard]] BL_FORCE_INLINE constexpr float nexttoward(float from, long double to) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nexttoward(from, to);
    return nextafter(from, static_cast<float>(to));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float nexttoward(float from, float to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] BL_FORCE_INLINE constexpr float exp(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::exp(x);
    return static_cast<float>(bl::exp(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float exp2(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::exp2(x);
    return static_cast<float>(bl::exp2(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float expm1(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::expm1(x);
    return static_cast<float>(bl::expm1(static_cast<double>(x)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log(static_cast<double>(x));
    return bl::log(static_cast<double>(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float log(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log(x);
    return static_cast<float>(bl::log(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float log2(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log2(x);
    return static_cast<float>(bl::log2(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float log10(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log10(x);
    return static_cast<float>(bl::log10(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float log1p(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log1p(x);
    return static_cast<float>(bl::log1p(static_cast<double>(x)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float sqrt(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sqrt(x);
    return static_cast<float>(bl::sqrt(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float cbrt(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cbrt(x);
    return static_cast<float>(bl::cbrt(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float hypot(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::hypot(x, y);
    return static_cast<float>(bl::hypot(static_cast<double>(x), static_cast<double>(y)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float sin(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sin(x);
    return static_cast<float>(bl::sin(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float cos(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cos(x);
    return static_cast<float>(bl::cos(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float tan(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tan(x);
    return static_cast<float>(bl::tan(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float atan(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atan(x);
    return static_cast<float>(bl::atan(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float atan2(float y, float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atan2(y, x);
    return static_cast<float>(bl::atan2(static_cast<double>(y), static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float asin(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::asin(x);
    return static_cast<float>(bl::asin(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float acos(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::acos(x);
    return static_cast<float>(bl::acos(static_cast<double>(x)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float sinh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sinh(x);
    return static_cast<float>(bl::sinh(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float cosh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cosh(x);
    return static_cast<float>(bl::cosh(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float tanh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tanh(x);
    return static_cast<float>(bl::tanh(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float asinh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::asinh(x);
    return static_cast<float>(bl::asinh(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float acosh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::acosh(x);
    return static_cast<float>(bl::acosh(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float atanh(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atanh(x);
    return static_cast<float>(bl::atanh(static_cast<double>(x)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float pow(float x, float y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::pow(x, y);
    return static_cast<float>(bl::pow(static_cast<double>(x), static_cast<double>(y)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr float erf(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::erf(x);
    return static_cast<float>(bl::erf(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float erfc(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::erfc(x);
    return static_cast<float>(bl::erfc(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float lgamma(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lgamma(x);
    return static_cast<float>(bl::lgamma(static_cast<double>(x)));
}
[[nodiscard]] BL_FORCE_INLINE constexpr float tgamma(float x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tgamma(x);
    return static_cast<float>(bl::tgamma(static_cast<double>(x)));
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

} // namespace bl

#endif