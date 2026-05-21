/**
 * fltx/detail/common/math.h - Shared constexpr scalar math helpers used by FLTX math headers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_COMMON_MATH_INCLUDED
#define FLTX_COMMON_MATH_INCLUDED
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "fltx/detail/fp.h"

namespace bl::detail::fp
{
BL_FORCE_INLINE constexpr double log_series_reduced(double z) noexcept
{
    const double z2 = z * z;
    const double poly =
        1.0 + z2 * (
        1.0 / 3.0 + z2 * (
        1.0 / 5.0 + z2 * (
        1.0 / 7.0 + z2 * (
        1.0 / 9.0 + z2 * (
        1.0 / 11.0 + z2 * (
        1.0 / 13.0 + z2 * (
        1.0 / 15.0 + z2 * (
        1.0 / 17.0 + z2 * (
        1.0 / 19.0 + z2 * (
        1.0 / 21.0 + z2 * (
        1.0 / 23.0 + z2 * (
        1.0 / 25.0 + z2 * (
        1.0 / 27.0 + z2 * (
        1.0 / 29.0 + z2 * (
        1.0 / 31.0)))))))))))))));

    return 2.0 * z * poly;
}

BL_FORCE_INLINE constexpr double log(double x) noexcept
{
    constexpr double ln2 = 0.6931471805599453094172321214581765680755;
    constexpr double sqrt_half = 0.7071067811865475244008443621048490392848;

    if (isnan(x)) return  std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return -std::numeric_limits<double>::infinity();
    if (x < 0.0)  return  std::numeric_limits<double>::quiet_NaN();
    if (isinf(x)) return  std::numeric_limits<double>::infinity();

    int e = frexp_exponent(x);
    double m = ldexp(x, -e);

    if (m < sqrt_half)
    {
        m *= 2.0;
        --e;
    }

    const double z = (m - 1.0) / (m + 1.0);
    return static_cast<double>(e) * ln2 + log_series_reduced(z);
}

BL_FORCE_INLINE constexpr double log1p(double x) noexcept
{
    if (x == -1.0) return -std::numeric_limits<double>::infinity();
    if (x < -1.0 || isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    if (isinf(x)) return x;
    if (x == 0.0) return x;

    const double ax = absd(x);
    if (ax < 0.5)
        return log_series_reduced(x / (2.0 + x));

    return log(1.0 + x);
}


BL_FORCE_INLINE constexpr double round_half_away_zero(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    constexpr double integer_threshold = double_integer_threshold;
    const double ax = signbit(x) ? -x : x;
    if (ax >= integer_threshold)
        return x;

    if (signbit(x))
    {
        const double y = -floor((-x) + 0.5);
        return (y == 0.0) ? -0.0 : y;
    }

    return floor(x + 0.5);
}

BL_FORCE_INLINE constexpr float round_half_away_zero(float x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0f)
        return x;

    constexpr float integer_threshold = 8388608.0f;
    const float ax = fabs(x);
    if (ax >= integer_threshold)
        return x;

    if (signbit(x))
    {
        const float y = static_cast<float>(-floor(static_cast<double>(-x) + 0.5));
        return (y == 0.0f) ? -0.0f : y;
    }

    return static_cast<float>(floor(static_cast<double>(x) + 0.5));
}

BL_FORCE_INLINE constexpr long long llround(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return 0;

    const double rounded = signbit(x) ? (x - 0.5) : (x + 0.5);

    constexpr double min_ll = static_cast<double>(std::numeric_limits<long long>::min());
    constexpr double max_ll = static_cast<double>(std::numeric_limits<long long>::max());

    if (rounded < min_ll || rounded > max_ll)
        return 0;

    return static_cast<long long>(rounded);
}

BL_FORCE_INLINE constexpr double fmod(double x, double y) noexcept
{
    if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(y) || x == 0.0)
        return x;

    const double q = trunc(x / y);
    return x - q * y;
}

BL_FORCE_INLINE constexpr double nextafter(double from, double to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<double>::quiet_NaN();

    if (from == to)
        return to;

    if (from == 0.0)
        return signbit(to)
            ? -std::numeric_limits<double>::denorm_min()
            :  std::numeric_limits<double>::denorm_min();

    std::uint64_t bits = std::bit_cast<std::uint64_t>(from);
    if ((from > 0.0) == (from < to))
        ++bits;
    else
        --bits;

    return std::bit_cast<double>(bits);
}

BL_FORCE_INLINE constexpr float nextafter(float from, float to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<float>::quiet_NaN();

    if (from == to)
        return to;

    if (from == 0.0f)
        return signbit(to)
            ? -std::numeric_limits<float>::denorm_min()
            :  std::numeric_limits<float>::denorm_min();

    std::uint32_t bits = std::bit_cast<std::uint32_t>(from);
    if ((from > 0.0f) == (from < to))
        ++bits;
    else
        --bits;

    return std::bit_cast<float>(bits);
}

BL_FORCE_INLINE constexpr double nearbyint_ties_even(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    const double t = floor(x);
    const double frac = x - t;
    if (frac < 0.5)
        return t;
    if (frac > 0.5)
        return t + 1.0;
    double out = double_integer_is_odd(t) ? (t + 1.0) : t;
    if (out == 0.0)
        return signbit(x) ? -0.0 : 0.0;
    return out;
}

BL_FORCE_INLINE constexpr double nearbyint(double x) noexcept
{
    const double y = nearbyint_ties_even(x);
    if (y == 0.0)
        return signbit(x) ? -0.0 : 0.0;
    return y;
}

BL_FORCE_INLINE constexpr float nearbyint(float x) noexcept
{
    const double y = nearbyint_ties_even(static_cast<double>(x));
    const float out = static_cast<float>(y);
    if (out == 0.0f)
        return signbit(x) ? -0.0f : 0.0f;
    return out;
}

template<typename SignedInt> BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(float x) noexcept
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

template<typename SignedInt> BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(double x) noexcept
{
    static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);

    if (isnan(x) || isinf(x))
        return 0;

    constexpr double lo = static_cast<double>(std::numeric_limits<SignedInt>::lowest());
    constexpr double hi = static_cast<double>(std::numeric_limits<SignedInt>::max());
    if (x < lo || x > hi)
        return 0;

    return static_cast<SignedInt>(x);
}

template<class Value> BL_FORCE_INLINE constexpr Value powi_by_squaring(Value base, std::int64_t exp)
{
    if (exp == 0)
        return Value{ 1.0 };

    const bool invert = exp < 0;
    std::uint64_t n = invert ? magnitude_u64(exp) : static_cast<std::uint64_t>(exp);
    Value result{ 1.0 };

    while (n != 0)
    {
        if ((n & 1u) != 0)
            result = result * base;

        n >>= 1;
        if (n != 0)
            base = base * base;
    }

    return invert ? (Value{ 1.0 } / result) : result;
}

template<class BigUInt> BL_FORCE_INLINE constexpr BigUInt append_decimal_digits(BigUInt coeff, const char* digits, int digit_count) noexcept
{
    for (int i = 0; i < digit_count; ++i)
    {
        coeff.mul_small(10);
        coeff.add_small(static_cast<std::uint32_t>(digits[i] - '0'));
    }

    return coeff;
}

BL_FORCE_INLINE constexpr double atan_series(double x) noexcept
{
    const double x2 = x * x;
    double term = x;
    double sum  = x;
    for (int k = 3; k <= 41; k += 2)
    {
        term *= -x2;
        sum += term / static_cast<double>(k);
    }
    return sum;
}

BL_FORCE_INLINE constexpr double atan_unit(double x) noexcept
{
    constexpr double pi_4     = 0.7853981633974483096156608458198757210493;
    constexpr double tan_pi_8 = 0.4142135623730950488016887242096980785697;

    if (x > tan_pi_8)
    {
        const double t = (x - 1.0) / (x + 1.0);
        return pi_4 + atan_series(t);
    }

    return atan_series(x);
}

BL_FORCE_INLINE constexpr double atan(double x) noexcept
{
    constexpr double pi_2 = 1.5707963267948966192313216916397514420986;

    if (isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(x))
        return signbit(x) ? -pi_2 : pi_2;
    if (x == 0.0)
        return x;

    const bool neg = x < 0.0;
    const double ax = neg ? -x : x;

    const double out = ax > 1.0
        ? pi_2 - atan_unit(1.0 / ax)
        : atan_unit(ax);

    return neg ? -out : out;
}

BL_MSVC_NOINLINE constexpr double atan2(double y, double x) noexcept
{
    constexpr double pi   = 3.1415926535897932384626433832795028841972;
    constexpr double pi_2 = 1.5707963267948966192313216916397514420986;

    if (isnan(x) || isnan(y))
        return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0)
    {
        if (y == 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        return signbit(y) ? -pi_2 : pi_2;
    }

    const double a = atan(y / x);
    if (x < 0.0)
        return signbit(y) ? (a - pi) : (a + pi);
    return a;
}

BL_FORCE_INLINE constexpr void reduce_pi_over_2(double x, int& quadrant, double& r) noexcept
{
    constexpr double pi_2_hi  = 0x1.921fb54442d18p+0;
    constexpr double pi_2_lo  = 0x1.1a62633145c07p-54;
    constexpr double inv_pi_2 = 0x1.45f306dc9c883p-1;
    constexpr double pi_4_hi  = 0x1.921fb54442d18p-1;

    const double n = nearbyint_ties_even(x * inv_pi_2);
    r = (x - n * pi_2_hi) - n * pi_2_lo;

    const int q0 = static_cast<int>(n) & 3;
    if (r > pi_4_hi)
    {
        r = (r - pi_2_hi) - pi_2_lo;
        quadrant = (q0 + 1) & 3;
    }
    else if (r < -pi_4_hi)
    {
        r = (r + pi_2_hi) + pi_2_lo;
        quadrant = (q0 + 3) & 3;
    }
    else
    {
        quadrant = q0;
    }
}

BL_FORCE_INLINE constexpr double sin_poly_reduced(double r) noexcept
{
    const double t = r * r;

    double p = 2.8114572543455206e-15;   //  1/17!
    p = p * t - 7.6471637318198164e-13;  // -1/15!
    p = p * t + 1.6059043836821615e-10;  //  1/13!
    p = p * t - 2.5052108385441719e-8;   // -1/11!
    p = p * t + 2.7557319223985891e-6;   //  1/9!
    p = p * t - 1.9841269841269841e-4;   // -1/7!
    p = p * t + 8.3333333333333332e-3;   //  1/5!
    p = p * t - 1.6666666666666666e-1;   // -1/3!

    return r + r * t * p;
}

BL_FORCE_INLINE constexpr double cos_poly_reduced(double r) noexcept
{
    const double t = r * r;

    double p = 4.7794773323873853e-14;   //  1/16!
    p = p * t - 1.1470745597729725e-11;  // -1/14!
    p = p * t + 2.0876756987868099e-9;   //  1/12!
    p = p * t - 2.7557319223985893e-7;   // -1/10!
    p = p * t + 2.4801587301587302e-5;   //  1/8!
    p = p * t - 1.3888888888888889e-3;   // -1/6!
    p = p * t + 4.1666666666666664e-2;   //  1/4!
    p = p * t - 5.0e-1;                  // -1/2

    return 1.0 + t * p;
}

BL_MSVC_NOINLINE constexpr void sincos(double x, double& s, double& c) noexcept
{
    if (isnan(x) || isinf(x))
    {
        s = std::numeric_limits<double>::quiet_NaN();
        c = s;
        return;
    }
    if (x == 0.0)
    {
        s = x;
        c = 1.0;
        return;
    }

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2(x, quadrant, r);

    const double sr = sin_poly_reduced(r);
    const double cr = cos_poly_reduced(r);

    switch (quadrant)
    {
    case 0: s = sr; c = cr; break;
    case 1: s = cr; c = -sr; break;
    case 2: s = -sr; c = -cr; break;
    default: s = -cr; c = sr; break;
    }
}

BL_FORCE_INLINE constexpr double sin(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0)
        return x;

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2(x, quadrant, r);

    switch (quadrant)
    {
    case 0: return sin_poly_reduced(r);
    case 1: return cos_poly_reduced(r);
    case 2: return -sin_poly_reduced(r);
    default: return -cos_poly_reduced(r);
    }
}

BL_FORCE_INLINE constexpr double cos(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2(x, quadrant, r);

    switch (quadrant)
    {
    case 0: return cos_poly_reduced(r);
    case 1: return -sin_poly_reduced(r);
    case 2: return -cos_poly_reduced(r);
    default: return sin_poly_reduced(r);
    }
}

BL_FORCE_INLINE constexpr double tan(double x) noexcept
{
    double s{}, c{};
    sincos(x, s, c);
    return s / c;
}

BL_FORCE_INLINE constexpr double sqrt_seed(double x) noexcept
{
    if (!(x > 0.0) || isnan(x) || isinf(x))
        return x;

    int exp2 = frexp_exponent(x);
    double m = ldexp(x, -exp2);

    if ((exp2 & 1) != 0)
    {
        m *= 2.0;
        --exp2;
    }

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(m);
    const std::uint64_t seed = (bits >> 1) + 0x1ff8000000000000ULL;
    double y = std::bit_cast<double>(seed);

    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);

    return ldexp(y, exp2 / 2);
}


} // namespace bl::detail::fp

#endif
