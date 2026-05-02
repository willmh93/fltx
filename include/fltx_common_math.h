
/**
 * fltx_common_math.h — constexpr math logic used by all float types
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */


#ifndef FLTX_COMMON_MATH_INCLUDED
#define FLTX_COMMON_MATH_INCLUDED

#include "fltx_common_base.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

namespace bl::detail::fp
{
        
BL_FORCE_INLINE constexpr bool isinf(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_inf bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL;
}
BL_FORCE_INLINE constexpr bool isfinite(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "isfinite bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

template<int bits_to_clear>
BL_FORCE_INLINE constexpr double zero_low_fraction_bits_finite(double value) noexcept
{
    static_assert(bits_to_clear >= 0 && bits_to_clear <= 52);

    if constexpr (bits_to_clear == 0)
        return value;

    if (!isfinite(value) || value == 0.0)
        return value;

    constexpr std::uint64_t fraction_mask = (std::uint64_t{ 1 } << 52) - 1ULL;
    constexpr std::uint64_t clear_mask = ~((std::uint64_t{ 1 } << bits_to_clear) - 1ULL);

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t sign_and_exponent = bits & ~fraction_mask;
    const std::uint64_t fraction = bits & fraction_mask;
    return std::bit_cast<double>(sign_and_exponent | (fraction & clear_mask));
}
BL_FORCE_INLINE constexpr bool isnan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t abs_bits = bits & 0x7fffffffffffffffULL;
    return abs_bits > 0x7ff0000000000000ULL;
}
BL_FORCE_INLINE constexpr double absd(double x) noexcept { return (x < 0.0) ? -x : x; }

BL_FORCE_INLINE constexpr int    frexp_exponent_constexpr(double x) noexcept
{
    if (x == 0.0 || !isfinite(x))
        return 0;

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
    if (exp_bits != 0)
        return static_cast<int>(exp_bits) - 1022;

    std::uint64_t frac = bits & ((std::uint64_t{ 1 } << 52) - 1);
    int e = -1022;
    while ((frac & (std::uint64_t{ 1 } << 52)) == 0)
    {
        frac <<= 1;
        --e;
    }
    return e + 1;
}
BL_FORCE_INLINE constexpr int    highest_bit_index_constexpr(std::uint64_t value) noexcept
{
    int index = -1;
    while (value != 0)
    {
        value >>= 1;
        ++index;
    }
    return index;
}
BL_FORCE_INLINE constexpr double scalbn_constexpr2(double value, int exp) noexcept
{
    if (value == 0.0 || isnan(value) || isinf(value) || exp == 0)
        return value;

    constexpr std::uint64_t sign_mask = 0x8000000000000000ull;
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
    constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;
    constexpr std::uint64_t hidden_bit = 0x0010000000000000ull;

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t sign = bits & sign_mask;
    const std::uint64_t fraction = bits & fraction_mask;
    const std::uint32_t exponent_bits = static_cast<std::uint32_t>((bits & exponent_mask) >> 52);

    std::uint64_t significand = 0;
    long long unbiased_exponent = 0;

    if (exponent_bits != 0)
    {
        significand = hidden_bit | fraction;
        unbiased_exponent = static_cast<int>(exponent_bits) - 1023;
    }
    else
    {
        const int msb_index = highest_bit_index_constexpr(fraction);
        significand = fraction << (52 - msb_index);
        unbiased_exponent = static_cast<long long>(msb_index) - 1074ll;
    }

    const long long new_unbiased_exponent = unbiased_exponent + static_cast<long long>(exp);

    if (new_unbiased_exponent > 1023)
        return std::bit_cast<double>(sign | exponent_mask);

    if (new_unbiased_exponent >= -1022)
    {
        const std::uint64_t new_exponent_bits =
            static_cast<std::uint64_t>(new_unbiased_exponent + 1023) << 52;
        const std::uint64_t new_fraction = significand & fraction_mask;
        return std::bit_cast<double>(sign | new_exponent_bits | new_fraction);
    }

    const long long shift = -1022ll - new_unbiased_exponent;
    if (shift >= 64)
        return std::bit_cast<double>(sign);

    const unsigned shift_u = static_cast<unsigned>(shift);

    std::uint64_t subnormal_fraction = 0;
    if (shift_u == 0)
    {
        subnormal_fraction = significand;
    }
    else
    {
        const std::uint64_t truncated = significand >> shift_u;
        const std::uint64_t remainder_mask = (std::uint64_t{ 1 } << shift_u) - 1;
        const std::uint64_t remainder = significand & remainder_mask;
        const std::uint64_t halfway = std::uint64_t{ 1 } << (shift_u - 1);
        const bool round_up =
            (remainder > halfway) ||
            (remainder == halfway && (truncated & 1u) != 0);

        subnormal_fraction = truncated + static_cast<std::uint64_t>(round_up);
    }

    if (subnormal_fraction >= hidden_bit)
        return std::bit_cast<double>(sign | (std::uint64_t{ 1 } << 52));

    if (subnormal_fraction == 0)
        return std::bit_cast<double>(sign);

    return std::bit_cast<double>(sign | subnormal_fraction);
}
BL_FORCE_INLINE constexpr double ldexp_constexpr2(double value, int exp) noexcept
{
    return scalbn_constexpr2(value, exp);
}
BL_FORCE_INLINE constexpr double log_series_reduced_constexpr(double z) noexcept
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
BL_FORCE_INLINE constexpr double log_constexpr(double x) noexcept
{
    constexpr double ln2 = 0.6931471805599453094172321214581765680755;
    constexpr double sqrt_half = 0.7071067811865475244008443621048490392848;

    if (isnan(x)) return  std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return -std::numeric_limits<double>::infinity();
    if (x < 0.0)  return  std::numeric_limits<double>::quiet_NaN();
    if (isinf(x)) return  std::numeric_limits<double>::infinity();

    int e = frexp_exponent_constexpr(x);
    double m = ldexp_constexpr2(x, -e);

    if (m < sqrt_half)
    {
        m *= 2.0;
        --e;
    }

    const double z = (m - 1.0) / (m + 1.0);
    return static_cast<double>(e) * ln2 + log_series_reduced_constexpr(z);
}
BL_FORCE_INLINE constexpr double log1p_constexpr(double x) noexcept
{
    if (x == -1.0) return -std::numeric_limits<double>::infinity();
    if (x < -1.0 || isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    if (isinf(x)) return x;
    if (x == 0.0) return x;

    const double ax = absd(x);
    if (ax < 0.5)
        return log_series_reduced_constexpr(x / (2.0 + x));

    return log_constexpr(1.0 + x);
}


BL_FORCE_INLINE constexpr bool   signbit_constexpr(double x) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    return (bits >> 63) != 0;
}
BL_FORCE_INLINE constexpr double fabs_constexpr(double x) noexcept
{
    return absd(x);
}
BL_FORCE_INLINE constexpr double floor_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    const double ax = absd(x);
    if (ax >= 4503599627370496.0)
        return x;

    const long long i = static_cast<long long>(x);
    double di = static_cast<double>(i);
    if (di > x) di -= 1.0;
    if (di == 0.0) return signbit_constexpr(x) ? -0.0 : 0.0;
    return di;
}
BL_FORCE_INLINE constexpr double ceil_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    const double ax = absd(x);
    if (ax >= 4503599627370496.0)
        return x;

    const long long i = static_cast<long long>(x);
    double di = static_cast<double>(i);
    if (di < x)
        di += 1.0;
    if (di == 0.0)
        return signbit_constexpr(x) ? -0.0 : 0.0;
    return di;
}
BL_FORCE_INLINE constexpr double trunc_constexpr(double x) noexcept
{
    return signbit_constexpr(x) ? ceil_constexpr(x) : floor_constexpr(x);
}
BL_FORCE_INLINE constexpr long long llround_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return 0;

    const double rounded = signbit_constexpr(x) ? (x - 0.5) : (x + 0.5);

    constexpr double min_ll = static_cast<double>(std::numeric_limits<long long>::min());
    constexpr double max_ll = static_cast<double>(std::numeric_limits<long long>::max());

    if (rounded < min_ll || rounded > max_ll)
        return 0;

    return static_cast<long long>(rounded);
}
BL_FORCE_INLINE constexpr double fmod_constexpr(double x, double y) noexcept
{
    if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(y) || x == 0.0)
        return x;

    const double q = trunc_constexpr(x / y);
    return x - q * y;
}
BL_FORCE_INLINE constexpr bool   double_integer_is_odd(double x) noexcept
{
    const double ax = absd(x);
    if (!isfinite(x) || ax < 1.0 || ax >= 9007199254740992.0)
        return false;
    const long long i = static_cast<long long>(x);
    return (i & 1ll) != 0;
}
BL_FORCE_INLINE constexpr double nearbyint_ties_even(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    const double t = floor_constexpr(x);
    const double frac = x - t;
    if (frac < 0.5)
        return t;
    if (frac > 0.5)
        return t + 1.0;
    double out = double_integer_is_odd(t) ? (t + 1.0) : t;
    if (out == 0.0)
        return signbit_constexpr(x) ? -0.0 : 0.0;
    return out;
}

BL_FORCE_INLINE constexpr double atan_series_constexpr(double x) noexcept
{
    const double x2 = x * x;
    double term = x;
    double sum = x;
    for (int k = 3; k <= 41; k += 2)
    {
        term *= -x2;
        sum += term / static_cast<double>(k);
    }
    return sum;
}
BL_FORCE_INLINE constexpr double atan_constexpr(double x) noexcept
{
    constexpr double pi_2 = 1.5707963267948966192313216916397514420986;
    constexpr double pi_4 = 0.7853981633974483096156608458198757210493;
    constexpr double tan_pi_8 = 0.4142135623730950488016887242096980785697;

    if (isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(x))
        return signbit_constexpr(x) ? -pi_2 : pi_2;
    if (x == 0.0)
        return x;

    const bool neg = x < 0.0;
    const double ax = neg ? -x : x;

    double out = 0.0;
    if (ax > 1.0)
    {
        out = pi_2 - atan_constexpr(1.0 / ax);
    }
    else if (ax > tan_pi_8)
    {
        const double t = (ax - 1.0) / (ax + 1.0);
        out = pi_4 + atan_series_constexpr(t);
    }
    else
        out = atan_series_constexpr(ax);

    return neg ? -out : out;
}
BL_NO_INLINE    constexpr double atan2_constexpr(double y, double x) noexcept
{
    constexpr double pi = 3.1415926535897932384626433832795028841972;
    constexpr double pi_2 = 1.5707963267948966192313216916397514420986;

    if (isnan(x) || isnan(y))
        return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0)
    {
        if (y == 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        return signbit_constexpr(y) ? -pi_2 : pi_2;
    }

    const double a = atan_constexpr(y / x);
    if (x < 0.0)
        return signbit_constexpr(y) ? (a - pi) : (a + pi);
    return a;
}
BL_FORCE_INLINE constexpr void   reduce_pi_over_2_constexpr(double x, int& quadrant, double& r) noexcept
{
    constexpr double pi_2_hi = 0x1.921fb54442d18p+0;
    constexpr double pi_2_lo = 0x1.1a62633145c07p-54;
    constexpr double inv_pi_2 = 0x1.45f306dc9c883p-1;
    constexpr double pi_4_hi = 0x1.921fb54442d18p-1;

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
BL_FORCE_INLINE constexpr double sin_poly_reduced_constexpr(double r) noexcept
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

BL_FORCE_INLINE constexpr double cos_poly_reduced_constexpr(double r) noexcept
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
BL_NO_INLINE    constexpr void   sincos_constexpr(double x, double& s, double& c) noexcept
{
    if (isnan(x) || isinf(x))
    {
        s = std::numeric_limits<double>::quiet_NaN();
        c = s;
        return;
    }

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2_constexpr(x, quadrant, r);

    const double sr = sin_poly_reduced_constexpr(r);
    const double cr = cos_poly_reduced_constexpr(r);

    switch (quadrant)
    {
    case 0: s = sr; c = cr; break;
    case 1: s = cr; c = -sr; break;
    case 2: s = -sr; c = -cr; break;
    default: s = -cr; c = sr; break;
    }
}
BL_FORCE_INLINE constexpr double sin_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2_constexpr(x, quadrant, r);

    switch (quadrant)
    {
    case 0: return sin_poly_reduced_constexpr(r);
    case 1: return cos_poly_reduced_constexpr(r);
    case 2: return -sin_poly_reduced_constexpr(r);
    default: return -cos_poly_reduced_constexpr(r);
    }
}
BL_FORCE_INLINE constexpr double cos_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();

    int quadrant = 0;
    double r = 0.0;
    reduce_pi_over_2_constexpr(x, quadrant, r);

    switch (quadrant)
    {
    case 0: return cos_poly_reduced_constexpr(r);
    case 1: return -sin_poly_reduced_constexpr(r);
    case 2: return -cos_poly_reduced_constexpr(r);
    default: return sin_poly_reduced_constexpr(r);
    }
}

BL_FORCE_INLINE constexpr double tan_constexpr(double x) noexcept
{
    double s{}, c{};
    sincos_constexpr(x, s, c);
    return s / c;
}
BL_FORCE_INLINE constexpr double sqrt_seed_constexpr(double x) noexcept
{
    if (!(x > 0.0) || isnan(x) || isinf(x))
        return x;

    int exp2 = frexp_exponent_constexpr(x);
    double m = ldexp_constexpr2(x, -exp2);

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

    return ldexp_constexpr2(y, exp2 / 2);
}

BL_PUSH_PRECISE
BL_FORCE_INLINE constexpr void two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    double bv = s - a;
    e = (a - (s - bv)) + (b - bv);
}
BL_FORCE_INLINE constexpr void quick_two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    e = b - (s - a);
}
BL_FORCE_INLINE constexpr void two_prod_precise_dekker(double a, double b, double& p, double& err) noexcept
{
    constexpr double split = 134217729.0;

    double a_c = a * split;
    double a_hi = a_c - (a_c - a);
    double a_lo = a - a_hi;

    double b_c = b * split;
    double b_hi = b_c - (b_c - b);
    double b_lo = b - b_hi;

    p = a * b;
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}
BL_POP_PRECISE

BL_FORCE_INLINE constexpr void two_prod_precise(double a, double b, double& p, double& err) noexcept
{
    #ifdef FMA_AVAILABLE
    if (bl::is_constant_evaluated())
    {
        two_prod_precise_dekker(a, b, p, err);
    }
    else
    {
        p = a * b;
        err = std::fma(a, b, -p);
    }
    #else
    two_prod_precise_dekker(a, b, p, err);
    #endif
}

BL_FORCE_INLINE constexpr void split_uint64_to_doubles(std::uint64_t value, double& hi, double& lo) noexcept
{
    hi = static_cast<double>(value >> 32) * 4294967296.0;
    lo = static_cast<double>(value & 0xFFFFFFFFull);
}
BL_FORCE_INLINE constexpr std::uint64_t magnitude_u64(std::int64_t value) noexcept
{
    return (value < 0) ? (std::uint64_t{0} - static_cast<std::uint64_t>(value)) : static_cast<std::uint64_t>(value);
}
BL_FORCE_INLINE constexpr void uint64_to_exact_double_pair(std::uint64_t value, double& sum, double& err) noexcept
{
    double hi{}, lo{};
    split_uint64_to_doubles(value, hi, lo);
    two_sum_precise(hi, lo, sum, err);
}

}

#endif