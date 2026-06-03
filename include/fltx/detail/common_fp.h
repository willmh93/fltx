/**
 * fltx/detail/common_fp.h - Shared low-level constexpr floating-point logic.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_COMMON_FP_INCLUDED
#define FLTX_DETAIL_COMMON_FP_INCLUDED
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "fltx/config.h"

namespace bl::detail::fp
{

inline constexpr std::uint64_t exact_double_integer_limit = 9007199254740992ull;
inline constexpr double exact_double_integer_limit_double = 9007199254740992.0;
inline constexpr double double_integer_threshold          = 4503599627370496.0;

template<class T>
inline constexpr bool is_integer_scalar_v = std::is_integral_v<std::remove_cv_t<T>> && (sizeof(std::remove_cv_t<T>) <= 8);

template<class T>
inline constexpr bool integer_type_fits_exact_double_v = std::is_integral_v<std::remove_cv_t<T>> && (sizeof(std::remove_cv_t<T>) < 8);

struct double_double
{
    double hi, lo;
};

BL_FORCE_INLINE constexpr bool isinf(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_inf bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL;
}

BL_FORCE_INLINE constexpr bool isinf(float value) noexcept
{
    static_assert(std::numeric_limits<float>::is_iec559,
        "is_inf bit-pattern check requires IEEE 754 / IEC 559 float");

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x7fffffffu) == 0x7f800000u;
}

BL_FORCE_INLINE constexpr bool isfinite(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "isfinite bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

BL_FORCE_INLINE constexpr bool isfinite(float value) noexcept
{
    static_assert(std::numeric_limits<float>::is_iec559,
        "isfinite bit-pattern check requires IEEE 754 / IEC 559 float");

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

BL_FORCE_INLINE constexpr bool isinf_or_nan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "isinf_or_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL;
}

BL_FORCE_INLINE constexpr bool iszero_or_nan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "iszero_or_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t abs_bits = std::bit_cast<std::uint64_t>(value) & 0x7fffffffffffffffULL;
    return (abs_bits - 1ULL) > 0x7fefffffffffffffULL;
}

BL_FORCE_INLINE constexpr bool iszero_or_inf_or_nan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "iszero_or_inf_or_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t abs_bits = std::bit_cast<std::uint64_t>(value) & 0x7fffffffffffffffULL;
    return (abs_bits - 1ULL) >= 0x7fefffffffffffffULL;
}

template<int bits_to_clear>
BL_FORCE_INLINE constexpr double zero_low_fraction_bits_finite(double value) noexcept
{
    static_assert(bits_to_clear >= 0 && bits_to_clear <= 52);

    if constexpr (bits_to_clear == 0)
        return value;

    if (iszero_or_inf_or_nan(value))
        return value;

    constexpr std::uint64_t fraction_mask = (std::uint64_t{ 1 } << 52) - 1ULL;
    constexpr std::uint64_t clear_mask    = ~((std::uint64_t{ 1 } << bits_to_clear) - 1ULL);

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t sign_and_exponent = bits & ~fraction_mask;
    const std::uint64_t fraction = bits & fraction_mask;
    return std::bit_cast<double>(sign_and_exponent | (fraction & clear_mask));
}

BL_FORCE_INLINE constexpr bool isnan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits     = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t abs_bits = bits & 0x7fffffffffffffffULL;
    return abs_bits > 0x7ff0000000000000ULL;
}

BL_FORCE_INLINE constexpr bool isnan(float value) noexcept
{
    static_assert(std::numeric_limits<float>::is_iec559,
        "is_nan bit-pattern check requires IEEE 754 / IEC 559 float");

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

BL_FORCE_INLINE constexpr double absd(double x) noexcept { return (x < 0.0) ? -x : x; }

BL_FORCE_INLINE constexpr int frexp_exponent(double x) noexcept
{
    if (iszero_or_inf_or_nan(x))
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

BL_FORCE_INLINE constexpr int frexp_exponent_limb(double value) noexcept
{
    if (bl::use_constexpr_math())
    {
        return frexp_exponent(value);
    }

    int exponent = 0;
    (void)std::frexp(value, &exponent);
    return exponent;
}

BL_FORCE_INLINE constexpr int highest_bit_index(std::uint64_t value) noexcept
{
    int index = -1;
    while (value != 0)
    {
        value >>= 1;
        ++index;
    }
    return index;
}

[[nodiscard]] BL_FORCE_INLINE constexpr int bit_length_u64(std::uint64_t value) noexcept
{
    int bits = 0;
    while (value != 0)
    {
        ++bits;
        value >>= 1;
    }
    return bits;
}

struct pow2_scale_info
{
    bool valid = false;
    bool negative = false;
    int exponent = 0;
};

BL_FORCE_INLINE constexpr pow2_scale_info exact_pow2_scale_info(double value) noexcept
{
    constexpr std::uint64_t sign_mask     = 0x8000000000000000ull;
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
    constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;

    const std::uint64_t bits     = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t abs_bits = bits & ~sign_mask;
    const bool negative = (bits & sign_mask) != 0;

    if (abs_bits == 0 || abs_bits >= exponent_mask)
        return { false, negative, 0 };

    const std::uint32_t exponent_bits = static_cast<std::uint32_t>((abs_bits & exponent_mask) >> 52);
    const std::uint64_t fraction = abs_bits & fraction_mask;

    if (exponent_bits != 0)
        return { fraction == 0, negative, static_cast<int>(exponent_bits) - 1023 };

    if ((fraction & (fraction - 1)) != 0)
        return { false, negative, 0 };

    return { true, negative, highest_bit_index(fraction) - 1074 };
}

BL_FORCE_INLINE constexpr bool abs_double_is_power_of_two(double value) noexcept
{
    return exact_pow2_scale_info(value).valid;
}

BL_FORCE_INLINE constexpr double scalbn(double value, int exp) noexcept
{
    if (exp == 0 || iszero_or_inf_or_nan(value))
        return value;

    constexpr std::uint64_t sign_mask     = 0x8000000000000000ull;
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
    constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;
    constexpr std::uint64_t hidden_bit    = 0x0010000000000000ull;

    const std::uint64_t bits     = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t sign     = bits & sign_mask;
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
        const int msb_index = highest_bit_index(fraction);
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

BL_FORCE_INLINE constexpr double ldexp(double value, int exp) noexcept
{
    return scalbn(value, exp);
}

BL_FORCE_INLINE constexpr double ldexp_limb(double value, int exponent) noexcept
{
    if (bl::use_constexpr_math())
    {
        return ldexp(value, exponent);
    }

    return std::ldexp(value, exponent);
}

BL_FORCE_INLINE constexpr bool signbit(double x) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    return (bits >> 63) != 0;
}

BL_FORCE_INLINE constexpr bool signbit(float x) noexcept
{
    return (std::bit_cast<std::uint32_t>(x) & 0x80000000u) != 0u;
}

BL_FORCE_INLINE constexpr double fabs(double x) noexcept
{
    return absd(x);
}

BL_FORCE_INLINE constexpr float fabs(float x) noexcept
{
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(x) & 0x7fffffffu);
}

BL_FORCE_INLINE constexpr double copysign(double magnitude, double sign_source) noexcept
{
    const std::uint64_t magnitude_bits = std::bit_cast<std::uint64_t>(magnitude) & 0x7fffffffffffffffULL;
    const std::uint64_t sign_bits = std::bit_cast<std::uint64_t>(sign_source) & 0x8000000000000000ULL;
    return std::bit_cast<double>(magnitude_bits | sign_bits);
}

BL_FORCE_INLINE constexpr float copysign(float magnitude, float sign_source) noexcept
{
    const std::uint32_t magnitude_bits = std::bit_cast<std::uint32_t>(magnitude) & 0x7fffffffu;
    const std::uint32_t sign_bits = std::bit_cast<std::uint32_t>(sign_source) & 0x80000000u;
    return std::bit_cast<float>(magnitude_bits | sign_bits);
}

BL_FORCE_INLINE constexpr double floor(double x) noexcept
{
    if (iszero_or_inf_or_nan(x))
        return x;

    const double ax = absd(x);
    if (ax >= double_integer_threshold)
        return x;

    const long long i = static_cast<long long>(x);
    double di = static_cast<double>(i);
    if (di > x) di -= 1.0;
    if (di == 0.0) return signbit(x) ? -0.0 : 0.0;
    return di;
}

BL_FORCE_INLINE constexpr double ceil(double x) noexcept
{
    if (iszero_or_inf_or_nan(x))
        return x;

    const double ax = absd(x);
    if (ax >= double_integer_threshold)
        return x;

    const long long i = static_cast<long long>(x);
    double di = static_cast<double>(i);
    if (di < x)
        di += 1.0;
    if (di == 0.0)
        return signbit(x) ? -0.0 : 0.0;
    return di;
}

BL_FORCE_INLINE constexpr double trunc(double x) noexcept
{
    return signbit(x) ? ceil(x) : floor(x);
}

BL_FORCE_INLINE constexpr bool double_integer_is_odd(double x) noexcept
{
    const double ax = absd(x);
    if (!isfinite(x) || ax < 1.0 || ax >= exact_double_integer_limit_double)
        return false;
    const long long i = static_cast<long long>(x);
    return (i & 1ll) != 0;
}

BL_PUSH_PRECISE
BL_FORCE_INLINE constexpr void two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    double bv = s - a;
    e = (a - (s - bv)) + (b - bv);
}

BL_FORCE_INLINE constexpr void two_diff_precise(double a, double b, double& s, double& e) noexcept
{
    s = a - b;
    double bv = s - a;
    e = (a - (s - bv)) - (b + bv);
}

BL_FORCE_INLINE constexpr void quick_two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    e = b - (s - a);
}

BL_FORCE_INLINE constexpr void two_prod_precise_dekker(double a, double b, double& p, double& err) noexcept
{
    constexpr double split = 134217729.0;

    double a_c  = a * split;
    double a_hi = a_c - (a_c - a);
    double a_lo = a - a_hi;

    double b_c  = b * split;
    double b_hi = b_c - (b_c - b);
    double b_lo = b - b_hi;

    p = a * b;
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}
BL_POP_PRECISE

BL_FORCE_INLINE constexpr void two_prod_precise(double a, double b, double& p, double& err) noexcept
{
    #ifdef FMA_AVAILABLE
    if (bl::is_constant_evaluated() || bl::use_constexpr_parity())
    {
        two_prod_precise_dekker(a, b, p, err);
    }
    else
    {
        p = a * b;
        #if defined(__clang__) || defined(__GNUC__)
        err = __builtin_fma(a, b, -p);
        #else
        err = std::fma(a, b, -p);
        #endif
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

BL_FORCE_INLINE constexpr void int64_to_exact_double_pair(std::int64_t value, double& sum, double& err) noexcept
{
    uint64_to_exact_double_pair(magnitude_u64(value), sum, err);
    if (value < 0)
    {
        sum = -sum;
        err = -err;
    }
}

template<class T> [[nodiscard]]
BL_FORCE_INLINE constexpr bool integer_fits_exact_double(T value) noexcept
{
    using clean_t = std::remove_cv_t<T>;
    static_assert(is_integer_scalar_v<clean_t>);

    if constexpr (sizeof(clean_t) < 8)
        return true;
    else if constexpr (std::is_signed_v<clean_t>)
        return magnitude_u64(static_cast<std::int64_t>(value)) <= exact_double_integer_limit;
    else
        return static_cast<std::uint64_t>(value) <= exact_double_integer_limit;
}

} // namespace bl::detail::fp

#endif
