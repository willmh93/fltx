#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <ios>
#include <limits>
#include <numbers>
#include <ostream>
#include <string>
#include <cstring>
#include <utility>
#include <type_traits>
#include <cstdint>

#ifndef FMA_AVAILABLE
#ifndef __EMSCRIPTEN__
#if defined(__FMA__) || defined(__FMA4__) || defined(_MSC_VER) || defined(__clang__)
#define FMA_AVAILABLE
#endif
#endif
#endif

#ifndef FORCE_INLINE
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif
#endif

#ifndef NO_INLINE
#if defined(_MSC_VER)
#define NO_INLINE [[msvc::noinline]]
#elif defined(__clang__)
#define NO_INLINE [[clang::noinline]]
#elif defined(__GNUC__)
#define NO_INLINE __attribute__((noinline))
#else
#define NO_INLINE
#endif
#endif

#ifndef BL_LIKELY
#if defined(__clang__) || defined(__GNUC__)
#define BL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define BL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BL_LIKELY(x)   (x)
#define BL_UNLIKELY(x) (x)
#endif
#endif

#ifndef BL_FAST_MATH
#if defined(__FAST_MATH__)
#define BL_FAST_MATH
#elif defined(_MSC_VER) && defined(_M_FP_FAST)
#define BL_FAST_MATH
#endif
#endif

#ifndef BL_PRINT_NOINLINE
#if defined(_MSC_VER) && defined(BL_FAST_MATH)
#define BL_PRINT_NOINLINE __declspec(noinline)
#else
#define BL_PRINT_NOINLINE
#endif
#endif

#if defined(_MSC_VER)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  __pragma(float_control(precise, on, push)) \
                         __pragma(fp_contract(off))
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   __pragma(float_control(pop))
#endif
#elif defined(__EMSCRIPTEN__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("clang fp reassociate(on)")  \
                         _Pragma("clang fp contract(fast)")
#endif
#elif defined(__clang__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("clang fp reassociate(on)")  \
                         _Pragma("clang fp contract(fast)")
#endif
#elif defined(__GNUC__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("GCC push_options")               \
                         _Pragma("GCC optimize(\"no-fast-math\")") \
                         _Pragma("STDC FP_CONTRACT OFF")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("GCC pop_options")
#endif
#else
#define BL_PUSH_PRECISE
#define BL_POP_PRECISE
#endif

// todo: 
// change bl::fltx::common to  bl::fltx::common
// put numeric_types in       bl::fltx
// constexpr_dispatch in      bl::

namespace bl
{
    // include in bl namespace in case user includes them f128/f256 manually
    enum struct FloatType
    {
        F32,
        F64,
        F128,
        F256,
        COUNT
    };

    static inline const char* FloatTypeNames[(int)FloatType::COUNT] =
    {
        "F32",
        "F64",
        "F128",
        "F256"
    };
}

namespace bl::fltx::common {

enum class format_kind : unsigned char { general, fixed_frac, scientific_frac, scientific_sig };

template<typename BigUInt>
struct parse_token
{
    using coeff_type = BigUInt;

    coeff_type coeff;
    int frac_digits = 0;
    int sig_digits = 0;
    int exp10 = 0;
    bool any_digit = false;
    bool seen_nonzero = false;
};

namespace fp {

FORCE_INLINE constexpr bool isinf(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_inf bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL;
}
FORCE_INLINE constexpr bool isfinite(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "isfinite bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

template<int bits_to_clear>
FORCE_INLINE constexpr double zero_low_fraction_bits_finite(double value) noexcept
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
FORCE_INLINE constexpr bool isnan(double value) noexcept
{
    static_assert(std::numeric_limits<double>::is_iec559,
        "is_nan bit-pattern check requires IEEE 754 / IEC 559 double");

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t abs_bits = bits & 0x7fffffffffffffffULL;
    return abs_bits > 0x7ff0000000000000ULL;
}
FORCE_INLINE constexpr double absd(double x) noexcept { return (x < 0.0) ? -x : x; }

// faster implementation
FORCE_INLINE constexpr int    frexp_exponent_constexpr(double x) noexcept
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
    return e;
}
FORCE_INLINE constexpr int    highest_bit_index_constexpr(std::uint64_t value) noexcept
{
    int index = -1;
    while (value != 0)
    {
        value >>= 1;
        ++index;
    }
    return index;
}
FORCE_INLINE constexpr double scalbn_constexpr2(double value, int exp) noexcept
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
FORCE_INLINE constexpr double ldexp_constexpr2(double value, int exp) noexcept
{
    return scalbn_constexpr2(value, exp);
}
FORCE_INLINE constexpr double log_series_reduced_constexpr(double z) noexcept
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
FORCE_INLINE constexpr double log_constexpr(double x) noexcept
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
FORCE_INLINE constexpr double log1p_constexpr(double x) noexcept
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


FORCE_INLINE constexpr bool   signbit_constexpr(double x) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    return (bits >> 63) != 0;
}
FORCE_INLINE constexpr double fabs_constexpr(double x) noexcept
{
    return absd(x);
}
FORCE_INLINE constexpr double floor_constexpr(double x) noexcept
{
    if (isnan(x) || isinf(x) || x == 0.0)
        return x;

    const double ax = absd(x);
    if (ax >= 4503599627370496.0)
        return x;

    const long long i = static_cast<long long>(x);
    double di = static_cast<double>(i);
    if (di > x)
        di -= 1.0;
    if (di == 0.0)
        return signbit_constexpr(x) ? -0.0 : 0.0;
    return di;
}
FORCE_INLINE constexpr double ceil_constexpr(double x) noexcept
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
FORCE_INLINE constexpr double trunc_constexpr(double x) noexcept
{
    return signbit_constexpr(x) ? ceil_constexpr(x) : floor_constexpr(x);
}
FORCE_INLINE constexpr long long llround_constexpr(double x) noexcept
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
FORCE_INLINE constexpr double fmod_constexpr(double x, double y) noexcept
{
    if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(y) || x == 0.0)
        return x;

    const double q = trunc_constexpr(x / y);
    return x - q * y;
}
FORCE_INLINE constexpr bool   double_integer_is_odd(double x) noexcept
{
    const double ax = absd(x);
    if (!isfinite(x) || ax < 1.0 || ax >= 9007199254740992.0)
        return false;
    const long long i = static_cast<long long>(x);
    return (i & 1ll) != 0;
}
FORCE_INLINE constexpr double nearbyint_ties_even(double x) noexcept
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

FORCE_INLINE constexpr double atan_series_constexpr(double x) noexcept
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
FORCE_INLINE constexpr double atan_constexpr(double x) noexcept
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
NO_INLINE constexpr double atan2_constexpr(double y, double x) noexcept
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
NO_INLINE constexpr void   sincos_constexpr(double x, double& s, double& c) noexcept
{
    constexpr double pi_2_hi = 0x1.921fb54442d18p+0;
    constexpr double pi_2_lo = 0x1.1a62633145c07p-54;
    constexpr double inv_pi_2 = 0x1.45f306dc9c883p-1;

    if (isnan(x) || isinf(x))
    {
        s = std::numeric_limits<double>::quiet_NaN();
        c = s;
        return;
    }

    const double n = nearbyint_ties_even(x * inv_pi_2);
    double r = (x - n * pi_2_hi) - n * pi_2_lo;

    if (r > 0x1.921fb54442d18p-1)
        r -= pi_2_hi;
    else if (r < -0x1.921fb54442d18p-1)
        r += pi_2_hi;

    const double t = r * r;

    double sp = 2.8114572543455206e-15;
    sp = sp * t - 7.6471637318198164e-13;
    sp = sp * t + 1.6059043836821615e-10;
    sp = sp * t - 2.5052108385441719e-8;
    sp = sp * t + 2.7557319223985891e-6;
    sp = sp * t - 1.9841269841269841e-4;
    sp = sp * t + 8.3333333333333332e-3;
    sp = sp * t - 1.6666666666666666e-1;
    const double sr = r + r * t * sp;

    double cp = -7.6471637318198164e-13;
    cp = cp * t + 1.6059043836821615e-10;
    cp = cp * t - 2.7557319223985891e-8;
    cp = cp * t + 2.4801587301587302e-5;
    cp = cp * t - 1.3888888888888889e-3;
    cp = cp * t + 4.1666666666666664e-2;
    cp = cp * t - 5.0e-1;
    const double cr = 1.0 + t * cp;

    const int q = static_cast<int>(n) & 3;
    switch (q)
    {
    case 0: s = sr; c = cr; break;
    case 1: s = cr; c = -sr; break;
    case 2: s = -sr; c = -cr; break;
    default: s = -cr; c = sr; break;
    }
}
FORCE_INLINE constexpr double sin_constexpr(double x) noexcept
{
    double s{}, c{};
    sincos_constexpr(x, s, c);
    return s;
}
FORCE_INLINE constexpr double cos_constexpr(double x) noexcept
{
    double s{}, c{};
    sincos_constexpr(x, s, c);
    return c;
}
FORCE_INLINE constexpr double tan_constexpr(double x) noexcept
{
    double s{}, c{};
    sincos_constexpr(x, s, c);
    return s / c;
}
FORCE_INLINE constexpr double sqrt_seed_constexpr(double x) noexcept
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
FORCE_INLINE constexpr void two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    double bv = s - a;
    e = (a - (s - bv)) + (b - bv);
}
FORCE_INLINE constexpr void quick_two_sum_precise(double a, double b, double& s, double& e) noexcept
{
    s = a + b;
    e = b - (s - a);
}
FORCE_INLINE constexpr void two_prod_precise_dekker(double a, double b, double& p, double& err) noexcept
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

#ifdef FMA_AVAILABLE
FORCE_INLINE double fma1(double a, double b, double c) noexcept
{
    return std::fma(a, b, c);
}
#endif

FORCE_INLINE constexpr void two_prod_precise(double a, double b, double& p, double& err) noexcept
{
    #ifdef FMA_AVAILABLE
    if (std::is_constant_evaluated())
    {
        two_prod_precise_dekker(a, b, p, err);
    }
    else
    {
        p = a * b;
        err = fma1(a, b, -p);
    }
    #else
    two_prod_precise_dekker(a, b, p, err);
    #endif
}

FORCE_INLINE constexpr void split_uint64_to_doubles(std::uint64_t value, double& hi, double& lo) noexcept
{
    hi = static_cast<double>(value >> 32) * 4294967296.0;
    lo = static_cast<double>(value & 0xFFFFFFFFull);
}
FORCE_INLINE constexpr std::uint64_t magnitude_u64(std::int64_t value) noexcept
{
    return (value < 0) ? (std::uint64_t{0} - static_cast<std::uint64_t>(value)) : static_cast<std::uint64_t>(value);
}

} // namespace fp

namespace exact_decimal {

struct biguint
{
    static constexpr int max_words = 320;

    std::uint32_t words[max_words]{};
    int size = 0;

    constexpr biguint() noexcept = default;

    constexpr explicit biguint(std::uint64_t value) noexcept
    {
        if (value == 0)
            return;

        words[0] = static_cast<std::uint32_t>(value);
        size = 1;

        const std::uint32_t hi = static_cast<std::uint32_t>(value >> 32);
        if (hi != 0)
        {
            words[1] = hi;
            size = 2;
        }
    }

    constexpr void clear() noexcept
    {
        size = 0;
    }

    [[nodiscard]] constexpr bool is_zero() const noexcept
    {
        return size == 0;
    }

    [[nodiscard]] constexpr bool is_odd() const noexcept
    {
        return size != 0 && (words[0] & 1u) != 0;
    }

    constexpr void trim() noexcept
    {
        while (size > 0 && words[size - 1] == 0)
            --size;
    }

    [[nodiscard]] constexpr int bit_length() const noexcept
    {
        if (size == 0)
            return 0;

        std::uint32_t msw = words[size - 1];
        int bits = 32 * (size - 1);
        while (msw != 0)
        {
            ++bits;
            msw >>= 1;
        }
        return bits;
    }

    [[nodiscard]] constexpr bool get_bit(int index) const noexcept
    {
        if (index < 0)
            return false;

        const int word_index = index >> 5;
        if (word_index >= size)
            return false;

        return ((words[word_index] >> (index & 31)) & 1u) != 0;
    }

    [[nodiscard]] constexpr std::uint64_t get_bits(int start, int count) const noexcept
    {
        std::uint64_t value = 0;
        for (int i = 0; i < count; ++i)
        {
            if (get_bit(start + i))
                value |= (std::uint64_t{1} << i);
        }
        return value;
    }

    constexpr void set_bit(int index) noexcept
    {
        if (index < 0)
            return;

        const int word_index = index >> 5;
        if (word_index >= max_words)
            return;

        while (size <= word_index)
            words[size++] = 0;

        words[word_index] |= (1u << (index & 31));
    }

    constexpr void add_small(std::uint32_t value) noexcept
    {
        std::uint64_t carry = value;
        int i = 0;

        while (carry != 0)
        {
            if (i == size)
            {
                if (size >= max_words)
                    return;
                words[size++] = 0;
            }

            const std::uint64_t sum = static_cast<std::uint64_t>(words[i]) + carry;
            words[i] = static_cast<std::uint32_t>(sum);
            carry = sum >> 32;
            ++i;
        }
    }

    constexpr void add_inplace(const biguint& other) noexcept
    {
        if (size < other.size)
        {
            while (size < other.size)
                words[size++] = 0;
        }

        std::uint64_t carry = 0;
        int i = 0;
        for (; i < other.size; ++i)
        {
            const std::uint64_t sum = static_cast<std::uint64_t>(words[i]) + other.words[i] + carry;
            words[i] = static_cast<std::uint32_t>(sum);
            carry = sum >> 32;
        }

        while (carry != 0)
        {
            if (i == size)
            {
                if (size >= max_words)
                    return;
                words[size++] = 0;
            }

            const std::uint64_t sum = static_cast<std::uint64_t>(words[i]) + carry;
            words[i] = static_cast<std::uint32_t>(sum);
            carry = sum >> 32;
            ++i;
        }
    }

    constexpr void mul_small(std::uint32_t factor) noexcept
    {
        if (factor == 0 || size == 0)
        {
            size = 0;
            return;
        }

        std::uint64_t carry = 0;
        for (int i = 0; i < size; ++i)
        {
            const std::uint64_t prod = static_cast<std::uint64_t>(words[i]) * factor + carry;
            words[i] = static_cast<std::uint32_t>(prod);
            carry = prod >> 32;
        }

        if (carry != 0 && size < max_words)
            words[size++] = static_cast<std::uint32_t>(carry);
    }

    constexpr std::uint32_t div_small(std::uint32_t divisor) noexcept
    {
        std::uint64_t rem = 0;
        for (int i = size - 1; i >= 0; --i)
        {
            const std::uint64_t cur = (rem << 32) | words[i];
            words[i] = static_cast<std::uint32_t>(cur / divisor);
            rem = cur % divisor;
        }
        trim();
        return static_cast<std::uint32_t>(rem);
    }

    constexpr void shl1() noexcept
    {
        if (size == 0)
            return;

        std::uint64_t carry = 0;
        for (int i = 0; i < size; ++i)
        {
            const std::uint64_t cur = (static_cast<std::uint64_t>(words[i]) << 1) | carry;
            words[i] = static_cast<std::uint32_t>(cur);
            carry = cur >> 32;
        }

        if (carry != 0 && size < max_words)
            words[size++] = static_cast<std::uint32_t>(carry);
    }

    constexpr void shr1() noexcept
    {
        if (size == 0)
            return;

        std::uint32_t carry = 0;
        for (int i = size - 1; i >= 0; --i)
        {
            const std::uint32_t next_carry = static_cast<std::uint32_t>(words[i] & 1u);
            words[i] = (words[i] >> 1) | (carry << 31);
            carry = next_carry;
        }
        trim();
    }

    constexpr void shl_bits(int bits) noexcept
    {
        if (bits <= 0 || size == 0)
            return;

        const int word_shift = bits >> 5;
        const int bit_shift = bits & 31;

        if (word_shift >= max_words)
        {
            size = 0;
            return;
        }

        const int old_size = size;
        const int src_count = std::min(old_size, max_words - word_shift);
        if (src_count <= 0)
        {
            size = 0;
            return;
        }

        if (bit_shift == 0)
        {
            for (int i = src_count - 1; i >= 0; --i)
                words[i + word_shift] = words[i];

            for (int i = 0; i < word_shift; ++i)
                words[i] = 0;

            size = src_count + word_shift;
            return;
        }

        std::uint32_t out[max_words]{};

        for (int i = 0; i < word_shift; ++i)
            out[i] = 0;

        std::uint32_t carry = 0;
        int dst = word_shift;

        for (int i = 0; i < src_count; ++i, ++dst)
        {
            const std::uint32_t word = words[i];
            out[dst] = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(word) << bit_shift) | carry);
            carry = static_cast<std::uint32_t>(word >> (32 - bit_shift));
        }

        int new_size = src_count + word_shift;
        if (carry != 0 && new_size < max_words)
        {
            out[new_size] = carry;
            ++new_size;
        }

        for (int i = 0; i < new_size; ++i)
            words[i] = out[i];

        size = new_size;
        trim();
    }

    [[nodiscard]] constexpr int compare(const biguint& other) const noexcept
    {
        if (size < other.size) return -1;
        if (size > other.size) return 1;

        for (int i = size - 1; i >= 0; --i)
        {
            if (words[i] < other.words[i]) return -1;
            if (words[i] > other.words[i]) return 1;
        }
        return 0;
    }

    constexpr void sub_inplace(const biguint& other) noexcept
    {
        std::uint64_t borrow = 0;
        for (int i = 0; i < size; ++i)
        {
            const std::uint64_t a = words[i];
            const std::uint64_t b = static_cast<std::uint64_t>(i < other.size ? other.words[i] : 0u) + borrow;

            if (a < b)
            {
                words[i] = static_cast<std::uint32_t>((std::uint64_t{1} << 32) + a - b);
                borrow = 1;
            }
            else
            {
                words[i] = static_cast<std::uint32_t>(a - b);
                borrow = 0;
            }
        }
        trim();
    }
};

[[nodiscard]] constexpr inline int compare(const biguint& a, const biguint& b) noexcept
{
    return a.compare(b);
}

[[nodiscard]] constexpr inline biguint shifted(biguint v, int bits) noexcept
{
    v.shl_bits(bits);
    return v;
}

[[nodiscard]] constexpr inline int high_word_index_shifted(const biguint& value, int bits) noexcept
{
    if (value.is_zero())
        return -1;

    const int word_shift = bits >> 5;
    const int bit_shift = bits & 31;
    const bool extra_word = bit_shift != 0 && (value.words[value.size - 1] >> (32 - bit_shift)) != 0;
    return word_shift + value.size - 1 + (extra_word ? 1 : 0);
}

[[nodiscard]] constexpr inline std::uint32_t shifted_word_at(const biguint& value, int index, int bits) noexcept
{
    if (index < 0 || value.is_zero())
        return 0;

    const int word_shift = bits >> 5;
    const int bit_shift = bits & 31;
    const int src = index - word_shift;

    if (bit_shift == 0)
        return (src >= 0 && src < value.size) ? value.words[src] : 0u;

    std::uint32_t out = 0;
    if (src >= 0 && src < value.size)
        out |= static_cast<std::uint32_t>(static_cast<std::uint64_t>(value.words[src]) << bit_shift);
    if (src - 1 >= 0 && src - 1 < value.size)
        out |= static_cast<std::uint32_t>(value.words[src - 1] >> (32 - bit_shift));
    return out;
}

[[nodiscard]] constexpr inline int compare_shifted(const biguint& a, const biguint& b, int bits) noexcept
{
    const int a_hi = a.size - 1;
    const int b_hi = high_word_index_shifted(b, bits);
    if (a_hi < b_hi) return -1;
    if (a_hi > b_hi) return 1;

    for (int i = a_hi; i >= 0; --i)
    {
        const std::uint32_t bw = shifted_word_at(b, i, bits);
        if (a.words[i] < bw) return -1;
        if (a.words[i] > bw) return 1;
    }
    return 0;
}

constexpr inline void sub_shifted_inplace(biguint& a, const biguint& b, int bits) noexcept
{
    const int a_size = a.size;
    std::uint64_t borrow = 0;
    for (int i = 0; i < a_size; ++i)
    {
        const std::uint64_t bi = shifted_word_at(b, i, bits);
        const std::uint64_t sub = bi + borrow;
        const std::uint64_t ai = a.words[i];
        if (ai < sub)
        {
            a.words[i] = static_cast<std::uint32_t>((std::uint64_t{1} << 32) + ai - sub);
            borrow = 1;
        }
        else
        {
            a.words[i] = static_cast<std::uint32_t>(ai - sub);
            borrow = 0;
        }
    }
    a.trim();
}

constexpr inline void mod_shift_subtract(const biguint& numerator, const biguint& denominator, biguint& remainder) noexcept
{
    remainder = numerator;
    if (denominator.is_zero())
        return;

    while (remainder.compare(denominator) >= 0)
    {
        int shift = (remainder.bit_length() - 1) - (denominator.bit_length() - 1);
        if (shift > 0 && compare_shifted(remainder, denominator, shift) < 0)
            --shift;
        sub_shifted_inplace(remainder, denominator, shift);
    }
}

constexpr inline biguint mul_big(const biguint& a, const biguint& b) noexcept
{
    biguint out;
    if (a.is_zero() || b.is_zero())
        return out;

    out.size = std::min(a.size + b.size, biguint::max_words);
    for (int i = 0; i < out.size; ++i)
        out.words[i] = 0;

    for (int i = 0; i < a.size; ++i)
    {
        std::uint64_t carry = 0;
        const int jmax = std::min(b.size, biguint::max_words - i);
        for (int j = 0; j < jmax; ++j)
        {
            const int k = i + j;
            const std::uint64_t cur =
                static_cast<std::uint64_t>(out.words[k]) +
                static_cast<std::uint64_t>(a.words[i]) * static_cast<std::uint64_t>(b.words[j]) +
                carry;
            out.words[k] = static_cast<std::uint32_t>(cur);
            carry = cur >> 32;
        }

        int k = i + jmax;
        while (carry != 0 && k < out.size)
        {
            const std::uint64_t cur = static_cast<std::uint64_t>(out.words[k]) + carry;
            out.words[k] = static_cast<std::uint32_t>(cur);
            carry = cur >> 32;
            ++k;
        }
    }

    out.trim();
    return out;
}

[[nodiscard]] constexpr inline biguint pow5_big(int exponent) noexcept
{
    biguint out{1};
    for (int i = 0; i < exponent; ++i)
        out.mul_small(5);
    return out;
}

[[nodiscard]] constexpr inline biguint pow10_big(int exponent) noexcept
{
    biguint out{1};
    for (int i = 0; i < exponent; ++i)
        out.mul_small(10);
    return out;
}

constexpr inline void divmod_bitwise(const biguint& numerator, const biguint& denominator, biguint& quotient, biguint& remainder) noexcept
{
    quotient.clear();
    remainder.clear();

    if (denominator.is_zero())
        return;

    const int nbits = numerator.bit_length();
    for (int i = nbits - 1; i >= 0; --i)
    {
        remainder.shl1();
        if (numerator.get_bit(i))
            remainder.add_small(1);

        if (remainder.compare(denominator) >= 0)
        {
            remainder.sub_inplace(denominator);
            quotient.set_bit(i);
        }
    }

    quotient.trim();
    remainder.trim();
}

[[nodiscard]] constexpr inline int floor_log2_ratio(const biguint& numerator, const biguint& denominator) noexcept
{
    int k = (numerator.bit_length() - 1) - (denominator.bit_length() - 1);

    if (k >= 0)
    {
        biguint shifted_den = denominator;
        shifted_den.shl_bits(k);
        if (numerator.compare(shifted_den) < 0)
            --k;
    }
    else
    {
        biguint shifted_num = numerator;
        shifted_num.shl_bits(-k);
        if (shifted_num.compare(denominator) < 0)
            --k;
    }

    return k;
}

struct signed_biguint
{
    biguint mag;
    bool neg = false;
};

constexpr inline void add_signed(signed_biguint& acc, biguint term, bool term_neg) noexcept
{
    if (acc.mag.is_zero())
    {
        acc.mag = term;
        acc.neg = term_neg;
        return;
    }

    if (acc.neg == term_neg)
    {
        acc.mag.add_inplace(term);
        return;
    }

    const int cmp = acc.mag.compare(term);
    if (cmp == 0)
    {
        acc.mag.clear();
        acc.neg = false;
        return;
    }

    if (cmp > 0)
    {
        acc.mag.sub_inplace(term);
        return;
    }

    term.sub_inplace(acc.mag);
    acc.mag = term;
    acc.neg = term_neg;
}

[[nodiscard]] constexpr inline std::uint64_t decompose_double_mantissa(double x, int& exponent, bool& neg) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    neg = (bits >> 63) != 0;
    const std::uint64_t frac = bits & ((std::uint64_t{1} << 52) - 1);
    const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
    if (exp_bits == 0)
    {
        exponent = -1074;
        return frac;
    }
    exponent = static_cast<int>(exp_bits) - 1023 - 52;
    return (std::uint64_t{1} << 52) | frac;
}

[[nodiscard]] constexpr inline int compare_scaled_with_pow10exp(const biguint& mag, int bin_exp, int dec_exp) noexcept
{
    if (dec_exp >= 0)
    {
        const biguint p10 = pow10_big(dec_exp);
        if (bin_exp >= 0)
            return compare(shifted(mag, bin_exp), p10);
        return compare(mag, shifted(p10, -bin_exp));
    }

    biguint lhs = mag;
    for (int i = 0; i < -dec_exp; ++i)
        lhs.mul_small(10);

    if (bin_exp >= 0)
    {
        lhs.shl_bits(bin_exp);
        return compare(lhs, biguint{ 1 });
    }

    return compare(lhs, shifted(biguint{ 1 }, -bin_exp));
}

[[nodiscard]] inline std::string to_decimal_string(biguint value)
{
    if (value.is_zero())
        return "0";

    std::uint32_t chunks[biguint::max_words * 2];
    int chunk_count = 0;
    while (!value.is_zero() && chunk_count < static_cast<int>(sizeof(chunks) / sizeof(chunks[0])))
        chunks[chunk_count++] = value.div_small(1000000000u);

    std::string out = std::to_string(chunks[chunk_count - 1]);
    for (int i = chunk_count - 2; i >= 0; --i)
    {
        std::string part = std::to_string(chunks[i]);
        out.append(static_cast<std::size_t>(9 - part.size()), '0');
        out += part;
    }
    return out;
}

template<class Traits>
[[nodiscard]] inline bool exact_scientific_digits(const typename Traits::value_type& x, int sig, std::string& digits, int& exp10)
{
    int common_exp = std::numeric_limits<int>::max();
    bool have_term = false;

    for (int i = 0; i < Traits::limb_count; ++i)
    {
        const double limb = Traits::limb(x, i);
        if (limb == 0.0)
            continue;

        int exponent = 0;
        bool limb_neg = false;
        const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
        if (mantissa == 0)
            continue;

        common_exp = std::min(common_exp, exponent);
        have_term = true;
    }

    if (!have_term)
        return false;

    signed_biguint acc{};
    for (int i = 0; i < Traits::limb_count; ++i)
    {
        const double limb = Traits::limb(x, i);
        if (limb == 0.0)
            continue;

        int exponent = 0;
        bool limb_neg = false;
        const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
        if (mantissa == 0)
            continue;

        biguint term{ mantissa };
        term.shl_bits(exponent - common_exp);
        add_signed(acc, term, limb_neg);
    }

    if (acc.mag.is_zero() || acc.neg)
        return false;

    const double log10_2 = 0.30102999566398119521373889472449;
    int approx = static_cast<int>(std::floor((acc.mag.bit_length() - 1 + common_exp) * log10_2));

    int guard = 0;
    while (compare_scaled_with_pow10exp(acc.mag, common_exp, approx) < 0)
    {
        --approx;
        if (++guard > 16)
            break;
    }

    guard = 0;
    while (compare_scaled_with_pow10exp(acc.mag, common_exp, approx + 1) >= 0)
    {
        ++approx;
        if (++guard > 16)
            break;
    }

    exp10 = approx;

    biguint num = acc.mag;
    biguint den{1};
    if (common_exp >= 0)
        num.shl_bits(common_exp);
    else
        den.shl_bits(-common_exp);

    const int scale10 = sig - 1 - exp10;
    if (scale10 >= 0)
    {
        for (int i = 0; i < scale10; ++i)
            num.mul_small(10);
    }
    else
    {
        for (int i = 0; i < -scale10; ++i)
            den.mul_small(10);
    }

    biguint q;
    biguint r;
    divmod_bitwise(num, den, q, r);
    if (!r.is_zero())
    {
        biguint twice_r = r;
        twice_r.shl1();
        const int cmp = compare(twice_r, den);
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    const biguint limit = pow10_big(sig);
    if (compare(q, limit) >= 0)
    {
        q.div_small(10);
        ++exp10;
    }

    digits = to_decimal_string(q);
    if (static_cast<int>(digits.size()) < sig)
        digits.insert(digits.begin(), sig - static_cast<int>(digits.size()), '0');

    return true;
}

template<class Traits>
constexpr inline typename Traits::value_type exact_decimal_to_value(const biguint& coeff, int dec_exp, bool neg) noexcept
{
    if (coeff.is_zero())
        return Traits::zero(neg);

    biguint numerator = coeff;
    biguint denominator{ 1 };
    int bin_exp = 0;

    if (dec_exp >= 0)
    {
        numerator = mul_big(coeff, pow5_big(dec_exp));
        bin_exp = dec_exp;
    }
    else
    {
        denominator = pow5_big(-dec_exp);
        bin_exp = dec_exp;
    }

    int ratio_exp = floor_log2_ratio(numerator, denominator);

    biguint scaled_num = numerator;
    biguint scaled_den = denominator;
    const int shift = (Traits::significand_bits - 1) - ratio_exp;
    if (shift >= 0)
        scaled_num.shl_bits(shift);
    else
        scaled_den.shl_bits(-shift);

    biguint q;
    biguint r;
    divmod_bitwise(scaled_num, scaled_den, q, r);

    if (!r.is_zero())
    {
        biguint twice_r = r;
        twice_r.shl1();
        const int cmp = compare(twice_r, scaled_den);
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    if (q.bit_length() > Traits::significand_bits)
    {
        q.shr1();
        ++ratio_exp;
    }

    const int e2 = bin_exp + ratio_exp;
    if (e2 > 1023)
        return Traits::infinity(neg);
    if (e2 < -1074)
        return Traits::zero(neg);

    return Traits::pack_from_significand(q, e2, neg);
}

} // namespace exact_decimal

template<typename CharsResult, typename Writer>
NO_INLINE void write_chars_to_string(std::string& out, std::size_t cap, Writer&& writer)
{
    out.resize(cap);
    char* first = out.data();
    auto r = std::forward<Writer>(writer)(first, first + out.size());
    if (!r.ok)
    {
        out.clear();
        return;
    }
    out.resize(static_cast<std::size_t>(r.ptr - first));
}

FORCE_INLINE constexpr bool valid_float_string(const char* s) noexcept
{
    for (const char* p = s; *p; ++p)
    {
        const char c = *p;
        if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+'))
            return false;
    }
    return true;
}

FORCE_INLINE constexpr unsigned char ascii_lower(char c) noexcept
{
    return static_cast<unsigned char>((c >= 'A' && c <= 'Z') ? (c | 0x20) : c);
}

FORCE_INLINE constexpr const char* skip_ascii_space(const char* p) noexcept
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v')
        ++p;
    return p;
}

inline NO_INLINE void ensure_decimal_point(std::string& s)
{
    const std::size_t e = s.find_first_of("eE");
    const std::size_t d = s.find('.');
    if (d != std::string::npos && (e == std::string::npos || d < e))
        return;
    if (e == std::string::npos)
        s.push_back('.');
    else
        s.insert(e, ".");
}

inline NO_INLINE void apply_stream_decorations(std::string& s, bool showpos, bool uppercase)
{
    if (showpos && (s.empty() || s[0] != '-'))
        s.insert(s.begin(), '+');
    if (!uppercase)
        return;
    for (char& c : s)
    {
        if (c == 'e')
            c = 'E';
        else if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - ('a' - 'A'));
    }
}

template<class Traits>
inline const char* special_text(const typename Traits::value_type& x, bool uppercase = false) noexcept
{
    if (Traits::isnan(x))
        return uppercase ? "NAN" : "nan";
    if (!Traits::isinf(x))
        return nullptr;
    return Traits::is_negative(x) ? (uppercase ? "-INF" : "-inf") : (uppercase ? "INF" : "inf");
}

template<class Traits>
inline bool assign_special_string(std::string& out, const typename Traits::value_type& x, bool uppercase = false) noexcept
{
    if (const char* text = special_text<Traits>(x, uppercase))
    {
        out = text;
        return true;
    }
    return false;
}

template<class Traits>
inline bool write_stream_special(std::ostream& os, const typename Traits::value_type& x, bool showpos, bool uppercase)
{
    const char* text = special_text<Traits>(x, uppercase);
    if (!text)
        return false;
    if (showpos && text[0] != '-')
        os << '+';
    os << text;
    return true;
}

template<class Traits>
inline void format_to_string(std::string& out, const typename Traits::value_type& x, int precision, format_kind kind, bool strip_trailing_zeros = false)
{
    if (assign_special_string<Traits>(out, x))
        return;

    if (kind == format_kind::scientific_sig)
    {
        if (precision < 1)
            precision = 1;
    }
    else if (precision < 0)
    {
        precision = 0;
    }

    const std::size_t cap = ((kind == format_kind::general || kind == format_kind::fixed_frac) ? 1u + 309u : 1u + 1u)
        + 1u + static_cast<std::size_t>(precision) + 32u;

    write_chars_to_string<typename Traits::chars_result>(out, cap, [&](char* first, char* last) {
        switch (kind)
        {
        case format_kind::general:
            return Traits::to_chars_general(first, last, x, precision, strip_trailing_zeros);
        case format_kind::fixed_frac:
            return Traits::to_chars_fixed(first, last, x, precision, strip_trailing_zeros);
        case format_kind::scientific_frac:
            return Traits::to_chars_scientific_frac(first, last, x, precision, strip_trailing_zeros);
        case format_kind::scientific_sig:
            return Traits::to_chars_scientific_sig(first, last, x, precision, strip_trailing_zeros);
        }
        return typename Traits::chars_result{ first, false };
    });
}

template<class Traits>
FORCE_INLINE void to_string_into(std::string& out, const typename Traits::value_type& x, int precision,
    bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    const format_kind kind = (fixed && !scientific) ? format_kind::fixed_frac : (scientific && !fixed) ? format_kind::scientific_frac : format_kind::general;
    format_to_string<Traits>(out, x, precision, kind, strip_trailing_zeros);
}

template<class Traits>
FORCE_INLINE void emit_scientific(std::string& out, const typename Traits::value_type& x, std::streamsize prec, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, static_cast<int>(prec), format_kind::scientific_frac, strip_trailing_zeros);
}

template<class Traits>
FORCE_INLINE void emit_fixed_dec(std::string& out, const typename Traits::value_type& x, int prec, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, prec, format_kind::fixed_frac, strip_trailing_zeros);
}

template<class Traits>
FORCE_INLINE void emit_scientific_sig(std::string& out, const typename Traits::value_type& x, std::streamsize sig_digits, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, static_cast<int>(sig_digits), format_kind::scientific_sig, strip_trailing_zeros);
}

template<typename Token>
constexpr NO_INLINE void scan_decimal_digits(const char*& p, Token& token, bool fractional) noexcept
{
    while (*p >= '0' && *p <= '9')
    {
        const int digit = *p - '0';
        if (digit != 0 || token.seen_nonzero)
        {
            token.coeff.mul_small(static_cast<std::uint32_t>(10));
            token.coeff.add_small(static_cast<std::uint32_t>(digit));
            ++token.sig_digits;
            token.seen_nonzero = true;
        }
        ++p;
        token.any_digit = true;
        if (fractional)
            ++token.frac_digits;
    }
}

template<typename Token>
constexpr NO_INLINE void scan_optional_exp10(const char*& p, Token& token) noexcept
{
    if (*p != 'e' && *p != 'E')
        return;

    const char* pe = p + 1;
    bool neg_exp = false;
    if (*pe == '+' || *pe == '-')
    {
        neg_exp = (*pe == '-');
        ++pe;
    }
    if (*pe < '0' || *pe > '9')
        return;

    int eacc = 0;
    while (*pe >= '0' && *pe <= '9')
    {
        const int digit = *pe - '0';
        if (eacc < 100000000)
            eacc = eacc * 10 + digit;
        ++pe;
    }

    token.exp10 = neg_exp ? -eacc : eacc;
    p = pe;
}

template<typename Token>
constexpr NO_INLINE bool scan_decimal_token(const char*& p, Token& token) noexcept
{
    scan_decimal_digits(p, token, false);
    if (*p == '.')
    {
        ++p;
        scan_decimal_digits(p, token, true);
    }
    if (!token.any_digit)
        return false;
    scan_optional_exp10(p, token);
    return true;
}

template<class Traits>
constexpr NO_INLINE bool parse_special(const char*& p, bool neg, typename Traits::value_type& out) noexcept
{
    if (ascii_lower(p[0]) == 'n' && ascii_lower(p[1]) == 'a' && ascii_lower(p[2]) == 'n')
    {
        out = Traits::quiet_nan();
        p += 3;
        return true;
    }

    if (ascii_lower(p[0]) != 'i' || ascii_lower(p[1]) != 'n' || ascii_lower(p[2]) != 'f')
        return false;

    p += 3;
    if (ascii_lower(p[0]) == 'i' && ascii_lower(p[1]) == 'n' && ascii_lower(p[2]) == 'i' && ascii_lower(p[3]) == 't' && ascii_lower(p[4]) == 'y')
        p += 5;

    out = Traits::infinity(neg);
    return true;
}

template<class Traits>
constexpr NO_INLINE bool parse_flt(const char* s, typename Traits::value_type& out, const char** endptr = nullptr) noexcept
{
    using token_type = typename Traits::parse_token;

    const char* p = skip_ascii_space(s);
    bool neg = false;
    if (*p == '+' || *p == '-')
    {
        neg = (*p == '-');
        ++p;
    }

    if (parse_special<Traits>(p, neg, out))
    {
        if (endptr)
            *endptr = p;
        return true;
    }

    token_type token;
    if (!scan_decimal_token(p, token))
    {
        if (endptr)
            *endptr = s;
        return false;
    }

    if (!token.seen_nonzero)
    {
        out = Traits::zero(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    const int dec_exp = token.exp10 - token.frac_digits;
    const int approx_dec_order = token.sig_digits + dec_exp - 1;

    if (approx_dec_order > Traits::max_parse_order)
    {
        out = Traits::infinity(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    if (approx_dec_order < Traits::min_parse_order)
    {
        out = Traits::zero(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    out = Traits::exact_decimal_to_value(token.coeff, dec_exp, neg);
    if (endptr)
        *endptr = p;
    return true;
}

template<class Traits>
NO_INLINE std::ostream& write_to_stream(std::ostream& os, const typename Traits::value_type& x)
{
    int prec = static_cast<int>(os.precision());
    if (prec < 0)
        prec = 6;

    const auto flags = os.flags();
    const bool fixed = (flags & std::ios_base::fixed) != 0;
    const bool scientific = (flags & std::ios_base::scientific) != 0;
    const bool showpoint = (flags & std::ios_base::showpoint) != 0;
    const bool showpos = (flags & std::ios_base::showpos) != 0;
    const bool uppercase = (flags & std::ios_base::uppercase) != 0;

    if (write_stream_special<Traits>(os, x, showpos, uppercase))
        return os;

    std::string s;
    if (fixed && !scientific)
    {
        format_to_string<Traits>(s, x, prec, format_kind::fixed_frac, false);
    }
    else if (scientific && !fixed)
    {
        format_to_string<Traits>(s, x, prec, format_kind::scientific_frac, false);
    }
    else
    {
        const int sig = (prec == 0) ? 1 : prec;
        if (Traits::iszero(x))
        {
            if (showpoint)
            {
                format_to_string<Traits>(s, x, std::max(0, sig - 1), format_kind::fixed_frac, false);
                ensure_decimal_point(s);
            }
            else
            {
                s = "0";
            }
            apply_stream_decorations(s, showpos, uppercase);
            os << s;
            return os;
        }

        typename Traits::value_type m;
        int e10 = 0;
        Traits::normalize10(Traits::abs(x), m, e10);

        if (e10 >= -4 && e10 < sig)
        {
            format_to_string<Traits>(s, x, std::max(0, sig - (e10 + 1)), format_kind::fixed_frac, !showpoint);
        }
        else if (showpoint)
        {
            format_to_string<Traits>(s, x, std::max(0, sig - 1), format_kind::scientific_frac, false);
            ensure_decimal_point(s);
        }
        else
        {
            format_to_string<Traits>(s, x, sig, format_kind::scientific_sig, true);
        }
    }

    if (showpoint)
        ensure_decimal_point(s);
    apply_stream_decorations(s, showpos, uppercase);
    os << s;
    return os;
}

} // namespace bl::fltx::common
