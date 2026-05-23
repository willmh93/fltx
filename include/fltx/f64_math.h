/**
 * fltx/f64_math.h - constexpr <cmath>-style functions for f64.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_MATH_INCLUDED
#define F64_MATH_INCLUDED

#include "fltx/aliases.h"
#include "fltx/detail/common_math.h"
#include "fltx/detail/math_utils.h"

namespace bl {

namespace detail::_f64_constexpr
{
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::signbit;
    using detail::fp::fabs;
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::trunc;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::nearbyint;
    using detail::fp::round_half_away_zero;
    using detail::fp::nextafter;
    using detail::fp::copysign;
    using detail::fp::to_signed_integer_or_zero;
    using detail::fp::frexp_exponent;
    using detail::fp::ldexp;
    using detail::fp::log;
    using detail::fp::log1p;
    using detail::fp::sin;
    using detail::fp::cos;
    using detail::fp::tan;
    using detail::fp::atan;
    using detail::fp::atan2;
    using detail::fp::sqrt_seed;

    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double pi_2 = 1.570796326794896619231321691639751442;
    constexpr double pi_4 = 0.785398163397448309615660845819875721;
    constexpr double ln2 = 0.693147180559945309417232121458176568;
    constexpr double inv_ln2 = 1.442695040888963407359924681001892137;
    constexpr double inv_ln10 = 0.434294481903251827651128918916605082;

    BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
    {
        return x == 0.0;
    }

    BL_FORCE_INLINE constexpr double abs(double x) noexcept
    {
        return fabs(x);
    }

    BL_FORCE_INLINE constexpr int ilogb_finite(double x) noexcept
    {
        return frexp_exponent(x) - 1;
    }

    BL_FORCE_INLINE constexpr double powi_nonneg(double base, std::uint64_t exp) noexcept
    {
        double result = 1.0;
        while (exp != 0)
        {
            if ((exp & 1u) != 0)
                result *= base;
            exp >>= 1u;
            if (exp != 0)
                base *= base;
        }
        return result;
    }

    BL_FORCE_INLINE constexpr double powi(double base, long long exp) noexcept
    {
        if (exp == 0)
            return 1.0;

        if (exp < 0)
            return 1.0 / powi_nonneg(base, static_cast<std::uint64_t>(-(exp + 1))) / base;

        return powi_nonneg(base, static_cast<std::uint64_t>(exp));
    }


    struct dyadic_u64
    {
        std::uint64_t coeff = 0;
        int exp2 = 0;
    };
    struct exact_divmod_result
    {
        std::uint64_t remainder   = 0;
        int remainder_exp2   = 0;
        std::uint64_t denominator = 0;
        int denominator_exp2 = 0;
        unsigned quotient_low_bits = 0;
        bool quotient_nonzero = false;
    };

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

    [[nodiscard]] BL_FORCE_INLINE constexpr dyadic_u64 decompose_normalized_abs(double value) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value) & 0x7fffffffffffffffull;
        const std::uint64_t frac = bits & ((std::uint64_t{ 1 } << 52) - 1u);
        const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);

        dyadic_u64 out;
        if (exp_bits == 0)
        {
            out.coeff = frac;
            out.exp2 = -1074;
        }
        else
        {
            out.coeff = (std::uint64_t{ 1 } << 52) | frac;
            out.exp2 = static_cast<int>(exp_bits) - 1023 - 52;
        }

        const int bits_used = bit_length_u64(out.coeff);
        const int shift     = 53 - bits_used;
        out.coeff <<= shift;
        out.exp2 -= shift;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr int compare_scaled_u64(std::uint64_t a, int a_exp2, std::uint64_t b, int b_exp2) noexcept
    {
        if (a == 0 || b == 0)
            return (a != 0) - (b != 0);

        const int a_bits = bit_length_u64(a);
        const int b_bits = bit_length_u64(b);
        const int a_top  = a_exp2 + a_bits - 1;
        const int b_top  = b_exp2 + b_bits - 1;
        if (a_top < b_top) return -1;
        if (a_top > b_top) return 1;

        if (a_exp2 >= b_exp2)
            a <<= (a_exp2 - b_exp2);
        else
            b <<= (b_exp2 - a_exp2);

        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr std::uint64_t rounded_shr_u64(std::uint64_t value, int bits) noexcept
    {
        if (bits <= 0)
            return value;

        const bool round_bit = bits <= 64 && ((value >> (bits - 1)) & 1u) != 0;
        const bool sticky = bits > 1
            ? (bits < 64 ? (value & ((std::uint64_t{ 1 } << (bits - 1)) - 1u)) != 0 : (value << 1) != 0)
            : false;
        const std::uint64_t shifted = bits >= 64 ? 0 : (value >> bits);
        return shifted + (round_bit && (sticky || (shifted & 1u) != 0) ? 1u : 0u);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double exact_dyadic_to_double(std::uint64_t coeff, int exp2, bool neg) noexcept
    {
        const std::uint64_t sign = neg ? (std::uint64_t{ 1 } << 63) : 0;
        if (coeff == 0)
            return std::bit_cast<double>(sign);

        const int top_bit = bit_length_u64(coeff) - 1;
        int unbiased_exp = exp2 + top_bit;
        if (unbiased_exp > 1023)
            return neg ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();

        if (unbiased_exp < -1022)
        {
            const int scale = exp2 + 1074;
            const std::uint64_t subnormal = scale >= 0
                ? (scale >= 64 ? 0 : (coeff << scale))
                : rounded_shr_u64(coeff, -scale);

            if (subnormal == 0)
                return std::bit_cast<double>(sign);
            if (subnormal >= (std::uint64_t{ 1 } << 52))
                return std::bit_cast<double>(sign | (std::uint64_t{ 1 } << 52));
            return std::bit_cast<double>(sign | subnormal);
        }

        const int shift = top_bit - 52;
        std::uint64_t significand = shift > 0
            ? rounded_shr_u64(coeff, shift)
            : (coeff << -shift);

        if (bit_length_u64(significand) > 53)
        {
            significand >>= 1;
            ++unbiased_exp;
            if (unbiased_exp > 1023)
                return neg ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        }

        const std::uint64_t exp_bits  = static_cast<std::uint64_t>(unbiased_exp + 1023);
        const std::uint64_t frac_bits = significand & ((std::uint64_t{ 1 } << 52) - 1u);
        return std::bit_cast<double>(sign | (exp_bits << 52) | frac_bits);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr dyadic_u64 subtract_scaled_u64(std::uint64_t a, int a_exp2, std::uint64_t b, int b_exp2) noexcept
    {
        const int common_exp2 = a_exp2 < b_exp2 ? a_exp2 : b_exp2;
        const int a_shift     = a_exp2 - common_exp2;
        const int b_shift     = b_exp2 - common_exp2;
        return dyadic_u64{ (a << a_shift) - (b << b_shift), common_exp2 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr exact_divmod_result exact_abs_divmod(double ax, double ay) noexcept
    {
        const dyadic_u64 x = decompose_normalized_abs(ax);
        const dyadic_u64 y = decompose_normalized_abs(ay);

        exact_divmod_result out;
        out.denominator = y.coeff;
        out.denominator_exp2 = y.exp2;

        if (compare_scaled_u64(x.coeff, x.exp2, y.coeff, y.exp2) < 0)
        {
            out.remainder = x.coeff;
            out.remainder_exp2 = x.exp2;
            return out;
        }

        std::uint64_t rem = x.coeff;
        const int shift = x.exp2 - y.exp2;

        for (int i = shift; i >= 0; --i)
        {
            if (rem >= y.coeff)
            {
                rem -= y.coeff;
                out.quotient_nonzero = true;
                if (i < 3)
                    out.quotient_low_bits |= (1u << i);
            }

            if (i != 0)
                rem <<= 1;
        }

        out.remainder = rem;
        out.remainder_exp2 = y.exp2;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr int normalized_remquo_bits(unsigned quotient_low_bits, bool quotient_nonzero, bool quotient_negative) noexcept
    {
        int bits = static_cast<int>(quotient_low_bits & 0x7u);
        if (bits == 0 && quotient_nonzero)
            bits = 8;
        return quotient_negative ? -bits : bits;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmod_exact(double x, double y) noexcept
    {
        if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y) || x == 0.0)
            return x;

        const exact_divmod_result divmod = exact_abs_divmod(abs(x), abs(y));
        double out = exact_dyadic_to_double(divmod.remainder, divmod.remainder_exp2, signbit(x));
        if (out == 0.0)
            out = signbit(x) ? -0.0 : 0.0;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double remainder_exact(double x, double y, int* quo = nullptr) noexcept
    {
        if (quo)
            *quo = 0;

        if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y))
            return x;

        const bool quotient_negative = signbit(x) != signbit(y);
        exact_divmod_result divmod = exact_abs_divmod(abs(x), abs(y));

        bool result_negative = signbit(x);
        std::uint64_t result_coeff = divmod.remainder;
        int result_exp2 = divmod.remainder_exp2;

        const int half_cmp = compare_scaled_u64(
            divmod.remainder,
            divmod.remainder_exp2 + 1,
            divmod.denominator,
            divmod.denominator_exp2);
        if (half_cmp > 0 || (half_cmp == 0 && (divmod.quotient_low_bits & 1u) != 0))
        {
            const dyadic_u64 adjusted = subtract_scaled_u64(
                divmod.denominator,
                divmod.denominator_exp2,
                divmod.remainder,
                divmod.remainder_exp2);

            result_coeff = adjusted.coeff;
            result_exp2 = adjusted.exp2;
            divmod.quotient_low_bits = (divmod.quotient_low_bits + 1u) & 0x7u;
            divmod.quotient_nonzero = true;
            result_negative = !result_negative;
        }

        if (quo)
            *quo = normalized_remquo_bits(divmod.quotient_low_bits, divmod.quotient_nonzero, quotient_negative);

        double out = exact_dyadic_to_double(result_coeff, result_exp2, result_negative);
        if (out == 0.0)
            out = signbit(x) ? -0.0 : 0.0;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmin(double a, double b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a < b) return a;
        if (b < a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? a : b;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a > b) return a;
        if (b > a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? b : a;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double frexp(double x, int* exp) noexcept
    {
        if (exp)
            *exp = 0;

        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        int e = frexp_exponent(x);
        double m = ldexp(x, -e);
        const double am = abs(m);

        if (am < 0.5)
        {
            m *= 2.0;
            --e;
        }
        else if (am >= 1.0)
        {
            m *= 0.5;
            ++e;
        }

        if (exp)
            *exp = e;

        return m;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
    {
        const double i = trunc(x);
        if (iptr)
            *iptr = i;

        double frac = x - i;
        if (iszero(frac))
            frac = signbit(x) ? -0.0 : 0.0;
        return frac;
    }

} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double abs(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::abs(x),
        std::abs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fabs(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::abs(x),
        std::fabs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::signbit(x),
        std::signbit(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::isnan(x),
        std::isnan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::isinf(x),
        std::isinf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::isfinite(x),
        std::isfinite(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
{
    return x == 0.0;
}

[[nodiscard]] BL_FORCE_INLINE constexpr double floor(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::floor(x),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ceil(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::ceil(x),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double trunc(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::trunc(x),
        std::trunc(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::round_half_away_zero(x),
        std::round(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nearbyint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::nearbyint(x),
        std::nearbyint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double rint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::nearbyint(x),
        std::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::to_signed_integer_or_zero<long>(detail::_f64_constexpr::round_half_away_zero(x)),
        std::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::to_signed_integer_or_zero<long long>(detail::_f64_constexpr::round_half_away_zero(x)),
        std::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::to_signed_integer_or_zero<long>(detail::_f64_constexpr::nearbyint(x)),
        std::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::to_signed_integer_or_zero<long long>(detail::_f64_constexpr::nearbyint(x)),
        std::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmod(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::fmod_exact(x, y),
        std::fmod(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remainder(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::remainder_exact(x, y),
        std::remainder(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remquo(double x, double y, int* quo) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::remainder_exact(x, y, quo),
        std::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fma(double x, double y, double z) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        x * y + z,
        std::fma(x, y, z)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmin(double a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::fmin(a, b),
        std::fmin(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::fmax(a, b),
        std::fmax(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fdim(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        (x > y) ? (x - y) : 0.0,
        std::fdim(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double copysign(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::copysign(x, y),
        std::copysign(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ldexp(double x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::ldexp(x, e),
        std::ldexp(x, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double scalbn(double x, int e) noexcept
{
    return ldexp(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double scalbln(double x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double frexp(double x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::frexp(x, exp),
        std::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::modf(x, iptr),
        std::modf(x, iptr)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(double x) noexcept
{
    if (isnan(x))
        return FP_ILOGBNAN;
    if (iszero(x))
        return FP_ILOGB0;
    if (isinf(x))
        return std::numeric_limits<int>::max();

    return detail::_f64_constexpr::ilogb_finite(abs(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double logb(double x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return -std::numeric_limits<double>::infinity();
    if (isinf(x))
        return std::numeric_limits<double>::infinity();

    return static_cast<double>(ilogb(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nextafter(double from, double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::nextafter(from, to),
        std::nextafter(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        nextafter(from, static_cast<double>(to)),
        std::nexttoward(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, double to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(double x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<double>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(double x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(double a, double b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(double a, double b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a != b;
}


// exp / log

namespace detail::_f64_constexpr
{
    BL_FORCE_INLINE constexpr double exp(double x) noexcept
    {
        constexpr double one = 1.0;
        constexpr double half_pos = 0.5;
        constexpr double half_neg = -0.5;
        constexpr double max_log = 7.09782712893383973096e+02;
        constexpr double min_log = -7.45133219101941108420e+02;
        constexpr double ln2_hi = 6.93147180369123816490e-01;
        constexpr double ln2_lo = 1.90821492927058770002e-10;
        constexpr double inv_ln2_local = 1.44269504088896338700e+00;
        constexpr double p1 = 1.66666666666666019037e-01;
        constexpr double p2 = -2.77777777770155933842e-03;
        constexpr double p3 = 6.61375632143793436117e-05;
        constexpr double p4 = -1.65339022054652515390e-06;
        constexpr double p5 = 4.13813679705723846039e-08;
        constexpr double half_ln2 = 3.46573590279972654709e-01;
        constexpr double one_and_half_ln2 = 1.03972077083991796413e+00;
        constexpr double tiny = 0x1.0p-28;

        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit(x) ? 0.0 : std::numeric_limits<double>::infinity();
        if (x > max_log)
            return std::numeric_limits<double>::infinity();
        if (x < min_log)
            return 0.0;

        const bool neg = signbit(x);
        const double ax = abs(x);

        double hi = 0.0;
        double lo = 0.0;
        int k = 0;

        if (ax > half_ln2)
        {
            if (ax < one_and_half_ln2)
            {
                hi = x - (neg ? -ln2_hi : ln2_hi);
                lo = neg ? -ln2_lo : ln2_lo;
                k = neg ? -1 : 1;
            }
            else
            {
                const double kd = trunc(x * inv_ln2_local + (neg ? half_neg : half_pos));
                k = static_cast<int>(kd);
                hi = x - kd * ln2_hi;
                lo = kd * ln2_lo;
            }

            x = hi - lo;
        }
        else if (ax < tiny)
        {
            return one + x;
        }

        const double t = x * x;
        const double c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))));

        if (k == 0)
            return one - ((x * c) / (c - 2.0) - x);

        const double y = one - ((lo - (x * c) / (2.0 - c)) - hi);
        return ldexp(y, k);
    }

    BL_FORCE_INLINE constexpr double exp2(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit(x) ? 0.0 : std::numeric_limits<double>::infinity();

        const double kd   = nearbyint_ties_even(x);
        const int k = static_cast<int>(kd);
        const double frac = x - kd;
        return ldexp(exp(frac * ln2), k);
    }

    BL_FORCE_INLINE constexpr double expm1(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x) ? -1.0 : std::numeric_limits<double>::infinity();
        if (x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax < 0.125)
        {
            double term = x;
            double sum  = x;
            for (int n = 2; n <= 32; ++n)
            {
                term *= x / static_cast<double>(n);
                sum += term;
            }
            return sum;
        }

        return exp(x) - 1.0;
    }

    BL_FORCE_INLINE constexpr double log2(double x) noexcept
    {
        return log(x) * inv_ln2;
    }

    BL_FORCE_INLINE constexpr double log10(double x) noexcept
    {
        return log(x) * inv_ln10;
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double exp(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::exp(x),
        std::exp(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double exp2(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::exp2(x),
        std::exp2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double expm1(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::expm1(x),
        std::expm1(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::log(x),
        std::log(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::log(x),
        std::log(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log2(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::log2(x),
        std::log2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log10(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::log10(x),
        std::log10(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log1p(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::log1p(x),
        std::log1p(x)
    );
}


// roots

namespace detail::_f64_constexpr
{
    BL_FORCE_INLINE constexpr double sqrt(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0 || isinf(x))
            return x;

        double y = sqrt_seed(x);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        return 0.5 * (y + x / y);
    }

    BL_FORCE_INLINE constexpr double cbrt(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        double y = exp(log(ax) / 3.0);
        for (int i = 0; i < 5; ++i)
            y = (y + y + ax / (y * y)) / 3.0;

        const double y_prev = nextafter(y, 0.0);
        const double y_next = nextafter(y, std::numeric_limits<double>::infinity());

        const double e = abs(y * y * y - ax);
        const double e_prev = abs(y_prev * y_prev * y_prev - ax);
        const double e_next = abs(y_next * y_next * y_next - ax);

        if (e_prev < e)
            y = y_prev;
        else if (e_next < e)
            y = y_next;

        return neg ? -y : y;
    }

    BL_FORCE_INLINE constexpr double hypot(double x, double y) noexcept
    {
        if (isinf(x) || isinf(y))
            return std::numeric_limits<double>::infinity();
        if (isnan(x))
            return x;
        if (isnan(y))
            return y;

        double ax = abs(x);
        double ay = abs(y);
        if (ax < ay)
        {
            const double tmp = ax;
            ax = ay;
            ay = tmp;
        }

        if (ax == 0.0)
            return 0.0;

        const double r = ay / ax;
        return ax * sqrt(1.0 + r * r);
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double sqrt(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::sqrt(x),
        std::sqrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cbrt(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::cbrt(x),
        std::cbrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double hypot(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::hypot(x, y),
        std::hypot(x, y)
    );
}


// pow

namespace detail::_f64_constexpr
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double pow(double x, double y) noexcept
    {
        if (iszero(y))
            return 1.0;

        if (isnan(x) || isnan(y))
            return std::numeric_limits<double>::quiet_NaN();

        const double yi = trunc(y);
        const bool y_is_int = (yi == y);

        if (y_is_int && yi >= static_cast<double>(std::numeric_limits<long long>::min()) &&
            yi <= static_cast<double>(std::numeric_limits<long long>::max()))
        {
            return powi(x, static_cast<long long>(yi));
        }

        if (x < 0.0 || (x == 0.0 && signbit(x)))
        {
            if (!y_is_int)
                return std::numeric_limits<double>::quiet_NaN();

            const double magnitude = exp(y * log(-x));
            const double parity    = fmod_exact(abs(yi), 2.0);
            return (parity == 1.0) ? -magnitude : magnitude;
        }

        return exp(y * log(x));
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double pow(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::pow(x, y),
        std::pow(x, y)
    );
}


// trig

[[nodiscard]] BL_FORCE_INLINE constexpr double sin(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::sin(x),
        std::sin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cos(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::cos(x),
        std::cos(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(double x, double& s_out, double& c_out) noexcept
{
    s_out = bl::sin(x);
    c_out = bl::cos(x);
    return bl::isfinite(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::tan(x),
        std::tan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::atan(x),
        std::atan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atan2(double y, double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::atan2(y, x),
        std::atan2(y, x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double asin(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        atan2(x, sqrt((1.0 - x) * (1.0 + x))),
        std::asin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double acos(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        2.0 * atan2(sqrt(1.0 - x), sqrt(1.0 + x)),
        std::acos(x)
    );
}

template<class Vec> requires detail::fp::sincos_vector_assignable<Vec, double>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(double x, Vec& out)
{
    double s_out{};
    double c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    detail::fp::assign_sincos_vector(out, s_out, c_out);
    return ok;
}

template<class Value> requires std::same_as<std::remove_cvref_t<Value>, double>
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<double> sincos(double x)
{
    double s_out{};
    double c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    return detail::fp::make_sincos_result(s_out, c_out, ok);
}


// hyperbolic

namespace detail::_f64_constexpr
{
    BL_FORCE_INLINE constexpr double sinh(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax < 0.25)
        {
            const double x2 = x * x;
            double p = 1.6059043836821613e-10;  // 1/13!
            p = p * x2 + 2.5052108385441720e-8; // 1/11!
            p = p * x2 + 2.7557319223985893e-6; // 1/9!
            p = p * x2 + 1.9841269841269841e-4; // 1/7!
            p = p * x2 + 8.3333333333333332e-3; // 1/5!
            p = p * x2 + 1.6666666666666666e-1; // 1/3!
            return x + x * x2 * p;
        }

        if (ax < 0.5)
        {
            const double em1 = expm1(ax);
            const double out = (em1 * (em1 + 2.0)) / ((em1 + 1.0) * 2.0);
            return signbit(x) ? -out : out;
        }

        const double ex = exp(ax);
        double out = (ex - 1.0 / ex) * 0.5;
        return signbit(x) ? -out : out;
    }

    BL_FORCE_INLINE constexpr double cosh(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return std::numeric_limits<double>::infinity();

        const double ax = abs(x);
        const double ex = exp(ax);
        return (ex + 1.0 / ex) * 0.5;
    }

    BL_FORCE_INLINE constexpr double tanh(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;
        if (isinf(x))
            return signbit(x) ? -1.0 : 1.0;

        const double ax = abs(x);
        if (ax > 20.0)
            return signbit(x) ? -1.0 : 1.0;

        const double em1 = expm1(ax + ax);
        double out = em1 / (em1 + 2.0);
        if (signbit(x))
            out = -out;
        return out;
    }

    BL_FORCE_INLINE constexpr double asinh(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        double out = 0.0;
        if (ax > 0x1p500)
            out = log(ax) + ln2;
        else
            out = log1p(ax + (ax * ax) / (1.0 + sqrt(1.0 + ax * ax)));

        return signbit(x) ? -out : out;
    }

    BL_FORCE_INLINE constexpr double acosh(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x < 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 1.0)
            return 0.0;
        if (isinf(x))
            return x;

        if (x > 0x1p500)
            return log(x) + ln2;

        return log(x + sqrt((x - 1.0) * (x + 1.0)));
    }

    BL_FORCE_INLINE constexpr double atanh(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (ax == 1.0)
            return signbit(x)
                ? -std::numeric_limits<double>::infinity()
                :  std::numeric_limits<double>::infinity();

        return 0.5 * (log1p(x) - log1p(-x));
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double sinh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::sinh(x),
        std::sinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cosh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::cosh(x),
        std::cosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tanh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::tanh(x),
        std::tanh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double asinh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::asinh(x),
        std::asinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double acosh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::acosh(x),
        std::acosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atanh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::atanh(x),
        std::atanh(x)
    );
}


// erf / erfc

namespace detail::_f64_constexpr
{
    constexpr double erf_erx  = 8.45062911510467529297e-01;
    constexpr double erf_efx8 = 1.02703333676410069053e+00;

    constexpr double erf_pp0 = 1.28379167095512558561e-01;
    constexpr double erf_pp1 = -3.25042107247001499370e-01;
    constexpr double erf_pp2 = -2.84817495755985104766e-02;
    constexpr double erf_pp3 = -5.77027029648944159157e-03;
    constexpr double erf_pp4 = -2.37630166566501626084e-05;

    constexpr double erf_qq1 = 3.97917223959155352819e-01;
    constexpr double erf_qq2 = 6.50222499887672944485e-02;
    constexpr double erf_qq3 = 5.08130628187576562776e-03;
    constexpr double erf_qq4 = 1.32494738004321644526e-04;
    constexpr double erf_qq5 = -3.96022827877536812320e-06;

    constexpr double erf_pa0 = -2.36211856075265944077e-03;
    constexpr double erf_pa1 = 4.14856118683748331666e-01;
    constexpr double erf_pa2 = -3.72207876035701323847e-01;
    constexpr double erf_pa3 = 3.18346619901161753674e-01;
    constexpr double erf_pa4 = -1.10894694282396677476e-01;
    constexpr double erf_pa5 = 3.54783043256182359371e-02;
    constexpr double erf_pa6 = -2.16637559486879084300e-03;

    constexpr double erf_qa1 = 1.06420880400844228286e-01;
    constexpr double erf_qa2 = 5.40397917702171048937e-01;
    constexpr double erf_qa3 = 7.18286544141962662868e-02;
    constexpr double erf_qa4 = 1.26171219808761642112e-01;
    constexpr double erf_qa5 = 1.36370839120290507362e-02;
    constexpr double erf_qa6 = 1.19844998467991074170e-02;

    constexpr double erf_ra0 = -9.86494403484714822705e-03;
    constexpr double erf_ra1 = -6.93858572707181764372e-01;
    constexpr double erf_ra2 = -1.05586262253232909814e+01;
    constexpr double erf_ra3 = -6.23753324503260060396e+01;
    constexpr double erf_ra4 = -1.62396669462573470355e+02;
    constexpr double erf_ra5 = -1.84605092906711035994e+02;
    constexpr double erf_ra6 = -8.12874355063065934246e+01;
    constexpr double erf_ra7 = -9.81432934416914548592e+00;

    constexpr double erf_sa1 = 1.96512716674392571292e+01;
    constexpr double erf_sa2 = 1.37657754143519042600e+02;
    constexpr double erf_sa3 = 4.34565877475229228821e+02;
    constexpr double erf_sa4 = 6.45387271733267880336e+02;
    constexpr double erf_sa5 = 4.29008140027567833386e+02;
    constexpr double erf_sa6 = 1.08635005541779435134e+02;
    constexpr double erf_sa7 = 6.57024977031928170135e+00;
    constexpr double erf_sa8 = -6.04244152148580987438e-02;

    constexpr double erf_rb0 = -9.86494292470009928597e-03;
    constexpr double erf_rb1 = -7.99283237680523006574e-01;
    constexpr double erf_rb2 = -1.77579549177547519889e+01;
    constexpr double erf_rb3 = -1.60636384855821916062e+02;
    constexpr double erf_rb4 = -6.37566443368389627722e+02;
    constexpr double erf_rb5 = -1.02509513161107724954e+03;
    constexpr double erf_rb6 = -4.83519191608651397019e+02;

    constexpr double erf_sb1 = 3.03380607434824582924e+01;
    constexpr double erf_sb2 = 3.25792512996573918826e+02;
    constexpr double erf_sb3 = 1.53672958608443695994e+03;
    constexpr double erf_sb4 = 3.19985821950859553908e+03;
    constexpr double erf_sb5 = 2.55305040643316442583e+03;
    constexpr double erf_sb6 = 4.74528541206955367215e+02;
    constexpr double erf_sb7 = -2.24409524465858183362e+01;

    BL_FORCE_INLINE constexpr double truncate_low_word(double x) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x) & 0xffffffff00000000ull;
        return std::bit_cast<double>(bits);
    }

    BL_FORCE_INLINE constexpr double erfc1(double ax) noexcept
    {
        const double s = ax - 1.0;
        const double p = erf_pa0 + s * (erf_pa1 + s * (erf_pa2 + s * (erf_pa3 + s * (erf_pa4 + s * (erf_pa5 + s * erf_pa6)))));
        const double q = 1.0 + s * (erf_qa1 + s * (erf_qa2 + s * (erf_qa3 + s * (erf_qa4 + s * (erf_qa5 + s * erf_qa6)))));
        return 1.0 - erf_erx - p / q;
    }

    BL_FORCE_INLINE constexpr double erfc2(double ax) noexcept
    {
        const double s = 1.0 / (ax * ax);

        double r = 0.0;
        double q = 0.0;
        if (ax < 2.85714285714285714286)
        {
            r = erf_ra0 + s * (erf_ra1 + s * (erf_ra2 + s * (erf_ra3 + s * (erf_ra4 + s * (erf_ra5 + s * (erf_ra6 + s * erf_ra7))))));
            q = 1.0 + s * (erf_sa1 + s * (erf_sa2 + s * (erf_sa3 + s * (erf_sa4 + s * (erf_sa5 + s * (erf_sa6 + s * (erf_sa7 + s * erf_sa8)))))));
        }
        else
        {
            r = erf_rb0 + s * (erf_rb1 + s * (erf_rb2 + s * (erf_rb3 + s * (erf_rb4 + s * (erf_rb5 + s * erf_rb6)))));
            q = 1.0 + s * (erf_sb1 + s * (erf_sb2 + s * (erf_sb3 + s * (erf_sb4 + s * (erf_sb5 + s * (erf_sb6 + s * erf_sb7))))));
        }

        const double z = truncate_low_word(ax);
        return exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + r / q) / ax;
    }

    BL_FORCE_INLINE constexpr double erf(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x) ? -1.0 : 1.0;
        if (x == 0.0)
            return x;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        double out = 0.0;
        if (ax < 0.84375)
        {
            if (ax < 0x1p-28)
                return x * (1.0 + erf_efx8 * 0.125);

            const double z = ax * ax;
            const double r = erf_pp0 + z * (erf_pp1 + z * (erf_pp2 + z * (erf_pp3 + z * erf_pp4)));
            const double s = 1.0 + z * (erf_qq1 + z * (erf_qq2 + z * (erf_qq3 + z * (erf_qq4 + z * erf_qq5))));
            out = ax + ax * (r / s);
        }
        else if (ax < 1.25)
        {
            const double s = ax - 1.0;
            const double p = erf_pa0 + s * (erf_pa1 + s * (erf_pa2 + s * (erf_pa3 + s * (erf_pa4 + s * (erf_pa5 + s * erf_pa6)))));
            const double q = 1.0 + s * (erf_qa1 + s * (erf_qa2 + s * (erf_qa3 + s * (erf_qa4 + s * (erf_qa5 + s * erf_qa6)))));
            out = erf_erx + p / q;
        }
        else if (ax < 6.0)
        {
            out = 1.0 - erfc2(ax);
        }
        else
        {
            out = 1.0 - 0x1p-1022;
        }

        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr double erfc(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit(x) ? 2.0 : 0.0;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        if (ax < 0.84375)
        {
            if (ax < 0x1p-56)
                return 1.0 - x;

            const double z = x * x;
            const double r = erf_pp0 + z * (erf_pp1 + z * (erf_pp2 + z * (erf_pp3 + z * erf_pp4)));
            const double s = 1.0 + z * (erf_qq1 + z * (erf_qq2 + z * (erf_qq3 + z * (erf_qq4 + z * erf_qq5))));
            const double y = r / s;

            if (neg || ax < 0.25)
                return 1.0 - (x + x * y);

            return 0.5 - (x - 0.5 + x * y);
        }

        if (ax < 28.0)
        {
            const double y = (ax < 1.25) ? erfc1(ax) : erfc2(ax);
            return neg ? 2.0 - y : y;
        }

        return neg ? 2.0 - 0x1p-1022 : 0x1p-1022 * 0x1p-1022;
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double erf(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::erf(x),
        std::erf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double erfc(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::erfc(x),
        std::erfc(x)
    );
}


// gamma

namespace detail::_f64_constexpr
{
    BL_FORCE_INLINE constexpr double lgamma_positive(double x) noexcept
    {
        if (x == 1.0 || x == 2.0)
            return 0.0;
        if (x == 0.5)
            return 0.57236494292470009;
        if (x == 1.5)
            return -0.12078223763524522;

        constexpr double coeffs[] =
        {
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        };

        double y = coeffs[0];
        const double z = x - 1.0;
        for (int i = 1; i < static_cast<int>(sizeof(coeffs) / sizeof(coeffs[0])); ++i)
            y += coeffs[i] / (z + static_cast<double>(i));

        const double t = z + 7.5;
        return 0.91893853320467274178032973640562 + (z + 0.5) * log(t) - t + log(y);
    }

    BL_FORCE_INLINE constexpr double lgamma(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
            return lgamma_positive(x);

        const double xi = trunc(x);
        if (xi == x)
            return std::numeric_limits<double>::infinity();

        const double sinpix = sin(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::infinity();

        return log(pi) - log(abs(sinpix)) - lgamma_positive(1.0 - x);
    }

    BL_FORCE_INLINE constexpr double tgamma(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
        {
            const double xi = trunc(x);
            if (xi == x && x <= 171.0)
            {
                double result = 1.0;
                for (int i = 2; i < static_cast<int>(x); ++i)
                    result *= static_cast<double>(i);
                return result;
            }
        }

        if (x > 0.0)
            return exp(lgamma_positive(x));

        const double xi = trunc(x);
        if (xi == x)
            return std::numeric_limits<double>::quiet_NaN();

        const double sinpix = sin(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::quiet_NaN();

        return pi / (sinpix * exp(lgamma_positive(1.0 - x)));
    }
} // namespace detail::_f64_constexpr

[[nodiscard]] BL_FORCE_INLINE constexpr double lgamma(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::lgamma(x),
        std::lgamma(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tgamma(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_constexpr::tgamma(x),
        std::tgamma(x)
    );
}


} // namespace bl

#endif
