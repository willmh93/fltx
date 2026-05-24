/**
 * fltx/f64_math_basic.h - constexpr <cmath>-style basic math functions for f64.
 *
 * f64 rounding, decomposition, remainder, min/max, and adjacent-value helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_MATH_BASIC_INCLUDED
#define F64_MATH_BASIC_INCLUDED

#include "fltx/f64_classification.h"


namespace bl {

namespace detail::_f64_impl
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

} // namespace detail::_f64_impl

[[nodiscard]] BL_FORCE_INLINE constexpr double floor(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::floor(x),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ceil(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::ceil(x),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double trunc(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::trunc(x),
        std::trunc(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::round_half_away_zero(x),
        std::round(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nearbyint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::nearbyint(x),
        std::nearbyint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double rint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::nearbyint(x),
        std::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long>(detail::_f64_impl::round_half_away_zero(x)),
        std::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long long>(detail::_f64_impl::round_half_away_zero(x)),
        std::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long>(detail::_f64_impl::nearbyint(x)),
        std::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long long>(detail::_f64_impl::nearbyint(x)),
        std::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmod(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::fmod_exact(x, y),
        std::fmod(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remainder(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::remainder_exact(x, y),
        std::remainder(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remquo(double x, double y, int* quo) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::remainder_exact(x, y, quo),
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
        detail::_f64_impl::fmin(a, b),
        std::fmin(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::fmax(a, b),
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
        detail::_f64_impl::copysign(x, y),
        std::copysign(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ldexp(double x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::ldexp(x, e),
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
        detail::_f64_impl::frexp(x, exp),
        std::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::modf(x, iptr),
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

    return detail::_f64_impl::ilogb_finite(abs(x));
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
        detail::_f64_impl::nextafter(from, to),
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

} // namespace bl

#endif
