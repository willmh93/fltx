/**
 * f256_math.h - constexpr <cmath>-style functions for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_MATH_INCLUDED
#define F256_MATH_INCLUDED

#include "f256.h"
#include "f256_consts.h"
#include "fltx_common_exact.h"
#include "fltx_math_utils.h"

namespace bl {

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_NO_INLINE f256_s    fmod(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_NO_INLINE f256_s    round(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s    round_to_decimals(f256_s v, int prec);
    [[nodiscard]] BL_NO_INLINE f256_s    sqrt(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s    nearbyint(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s    rint(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE long      lround(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE long long llround(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE long      lrint(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE long long llrint(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s    ldexp(const f256_s& a, int e);

    [[nodiscard]] BL_NO_INLINE f256_s exp(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s exp2(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s log(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s log2(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s log10(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_NO_INLINE f256_s pow(const f256_s& x, double y);

    [[nodiscard]] BL_NO_INLINE bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out);
    [[nodiscard]] BL_NO_INLINE f256_s sin(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s cos(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s tan(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s atan(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s asin(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s acos(const f256_s& x);

    [[nodiscard]] BL_NO_INLINE f256_s expm1(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s log1p(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s sinh(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s cosh(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s tanh(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s asinh(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s acosh(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s atanh(const f256_s& x);

    [[nodiscard]] BL_NO_INLINE f256_s cbrt(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s hypot(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_NO_INLINE f256_s remquo(const f256_s& x, const f256_s& y, int* quo);
    [[nodiscard]] BL_NO_INLINE f256_s remainder(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_NO_INLINE f256_s frexp(const f256_s& x, int* exp) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s modf(const f256_s& x, f256_s* iptr) noexcept;
    [[nodiscard]] BL_NO_INLINE int    ilogb(const f256_s& x) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s logb(const f256_s& x) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s scalbn(const f256_s& x, int e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s scalbln(const f256_s& x, long e) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s nextafter(const f256_s& from, const f256_s& to) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s nexttoward(const f256_s& from, long double to) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept;

    [[nodiscard]] BL_NO_INLINE f256_s erf(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s erfc(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s lgamma(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s tgamma(const f256_s& x);
    [[nodiscard]] BL_NO_INLINE f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
}
namespace detail::_f256_constexpr
{
    using namespace detail::_f256;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s round(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s round_to_decimals(f256_s v, int prec);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqrt(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s nearbyint(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s ldexp(const f256_s& a, int e);

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s exp(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s exp2(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s log(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s log2(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s log10(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s pow(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s pow(const f256_s& x, double y);

    [[nodiscard]] BL_FORCE_INLINE constexpr bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sin(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s cos(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s tan(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s atan(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s asin(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s acos(const f256_s& x);

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s expm1(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s log1p(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sinh(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s cosh(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s tanh(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s asinh(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s acosh(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s atanh(const f256_s& x);

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s cbrt(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr int    ilogb(const f256_s& x) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s logb(const f256_s& x) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s erf(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s erfc(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s lgamma(const f256_s& x);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s tgamma(const f256_s& x);
}

// forward declare wrappers to runtime/constexpr calls
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s round(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s round_to_decimals(f256_s v, int prec);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sqrt(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nearbyint(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s ldexp(const f256_s& a, int e);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s exp(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s exp2(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log2(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log10(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, const f256_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, double y);

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sin(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cos(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tan(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atan(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s asin(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s acos(const f256_s& x);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s expm1(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log1p(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sinh(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cosh(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tanh(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s asinh(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s acosh(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atanh(const f256_s& x);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cbrt(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr int ilogb(const f256_s& x) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s logb(const f256_s& x) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept;

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s lgamma(const f256_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tgamma(const f256_s& x);

/// ============= Math =============

namespace detail::_f256
{
    using detail::exact_decimal::add_signed;
    using detail::exact_decimal::biguint;
    using detail::exact_decimal::decompose_double_mantissa;
    using detail::exact_decimal::mod_shift_subtract;
    using detail::exact_decimal::signed_biguint;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept;

    // frexp / ldexp
    BL_FORCE_INLINE  constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
            return frexp_exponent_constexpr(value);

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }
    BL_FORCE_INLINE  constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::ldexp_constexpr2(value, exponent);

        return std::ldexp(value, exponent);
    }
    BL_FORCE_INLINE  constexpr f256_s ldexp_terms(const f256_s& value, int exponent) noexcept
    {
        return renorm(
            ldexp_limb(value.x0, exponent),
            ldexp_limb(value.x1, exponent),
            ldexp_limb(value.x2, exponent),
            ldexp_limb(value.x3, exponent));
    }

    // fmod
    struct exact_dyadic_fmod
    {
        int exp2 = 0;
        biguint mant{};
    };
    BL_FORCE_INLINE  constexpr bool biguint_is_odd(const biguint& value)
    {
        return !value.is_zero() && (value.words[0] & 1u) != 0;
    }
    BL_FORCE_INLINE  constexpr bool biguint_any_low_bits_set(const biguint& value, int bit_count)
    {
        if (bit_count <= 0)
            return false;

        const int full_words = bit_count >> 5;
        const int rem_bits = bit_count & 31;

        for (int i = 0; i < full_words && i < value.size; ++i)
        {
            if (value.words[i] != 0)
                return true;
        }

        if (rem_bits != 0 && full_words < value.size)
        {
            const std::uint32_t mask = (std::uint32_t{ 1 } << rem_bits) - 1u;
            if ((value.words[full_words] & mask) != 0)
                return true;
        }

        return false;
    }
    BL_FORCE_INLINE  constexpr int biguint_trailing_zero_bits(const biguint& value)
    {
        int count = 0;
        for (int i = 0; i < value.size; ++i)
        {
            const std::uint32_t word = value.words[i];
            if (word == 0)
            {
                count += 32;
                continue;
            }

            std::uint32_t bits = word;
            while ((bits & 1u) == 0u)
            {
                bits >>= 1;
                ++count;
            }
            break;
        }
        return count;
    }
    BL_FORCE_INLINE  constexpr biguint biguint_shr_bits(biguint value, int bits)
    {
        if (bits <= 0 || value.is_zero())
            return value;

        const int word_shift = bits >> 5;
        const int bit_shift = bits & 31;

        if (word_shift >= value.size)
        {
            value.clear();
            return value;
        }

        if (word_shift > 0)
        {
            for (int i = 0; i + word_shift < value.size; ++i)
                value.words[i] = value.words[i + word_shift];
            value.size -= word_shift;
        }

        if (bit_shift != 0)
        {
            std::uint32_t carry = 0;
            for (int i = value.size - 1; i >= 0; --i)
            {
                const std::uint32_t next_carry = static_cast<std::uint32_t>(value.words[i] << (32 - bit_shift));
                value.words[i] = static_cast<std::uint32_t>((value.words[i] >> bit_shift) | carry);
                carry = next_carry;
            }
        }

        value.trim();
        return value;
    }
    BL_FORCE_INLINE  constexpr biguint biguint_mod(const biguint& numerator, const biguint& modulus)
    {
        biguint remainder{};
        mod_shift_subtract(numerator, modulus, remainder);
        return remainder;
    }
    BL_FORCE_INLINE  constexpr biguint biguint_mul_mod(const biguint& a, const biguint& b, const biguint& modulus)
    {
        if (a.is_zero() || b.is_zero())
            return {};

        return biguint_mod(mul_big(a, b), modulus);
    }
    BL_FORCE_INLINE  constexpr biguint biguint_pow2_mod(int exponent, const biguint& modulus)
    {
        if (modulus.is_zero())
            return {};
        if (exponent <= 0)
            return biguint_mod(biguint{ 1u }, modulus);

        biguint result = biguint_mod(biguint{ 1u }, modulus);
        biguint base = biguint_mod(biguint{ 2u }, modulus);

        while (exponent > 0)
        {
            if ((exponent & 1) != 0)
                result = biguint_mul_mod(result, base, modulus);

            exponent >>= 1;
            if (exponent != 0)
                base = biguint_mul_mod(base, base, modulus);
        }

        return result;
    }
    BL_FORCE_INLINE  constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
    {
        if (value.mant.is_zero())
        {
            value.exp2 = 0;
            return;
        }

        const int tz = biguint_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = biguint_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }
    BL_MSVC_NOINLINE constexpr exact_dyadic_fmod exact_from_f256_fmod(const f256_s& x)
    {
        int common_exp = std::numeric_limits<int>::max();
        const double limbs[4] = { x.x0, x.x1, x.x2, x.x3 };

        for (double limb : limbs)
        {
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            if (exponent < common_exp)
                common_exp = exponent;
        }

        exact_dyadic_fmod out{};
        if (common_exp == std::numeric_limits<int>::max())
            return out;

        signed_biguint acc{};
        for (double limb : limbs)
        {
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

        if (acc.neg || acc.mag.is_zero())
            return out;

        out.exp2 = common_exp;
        out.mant = acc.mag;
        normalize_exact_dyadic_fmod(out);
        return out;
    }
    BL_MSVC_NOINLINE constexpr f256_s exact_dyadic_to_f256_fmod(const biguint& coeff, int exp2, bool neg)
    {
        if (coeff.is_zero())
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        constexpr int kept_bits = 53 * 5;
        int ratio_exp = coeff.bit_length() - 1;
        biguint q = coeff;

        if (ratio_exp > (kept_bits - 1))
        {
            const int right_shift = ratio_exp - (kept_bits - 1);
            const bool round_bit = q.get_bit(right_shift - 1);
            const bool sticky = biguint_any_low_bits_set(q, right_shift - 1);

            q = biguint_shr_bits(q, right_shift);

            if (round_bit && (sticky || biguint_is_odd(q)))
                q.add_small(1u);

            if (q.bit_length() > kept_bits)
            {
                q = biguint_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < (kept_bits - 1))
        {
            q.shl_bits((kept_bits - 1) - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f256_s>::infinity() : std::numeric_limits<f256_s>::infinity();
        if (e2 < -1074)
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        const std::uint64_t c4 = q.get_bits(0, 53);
        const std::uint64_t c3 = q.get_bits(53, 53);
        const std::uint64_t c2 = q.get_bits(106, 53);
        const std::uint64_t c1 = q.get_bits(159, 53);
        const std::uint64_t c0 = q.get_bits(212, 53);

        const double x0 = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp_constexpr2(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp_constexpr2(static_cast<double>(c3), e2 - 211) : 0.0;
        const double x4 = c4 ? detail::fp::ldexp_constexpr2(static_cast<double>(c4), e2 - 264) : 0.0;

        f256_s out = renorm5(x0, x1, x2, x3, x4);
        return neg ? -out : out;
    }
    BL_MSVC_NOINLINE constexpr f256_s fmod_exact(const f256_s& x, const f256_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f256_fmod(abs(x));
        const exact_dyadic_fmod dy = exact_from_f256_fmod(abs(y));

        if (dx.mant.is_zero() || dy.mant.is_zero())
            return f256_s{ signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

        biguint remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            biguint denominator = dy.mant;
            denominator.shl_bits(shift);
            mod_shift_subtract(dx.mant, denominator, remainder);
            out_exp = dx.exp2;
        }
        else
        {
            remainder = biguint_mod(dx.mant, dy.mant);
            const int shift = dx.exp2 - dy.exp2;
            if (!remainder.is_zero() && shift != 0)
            {
                const biguint scale = biguint_pow2_mod(shift, dy.mant);
                remainder = biguint_mul_mod(remainder, scale, dy.mant);
            }
            out_exp = dy.exp2;
        }

        f256_s out = exact_dyadic_to_f256_fmod(remainder, out_exp, !ispositive(x));
        if (iszero(out))
            return f256_s{ signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return out;
    }
    BL_FORCE_INLINE  constexpr bool fmod_normalize_remainder(f256_s& r, const f256_s& modulus) noexcept
    {
        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
            {
                r += modulus;
                continue;
            }

            if (r >= modulus)
            {
                r -= modulus;
                continue;
            }

            return true;
        }

        return r >= 0.0 && r < modulus;
    }
    BL_FORCE_INLINE  constexpr void fmod_append_expansion_term(double* expansion, int& count, double value) noexcept
    {
        if (value != 0.0)
            expansion[count++] = value;
    }
    BL_MSVC_NOINLINE constexpr f256_s fmod_sub_mul_scalar_expansion(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double r_exp[4]{};
        int r_count = 0;
        fmod_append_expansion_term(r_exp, r_count, r.x3);
        fmod_append_expansion_term(r_exp, r_count, r.x2);
        fmod_append_expansion_term(r_exp, r_count, r.x1);
        fmod_append_expansion_term(r_exp, r_count, r.x0);

        double b_exp[4]{};
        int b_count = 0;
        fmod_append_expansion_term(b_exp, b_count, b.x3);
        fmod_append_expansion_term(b_exp, b_count, b.x2);
        fmod_append_expansion_term(b_exp, b_count, b.x1);
        fmod_append_expansion_term(b_exp, b_count, b.x0);

        double product_exp[16]{};
        const int product_count = scale_expansion_zeroelim(b_count, b_exp, q, product_exp);
        for (int i = 0; i < product_count; ++i)
            product_exp[i] = -product_exp[i];

        double diff_exp[32]{};
        const int diff_count = fast_expansion_sum_zeroelim(r_count, r_exp, product_count, product_exp, diff_exp);
        return from_expansion_fast(diff_exp, diff_count);
    }
    BL_MSVC_NOINLINE constexpr f256_s fmod_runtime(const f256_s& x, const f256_s& y)
    {
        const f256_s ay = abs(y);
        f256_s r = abs(x);

        for (int iteration = 0; iteration < 128 && r >= ay; ++iteration)
        {
            const int ex = frexp_exponent(r.x0);
            const int ey = frexp_exponent(ay.x0);
            int shift = ex - ey - 52;
            if (shift < 0)
                shift = 0;

            f256_s scaled = ldexp_terms(ay, shift);
            while (shift > 0 && scaled > r)
            {
                --shift;
                scaled = ldexp_terms(ay, shift);
            }

            if (!(scaled > 0.0) || scaled > r)
                return fmod_exact(x, y);

            const f256_s q_floor = floor(r / scaled);
            if (q_floor.x1 != 0.0 || q_floor.x2 != 0.0 || q_floor.x3 != 0.0)
                return fmod_exact(x, y);
            if (!(q_floor.x0 > 0.0) || absd(q_floor.x0) >= 0x1p53)
                return fmod_exact(x, y);

            r = fmod_sub_mul_scalar_expansion(r, scaled, q_floor.x0);
            if (!fmod_normalize_remainder(r, scaled))
                return fmod_exact(x, y);
        }

        if (!fmod_normalize_remainder(r, ay))
            return fmod_exact(x, y);

        if (iszero(r))
            return f256_s{ signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

        return ispositive(x) ? r : -r;
    }
    BL_MSVC_NOINLINE constexpr bool fmod_fast_double_divisor_abs(const f256_s& ax, double ay, f256_s& out)
    {
        if (!(ay > 0.0) || !isfinite(ay))
            return false;

        const f256_s mod{ ay, 0.0, 0.0, 0.0 };

        if (ax.x1 == 0.0 && ax.x2 == 0.0 && ax.x3 == 0.0)
        {
            out = f256_s{ fmod_constexpr(ax.x0, ay), 0.0, 0.0, 0.0 };
            return true;
        }

        const double r0 = (ax.x0 < ay) ? ax.x0 : fmod_constexpr(ax.x0, ay);
        const double r1 = (absd(ax.x1) < ay) ? ax.x1 : fmod_constexpr(ax.x1, ay);
        const double r2 = (absd(ax.x2) < ay) ? ax.x2 : fmod_constexpr(ax.x2, ay);
        const double r3 = (absd(ax.x3) < ay) ? ax.x3 : fmod_constexpr(ax.x3, ay);

        f256_s r = f256_s{ r0, 0.0, 0.0, 0.0 } +
            f256_s{ r1, 0.0, 0.0, 0.0 } +
            f256_s{ r2, 0.0, 0.0, 0.0 } +
            f256_s{ r3, 0.0, 0.0, 0.0 };

        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
                r += mod;
            if (r >= mod)
                r -= mod;
        }

        if (r < 0.0 || r >= mod)
            return false;

        // reject boundary-adjacent results so the exact fallback handles the
        // cases where double-limb modular reduction is not strong enough
        const f256_s ar = abs(r);
        const f256_s slack = mod * f256_s{ 0x1p-160 };
        if (ar <= slack || ar >= mod - slack)
            return false;

        out = r;
        return true;
    }

    // integer conversion
    BL_FORCE_INLINE  constexpr bool f256_try_get_int64(const f256_s& x, int64_t& out)
    {
        const f256_s xi = detail::_f256_constexpr::trunc(x);
        if (xi != x)
            return false;

        if (absd(xi.x0) >= 0x1p63)
            return false;

        const int64_t p0 = static_cast<int64_t>(xi.x0);
        const f256_s r0 = sub_inline(xi, detail::_f256_constexpr::to_f256(p0));
        const int64_t p1 = static_cast<int64_t>(r0.x0);
        const f256_s r1 = sub_inline(r0, detail::_f256_constexpr::to_f256(p1));
        const int64_t p2 = static_cast<int64_t>(r1.x0);
        const f256_s r2 = sub_inline(r1, detail::_f256_constexpr::to_f256(p2));
        const int64_t p3 = static_cast<int64_t>(r2.x0 + r2.x1 + r2.x2 + r2.x3);

        out = p0 + p1 + p2 + p3;
        return true;
    }

    // decimal conversion
    BL_FORCE_INLINE  constexpr f256_s pack_decimal_significand(const biguint& q, int e2, bool neg) noexcept
    {
        const std::uint64_t c3 = q.get_bits(0, 53);
        const std::uint64_t c2 = q.get_bits(53, 53);
        const std::uint64_t c1 = q.get_bits(106, 53);
        const std::uint64_t c0 = q.get_bits(159, 53);

        const double x0 = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp_constexpr2(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp_constexpr2(static_cast<double>(c3), e2 - 211) : 0.0;

        f256_s out = renorm(x0, x1, x2, x3);
        return neg ? -out : out;
    }
    BL_MSVC_NOINLINE constexpr f256_s round_decimal_exact_to_f256(const biguint& coeff, int dec_exp, bool neg) noexcept
    {
        if (coeff.is_zero())
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        biguint numerator = coeff;
        biguint denominator{ 1 };
        int bin_exp = 0;

        if (dec_exp >= 0)
        {
            numerator = detail::exact_decimal::mul_big(coeff, detail::exact_decimal::pow5_big(dec_exp));
            bin_exp = dec_exp;
        }
        else
        {
            denominator = detail::exact_decimal::pow5_big(-dec_exp);
            bin_exp = dec_exp;
        }

        int ratio_exp = detail::exact_decimal::floor_log2_ratio(numerator, denominator);
        biguint q = detail::exact_decimal::extract_rounded_significand_chunks(numerator, denominator, ratio_exp, std::numeric_limits<f256_s>::digits);
        if (q.bit_length() > std::numeric_limits<f256_s>::digits)
        {
            q.shr1();
            ++ratio_exp;
        }

        const int e2 = bin_exp + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f256_s>::infinity() : std::numeric_limits<f256_s>::infinity();
        if (e2 < -1074)
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        return pack_decimal_significand(q, e2, neg);
    }
    BL_FORCE_INLINE  constexpr bool try_rounded_decimal_to_f256(const f256_s& integer_part, const char* digits, int digit_count, bool neg, f256_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!f256_try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        const biguint coeff = detail::fp::append_decimal_digits(
            biguint{ static_cast<std::uint64_t>(integer_value) },
            digits,
            digit_count);

        out = round_decimal_exact_to_f256(coeff, -digit_count, neg);
        return true;
    }

    // pow
    BL_FORCE_INLINE  constexpr double limb_mod2(double value) noexcept
    {
        if (value == 0.0 || !isfinite(value) || absd(value) >= 0x1p53)
            return 0.0;

        return fmod_constexpr(value, 2.0);
    }
    BL_FORCE_INLINE  constexpr bool is_odd_integer(const f256_s& x) noexcept
    {
        double mod2 =
            limb_mod2(x.x0) +
            limb_mod2(x.x1) +
            limb_mod2(x.x2) +
            limb_mod2(x.x3);

        mod2 = fmod_constexpr(mod2, 2.0);
        if (mod2 < 0.0)
            mod2 += 2.0;

        return detail::fp::double_integer_is_odd(nearbyint_ties_even(mod2));
    }
    BL_FORCE_INLINE constexpr f256_s powi(f256_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

    // remquo
    BL_FORCE_INLINE  constexpr double limb_mod_power_of_two(double value, double modulus, double zero_threshold) noexcept
    {
        if (value == 0.0 || !isfinite(value) || absd(value) >= zero_threshold)
            return 0.0;

        return fmod_constexpr(value, modulus);
    }
    BL_FORCE_INLINE  constexpr int low_quotient_bits(const f256_s& x) noexcept
    {
        constexpr double modulus = 2147483648.0;
        constexpr double zero_threshold = 0x1p83;

        double bits =
            limb_mod_power_of_two(x.x0, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x1, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x2, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x3, modulus, zero_threshold);

        bits = fmod_constexpr(bits, modulus);
        return static_cast<int>(static_cast<long long>(nearbyint_ties_even(bits)));
    }

    // horner
    [[nodiscard]] constexpr f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;

    // sqrt
    BL_FORCE_INLINE  constexpr f256_s canonicalize_sqrt_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<16>(value.x3);
        return value;
    }
    BL_FORCE_INLINE  constexpr void sqrt_step_seed_recip(const f256_s& scaled_a, f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        const f256_s residual = sub_inline(scaled_a, sqr_inline(y));
        y = add_inline(y, mul_double_inline(residual, half_inv_y0));
    }
    BL_FORCE_INLINE double sqrt_tail_residual_head(const f256_s& scaled_a, const f256_s& y)
    {
        using namespace detail::_f256;

        double p00{}, q00{};
        double p01{}, q01{};
        double p02{}, q02{};
        double p03{}, q03{};
        double p11{}, q11{};
        double p12{}, q12{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p00p12{}, q00q12{};
            f256_simd2d p01p02{}, q01q02{};
            f256_simd2d p03p11{}, q03q11{};

            f256_simd_two_prod_precise(f256_simd_set(y.x0, y.x1), f256_simd_set(y.x0, y.x2), p00p12, q00q12);
            f256_simd_two_prod_precise(f256_simd_set(y.x0, y.x0), f256_simd_set(y.x1, y.x2), p01p02, q01q02);
            f256_simd_two_prod_precise(f256_simd_set(y.x0, y.x1), f256_simd_set(y.x3, y.x1), p03p11, q03q11);

            f256_simd_store(p00p12, p00, p12);
            f256_simd_store(q00q12, q00, q12);
            f256_simd_store(p01p02, p01, p02);
            f256_simd_store(q01q02, q01, q02);
            f256_simd_store(p03p11, p03, p11);
            f256_simd_store(q03q11, q03, q11);
        }
        else
        #endif
        {
            two_prod_precise(y.x0, y.x0, p00, q00);
            two_prod_precise(y.x0, y.x1, p01, q01);
            two_prod_precise(y.x0, y.x2, p02, q02);
            two_prod_precise(y.x0, y.x3, p03, q03);
            two_prod_precise(y.x1, y.x1, p11, q11);
            two_prod_precise(y.x1, y.x2, p12, q12);
        }

        p01 *= 2.0; q01 *= 2.0;
        p02 *= 2.0; q02 *= 2.0;
        p03 *= 2.0; q03 *= 2.0;
        p12 *= 2.0; q12 *= 2.0;

        const double terms[] = {
            -q12,
            -q03,
            scaled_a.x3, -p03, -p12, -q02, -q11,
            scaled_a.x2, -p02, -p11, -q01,
            scaled_a.x1, -p01, -q00,
            scaled_a.x0, -p00
        };

        double compressed[24];
        const int count = compress_expansion_zeroelim(static_cast<int>(sizeof(terms) / sizeof(terms[0])), terms, compressed);
        double residual = 0.0;
        for (int i = 0; i < count; ++i)
            residual += compressed[i];
        return residual;
    }
    BL_FORCE_INLINE f256_s sqrt_step_tail_head(const f256_s& scaled_a, const f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        return add_double_inline(y, sqrt_tail_residual_head(scaled_a, y) * half_inv_y0);
    }
    BL_MSVC_NOINLINE constexpr f256_s sqrt_constexpr_impl(const f256_s& a)
    {
        const int exp2 = frexp_exponent(a.x0);
        const int result_scale = exp2 / 2;
        const int input_scale = -2 * result_scale;
        const f256_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

        const double y0 = sqrt_seed_constexpr(scaled_a.x0);
        const double half_inv_y0 = 0.5 / y0;
        f256_s y{ y0, 0.0, 0.0, 0.0 };
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_sqrt_result(y);
    }
    BL_FORCE_INLINE f256_s sqrt_runtime_impl(const f256_s& a)
    {
        f256_s scaled_a = a;
        int result_scale = 0;
        if (a.x0 < 0x1p-900 || a.x0 > 0x1p900)
        {
            const int exp2 = frexp_exponent(a.x0);
            result_scale = exp2 / 2;
            const int input_scale = -2 * result_scale;
            if (input_scale != 0)
                scaled_a = ldexp_terms(a, input_scale);
        }

        const double y0 = std::sqrt(scaled_a.x0);
        const double half_inv_y0 = 0.5 / y0;
        f256_s y{ y0, 0.0, 0.0, 0.0 };
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        y = sqrt_step_tail_head(scaled_a, y, half_inv_y0);
        y = sqrt_step_tail_head(scaled_a, y, half_inv_y0);

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_sqrt_result(y);
    }

    // exp / log helpers
    BL_FORCE_INLINE constexpr double log_as_double_impl(f256_s a) noexcept
    {
        const double hi = a.x0;
        if (hi <= 0.0)
            return detail::fp::log_constexpr(static_cast<double>(a));

        const double lo = (a.x1 + a.x2) + a.x3;
        return detail::fp::log_constexpr(hi) + detail::fp::log1p_constexpr(lo / hi);
    }
    BL_FORCE_INLINE constexpr f256_s log1p_double_seed_residual(const f256_s& r) noexcept
    {
        const f256_s r2 = mul_inline(r, r);
        const f256_s r3 = mul_inline(r2, r);
        const f256_s r4 = mul_inline(r2, r2);
        const f256_s r5 = mul_inline(r4, r);

        f256_s correction = r;
        correction = sub_inline(correction, mul_double_inline(r2, 0.5));
        correction = add_inline(correction, div_double_inline(r3, 3.0));
        correction = sub_inline(correction, mul_double_inline(r4, 0.25));
        correction = add_inline(correction, mul_double_inline(r5, 0.2));
        return correction;
    }
    BL_MSVC_NOINLINE constexpr f256_s f256_expm1_tiny(const f256_s& r)
    {
        f256_s p = exp_inv_fact[(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 1];
        for (int i = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 2; i >= 0; --i)
            p = f256_mul_add_horner_step(p, r, exp_inv_fact[i]);
        p = f256_mul_add_horner_step(p, r, f256_s{ 0.5 });
        return f256_mul_add_horner_step(mul_inline(r, r), p, r);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s _ldexp(const f256_s& a, int e);
    BL_MSVC_NOINLINE constexpr f256_s f256_expm1_reduced(const f256_s& x)
    {
        const f256_s t = x * inv_ln2;

        double kd = nearbyint_ties_even(t.x0);
        const f256_s delta = t - f256_s{ kd };
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s kd_ln2 = mul_double_inline(ln2, kd);
        const f256_s r = mul_double_inline(sub_inline(x, kd_ln2), 0.0009765625);

        f256_s e = f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = f256_mul_add_horner_step(e, e, mul_double_inline(e, 2.0));

        if (k == 0)
            return e;

        return add_scalar_precise(_ldexp(add_scalar_precise(e, 1.0), k), -1.0);
    }
    BL_MSVC_NOINLINE constexpr f256_s f256_log1p_series_reduced(const f256_s& x)
    {
        const f256_s z = x / (f256_s{ 2.0 } + x);
        const f256_s z2 = z * z;

        f256_s term = z;
        f256_s sum = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term *= z2;
            const f256_s add = term / f256_s{ static_cast<double>(k) };
            sum += add;

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(add) <= f256_s::eps() * scale)
                break;
        }

        return sum + sum;
    }

    // trig helpers
    BL_FORCE_INLINE constexpr bool f256_remainder_pi2(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        if (!isfinite(x.x0))
            return false;

        if (abs(x) <= pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256_s q = nearbyint(x * invpi2);
        const double qd = q.x0;

        if (!detail::fp::isfinite(qd) || detail::fp::absd(qd) > 9.0e15)
        {
            const double xd = static_cast<double>(x);
            const double fallback_qd = (double)detail::fp::llround_constexpr(xd * static_cast<double>(invpi2));

            if (!detail::fp::isfinite(fallback_qd) || detail::fp::absd(fallback_qd) > 9.0e15)
                return false;

            const long long n = (long long)fallback_qd;
            const f256_s qf{ (double)n };

            f256_s r = x;
            r -= qf * pi_2.x0;
            r -= qf * pi_2.x1;
            r -= qf * pi_2.x2;
            r -= qf * pi_2.x3;

            if (r > pi_4)
            {
                r -= pi_2;
                n_out = n + 1;
            }
            else if (r < -pi_4)
            {
                r += pi_2;
                n_out = n - 1;
            }
            else
            {
                n_out = n;
            }

            r_out = r;
            return true;
        }

        long long n = (long long)qd;
        f256_s r = x;
        r -= q * pi_2.x0;
        r -= q * pi_2.x1;
        r -= q * pi_2.x2;
        r -= q * pi_2.x3;

        if (r > pi_4)
        {
            r -= pi_2;
            ++n;
        }
        else if (r < -pi_4)
        {
            r += pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }

    #if BL_F256_ENABLE_SIMD
    BL_FORCE_INLINE constexpr f256_s f256_mul_from_two_prod_terms(
        double p0, double p1, double p2, double p3, double p4, double p5,
        double p6, double p7, double p8, double p9,
        double q0, double q1, double q2, double q3, double q4, double q5,
        double q6, double q7, double q8, double q9,
        double tail_mul0, double tail_mul1, double tail_mul2) noexcept
    {
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);
        t1 += (q3 + q5);

        two_sum_precise(p6, p8, r0, r1);
        r1 += (p7 + p9);

        two_sum_precise(t0, r0, q3, q4);
        q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;

        t1 += tail_mul0 + tail_mul1 + tail_mul2
            + q6 + q7 + q8 + q9 + s2;

        return renorm5(p0, p1, s0, t0, t1);
    }
    BL_FORCE_INLINE void f256_mul_pair_simd(
        const f256_s& a0, const f256_s& b0,
        const f256_s& a1, const f256_s& b1,
        f256_s& out0, f256_s& out1) noexcept
    {
        double p00{}, p10{}, p20{}, p30{}, p40{}, p50{};
        double q00{}, q10{}, q20{}, q30{}, q40{}, q50{};

        double p01{}, p11{}, p21{}, p31{}, p41{}, p51{};
        double q01{}, q11{}, q21{}, q31{}, q41{}, q51{};

        two_prod_precise(a0.x0, b0.x0, p00, q00);
        two_prod_precise(a0.x0, b0.x1, p10, q10);
        two_prod_precise(a0.x1, b0.x0, p20, q20);
        two_prod_precise(a0.x0, b0.x2, p30, q30);
        two_prod_precise(a0.x1, b0.x1, p40, q40);
        two_prod_precise(a0.x2, b0.x0, p50, q50);

        two_prod_precise(a1.x0, b1.x0, p01, q01);
        two_prod_precise(a1.x0, b1.x1, p11, q11);
        two_prod_precise(a1.x1, b1.x0, p21, q21);
        two_prod_precise(a1.x0, b1.x2, p31, q31);
        two_prod_precise(a1.x1, b1.x1, p41, q41);
        two_prod_precise(a1.x2, b1.x0, p51, q51);

        const f256_simd2d ax0 = f256_simd_set(a0.x0, a1.x0);
        const f256_simd2d ax1 = f256_simd_set(a0.x1, a1.x1);
        const f256_simd2d ax2 = f256_simd_set(a0.x2, a1.x2);
        const f256_simd2d ax3 = f256_simd_set(a0.x3, a1.x3);

        const f256_simd2d bx0 = f256_simd_set(b0.x0, b1.x0);
        const f256_simd2d bx1 = f256_simd_set(b0.x1, b1.x1);
        const f256_simd2d bx2 = f256_simd_set(b0.x2, b1.x2);
        const f256_simd2d bx3 = f256_simd_set(b0.x3, b1.x3);

        f256_simd2d p6{}, p7{}, p8{}, p9{};
        f256_simd2d q6{}, q7{}, q8{}, q9{};

        f256_simd_two_prod_precise(ax0, bx3, p6, q6);
        f256_simd_two_prod_precise(ax1, bx2, p7, q7);
        f256_simd_two_prod_precise(ax2, bx1, p8, q8);
        f256_simd_two_prod_precise(ax3, bx0, p9, q9);

        alignas(16) double p6v[2], p7v[2], p8v[2], p9v[2];
        alignas(16) double q6v[2], q7v[2], q8v[2], q9v[2];

        f256_simd_store_array(p6, p6v);
        f256_simd_store_array(p7, p7v);
        f256_simd_store_array(p8, p8v);
        f256_simd_store_array(p9, p9v);
        f256_simd_store_array(q6, q6v);
        f256_simd_store_array(q7, q7v);
        f256_simd_store_array(q8, q8v);
        f256_simd_store_array(q9, q9v);

        out0 = f256_mul_from_two_prod_terms(
            p00, p10, p20, p30, p40, p50,
            p6v[0], p7v[0], p8v[0], p9v[0],
            q00, q10, q20, q30, q40, q50,
            q6v[0], q7v[0], q8v[0], q9v[0],
            a0.x1 * b0.x3, a0.x2 * b0.x2, a0.x3 * b0.x1
        );

        out1 = f256_mul_from_two_prod_terms(
            p01, p11, p21, p31, p41, p51,
            p6v[1], p7v[1], p8v[1], p9v[1],
            q01, q11, q21, q31, q41, q51,
            q6v[1], q7v[1], q8v[1], q9v[1],
            a1.x1 * b1.x3, a1.x2 * b1.x2, a1.x3 * b1.x1
        );
    }
    #endif
    BL_FORCE_INLINE constexpr f256_s f256_mul_add_horner_step_constexpr(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double m0{}, m1{}, m2{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            f256_simd2d p45{}, q45{};
            f256_simd2d p67{}, q67{};
            f256_simd2d p89{}, q89{};

            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x0), f256_simd_set(b.x0, b.x1), p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x0), f256_simd_set(b.x0, b.x2), p23, q23);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x2), f256_simd_set(b.x1, b.x0), p45, q45);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(b.x3, b.x2), p67, q67);
            f256_simd_two_prod_precise(f256_simd_set(a.x2, a.x3), f256_simd_set(b.x1, b.x0), p89, q89);

            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, q3);
            f256_simd_store(p45, p4, p5);
            f256_simd_store(q45, q4, q5);
            f256_simd_store(p67, p6, p7);
            f256_simd_store(q67, q6, q7);
            f256_simd_store(p89, p8, p9);
            f256_simd_store(q89, q8, q9);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, b.x0, p0, q0);
            two_prod_precise(a.x0, b.x1, p1, q1);
            two_prod_precise(a.x1, b.x0, p2, q2);
            two_prod_precise(a.x0, b.x2, p3, q3);
            two_prod_precise(a.x1, b.x1, p4, q4);
            two_prod_precise(a.x2, b.x0, p5, q5);
            two_prod_precise(a.x0, b.x3, p6, q6);
            two_prod_precise(a.x1, b.x2, p7, q7);
            two_prod_precise(a.x2, b.x1, p8, q8);
            two_prod_precise(a.x3, b.x0, p9, q9);
        }

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, m0, t0);
        two_sum_precise(q1, p4, m1, t1); m2 = q2 + p5;
        two_sum_precise(m1, t0, m1, t0); m2 += (t0 + t1);

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1); t1 += (q3 + q5);
        two_sum_precise(p6, p8, r0, r1); r1 += (p7 + p9);
        two_sum_precise(t0, r0, q3, q4); q4 += (t1 + r1);

        two_sum_precise(q3, m1, t0, t1);
        t1 += q4;
        t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + m2;

        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        two_sum_precise(p0, c.x0, s0, e0);
        two_sum_precise(p1, c.x1, s1, e1);
        two_sum_precise(m0, c.x2, s2, e2);
        two_sum_precise(t0, c.x3, s3, e3);

        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3 + t1;
        return renorm5(s0, s1, s2, s3, e0);
    }
    BL_FORCE_INLINE constexpr f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::use_constexpr_math())
            return f256_mul_add_horner_step_constexpr(a, b, c);

        return detail::_f256_runtime::f256_mul_add_horner_step(a, b, c);
    }
    BL_MSVC_NOINLINE constexpr f256_s f256_sin_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s ps = f256_sin_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            ps = f256_mul_add_horner_step(ps, t, f256_sin_coeffs_pi4[i]);

        const f256_s rt = r * t;
        return f256_mul_add_horner_step(rt, ps, r);
    }
    BL_MSVC_NOINLINE constexpr f256_s f256_cos_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s pc = f256_cos_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            pc = f256_mul_add_horner_step(pc, t, f256_cos_coeffs_pi4[i]);

        return f256_mul_add_horner_step(t, pc, f256_s{ 1.0 });
    }
    BL_MSVC_NOINLINE constexpr void f256_sincos_kernel_pi4(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        using namespace detail::_f256;

        const f256_s t = mul_inline(r, r);

        f256_s ps = f256_sin_coeffs_pi4[0];
        f256_s pc = f256_cos_coeffs_pi4[0];

        #if BL_F256_ENABLE_SIMD
        if (f256_runtime_trig_simd_enabled())
        {
            for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            {
                f256_s next_ps{}, next_pc{};
                f256_mul_pair_simd(ps, t, pc, t, next_ps, next_pc);
                ps = add_inline(next_ps, f256_sin_coeffs_pi4[i]);
                pc = add_inline(next_pc, f256_cos_coeffs_pi4[i]);
            }

            const f256_s rt = mul_inline(r, t);
            f256_s sin_tail{}, cos_tail{};
            f256_mul_pair_simd(ps, rt, pc, t, sin_tail, cos_tail);
            s_out = add_inline(r, sin_tail);
            c_out = add_scalar_precise(cos_tail, 1.0);

            return;
        }
        #endif

        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
        {
            ps = f256_mul_add_horner_step(ps, t, f256_sin_coeffs_pi4[i]);
            pc = f256_mul_add_horner_step(pc, t, f256_cos_coeffs_pi4[i]);
        }

        const f256_s rt = mul_inline(r, t);
        s_out = f256_mul_add_horner_step(rt, ps, r);
        c_out = f256_mul_add_horner_step(t, pc, f256_s{ 1.0 });
    }

    // exp / log kernels
    BL_FORCE_INLINE constexpr f256_s _ldexp(const f256_s& a, int e)
    {
        double s;
        if (bl::use_constexpr_math())
        {
            s = bl::detail::fp::ldexp_constexpr2(1.0, e);
        }
        else
        {
            s = std::ldexp(1.0, e);
        }

        if (bl::use_constexpr_math())
        {
            return renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
        }
        else
        {
            #if BL_F256_ENABLE_SIMD
            if (f256_runtime_simd_enabled())
            {
                const f256_simd2d scale = f256_simd_splat(s);
                f256_simd2d lo = f256_simd_mul(f256_simd_set(a.x0, a.x1), scale);
                f256_simd2d hi = f256_simd_mul(f256_simd_set(a.x2, a.x3), scale);
                double x0{}, x1{}, x2{}, x3{};
                f256_simd_store(lo, x0, x1);
                f256_simd_store(hi, x2, x3);
                return renorm(x0, x1, x2, x3);
            }
            else
            #endif
            {
                return renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
            }
        }
    }
    BL_MSVC_NOINLINE constexpr f256_s _exp(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 709.782712893384)
            return std::numeric_limits<f256_s>::infinity();

        if (x.x0 < -745.133219101941)
            return f256_s{ 0.0 };

        if (iszero(x))
            return f256_s{ 1.0 };

        const f256_s t = x * inv_ln2;

        double kd = nearbyint_ties_even(t.x0);
        const f256_s delta = t - f256_s{ kd };
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s kd_ln2 = mul_double_inline(ln2, kd);
        const f256_s r = mul_double_inline(sub_inline(x, kd_ln2), 0.0009765625);

        f256_s e = f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = f256_mul_add_horner_step(e, e, mul_double_inline(e, 2.0));

        return _ldexp(add_scalar_precise(e, 1.0), k);
    }
    BL_MSVC_NOINLINE constexpr f256_s _exp2(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 1023.0 || x.x0 < -1074.0)
            return _exp(mul_inline(x, ln2));

        if (iszero(x))
            return f256_s{ 1.0 };

        double kd = nearbyint_ties_even(x.x0);
        const f256_s delta = sub_inline(x, f256_s{ kd });
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s reduced = sub_inline(x, f256_s{ kd });
        const f256_s r = mul_double_inline(mul_inline(reduced, ln2), 0.0009765625);

        f256_s e = f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = f256_mul_add_horner_step(e, e, mul_double_inline(e, 2.0));

        return _ldexp(add_scalar_precise(e, 1.0), k);
    }
    BL_MSVC_NOINLINE constexpr f256_s _log(const f256_s& a)
    {
        if (isnan(a))
            return a;
        if (iszero(a))
            return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
        if (a.x0 < 0.0 || (a.x0 == 0.0 && (a.x1 < 0.0 || (a.x1 == 0.0 && (a.x2 < 0.0 || (a.x2 == 0.0 && a.x3 < 0.0))))))
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (isinf(a))
            return a;

        int exp2 = 0;
        if (bl::use_constexpr_math()) {
            exp2 = frexp_exponent_constexpr(a.x0);
        }
        else {
            (void)std::frexp(a.x0, &exp2);
        }

        f256_s m = _ldexp(a, -exp2);
        if (m < sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }

        const f256_s exp2_ln2 = mul_double_inline(ln2, static_cast<double>(exp2));
        f256_s y = exp2_ln2 + f256_s{ log_as_double_impl(m), 0.0, 0.0, 0.0 };
        if (bl::use_constexpr_math())
        {
            y += m * _exp(exp2_ln2 - y) - 1.0;
            y += m * _exp(exp2_ln2 - y) - 1.0;
        }
        else
        {
            const f256_s residual = m * _exp(exp2_ln2 - y) - 1.0;
            y += log1p_double_seed_residual(residual);
        }
        return y;
    }
    BL_MSVC_NOINLINE constexpr f256_s _expm1(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (x == f256_s{ 0.0 })
            return x;
        if (isinf(x))
            return signbit_constexpr(x.x0)
                ? f256_s{ -1.0, 0.0, 0.0, 0.0 }
                : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 709.782712893384)
            return std::numeric_limits<f256_s>::infinity();

        if (x.x0 < -745.133219101941)
            return f256_s{ -1.0, 0.0, 0.0, 0.0 };

        return f256_expm1_reduced(x);
    }

    // trig functions
    BL_MSVC_NOINLINE constexpr bool _sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
    {
        const double ax = fabs_constexpr(x.x0);
        if (!isfinite(ax))
        {
            s_out = f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
            c_out = s_out;
            return false;
        }

        if (ax <= static_cast<double>(pi_4))
        {
            f256_sincos_kernel_pi4(x, s_out, c_out);
            return true;
        }

        long long n = 0;
        f256_s r{};
        if (!f256_remainder_pi2(x, n, r))
            return false;

        f256_s sr{}, cr{};
        f256_sincos_kernel_pi4(r, sr, cr);

        switch ((int)(n & 3LL))
        {
        case 0: s_out = sr;  c_out = cr;  break;
        case 1: s_out = cr;  c_out = -sr; break;
        case 2: s_out = -sr; c_out = -cr; break;
        default: s_out = -cr; c_out = sr;  break;
        }

        return true;
    }
    BL_MSVC_NOINLINE constexpr f256_s atan_core_unit(const f256_s& z)
    {
        using namespace detail::_f256;

        if (z.x0 <= 0x1p-9)
        {
            const f256_s z2 = z * z;
            f256_s p = f256_atan_tiny_coeffs[f256_atan_tiny_coeff_count - 1];
            for (int i = static_cast<int>(f256_atan_tiny_coeff_count) - 2; i >= 0; --i)
                p = f256_mul_add_horner_step(p, z2, f256_atan_tiny_coeffs[i]);

            return z * p;
        }

        f256_s v = f256_s{ detail::fp::atan_constexpr(static_cast<double>(z)) };

        for (int i = 0; i < 2; ++i)
        {
            f256_s sv{}, cv{};
            f256_sincos_kernel_pi4(v, sv, cv);

            #if BL_F256_ENABLE_SIMD
            if (f256_runtime_trig_simd_enabled())
            {
                f256_s zcv{}, zsv{};
                f256_mul_pair_simd(z, cv, z, sv, zcv, zsv);

                const f256_s f = sv - zcv;
                const f256_s fp = cv + zsv;
                v = v - f / fp;
                continue;
            }
            #endif

            const f256_s f  = sv - z * cv;
            const f256_s fp = cv + z * sv;
            v = v - f / fp;
        }

        return v;
    }
    BL_MSVC_NOINLINE constexpr f256_s _atan(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))  return x;
        if (iszero(x)) return x;
        if (isinf(x))  return signbit_constexpr(x.x0) ? -pi_2 : pi_2;

        const bool neg = x.x0 < 0.0;
        const f256_s ax = neg ? -x : x;

        if (ax > f256_s{ 1.0 })
        {
            const f256_s core = atan_core_unit(recip(ax));
            const f256_s out = sub_inline(pi_2, core);
            return neg ? -out : out;
        }

        const f256_s out = atan_core_unit(ax);
        return neg ? -out : out;
    }
    BL_FORCE_INLINE constexpr f256_s _asin(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (ax == f256_s{ 1.0 })
            return (x.x0 < 0.0) ? -pi_2 : pi_2;

        if (ax <= f256_s{ 0.5 })
            return _atan(x / sqrt(f256_s{ 1.0 } - x * x));

        const f256_s t = sqrt((f256_s{ 1.0 } - ax) / (f256_s{ 1.0 } + ax));
        const f256_s a = pi_2 - mul_double_inline(_atan(t), 2.0);
        return (x.x0 < 0.0) ? -a : a;
    }
    BL_FORCE_INLINE constexpr f256_s _acos(const f256_s& x)
    {
        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (x == f256_s{ 1.0 })
            return f256_s{ 0.0 };
        if (x == f256_s{ -1.0 })
            return pi;

        return pi_2 - _asin(x);
    }

    // classification / signs
    BL_FORCE_INLINE constexpr f256_s fabs_impl(const f256_s& a) noexcept
    {
        return abs(a);
    }
    BL_FORCE_INLINE constexpr bool signbit_impl(const f256_s& x) noexcept
    {
        return signbit_constexpr(x.x0)
            || (x.x0 == 0.0 && (signbit_constexpr(x.x1)
            || (x.x1 == 0.0 && (signbit_constexpr(x.x2)
            || (x.x2 == 0.0 && signbit_constexpr(x.x3))))));
    }
    BL_FORCE_INLINE constexpr int fpclassify_impl(const f256_s& x) noexcept
    {
        if (isnan(x))  return FP_NAN;
        if (isinf(x))  return FP_INFINITE;
        if (iszero(x)) return FP_ZERO;
        return abs(x) < std::numeric_limits<f256_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
    }
    BL_FORCE_INLINE constexpr bool isnormal_impl(const f256_s& x) noexcept
    {
        return fpclassify_impl(x) == FP_NORMAL;
    }
    BL_FORCE_INLINE constexpr bool isunordered_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return isnan(a) || isnan(b);
    }
    BL_FORCE_INLINE constexpr bool isgreater_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return !isunordered_impl(a, b) && a > b;
    }
    BL_FORCE_INLINE constexpr bool isgreaterequal_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return !isunordered_impl(a, b) && a >= b;
    }
    BL_FORCE_INLINE constexpr bool isless_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return !isunordered_impl(a, b) && a < b;
    }
    BL_FORCE_INLINE constexpr bool islessequal_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return !isunordered_impl(a, b) && a <= b;
    }
    BL_FORCE_INLINE constexpr bool islessgreater_impl(const f256_s& a, const f256_s& b) noexcept
    {
        return !isunordered_impl(a, b) && a != b;
    }

    // rounding
    BL_FORCE_INLINE constexpr f256_s round_half_away_zero(const f256_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit_impl(x))
        {
            f256_s y = -detail::_f256_constexpr::floor(
                add_inline(-x, f256_s{ 0.5 }));
            if (iszero(y))
                return f256_s{ -0.0, 0.0, 0.0, 0.0 };
            return y;
        }

        return detail::_f256_constexpr::floor(
            add_inline(x, f256_s{ 0.5 }));
    }

    // nextafter / nexttoward
    using detail::fp::nextafter_double_constexpr;
    BL_FORCE_INLINE constexpr f256_s normalize_nextafter_tail(const f256_s& from, double stepped_x3) noexcept
    {
        if (from.x1 == 0.0)
            return f256_s{ from.x0, stepped_x3, 0.0, 0.0 };

        if (from.x2 == 0.0 && (from.x1 + stepped_x3) == from.x1)
            return f256_s{ from.x0, from.x1, stepped_x3, 0.0 };

        if (from.x1 != 0.0 && from.x2 != 0.0 && (from.x2 + stepped_x3) == from.x2)
            return f256_s{ from.x0, from.x1, from.x2, stepped_x3 };

        return renorm4(from.x0, from.x1, from.x2, stepped_x3);
    }
    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f256_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (bl::isnan(x) || bl::isinf(x))
            return 0;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        const f256_s lo = detail::_f256_constexpr::to_f256(lo_i);
        const f256_s hi = detail::_f256_constexpr::to_f256(hi_i);

        if (x < lo || x > hi)
            return 0;

        std::int64_t out = 0;
        if (!f256_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }
    BL_FORCE_INLINE constexpr f256_s nearest_integer_ties_even(const f256_s& q) noexcept
    {
        f256_s n = detail::_f256_constexpr::trunc(q);
        const f256_s frac = sub_inline(q, n);
        const f256_s half{ 0.5 };
        const f256_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n = add_inline(n, signbit_impl(frac) ? -one : one);
        }
        else if (abs(frac) == half)
        {
            if (is_odd_integer(n))
                n = add_inline(n, signbit_impl(frac) ? -one : one);
        }

        return n;
    }

    // hyperbolic
    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series_constexpr(const f256_s& x)
    {
        const f256_s x2 = x * x;
        f256_s sum = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = power * x2;
            const f256_s term = power / static_cast<double>(2 * k + 1);
            sum = sum + term;

            if (abs(term) <= f256_s::eps())
                break;
        }

        return sum;
    }
    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series_runtime(const f256_s& x)
    {
        const f256_s x2 = x * x;
        f256_s sum = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = power * x2;
            const f256_s term = power / static_cast<double>(2 * k + 1);
            sum = sum + term;

            if (abs(term) <= f256_s::eps())
                break;
        }

        return sum;
    }

    // hypot
    BL_FORCE_INLINE constexpr f256_s hypot_impl(const f256_s& x, const f256_s& y)
    {
        if (isinf(x) || isinf(y))
            return std::numeric_limits<f256_s>::infinity();
        if (isnan(x))
            return x;
        if (isnan(y))
            return y;

        f256_s ax = abs(x);
        f256_s ay = abs(y);
        if (ax < ay)
            std::swap(ax, ay);

        if (iszero(ax))
            return f256_s{ 0.0 };
        if (iszero(ay))
            return canonicalize_math_result(ax);

        int ex = 0;
        int ey = 0;
        if (bl::use_constexpr_math())
        {
            ex = frexp_exponent_constexpr(ax.x0);
            ey = frexp_exponent_constexpr(ay.x0);
        }
        else
        {
            (void)std::frexp(ax.x0, &ex);
            (void)std::frexp(ay.x0, &ey);
        }

        if ((ex - ey) > 110)
            return canonicalize_math_result(ax);

        if (!bl::use_constexpr_math() && ex > -450 && ex < 450)
            return canonicalize_math_result(sqrt(add_inline(sqr_inline(ax), sqr_inline(ay))));

        const f256_s r = div_inline(ay, ax);
        return canonicalize_math_result(mul_inline(ax, sqrt(f256_s{ 1.0 } + mul_inline(r, r))));
    }

    // integer rounding
    BL_FORCE_INLINE constexpr f256_s rint_impl(const f256_s& x)
    {
        return detail::_f256_constexpr::nearbyint(x);
    }
    BL_FORCE_INLINE constexpr long lround_impl(const f256_s& x)
    {
        if (bl::use_constexpr_math())
            return to_signed_integer_or_zero<long>(round_half_away_zero(x));

        return detail::_f256_runtime::lround(x);
    }
    BL_FORCE_INLINE constexpr long long llround_impl(const f256_s& x)
    {
        if (bl::use_constexpr_math())
            return to_signed_integer_or_zero<long long>(round_half_away_zero(x));

        return detail::_f256_runtime::llround(x);
    }
    BL_FORCE_INLINE constexpr long lrint_impl(const f256_s& x)
    {
        if (bl::use_constexpr_math())
            return to_signed_integer_or_zero<long>(detail::_f256_constexpr::nearbyint(x));

        return detail::_f256_runtime::lrint(x);
    }
    BL_FORCE_INLINE constexpr long long llrint_impl(const f256_s& x)
    {
        if (bl::use_constexpr_math())
            return to_signed_integer_or_zero<long long>(detail::_f256_constexpr::nearbyint(x));

        return detail::_f256_runtime::llrint(x);
    }

    // fused arithmetic / min-max
    BL_FORCE_INLINE constexpr f256_s fma_impl(const f256_s& x, const f256_s& y, const f256_s& z)
    {
        return canonicalize_math_result(x * y + z);
    }
    BL_FORCE_INLINE constexpr f256_s fmin_impl(const f256_s& a, const f256_s& b)
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a < b) return a;
        if (b < a) return b;
        if (iszero(a) && iszero(b))
            return signbit_impl(a) ? a : b;
        return a;
    }
    BL_FORCE_INLINE constexpr f256_s fmax_impl(const f256_s& a, const f256_s& b)
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a > b) return a;
        if (b > a) return b;
        if (iszero(a) && iszero(b))
            return signbit_impl(a) ? b : a;
        return a;
    }
    BL_FORCE_INLINE constexpr f256_s fdim_impl(const f256_s& x, const f256_s& y)
    {
        return (x > y) ? canonicalize_math_result(x - y) : f256_s{ 0.0 };
    }
    BL_FORCE_INLINE constexpr f256_s copysign_impl(const f256_s& x, const f256_s& y)
    {
        return signbit_impl(x) == signbit_impl(y) ? x : -x;
    }

    // erf / erfc
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_cheb_eval(const f256_s& x, const f256_s* coeffs, double shift)
    {
        const f256_s t = sub_inline(mul_double_inline(x, 2.0), f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (int i = static_cast<int>(f256_erf_cheb_coeff_count) - 1; i >= 1; --i)
        {
            const f256_s b0 = add_inline(
                sub_inline(mul_double_inline(mul_inline(t, b1), 2.0), b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return add_inline(sub_inline(mul_inline(t, b1), b2), coeffs[0]);
    }
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_cheb(const f256_s& x)
    {
        if (x < f256_s{ 1.0 })
            return erf_cheb_eval(x, f256_erf_cheb_0_1, 1.0);
        if (x < f256_s{ 2.0 })
            return erf_cheb_eval(x, f256_erf_cheb_1_2, 3.0);
        return erf_cheb_eval(x, f256_erf_cheb_2_3, 5.0);
    }
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_series(const f256_s& x)
    {
        const f256_s xx = mul_inline(x, x);
        f256_s power = x;
        f256_s sum = x;

        for (int n = 1; n < 512; ++n)
        {
            power = mul_inline(
                power,
                div_inline(-xx, f256_s{ static_cast<double>(n) }));

            const f256_s term = div_inline(
                power,
                f256_s{ static_cast<double>(2 * n + 1) });

            sum = add_inline(sum, term);
            if (abs(term) < f256_s::eps())
                break;
        }

        return mul_inline(
            mul_inline(f256_s{ 2.0 }, inv_sqrtpi),
            sum);
    }
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cheb_3_4(const f256_s& x)
    {
        return erf_cheb_eval(x, f256_erfc_cheb_3_4, 7.0);
    }
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cf(const f256_s& x)
    {
        const f256_s z = x * x;
        constexpr f256_s a = f256_s{ 0.5 };
        constexpr f256_s tiny = f256_s{ 1.0e-300, 0.0, 0.0, 0.0 };

        f256_s b = z + f256_s{ 1.0 } - a;
        f256_s c = f256_s{ 1.0 } / tiny;
        f256_s d = f256_s{ 1.0 } / b;
        f256_s h = d;

        for (int i = 1; i <= 160; ++i)
        {
            const f256_s ii = f256_s{ static_cast<double>(i) };
            const f256_s an = -(ii * (ii - a));

            b += f256_s{ 2.0 };

            d = an * d + b;
            if (abs(d) < tiny)
                d = tiny;

            c = b + an / c;
            if (abs(c) < tiny)
                c = tiny;

            d = f256_s{ 1.0 } / d;
            const f256_s delta = d * c;
            h *= delta;

            if (abs(delta - f256_s{ 1.0 }) <= f256_s{ 64.0 } * f256_s::eps())
                break;
        }

        return mul_inline(
            mul_inline(_exp(-z), x),
            mul_inline(inv_sqrtpi, h));
    }

    // gamma
    BL_MSVC_NOINLINE constexpr f256_s lgamma1p_series(const f256_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f256_s p = lgamma1p_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = p * y + lgamma1p_coeff[i];

        return y * (-egamma + y * p);
    }
    BL_MSVC_NOINLINE constexpr bool try_lgamma_near_one_or_two(const f256_s& x, f256_s& out) noexcept
    {
        const f256_s y1 = x - f256_s{ 1.0 };
        if (abs(y1) <= f256_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f256_s y2 = x - f256_s{ 2.0 };
        if (abs(y2) <= f256_s{ 0.25 })
        {
            out = f256_log1p_series_reduced(y2) + lgamma1p_series(y2);
            return true;
        }

        return false;
    }
    BL_MSVC_NOINLINE constexpr void positive_recurrence_product(const f256_s& x, const f256_s& asymptotic_min, f256_s& z, f256_s& product, int& product_exp2) noexcept
    {
        z = x;
        product = f256_s{ 1.0 };
        product_exp2 = 0;

        while (z < asymptotic_min)
        {
            product *= z;

            const double hi = product.x0;
            if (hi != 0.0)
            {
                const int e = frexp_exponent_constexpr(hi);
                if (e > 512 || e < -512)
                {
                    product = ldexp(product, -e);
                    product_exp2 += e;
                }
            }

            z += f256_s{ 1.0 };
        }
    }
    BL_MSVC_NOINLINE constexpr f256_s lgamma_stirling_asymptotic(const f256_s& z) noexcept
    {
        const f256_s inv = f256_s{ 1.0 } / z;
        const f256_s inv2 = inv * inv;

        f256_s series = inv / f256_s{ 12.0 };
        f256_s invpow = inv * inv2;

        series -= invpow / f256_s{ 360.0 };
        invpow *= inv2;
        series += invpow / f256_s{ 1260.0 };
        invpow *= inv2;
        series -= invpow / f256_s{ 1680.0 };
        invpow *= inv2;
        series += invpow / f256_s{ 1188.0 };
        invpow *= inv2;
        series -= invpow * (f256_s{ 691.0 } / f256_s{ 360360.0 });
        invpow *= inv2;
        series += invpow / f256_s{ 156.0 };
        invpow *= inv2;
        series -= invpow * (f256_s{ 3617.0 } / f256_s{ 122400.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 43867.0 } / f256_s{ 244188.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 174611.0 } / f256_s{ 125400.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 77683.0 } / f256_s{ 5796.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 236364091.0 } / f256_s{ 1506960.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 657931.0 } / f256_s{ 300.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 3392780147.0 } / f256_s{ 93960.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 1723168255201.0 } / f256_s{ 2492028.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 7709321041217.0 } / f256_s{ 505920.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 151628697551.0 } / f256_s{ 3960.0 });
        invpow *= inv2;

        const f256_s b28_num = to_f256(std::uint64_t{ 2631527155305347737 }) * f256_s{ 10.0 } + f256_s{ 3.0 };
        series -= invpow * (b28_num / f256_s{ 5609403360.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 154210205991661.0 } / f256_s{ 444.0 });

        return (z - f256_s{ 0.5 }) * log(z) - z + half_log_two_pi + series;
    }
    BL_MSVC_NOINLINE constexpr f256_s lgamma_positive_recurrence(const f256_s& x) noexcept
    {
        f256_s near_value{};
        if (try_lgamma_near_one_or_two(x, near_value))
            return near_value;

        constexpr f256_s asymptotic_min = f256_s{ 128.0 };

        f256_s z{};
        f256_s product{};
        int product_exp2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_exp2);

        return lgamma_stirling_asymptotic(z)
            - log(product)
            - f256_s{ static_cast<double>(product_exp2) } * ln2;
    }
    BL_MSVC_NOINLINE constexpr f256_s gamma_positive_recurrence(const f256_s& x) noexcept
    {
        f256_s near_lgamma{};
        if (try_lgamma_near_one_or_two(x, near_lgamma))
            return exp(near_lgamma);

        constexpr f256_s asymptotic_min = f256_s{ 128.0 };

        f256_s z{};
        f256_s product{};
        int product_exp2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_exp2);

        f256_s result = exp(lgamma_stirling_asymptotic(z)) / product;
        if (product_exp2 != 0)
            result = ldexp(result, -product_exp2);

        return result;
    }
}

/// ============= f256 inline public wrappers =============

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept { return detail::_f256::log_as_double_impl(a); }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fabs(const f256_s& a) noexcept { return detail::_f256::fabs_impl(a); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(const f256_s& x) noexcept { return detail::_f256::signbit_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(const f256_s& x) noexcept { return detail::_f256::fpclassify_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(const f256_s& x) noexcept { return detail::_f256::isnormal_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::isunordered_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::isgreater_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::isgreaterequal_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::isless_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::islessequal_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(const f256_s& a, const f256_s& b) noexcept { return detail::_f256::islessgreater_impl(a, b); }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s rint(const f256_s& x) { BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::nearbyint(x), detail::_f256_runtime::rint(x)); }
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f256_s& x) { BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x)), detail::_f256_runtime::lround(x)); }
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f256_s& x) { BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x)), detail::_f256_runtime::llround(x)); }
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f256_s& x) { BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_constexpr::nearbyint(x)), detail::_f256_runtime::lrint(x)); }
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f256_s& x) { BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_constexpr::nearbyint(x)), detail::_f256_runtime::llrint(x)); }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s hypot(const f256_s& x, const f256_s& y)
{
#if BL_CONSTEXPR_RUNTIME_DISPATCH_USES_CONSTEVAL
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256::hypot_impl(x, y), detail::_f256_runtime::hypot(x, y));
#else
    return detail::_f256::hypot_impl(x, y);
#endif
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fma(const f256_s& x, const f256_s& y, const f256_s& z) { return detail::_f256::fma_impl(x, y, z); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmin(const f256_s& a, const f256_s& b) { return detail::_f256::fmin_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmax(const f256_s& a, const f256_s& b) { return detail::_f256::fmax_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fdim(const f256_s& x, const f256_s& y) { return detail::_f256::fdim_impl(x, y); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s copysign(const f256_s& x, const f256_s& y) { return detail::_f256::copysign_impl(x, y); }

/// ============= f256 constexpr implementations =============

[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::fmod(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax < ay)
        return x;

    f256_s fast{};
    if (y.x1 == 0.0 && y.x2 == 0.0 && y.x3 == 0.0 && fmod_fast_double_divisor_abs(ax, ay.x0, fast))
    {
        if (iszero(fast))
            return f256_s{ signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        const f256_s out = ispositive(x) ? fast : -fast;
        return canonicalize_math_result(out);
    }

    const f256_s out = bl::use_constexpr_math()
        ? fmod_exact(x, y)
        : fmod_runtime(x, y);

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::round(const f256_s& a)
{
    return round_half_away_zero(a);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::round_to_decimals(f256_s v, int prec)
{
    constexpr int local_capacity = std::numeric_limits<f256_s>::max_digits10;

    if (prec <= 0) return v;
    if (prec > local_capacity) prec = local_capacity;

    constexpr f256_s inv10_qd{
         0x1.999999999999ap-4,
        -0x1.999999999999ap-58,
         0x1.999999999999ap-112,
        -0x1.999999999999ap-166
    };

    char digits[local_capacity];

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f256_s ip = detail::_f256_constexpr::floor(v);
    f256_s frac = v - ip;

    f256_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;

        int di = static_cast<int>(detail::_f256_constexpr::floor(w).x0);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = w - f256_s{ static_cast<double>(di) };
    }

    f256_s la = w * 10.0;

    const f256_s tie_slop = f256_s::eps() * f256_s{ 65536.0 };
    int next = static_cast<int>(detail::_f256_constexpr::floor(la).x0);
    if (next < 0) next = 0;

    f256_s rem = la - f256_s{ static_cast<double>(next) };
    if (next < 10 && rem >= f256_s{ 1.0 } - tie_slop)
    {
        ++next;
        rem -= f256_s{ 1.0 };
    }

    const int last = digits[prec - 1] - '0';
    const bool beyond_half = rem > tie_slop;
    const bool round_up =
        (next > 5) ||
        (next == 5 && (beyond_half || (last & 1)));

    if (round_up)
    {
        int i = prec - 1;
        for (; i >= 0; --i)
        {
            if (digits[i] == '9')
            {
                digits[i] = '0';
            }
            else
            {
                ++digits[i];
                break;
            }
        }

        if (i < 0)
            ip = ip + 1.0;
    }

    f256_s exact_out{};
    if (try_rounded_decimal_to_f256(ip, digits, prec, neg, exact_out))
        return exact_out;

    f256_s frac_val{ 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = frac_val + f256_s{ static_cast<double>(digits[i] - '0') };
        frac_val = frac_val * inv10_qd;
    }

    f256_s out = ip + frac_val;
    return neg ? -out : out;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::sqrt(const f256_s& a)
{
    using namespace detail::_f256;

    if (a.x0 <= 0.0)
    {
        if (iszero(a))
            return a;
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    if (isinf(a))
        return a;

    if (bl::use_constexpr_math())
        return sqrt_constexpr_impl(a);

    return sqrt_runtime_impl(a);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::nearbyint(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f256_s t = detail::_f256_constexpr::floor(a);
    const f256_s frac = sub_inline(a, t);

    if (frac < f256_s{ 0.5 })
        return t;

    if (frac > f256_s{ 0.5 })
    {
        t = add_inline(t, f256_s{ 1.0 });
        if (iszero(t))
            return f256_s{ signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return t;
    }

    if (is_odd_integer(t))
        t = add_inline(t, f256_s{ 1.0 });

    if (iszero(t))
        return f256_s{ signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return t;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::ldexp(const f256_s& a, int e)
{
    return canonicalize_math_result(_ldexp(a, e));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::exp(const f256_s& x)
{
    return canonicalize_math_result(_exp(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::exp2(const f256_s& x)
{
    return canonicalize_math_result(_exp2(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::log(const f256_s& a)
{
    return canonicalize_math_result(_log(a));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::log2(const f256_s& a)
{
    return canonicalize_math_result(_log(a) * inv_ln2);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::log10(const f256_s& a)
{
    return canonicalize_math_result(_log(a) / ln10);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::pow(const f256_s& x, const f256_s& y)
{
    if (iszero(y))
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s yi = detail::_f256_constexpr::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && f256_try_get_int64(yi, yi64))
        return powi(x, yi64);

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = _exp(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return canonicalize_math_result(_exp(mul_inline(y, _log(x))));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::pow(const f256_s& x, double y)
{
    if (y == 0.0)
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (y == 1.0) return x;
    if (y == 2.0) return canonicalize_math_result(x * x);
    if (y == -1.0) return canonicalize_math_result(f256_s{ 1.0 } / x);
    if (y == 0.5) return canonicalize_math_result(detail::_f256_constexpr::sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? ceil_constexpr(y)
            : floor_constexpr(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && absd(yi) < 0x1p63)
        return powi(x, static_cast<int64_t>(yi));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = _exp(mul_double_inline(_log(-x), y));
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return canonicalize_math_result(_exp(mul_double_inline(_log(x), y)));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr bool   detail::_f256_constexpr::sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    bool ret = _sincos(x, s_out, c_out);
    s_out = canonicalize_math_result(s_out);
    c_out = canonicalize_math_result(c_out);
    return ret;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::sin(const f256_s& x)
{
    const double ax = fabs_constexpr(x.x0);
    if (!isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(pi_4))
        return f256_sin_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!f256_remainder_pi2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return f256_s{ detail::fp::sin_constexpr(static_cast<double>(x)) };
        }
        else
        {
            return f256_s{ std::sin(static_cast<double>(x)) };
        }
    }
    switch ((int)(n & 3LL))
    {
    case 0: return f256_sin_kernel_pi4(r);
    case 1: return f256_cos_kernel_pi4(r);
    case 2: return -f256_sin_kernel_pi4(r);
    default: return -f256_cos_kernel_pi4(r);
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::cos(const f256_s& x)
{
    const double ax = fabs_constexpr(x.x0);
    if (!isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(pi_4))
        return f256_cos_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!f256_remainder_pi2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return f256_s{ detail::fp::cos_constexpr(static_cast<double>(x)) };
        }
        else
        {
            return f256_s{ std::cos(static_cast<double>(x)) };
        }
    }

    switch ((int)(n & 3LL))
    {
    case 0: return f256_cos_kernel_pi4(r);
    case 1: return -f256_sin_kernel_pi4(r);
    case 2: return -f256_cos_kernel_pi4(r);
    default: return f256_sin_kernel_pi4(r);
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::tan(const f256_s& x)
{
    f256_s s{}, c{};
    if (_sincos(x, s, c))
        return canonicalize_math_result(s / c);

    if (bl::use_constexpr_math())
    {
        return canonicalize_math_result(f256_s{ detail::fp::tan_constexpr(static_cast<double>(x)) });
    }
    else
    {
        return canonicalize_math_result(f256_s{ std::tan(static_cast<double>(x)) });
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::atan(const f256_s& x)
{
    return canonicalize_math_result(_atan(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::atan2(const f256_s& y, const f256_s& x)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (iszero(x))
    {
        if (iszero(y))
            return f256_s{ std::numeric_limits<double>::quiet_NaN() };
        return ispositive(y) ? pi_2 : -pi_2;
    }

    if (iszero(y))
    {
        if (x.x0 < 0.0)
            return signbit_constexpr(y.x0) ? -pi : pi;
        return y;
    }

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax == ay)
    {
        if (x.x0 < 0.0)
        {
            return canonicalize_math_result(
                (y.x0 < 0.0) ? -pi_3_4 : pi_3_4);
        }

        return canonicalize_math_result(
            (y.x0 < 0.0) ? -pi_4 : pi_4);
    }

    if (ax >= ay)
    {
        f256_s a = _atan(y / x);

        if (x.x0 < 0.0)
            a += (y.x0 < 0.0) ? -pi : pi;
        return canonicalize_math_result(a);
    }

    f256_s a = _atan(x / y);
    return canonicalize_math_result((y.x0 < 0.0) ? (-pi_2 - a) : (pi_2 - a));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::asin(const f256_s& x)
{
    return canonicalize_math_result(_asin(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::acos(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_acos(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::expm1(const f256_s& x)
{
    return canonicalize_math_result(_expm1(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::log1p(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ -1.0 })
        return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (x < f256_s{ -1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(x))
        return x;
    if (iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (ax <= f256_s{ 0.5 })
        return canonicalize_math_result(f256_log1p_series_reduced(x));

    const f256_s u = f256_s{ 1.0 } + x;
    if ((u - f256_s{ 1.0 }) == x)
        return canonicalize_math_result(detail::_f256_constexpr::log(u));

    if (x > f256_s{ 0.0 } && x <= f256_s{ 1.0 })
    {
        const f256_s t = x / (f256_s{ 1.0 } + detail::_f256_constexpr::sqrt(f256_s{ 1.0 } + x));
        return canonicalize_math_result(f256_log1p_series_reduced(t) * f256_s{ 2.0 });
    }

    if (x > f256_s{ 0.0 })
        return canonicalize_math_result(detail::_f256_constexpr::log(u));

    const f256_s y = u - f256_s{ 1.0 };
    if (iszero(y))
        return x;

    return canonicalize_math_result(detail::_f256_constexpr::log(u) * (x / y));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::sinh(const f256_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (!bl::use_constexpr_math())
    {
        const f256_s em1 = _expm1(ax);
        f256_s out = div_inline(
            mul_inline(em1, add_scalar_precise(em1, 2.0)),
            mul_double_inline(add_scalar_precise(em1, 1.0), 2.0));
        if (signbit(x))
            out = -out;
        return canonicalize_math_result(out);
    }

    if (ax <= f256_s{ 0.5 })
    {
        const f256_s x2 = mul_inline(x, x);
        f256_s term = x;
        f256_s sum = x;

        for (int n = 1; n <= 256; ++n)
        {
            term = div_double_inline(
                mul_inline(term, x2),
                static_cast<double>((2 * n) * (2 * n + 1)));
            sum = add_inline(sum, term);

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }

        return canonicalize_math_result(sum);
    }

    const f256_s ex = _exp(ax);
    const f256_s inv_ex = recip(ex);
    f256_s out = mul_double_inline(sub_inline(ex, inv_ex), 0.5);
    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::cosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s ax = abs(x);
    const f256_s ex = _exp(ax);
    const f256_s inv_ex = recip(ex);
    return canonicalize_math_result(
        mul_double_inline(add_inline(ex, inv_ex), 0.5));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::tanh(const f256_s& x)
{
    using namespace detail::_f256;
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s ax = abs(x);
    if (ax > f256_s{ 20.0 })
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s em1 = _expm1(mul_double_inline(ax, 2.0));
    const f256_s denom = add_scalar_precise(em1, 2.0);

    f256_s out = div_inline(em1, denom);

    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::asinh(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f256_s ax = abs(x);
    f256_s out{};
    if (ax > f256_s{ 0x1p500 })
        out = detail::_f256_constexpr::log(ax) + ln2;
    else
        out = detail::_f256_constexpr::log(ax + detail::_f256_constexpr::sqrt(ax * ax + f256_s{ 1.0 }));

    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::acosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x < f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (x == f256_s{ 1.0 })
        return f256_s{ 0.0 };
    if (isinf(x))
        return x;

    f256_s out{};
    if (x > f256_s{ 0x1p500 })
        out = detail::_f256_constexpr::log(x) + ln2;
    else
        out = detail::_f256_constexpr::log(x + detail::_f256_constexpr::sqrt((x - f256_s{ 1.0 }) * (x + f256_s{ 1.0 })));

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::atanh(const f256_s& x)
{
    if (isnan(x) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (ax > f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (ax == f256_s{ 1.0 })
        return signbit(x)
        ? f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }
        : f256_s{ std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };

    if (ax <= f256_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
            return canonicalize_math_result(atanh_small_series_constexpr(x));

        return canonicalize_math_result(atanh_small_series_runtime(x));
    }

    const f256_s out = detail::_f256_constexpr::log((f256_s{ 1.0 } + x) / (f256_s{ 1.0 } - x)) * f256_s { 0.5 };
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::cbrt(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s y{};
    if (bl::use_constexpr_math())
    {
        y = detail::_f256_constexpr::exp(detail::_f256_constexpr::log(ax) / f256_s{ 3.0 });
    }
    else
    {
        int exp2 = 0;
        double mantissa = std::frexp(ax.x0, &exp2);
        int rem = exp2 % 3;
        if (rem < 0)
            rem += 3;
        if (rem != 0)
        {
            mantissa = std::ldexp(mantissa, rem);
            exp2 -= rem;
        }

        y = f256_s{ std::cbrt(mantissa), 0.0, 0.0, 0.0 };
        if (exp2 != 0)
            y = detail::_f256_constexpr::ldexp(y, exp2 / 3);
    }

    const auto cbrt_newton_step = [&](const f256_s& current) constexpr -> f256_s
    {
        const f256_s current_squared = mul_inline(current, current);
        const f256_s quotient = div_inline(ax, current_squared);
        return div_double_inline(add_inline(add_inline(current, current), quotient), 3.0);
    };
    const auto cbrt_tail_step = [&](const f256_s& current) constexpr -> f256_s
    {
        const double inv_derivative = 1.0 / (3.0 * current.x0 * current.x0);
        const f256_s current_squared = mul_inline(current, current);
        const f256_s residual = sub_inline(ax, mul_inline(current_squared, current));
        return add_inline(current, mul_double_inline(residual, inv_derivative));
    };

    if (bl::use_constexpr_math())
    {
        y = cbrt_newton_step(y);
        y = cbrt_newton_step(y);
        y = cbrt_newton_step(y);
    }
    else
    {
        y = cbrt_tail_step(y);
        y = cbrt_tail_step(y);
        y = cbrt_tail_step(y);
    }

    if (neg)
        y = -y;

    return canonicalize_math_result(y);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::remquo(const f256_s& x, const f256_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = nearest_integer_ties_even(x / y);
    f256_s r = x - n * y;

    if (quo)
        *quo = low_quotient_bits(n);

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return canonicalize_math_result(r);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::remainder(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = nearest_integer_ties_even(x / y);
    f256_s r = x - n * y;

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return canonicalize_math_result(r);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::frexp(const f256_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const double lead =
        (x.x0 != 0.0) ? x.x0 :
        (x.x1 != 0.0) ? x.x1 :
        (x.x2 != 0.0) ? x.x2 : x.x3;
    int e = 0;

    if (bl::use_constexpr_math())
        e = detail::fp::frexp_exponent_constexpr(lead);
    else
        (void)std::frexp(lead, &e);

    f256_s m = detail::_f256_constexpr::ldexp(x, -e);
    const f256_s am = abs(m);

    if (am < f256_s{ 0.5 })
    {
        m *= f256_s{ 2.0 };
        --e;
    }
    else if (am >= f256_s{ 1.0 })
    {
        m *= f256_s{ 0.5 };
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::modf(const f256_s& x, f256_s* iptr) noexcept
{
    const f256_s i = detail::_f256_constexpr::trunc(x);
    if (iptr)
        *iptr = i;

    f256_s frac = x - i;
    if (iszero(frac))
        frac = f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
    return frac;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr int    detail::_f256_constexpr::ilogb(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_ILOGBNAN;
    if (iszero(x)) return FP_ILOGB0;
    if (isinf(x))  return std::numeric_limits<int>::max();

    int e = 0;
    (void)detail::_f256_constexpr::frexp(abs(x), &e);
    return e - 1;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::logb(const f256_s& x) noexcept
{
    if (isnan(x))  return x;
    if (iszero(x)) return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (isinf(x))  return std::numeric_limits<f256_s>::infinity();

    return f256_s{ static_cast<double>(detail::_f256_constexpr::ilogb(x)), 0.0, 0.0, 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::scalbn(const f256_s& x, int e) noexcept
{
    return detail::_f256_constexpr::ldexp(x, e);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::scalbln(const f256_s& x, long e) noexcept
{
    return detail::_f256_constexpr::ldexp(x, static_cast<int>(e));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::nextafter(const f256_s& from, const f256_s& to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (from == to)
        return to;
    if (iszero(from))
        return signbit(to)
        ? f256_s{ -std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 }
        : f256_s{ std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 };
    if (isinf(from))
        return signbit(from)
        ? -std::numeric_limits<f256_s>::max()
        : std::numeric_limits<f256_s>::max();

    const double toward = (from < to)
        ? std::numeric_limits<double>::infinity()
        : -std::numeric_limits<double>::infinity();

    return normalize_nextafter_tail(
        from,
        nextafter_double_constexpr(from.x3, toward));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::nexttoward(const f256_s& from, long double to) noexcept
{
    return detail::_f256_constexpr::nextafter(from, f256_s{ static_cast<double>(to), 0.0, 0.0, 0.0 });
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    return detail::_f256_constexpr::nextafter(from, to);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::erf(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s out{};

    if (ax < f256_s{ 1.0 })
    {
        out = erf_positive_series(ax);
    }
    else if (ax < f256_s{ 3.0 })
    {
        out = erf_positive_cheb(ax);
    }
    else if (ax < f256_s{ 4.0 })
    {
        out = f256_s{ 1.0 } - erfc_positive_cheb_3_4(ax);
    }
    else
    {
        out = f256_s{ 1.0 } - erfc_positive_cf(ax);
    }

    if (neg)
        out = -out;

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_constexpr::erfc(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return f256_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f256_s{ 2.0 } : f256_s{ 0.0 };

    if (signbit(x))
        return canonicalize_math_result(f256_s{ 1.0 } + detail::_f256_constexpr::erf(-x));

    if (x < f256_s{ 1.0 })
        return canonicalize_math_result(f256_s{ 1.0 } - erf_positive_series(x));

    if (x < f256_s{ 3.0 })
        return canonicalize_math_result(f256_s{ 1.0 } - erf_positive_cheb(x));

    if (x < f256_s{ 4.0 })
        return canonicalize_math_result(erfc_positive_cheb_3_4(x));

    if (x > f256_s{ 40.0 })
        return f256_s{ 0.0 };

    return canonicalize_math_result(erfc_positive_cf(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::lgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return canonicalize_math_result(lgamma_positive_recurrence(x));

    const f256_s xi = detail::_f256_constexpr::trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::infinity();

    const f256_s sinpix = detail::_f256_constexpr::sin(pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s out =
        detail::_f256_constexpr::log(pi)
        - detail::_f256_constexpr::log(abs(sinpix))
        - lgamma_positive_recurrence(f256_s{ 1.0 } - x);

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f256_s detail::_f256_constexpr::tgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return canonicalize_math_result(gamma_positive_recurrence(x));

    const f256_s xi = detail::_f256_constexpr::trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s sinpix = detail::_f256_constexpr::sin(pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s out = pi / (sinpix * gamma_positive_recurrence(f256_s{ 1.0 } - x));
    return canonicalize_math_result(out);
}

/// ============= f256 public wrappers =============

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::fmod(x, y), detail::_f256_runtime::fmod(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s round(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::round(a), detail::_f256_runtime::round(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s round_to_decimals(f256_s v, int prec)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::round_to_decimals(v, prec), detail::_f256_runtime::round_to_decimals(v, prec));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sqrt(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::sqrt(a), detail::_f256_runtime::sqrt(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nearbyint(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::nearbyint(a), detail::_f256_runtime::nearbyint(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s ldexp(const f256_s& a, int e)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::ldexp(a, e), detail::_f256_runtime::ldexp(a, e));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s exp(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::exp(x), detail::_f256_runtime::exp(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s exp2(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::exp2(x), detail::_f256_runtime::exp2(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::log(a), detail::_f256_runtime::log(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log2(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::log2(a), detail::_f256_runtime::log2(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log10(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::log10(a), detail::_f256_runtime::log10(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::pow(x, y), detail::_f256_runtime::pow(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::pow(x, y), detail::_f256_runtime::pow(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::sincos(x, s_out, c_out), detail::_f256_runtime::sincos(x, s_out, c_out));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::sin(x), detail::_f256_runtime::sin(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::cos(x), detail::_f256_runtime::cos(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::tan(x), detail::_f256_runtime::tan(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::atan(x), detail::_f256_runtime::atan(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::atan2(y, x), detail::_f256_runtime::atan2(y, x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s asin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::asin(x), detail::_f256_runtime::asin(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s acos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::acos(x), detail::_f256_runtime::acos(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s expm1(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::expm1(x), detail::_f256_runtime::expm1(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s log1p(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::log1p(x), detail::_f256_runtime::log1p(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s sinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::sinh(x), detail::_f256_runtime::sinh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::cosh(x), detail::_f256_runtime::cosh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::tanh(x), detail::_f256_runtime::tanh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s asinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::asinh(x), detail::_f256_runtime::asinh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s acosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::acosh(x), detail::_f256_runtime::acosh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s atanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::atanh(x), detail::_f256_runtime::atanh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s cbrt(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::cbrt(x), detail::_f256_runtime::cbrt(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::remquo(x, y, quo), detail::_f256_runtime::remquo(x, y, quo));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::remainder(x, y), detail::_f256_runtime::remainder(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::frexp(x, exp), detail::_f256_runtime::frexp(x, exp));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::modf(x, iptr), detail::_f256_runtime::modf(x, iptr));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr int    ilogb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::ilogb(x), detail::_f256_runtime::ilogb(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s logb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::logb(x), detail::_f256_runtime::logb(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::scalbn(x, e), detail::_f256_runtime::scalbn(x, e));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::scalbln(x, e), detail::_f256_runtime::scalbln(x, e));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::nextafter(from, to), detail::_f256_runtime::nextafter(from, to));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::nexttoward(from, to), detail::_f256_runtime::nexttoward(from, to));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::nexttoward(from, to), detail::_f256_runtime::nexttoward(from, to));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::erf(x), detail::_f256_runtime::erf(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::erfc(x), detail::_f256_runtime::erfc(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s lgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::lgamma(x), detail::_f256_runtime::lgamma(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s tgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f256_constexpr::tgamma(x), detail::_f256_runtime::tgamma(x));
}

}

#endif
