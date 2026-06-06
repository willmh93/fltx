/**
 * fltx/detail/f256_math_kernels.h - f256 math kernels and helper algorithms.
 *
 * Low-level f256 helper logic used by grouped math implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_DETAIL_MATH_KERNELS_INCLUDED
#define F256_DETAIL_MATH_KERNELS_INCLUDED
#include "fltx/detail/f256_declarations.h"

namespace bl {

namespace detail::_f256 // primitives and kernels
{
    using detail::exact_decimal::add_signed;
    using detail::exact_decimal::biguint;
    using detail::exact_decimal::decompose_double_mantissa;
    using detail::exact_decimal::mod_shift_subtract;
    using detail::exact_decimal::signed_biguint;
    using detail::fp::fmod;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::sqrt_seed;
    using detail::fp::trunc;
    using detail::fp::frexp_exponent_limb;
    using detail::fp::ldexp_limb;

    BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s signed_zero_from(double sign_source) noexcept
    {
        return f256_s{ detail::fp::copysign(0.0, sign_source), 0.0, 0.0, 0.0 };
    }
    BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept;

    // expression evaluation
    BL_FORCE_INLINE constexpr f256_s add_eval(const f256_s& a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            add_inline(a, b),
            detail::_f256_runtime::add(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s sub_eval(const f256_s& a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_inline(a, b),
            detail::_f256_runtime::sub(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s mul_eval(const f256_s& a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_inline(a, b),
            detail::_f256_runtime::mul(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s div_eval(const f256_s& a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            div_inline(a, b),
            detail::_f256_runtime::div(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s add_double_eval(const f256_s& a, double b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            add_double_inline(a, b),
            detail::_f256_runtime::add_double(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s sub_double_eval(const f256_s& a, double b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_double_inline(a, b),
            detail::_f256_runtime::sub_double(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s sub_double_eval(double a, const f256_s& b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_double_inline(a, b),
            detail::_f256_runtime::sub_double(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s mul_double_eval(const f256_s& a, double b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_double_inline(a, b),
            detail::_f256_runtime::mul_double(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s div_double_eval(const f256_s& a, double b) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            div_double_inline(a, b),
            detail::_f256_runtime::div_double(a, b)
        );
    }

    BL_FORCE_INLINE constexpr f256_s sqr_eval(const f256_s& a) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sqr_inline(a),
            detail::_f256_runtime::sqr(a)
        );
    }

    BL_FORCE_INLINE constexpr f256_s mul_add_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_add_inline(a, b, c),
            detail::_f256_runtime::mul_add(a, b, c)
        );
    }

    BL_FORCE_INLINE constexpr f256_s mul_sub_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_sub_inline(a, b, c),
            detail::_f256_runtime::mul_sub(a, b, c)
        );
    }

    BL_FORCE_INLINE constexpr f256_s value_sub_mul_eval(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            value_sub_mul_inline(a, b, c),
            detail::_f256_runtime::value_sub_mul(a, b, c)
        );
    }

    BL_FORCE_INLINE constexpr f256_s add_mul_double_eval(const f256_s& addend, const f256_s& value, double scalar) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            add_mul_double_inline(addend, value, scalar),
            detail::_f256_runtime::add_mul_double(addend, value, scalar)
        );
    }

    BL_FORCE_INLINE constexpr f256_s sub_mul_double_eval(const f256_s& minuend, const f256_s& value, double scalar) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sub_mul_double_inline(minuend, value, scalar),
            detail::_f256_runtime::sub_mul_double(minuend, value, scalar)
        );
    }

    BL_FORCE_INLINE constexpr f256_s mul_double_sub_eval(const f256_s& value, double scalar, const f256_s& subtrahend) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            mul_double_sub_inline(value, scalar, subtrahend),
            detail::_f256_runtime::mul_double_sub(value, scalar, subtrahend)
        );
    }

    BL_FORCE_INLINE constexpr f256_s div_add_double_eval(const f256_s& numerator, const f256_s& base_denominator, double scalar) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            div_add_double_inline(numerator, base_denominator, scalar),
            detail::_f256_runtime::div_add_double(numerator, base_denominator, scalar)
        );
    }

    // scaling helpers
    BL_FORCE_INLINE constexpr f256_s ldexp_terms(const f256_s& value, int exponent) noexcept
    {
        return renorm(
            ldexp_limb(value.x0, exponent),
            ldexp_limb(value.x1, exponent),
            ldexp_limb(value.x2, exponent),
            ldexp_limb(value.x3, exponent));
    }

    struct exact_dyadic_fmod
    {
        int exp2 = 0;
        biguint mant{};
    };

    // exact integer helpers
    BL_FORCE_INLINE constexpr bool biguint_is_odd(const biguint& value)
    {
        return !value.is_zero() && (value.words[0] & 1u) != 0;
    }

    BL_FORCE_INLINE constexpr bool biguint_any_low_bits_set(const biguint& value, int bit_count)
    {
        if (bit_count <= 0)
            return false;

        const int full_words = bit_count >> 5;
        const int rem_bits   = bit_count & 31;

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

    BL_FORCE_INLINE constexpr int biguint_trailing_zero_bits(const biguint& value)
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

    BL_FORCE_INLINE constexpr biguint biguint_shr_bits(biguint value, int bits)
    {
        if (bits <= 0 || value.is_zero())
            return value;

        const int word_shift = bits >> 5;
        const int bit_shift  = bits & 31;

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

    BL_FORCE_INLINE constexpr biguint biguint_mod(const biguint& numerator, const biguint& modulus)
    {
        biguint remainder{};
        mod_shift_subtract(numerator, modulus, remainder);
        return remainder;
    }

    BL_FORCE_INLINE constexpr biguint biguint_mul_mod(const biguint& a, const biguint& b, const biguint& modulus)
    {
        if (a.is_zero() || b.is_zero())
            return {};

        return biguint_mod(mul_big(a, b), modulus);
    }

    BL_FORCE_INLINE constexpr biguint biguint_pow2_mod(int exponent, const biguint& modulus)
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

    struct fmod_u320
    {
        std::uint64_t word[5]{};
    };

    struct exact_dyadic_fmod_fixed
    {
        int exp2 = 0;
        fmod_u320 mant{};
    };

    struct signed_fmod_u320
    {
        bool neg = false;
        fmod_u320 mag{};
    };

    BL_FORCE_INLINE constexpr bool fmod_u320_is_zero(const fmod_u320& value)
    {
        return (value.word[0] | value.word[1] | value.word[2] | value.word[3] | value.word[4]) == 0;
    }

    BL_FORCE_INLINE constexpr bool fmod_u320_is_odd(const fmod_u320& value)
    {
        return (value.word[0] & 1u) != 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u320_compare(const fmod_u320& a, const fmod_u320& b)
    {
        for (int i = 4; i >= 0; --i)
        {
            if (a.word[i] < b.word[i]) return -1;
            if (a.word[i] > b.word[i]) return 1;
        }
        return 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u320_bit_length(const fmod_u320& value)
    {
        for (int i = 4; i >= 0; --i)
        {
            if (value.word[i] != 0)
                return i * 64 + 64 - static_cast<int>(std::countl_zero(value.word[i]));
        }
        return 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u320_trailing_zero_bits(const fmod_u320& value)
    {
        for (int i = 0; i < 5; ++i)
        {
            if (value.word[i] != 0)
                return i * 64 + static_cast<int>(std::countr_zero(value.word[i]));
        }
        return 0;
    }

    BL_FORCE_INLINE constexpr bool fmod_u320_get_bit(const fmod_u320& value, int index)
    {
        if (index < 0 || index >= 320)
            return false;

        return ((value.word[index >> 6] >> (index & 63)) & 1u) != 0;
    }

    BL_FORCE_INLINE constexpr std::uint64_t fmod_u320_get_bits(const fmod_u320& value, int start, int count)
    {
        if (count <= 0 || start < 0 || start >= 320)
            return 0;

        if (count >= 64)
            count = 64;

        const int word_index = start >> 6;
        const int bit_index = start & 63;
        std::uint64_t out = value.word[word_index] >> bit_index;
        if (bit_index != 0 && word_index + 1 < 5)
            out |= value.word[word_index + 1] << (64 - bit_index);

        if (count < 64)
            out &= (std::uint64_t{ 1 } << count) - 1u;

        return out;
    }

    BL_FORCE_INLINE constexpr bool fmod_u320_any_low_bits_set(const fmod_u320& value, int count)
    {
        if (count <= 0)
            return false;
        if (count >= 320)
            return !fmod_u320_is_zero(value);

        const int full_words = count >> 6;
        const int rem_bits = count & 63;
        for (int i = 0; i < full_words; ++i)
        {
            if (value.word[i] != 0)
                return true;
        }

        if (rem_bits == 0)
            return false;

        const std::uint64_t mask = (std::uint64_t{ 1 } << rem_bits) - 1u;
        return (value.word[full_words] & mask) != 0;
    }

    BL_FORCE_INLINE constexpr void fmod_u320_add_inplace(fmod_u320& a, const fmod_u320& b)
    {
        std::uint64_t carry = 0;
        for (int i = 0; i < 5; ++i)
        {
            const std::uint64_t old = a.word[i];
            std::uint64_t sum = old + b.word[i];
            const std::uint64_t carry0 = sum < old ? 1u : 0u;
            const std::uint64_t before_carry = sum;
            sum += carry;
            a.word[i] = sum;
            carry = carry0 | (sum < before_carry ? 1u : 0u);
        }
    }

    BL_FORCE_INLINE constexpr void fmod_u320_add_small(fmod_u320& a, std::uint32_t value)
    {
        std::uint64_t add = value;
        for (int i = 0; i < 5 && add != 0; ++i)
        {
            const std::uint64_t old = a.word[i];
            a.word[i] += add;
            add = a.word[i] < old ? 1u : 0u;
        }
    }

    BL_FORCE_INLINE constexpr void fmod_u320_sub_inplace(fmod_u320& a, const fmod_u320& b)
    {
        std::uint64_t borrow = 0;
        for (int i = 0; i < 5; ++i)
        {
            const std::uint64_t ai = a.word[i];
            const std::uint64_t bi = b.word[i];
            a.word[i] = ai - bi - borrow;
            borrow = (ai < bi) || (borrow != 0 && ai == bi) ? 1u : 0u;
        }
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_shl_bits(fmod_u320 value, int bits)
    {
        if (bits <= 0 || fmod_u320_is_zero(value))
            return value;
        if (bits >= 320)
            return {};

        fmod_u320 out{};
        const int word_shift = bits >> 6;
        const int bit_shift = bits & 63;

        for (int i = 4; i >= word_shift; --i)
        {
            const int src = i - word_shift;
            std::uint64_t word = value.word[src] << bit_shift;
            if (bit_shift != 0 && src > 0)
                word |= value.word[src - 1] >> (64 - bit_shift);
            out.word[i] = word;
        }
        return out;
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_shr_bits(fmod_u320 value, int bits)
    {
        if (bits <= 0 || fmod_u320_is_zero(value))
            return value;
        if (bits >= 320)
            return {};

        fmod_u320 out{};
        const int word_shift = bits >> 6;
        const int bit_shift = bits & 63;

        for (int i = 0; i + word_shift < 5; ++i)
        {
            const int src = i + word_shift;
            std::uint64_t word = value.word[src] >> bit_shift;
            if (bit_shift != 0 && src + 1 < 5)
                word |= value.word[src + 1] << (64 - bit_shift);
            out.word[i] = word;
        }
        return out;
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_shl1(fmod_u320 value)
    {
        std::uint64_t carry = 0;
        for (int i = 0; i < 5; ++i)
        {
            const std::uint64_t next_carry = value.word[i] >> 63;
            value.word[i] = (value.word[i] << 1) | carry;
            carry = next_carry;
        }
        return value;
    }

    BL_FORCE_INLINE constexpr bool fmod_u320_shift_exceeds_capacity(const fmod_u320& value, int bits)
    {
        return bits > 0 && !fmod_u320_is_zero(value) &&
            fmod_u320_bit_length(value) + bits > 320;
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_from_u64(std::uint64_t value)
    {
        fmod_u320 out{};
        out.word[0] = value;
        return out;
    }

    BL_FORCE_INLINE constexpr void fmod_u320_add_signed(
        signed_fmod_u320& acc,
        const fmod_u320& value,
        bool value_neg)
    {
        if (fmod_u320_is_zero(value))
            return;

        if (fmod_u320_is_zero(acc.mag))
        {
            acc.neg = value_neg;
            acc.mag = value;
            return;
        }

        if (acc.neg == value_neg)
        {
            fmod_u320_add_inplace(acc.mag, value);
            return;
        }

        const int cmp = fmod_u320_compare(acc.mag, value);
        if (cmp >= 0)
        {
            fmod_u320_sub_inplace(acc.mag, value);
            if (fmod_u320_is_zero(acc.mag))
                acc.neg = false;
            return;
        }

        fmod_u320 diff = value;
        fmod_u320_sub_inplace(diff, acc.mag);
        acc.neg = value_neg;
        acc.mag = diff;
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_mod_shift_subtract_with_quotient_mod(
        fmod_u320 numerator,
        const fmod_u320& denominator,
        std::uint64_t& quotient_mod)
    {
        quotient_mod = 0;
        if (fmod_u320_is_zero(denominator))
            return {};
        if (fmod_u320_compare(numerator, denominator) < 0)
            return numerator;

        int shift = fmod_u320_bit_length(numerator) - fmod_u320_bit_length(denominator);
        fmod_u320 shifted = fmod_u320_shl_bits(denominator, shift);

        for (; shift >= 0; --shift)
        {
            if (fmod_u320_compare(numerator, shifted) >= 0)
            {
                fmod_u320_sub_inplace(numerator, shifted);
                if (shift < 31)
                    quotient_mod |= std::uint64_t{ 1 } << shift;
            }
            shifted = fmod_u320_shr_bits(shifted, 1);
        }
        return numerator;
    }

    BL_FORCE_INLINE constexpr fmod_u320 fmod_u320_double_mod_with_quotient_bit(
        fmod_u320 value,
        const fmod_u320& modulus,
        std::uint64_t& bit)
    {
        const bool overflow = (value.word[4] >> 63) != 0u;
        value = fmod_u320_shl1(value);
        if (overflow || fmod_u320_compare(value, modulus) >= 0)
        {
            fmod_u320_sub_inplace(value, modulus);
            bit = 1;
        }
        else
        {
            bit = 0;
        }
        return value;
    }

    BL_FORCE_INLINE constexpr void normalize_exact_dyadic_fmod_fixed(exact_dyadic_fmod_fixed& value)
    {
        if (fmod_u320_is_zero(value.mant))
        {
            value.exp2 = 0;
            return;
        }

        const int tz = fmod_u320_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = fmod_u320_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }

    BL_NO_INLINE constexpr bool exact_from_f256_fmod_fixed(
        const f256_s& x,
        exact_dyadic_fmod_fixed& out)
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

        out = {};
        if (common_exp == std::numeric_limits<int>::max())
            return true;

        signed_fmod_u320 acc{};
        for (double limb : limbs)
        {
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            const int shift = exponent - common_exp;
            fmod_u320 term = fmod_u320_from_u64(mantissa);
            if (fmod_u320_shift_exceeds_capacity(term, shift))
                return false;
            term = fmod_u320_shl_bits(term, shift);
            fmod_u320_add_signed(acc, term, limb_neg);
        }

        if (acc.neg)
            return false;

        out.exp2 = common_exp;
        out.mant = acc.mag;
        normalize_exact_dyadic_fmod_fixed(out);
        return true;
    }

    BL_NO_INLINE constexpr f256_s exact_dyadic_to_f256_fmod_fixed(
        const fmod_u320& coeff,
        int exp2,
        bool neg)
    {
        if (fmod_u320_is_zero(coeff))
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        constexpr int kept_bits = 53 * 5;
        int ratio_exp = fmod_u320_bit_length(coeff) - 1;
        fmod_u320 q = coeff;

        if (ratio_exp > (kept_bits - 1))
        {
            const int right_shift = ratio_exp - (kept_bits - 1);
            const bool round_bit = fmod_u320_get_bit(q, right_shift - 1);
            const bool sticky = fmod_u320_any_low_bits_set(q, right_shift - 1);

            q = fmod_u320_shr_bits(q, right_shift);

            if (round_bit && (sticky || fmod_u320_is_odd(q)))
                fmod_u320_add_small(q, 1u);

            if (fmod_u320_bit_length(q) > kept_bits)
            {
                q = fmod_u320_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < (kept_bits - 1))
        {
            q = fmod_u320_shl_bits(q, (kept_bits - 1) - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f256_s>::infinity() : std::numeric_limits<f256_s>::infinity();
        if (e2 < -1074)
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        const std::uint64_t c4 = fmod_u320_get_bits(q, 0, 53);
        const std::uint64_t c3 = fmod_u320_get_bits(q, 53, 53);
        const std::uint64_t c2 = fmod_u320_get_bits(q, 106, 53);
        const std::uint64_t c1 = fmod_u320_get_bits(q, 159, 53);
        const std::uint64_t c0 = fmod_u320_get_bits(q, 212, 53);

        const double x0 = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp(static_cast<double>(c3), e2 - 211) : 0.0;
        const double x4 = c4 ? detail::fp::ldexp(static_cast<double>(c4), e2 - 264) : 0.0;

        f256_s out = renorm5(x0, x1, x2, x3, x4);
        return neg ? -out : out;
    }

    BL_NO_INLINE constexpr bool fmod_exact_fixed_limb_abs_with_quotient_mod(
        const f256_s& ax,
        const f256_s& ay,
        std::uint64_t& quotient_mod,
        f256_s& out)
    {
        constexpr std::uint64_t quotient_mask = 0x7fffffffull;

        exact_dyadic_fmod_fixed dx{};
        exact_dyadic_fmod_fixed dy{};
        if (!exact_from_f256_fmod_fixed(ax, dx) ||
            !exact_from_f256_fmod_fixed(ay, dy))
        {
            return false;
        }

        fmod_u320 remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            if (fmod_u320_shift_exceeds_capacity(dy.mant, shift))
            {
                remainder = dx.mant;
                quotient_mod = 0;
            }
            else
            {
                const fmod_u320 denominator = fmod_u320_shl_bits(dy.mant, shift);
                remainder = fmod_u320_mod_shift_subtract_with_quotient_mod(dx.mant, denominator, quotient_mod);
            }
            out_exp = dx.exp2;
        }
        else
        {
            remainder = fmod_u320_mod_shift_subtract_with_quotient_mod(dx.mant, dy.mant, quotient_mod);
            const int shift = dx.exp2 - dy.exp2;

            int i = 0;
            for (; i < shift && !fmod_u320_is_zero(remainder); ++i)
            {
                std::uint64_t bit = 0;
                remainder = fmod_u320_double_mod_with_quotient_bit(remainder, dy.mant, bit);
                quotient_mod = ((quotient_mod << 1) | bit) & quotient_mask;
            }

            const int remaining = shift - i;
            if (remaining >= 31)
                quotient_mod = 0;
            else if (remaining > 0)
                quotient_mod = (quotient_mod << remaining) & quotient_mask;

            out_exp = dy.exp2;
        }

        out = exact_dyadic_to_f256_fmod_fixed(remainder, out_exp, false);
        if (iszero(out))
            out = f256_s{ 0.0 };
        return true;
    }

    // exact fmod conversion
    BL_FORCE_INLINE constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
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
            const bool sticky    = biguint_any_low_bits_set(q, right_shift - 1);

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

        const double x0 = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp(static_cast<double>(c3), e2 - 211) : 0.0;
        const double x4 = c4 ? detail::fp::ldexp(static_cast<double>(c4), e2 - 264) : 0.0;

        f256_s out = renorm5(x0, x1, x2, x3, x4);
        return neg ? -out : out;
    }

    // fmod kernels
    BL_MSVC_NOINLINE constexpr f256_s fmod_exact(const f256_s& x, const f256_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f256_fmod(mag(x));
        const exact_dyadic_fmod dy = exact_from_f256_fmod(mag(y));

        if (dx.mant.is_zero() || dy.mant.is_zero())
            return signed_zero_from(x.x0);

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
            return signed_zero_from(x.x0);
        return out;
    }

    BL_FORCE_INLINE constexpr bool fmod_normalize_remainder(f256_s& r, const f256_s& modulus) noexcept
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

    BL_FORCE_INLINE constexpr bool fmod_normalize_remainder_with_quotient(
        f256_s& r,
        const f256_s& modulus,
        std::uint64_t& quotient) noexcept
    {
        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
            {
                r += modulus;
                if (quotient == 0u)
                    return false;
                --quotient;
                continue;
            }

            if (r >= modulus)
            {
                r -= modulus;
                ++quotient;
                continue;
            }

            return true;
        }

        return r >= 0.0 && r < modulus;
    }

    BL_FORCE_INLINE constexpr void fmod_append_expansion_term(double* expansion, int& count, double value) noexcept
    {
        if (value != 0.0)
            expansion[count++] = value;
    }

    BL_FORCE_INLINE constexpr f256_s fmod_sub_mul_scalar_expansion(const f256_s& r, const f256_s& b, double q) noexcept
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

    BL_FORCE_INLINE constexpr int fmod_compare_remainder_to_half(const f256_s& r_abs, const f256_s& half) noexcept
    {
        const f256_s delta = sub_inline(r_abs, half);
        if (iszero(delta))
            return 0;
        return delta < 0.0 ? -1 : 1;
    }

    BL_FORCE_INLINE constexpr bool fmod_try_small_quotient_abs(
        const f256_s& ax,
        const f256_s& ay,
        double q,
        f256_s& out,
        std::uint64_t& quotient) noexcept
    {
        std::uint64_t candidate = static_cast<std::uint64_t>(q);
        f256_s r = fmod_sub_mul_scalar_expansion(ax, ay, q);
        if (!fmod_normalize_remainder_with_quotient(r, ay, candidate))
            return false;

        const f256_s edge_slack = mul_double_inline(ay, 0x1p-160);
        if (r <= edge_slack || sub_inline(ay, r) <= edge_slack)
            return false;

        out = r;
        quotient = candidate;
        return true;
    }

    BL_MSVC_NOINLINE constexpr bool fmod_fast_small_quotient_abs_with_quotient(
        const f256_s& ax,
        const f256_s& ay,
        f256_s& out,
        std::uint64_t& quotient) noexcept
    {
        if (!(ay.x0 > 0.0) || detail::fp::isinf_or_nan(ay.x0) || !(ax >= ay))
            return false;

        const double q = detail::fp::trunc(ax.x0 / ay.x0);
        if (!(q > 0.0) || q >= 0x1p42)
            return false;

        quotient = static_cast<std::uint64_t>(q);
        f256_s r = fmod_sub_mul_scalar_expansion(ax, ay, q);
        if (!fmod_normalize_remainder_with_quotient(r, ay, quotient))
            return false;

        const f256_s edge_slack = mul_double_inline(ay, 0x1p-160);
        if (r <= edge_slack || sub_inline(ay, r) <= edge_slack)
            return false;

        out = r;
        return true;
    }

    BL_MSVC_NOINLINE constexpr bool fmod_fast_medium_quotient_abs_with_quotient(
        const f256_s& ax,
        const f256_s& ay,
        f256_s& out,
        std::uint64_t& quotient,
        bool refine_quotient = true,
        double quotient_limit = 0x1p53) noexcept
    {
        if (!(ay.x0 > 0.0) || detail::fp::isinf_or_nan(ay.x0) || !(ax >= ay))
            return false;

        const double q = detail::fp::trunc(ax.x0 / ay.x0);
        if (!(q > 0.0) || q >= quotient_limit)
            return false;

        if (fmod_try_small_quotient_abs(ax, ay, q, out, quotient))
            return true;

        if (!refine_quotient)
            return false;

        const f256_s q_floor = detail::_f256_impl::floor(ax / ay);
        if (q_floor.x1 != 0.0 || q_floor.x2 != 0.0 || q_floor.x3 != 0.0)
            return false;
        if (!(q_floor.x0 > 0.0) || q_floor.x0 >= quotient_limit || q_floor.x0 == q)
            return false;

        return fmod_try_small_quotient_abs(ax, ay, q_floor.x0, out, quotient);
    }

    BL_MSVC_NOINLINE constexpr bool fmod_fast_small_quotient_abs(const f256_s& ax, const f256_s& ay, f256_s& out) noexcept
    {
        if (!(ay.x0 > 0.0) || detail::fp::isinf_or_nan(ay.x0) || !(ax >= ay))
            return false;

        const double q = detail::fp::trunc(ax.x0 / ay.x0);
        if (!(q > 0.0) || q >= 0x1p53)
            return false;

        f256_s r = fmod_sub_mul_scalar_expansion(ax, ay, q);
        if (!fmod_normalize_remainder(r, ay))
            return false;

        const f256_s edge_slack = mul_double_inline(ay, 0x1p-160);
        if (r <= edge_slack || sub_inline(ay, r) <= edge_slack)
            return false;

        out = r;
        return true;
    }

    BL_FORCE_INLINE constexpr void biguint_mod_shift_subtract_with_quotient_mod(
        const biguint& numerator,
        const biguint& denominator,
        biguint& remainder,
        std::uint64_t& quotient_mod) noexcept
    {
        constexpr std::uint64_t quotient_mask = 0x7fffffffull;

        remainder = numerator;
        if (denominator.is_zero())
            return;

        while (remainder.compare(denominator) >= 0)
        {
            int shift = (remainder.bit_length() - 1) - (denominator.bit_length() - 1);
            if (shift > 0 && detail::exact_decimal::compare_shifted(remainder, denominator, shift) < 0)
                --shift;

            detail::exact_decimal::sub_shifted_inplace(remainder, denominator, shift);
            if (shift < 31)
                quotient_mod = (quotient_mod + (std::uint64_t{ 1 } << shift)) & quotient_mask;
        }
    }

    BL_FORCE_INLINE constexpr biguint biguint_double_mod_with_quotient_bit(
        const biguint& value,
        const biguint& modulus,
        std::uint64_t& bit) noexcept
    {
        biguint out = value;
        out.shl1();

        bit = 0;
        if (out.compare(modulus) >= 0)
        {
            out.sub_inplace(modulus);
            bit = 1;
        }

        return out;
    }

    BL_FORCE_INLINE constexpr f256_s fmod_exact_abs_with_quotient_mod(
        const f256_s& ax,
        const f256_s& ay,
        std::uint64_t& quotient_mod)
    {
        constexpr std::uint64_t quotient_mask = 0x7fffffffull;

        f256_s fixed{};
        if (fmod_exact_fixed_limb_abs_with_quotient_mod(ax, ay, quotient_mod, fixed))
            return fixed;

        const exact_dyadic_fmod dx = exact_from_f256_fmod(ax);
        const exact_dyadic_fmod dy = exact_from_f256_fmod(ay);

        biguint remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            biguint denominator = dy.mant;
            denominator.shl_bits(shift);
            biguint_mod_shift_subtract_with_quotient_mod(dx.mant, denominator, remainder, quotient_mod);
            out_exp = dx.exp2;
        }
        else
        {
            biguint_mod_shift_subtract_with_quotient_mod(dx.mant, dy.mant, remainder, quotient_mod);
            const int shift = dx.exp2 - dy.exp2;

            int i = 0;
            for (; i < shift && !remainder.is_zero(); ++i)
            {
                std::uint64_t bit = 0;
                remainder = biguint_double_mod_with_quotient_bit(remainder, dy.mant, bit);
                quotient_mod = ((quotient_mod << 1) | bit) & quotient_mask;
            }

            const int remaining = shift - i;
            if (remaining >= 31)
                quotient_mod = 0;
            else if (remaining > 0)
                quotient_mod = (quotient_mod << remaining) & quotient_mask;

            out_exp = dy.exp2;
        }

        f256_s out = exact_dyadic_to_f256_fmod(remainder, out_exp, false);
        if (iszero(out))
            return f256_s{ 0.0 };
        return out;
    }

    BL_MSVC_NOINLINE constexpr f256_s fmod_reduced_or_exact(const f256_s& x, const f256_s& y)
    {
        const f256_s ay = mag(y);
        f256_s r = mag(x);

        constexpr int exact_reduction_exponent_gap = 96;
        if (frexp_exponent_limb(r.x0) - frexp_exponent_limb(ay.x0) > exact_reduction_exponent_gap)
            return fmod_exact(x, y);

        for (int iteration = 0; iteration < 128 && r >= ay; ++iteration)
        {
            const int ex = frexp_exponent_limb(r.x0);
            const int ey = frexp_exponent_limb(ay.x0);
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

            const double q = detail::fp::trunc(r.x0 / scaled.x0);
            if (!(q > 0.0) || absd(q) >= 0x1p53)
                return fmod_exact(x, y);

            r = fmod_sub_mul_scalar_expansion(r, scaled, q);
            if (!fmod_normalize_remainder(r, scaled))
                return fmod_exact(x, y);
        }

        if (!fmod_normalize_remainder(r, ay))
            return fmod_exact(x, y);

        if (iszero(r))
            return signed_zero_from(x.x0);

        return ispositive(x) ? r : -r;
    }

    // decimal conversion
    BL_FORCE_INLINE constexpr bool try_get_int64(const f256_s& x, int64_t& out)
    {
        const f256_s xi = detail::_f256_impl::trunc(x);
        if (xi != x)
            return false;

        if (absd(xi.x0) >= 0x1p63)
            return false;

        const int64_t p0 = static_cast<int64_t>(xi.x0);
        const f256_s r0 = sub_inline(xi, detail::_f256_impl::to_f256(p0));
        const int64_t p1 = static_cast<int64_t>(r0.x0);
        const f256_s r1 = sub_inline(r0, detail::_f256_impl::to_f256(p1));
        const int64_t p2 = static_cast<int64_t>(r1.x0);
        const f256_s r2 = sub_inline(r1, detail::_f256_impl::to_f256(p2));
        const int64_t p3 = static_cast<int64_t>(r2.x0 + r2.x1 + r2.x2 + r2.x3);

        out = p0 + p1 + p2 + p3;
        return true;
    }

    BL_FORCE_INLINE constexpr f256_s pack_decimal_significand(const biguint& q, int e2, bool neg) noexcept
    {
        const std::uint64_t c3 = q.get_bits(0, 53);
        const std::uint64_t c2 = q.get_bits(53, 53);
        const std::uint64_t c1 = q.get_bits(106, 53);
        const std::uint64_t c0 = q.get_bits(159, 53);

        const double x0 = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp(static_cast<double>(c3), e2 - 211) : 0.0;

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

    BL_FORCE_INLINE constexpr bool try_rounded_decimal_to_f256(const f256_s& integer_part, const char* digits, int digit_count, bool neg, f256_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        const biguint coeff = detail::fp::append_decimal_digits(
            biguint{ static_cast<std::uint64_t>(integer_value) },
            digits,
            digit_count);

        out = round_decimal_exact_to_f256(coeff, -digit_count, neg);
        return true;
    }

    // quotient helpers
    BL_FORCE_INLINE constexpr double limb_mod2(double value) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(value) || absd(value) >= 0x1p53)
            return 0.0;

        return fmod(value, 2.0);
    }

    BL_FORCE_INLINE constexpr bool is_odd_integer(const f256_s& x) noexcept
    {
        double mod2 =
            limb_mod2(x.x0) +
            limb_mod2(x.x1) +
            limb_mod2(x.x2) +
            limb_mod2(x.x3);

        mod2 = fmod(mod2, 2.0);
        if (mod2 < 0.0)
            mod2 += 2.0;

        return detail::fp::double_integer_is_odd(nearbyint_ties_even(mod2));
    }

    // sqrt kernels
    BL_FORCE_INLINE constexpr f256_s canonicalize_sqrt_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<16>(value.x3);
        return value;
    }

    #if defined(FLTX_CONSTEXPR_PARITY)
        #define F256_CANONICALIZE_SQRT_RESULT(value) bl::detail::_f256::canonicalize_sqrt_result(value)
    #else
        #define F256_CANONICALIZE_SQRT_RESULT(value) (value)
    #endif

    BL_FORCE_INLINE constexpr double sqrt_compress_sum16(const double* e) noexcept
    {
        using namespace detail::_f256;

        double g[16];
        double q = e[15];
        for (int i = 14; i >= 0; --i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, e[i], q_new, low);
            q = q_new;
            g[i + 1] = low;
        }
        g[0] = q;

        double residual = 0.0;
        q = g[0];
        for (int i = 1; i < 16; ++i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, g[i], q_new, low);
            if (low != 0.0)
                residual += low;
            q = q_new;
        }

        return residual + q;
    }

    BL_FORCE_INLINE constexpr double sqrt_compress_sum6(const double* e) noexcept
    {
        using namespace detail::_f256;

        double g[6];
        double q = e[5];
        for (int i = 4; i >= 0; --i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, e[i], q_new, low);
            q = q_new;
            g[i + 1] = low;
        }
        g[0] = q;

        double residual = 0.0;
        q = g[0];
        for (int i = 1; i < 6; ++i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, g[i], q_new, low);
            if (low != 0.0)
                residual += low;
            q = q_new;
        }

        return residual + q;
    }

    BL_FORCE_INLINE constexpr double sqrt_compress_sum10(const double* e) noexcept
    {
        using namespace detail::_f256;

        double g[10];
        double q = e[9];
        for (int i = 8; i >= 0; --i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, e[i], q_new, low);
            q = q_new;
            g[i + 1] = low;
        }
        g[0] = q;

        double residual = 0.0;
        q = g[0];
        for (int i = 1; i < 10; ++i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, g[i], q_new, low);
            if (low != 0.0)
                residual += low;
            q = q_new;
        }

        return residual + q;
    }

    BL_FORCE_INLINE constexpr double sqrt_tail_residual_head_limb(const f256_s& scaled_a, double y0)
    {
        using namespace detail::_f256;

        double p00{}, q00{};
        two_prod_precise(y0, y0, p00, q00);

        const double terms[] = {
            scaled_a.x3,
            scaled_a.x2,
            scaled_a.x1,
            -q00,
            scaled_a.x0,
            -p00
        };

        return sqrt_compress_sum6(terms);
    }

    BL_FORCE_INLINE constexpr double sqrt_tail_residual_head_dd(const f256_s& scaled_a, const f256_s& y)
    {
        using namespace detail::_f256;

        double p00{}, q00{};
        double p01{}, q01{};
        double p11{}, q11{};

        #if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p00p01{}, q00q01{};
            simd::f64x2_two_prod_precise(
                simd::f64x2_set(y.x0, y.x0),
                simd::f64x2_set(y.x0, y.x1),
                p00p01,
                q00q01);
            simd::f64x2_store(p00p01, p00, p01);
            simd::f64x2_store(q00q01, q00, q01);
            two_prod_precise(y.x1, y.x1, p11, q11);
        }
        else
        #endif
        {
            two_prod_precise(y.x0, y.x0, p00, q00);
            two_prod_precise(y.x0, y.x1, p01, q01);
            two_prod_precise(y.x1, y.x1, p11, q11);
        }

        p01 *= 2.0; q01 *= 2.0;

        const double terms[] = {
            scaled_a.x3,
            -q11,
            scaled_a.x2,
            -p11,
            -q01,
            scaled_a.x1,
            -p01,
            -q00,
            scaled_a.x0,
            -p00
        };

        return sqrt_compress_sum10(terms);
    }

    BL_FORCE_INLINE constexpr double sqrt_tail_residual_head(const f256_s& scaled_a, const f256_s& y)
    {
        using namespace detail::_f256;

        double p00{}, q00{};
        double p01{}, q01{};
        double p02{}, q02{};
        double p03{}, q03{};
        double p11{}, q11{};
        double p12{}, q12{};

        #if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            simd::f64x2 p00p12{}, q00q12{};
            simd::f64x2 p01p02{}, q01q02{};
            simd::f64x2 p03p11{}, q03q11{};

            simd::f64x2_two_prod_precise(simd::f64x2_set(y.x0, y.x1), simd::f64x2_set(y.x0, y.x2), p00p12, q00q12);
            simd::f64x2_two_prod_precise(simd::f64x2_set(y.x0, y.x0), simd::f64x2_set(y.x1, y.x2), p01p02, q01q02);
            simd::f64x2_two_prod_precise(simd::f64x2_set(y.x0, y.x1), simd::f64x2_set(y.x3, y.x1), p03p11, q03q11);

            simd::f64x2_store(p00p12, p00, p12);
            simd::f64x2_store(q00q12, q00, q12);
            simd::f64x2_store(p01p02, p01, p02);
            simd::f64x2_store(q01q02, q01, q02);
            simd::f64x2_store(p03p11, p03, p11);
            simd::f64x2_store(q03q11, q03, q11);
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

        return sqrt_compress_sum16(terms);
    }

    BL_FORCE_INLINE constexpr f256_s sqrt_step_tail_head_limb(const f256_s& scaled_a, double y0, double half_inv_y0)
    {
        using namespace detail::_f256;
        return add_double_inline(f256_s{ y0, 0.0, 0.0, 0.0 }, sqrt_tail_residual_head_limb(scaled_a, y0) * half_inv_y0);
    }

    BL_FORCE_INLINE constexpr f256_s sqrt_step_tail_head_dd(const f256_s& scaled_a, const f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        return add_double_inline(y, sqrt_tail_residual_head_dd(scaled_a, y) * half_inv_y0);
    }

    BL_FORCE_INLINE constexpr f256_s sqrt_step_tail_head(const f256_s& scaled_a, const f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        return add_double_inline(y, sqrt_tail_residual_head(scaled_a, y) * half_inv_y0);
    }

    BL_PUSH_PRECISE
    BL_FORCE_INLINE constexpr double sqrt_renorm4_head(double c0, double c1, double c2, double c3) noexcept
    {
        double s{}, e{};
        s = c2 + c3;  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        if (c1 == 0.0 && c2 != 0.0)
            c0 += c2;
        if (c1 == 0.0 && c3 != 0.0)
            c0 += c3;

        return c0;
    }

    BL_FORCE_INLINE constexpr double sqrt_renorm5_head(double c0, double c1, double c2, double c3, double c4) noexcept
    {
        double s{}, e{};
        s = c3 + c4;  e = c4 - (s - c3);  c3 = s;  c4 = e;
        s = c2 + c3;  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        if (c1 == 0.0 && c2 != 0.0)
            c0 += c2;
        if (c1 == 0.0 && c3 != 0.0)
            c0 += c3;
        if (c1 == 0.0 && c4 != 0.0)
            c0 += c4;

        return c0;
    }
    BL_POP_PRECISE

    BL_FORCE_INLINE constexpr double sqrt_raw_residual_head(const f256_s& scaled_a, const f256_s& y) noexcept
    {
        using namespace detail::_f256;

        const f256_raw5 p = neg_raw5(sqr_raw5_inline(y));

        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        two_sum_precise(p.x0, scaled_a.x0, s0, e0);
        two_sum_precise(p.x1, scaled_a.x1, s1, e1);
        two_sum_precise(p.x2, scaled_a.x2, s2, e2);
        two_sum_precise(p.x3, scaled_a.x3, s3, e3);

        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3 + p.x4;

        if (e0 == 0.0)
            return sqrt_renorm4_head(s0, s1, s2, s3);

        return sqrt_renorm5_head(s0, s1, s2, s3, e0);
    }

    BL_FORCE_INLINE constexpr f256_s sqrt_step_full_recip(const f256_s& scaled_a, const f256_s& y)
    {
        using namespace detail::_f256;
        return add_double_inline(y, sqrt_raw_residual_head(scaled_a, y) * (0.5 / y.x0));
    }

    BL_FORCE_INLINE constexpr double sqrt_limb_seed(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            sqrt_seed(x),
            std::sqrt(x)
        );
    }

    BL_FORCE_INLINE constexpr int scale_sqrt_input(f256_s& scaled_a) noexcept
    {
        int result_scale = 0;
        if (scaled_a.x0 < 0x1p-900 || scaled_a.x0 > 0x1p900)
        {
            const int exp2 = frexp_exponent_limb(scaled_a.x0);
            result_scale = exp2 / 2;
            const int input_scale = -2 * result_scale;
            if (input_scale != 0)
                scaled_a = ldexp_terms(scaled_a, input_scale);
        }

        return result_scale;
    }

    BL_FORCE_INLINE constexpr f256_s rescale_sqrt_result(f256_s y, int result_scale) noexcept
    {
        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return F256_CANONICALIZE_SQRT_RESULT(y);
    }

    BL_MSVC_NOINLINE constexpr f256_s sqrt_impl_fast(const f256_s& a)
    {
        f256_s scaled_a = a;
        const int result_scale = scale_sqrt_input(scaled_a);

        const double y0 = sqrt_limb_seed(scaled_a.x0);
        f256_s y = sqrt_step_tail_head_limb(scaled_a, y0, 0.5 / y0);
        y = sqrt_step_tail_head_dd(scaled_a, y, 0.5 / y.x0);
        if (scaled_a.x0 < 1.0)
        {
            y = sqrt_step_full_recip(scaled_a, y);
        }
        else
        {
            y = sqrt_step_tail_head(scaled_a, y, 0.5 / y.x0);
            y = sqrt_step_tail_head(scaled_a, y, 0.5 / y.x0);
        }

        return rescale_sqrt_result(y, result_scale);
    }

    BL_MSVC_NOINLINE constexpr f256_s sqrt_impl(const f256_s& a)
    {
        f256_s scaled_a = a;
        const int result_scale = scale_sqrt_input(scaled_a);

        const double y0 = sqrt_limb_seed(scaled_a.x0);
        f256_s y = sqrt_step_tail_head_limb(scaled_a, y0, 0.5 / y0);
        y = sqrt_step_tail_head_dd(scaled_a, y, 0.5 / y.x0);
        y = sqrt_step_tail_head(scaled_a, y, 0.5 / y.x0);
        y = sqrt_step_full_recip(scaled_a, y);

        return rescale_sqrt_result(y, result_scale);
    }

    // rounding helpers
    BL_FORCE_INLINE constexpr f256_s round_half_away_zero(const f256_s& x) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(x.x0))
            return x;

        if (bl::signbit(x))
        {
            f256_s y = -detail::_f256_impl::floor(
                add_inline(-x, f256_s{ 0.5 }));
            if (iszero(y))
                return f256_s{ -0.0, 0.0, 0.0, 0.0 };
            return y;
        }

        return detail::_f256_impl::floor(
            add_inline(x, f256_s{ 0.5 }));
    }

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

        if (detail::fp::isinf_or_nan(x.x0))
            return 0;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        const f256_s lo = detail::_f256_impl::to_f256(lo_i);
        const f256_s hi = detail::_f256_impl::to_f256(hi_i);

        if (x < lo || x > hi)
            return 0;

        std::int64_t out = 0;
        if (!try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr bool try_round_to_signed_integer(const f256_s& x, bool ties_to_even, SignedInt& out) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (detail::fp::isinf_or_nan(x.x0) || absd(x.x0) >= 0x1p52)
            return false;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        if (x < detail::_f256_impl::to_f256(lo_i) || x > detail::_f256_impl::to_f256(hi_i))
            return false;

        const std::int64_t base = static_cast<std::int64_t>(x.x0);
        const f256_s frac     = sub_double_inline(x, static_cast<double>(base));
        const f256_s abs_frac = mag(frac);
        std::int64_t rounded = base;

        if (abs_frac > f256_s{ 0.5 } || (!ties_to_even && abs_frac == f256_s{ 0.5 }) ||
            (ties_to_even && abs_frac == f256_s{ 0.5 } && (base & 1ll) != 0))
        {
            rounded += (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0))) ? -1 : 1;
        }

        if (rounded < lo_i || rounded > hi_i)
            return false;

        out = static_cast<SignedInt>(rounded);
        return true;
    }

    // scaling kernels
    BL_FORCE_INLINE constexpr f256_s _ldexp(const f256_s& a, int e)
    {
        if (e > 1023 || e < -1074) [[unlikely]]
            return ldexp_terms(a, e);

        double s;
        if (bl::use_constexpr_math())
        {
            s = bl::detail::fp::ldexp(1.0, e);
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
                const simd::f64x2 scale = simd::f64x2_splat(s);
                simd::f64x2 lo = simd::f64x2_mul(simd::f64x2_set(a.x0, a.x1), scale);
                simd::f64x2 hi = simd::f64x2_mul(simd::f64x2_set(a.x2, a.x3), scale);
                double x0{}, x1{}, x2{}, x3{};
                simd::f64x2_store(lo, x0, x1);
                simd::f64x2_store(hi, x2, x3);
                return renorm(x0, x1, x2, x3);
            }
            else
            #endif
            {
                return renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
            }
        }
    }

} // namespace detail::_f256

} // namespace bl

#endif
