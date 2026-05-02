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
#include "fltx_common_exact.h"
#include "fltx_math_utils.h"

namespace bl {

/// ------------------ math ------------------

namespace detail::_f256
{
    using detail::exact_decimal::add_signed;
    using detail::exact_decimal::biguint;
    using detail::exact_decimal::decompose_double_mantissa;
    using detail::exact_decimal::mod_shift_subtract;
    using detail::exact_decimal::signed_biguint;
    struct exact_dyadic_fmod
    {
        int exp2 = 0;
        biguint mant{};
    };

    BL_FORCE_INLINE constexpr bool biguint_is_odd(const biguint& value)
    {
        return !value.is_zero() && (value.words[0] & 1u) != 0;
    }
    BL_FORCE_INLINE constexpr bool biguint_any_low_bits_set(const biguint& value, int bit_count)
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
    BL_NO_INLINE constexpr exact_dyadic_fmod exact_from_f256_fmod(const f256_s& x)
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
    BL_NO_INLINE constexpr f256_s exact_dyadic_to_f256_fmod(const biguint& coeff, int exp2, bool neg)
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

        f256_s out = detail::_f256::renorm5(x0, x1, x2, x3, x4);
        return neg ? -out : out;
    }
    BL_NO_INLINE constexpr f256_s fmod_exact(const f256_s& x, const f256_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f256_fmod(abs(x));
        const exact_dyadic_fmod dy = exact_from_f256_fmod(abs(y));

        if (dx.mant.is_zero() || dy.mant.is_zero())
            return f256_s{ detail::_f256::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

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
            return f256_s{ detail::_f256::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return out;
    }
    BL_FORCE_INLINE constexpr f256_s canonicalize_sqrt_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<16>(value.x3);
        return value;
    }
    BL_FORCE_INLINE constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::_f256::frexp_exponent_constexpr(value);

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }
    BL_FORCE_INLINE constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::ldexp_constexpr2(value, exponent);

        return std::ldexp(value, exponent);
    }
    BL_FORCE_INLINE constexpr f256_s ldexp_terms(const f256_s& value, int exponent) noexcept
    {
        return detail::_f256::renorm(
            detail::_f256::ldexp_limb(value.x0, exponent),
            detail::_f256::ldexp_limb(value.x1, exponent),
            detail::_f256::ldexp_limb(value.x2, exponent),
            detail::_f256::ldexp_limb(value.x3, exponent));
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
    BL_FORCE_INLINE constexpr void fmod_append_expansion_term(double* expansion, int& count, double value) noexcept
    {
        if (value != 0.0)
            expansion[count++] = value;
    }
    BL_NO_INLINE constexpr f256_s fmod_sub_mul_scalar_expansion(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double r_exp[4]{};
        int r_count = 0;
        detail::_f256::fmod_append_expansion_term(r_exp, r_count, r.x3);
        detail::_f256::fmod_append_expansion_term(r_exp, r_count, r.x2);
        detail::_f256::fmod_append_expansion_term(r_exp, r_count, r.x1);
        detail::_f256::fmod_append_expansion_term(r_exp, r_count, r.x0);

        double b_exp[4]{};
        int b_count = 0;
        detail::_f256::fmod_append_expansion_term(b_exp, b_count, b.x3);
        detail::_f256::fmod_append_expansion_term(b_exp, b_count, b.x2);
        detail::_f256::fmod_append_expansion_term(b_exp, b_count, b.x1);
        detail::_f256::fmod_append_expansion_term(b_exp, b_count, b.x0);

        double product_exp[16]{};
        const int product_count = detail::_f256::scale_expansion_zeroelim(b_count, b_exp, q, product_exp);
        for (int i = 0; i < product_count; ++i)
            product_exp[i] = -product_exp[i];

        double diff_exp[32]{};
        const int diff_count = detail::_f256::fast_expansion_sum_zeroelim(r_count, r_exp, product_count, product_exp, diff_exp);
        return detail::_f256::from_expansion_fast(diff_exp, diff_count);
    }
    BL_NO_INLINE constexpr f256_s fmod_runtime(const f256_s& x, const f256_s& y)
    {
        const f256_s ay = abs(y);
        f256_s r = abs(x);

        for (int iteration = 0; iteration < 128 && r >= ay; ++iteration)
        {
            const int ex = detail::_f256::frexp_exponent(r.x0);
            const int ey = detail::_f256::frexp_exponent(ay.x0);
            int shift = ex - ey - 52;
            if (shift < 0)
                shift = 0;

            f256_s scaled = detail::_f256::ldexp_terms(ay, shift);
            while (shift > 0 && scaled > r)
            {
                --shift;
                scaled = detail::_f256::ldexp_terms(ay, shift);
            }

            if (!(scaled > 0.0) || scaled > r)
                return detail::_f256::fmod_exact(x, y);

            const f256_s q_floor = floor(r / scaled);
            if (q_floor.x1 != 0.0 || q_floor.x2 != 0.0 || q_floor.x3 != 0.0)
                return detail::_f256::fmod_exact(x, y);
            if (!(q_floor.x0 > 0.0) || detail::_f256::absd(q_floor.x0) >= 0x1p53)
                return detail::_f256::fmod_exact(x, y);

            r = detail::_f256::fmod_sub_mul_scalar_expansion(r, scaled, q_floor.x0);
            if (!detail::_f256::fmod_normalize_remainder(r, scaled))
                return detail::_f256::fmod_exact(x, y);
        }

        if (!detail::_f256::fmod_normalize_remainder(r, ay))
            return detail::_f256::fmod_exact(x, y);

        if (iszero(r))
            return f256_s{ detail::_f256::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

        return ispositive(x) ? r : -r;
    }
    BL_NO_INLINE constexpr bool fmod_fast_double_divisor_abs(const f256_s& ax, double ay, f256_s& out)
    {
        if (!(ay > 0.0) || !detail::_f256::isfinite(ay))
            return false;

        const f256_s mod{ ay, 0.0, 0.0, 0.0 };

        if (ax.x1 == 0.0 && ax.x2 == 0.0 && ax.x3 == 0.0)
        {
            out = f256_s{ detail::_f256::fmod_constexpr(ax.x0, ay), 0.0, 0.0, 0.0 };
            return true;
        }

        const double r0 = (ax.x0 < ay) ? ax.x0 : detail::_f256::fmod_constexpr(ax.x0, ay);
        const double r1 = (detail::_f256::absd(ax.x1) < ay) ? ax.x1 : detail::_f256::fmod_constexpr(ax.x1, ay);
        const double r2 = (detail::_f256::absd(ax.x2) < ay) ? ax.x2 : detail::_f256::fmod_constexpr(ax.x2, ay);
        const double r3 = (detail::_f256::absd(ax.x3) < ay) ? ax.x3 : detail::_f256::fmod_constexpr(ax.x3, ay);

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
    BL_FORCE_INLINE constexpr bool f256_try_get_int64(const f256_s& x, int64_t& out)
    {
        const f256_s xi = trunc(x);
        if (xi != x)
            return false;

        if (detail::_f256::absd(xi.x0) >= 0x1p63)
            return false;

        const int64_t p0 = static_cast<int64_t>(xi.x0);
        const f256_s r0 = xi - to_f256(p0);
        const int64_t p1 = static_cast<int64_t>(r0.x0);
        const f256_s r1 = r0 - to_f256(p1);
        const int64_t p2 = static_cast<int64_t>(r1.x0);
        const f256_s r2 = r1 - to_f256(p2);
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

        const double x0 = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? detail::fp::ldexp_constexpr2(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? detail::fp::ldexp_constexpr2(static_cast<double>(c3), e2 - 211) : 0.0;

        f256_s out = detail::_f256::renorm(x0, x1, x2, x3);
        return neg ? -out : out;
    }
    BL_NO_INLINE constexpr f256_s round_decimal_exact_to_f256(const biguint& coeff, int dec_exp, bool neg) noexcept
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

        return detail::_f256::pack_decimal_significand(q, e2, neg);
    }
    BL_FORCE_INLINE constexpr bool try_rounded_decimal_to_f256(const f256_s& integer_part, const char* digits, int digit_count, bool neg, f256_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!detail::_f256::f256_try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        const biguint coeff = detail::fp::append_decimal_digits(
            biguint{ static_cast<std::uint64_t>(integer_value) },
            digits,
            digit_count);

        out = detail::_f256::round_decimal_exact_to_f256(coeff, -digit_count, neg);
        return true;
    }
    BL_FORCE_INLINE constexpr double limb_mod2(double value) noexcept
    {
        if (value == 0.0 || !detail::_f256::isfinite(value) || detail::_f256::absd(value) >= 0x1p53)
            return 0.0;

        return detail::_f256::fmod_constexpr(value, 2.0);
    }
    BL_FORCE_INLINE constexpr bool is_odd_integer(const f256_s& x) noexcept
    {
        double mod2 =
            detail::_f256::limb_mod2(x.x0) +
            detail::_f256::limb_mod2(x.x1) +
            detail::_f256::limb_mod2(x.x2) +
            detail::_f256::limb_mod2(x.x3);

        mod2 = detail::_f256::fmod_constexpr(mod2, 2.0);
        if (mod2 < 0.0)
            mod2 += 2.0;

        return detail::fp::double_integer_is_odd(detail::_f256::nearbyint_ties_even(mod2));
    }
    BL_FORCE_INLINE constexpr double limb_mod_power_of_two(double value, double modulus, double zero_threshold) noexcept
    {
        if (value == 0.0 || !detail::_f256::isfinite(value) || detail::_f256::absd(value) >= zero_threshold)
            return 0.0;

        return detail::_f256::fmod_constexpr(value, modulus);
    }
    BL_FORCE_INLINE constexpr int low_quotient_bits(const f256_s& x) noexcept
    {
        constexpr double modulus = 2147483648.0;
        constexpr double zero_threshold = 0x1p83;

        double bits =
            detail::_f256::limb_mod_power_of_two(x.x0, modulus, zero_threshold) +
            detail::_f256::limb_mod_power_of_two(x.x1, modulus, zero_threshold) +
            detail::_f256::limb_mod_power_of_two(x.x2, modulus, zero_threshold) +
            detail::_f256::limb_mod_power_of_two(x.x3, modulus, zero_threshold);

        bits = detail::_f256::fmod_constexpr(bits, modulus);
        return static_cast<int>(static_cast<long long>(detail::_f256::nearbyint_ties_even(bits)));
    }

    BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (a.x2 == 0.0 && a.x3 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return detail::_f256::add_dd_qd(a, b);

        return detail::_f256::add_qd_qd(a, b);
    }
    BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (a.x2 == 0.0 && a.x3 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return detail::_f256::sub_dd_qd(a, b);

        return detail::_f256::sub_qd_qd(a, b);
    }
    BL_FORCE_INLINE constexpr f256_s mul_inline(const f256_s& a, const f256_s& b) noexcept
    {
        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        detail::_f256::two_prod_precise(a.x0, b.x0, p0, q0);
        detail::_f256::two_prod_precise(a.x0, b.x1, p1, q1);
        detail::_f256::two_prod_precise(a.x1, b.x0, p2, q2);
        detail::_f256::two_prod_precise(a.x0, b.x2, p3, q3);
        detail::_f256::two_prod_precise(a.x1, b.x1, p4, q4);
        detail::_f256::two_prod_precise(a.x2, b.x0, p5, q5);

        detail::_f256::three_sum(p1, p2, q0);
        detail::_f256::three_sum(p2, q1, q2);
        detail::_f256::three_sum(p3, p4, p5);

        detail::_f256::two_sum_precise(p2, p3, s0, t0);
        detail::_f256::two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        detail::_f256::two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        detail::_f256::two_prod_precise(a.x0, b.x3, p6, q6);
        detail::_f256::two_prod_precise(a.x1, b.x2, p7, q7);
        detail::_f256::two_prod_precise(a.x2, b.x1, p8, q8);
        detail::_f256::two_prod_precise(a.x3, b.x0, p9, q9);

        detail::_f256::two_sum_precise(q0, q3, q0, q3);
        detail::_f256::two_sum_precise(q4, q5, q4, q5);
        detail::_f256::two_sum_precise(p6, p7, p6, p7);
        detail::_f256::two_sum_precise(p8, p9, p8, p9);

        detail::_f256::two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        detail::_f256::two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        detail::_f256::two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        detail::_f256::two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + s2;

        return detail::_f256::renorm5(p0, p1, s0, t0, t1);
    }
    BL_FORCE_INLINE constexpr f256_s sqr_inline(const f256_s& a) noexcept
    {
        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        detail::_f256::two_prod_precise(a.x0, a.x0, p0, q0);
        detail::_f256::two_prod_precise(a.x0, a.x1, p1, q1);
        p2 = p1;
        q2 = q1;
        detail::_f256::two_prod_precise(a.x0, a.x2, p3, q3);
        detail::_f256::two_prod_precise(a.x1, a.x1, p4, q4);
        p5 = p3;
        q5 = q3;

        detail::_f256::three_sum(p1, p2, q0);
        detail::_f256::three_sum(p2, q1, q2);
        detail::_f256::three_sum(p3, p4, p5);

        detail::_f256::two_sum_precise(p2, p3, s0, t0);
        detail::_f256::two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        detail::_f256::two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        detail::_f256::two_prod_precise(a.x0, a.x3, p6, q6);
        detail::_f256::two_prod_precise(a.x1, a.x2, p7, q7);
        p8 = p7;
        q8 = q7;
        p9 = p6;
        q9 = q6;

        detail::_f256::two_sum_precise(q0, q3, q0, q3);
        detail::_f256::two_sum_precise(q4, q5, q4, q5);
        detail::_f256::two_sum_precise(p6, p7, p6, p7);
        detail::_f256::two_sum_precise(p8, p9, p8, p9);

        detail::_f256::two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        detail::_f256::two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        detail::_f256::two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        detail::_f256::two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * a.x3 + a.x2 * a.x2 + a.x3 * a.x1 + q6 + q7 + q8 + q9 + s2;

        return detail::_f256::renorm5(p0, p1, s0, t0, t1);
    }
    BL_FORCE_INLINE constexpr f256_s div_inline(const f256_s& a, const f256_s& b) noexcept
    {
        if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return a / b.x0;

        const double inv_b0 = 1.0 / b.x0;

        const double q0 = a.x0 * inv_b0;
        f256_s r = detail::_f256::sub_mul_scalar_exact(a, b, q0);

        const double q1 = r.x0 * inv_b0;
        r = detail::_f256::sub_mul_scalar_fast(r, b, q1);

        const double q2 = r.x0 * inv_b0;
        r = detail::_f256::sub_mul_scalar_fast(r, b, q2);

        const double q3 = r.x0 * inv_b0;
        r = detail::_f256::sub_mul_scalar_fast(r, b, q3);

        const double q4 = r.x0 * inv_b0;

        return detail::_f256::renorm5(q0, q1, q2, q3, q4);
    }
    BL_FORCE_INLINE constexpr f256_s mul_double_inline(const f256_s& a, double b) noexcept
    {
        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        detail::_f256::two_prod_precise(a.x0, b, p0, q0);
        detail::_f256::two_prod_precise(a.x1, b, p1, q1);
        detail::_f256::two_prod_precise(a.x2, b, p2, q2);
        p3 = a.x3 * b;

        s0 = p0;
        detail::_f256::two_sum_precise(q0, p1, s1, s2);
        detail::_f256::three_sum(s2, q1, p2);
        detail::_f256::three_sum2(q1, q2, p3);
        s3 = q1;
        s4 = q2 + p2;

        return detail::_f256::renorm5(s0, s1, s2, s3, s4);
    }
    BL_FORCE_INLINE constexpr f256_s div_double_inline(const f256_s& a, double b) noexcept
    {
        if (bl::use_constexpr_math())
        {
            if (isnan(a) || detail::_f256::isnan(b))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (detail::_f256::isinf(b))
            {
                if (isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = detail::_f256::signbit_constexpr(a.x0) ^ detail::_f256::signbit_constexpr(b);
                return f256_s{ neg ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
            }

            if (b == 0.0)
            {
                if (iszero(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = detail::_f256::signbit_constexpr(a.x0) ^ detail::_f256::signbit_constexpr(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   :  std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (isinf(a))
            {
                const bool neg = detail::_f256::signbit_constexpr(a.x0) ^ detail::_f256::signbit_constexpr(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   :  std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        const double inv_b = 1.0 / b;
        const f256_s divisor{ b, 0.0, 0.0, 0.0 };

        const double q0 = a.x0 * inv_b;
        f256_s r = detail::_f256::sub_mul_scalar_exact(a, divisor, q0);

        const double q1 = r.x0 * inv_b; r = detail::_f256::sub_mul_scalar_fast(r, divisor, q1);
        const double q2 = r.x0 * inv_b; r = detail::_f256::sub_mul_scalar_fast(r, divisor, q2);
        const double q3 = r.x0 * inv_b; r = detail::_f256::sub_mul_scalar_fast(r, divisor, q3);
        const double q4 = r.x0 * inv_b;

        return detail::_f256::renorm5(q0, q1, q2, q3, q4);
    }
    BL_FORCE_INLINE constexpr f256_s div_double_inline(double a, const f256_s& b) noexcept
    {
        if (bl::use_constexpr_math())
        {
            if (detail::_f256::isnan(a) || isnan(b))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (isinf(b))
            {
                if (detail::_f256::isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = detail::_f256::signbit_constexpr(a) ^ detail::_f256::signbit_constexpr(b.x0);
                return f256_s{ neg ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
            }

            if (iszero(b))
            {
                if (a == 0.0)
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = detail::_f256::signbit_constexpr(a) ^ detail::_f256::signbit_constexpr(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (detail::_f256::isinf(a))
            {
                const bool neg = detail::_f256::signbit_constexpr(a) ^ detail::_f256::signbit_constexpr(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return div_double_inline(f256_s{ a, 0.0, 0.0, 0.0 }, b.x0);

        const double inv_b0 = 1.0 / b.x0;
        const double q0 = a * inv_b0;

        double p0{}, p1{}, p2{}, p3{};
        double e0{}, e1{}, e2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        detail::_f256::two_prod_precise(b.x0, q0, p0, e0);
        detail::_f256::two_prod_precise(b.x1, q0, p1, e1);
        detail::_f256::two_prod_precise(b.x2, q0, p2, e2);
        p3 = b.x3 * q0;

        s0 = p0;
        detail::_f256::two_sum_precise(e0, p1, s1, s2);
        detail::_f256::three_sum(s2, e1, p2);
        detail::_f256::three_sum2(e1, e2, p3);
        s3 = e1;
        s4 = e2 + p2;

        double c0{}, t0{};
        detail::_f256::two_sum_precise(a, -s0, c0, t0);

        double c1 = -s1;
        double c2 = -s2;
        double c3 = -s3;
        double t1 = 0.0;
        double t2 = 0.0;

        detail::_f256::two_sum_precise(c1, t0, c1, t0);
        detail::_f256::three_sum(c2, t0, t1);
        detail::_f256::three_sum2(c3, t0, t2);
        t0 += t1 - s4;

        f256_s r = detail::_f256::renorm5(c0, c1, c2, c3, t0);

        const double q1 = r.x0 * inv_b0; r = detail::_f256::sub_mul_scalar_fast(r, b, q1);
        const double q2 = r.x0 * inv_b0; r = detail::_f256::sub_mul_scalar_fast(r, b, q2);
        const double q3 = r.x0 * inv_b0; r = detail::_f256::sub_mul_scalar_fast(r, b, q3);
        const double q4 = r.x0 * inv_b0;

        return detail::_f256::renorm5(q0, q1, q2, q3, q4);
    }

    BL_NO_INLINE constexpr f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    struct powi_ops
    {
        BL_FORCE_INLINE constexpr f256_s multiply(f256_s a, const f256_s& b) const noexcept
        {
            a *= b;
            return a;
        }

        BL_FORCE_INLINE constexpr f256_s divide(const f256_s& a, const f256_s& b) const noexcept
        {
            return a / b;
        }
    };

    BL_FORCE_INLINE constexpr f256_s powi(f256_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp, powi_ops{});
    }
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y)
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
    if (y.x1 == 0.0 && y.x2 == 0.0 && y.x3 == 0.0 && detail::_f256::fmod_fast_double_divisor_abs(ax, ay.x0, fast))
    {
        if (iszero(fast))
            return f256_s{ detail::_f256::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        const f256_s out = ispositive(x) ? fast : -fast;
        return detail::_f256::canonicalize_math_result(out);
    }

    const f256_s out = bl::use_constexpr_math()
        ? detail::_f256::fmod_exact(x, y)
        : detail::_f256::fmod_runtime(x, y);

    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s round(const f256_s& a)
{
    f256_s t = floor(a + f256_s{ 0.5 });
    if ((t - a) == f256_s{ 0.5 } && detail::_f256::is_odd_integer(t))
        t -= f256_s{ 1.0 };
    return t;
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s round_to_decimals(f256_s v, int prec)
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

    f256_s ip = floor(v);
    f256_s frac = v - ip;

    f256_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;

        int di = static_cast<int>(floor(w).x0);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = w - f256_s{ static_cast<double>(di) };
    }

    f256_s la = w * 10.0;

    const f256_s tie_slop = f256_s::eps() * f256_s{ 65536.0 };
    int next = static_cast<int>(floor(la).x0);
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
    if (detail::_f256::try_rounded_decimal_to_f256(ip, digits, prec, neg, exact_out))
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

namespace detail::_f256
{
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqrt_double_terms(const f256_s& y)
    {
        return f256_s{ y.x0 + y.x0, y.x1 + y.x1, y.x2 + y.x2, y.x3 + y.x3 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr void sqrt_step_2y(const f256_s& scaled_a, f256_s& y)
    {
        using namespace detail::_f256;
        const f256_s residual = sub_inline(scaled_a, sqr_inline(y));
        y = add_inline(y, div_inline(residual, sqrt_double_terms(y)));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr void sqrt_step_seed_recip(const f256_s& scaled_a, f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        const f256_s residual = sub_inline(scaled_a, sqr_inline(y));
        y = add_inline(y, mul_double_inline(residual, half_inv_y0));
    }

    [[nodiscard]] BL_NO_INLINE constexpr f256_s sqrt_constexpr_impl(const f256_s& a)
    {
        const int exp2 = detail::_f256::frexp_exponent(a.x0);
        const int result_scale = exp2 / 2;
        const int input_scale = -2 * result_scale;
        const f256_s scaled_a = input_scale == 0 ? a : detail::_f256::ldexp_terms(a, input_scale);

        const double y0 = detail::_f256::sqrt_seed_constexpr(scaled_a.x0);
        f256_s y{ y0, 0.0, 0.0, 0.0 };
        sqrt_step_2y(scaled_a, y);
        sqrt_step_2y(scaled_a, y);
        sqrt_step_2y(scaled_a, y);

        if (result_scale != 0)
            y = detail::_f256::ldexp_terms(y, result_scale);

        return detail::_f256::canonicalize_sqrt_result(y);
    }

    [[nodiscard]] BL_FORCE_INLINE f256_s sqrt_runtime_impl(const f256_s& a)
    {
        f256_s scaled_a = a;
        int result_scale = 0;
        if (a.x0 < 0x1p-900 || a.x0 > 0x1p900)
        {
            const int exp2 = detail::_f256::frexp_exponent(a.x0);
            result_scale = exp2 / 2;
            const int input_scale = -2 * result_scale;
            if (input_scale != 0)
                scaled_a = detail::_f256::ldexp_terms(a, input_scale);
        }

        const double y0 = std::sqrt(scaled_a.x0);
        const double half_inv_y0 = 0.5 / y0;
        f256_s y{ y0, 0.0, 0.0, 0.0 };
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);

        if (result_scale != 0)
            y = detail::_f256::ldexp_terms(y, result_scale);

        return detail::_f256::canonicalize_sqrt_result(y);
    }

}

[[nodiscard]] BL_NO_INLINE constexpr f256_s sqrt(const f256_s& a)
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
        return detail::_f256::sqrt_constexpr_impl(a);

    return detail::_f256::sqrt_runtime_impl(a);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s nearbyint(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f256_s t = floor(a);
    const f256_s frac = a - t;

    if (frac < f256_s{ 0.5 })
        return t;

    if (frac > f256_s{ 0.5 })
    {
        t += f256_s{ 1.0 };
        if (iszero(t))
            return f256_s{ detail::_f256::signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return t;
    }

    if (detail::_f256::is_odd_integer(t))
        t += f256_s{ 1.0 };

    if (iszero(t))
        return f256_s{ detail::_f256::signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return t;
}

/// ------------------ transcendentals ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept
{
    const double hi = a.x0;
    if (hi <= 0.0)
        return detail::fp::log_constexpr(static_cast<double>(a));

    const double lo = (a.x1 + a.x2) + a.x3;
    return detail::fp::log_constexpr(hi) + detail::fp::log1p_constexpr(lo / hi);
}

namespace detail::_f256
{
    inline constexpr f256_s e = std::numbers::e_v<f256_s>;
    inline constexpr f256_s log2e = std::numbers::log2e_v<f256_s>;
    inline constexpr f256_s log10e = std::numbers::log10e_v<f256_s>;
    inline constexpr f256_s pi = std::numbers::pi_v<f256_s>;
    inline constexpr f256_s inv_pi = std::numbers::inv_pi_v<f256_s>;
    inline constexpr f256_s inv_sqrtpi = std::numbers::inv_sqrtpi_v<f256_s>;
    inline constexpr f256_s ln2 = std::numbers::ln2_v<f256_s>;
    inline constexpr f256_s ln10 = std::numbers::ln10_v<f256_s>;
    inline constexpr f256_s sqrt2 = std::numbers::sqrt2_v<f256_s>;
    inline constexpr f256_s sqrt3 = std::numbers::sqrt3_v<f256_s>;
    inline constexpr f256_s inv_sqrt3 = std::numbers::inv_sqrt3_v<f256_s>;
    inline constexpr f256_s egamma = std::numbers::egamma_v<f256_s>;
    inline constexpr f256_s phi = std::numbers::phi_v<f256_s>;

    inline constexpr f256_s pi_2 = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54, -0x1.f1976b7ed8fbcp-110,  0x1.4cf98e804177dp-164 };
    inline constexpr f256_s pi_4 = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55, -0x1.f1976b7ed8fbcp-111,  0x1.4cf98e804177dp-165 };
    inline constexpr f256_s invpi2 = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55, -0x1.6447e493ad4cep-109,  0x1.e21c820ff28b2p-163 };
    inline constexpr f256_s pi_3_4 = pi_2 + pi_4;
    inline constexpr f256_s inv_ln2 = log2e;
    inline constexpr f256_s inv_ln10 = log10e;
    inline constexpr f256_s sqrt_half = { 0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55,  0x1.57d3e3adec175p-109,  0x1.2775099da2f59p-165 };
    inline constexpr f256_s half_log_two_pi = { 0x1.d67f1c864beb5p-1, -0x1.65b5a1b7ff5dfp-55, -0x1.b7f70c13dc1ccp-110, 0x1.3458b4ddec6a3p-164 };

    inline constexpr f256_s exp_inv_fact[] = {
        f256_s{ 1.66666666666666657e-01,  9.25185853854297066e-18,  5.13581318503262866e-34,  2.85094902409834186e-50 },
        f256_s{ 4.16666666666666644e-02,  2.31296463463574266e-18,  1.28395329625815716e-34,  7.12737256024585466e-51 },
        f256_s{ 8.33333333333333322e-03,  1.15648231731787138e-19,  1.60494162032269652e-36,  2.22730392507682967e-53 },
        f256_s{ 1.38888888888888894e-03, -5.30054395437357706e-20, -1.73868675534958776e-36, -1.63335621172300840e-52 },
        f256_s{ 1.98412698412698413e-04,  1.72095582934207053e-22,  1.49269123913941271e-40,  1.29470326746002471e-58 },
        f256_s{ 2.48015873015873016e-05,  2.15119478667758816e-23,  1.86586404892426588e-41,  1.61837908432503088e-59 },
        f256_s{ 2.75573192239858925e-06, -1.85839327404647208e-22,  8.49175460488199287e-39, -5.72661640789429621e-55 },
        f256_s{ 2.75573192239858883e-07,  2.37677146222502973e-23, -3.26318890334088294e-40,  1.61435111860404415e-56 },
        f256_s{ 2.50521083854417202e-08, -1.44881407093591197e-24,  2.04267351467144546e-41, -8.49632672007163175e-58 },
        f256_s{ 2.08767569878681002e-09, -1.20734505911325997e-25,  1.70222792889287100e-42,  1.41609532150396700e-58 },
        f256_s{ 1.60590438368216133e-10,  1.25852945887520981e-26, -5.31334602762985031e-43,  3.54021472597605528e-59 },
        f256_s{ 1.14707455977297245e-11,  2.06555127528307454e-28,  6.88907923246664603e-45,  5.72920002655109095e-61 },
        f256_s{ 7.64716373181981641e-13,  7.03872877733453001e-30, -7.82753927716258345e-48,  1.92138649443790242e-64 },
        f256_s{ 4.77947733238738525e-14,  4.39920548583408126e-31, -4.89221204822661465e-49,  1.20086655902368901e-65 },
        f256_s{ 2.81145725434552060e-15,  1.65088427308614326e-31, -2.87777179307447918e-50,  4.27110689256293549e-67 }
    };

    BL_NO_INLINE constexpr f256_s f256_expm1_tiny(const f256_s& r)
    {
        f256_s p = exp_inv_fact[(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 1];
        for (int i = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 2; i >= 0; --i)
            p = detail::_f256::f256_mul_add_horner_step(p, r, exp_inv_fact[i]);
        p = detail::_f256::f256_mul_add_horner_step(p, r, f256_s{ 0.5 });
        return detail::_f256::f256_mul_add_horner_step(detail::_f256::mul_inline(r, r), p, r);
    }
    BL_FORCE_INLINE constexpr f256_s f256_log1p_series_reduced(const f256_s& x)
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
    inline constexpr f256_s lgamma1p_coeff[] = {
        f256_s{ 0x1.a51a6625307d3p-1, 0x1.1873d8912200cp-56, -0x1.4c68528ddc956p-110, 0x1.162d8b33582c0p-168 },
        f256_s{ -0x1.9a4d55beab2d7p-2, 0x1.4c26d1b465993p-59, -0x1.aa121007a9210p-113, 0x1.e77545b273b41p-167 },
        f256_s{ 0x1.151322ac7d848p-2, 0x1.b5f91211196e5p-57, 0x1.1afde2c358986p-112, -0x1.ff5e9b485c055p-167 },
        f256_s{ -0x1.a8b9c17aa6149p-3, -0x1.2e826a4fdae1ap-58, 0x1.bec8fd99e4b23p-112, 0x1.c4d2077d59b06p-166 },
        f256_s{ 0x1.5b40cb100c306p-3, 0x1.4a79940f15696p-59, -0x1.38825ea888f47p-113, -0x1.465029c2b0433p-167 },
        f256_s{ -0x1.2703a1dcea3aep-3, -0x1.6307fd0794ac4p-57, 0x1.fcb7807245585p-111, 0x1.f0664358361f4p-166 },
        f256_s{ 0x1.010b36af86397p-3, -0x1.741a635b224a6p-59, 0x1.9336e1bce5c27p-113, 0x1.49afd8254897cp-167 },
        f256_s{ -0x1.c806706d57db4p-4, -0x1.56aa806fdd3eep-58, 0x1.bbb9c2de4a62ap-112, -0x1.3011b58722bd7p-167 },
        f256_s{ 0x1.9a01e385d5f8fp-4, 0x1.813418f3768cdp-59, 0x1.9ac3b8f78d2dbp-113, 0x1.f9d6fac7bc2bep-167 },
        f256_s{ -0x1.748c33114c6d6p-4, -0x1.ea57624080720p-61, 0x1.d4f09980d4de7p-116, -0x1.ecb44a07a7c5dp-170 },
        f256_s{ 0x1.556ad63243bc4p-4, 0x1.5de8580fae81dp-62, 0x1.cccd6abe647edp-119, 0x1.1be9a3144317ap-173 },
        f256_s{ -0x1.3b1d971fc5985p-4, 0x1.e58607e493dfdp-59, -0x1.abfc7225b8175p-113, -0x1.2ecb61bf48473p-169 },
        f256_s{ 0x1.2496df8320c5fp-4, 0x1.cf4b4ae040be8p-58, 0x1.4c882cc4762e8p-112, 0x1.7476f52945b0fp-166 },
        f256_s{ -0x1.11133476e7fe0p-4, -0x1.dc9a4ff396ee3p-59, -0x1.3b08c41a7a8b6p-113, 0x1.cf5c5597a8f3ep-168 },
        f256_s{ 0x1.00010064cdeb2p-4, 0x1.7879d0156affep-59, -0x1.0fbd29f2ffe91p-113, 0x1.89bce2341cdd7p-167 },
        f256_s{ -0x1.e1e2d311e8abdp-5, 0x1.8d2a110ce956bp-59, 0x1.63ee8a858cae0p-113, -0x1.e39f4153afc89p-167 },
        f256_s{ 0x1.c71ce3a20b419p-5, -0x1.be9617d035b06p-59, 0x1.89baba83cec5cp-115, -0x1.dcfb4b6decc54p-169 },
        f256_s{ -0x1.af28a1b5688a0p-5, -0x1.74741e885fefbp-59, -0x1.1b477c35fac2fp-113, 0x1.d7c330022aca2p-168 },
        f256_s{ 0x1.9999b3352d5bap-5, 0x1.4951b4c6be56dp-62, -0x1.ab8f8f67d6af8p-118, 0x1.986c8bd56b1b5p-172 },
        f256_s{ -0x1.86186db77bfbfp-5, -0x1.6dedef1f58778p-59, 0x1.f067254ca5106p-114, -0x1.fba80968fedbfp-168 },
        f256_s{ 0x1.745d1d1778df9p-5, 0x1.02b8fe0a898e7p-61, 0x1.c5b2f696e8978p-115, 0x1.8192959451748p-169 },
        f256_s{ -0x1.642c88591b66dp-5, 0x1.1074551cafc60p-59, -0x1.d738a4a5cbb56p-116, 0x1.1369f37f73d94p-170 },
        f256_s{ 0x1.555556aaafdcdp-5, 0x1.54a05fce04ef6p-59, 0x1.7588d216e6fc9p-114, -0x1.c8c5449ed87ecp-170 },
        f256_s{ -0x1.47ae151eb9fb7p-5, -0x1.d038d4d4653c2p-59, -0x1.7ebe42832f35dp-113, -0x1.28f5805cfb70ap-167 },
        f256_s{ 0x1.3b13b189d925ep-5, 0x1.f4ad5a89f860cp-59, 0x1.c3ba6ba46072cp-113, 0x1.dbc96b627a7eap-167 },
        f256_s{ -0x1.2f684c00002bcp-5, -0x1.055a3ba5e6a12p-59, 0x1.15f6f174f29fcp-114, -0x1.a4d95a3ce102dp-168 },
        f256_s{ 0x1.24924936db7bcp-5, 0x1.f2631c34f2cbcp-59, 0x1.6d33663f9d067p-116, -0x1.8e49cb0d4669dp-172 },
        f256_s{ -0x1.1a7b961a7b9aap-5, 0x1.e116d2f11b9bcp-59, -0x1.7ae84ff57e3e8p-113, 0x1.48411ad6c2f3cp-168 },
        f256_s{ 0x1.111111155556dp-5, -0x1.527ce242d7c8fp-59, -0x1.c6b75ac93ef74p-116, -0x1.59689b24da75ap-170 },
        f256_s{ -0x1.08421086318cep-5, 0x1.1db4d8fcae8c6p-59, -0x1.8aff5794eab89p-114, 0x1.29680715cc491p-169 },
        f256_s{ 0x1.0000000100002p-5, 0x1.b8fd913d3546ap-59, 0x1.815322257c298p-113, -0x1.6a18198001fe7p-168 },
        f256_s{ -0x1.f07c1f08ba2eap-6, -0x1.31bb2e9036633p-60, 0x1.151f1a77d048fp-115, -0x1.19b08afd4a13dp-171 },
        f256_s{ 0x1.e1e1e1e25a5a6p-6, 0x1.3e46eaa03f9ccp-61, 0x1.2f952ffc8aa9bp-115, 0x1.b1cb7727dbed5p-172 },
        f256_s{ -0x1.d41d41d457c58p-6, 0x1.0600661f0f0e3p-62, 0x1.8207e60d9c037p-116, 0x1.aa0de68815246p-170 },
        f256_s{ 0x1.c71c71c738e39p-6, -0x1.d93a55599cf57p-63, 0x1.4c492654868edp-117, 0x1.710b595a6ccadp-171 },
        f256_s{ -0x1.bacf914c29837p-6, -0x1.797fe7c73f29ap-60, 0x1.9a39cef32197ap-116, -0x1.a972d739b9af9p-171 },
        f256_s{ 0x1.af286bca21af3p-6, -0x1.df4d835f028bdp-60, 0x1.a8048d2fde67cp-114, -0x1.762e068021af3p-168 },
        f256_s{ -0x1.a41a41a41d89ep-6, 0x1.d6bf77cbc25c7p-60, 0x1.aa5ec7c867b9bp-114, 0x1.aec81d4fdf930p-169 },
        f256_s{ 0x1.999999999b333p-6, 0x1.9ad0584412591p-61, -0x1.5494ee7aa6acbp-115, 0x1.017dce762d13fp-170 },
        f256_s{ -0x1.8f9c18f9c2577p-6, 0x1.766fd061292d7p-60, 0x1.8f5129cb00964p-114, 0x1.c84bbfd5adbcep-170 },
        f256_s{ 0x1.8618618618c31p-6, -0x1.e77d97e1c5a45p-61, -0x1.917b2ef884989p-116, -0x1.2370e4fefbcf8p-171 },
        f256_s{ -0x1.7d05f417d08eep-6, -0x1.1dcf2bd1488c1p-61, 0x1.653c4b9cdc9cbp-119, 0x1.ce66a9ae95282p-173 },
        f256_s{ 0x1.745d1745d18bap-6, 0x1.7460941753bf5p-61, -0x1.7ff16da4c5b6dp-115, -0x1.a4622b9427056p-169 },
        f256_s{ -0x1.6c16c16c16ccdp-6, 0x1.9998769b89af0p-61, 0x1.e206e60394ebap-115, 0x1.da1d89108d728p-174 },
        f256_s{ 0x1.642c8590b21bdp-6, 0x1.bd3805d865a75p-61, 0x1.4fa64231f4873p-116, 0x1.48681770a7db2p-172 },
        f256_s{ -0x1.5c9882b931083p-6, 0x1.1b3bdabc05a8dp-60, -0x1.e180a2e6d8d8ap-115, -0x1.adf0f5b6511bdp-169 },
        f256_s{ 0x1.555555555556bp-6, -0x1.555550480911cp-60, -0x1.7e33c2f812c41p-115, 0x1.fc4f9b927ed2fp-173 },
        f256_s{ -0x1.4e5e0a72f0544p-6, 0x1.4e5e03d9bbd88p-62, 0x1.8ad74f7d864c5p-118, 0x1.de83badc29713p-174 },
        f256_s{ 0x1.47ae147ae1480p-6, 0x1.13e7474dcd9a5p-85, 0x1.f485e7aa34c2bp-140, 0x1.03954ca6cce4bp-194 },
        f256_s{ -0x1.4141414141417p-6, 0x1.a5a5a57890971p-60, 0x1.683bc59ca465cp-116, -0x1.bec1a58fd95fbp-170 },
        f256_s{ 0x1.3b13b13b13b15p-6, -0x1.3b13b1001f8aep-62, -0x1.204bb43c1d47fp-117, -0x1.0119b57ca7085p-171 },
        f256_s{ -0x1.3521cfb2b78c2p-6, 0x1.826a4395c1891p-61, 0x1.a4b38e003722bp-116, -0x1.cce6803948c97p-170 },
        f256_s{ 0x1.2f684bda12f69p-6, -0x1.a12f684a465ffp-60, 0x1.bad2fc9ce5f0bp-114, 0x1.bfc93a89e124bp-168 },
        f256_s{ -0x1.29e4129e4129ep-6, -0x1.9999999a1db84p-60, -0x1.4b1b19906daf9p-114, 0x1.dc501d5c7a0cep-168 },
        f256_s{ 0x1.2492492492492p-6, 0x1.6db6db6de21c5p-60, 0x1.c06a19ff7e3e7p-115, -0x1.c1d6ed17c4a8ep-173 },
        f256_s{ -0x1.1f7047dc11f70p-6, -0x1.435e50d7a2602p-60, 0x1.2b58ebdccf2bcp-114, 0x1.3df12156c2cb2p-168 },
        f256_s{ 0x1.1a7b9611a7b96p-6, 0x1.611a7b9624375p-62, -0x1.53910f155f6e2p-122, 0x1.cdf4e01c19e6fp-176 },
        f256_s{ -0x1.15b1e5f75270dp-6, -0x1.a08ad8f313fd5p-64, 0x1.694e002134230p-118, -0x1.19c40b553edeep-172 },
        f256_s{ 0x1.1111111111111p-6, 0x1.2222222224208p-62, -0x1.660eeab940148p-117, 0x1.3f841c542be73p-175 },
        f256_s{ -0x1.0c9714fbcda3bp-6, 0x1.f368eb043208bp-61, -0x1.9180e02b1ebb4p-115, -0x1.ae60aa3d03da9p-169 },
        f256_s{ 0x1.0842108421084p-6, 0x1.0a5294a52965cp-61, 0x1.6a716f3fe41bbp-116, -0x1.3966f9e59e4cep-171 },
        f256_s{ -0x1.0410410410410p-6, -0x1.04924924924dap-60, -0x1.2c9f914026d3ep-114, 0x1.1aa441e1febb5p-170 },
        f256_s{ 0x1.0000000000000p-6, 0x1.0000000005e83p-70, -0x1.6ea6b12420976p-124, 0x1.46b6024a71bf4p-179 },
        f256_s{ -0x1.f81f81f81f820p-7, 0x1.f7e07e07e07d1p-61, -0x1.17cedd5cd4340p-119, -0x1.231cb51f5fa1ap-175 },
        f256_s{ 0x1.f07c1f07c1f08p-7, -0x1.f03e0f83e0f7ap-62, 0x1.37710ca3a5bb1p-116, -0x1.c5ed76f77cd34p-171 },
        f256_s{ -0x1.e9131abf0b767p-7, -0x1.505bb39503d26p-62, 0x1.12bb0a2d9bbe6p-116, -0x1.08b90d7460356p-173 },
        f256_s{ 0x1.e1e1e1e1e1e1ep-7, 0x1.e200000000002p-63, 0x1.920f4caafc2b6p-118, -0x1.71978c07e1374p-172 },
        f256_s{ -0x1.dae6076b981dbp-7, 0x1.9f7a6f4de9bd3p-63, -0x1.f2252e21e87a9p-118, 0x1.83d4b1dbeca59p-172 },
        f256_s{ 0x1.d41d41d41d41dp-7, 0x1.0752492492492p-61, 0x1.614270fa6214ep-115, -0x1.f77a00fa73fe6p-171 },
        f256_s{ -0x1.cd85689039b0bp-7, 0x1.76fa976fc64f5p-62, 0x1.2735476e24ed0p-117, -0x1.f8b4320455699p-173 },
        f256_s{ 0x1.c71c71c71c71cp-7, 0x1.c71ce38e38e39p-61, -0x1.ace34a7163d76p-117, -0x1.1f77c846a312ap-172 },
        f256_s{ -0x1.c0e070381c0e0p-7, -0x1.c0e0a8542a151p-61, 0x1.5c87938fc211bp-115, 0x1.f7dd13bd0c614p-170 },
        f256_s{ 0x1.bacf914c1bad0p-7, -0x1.bacf759f22983p-61, -0x1.d5c71a9cd2d7bp-115, -0x1.f2ba4ed0c813cp-171 },
        f256_s{ -0x1.b4e81b4e81b4fp-7, 0x1.f92c51eb851ecp-61, -0x1.ebc0cadfc39dcp-115, -0x1.d86dcb378da2fp-172 },
        f256_s{ 0x1.af286bca1af28p-7, 0x1.af287286bca1bp-61, -0x1.ae8b63624e57ep-118, -0x1.858b16f233d73p-176 },
        f256_s{ -0x1.a98ef606a63bep-7, 0x1.f959c0d4c77b0p-61, 0x1.a9820b7fc62f2p-116, -0x1.f72b8dbc6b5d5p-175 },
        f256_s{ 0x1.a41a41a41a41ap-7, 0x1.06906aaaaaaabp-61, -0x1.5553354f8d76dp-115, 0x1.0d86d971be33fp-169 },
        f256_s{ -0x1.9ec8e951033d9p-7, -0x1.d2a209b8b577ep-63, -0x1.84df26ea720dcp-117, 0x1.af04d98dfad02p-172 },
        f256_s{ 0x1.999999999999ap-7, -0x1.9999993333333p-61, -0x1.999923ba7b188p-116, 0x1.c2ee41f7b15dcp-170 },
        f256_s{ -0x1.948b0fcd6e9e0p-7, -0x1.948b100000000p-61, -0x1.3671909a8aa7dp-135, -0x1.13181f3d7138fp-189 },
        f256_s{ 0x1.8f9c18f9c18fap-7, -0x1.f3831f063e706p-62, -0x1.f38305aa29f35p-117, 0x1.a672177cf91d8p-171 },
        f256_s{ -0x1.8acb90f6bf3aap-7, 0x1.721ed7dafcea7p-61, -0x1.c87b61b7e6f4bp-115, -0x1.5d61cf8be7a0cp-169 },
        f256_s{ 0x1.8618618618618p-7, 0x1.8618618c30c31p-61, -0x1.e79e7884d38b8p-116, -0x1.9d9fb59a71375p-174 },
        f256_s{ -0x1.8181818181818p-7, -0x1.8181818d8d8d9p-63, 0x1.39393765bb628p-118, -0x1.3b4d334cdaa33p-172 },
        f256_s{ 0x1.7d05f417d05f4p-7, 0x1.7d05f41dc4771p-63, 0x1.dc477251cdf4bp-119, 0x1.b79aeab06c17dp-176 },
        f256_s{ -0x1.78a4c8178a4c8p-7, -0x1.78a4c81a7b961p-63, -0x1.a7b9617ffb47dp-119, 0x1.cb1cecaaf7e8ap-174 },
        f256_s{ 0x1.745d1745d1746p-7, -0x1.745d17451745dp-62, -0x1.745d1735180cep-118, -0x1.65e2a01ab292dp-172 },
        f256_s{ -0x1.702e05c0b8170p-7, -0x1.702e05c114228p-62, -0x1.14228451ead7ap-116, -0x1.83351baaf0d6fp-174 },
        f256_s{ 0x1.6c16c16c16c17p-7, -0x1.f49f49f471c72p-62, 0x1.c71c71c80503cp-117, -0x1.a7fcfb6c46765p-171 },
        f256_s{ -0x1.6816816816817p-7, 0x1.fa5fa5fa54654p-61, 0x1.951951950626ap-115, -0x1.09210a392fa45p-169 },
        f256_s{ 0x1.642c8590b2164p-7, 0x1.642c8590bd37ap-62, 0x1.bd37a6f4eb3f9p-116, 0x1.60eb1b25b5ad2p-170 },
        f256_s{ -0x1.6058160581606p-7, 0x1.fa7e9fa7e739dp-61, -0x1.8c6318c639e26p-117, 0x1.7351b5a7b933ap-172 },
        f256_s{ 0x1.5c9882b931057p-7, 0x1.310572620d9dfp-62, 0x1.46cefa8d9f550p-116, -0x1.db2f84890bedcp-171 },
        f256_s{ -0x1.58ed2308158edp-7, -0x1.1840ac7692dcfp-62, -0x1.fa9c4b73e01ddp-116, -0x1.d473788da23dbp-171 },
        f256_s{ 0x1.5555555555555p-7, 0x1.5555555555aabp-61, -0x1.5555555555423p-115, -0x1.174105986442cp-171 },
        f256_s{ -0x1.51d07eae2f815p-7, -0x1.d07eae2f81facp-63, 0x1.d07eae2f81389p-117, -0x1.cdf5daadab33dp-173 },
        f256_s{ 0x1.4e5e0a72f0539p-7, 0x1.e0a72f05398d1p-61, -0x1.4e5e0a72f0324p-119, -0x1.c12afec73670cp-175 },
        f256_s{ -0x1.4afd6a052bf5bp-7, 0x1.fad40a57eb45dp-61, 0x1.745d1745d171ap-117, -0x1.6fa0eb9403245p-172 },
        f256_s{ 0x1.47ae147ae147bp-7, -0x1.eb851eb851d71p-63, 0x1.70a3d70a3d719p-117, -0x1.ed1522e63fd11p-172 },
        f256_s{ -0x1.446f86562d9fbp-7, 0x1.1be1958b67e19p-63, 0x1.62d9faee41e66p-117, -0x1.58747c5a6bc0cp-171 },
        f256_s{ 0x1.4141414141414p-7, 0x1.4141414141464p-63, 0x1.919191919191bp-117, -0x1.480866d9ca729p-171 },
        f256_s{ -0x1.3e22cbce4a902p-7, -0x1.f1165e725481ep-61, 0x1.65e7254813e23p-116, -0x1.dc15cf8adce0bp-170 },
        f256_s{ 0x1.3b13b13b13b14p-7, -0x1.3b13b13b13b0fp-61, 0x1.d89d89d89d89ep-116, -0x1.805e8b12fa608p-170 },
        f256_s{ -0x1.3813813813814p-7, 0x1.fb1fb1fb1fb1dp-61, 0x1.0750750750750p-115, 0x1.c58bf33315042p-169 },
        f256_s{ 0x1.3521cfb2b78c1p-7, 0x1.a90e7d95bc60cp-62, 0x1.3521cfb2b78c1p-118, 0x1.f6047d965bd53p-173 },
        f256_s{ -0x1.323e34a2b10bfp-7, -0x1.9b8396ba9de82p-61, 0x1.a515885fb3707p-116, 0x1.654f614bb639ep-171 },
        f256_s{ 0x1.2f684bda12f68p-7, 0x1.2f684bda12f69p-61, -0x1.a12f684bda12fp-115, -0x1.a0a91f24de414p-169 },
        f256_s{ -0x1.2c9fb4d812ca0p-7, 0x1.2c9fb4d812ca0p-61, -0x1.c2ef8f441c2f0p-115, 0x1.c2c335581d907p-169 },
        f256_s{ 0x1.29e4129e4129ep-7, 0x1.04a7904a7904bp-61, -0x1.d1745d1745d17p-115, -0x1.17372b38419eap-169 },
        f256_s{ -0x1.27350b8812735p-7, -0x1.71024e6a17103p-64, 0x1.9f22983759f23p-118, -0x1.9f494ebf2163bp-172 },
        f256_s{ 0x1.2492492492492p-7, 0x1.2492492492492p-61, 0x1.36db6db6db6dbp-115, 0x1.b6dd06f7523c6p-169 }
    };

    BL_NO_INLINE constexpr f256_s lgamma1p_series(const f256_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f256_s p = lgamma1p_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = p * y + lgamma1p_coeff[i];

        return y * (-detail::_f256::egamma + y * p);
    }

    BL_NO_INLINE constexpr bool try_lgamma_near_one_or_two(const f256_s& x, f256_s& out) noexcept
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

    BL_FORCE_INLINE constexpr bool f256_remainder_pi2(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        if (!detail::_f256::isfinite(x.x0))
            return false;

        if (abs(x) <= detail::_f256::pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256_s q = nearbyint(x * detail::_f256::invpi2);
        const double qd = q.x0;

        if (!detail::fp::isfinite(qd) || detail::fp::absd(qd) > 9.0e15)
        {
            const double xd = static_cast<double>(x);
            const double fallback_qd = (double)detail::fp::llround_constexpr(xd * static_cast<double>(detail::_f256::invpi2));

            if (!detail::fp::isfinite(fallback_qd) || detail::fp::absd(fallback_qd) > 9.0e15)
                return false;

            const long long n = (long long)fallback_qd;
            const f256_s qf{ (double)n };

            f256_s r = x;
            r -= qf * detail::_f256::pi_2.x0;
            r -= qf * detail::_f256::pi_2.x1;
            r -= qf * detail::_f256::pi_2.x2;
            r -= qf * detail::_f256::pi_2.x3;

            if (r > detail::_f256::pi_4)
            {
                r -= detail::_f256::pi_2;
                n_out = n + 1;
            }
            else if (r < -detail::_f256::pi_4)
            {
                r += detail::_f256::pi_2;
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
        r -= q * detail::_f256::pi_2.x0;
        r -= q * detail::_f256::pi_2.x1;
        r -= q * detail::_f256::pi_2.x2;
        r -= q * detail::_f256::pi_2.x3;

        if (r > detail::_f256::pi_4)
        {
            r -= detail::_f256::pi_2;
            ++n;
        }
        else if (r < -detail::_f256::pi_4)
        {
            r += detail::_f256::pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }
    inline constexpr f256_s f256_sin_coeffs_pi4[] = {
        {  0x1.5a42f0dfeb086p-209, -0x1.35ae015f78f6ep-264, -0x1.c71a521ce2e79p-318,  0x1.6a300230ce998p-372 },
        { -0x1.8da8e0a127ebap-198,  0x1.21d2eac9d275cp-252,  0x1.ad541d26964afp-306,  0x1.1c066ebdf95dep-360 },
        {  0x1.a3cb872220648p-187, -0x1.c7f4e85b8e6cdp-241, -0x1.413a0bc5fc28ap-295, -0x1.16ae534063fabp-352 },
        { -0x1.95db45257e512p-176, -0x1.6e5d72b6f79b9p-231, -0x1.b830cf0b5b5c6p-291, -0x1.29276833f5728p-345 },
        {  0x1.65e61c39d0241p-165, -0x1.c0ed181727269p-220, -0x1.abbd2f56bbc2fp-276, -0x1.18ff57fdc2e4ep-330 },
        { -0x1.1e99449a4bacep-154,  0x1.fefbb89514b3cp-210,  0x1.53433f743a2d9p-264, -0x1.25f70d1395dd7p-320 },
        {  0x1.9ec8d1c94e85bp-144, -0x1.670e9d4784ec6p-201,  0x1.79fe5954939a2p-255,  0x1.82e418d9b0c9ep-311 },
        { -0x1.0dc59c716d91fp-133, -0x1.419e3fad3f031p-188, -0x1.d9d7ed1981ffcp-244,  0x1.345ea5d66a84bp-300 },
        {  0x1.3981254dd0d52p-123, -0x1.2b1f4c8015a2fp-177, -0x1.d82af23edb6dbp-231,  0x1.a1cd20123a99bp-285 },
        { -0x1.434d2e783f5bcp-113, -0x1.0b87b91be9affp-167, -0x1.c89db1796db75p-224,  0x1.8923b7699c8bep-278 },
        {  0x1.259f98b4358adp-103,  0x1.eaf8c39dd9bc5p-157, -0x1.6e29990a26fb6p-211, -0x1.2d867809b5568p-267 },
        { -0x1.d1ab1c2dccea3p-94,  -0x1.054d0c78aea14p-149,  0x1.196bf16c33a56p-203, -0x1.f0e65ed04d346p-257 },
        {  0x1.3f3ccdd165fa9p-84,  -0x1.58ddadf344487p-139, -0x1.e8ed8001ad67ep-193,  0x1.80a5edffcced7p-247 },
        { -0x1.761b41316381ap-75,   0x1.3423c7d91404fp-130, -0x1.e6135bfc1194ap-185,  0x1.ba7b1a3077b39p-239 },
        {  0x1.71b8ef6dcf572p-66,  -0x1.d043ae40c4647p-120,  0x1.486121e81d5fep-176, -0x1.2d4ba8e1e64c7p-230 },
        { -0x1.2f49b46814157p-57,  -0x1.2650f61dbdcb4p-112,  0x1.69502917cbf3bp-166, -0x1.e35fbddac4553p-223 },
        {  0x1.952c77030ad4ap-49,   0x1.ac981465ddc6cp-103, -0x1.588b72e53bc5fp-165,  0x1.7079e8909271ap-221 },
        { -0x1.ae7f3e733b81fp-41,  -0x1.1d8656b0ee8cbp-97,   0x1.6e142a138f825p-157, -0x1.43c0c38ccdcc6p-212 },
        {  0x1.6124613a86d09p-33,   0x1.f28e0cc748ebep-87,  -0x1.7b2c4c8a840bcp-141,  0x1.c71cca1034c07p-195 },
        { -0x1.ae64567f544e4p-26,   0x1.c062e06d1f209p-80,  -0x1.c7880adcbc46ep-136,  0x1.5553a6f0fed60p-190 },
        {  0x1.71de3a556c734p-19,  -0x1.c154f8ddc6c00p-73,   0x1.71de3a556c734p-127, -0x1.c154f8ddc6c00p-181 },
        { -0x1.a01a01a01a01ap-13,  -0x1.a01a01a01a01ap-73,  -0x1.a01a01a01a01ap-133, -0x1.a01a01a01a01ap-193 },
        {  0x1.1111111111111p-7,    0x1.1111111111111p-63,   0x1.1111111111111p-119,  0x1.1111111111111p-175 },
        { -0x1.5555555555555p-3,   -0x1.5555555555555p-57,  -0x1.5555555555555p-111, -0x1.5555555555555p-165 }
    };
    inline constexpr f256_s f256_cos_coeffs_pi4[] = {
        {  0x1.091b406b6ff26p-203,  0x1.e973637973b18p-257, -0x1.1e38136f0edcap-311, -0x1.7ab33e52a1d28p-366 },
        { -0x1.240804f659510p-192, -0x1.8b291b93c9718p-246, -0x1.096c752f5341fp-301,  0x1.c12972a70641ep-355 },
        {  0x1.272b1b03fec6ap-181,  0x1.3f67cc9f9fdb8p-235, -0x1.71dcd047354c9p-289, -0x1.c3f29289464c4p-346 },
        { -0x1.10af527530de8p-170, -0x1.b626c912ee5c8p-225, -0x1.349f032c6e859p-279,  0x1.ec616617f45c6p-333 },
        {  0x1.ca8ed42a12ae3p-160,  0x1.a07244abad2abp-224,  0x1.facdac6fb71b7p-278, -0x1.ca2f486d514e1p-339 },
        { -0x1.5d4acb9c0c3abp-149,  0x1.6ec2c8f5b13b2p-205, -0x1.e2860aaa59188p-259,  0x1.866eba0408569p-313 },
        {  0x1.df983290c2ca9p-139,  0x1.5835c6895393bp-194, -0x1.0578f45b1aaaep-249, -0x1.281508688972dp-303 },
        { -0x1.2710231c0fd7ap-128, -0x1.3f8a2b4af9d6bp-184, -0x1.c32215a9f317ep-238,  0x1.d451e158a1205p-293 },
        {  0x1.434d2e783f5bcp-118,  0x1.0b87b91be9affp-172,  0x1.c89db1796db75p-229, -0x1.8923b7699c8bep-283 },
        { -0x1.3932c5047d60ep-108, -0x1.832b7b530a627p-162, -0x1.5d2c61f6d124cp-218, -0x1.f192b328d82c4p-272 },
        {  0x1.0a18a2635085dp-98,   0x1.b9e2e28e1aa54p-153,  0x1.a8549a9d99586p-207, -0x1.141dcc8cc5668p-266 },
        { -0x1.88e85fc6a4e5ap-89,   0x1.71c37ebd16540p-143, -0x1.494676265a364p-197,  0x1.397b40007db79p-253 },
        {  0x1.f2cf01972f578p-80,  -0x1.9ada5fcc1ab14p-135,  0x1.440ce7fd610dcp-189, -0x1.26fcbc204fcd1p-243 },
        { -0x1.0ce396db7f853p-70,   0x1.aebcdbd20331cp-124,  0x1.38a88578b4d75p-178, -0x1.c0fbc29694fb8p-233 },
        {  0x1.e542ba4020225p-62,   0x1.ea72b4afe3c2fp-120, -0x1.44020dfd65c8cp-174, -0x1.6e69b50fc88abp-231 },
        { -0x1.6827863b97d97p-53,  -0x1.eec01221a8b0bp-107,  0x1.568798662118bp-161, -0x1.f00d8b9e49291p-222 },
        {  0x1.ae7f3e733b81fp-45,   0x1.1d8656b0ee8cbp-101, -0x1.6e142a138f825p-161,  0x1.43c0c38ccdcc6p-216 },
        { -0x1.93974a8c07c9dp-37,  -0x1.05d6f8a2efd1fp-92,  -0x1.3aa3346236a5dp-147, -0x1.d75f096ea801ep-201 },
        {  0x1.1eed8eff8d898p-29,  -0x1.2aec959e14c06p-83,   0x1.2fb0073dd2d9ep-139,  0x1.c71d90b4ab715p-193 },
        { -0x1.27e4fb7789f5cp-22,  -0x1.cbbc05b4fa99ap-76,   0x1.c6d278883e8f5p-132, -0x1.95567d3a50ccep-186 },
        {  0x1.a01a01a01a01ap-16,   0x1.a01a01a01a01ap-76,   0x1.a01a01a01a01ap-136,  0x1.a01a01a01a01ap-196 },
        { -0x1.6c16c16c16c17p-10,   0x1.f49f49f49f49fp-65,   0x1.27d27d27d27d2p-119,  0x1.f49f49f49f49fp-173 },
        {  0x1.5555555555555p-5,    0x1.5555555555555p-59,   0x1.5555555555555p-113,  0x1.5555555555555p-167 },
        { -0x1.0000000000000p-1,    0x0.0p+0,                0x0.0p+0,                0x0.0p+0 }
    };
    inline constexpr std::size_t f256_trig_coeff_count_pi4 = sizeof(f256_sin_coeffs_pi4) / sizeof(f256_sin_coeffs_pi4[0]);
    inline constexpr f256_s f256_atan_tiny_coeffs[] = {
        {  0x1.0000000000000p+0,  0x0.0p+0,               0x0.0p+0,                0x0.0p+0 },
        { -0x1.5555555555555p-2, -0x1.5555555555555p-56, -0x1.5555555555555p-110, -0x1.5555555555555p-164 },
        {  0x1.999999999999ap-3, -0x1.999999999999ap-57,  0x1.999999999999ap-111, -0x1.999999999999ap-165 },
        { -0x1.2492492492492p-3, -0x1.2492492492492p-57, -0x1.2492492492492p-111, -0x1.2492492492492p-165 },
        {  0x1.c71c71c71c71cp-4,  0x1.c71c71c71c71cp-58,  0x1.c71c71c71c71cp-112,  0x1.c71c71c71c71cp-166 },
        { -0x1.745d1745d1746p-4,  0x1.745d1745d1746p-59, -0x1.745d1745d1746p-114,  0x1.745d1745d1746p-169 },
        {  0x1.3b13b13b13b14p-4, -0x1.3b13b13b13b14p-58,  0x1.3b13b13b13b14p-112, -0x1.3b13b13b13b14p-166 },
        { -0x1.1111111111111p-4, -0x1.1111111111111p-60, -0x1.1111111111111p-116, -0x1.1111111111111p-172 },
        {  0x1.e1e1e1e1e1e1ep-5,  0x1.e1e1e1e1e1e1ep-61,  0x1.e1e1e1e1e1e1ep-117,  0x1.e1e1e1e1e1e1ep-173 },
        { -0x1.af286bca1af28p-5, -0x1.af286bca1af28p-59, -0x1.af286bca1af28p-113, -0x1.af286bca1af28p-167 },
        {  0x1.8618618618618p-5,  0x1.8618618618618p-59,  0x1.8618618618618p-113,  0x1.8618618618618p-167 }
    };
    inline constexpr std::size_t f256_atan_tiny_coeff_count =
        sizeof(f256_atan_tiny_coeffs) / sizeof(f256_atan_tiny_coeffs[0]);

    #if BL_F256_ENABLE_SIMD
    BL_FORCE_INLINE __m128d f256_trig_simd_set(double lane0, double lane1) noexcept
    {
        return _mm_set_pd(lane1, lane0);
    }
    BL_FORCE_INLINE void f256_trig_simd_store(__m128d value, double& lane0, double& lane1) noexcept
    {
        alignas(16) double lanes[2];
        _mm_storeu_pd(lanes, value);
        lane0 = lanes[0];
        lane1 = lanes[1];
    }
    BL_FORCE_INLINE void f256_trig_simd_two_sum(__m128d a, __m128d b, __m128d& s, __m128d& e) noexcept
    {
        s = _mm_add_pd(a, b);
        const __m128d bb = _mm_sub_pd(s, a);
        e = _mm_add_pd(_mm_sub_pd(a, _mm_sub_pd(s, bb)), _mm_sub_pd(b, bb));
    }
    BL_FORCE_INLINE void f256_trig_simd_two_prod(__m128d a, __m128d b, __m128d& p, __m128d& e) noexcept
    {
        p = _mm_mul_pd(a, b);

        const __m128d split = _mm_set1_pd(134217729.0);
        const __m128d a_scaled = _mm_mul_pd(a, split);
        const __m128d b_scaled = _mm_mul_pd(b, split);

        const __m128d a_hi = _mm_sub_pd(a_scaled, _mm_sub_pd(a_scaled, a));
        const __m128d b_hi = _mm_sub_pd(b_scaled, _mm_sub_pd(b_scaled, b));
        const __m128d a_lo = _mm_sub_pd(a, a_hi);
        const __m128d b_lo = _mm_sub_pd(b, b_hi);

        e = _mm_add_pd(
            _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a_hi, b_hi), p), _mm_mul_pd(a_hi, b_lo)),
            _mm_add_pd(_mm_mul_pd(a_lo, b_hi), _mm_mul_pd(a_lo, b_lo))
        );
    }
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

        detail::_f256::three_sum(p1, p2, q0);
        detail::_f256::three_sum(p2, q1, q2);
        detail::_f256::three_sum(p3, p4, p5);

        detail::_f256::two_sum_precise(p2, p3, s0, t0);
        detail::_f256::two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        detail::_f256::two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        detail::_f256::two_sum_precise(q0, q3, q0, q3);
        detail::_f256::two_sum_precise(q4, q5, q4, q5);
        detail::_f256::two_sum_precise(p6, p7, p6, p7);
        detail::_f256::two_sum_precise(p8, p9, p8, p9);

        detail::_f256::two_sum_precise(q0, q4, t0, t1);
        t1 += (q3 + q5);

        detail::_f256::two_sum_precise(p6, p8, r0, r1);
        r1 += (p7 + p9);

        detail::_f256::two_sum_precise(t0, r0, q3, q4);
        q4 += (t1 + r1);

        detail::_f256::two_sum_precise(q3, s1, t0, t1);
        t1 += q4;

        t1 += tail_mul0 + tail_mul1 + tail_mul2
            + q6 + q7 + q8 + q9 + s2;

        return detail::_f256::renorm5(p0, p1, s0, t0, t1);
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

        detail::_f256::two_prod_precise(a0.x0, b0.x0, p00, q00);
        detail::_f256::two_prod_precise(a0.x0, b0.x1, p10, q10);
        detail::_f256::two_prod_precise(a0.x1, b0.x0, p20, q20);
        detail::_f256::two_prod_precise(a0.x0, b0.x2, p30, q30);
        detail::_f256::two_prod_precise(a0.x1, b0.x1, p40, q40);
        detail::_f256::two_prod_precise(a0.x2, b0.x0, p50, q50);

        detail::_f256::two_prod_precise(a1.x0, b1.x0, p01, q01);
        detail::_f256::two_prod_precise(a1.x0, b1.x1, p11, q11);
        detail::_f256::two_prod_precise(a1.x1, b1.x0, p21, q21);
        detail::_f256::two_prod_precise(a1.x0, b1.x2, p31, q31);
        detail::_f256::two_prod_precise(a1.x1, b1.x1, p41, q41);
        detail::_f256::two_prod_precise(a1.x2, b1.x0, p51, q51);

        const __m128d ax0 = f256_trig_simd_set(a0.x0, a1.x0);
        const __m128d ax1 = f256_trig_simd_set(a0.x1, a1.x1);
        const __m128d ax2 = f256_trig_simd_set(a0.x2, a1.x2);
        const __m128d ax3 = f256_trig_simd_set(a0.x3, a1.x3);

        const __m128d bx0 = f256_trig_simd_set(b0.x0, b1.x0);
        const __m128d bx1 = f256_trig_simd_set(b0.x1, b1.x1);
        const __m128d bx2 = f256_trig_simd_set(b0.x2, b1.x2);
        const __m128d bx3 = f256_trig_simd_set(b0.x3, b1.x3);

        __m128d p6{}, p7{}, p8{}, p9{};
        __m128d q6{}, q7{}, q8{}, q9{};

        f256_trig_simd_two_prod(ax0, bx3, p6, q6);
        f256_trig_simd_two_prod(ax1, bx2, p7, q7);
        f256_trig_simd_two_prod(ax2, bx1, p8, q8);
        f256_trig_simd_two_prod(ax3, bx0, p9, q9);

        alignas(16) double p6v[2], p7v[2], p8v[2], p9v[2];
        alignas(16) double q6v[2], q7v[2], q8v[2], q9v[2];

        _mm_storeu_pd(p6v, p6);
        _mm_storeu_pd(p7v, p7);
        _mm_storeu_pd(p8v, p8);
        _mm_storeu_pd(p9v, p9);
        _mm_storeu_pd(q6v, q6);
        _mm_storeu_pd(q7v, q7);
        _mm_storeu_pd(q8v, q8);
        _mm_storeu_pd(q9v, q9);

        out0 = detail::_f256::f256_mul_from_two_prod_terms(
            p00, p10, p20, p30, p40, p50,
            p6v[0], p7v[0], p8v[0], p9v[0],
            q00, q10, q20, q30, q40, q50,
            q6v[0], q7v[0], q8v[0], q9v[0],
            a0.x1 * b0.x3, a0.x2 * b0.x2, a0.x3 * b0.x1
        );

        out1 = detail::_f256::f256_mul_from_two_prod_terms(
            p01, p11, p21, p31, p41, p51,
            p6v[1], p7v[1], p8v[1], p9v[1],
            q01, q11, q21, q31, q41, q51,
            q6v[1], q7v[1], q8v[1], q9v[1],
            a1.x1 * b1.x3, a1.x2 * b1.x2, a1.x3 * b1.x1
        );
    }
    #endif

    BL_NO_INLINE constexpr f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double m0{}, m1{}, m2{};

        detail::_f256::two_prod_precise(a.x0, b.x0, p0, q0);
        detail::_f256::two_prod_precise(a.x0, b.x1, p1, q1);
        detail::_f256::two_prod_precise(a.x1, b.x0, p2, q2);
        detail::_f256::two_prod_precise(a.x0, b.x2, p3, q3);
        detail::_f256::two_prod_precise(a.x1, b.x1, p4, q4);
        detail::_f256::two_prod_precise(a.x2, b.x0, p5, q5);

        detail::_f256::three_sum(p1, p2, q0);
        detail::_f256::three_sum(p2, q1, q2);
        detail::_f256::three_sum(p3, p4, p5);

        detail::_f256::two_sum_precise(p2, p3, m0, t0);
        detail::_f256::two_sum_precise(q1, p4, m1, t1); m2 = q2 + p5;
        detail::_f256::two_sum_precise(m1, t0, m1, t0); m2 += (t0 + t1);

        detail::_f256::two_prod_precise(a.x0, b.x3, p6, q6);
        detail::_f256::two_prod_precise(a.x1, b.x2, p7, q7);
        detail::_f256::two_prod_precise(a.x2, b.x1, p8, q8);
        detail::_f256::two_prod_precise(a.x3, b.x0, p9, q9);

        detail::_f256::two_sum_precise(q0, q3, q0, q3);
        detail::_f256::two_sum_precise(q4, q5, q4, q5);
        detail::_f256::two_sum_precise(p6, p7, p6, p7);
        detail::_f256::two_sum_precise(p8, p9, p8, p9);

        detail::_f256::two_sum_precise(q0, q4, t0, t1); t1 += (q3 + q5);
        detail::_f256::two_sum_precise(p6, p8, r0, r1); r1 += (p7 + p9);
        detail::_f256::two_sum_precise(t0, r0, q3, q4); q4 += (t1 + r1);

        detail::_f256::two_sum_precise(q3, m1, t0, t1);
        t1 += q4;
        t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + m2;

        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        detail::_f256::two_sum_precise(p0, c.x0, s0, e0);
        detail::_f256::two_sum_precise(p1, c.x1, s1, e1);
        detail::_f256::two_sum_precise(m0, c.x2, s2, e2);
        detail::_f256::two_sum_precise(t0, c.x3, s3, e3);

        detail::_f256::two_sum_precise(s1, e0, s1, e0);
        detail::_f256::three_sum(s2, e0, e1);
        detail::_f256::three_sum2(s3, e0, e2);

        e0 += e1 + e3 + t1;
        return detail::_f256::renorm5(s0, s1, s2, s3, e0);
    }

    BL_NO_INLINE constexpr f256_s f256_sin_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s ps = f256_sin_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            ps = f256_mul_add_horner_step(ps, t, f256_sin_coeffs_pi4[i]);

        const f256_s rt = r * t;
        return f256_mul_add_horner_step(rt, ps, r);
    }
    BL_NO_INLINE constexpr f256_s f256_cos_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s pc = f256_cos_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            pc = f256_mul_add_horner_step(pc, t, f256_cos_coeffs_pi4[i]);

        return f256_mul_add_horner_step(t, pc, f256_s{ 1.0 });
    }
    BL_NO_INLINE constexpr void f256_sincos_kernel_pi4(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        using namespace detail::_f256;

        const f256_s t = mul_inline(r, r);

        f256_s ps = f256_sin_coeffs_pi4[0];
        f256_s pc = f256_cos_coeffs_pi4[0];

        #if BL_F256_ENABLE_SIMD
        if (detail::_f256::f256_runtime_trig_simd_enabled())
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
            return detail::_f256::renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
        }
        else
        {
            #if BL_F256_ENABLE_SIMD
            if (detail::_f256::f256_runtime_simd_enabled())
            {
                const __m128d scale = detail::_f256::f256_simd_splat(s);
                __m128d lo = _mm_mul_pd(detail::_f256::f256_simd_set(a.x0, a.x1), scale);
                __m128d hi = _mm_mul_pd(detail::_f256::f256_simd_set(a.x2, a.x3), scale);
                double x0{}, x1{}, x2{}, x3{};
                detail::_f256::f256_simd_store(lo, x0, x1);
                detail::_f256::f256_simd_store(hi, x2, x3);
                return detail::_f256::renorm(x0, x1, x2, x3);
            }
            else
            #endif
            {
                return detail::_f256::renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
            }
        }
    }
    BL_NO_INLINE constexpr f256_s _exp(const f256_s& x)
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

        const f256_s t = x * detail::_f256::inv_ln2;

        double kd = detail::_f256::nearbyint_ties_even(t.x0);
        const f256_s delta = t - f256_s{ kd };
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s kd_ln2 = detail::_f256::mul_double_inline(detail::_f256::ln2, kd);
        const f256_s r = detail::_f256::mul_double_inline(detail::_f256::sub_inline(x, kd_ln2), 0.0009765625);

        f256_s e = detail::_f256::f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = detail::_f256::f256_mul_add_horner_step(e, e, detail::_f256::mul_double_inline(e, 2.0));

        return _ldexp(detail::_f256::add_scalar_precise(e, 1.0), k);
    }
    BL_NO_INLINE constexpr f256_s _exp2(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 1023.0 || x.x0 < -1074.0)
            return detail::_f256::_exp(detail::_f256::mul_inline(x, detail::_f256::ln2));

        if (iszero(x))
            return f256_s{ 1.0 };

        double kd = detail::_f256::nearbyint_ties_even(x.x0);
        const f256_s delta = detail::_f256::sub_inline(x, f256_s{ kd });
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s reduced = detail::_f256::sub_inline(x, f256_s{ kd });
        const f256_s r = detail::_f256::mul_double_inline(detail::_f256::mul_inline(reduced, detail::_f256::ln2), 0.0009765625);

        f256_s e = detail::_f256::f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = detail::_f256::f256_mul_add_horner_step(e, e, detail::_f256::mul_double_inline(e, 2.0));

        return detail::_f256::_ldexp(detail::_f256::add_scalar_precise(e, 1.0), k);
    }
    BL_NO_INLINE constexpr f256_s _log(const f256_s& a)
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
            exp2 = detail::_f256::frexp_exponent_constexpr(a.x0);
        }
        else {
            (void)std::frexp(a.x0, &exp2);
        }

        f256_s m = _ldexp(a, -exp2);
        if (m < detail::_f256::sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }

        f256_s y = f256_s{ (double)exp2 } * detail::_f256::ln2 + f256_s{ log_as_double(m), 0.0, 0.0, 0.0 };
        y += m * _exp(-y + f256_s{ (double)exp2 } * detail::_f256::ln2) - 1.0;
        y += m * _exp(-y + f256_s{ (double)exp2 } * detail::_f256::ln2) - 1.0;
        return y;
    }
}

// exp
[[nodiscard]] BL_NO_INLINE constexpr f256_s ldexp(const f256_s& a, int e)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_ldexp(a, e));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s exp(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_exp(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s exp2(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_exp2(x));
}

// log
[[nodiscard]] BL_NO_INLINE constexpr f256_s log(const f256_s& a)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_log(a));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s log2(const f256_s& a)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_log(a) * detail::_f256::inv_ln2);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s log10(const f256_s& a)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_log(a) / detail::_f256::ln10);
}

// pow
[[nodiscard]] BL_NO_INLINE constexpr f256_s pow(const f256_s& x, const f256_s& y)
{
    if (iszero(y))
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s yi = trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && detail::_f256::f256_try_get_int64(yi, yi64))
        return detail::_f256::powi(x, yi64);

    if (x.x0 < 0.0 || (x.x0 == 0.0 && detail::_f256::signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = detail::_f256::_exp(detail::_f256::mul_inline(y, detail::_f256::_log(-x)));
        return detail::_f256::is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return detail::_f256::canonicalize_math_result(detail::_f256::_exp(detail::_f256::mul_inline(y, detail::_f256::_log(x))));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s pow(const f256_s& x, double y)
{
    if (y == 0.0)
        return f256_s{ 1.0 };

    if (isnan(x) || detail::_f256::isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (y == 1.0) return x;
    if (y == 2.0) return detail::_f256::canonicalize_math_result(x * x);
    if (y == -1.0) return detail::_f256::canonicalize_math_result(f256_s{ 1.0 } / x);
    if (y == 0.5) return detail::_f256::canonicalize_math_result(sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? detail::_f256::ceil_constexpr(y)
            : detail::_f256::floor_constexpr(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && detail::_f256::absd(yi) < 0x1p63)
        return detail::_f256::powi(x, static_cast<int64_t>(yi));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && detail::_f256::signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = detail::_f256::_exp(detail::_f256::mul_double_inline(detail::_f256::_log(-x), y));
        const bool y_is_odd =
            (detail::_f256::absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return detail::_f256::canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return detail::_f256::canonicalize_math_result(detail::_f256::_exp(detail::_f256::mul_double_inline(detail::_f256::_log(x), y)));
}


// trig
namespace detail::_f256
{
    BL_NO_INLINE constexpr bool _sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
    {
        const double ax = detail::_f256::fabs_constexpr(x.x0);
        if (!detail::_f256::isfinite(ax))
        {
            s_out = f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
            c_out = s_out;
            return false;
        }

        if (ax <= static_cast<double>(detail::_f256::pi_4))
        {
            detail::_f256::f256_sincos_kernel_pi4(x, s_out, c_out);
            return true;
        }

        long long n = 0;
        f256_s r{};
        if (!detail::_f256::f256_remainder_pi2(x, n, r))
            return false;

        f256_s sr{}, cr{};
        detail::_f256::f256_sincos_kernel_pi4(r, sr, cr);

        switch ((int)(n & 3LL))
        {
        case 0: s_out = sr;  c_out = cr;  break;
        case 1: s_out = cr;  c_out = -sr; break;
        case 2: s_out = -sr; c_out = -cr; break;
        default: s_out = -cr; c_out = sr;  break;
        }

        return true;
    }

    BL_NO_INLINE constexpr f256_s atan_core_unit(const f256_s& z)
    {
        using namespace detail::_f256;

        if (z.x0 <= 0x1p-9)
        {
            //const f256_s z2 = mul_inline(z, z);
            const f256_s z2 = z * z;
            f256_s p = f256_atan_tiny_coeffs[f256_atan_tiny_coeff_count - 1];
            for (int i = static_cast<int>(f256_atan_tiny_coeff_count) - 2; i >= 0; --i)
                p = f256_mul_add_horner_step(p, z2, f256_atan_tiny_coeffs[i]);

            //return mul_inline(z, p);
            return z * p;
        }

        f256_s v = f256_s{ detail::fp::atan_constexpr(static_cast<double>(z)) };

        for (int i = 0; i < 2; ++i)
        {
            f256_s sv{}, cv{};
            detail::_f256::f256_sincos_kernel_pi4(v, sv, cv);

            #if BL_F256_ENABLE_SIMD
            if (detail::_f256::f256_runtime_trig_simd_enabled())
            {
                f256_s zcv{}, zsv{};
                detail::_f256::f256_mul_pair_simd(z, cv, z, sv, zcv, zsv);

                const f256_s f = sv - zcv;
                const f256_s fp = cv + zsv;
                v = v - f / fp;

                //const f256_s f  = sub_inline(sv, zcv);
                //const f256_s fp = add_inline(cv, zsv);
                //v = sub_inline(v, div_inline(f, fp));
                continue;
            }
            #endif

            const f256_s f  = sv - z * cv;
            const f256_s fp = cv + z * sv;
            v = v - f / fp;

            //const f256_s f  = sub_inline(sv, mul_inline(z, cv));
            //const f256_s fp = add_inline(cv, mul_inline(z, sv));
            //v = sub_inline(v, div_inline(f, fp));
        }

        return v;
    }
    BL_NO_INLINE constexpr f256_s _atan(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))  return x;
        if (iszero(x)) return x;
        if (isinf(x))  return detail::_f256::signbit_constexpr(x.x0) ? -detail::_f256::pi_2 : detail::_f256::pi_2;

        const bool neg = x.x0 < 0.0;
        const f256_s ax = neg ? -x : x;

        if (ax > f256_s{ 1.0 })
        {
            const f256_s core = detail::_f256::atan_core_unit(recip(ax));
            const f256_s out = sub_inline(detail::_f256::pi_2, core);
            return neg ? -out : out;
        }

        const f256_s out = detail::_f256::atan_core_unit(ax);
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
            return (x.x0 < 0.0) ? -detail::_f256::pi_2 : detail::_f256::pi_2;

        if (ax <= f256_s{ 0.5 })
            return _atan(x / sqrt(f256_s{ 1.0 } - x * x));

        const f256_s t = sqrt((f256_s{ 1.0 } - ax) / (f256_s{ 1.0 } + ax));
        const f256_s a = detail::_f256::pi_2 - mul_double_inline(_atan(t), 2.0);
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
            return detail::_f256::pi;

        return detail::_f256::pi_2 - _asin(x);
    }
}

[[nodiscard]] BL_NO_INLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    bool ret = detail::_f256::_sincos(x, s_out, c_out);
    s_out = detail::_f256::canonicalize_math_result(s_out);
    c_out = detail::_f256::canonicalize_math_result(c_out);
    return ret;
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s sin(const f256_s& x)
{
    const double ax = detail::_f256::fabs_constexpr(x.x0);
    if (!detail::_f256::isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(detail::_f256::pi_4))
        return detail::_f256::f256_sin_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!detail::_f256::f256_remainder_pi2(x, n, r))
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
    case 0: return detail::_f256::f256_sin_kernel_pi4(r);
    case 1: return detail::_f256::f256_cos_kernel_pi4(r);
    case 2: return -detail::_f256::f256_sin_kernel_pi4(r);
    default: return -detail::_f256::f256_cos_kernel_pi4(r);
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s cos(const f256_s& x)
{
    const double ax = detail::_f256::fabs_constexpr(x.x0);
    if (!detail::_f256::isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(detail::_f256::pi_4))
        return detail::_f256::f256_cos_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!detail::_f256::f256_remainder_pi2(x, n, r))
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
    case 0: return detail::_f256::f256_cos_kernel_pi4(r);
    case 1: return -detail::_f256::f256_sin_kernel_pi4(r);
    case 2: return -detail::_f256::f256_cos_kernel_pi4(r);
    default: return detail::_f256::f256_sin_kernel_pi4(r);
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tan(const f256_s& x)
{
    f256_s s{}, c{};
    if (detail::_f256::_sincos(x, s, c))
        return detail::_f256::canonicalize_math_result(s / c);

    if (bl::use_constexpr_math())
    {
        return detail::_f256::canonicalize_math_result(f256_s{ detail::fp::tan_constexpr(static_cast<double>(x)) });
    }
    else
    {
        return detail::_f256::canonicalize_math_result(f256_s{ std::tan(static_cast<double>(x)) });
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atan(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_atan(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (iszero(x))
    {
        if (iszero(y))
            return f256_s{ std::numeric_limits<double>::quiet_NaN() };
        return ispositive(y) ? detail::_f256::pi_2 : -detail::_f256::pi_2;
    }

    if (iszero(y))
    {
        if (x.x0 < 0.0)
            return detail::_f256::signbit_constexpr(y.x0) ? -detail::_f256::pi : detail::_f256::pi;
        return y;
    }

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax == ay)
    {
        if (x.x0 < 0.0)
        {
            return detail::_f256::canonicalize_math_result(
                (y.x0 < 0.0) ? -detail::_f256::pi_3_4 : detail::_f256::pi_3_4);
        }

        return detail::_f256::canonicalize_math_result(
            (y.x0 < 0.0) ? -detail::_f256::pi_4 : detail::_f256::pi_4);
    }

    if (ax >= ay)
    {
        f256_s a = detail::_f256::_atan(y / x);

        if (x.x0 < 0.0)
            a += (y.x0 < 0.0) ? -detail::_f256::pi : detail::_f256::pi;
        return detail::_f256::canonicalize_math_result(a);
    }

    f256_s a = detail::_f256::_atan(x / y);
    return detail::_f256::canonicalize_math_result((y.x0 < 0.0) ? (-detail::_f256::pi_2 - a) : (detail::_f256::pi_2 - a));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s asin(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_asin(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s acos(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_acos(x));
}


[[nodiscard]] BL_NO_INLINE constexpr f256_s fabs(const f256_s& a) noexcept
{
    return abs(a);
}

[[nodiscard]] BL_NO_INLINE constexpr bool signbit(const f256_s& x) noexcept
{
    return detail::_f256::signbit_constexpr(x.x0)
        || (x.x0 == 0.0 && (detail::_f256::signbit_constexpr(x.x1)
        || (x.x1 == 0.0 && (detail::_f256::signbit_constexpr(x.x2)
        || (x.x2 == 0.0 && detail::_f256::signbit_constexpr(x.x3))))));
}
[[nodiscard]] BL_NO_INLINE constexpr int fpclassify(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<f256_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}
[[nodiscard]] BL_NO_INLINE constexpr bool isnormal(const f256_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}
[[nodiscard]] BL_NO_INLINE constexpr bool isunordered(const f256_s& a, const f256_s& b) noexcept
{
    return isnan(a) || isnan(b);
}
[[nodiscard]] BL_NO_INLINE constexpr bool isgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}
[[nodiscard]] BL_NO_INLINE constexpr bool isgreaterequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}
[[nodiscard]] BL_NO_INLINE constexpr bool isless(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}
[[nodiscard]] BL_NO_INLINE constexpr bool islessequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}
[[nodiscard]] BL_NO_INLINE constexpr bool islessgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

namespace detail::_f256
{
    BL_FORCE_INLINE constexpr f256_s round_half_away_zero(const f256_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f256_s y = -floor((-x) + f256_s{ 0.5 });
            if (iszero(y))
                return f256_s{ -0.0, 0.0, 0.0, 0.0 };
            return y;
        }

        return floor(x + f256_s{ 0.5 });
    }

    using detail::fp::nextafter_double_constexpr;

    struct signed_integer_conversion_ops
    {
        BL_FORCE_INLINE constexpr bool is_nan(const f256_s& x) const noexcept { return bl::isnan(x); }
        BL_FORCE_INLINE constexpr bool is_inf(const f256_s& x) const noexcept { return bl::isinf(x); }
        BL_FORCE_INLINE constexpr f256_s from_int64(std::int64_t value) const noexcept { return to_f256(value); }
        BL_FORCE_INLINE constexpr bool try_get_int64(const f256_s& x, std::int64_t& out) const noexcept
        {
            return detail::_f256::f256_try_get_int64(x, out);
        }
    };

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f256_s& x) noexcept
    {
        return detail::fp::to_signed_integer_or_zero<SignedInt>(x, signed_integer_conversion_ops{});
    }

    BL_FORCE_INLINE constexpr f256_s nearest_integer_ties_even(const f256_s& q) noexcept
    {
        f256_s n = trunc(q);
        const f256_s frac = q - n;
        const f256_s half{ 0.5 };
        const f256_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n += signbit(frac) ? -one : one;
        }
        else if (abs(frac) == half)
        {
            if (detail::_f256::is_odd_integer(n))
                n += signbit(frac) ? -one : one;
        }

        return n;
    }

    BL_NO_INLINE constexpr f256_s _expm1(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (x == f256_s{ 0.0 })
            return x;
        if (isinf(x))
            return signbit(x)
            ? f256_s{ -1.0, 0.0, 0.0, 0.0 }
        : std::numeric_limits<f256_s>::infinity();

        const f256_s ax = abs(x);
        if (ax <= f256_s{ 0.5 })
        {
            f256_s term = x;
            f256_s sum = x;

            for (int n = 2; n <= 256; ++n)
            {
                term = detail::_f256::div_double_inline(detail::_f256::mul_inline(term, x), static_cast<double>(n));
                sum = detail::_f256::add_inline(sum, term);

                const f256_s asum = abs(sum);
                const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
                if (abs(term) <= f256_s::eps() * scale)
                    break;
            }

            return sum;
        }

        return detail::_f256::sub_inline(detail::_f256::_exp(x), f256_s{ 1.0 });
    }

    [[nodiscard]] BL_NO_INLINE constexpr f256_s atanh_small_series_constexpr(const f256_s& x)
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
    [[nodiscard]] BL_NO_INLINE inline f256_s atanh_small_series_runtime(const f256_s& x)
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

}

[[nodiscard]] BL_NO_INLINE constexpr f256_s expm1(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_expm1(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s log1p(const f256_s& x)
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
        return detail::_f256::canonicalize_math_result(detail::_f256::f256_log1p_series_reduced(x));

    const f256_s u = f256_s{ 1.0 } + x;
    if ((u - f256_s{ 1.0 }) == x)
        return detail::_f256::canonicalize_math_result(log(u));

    if (x > f256_s{ 0.0 } && x <= f256_s{ 1.0 })
    {
        const f256_s t = x / (f256_s{ 1.0 } + sqrt(f256_s{ 1.0 } + x));
        return detail::_f256::canonicalize_math_result(detail::_f256::f256_log1p_series_reduced(t) * f256_s{ 2.0 });
    }

    if (x > f256_s{ 0.0 })
        return detail::_f256::canonicalize_math_result(log(u));

    const f256_s y = u - f256_s{ 1.0 };
    if (iszero(y))
        return x;

    return detail::_f256::canonicalize_math_result(log(u) * (x / y));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s sinh(const f256_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (ax <= f256_s{ 0.5 })
    {
        const f256_s x2 = detail::_f256::mul_inline(x, x);
        f256_s term = x;
        f256_s sum = x;

        for (int n = 1; n <= 256; ++n)
        {
            term = detail::_f256::div_double_inline(
                detail::_f256::mul_inline(term, x2),
                static_cast<double>((2 * n) * (2 * n + 1)));
            sum = detail::_f256::add_inline(sum, term);

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }

        return detail::_f256::canonicalize_math_result(sum);
    }

    const f256_s ex = detail::_f256::_exp(ax);
    const f256_s inv_ex = recip(ex);
    f256_s out = detail::_f256::mul_double_inline(detail::_f256::sub_inline(ex, inv_ex), 0.5);
    if (signbit(x))
        out = -out;
    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s cosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s ax = abs(x);
    const f256_s ex = detail::_f256::_exp(ax);
    const f256_s inv_ex = recip(ex);
    return detail::_f256::canonicalize_math_result(
        detail::_f256::mul_double_inline(detail::_f256::add_inline(ex, inv_ex), 0.5));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tanh(const f256_s& x)
{
    using namespace detail::_f256;
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s ax = abs(x);
    if (ax > f256_s{ 20.0 })
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s em1 = detail::_f256::_expm1(detail::_f256::mul_double_inline(ax, 2.0));

    //f256_s out = em1 / detail::_f256::add_scalar_precise(em1, 2.0);
    f256_s out = div_inline(em1, detail::_f256::add_scalar_precise(em1, 2.0));

    if (signbit(x))
        out = -out;
    return detail::_f256::canonicalize_math_result(out);
}

/*[[nodiscard]] BL_NO_INLINE constexpr f256_s asinh(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::asinh_newton_impl(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s acosh(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::acosh_newton_impl(x));
}*/
[[nodiscard]] BL_NO_INLINE constexpr f256_s asinh(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f256_s ax = abs(x);
    f256_s out{};
    if (ax > f256_s{ 0x1p500 })
        out = log(ax) + detail::_f256::ln2;
    else
        out = log(ax + sqrt(ax * ax + f256_s{ 1.0 }));

    if (signbit(x))
        out = -out;
    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s acosh(const f256_s& x)
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
        out = log(x) + detail::_f256::ln2;
    else
        out = log(x + sqrt((x - f256_s{ 1.0 }) * (x + f256_s{ 1.0 })));

    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atanh(const f256_s& x)
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
            return detail::_f256::canonicalize_math_result(detail::_f256::atanh_small_series_constexpr(x));

        return detail::_f256::canonicalize_math_result(detail::_f256::atanh_small_series_runtime(x));
    }

    const f256_s out = log((f256_s{ 1.0 } + x) / (f256_s{ 1.0 } - x)) * f256_s { 0.5 };
    return detail::_f256::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s cbrt(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s y{};
    if (bl::use_constexpr_math())
    {
        y = exp(log(ax) / f256_s{ 3.0 });
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
            y = ldexp(y, exp2 / 3);
    }

    y = (y + y + ax / (y * y)) / f256_s{ 3.0 };
    y = (y + y + ax / (y * y)) / f256_s{ 3.0 };

    if (bl::use_constexpr_math())
        y = (y + y + ax / (y * y)) / f256_s{ 3.0 };

    if (neg)
        y = -y;

    return detail::_f256::canonicalize_math_result(y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s hypot(const f256_s& x, const f256_s& y)
{
    using namespace detail::_f256;

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
        return detail::_f256::canonicalize_math_result(ax);

    int ex = 0;
    int ey = 0;
    if (bl::use_constexpr_math())
    {
        ex = detail::_f256::frexp_exponent_constexpr(ax.x0);
        ey = detail::_f256::frexp_exponent_constexpr(ay.x0);
    }
    else
    {
        (void)std::frexp(ax.x0, &ex);
        (void)std::frexp(ay.x0, &ey);
    }

    if ((ex - ey) > 110)
        return detail::_f256::canonicalize_math_result(ax);

    const f256_s r = div_inline(ay, ax);
    return detail::_f256::canonicalize_math_result(mul_inline(ax, sqrt(f256_s{ 1.0 }) + mul_inline(r, r)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s rint(const f256_s& x)
{
    return nearbyint(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = detail::_f256::nearest_integer_ties_even(x / y);
    f256_s r = x - n * y;

    if (quo)
        *quo = detail::_f256::low_quotient_bits(n);

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return detail::_f256::canonicalize_math_result(r);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = detail::_f256::nearest_integer_ties_even(x / y);
    f256_s r = x - n * y;

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return detail::_f256::canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    return detail::_f256::canonicalize_math_result(x * y + z);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmin(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmax(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fdim(const f256_s& x, const f256_s& y)
{
    return (x > y) ? detail::_f256::canonicalize_math_result(x - y) : f256_s{ 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s copysign(const f256_s& x, const f256_s& y)
{
    return signbit(x) == signbit(y) ? x : -x;
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept
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

    f256_s m = ldexp(x, -e);
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
[[nodiscard]] BL_NO_INLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept
{
    const f256_s i = trunc(x);
    if (iptr)
        *iptr = i;

    f256_s frac = x - i;
    if (iszero(frac))
        frac = f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
    return frac;
}
[[nodiscard]] BL_NO_INLINE constexpr int ilogb(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_ILOGBNAN;
    if (iszero(x)) return FP_ILOGB0;
    if (isinf(x))  return std::numeric_limits<int>::max();

    int e = 0;
    (void)frexp(abs(x), &e);
    return e - 1;
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s logb(const f256_s& x) noexcept
{
    if (isnan(x))  return x;
    if (iszero(x)) return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (isinf(x))  return std::numeric_limits<f256_s>::infinity();

    return f256_s{ static_cast<double>(ilogb(x)), 0.0, 0.0, 0.0 };
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept
{
    return ldexp(x, e);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept
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

    return f256_s{
        from.x0,
        from.x1,
        from.x2,
        detail::_f256::nextafter_double_constexpr(from.x3, toward)
    };
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept
{
    return nextafter(from, f256_s{ static_cast<double>(to), 0.0, 0.0, 0.0 });
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    return nextafter(from, to);
}

// special functions

namespace detail::_f256
{
    inline constexpr std::size_t f256_erf_cheb_coeff_count = 52;

    inline constexpr f256_s f256_erf_cheb_0_1[] = {
        {     0x1.e0e7739ddec39p-2,    0x1.3a4ee10fce73ep-56,   0x1.ebd9f663f231fp-110,  -0x1.4132792c8f663p-164 },
        {     0x1.b40b53a48a550p-2,    0x1.61466f0de5c3ep-56,  -0x1.0d13b8c31c903p-111,  -0x1.dbe30fcee3951p-165 },
        {    -0x1.95ef63f0a8ae4p-5,    0x1.4afb1bf3375b4p-61,   0x1.4b4a41d443caap-115,   0x1.b71d8684a92ebp-170 },
        {    -0x1.2672f067d94c9p-8,   -0x1.b0c79ed69d826p-62,   0x1.888f7ad65e642p-117,  -0x1.4859d949e5ba6p-171 },
        {    0x1.52acb22c7c2dfp-10,   -0x1.7ba857512c10ep-64,   0x1.6d8ddb1817e40p-121,   0x1.06081d2af6d7fp-175 },
        {    0x1.341851cc64fd9p-16,   -0x1.64e4d7a520fc0p-70,  -0x1.92ea557866fb3p-125,  -0x1.287773a1b02c5p-181 },
        {   -0x1.72f6a74668f7ep-16,    0x1.52d222adc2891p-70,  -0x1.a9b1ecd235440p-126,  -0x1.963692b13e67bp-182 },
        {    0x1.18f6f4cefae30p-21,   -0x1.ffb9b3e3e6634p-75,  -0x1.b9e20cdb201a6p-137,   0x1.5c3dfaddcd64cp-192 },
        {    0x1.2ab2da3d52cd5p-22,    0x1.ee1fdc14ae669p-80,  -0x1.40aae46d25039p-136,  -0x1.34aaad1430777p-190 },
        {   -0x1.e503f8b1f83fdp-27,    0x1.b8eb6770c66c5p-83,   0x1.38731c6ff5ce5p-138,   0x1.fd371065fe80bp-192 },
        {   -0x1.769e7fc58fb7dp-29,   -0x1.80f1a0d9feebfp-83,  -0x1.cc8942db0ad27p-140,  -0x1.bd59adddc7cb3p-194 },
        {    0x1.c5c14aa8b09e2p-33,   -0x1.5ce4de86ca77ep-89,  -0x1.eb62d716b8742p-143,   0x1.8ef06e22556a0p-197 },
        {    0x1.78f46d533a139p-36,    0x1.b2c073468c41fp-90,  -0x1.493fe4cd2a9adp-145,  -0x1.9d0d7b067ad58p-204 },
        {   -0x1.39d4b4dfd3f8ap-39,    0x1.6bc02cd97f21cp-93,  -0x1.8fc543fe32ab7p-147,  -0x1.1146030565976p-201 },
        {   -0x1.32ced20e8d32ep-43,    0x1.a6f7634d03bdfp-97,  -0x1.fad17c453fc6dp-153,   0x1.b4e1163c34be8p-207 },
        {    0x1.5f8a4a69ad905p-46,  -0x1.926d8dc98a486p-101,   0x1.fb821fb2325edp-158,   0x1.ae03893d520fbp-213 },
        {    0x1.8b3cb61c1ba41p-51,   0x1.ca9fa171b6f73p-106,   0x1.0482c511a3251p-162,  -0x1.805627b3d81dfp-218 },
        {   -0x1.4d3593caa0defp-53,  -0x1.4dd3e3fe4f996p-107,  -0x1.082cadb4337f0p-161,  -0x1.6ced334267182p-215 },
        {   -0x1.6aefbb5ad4104p-59,   0x1.91c76ee65a514p-113,   0x1.7da58505fb416p-167,  -0x1.bf12cd531c21cp-222 },
        {    0x1.125e8e95591aap-60,  -0x1.fc915dcc0ec16p-116,   0x1.7c69b8f8ed2a1p-172,   0x1.9e69db66c181dp-228 },
        {    0x1.b25cb0eff6240p-69,  -0x1.7bd21124aa731p-124,  -0x1.df68ca6705063p-179,  -0x1.cbe468e4e1f54p-233 },
        {   -0x1.8f7ce46504a4bp-68,  -0x1.75ccf4213d82ap-122,   0x1.ac2adaa38cc10p-177,   0x1.3c3810f1d87a7p-233 },
        {    0x1.b159b64e36661p-75,   0x1.e1a47ecc7a328p-131,  -0x1.dc01028000648p-185,   0x1.ab648ecd732adp-240 },
        {    0x1.04610da36963cp-75,  -0x1.5ebf38d263c1fp-131,  -0x1.659da6445055dp-187,   0x1.7398e8acb227cp-242 },
        {   -0x1.386823cd195b4p-81,  -0x1.841a7ddff471ap-135,   0x1.2687882d944aap-190,  -0x1.5bd35c40b13afp-245 },
        {   -0x1.32af712a7fcbcp-83,  -0x1.33e4e222b4f4cp-138,  -0x1.bc6697ca8c897p-193,   0x1.78a0ecbadccd9p-247 },
        {    0x1.17356febe6a9dp-88,  -0x1.92e807826cb31p-145,   0x1.685c53c1f2ceap-201,   0x1.99426922241e7p-255 },
        {    0x1.4894f1715a815p-91,  -0x1.ee86ae3445f6bp-145,  -0x1.675b5b1a6feabp-204,  -0x1.a101498ee0be9p-261 },
        {   -0x1.9161370246becp-96,   0x1.28010616b9efep-152,  -0x1.81cf003132809p-209,   0x1.00c636655ab79p-263 },
        {   -0x1.41b514f4a98fep-99,   0x1.436019911dee9p-154,   0x1.bac858cbf6aeap-208,   0x1.445841bcb581ep-266 },
        {   0x1.f350e8b772adep-104,  -0x1.88f9c5f1eb139p-161,   0x1.62ec0a6004c4cp-216,  -0x1.1fb50c111a549p-270 },
        {   0x1.20952427a781fp-107,   0x1.2530553b6a9d5p-162,   0x1.69a14acbcf7aap-216,   0x1.9d95ff54bb991p-271 },
        {  -0x1.15b88d8f79155p-111,  -0x1.47747724260e4p-166,  -0x1.596a6b13ab96bp-223,  -0x1.0b1e6dca2f90bp-278 },
        {  -0x1.da7a6ee2310f0p-116,   0x1.4b0ab47e0e729p-174,  -0x1.a867a415e0ac6p-229,  -0x1.211dfacb263e1p-283 },
        {   0x1.1964286c4c431p-119,   0x1.f8bc0fa5f9b47p-173,   0x1.27fd4c4e5a985p-228,   0x1.59daf2de65ae7p-282 },
        {   0x1.646bb3b838267p-124,  -0x1.d504930ea0b32p-178,   0x1.e50aa3fa0fa80p-233,  -0x1.b97b210394528p-289 },
        {  -0x1.06c76ccac0f04p-127,  -0x1.43a31b89b4413p-181,   0x1.44e4a08d9ae86p-235,  -0x1.da340ae6fa754p-292 },
        {  -0x1.e5462c037d190p-133,   0x1.57e28e4403657p-189,  -0x1.35f0f4a3463b5p-243,  -0x1.cb83f41184934p-298 },
        {   0x1.c821836e72befp-136,  -0x1.668ff29d1d830p-191,  -0x1.39b1f6aadfe04p-246,   0x1.c43f8658eb6e9p-304 },
        {   0x1.25c87d4bf090cp-141,  -0x1.c2a56911264efp-198,   0x1.7b8cd77eba2b9p-252,   0x1.76126fb1ea422p-306 },
        {  -0x1.723498a144e9cp-144,  -0x1.5d2ccb10541c8p-198,  -0x1.1200b08b095f3p-256,  -0x1.6d36613ef5b42p-310 },
        {  -0x1.2e2d2c629c6fap-150,   0x1.f699661475367p-204,  -0x1.9d9081aa9c678p-259,   0x1.8cb997513ca69p-316 },
        {   0x1.1a58375d0659ap-152,  -0x1.b6e77489f8e74p-206,   0x1.09c50fed61f03p-260,   0x1.06aee6eac1ba2p-315 },
        {   0x1.c89bbf3925b19p-160,  -0x1.664fba475f7e1p-216,  -0x1.44a575e353421p-272,   0x1.8bb1a496a6ba0p-326 },
        {  -0x1.9644409585efbp-161,   0x1.88c61900f0a4bp-215,   0x1.796f5e09cb051p-269,  -0x1.af310ac3aad79p-324 },
        {  -0x1.ba632b6923566p-172,  -0x1.fecde9c4bf934p-238,  -0x1.374a38b9ef099p-292,  -0x1.026f4abc5a38bp-346 },
        {   0x1.1499355d41176p-169,   0x1.54a032b06d5f2p-223,   0x1.fa75509f541cfp-277,   0x1.158ebd26c393ap-331 },
        {  -0x1.54b8609dc0ce3p-177,   0x1.f8d821d2e866dp-235,  -0x1.fd1a85f328e91p-289,  -0x1.3dbf5a746937fp-344 },
        {  -0x1.6557b30922e98p-178,   0x1.bbf1af6b2adabp-232,  -0x1.8996a5856e2aep-286,   0x1.94810c06197a8p-340 },
        {   0x1.c3c02eb70e795p-185,   0x1.c38e14d305cfcp-239,   0x1.a09387e89b43ep-293,   0x1.94823058e7bc1p-348 },
        {   0x1.b6f29b53de702p-187,   0x1.69c2b2554e312p-242,   0x1.5e1b01637ae8dp-296,   0x1.1c58b694566b5p-350 },
        {  -0x1.a1075d187b605p-193,   0x1.05d2ede435676p-247,   0x1.280b78188a063p-302,   0x1.240951b58a95fp-356 },
    };
    inline constexpr f256_s f256_erf_cheb_1_2[] = {
        {     0x1.e2bd87e6b1fb4p-1,   -0x1.94ab141b6f38cp-56,   0x1.ec081debfe542p-110,   0x1.e4cccfad26832p-164 },
        {     0x1.27958e4629c8fp-4,   -0x1.bcef94378a6e8p-60,  -0x1.b546158dbeedep-118,  -0x1.93e331d562d28p-176 },
        {    -0x1.81ef9beca3846p-6,    0x1.b4450017b890ap-61,   0x1.f01f9ec5e656bp-115,  -0x1.9f3d4607981b7p-169 },
        {     0x1.11ec0d93eaaa2p-8,    0x1.aba1e79f04554p-66,  -0x1.5d1acf04bac83p-127,   0x1.2d8ca3a95bf8dp-181 },
        {   -0x1.341e56f165d5ep-12,    0x1.bc81ab8936810p-67,  -0x1.fd164467d5552p-122,  -0x1.445ab1417fe62p-177 },
        {   -0x1.11caef6ccc294p-15,    0x1.679785875c37fp-69,   0x1.0f202deb6779dp-126,  -0x1.c60aaae0580e1p-180 },
        {    0x1.2915116c85aecp-17,    0x1.ef7a640c00f59p-72,  -0x1.bcf0f15b5a596p-130,  -0x1.b4dd0d1f07670p-184 },
        {   -0x1.f69e94c30021bp-22,   -0x1.0b1914732ec47p-80,  -0x1.6bc75b2fceec9p-136,  -0x1.8bb31babc5760p-190 },
        {   -0x1.3ad36796a5e59p-24,   -0x1.32d26ef1f688bp-78,   0x1.7e68738db48bcp-133,  -0x1.b0dd78c25235fp-187 },
        {    0x1.93cf442d68490p-27,    0x1.48409c11c5e63p-83,  -0x1.735fb3c25bd76p-140,   0x1.069d94612fb8cp-194 },
        {   -0x1.4af16f8c3b21fp-34,    0x1.e476d9225d31cp-89,  -0x1.8d0ac7e6454ddp-145,   0x1.da8c2eb59a3a5p-199 },
        {   -0x1.f69f041741747p-34,    0x1.6e7b70a63d2c7p-89,  -0x1.7cee826077531p-143,   0x1.0252e92f6f2f9p-201 },
        {    0x1.151018c6e107bp-37,    0x1.ddd226b82c9bbp-91,   0x1.e5e5cc95cecdcp-145,   0x1.cef12099beb47p-200 },
        {    0x1.33ccd7063e94dp-41,   -0x1.a4b1535d338a9p-95,   0x1.7bc05149dd478p-151,   0x1.b64d7b33d9b9bp-205 },
        {   -0x1.a78200513fe64p-44,    0x1.b106e770bb435p-98,   0x1.752f41b709ad9p-153,  -0x1.2a959b5984ea2p-208 },
        {    0x1.2207b8e2b8989p-51,  -0x1.0531378be6d1ep-105,   0x1.f2994bff2354ep-161,  -0x1.e1b49924d9019p-216 },
        {    0x1.7c68a583bbbbdp-51,   0x1.bf7dc3e8d2198p-111,   0x1.05174a357d73cp-168,   0x1.bf5494f0e48b6p-222 },
        {   -0x1.2d6c2fd7cf7c7p-55,   0x1.900fe150afc9cp-111,   0x1.9853d541345f5p-165,   0x1.99726ef74f533p-223 },
        {   -0x1.b1401cab07fc2p-59,   0x1.c341caf411d51p-117,   0x1.9cef808d865d8p-172,   0x1.32a5b34281d0ep-228 },
        {    0x1.7883e84126acfp-62,  -0x1.5b183cd1ccf32p-116,   0x1.fb9365ee10a00p-172,  -0x1.cb609aecdbbbcp-226 },
        {    0x1.93eb4cbb924a0p-68,   0x1.d9fbd28135da6p-123,   0x1.7da7c79421adcp-177,   0x1.d1697a89e0f2cp-231 },
        {   -0x1.2cfc3dc43dc2bp-69,  -0x1.5576b8b252e79p-123,  -0x1.ffb0eeb6203f7p-179,   0x1.61a64978f57bdp-233 },
        {    0x1.7badb451d1ee0p-75,  -0x1.b5a654d980ec4p-130,   0x1.abf61b1616ba0p-185,  -0x1.a8caa2472a28dp-240 },
        {    0x1.5d6dc874c30d2p-77,   0x1.87386cfc353fap-132,   0x1.469cf7068937ap-188,  -0x1.ad165db1d14c7p-242 },
        {   -0x1.28309bbad20ebp-81,   0x1.b3861e0b301a6p-135,  -0x1.9db2e8a415fe6p-190,  -0x1.1a3828c0e3a2ep-244 },
        {   -0x1.1d7117ae3b581p-85,  -0x1.1d3da4b156ea4p-141,   0x1.2b07c98fc0cf4p-196,   0x1.b7597aed09d58p-256 },
        {    0x1.e1b606c21acc5p-89,   0x1.0ad07a22b8c34p-144,  -0x1.74886fb34561ep-202,  -0x1.007daff569101p-256 },
        {    0x1.b8a297f4ee1d5p-95,   0x1.2e1d3fcfbe3bcp-149,  -0x1.45f2324f75d01p-205,   0x1.2757b39c85599p-264 },
        {   -0x1.20797a3a6c93bp-96,   0x1.19540fc7c8158p-150,  -0x1.87876d56e4e0cp-204,   0x1.0618dbfd72d95p-259 },
        {   0x1.e972f3aa01dcap-103,   0x1.e6b815be1cd18p-157,  -0x1.24b5d07791598p-213,   0x1.e7ceb07b8df87p-267 },
        {   0x1.104898c0d97ccp-104,   0x1.33810b508ed6dp-159,   0x1.f5dfdfa48ac37p-214,  -0x1.118625827aea8p-269 },
        {  -0x1.4d696309b9a67p-109,  -0x1.4296035a8f10ep-164,  -0x1.0967297b68768p-219,  -0x1.d00c14a3359c0p-273 },
        {  -0x1.9132754ed8153p-113,  -0x1.ffb7f01bc107bp-170,  -0x1.94faff4bc28e6p-226,   0x1.4d94c06221236p-280 },
        {   0x1.cb4554289eccbp-117,  -0x1.edb866e8c0ff4p-171,   0x1.19ecb9db439aap-226,  -0x1.38c0d88cd7f2fp-280 },
        {   0x1.96b6e2b92e7a3p-122,  -0x1.9be6915b062f6p-178,  -0x1.3926cfc8d0da0p-232,   0x1.ff31056c4ee39p-287 },
        {  -0x1.dd31f9908024dp-125,  -0x1.ead442e7e20d8p-179,   0x1.366d8e0c01319p-233,   0x1.f51406643361cp-288 },
        {  -0x1.0007996bf5853p-133,  -0x1.faa0e7087cf58p-187,   0x1.cc625e03382afp-241,  -0x1.647d3db86e94cp-295 },
        {   0x1.9635e64e7a18cp-133,   0x1.6da1af2326550p-190,  -0x1.0ebeb5c2b3d96p-244,  -0x1.a7c481aafc2e7p-301 },
        {  -0x1.cdffc7dc4ff3ap-139,   0x1.f480035899f9ap-193,  -0x1.a9d71bcb5e64fp-247,  -0x1.2aac9f1a7fa90p-301 },
        {  -0x1.20bab525ac830p-141,  -0x1.9d193b0ff85e0p-195,  -0x1.bd97b9eb35f0bp-249,   0x1.c1cbab3c75890p-303 },
        {   0x1.61a1d49ecace4p-146,  -0x1.06b9805f350b9p-201,   0x1.1d6c116beabfbp-255,  -0x1.ffe984d47e0d9p-309 },
        {   0x1.4f77cc2940248p-150,   0x1.548627f166e3fp-205,  -0x1.68e72a1ee9b06p-259,  -0x1.c7307c0cc5097p-314 },
        {  -0x1.66d0e07826994p-154,   0x1.5c452eeea8e58p-208,  -0x1.71419ad8773bcp-262,  -0x1.7418903ffa14bp-316 },
        {  -0x1.1e76adc83a915p-159,  -0x1.54930ace2a06ap-213,  -0x1.8e7bbc8a64a4fp-269,  -0x1.f3dc20125d556p-323 },
        {   0x1.25eef91b5c4c7p-162,   0x1.429eb292044b9p-216,   0x1.6e6e2c205ec3cp-271,   0x1.bc4d28e152a32p-329 },
        {   0x1.4f0c84c368158p-170,   0x1.acb8df63ffa7cp-224,  -0x1.7c39d074f7a53p-278,   0x1.1f1007f6714f6p-333 },
        {  -0x1.9aa8a41875d65p-171,  -0x1.d46c081ac7c21p-225,   0x1.41cad059cbb91p-281,  -0x1.7298aa2c5b800p-337 },
        {   0x1.34c4aebbd1785p-177,   0x1.90139f667d387p-231,  -0x1.08e4ddfdf2ae3p-285,  -0x1.4081cc74be0f9p-339 },
        {   0x1.f106e5df28ea0p-180,  -0x1.709d634e0b541p-234,   0x1.d2c5e8d87eac5p-289,  -0x1.c02a55678068bp-344 },
        {  -0x1.b954769a95addp-185,   0x1.3c9494f7553abp-239,  -0x1.209dbd3bcc509p-295,  -0x1.a35de1f7ed100p-351 },
        {  -0x1.026f2330fa993p-188,  -0x1.4e0e1ab6411edp-243,   0x1.b1c87802de012p-297,  -0x1.b96ea1de23ac6p-352 },
        {   0x1.8a4218c17c55dp-193,  -0x1.5b7ab0a08eb2ep-249,  -0x1.e84615fcdfdabp-303,   0x1.da09c854de039p-357 },
    };
    inline constexpr f256_s f256_erf_cheb_2_3[] = {
        {     0x1.ff553286b4be4p-1,   -0x1.7128d27da60e3p-56,  -0x1.d6a130418cda9p-110,  -0x1.247898342147cp-164 },
        {     0x1.054b740da9f1ep-9,   -0x1.8f149fe75eb3fp-66,  -0x1.4a17f0f2aef59p-122,   0x1.bb5b8329ab90fp-176 },
        {   -0x1.fca7e70e4f64dp-11,   -0x1.464b696378824p-66,   0x1.df2ed2450d6aep-120,  -0x1.da9b5e72f5f80p-174 },
        {    0x1.518c57ec05f7bp-12,   -0x1.8254621fe7cc8p-66,  -0x1.a5aec1e7d1702p-120,  -0x1.4e0c31e1313bbp-175 },
        {   -0x1.3beb5ea9a13a3p-14,    0x1.d13ad35cb3702p-71,   0x1.449e01524a689p-126,  -0x1.7a9eb6f8d2278p-180 },
        {    0x1.9e2b2e9c4437dp-17,    0x1.4b571afee48dbp-72,  -0x1.eb7c50ddf72a3p-127,   0x1.129ec5f555191p-183 },
        {   -0x1.5c00e5ab3c9c1p-20,    0x1.6ebd85fc8bd49p-74,  -0x1.69462096eaf5ap-128,  -0x1.b9f208d7b2572p-182 },
        {    0x1.a69ee4908615bp-25,   -0x1.515d20ab0e449p-79,   0x1.8ef7be5325119p-134,   0x1.5d32e5051cc8ap-188 },
        {    0x1.4130f9cd8475dp-27,   -0x1.c320a565d4683p-84,   0x1.8a15e6d4d3847p-140,  -0x1.d2ccfac1da6f0p-195 },
        {   -0x1.0150b4e1fda41p-29,   -0x1.e8e5ef39339c2p-85,  -0x1.f49983b10f0a9p-140,   0x1.4a6e7045b5428p-195 },
        {    0x1.1e97ebed4506cp-33,    0x1.e69fd343266afp-87,  -0x1.c10b78100e9a5p-145,   0x1.b533192d57227p-202 },
        {    0x1.238abb8c6e614p-38,   -0x1.b4d704fd3ad3ap-92,  -0x1.bc36f4ca72975p-146,   0x1.121943b91d9ddp-201 },
        {   -0x1.d0756e3d26ab5p-40,   -0x1.eb91764c9456bp-94,   0x1.21565b38e3821p-148,   0x1.562951e959f17p-202 },
        {    0x1.132b546b338edp-43,   -0x1.97c0286e53edap-98,   0x1.50875187114d7p-156,   0x1.43af7d385ab98p-211 },
        {    0x1.744b729dee8c0p-49,  -0x1.d1e15438c74dfp-105,  -0x1.4468366054016p-159,  -0x1.ac0799551cd76p-213 },
        {   -0x1.4c811f9a4c378p-50,  -0x1.120dea90e9f84p-105,  -0x1.1e35189ae1b5ap-159,  -0x1.9efaf53765d1bp-215 },
        {    0x1.495e821f36660p-54,   0x1.b2193523fd017p-108,   0x1.237168379834cp-162,  -0x1.28b676f1595ccp-216 },
        {    0x1.8736345ccb15bp-59,   0x1.f9b35fe0123b3p-115,   0x1.9e95789f2cbcap-169,   0x1.096e53179feb1p-225 },
        {   -0x1.7ed604ad93ee4p-61,   0x1.08a01279e14fap-118,  -0x1.f42e2363b2423p-172,   0x1.ec61b2e3ed796p-227 },
        {    0x1.f1279334caabdp-66,   0x1.156e4eae65de8p-125,   0x1.45018d9fbf900p-180,   0x1.fe019f1a89952p-234 },
        {    0x1.48e4e484104e2p-69,   0x1.dba4f2f5de2e1p-124,   0x1.885ae82126e9bp-178,   0x1.cf30060cc25c4p-233 },
        {   -0x1.500b839965f6ap-72,   0x1.081684944a94cp-126,  -0x1.429f60c343e8bp-180,   0x1.be72e7938ef8bp-237 },
        {    0x1.3be63c36eac12p-78,   0x1.9f04c3286be56p-132,  -0x1.e237b1c9b1f6ap-189,   0x1.332ba667108b9p-243 },
        {    0x1.7806ed884e30fp-80,  -0x1.4b16a9261e495p-142,   0x1.9348ea962a5c3p-196,   0x1.dc92b60e0630bp-250 },
        {   -0x1.9e4d24b12abb8p-84,  -0x1.4026c15d48922p-138,   0x1.534f4a601b2e0p-193,   0x1.bd28bd2bc6e27p-249 },
        {   -0x1.ffc21e5a74bebp-90,  -0x1.7f12ffd2d2c2ap-147,  -0x1.f459ccd1beb1ap-203,   0x1.781c72ab8976cp-257 },
        {    0x1.256a52b8563cep-91,   0x1.8e04549183dbap-145,   0x1.5a6e6c3a3a13fp-199,  -0x1.c5e2b4f6297f5p-254 },
        {   -0x1.220acc94fa318p-96,   0x1.468e4d1381ed4p-151,   0x1.6531ff64369eap-205,   0x1.940bc25f7cb43p-259 },
        {  -0x1.b4cf417bd2222p-100,  -0x1.99e5cd64ee95bp-156,  -0x1.d9a9fde8d0318p-211,  -0x1.e4d336894cb45p-267 },
        {   0x1.30e45cdf2a2fap-103,   0x1.e0c784e62d6b5p-157,  -0x1.7566451fd42d8p-214,   0x1.b5c9c8c46a31dp-268 },
        {   0x1.4d77e72743eb2p-111,  -0x1.79fed23b3fec0p-166,   0x1.c2a877f7b2ba6p-221,   0x1.7cda2bc32b8e4p-275 },
        {  -0x1.3d25b25bb3443p-111,   0x1.60188734b92c3p-165,  -0x1.cca29bb1dd6d7p-220,  -0x1.efd48169823d5p-278 },
        {   0x1.64ec2ad6e04bep-116,  -0x1.48f8b6eb0e3d9p-174,  -0x1.5b83dbcca89c3p-229,   0x1.bb2412b64f691p-283 },
        {   0x1.7a26a16af2205p-120,   0x1.e29f686a0f521p-174,  -0x1.31adf83d4d596p-233,   0x1.765dd505a6794p-296 },
        {  -0x1.120e5f57528bap-123,  -0x1.8ea4b4003e4a7p-179,   0x1.800160831a719p-235,   0x1.99ed91b354d59p-289 },
        {  -0x1.547a7b371d3ffp-132,   0x1.bd4fa447197afp-188,   0x1.9d3631f452b24p-242,   0x1.4f422fc13509cp-296 },
        {   0x1.e47c22ce0c79ap-132,   0x1.bcf6f7814a243p-190,   0x1.630a92b35977cp-244,   0x1.aceb0e0db5b88p-298 },
        {  -0x1.e9034cef66214p-137,   0x1.073704ef9a132p-191,   0x1.7a9a89c1eb6cfp-245,   0x1.3853162eb23ffp-300 },
        {  -0x1.0b8ad89d731c5p-140,   0x1.18b9df6e3c5a4p-194,   0x1.b2283eba970a8p-248,   0x1.15d23336026b9p-303 },
        {   0x1.4c915982a9767p-144,   0x1.431415b3f9aaap-198,  -0x1.04db9aa3800b1p-255,   0x1.a1bfb7c706c3cp-310 },
        {   0x1.4dc8e3eb0dd0dp-151,  -0x1.8353d09274b4fp-208,   0x1.e5c963b5e578ap-262,  -0x1.fb58eea09a71ap-316 },
        {  -0x1.113266a01b504p-152,  -0x1.6d7a376d5a450p-209,   0x1.403e1ac566dbdp-263,  -0x1.8ddd5587ea2a8p-319 },
        {   0x1.8d635cd0819dfp-158,  -0x1.ebbeccb56780fp-212,  -0x1.ba03c6cca2224p-266,   0x1.b8cdb80db27e2p-320 },
        {   0x1.2ff8b8140934ep-161,  -0x1.733ac9d54b197p-215,  -0x1.b2e406690f322p-270,   0x1.88471a759c8f3p-326 },
        {  -0x1.176da24b03525p-165,   0x1.94a5b7fdd9213p-223,  -0x1.d9bac9c38cd1fp-280,  -0x1.0b40f505d2ddap-334 },
        {  -0x1.5a98a43547acbp-171,  -0x1.7f0a99491a422p-226,  -0x1.3f834ede1eef6p-280,   0x1.d6df32aa315b8p-334 },
        {   0x1.c7497a53702cdp-174,   0x1.60e4f38ac70d9p-229,  -0x1.a1f195ec28b13p-283,  -0x1.23c7731c20794p-339 },
        {  -0x1.3af7f24812e92p-180,   0x1.6cc07d179b8aep-235,   0x1.82c503ba202c0p-289,  -0x1.581df93cf9555p-343 },
        {  -0x1.07f8607306b4fp-182,  -0x1.8320772b8ecb2p-236,   0x1.404dd5f26b7c3p-291,  -0x1.2878aa0954f7ep-345 },
        {   0x1.3c7ac2ee69d5bp-187,  -0x1.644bdb0376a89p-245,  -0x1.9b9330513bcc2p-301,   0x1.5adfce07cd0d3p-357 },
        {   0x1.97e4357cb6176p-192,  -0x1.a474da8a56468p-246,  -0x1.308a68610ffd2p-301,   0x1.f491ee2e05c16p-355 },
        {  -0x1.133d9b77ae048p-195,  -0x1.15bc8a17aa4c7p-249,   0x1.f836a806a4f11p-306,  -0x1.d64b179bffbbdp-362 },
    };
    inline constexpr f256_s f256_erfc_cheb_3_4[] = {
        {    0x1.56e94c5092e6ep-18,   -0x1.f938cfaa5e33bp-75,  -0x1.e9f968ce9ca2cp-129,   0x1.53aa91820ffb1p-183 },
        {   -0x1.1e4b0457b2d90p-17,   -0x1.025f660f571d9p-72,  -0x1.1c9536f82274cp-127,   0x1.84bcdef48d817p-182 },
        {    0x1.56ce61801138dp-18,    0x1.21d8e2d52c25dp-73,   0x1.cfc4dc159b8f4p-136,   0x1.43ec288d34c61p-191 },
        {   -0x1.326157092f054p-19,    0x1.79e3bd1e6dec8p-77,  -0x1.29badd463ed20p-131,  -0x1.58f27cf03e24ep-185 },
        {    0x1.a69509fb57889p-21,   -0x1.c7353a2b7e928p-75,   0x1.1852fcf9df19bp-130,   0x1.3adba02ad4da6p-184 },
        {   -0x1.cc4efcd347bd5p-23,   -0x1.2305502e667bcp-77,   0x1.744c4eeae2898p-133,  -0x1.654ca71a346b3p-187 },
        {    0x1.913ee5a1ed735p-25,   -0x1.935ca2f99e315p-79,   0x1.7e4ebb3fa1249p-133,  -0x1.dd1ed40fb30a5p-189 },
        {   -0x1.188f54d77a769p-27,    0x1.95865d9580f76p-82,   0x1.4cbfcaf6c9150p-136,  -0x1.3945fd22f80b3p-190 },
        {    0x1.36b147aabdab0p-30,   -0x1.6398975d70977p-84,   0x1.ab522761cf2d9p-138,   0x1.e8d12664adbb8p-193 },
        {   -0x1.04c9618831f91p-33,   -0x1.aa3480d238001p-88,   0x1.fdf9776c02796p-143,   0x1.1ecf3a8d01819p-197 },
        {    0x1.1efe46a4a7cdbp-37,    0x1.75463902b2580p-91,   0x1.9c48d6f4c6324p-145,  -0x1.928fb39c21ba2p-199 },
        {   -0x1.a900b897ab008p-44,  -0x1.06364a27b1b69p-102,   0x1.91195e7187582p-157,   0x1.a88391e87818fp-211 },
        {   -0x1.177720be78023p-44,   0x1.fd2062f317e34p-100,   0x1.a95946cde1f58p-154,   0x1.19dc03e09b52ep-208 },
        {    0x1.476be39a8bfffp-47,  -0x1.19d80102e752dp-101,   0x1.410f1594b530cp-159,   0x1.fc9a6e5cfb119p-213 },
        {   -0x1.6754c1b54b70cp-51,   0x1.f2503dd3fcde4p-105,   0x1.80c84a495e7bfp-161,  -0x1.ec5884c9ed86cp-215 },
        {    0x1.a92dcd268c18dp-59,  -0x1.4a32ded90850fp-113,   0x1.5741a5c8ebd96p-167,   0x1.60aa0156425e6p-221 },
        {    0x1.3437bcfbe98aap-58,   0x1.c068780433d5cp-112,  -0x1.5d16a8846834ep-167,  -0x1.6c3dcb6836f34p-223 },
        {   -0x1.0852674b60f43p-61,   0x1.192a94bbb26c4p-115,  -0x1.7a0ff46c3ea8bp-169,   0x1.2c987dbff142bp-224 },
        {    0x1.34aa7e4161112p-66,   0x1.dc583488863b4p-120,   0x1.e2728e92f91dep-178,  -0x1.7f54f0ad5883bp-232 },
        {    0x1.7c095bacf5a79p-70,   0x1.3486c0ed3bb17p-125,  -0x1.393e37edb6f35p-179,   0x1.bf3d3df9c4133p-234 },
        {   -0x1.f1848dc222a24p-73,  -0x1.be04f9add21d1p-132,   0x1.8254509bef148p-188,  -0x1.bf8202d60c728p-243 },
        {    0x1.8508ec82ac24ep-77,  -0x1.0e33a5d7e641dp-133,  -0x1.ca4f1cb5448d5p-189,  -0x1.b9e6b95415dd1p-243 },
        {    0x1.7c8ef574fe1a7p-82,  -0x1.534bb08400228p-137,  -0x1.ed803fbef86e4p-193,   0x1.523b45faad99dp-247 },
        {   -0x1.747bdd308aafcp-84,   0x1.d86424589a503p-138,   0x1.cedc8d69c86b9p-194,  -0x1.9ec1e76d3fd9fp-250 },
        {    0x1.398d7af94eb18p-88,  -0x1.9cd2a00fd70a3p-143,   0x1.c83d3522acd19p-198,  -0x1.625be1578a735p-252 },
        {    0x1.9e65fde1d0f34p-94,  -0x1.b1b6a53018399p-148,   0x1.58c540d073ff2p-203,   0x1.585429b733aa1p-264 },
        {   -0x1.e0432029c8e37p-96,   0x1.cab65b56aee74p-155,  -0x1.7cbadbb58c2d0p-211,   0x1.156f133372620p-265 },
        {   0x1.7c7015484e7e5p-100,   0x1.638519eb5813bp-159,  -0x1.d177dde54d0b6p-216,   0x1.bb9eb2df708b4p-270 },
        {   0x1.23e0af450d49cp-105,   0x1.ae8987b4966b2p-159,  -0x1.d6a23cfb01a9dp-213,   0x1.509c0279fb68cp-267 },
        {  -0x1.101859704e903p-107,  -0x1.2c7fcec30ea0ap-161,  -0x1.bdf2ab0c7ec69p-216,  -0x1.aefa7640660dfp-271 },
        {   0x1.664b790508b2fp-112,  -0x1.fad9c34cc65fap-166,  -0x1.da6678f9e69fcp-220,  -0x1.5ea4033726d04p-274 },
        {   0x1.b2e8ff9e9838ap-117,   0x1.9208bb5b47c7dp-171,   0x1.f4e0bd6087d20p-227,   0x1.0998c9b86c698p-283 },
        {  -0x1.0c01fa961de91p-119,   0x1.32efcebccea3cp-173,  -0x1.895954040d3e8p-227,   0x1.68a8b7558f25dp-284 },
        {   0x1.f6c809b93b543p-125,   0x1.ebe68a72fa5fdp-181,  -0x1.1d7497c1d47c9p-235,  -0x1.a83ebfb70d318p-289 },
        {   0x1.18ac1cd06d5c2p-128,   0x1.9c65fa7b344b4p-182,  -0x1.19b40bfcb1927p-237,  -0x1.15d9808c5e8dap-291 },
        {  -0x1.bf315eb218ca3p-132,  -0x1.fa9b2b1e5e0bep-186,  -0x1.e2b133109e5abp-240,  -0x1.181b8dda743e9p-294 },
        {   0x1.a950854ad9473p-138,   0x1.12a5d3aa3dd8ap-193,  -0x1.f1e4cdcaa0b3cp-247,  -0x1.8a195624b4cdep-302 },
        {   0x1.26998fbb63f7cp-140,  -0x1.4fb34df256304p-196,   0x1.04c36daf5d5dep-251,  -0x1.05f255f695c5bp-305 },
        {  -0x1.302f321af596ep-144,   0x1.efc793b2f1170p-198,  -0x1.6fd77720da3b8p-254,   0x1.71a621ab459b7p-309 },
        {  -0x1.046dbbe85f84bp-152,  -0x1.128f44521510fp-208,   0x1.e29a77da1f383p-263,   0x1.2da3724423a22p-317 },
        {   0x1.f0106022c3b37p-153,  -0x1.cc1971e7d0aabp-209,   0x1.73ea77e1b1e70p-264,  -0x1.eaae10b0e2cc3p-319 },
        {  -0x1.3a5fd23fb2217p-157,   0x1.c9ac59e96388cp-211,   0x1.969e4e9e83643p-265,   0x1.8d1330bfdce34p-319 },
        {  -0x1.3c8989fe4de15p-162,   0x1.cefafa4b87783p-216,  -0x1.3a567b59a3634p-270,  -0x1.c7aaee9c15e39p-325 },
        {   0x1.4b24bba86ba3bp-165,  -0x1.a9a7a4eb123b0p-219,  -0x1.cfe0599af4150p-276,  -0x1.0aa311c00550dp-330 },
        {  -0x1.8ad095857cce9p-171,  -0x1.ec7e1695c2819p-225,   0x1.bbf7edcebe170p-283,   0x1.371cdc0a8b233p-337 },
        {  -0x1.508f0136e4c55p-174,  -0x1.610f5fbe0553fp-229,  -0x1.3c032cb2c6f71p-285,  -0x1.398ab14eef776p-340 },
        {   0x1.5325472cd461ep-178,  -0x1.be070aa9e6708p-232,  -0x1.9424187dd7358p-286,   0x1.a01887d174e65p-340 },
        {   0x1.5936fbf9ec655p-186,   0x1.36b1efd62ae1fp-243,  -0x1.5197195d42727p-297,   0x1.396f96fdbf03dp-352 },
        {  -0x1.d3257cb381cb2p-187,  -0x1.a9acc2bacdf6ap-241,   0x1.3204cc65fcf88p-298,  -0x1.540bd126cad21p-352 },
        {   0x1.df5871880e591p-192,  -0x1.e091979f28c8ap-246,   0x1.acea2fc38ae40p-301,   0x1.0ee226696b3aep-355 },
        {   0x1.3c421b3b2b2dfp-196,  -0x1.a2c79eef97c8bp-252,  -0x1.4da91e762e58ap-308,   0x1.f465abcf26a83p-363 },
        {  -0x1.d527b931bde63p-200,  -0x1.a90490031a12ep-254,  -0x1.3a30d8ce0d4c0p-308,  -0x1.5aa90a6cd2fa6p-362 },
    };

    [[nodiscard]] BL_NO_INLINE constexpr f256_s erf_cheb_eval(const f256_s& x, const f256_s* coeffs, double shift)
    {
        const f256_s t = detail::_f256::sub_inline(detail::_f256::mul_double_inline(x, 2.0), f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (int i = static_cast<int>(f256_erf_cheb_coeff_count) - 1; i >= 1; --i)
        {
            const f256_s b0 = detail::_f256::add_inline(
                detail::_f256::sub_inline(detail::_f256::mul_double_inline(detail::_f256::mul_inline(t, b1), 2.0), b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return detail::_f256::add_inline(detail::_f256::sub_inline(detail::_f256::mul_inline(t, b1), b2), coeffs[0]);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s erf_positive_cheb(const f256_s& x)
    {
        if (x < f256_s{ 1.0 })
            return detail::_f256::erf_cheb_eval(x, detail::_f256::f256_erf_cheb_0_1, 1.0);
        if (x < f256_s{ 2.0 })
            return detail::_f256::erf_cheb_eval(x, detail::_f256::f256_erf_cheb_1_2, 3.0);
        return detail::_f256::erf_cheb_eval(x, detail::_f256::f256_erf_cheb_2_3, 5.0);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s erfc_positive_cheb_3_4(const f256_s& x)
    {
        return detail::_f256::erf_cheb_eval(x, detail::_f256::f256_erfc_cheb_3_4, 7.0);
    }
    [[nodiscard]] BL_NO_INLINE constexpr f256_s erfc_positive_cf(const f256_s& x)
    {
        //const f256_s z = detail::_f256::mul_inline(x, x);
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
            //const f256_s an = -detail::_f256::mul_inline(ii, ii - a);
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

        return detail::_f256::mul_inline(
            detail::_f256::mul_inline(detail::_f256::_exp(-z), x),
            detail::_f256::mul_inline(detail::_f256::inv_sqrtpi, h));
    }

    BL_NO_INLINE constexpr f256_s lgamma_stirling_asymptotic(const f256_s& z) noexcept
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

        return (z - f256_s{ 0.5 }) * log(z) - z + detail::_f256::half_log_two_pi + series;
    }
    BL_NO_INLINE constexpr void positive_recurrence_product(const f256_s& x, const f256_s& asymptotic_min, f256_s& z, f256_s& product, int& product_exp2) noexcept
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
    BL_NO_INLINE constexpr f256_s lgamma_positive_recurrence(const f256_s& x) noexcept
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
            - f256_s{ static_cast<double>(product_exp2) } * detail::_f256::ln2;
    }
    BL_NO_INLINE constexpr f256_s gamma_positive_recurrence(const f256_s& x) noexcept
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

[[nodiscard]] BL_NO_INLINE constexpr f256_s erf(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s out{ 0.0 };

    if (ax < f256_s{ 3.0 })
    {
        out = detail::_f256::erf_positive_cheb(ax);
    }
    else if (ax < f256_s{ 4.0 })
    {
        out = f256_s{ 1.0 } - detail::_f256::erfc_positive_cheb_3_4(ax);
    }
    else
    {
        out = f256_s{ 1.0 } - detail::_f256::erfc_positive_cf(ax);
    }

    if (neg)
        out = -out;

    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s erfc(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return f256_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f256_s{ 2.0 } : f256_s{ 0.0 };

    if (signbit(x))
        return detail::_f256::canonicalize_math_result(f256_s{ 1.0 } + erf(-x));

    if (x < f256_s{ 3.0 })
        return detail::_f256::canonicalize_math_result(f256_s{ 1.0 } - detail::_f256::erf_positive_cheb(x));

    if (x < f256_s{ 4.0 })
        return detail::_f256::canonicalize_math_result(detail::_f256::erfc_positive_cheb_3_4(x));

    if (x > f256_s{ 40.0 })
        return f256_s{ 0.0 };

    return detail::_f256::canonicalize_math_result(detail::_f256::erfc_positive_cf(x));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s lgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return detail::_f256::canonicalize_math_result(detail::_f256::lgamma_positive_recurrence(x));

    const f256_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::infinity();

    const f256_s sinpix = sin(detail::_f256::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s out =
        log(detail::_f256::pi)
        - log(abs(sinpix))
        - detail::_f256::lgamma_positive_recurrence(f256_s{ 1.0 } - x);

    return detail::_f256::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return detail::_f256::canonicalize_math_result(detail::_f256::gamma_positive_recurrence(x));

    const f256_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s sinpix = sin(detail::_f256::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s out = detail::_f256::pi / (sinpix * detail::_f256::gamma_positive_recurrence(f256_s{ 1.0 } - x));
    return detail::_f256::canonicalize_math_result(out);
}

}

#endif
