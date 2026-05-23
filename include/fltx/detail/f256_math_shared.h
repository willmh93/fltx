/**
 * fltx/detail/f256_math_shared.h - Shared f256 math implementation support.
 *
 * Shared f256 helpers used by math implementation headers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_MATH_SHARED_INCLUDED
#define FLTX_F256_DETAIL_MATH_SHARED_INCLUDED
#include "fltx/detail/f256_declarations.h"
#include "fltx/detail/f256_rounding.h"

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

    BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept;
    BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept;

    // expression fallbacks
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
    BL_FORCE_INLINE constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return detail::fp::frexp_exponent(value);
        }

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }

    BL_FORCE_INLINE constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return detail::fp::ldexp(value, exponent);
        }

        return std::ldexp(value, exponent);
    }

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
            return f256_s{ signbit(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

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
            return f256_s{ signbit(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
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

    BL_FORCE_INLINE constexpr void fmod_append_expansion_term(double* expansion, int& count, double value) noexcept
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

    BL_MSVC_NOINLINE constexpr bool fmod_fast_small_quotient_abs(const f256_s& ax, const f256_s& ay, f256_s& out) noexcept
    {
        if (!(ay.x0 > 0.0) || !isfinite(ay) || !(ax >= ay))
            return false;

        const double q = detail::fp::trunc(ax.x0 / ay.x0);
        if (!(q > 0.0) || q >= 0x1p42)
            return false;

        f256_s r = fmod_sub_mul_scalar_expansion(ax, ay, q);
        if (!fmod_normalize_remainder(r, ay))
            return false;

        out = r;
        return true;
    }

    BL_MSVC_NOINLINE constexpr f256_s fmod_runtime(const f256_s& x, const f256_s& y)
    {
        const f256_s ay = mag(y);
        f256_s r = mag(x);

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

            const f256_s q_floor = detail::_f256_impl::floor(r / scaled);
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
            return f256_s{ signbit(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

        return ispositive(x) ? r : -r;
    }

    BL_MSVC_NOINLINE constexpr bool fmod_fast_double_divisor_abs(const f256_s& ax, double ay, f256_s& out)
    {
        if (!(ay > 0.0) || !isfinite(ay))
            return false;

        const f256_s mod{ ay, 0.0, 0.0, 0.0 };

        if (ax.x1 == 0.0 && ax.x2 == 0.0 && ax.x3 == 0.0)
        {
            out = f256_s{ fmod(ax.x0, ay), 0.0, 0.0, 0.0 };
            return true;
        }

        const double r0 = (ax.x0 < ay) ? ax.x0 : fmod(ax.x0, ay);
        const double r1 = (absd(ax.x1) < ay) ? ax.x1 : fmod(ax.x1, ay);
        const double r2 = (absd(ax.x2) < ay) ? ax.x2 : fmod(ax.x2, ay);
        const double r3 = (absd(ax.x3) < ay) ? ax.x3 : fmod(ax.x3, ay);

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
        const f256_s ar    = mag(r);
        const f256_s slack = mul_double_inline(mod, 0x1p-160);
        if (ar <= slack || ar >= mod - slack)
            return false;

        out = r;
        return true;
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
        if (value == 0.0 || !isfinite(value) || absd(value) >= 0x1p53)
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

    BL_FORCE_INLINE constexpr double limb_mod_power_of_two(double value, double modulus, double zero_threshold) noexcept
    {
        if (value == 0.0 || !isfinite(value) || absd(value) >= zero_threshold)
            return 0.0;

        return fmod(value, modulus);
    }

    BL_FORCE_INLINE constexpr int low_quotient_bits(const f256_s& x) noexcept
    {
        constexpr double modulus = 2147483648.0;
        constexpr double zero_threshold = 0x1p83;

        double bits =
            limb_mod_power_of_two(x.x0, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x1, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x2, modulus, zero_threshold) +
            limb_mod_power_of_two(x.x3, modulus, zero_threshold);

        bits = fmod(bits, modulus);
        return static_cast<int>(static_cast<long long>(nearbyint_ties_even(bits)));
    }

    // polynomial evaluation
    BL_FORCE_INLINE constexpr f256_s mul_add_horner_step_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return mul_add_inline(a, b, c);
    }

    BL_FORCE_INLINE constexpr f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return mul_add_horner_step_inline(a, b, c);
        }

        return detail::_f256_runtime::mul_add_horner_step(a, b, c);
    }

    BL_FORCE_INLINE constexpr f256_s horner_forward_inline(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_FORCE_INLINE constexpr f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_forward_inline(coeffs, count, x);
        }

        return detail::_f256_runtime::horner_forward(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr f256_s horner_reverse_inline(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_FORCE_INLINE constexpr f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_reverse_inline(coeffs, count, x);
        }

        return detail::_f256_runtime::horner_reverse(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward_inline(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f256_s{};
            right_out = f256_s{};
            return;
        }

        f256_s left  = left_coeffs[0];
        f256_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            left = mul_add_inline(left, x, left_coeffs[i]);
            right = mul_add_inline(right, x, right_coeffs[i]);
        }

        left_out = left;
        right_out = right;
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (bl::use_constexpr_math())
        {
            horner_pair_forward_inline(left_coeffs, right_coeffs, count, x, left_out, right_out);
            return;
        }

        detail::_f256_runtime::horner_pair_forward(left_coeffs, right_coeffs, count, x, left_out, right_out);
    }

    // sqrt kernels
    BL_FORCE_INLINE constexpr f256_s canonicalize_sqrt_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<16>(value.x3);
        return value;
    }

    #if defined(FLTX_CONSTEXPR_PARITY)
        #define F256_CANONICALIZE_SQRT_RESULT(value) ::bl::detail::_f256::canonicalize_sqrt_result(value)
    #else
        #define F256_CANONICALIZE_SQRT_RESULT(value) (value)
    #endif

    BL_FORCE_INLINE constexpr void sqrt_step_seed_recip(const f256_s& scaled_a, f256_s& y, double half_inv_y0)
    {
        using namespace detail::_f256;
        const f256_s residual = sub_inline(scaled_a, sqr_inline(y));
        y = add_inline(y, mul_double_inline(residual, half_inv_y0));
    }

    BL_FORCE_INLINE  double sqrt_tail_residual_head(const f256_s& scaled_a, const f256_s& y)
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

        double compressed[24];
        const int count = compress_expansion_zeroelim(static_cast<int>(sizeof(terms) / sizeof(terms[0])), terms, compressed);
        double residual = 0.0;
        for (int i = 0; i < count; ++i)
            residual += compressed[i];
        return residual;
    }

    BL_FORCE_INLINE  f256_s sqrt_step_tail_head(const f256_s& scaled_a, const f256_s& y, double half_inv_y0)
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

        const double y0 = sqrt_seed(scaled_a.x0);
        const double half_inv_y0 = 0.5 / y0;
        f256_s y{ y0, 0.0, 0.0, 0.0 };
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);
        sqrt_step_seed_recip(scaled_a, y, half_inv_y0);

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return F256_CANONICALIZE_SQRT_RESULT(y);
    }

    BL_FORCE_INLINE  f256_s sqrt_runtime_impl(const f256_s& a)
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

        return F256_CANONICALIZE_SQRT_RESULT(y);
    }

    // rounding helpers
    BL_FORCE_INLINE constexpr f256_s round_half_away_zero(const f256_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
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

        if (bl::isnan(x) || bl::isinf(x))
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

        if (bl::isnan(x) || bl::isinf(x) || absd(x.x0) >= 0x1p52)
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

    BL_FORCE_INLINE constexpr f256_s nearest_integer_ties_even(const f256_s& q) noexcept
    {
        f256_s n = detail::_f256_impl::trunc(q);
        const f256_s frac = sub_inline(q, n);
        const f256_s half{ 0.5 };
        const f256_s one{ 1.0 };

        if (mag(frac) > half)
        {
            n = add_inline(n, bl::signbit(frac) ? -one : one);
        }
        else if (mag(frac) == half)
        {
            if (is_odd_integer(n))
                n = add_inline(n, bl::signbit(frac) ? -one : one);
        }

        return n;
    }

    // scaling kernels
    BL_FORCE_INLINE constexpr f256_s _ldexp(const f256_s& a, int e)
    {
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

// remainders
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmod(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    const f256_s ay = detail::_f256::mag(y);

    if (ax < ay)
        return x;

    f256_s fast{};
    if (y.x1 == 0.0 && y.x2 == 0.0 && y.x3 == 0.0 && fmod_fast_double_divisor_abs(ax, ay.x0, fast))
    {
        if (iszero(fast))
            return f256_s{ signbit(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        const f256_s out = ispositive(x) ? fast : -fast;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    if (!bl::use_constexpr_math() && fmod_fast_small_quotient_abs(ax, ay, fast))
    {
        if (iszero(fast))
            return f256_s{ signbit(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        const f256_s out = ispositive(x) ? fast : -fast;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    const f256_s out = bl::use_constexpr_math()
        ? fmod_exact(x, y)
        : fmod_runtime(x, y);

    return F256_CANONICALIZE_MATH_RESULT(out);
}

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sqrt(const f256_s& a)
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
    {
        return sqrt_constexpr_impl(a);
    }

    return sqrt_runtime_impl(a);
}

// rounding
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nearbyint(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f256_s t = detail::_f256_impl::floor(a);
    const f256_s frac = sub_inline(a, t);

    if (frac < f256_s{ 0.5 })
        return t;

    if (frac > f256_s{ 0.5 })
    {
        t = add_inline(t, f256_s{ 1.0 });
        if (iszero(t))
            return f256_s{ signbit(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return t;
    }

    if (is_odd_integer(t))
        t = add_inline(t, f256_s{ 1.0 });

    if (iszero(t))
        return f256_s{ signbit(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return t;
}

// decomposition and scaling
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::ldexp(const f256_s& a, int e)
{
    return F256_CANONICALIZE_MATH_RESULT(_ldexp(a, e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::frexp(const f256_s& x, int* exp) noexcept
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
    {
        e = detail::fp::frexp_exponent(lead);
    }
    else
    {
        (void)std::frexp(lead, &e);
    }

    f256_s m = detail::_f256_impl::ldexp(x, -e);
    const f256_s am = detail::_f256::mag(m);

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

// adjacent values
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nextafter(const f256_s& from, const f256_s& to) noexcept
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
        detail::fp::nextafter(from.x3, toward));
}

} // namespace bl

#endif
