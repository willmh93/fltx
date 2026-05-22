/**
 * fltx/detail/f128/math_shared.h - Shared f128 math implementation support.
 *
 * Shared f128 helpers used by math implementation headers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_MATH_SHARED_INCLUDED
#define FLTX_F128_DETAIL_MATH_SHARED_INCLUDED
#include "fltx/detail/f128/declarations.h"
#include "fltx/f128/rounding.h"

namespace bl {

namespace detail::_f128
{
    using detail::fp::signbit;
    using detail::fp::fabs;
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::double_integer_is_odd;
    using detail::fp::fmod;
    using detail::fp::sqrt_seed;
    using detail::fp::nearbyint_ties_even;

    // scaling helpers
    BL_FORCE_INLINE constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::frexp_exponent(value);

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }

    BL_FORCE_INLINE constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::ldexp(value, exponent);

        return std::ldexp(value, exponent);
    }

    BL_FORCE_INLINE constexpr f128_s ldexp_terms(const f128_s& value, int exponent) noexcept
    {
        return renorm(
            ldexp_limb(value.hi, exponent),
            ldexp_limb(value.lo, exponent));
    }

    BL_FORCE_INLINE constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        if (bl::use_constexpr_math())
        {
            return renorm(
                detail::fp::ldexp(x.hi, e),
                detail::fp::ldexp(x.lo, e)
            );
        }
        else
        {
            return renorm(
                std::ldexp(x.hi, e),
                std::ldexp(x.lo, e)
            );
        }
    }

    struct fmod_u128
    {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
    };
    struct exact_dyadic_fmod
    {
        bool neg = false;
        int exp2 = 0;
        fmod_u128 mant{};
    };

    // exact integer helpers
    BL_FORCE_INLINE constexpr bool fmod_u128_is_zero(const fmod_u128& value)
    {
        return value.lo == 0 && value.hi == 0;
    }

    BL_FORCE_INLINE constexpr bool fmod_u128_is_odd(const fmod_u128& value)
    {
        return (value.lo & 1u) != 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u128_compare(const fmod_u128& a, const fmod_u128& b)
    {
        if (a.hi < b.hi) return -1;
        if (a.hi > b.hi) return 1;
        if (a.lo < b.lo) return -1;
        if (a.lo > b.lo) return 1;
        return 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u128_bit_length(const fmod_u128& value)
    {
        if (value.hi != 0)
            return 128 - static_cast<int>(std::countl_zero(value.hi));
        if (value.lo != 0)
            return 64 - static_cast<int>(std::countl_zero(value.lo));
        return 0;
    }

    BL_FORCE_INLINE constexpr int fmod_u128_trailing_zero_bits(const fmod_u128& value)
    {
        if (value.lo != 0)
            return static_cast<int>(std::countr_zero(value.lo));
        if (value.hi != 0)
            return 64 + static_cast<int>(std::countr_zero(value.hi));
        return 0;
    }

    BL_FORCE_INLINE constexpr bool fmod_u128_get_bit(const fmod_u128& value, int index)
    {
        if (index < 0 || index >= 128)
            return false;
        if (index < 64)
            return ((value.lo >> index) & 1u) != 0;
        return ((value.hi >> (index - 64)) & 1u) != 0;
    }

    BL_FORCE_INLINE constexpr std::uint64_t fmod_u128_get_bits(const fmod_u128& value, int start, int count)
    {
        std::uint64_t out = 0;
        for (int i = 0; i < count; ++i)
        {
            if (fmod_u128_get_bit(value, start + i))
                out |= (std::uint64_t{ 1 } << i);
        }
        return out;
    }

    BL_FORCE_INLINE constexpr bool fmod_u128_any_low_bits_set(const fmod_u128& value, int count)
    {
        if (count <= 0)
            return false;

        if (count >= 64)
        {
            if (value.lo != 0)
                return true;
            count -= 64;
            if (count >= 64)
                return value.hi != 0;
            return (value.hi & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
        }

        return (value.lo & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
    }

    BL_FORCE_INLINE constexpr void fmod_u128_add_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += b.lo;
        a.hi += b.hi + (a.lo < old_lo ? 1u : 0u);
    }

    BL_FORCE_INLINE constexpr void fmod_u128_add_small(fmod_u128& a, std::uint32_t value)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += value;
        if (a.lo < old_lo)
            ++a.hi;
    }

    BL_FORCE_INLINE constexpr void fmod_u128_sub_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t borrow = (a.lo < b.lo) ? 1u : 0u;
        a.lo -= b.lo;
        a.hi -= b.hi + borrow;
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shl_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { 0, value.lo << (bits - 64) };

        return {
            value.lo << bits,
            (value.hi << bits) | (value.lo >> (64 - bits))
        };
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shr_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { value.hi >> (bits - 64), 0 };

        return {
            (value.lo >> bits) | (value.hi << (64 - bits)),
            value.hi >> bits
        };
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shl1(fmod_u128 value)
    {
        return { value.lo << 1, (value.hi << 1) | (value.lo >> 63) };
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_mod_shift_subtract(fmod_u128 numerator, const fmod_u128& denominator)
    {
        if (fmod_u128_is_zero(denominator))
            return {};
        if (fmod_u128_compare(numerator, denominator) < 0)
            return numerator;

        int shift = fmod_u128_bit_length(numerator) - fmod_u128_bit_length(denominator);
        fmod_u128 shifted = fmod_u128_shl_bits(denominator, shift);

        for (; shift >= 0; --shift)
        {
            if (fmod_u128_compare(numerator, shifted) >= 0)
                fmod_u128_sub_inplace(numerator, shifted);
            shifted = fmod_u128_shr_bits(shifted, 1);
        }

        return numerator;
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_double_mod(fmod_u128 value, const fmod_u128& modulus)
    {
        value = fmod_u128_shl1(value);
        if (fmod_u128_compare(value, modulus) >= 0)
            fmod_u128_sub_inplace(value, modulus);
        return value;
    }

    // exact fmod conversion
    BL_FORCE_INLINE constexpr exact_dyadic_fmod exact_from_double_fmod(double value)
    {
        exact_dyadic_fmod out;
        if (value == 0.0)
            return out;

        int exponent = 0;
        bool neg = false;
        const std::uint64_t mantissa = detail::exact_decimal::decompose_double_mantissa(value, exponent, neg);
        if (mantissa == 0)
            return out;

        out.neg = neg;
        out.exp2 = exponent;
        out.mant.lo = mantissa;
        return out;
    }

    BL_FORCE_INLINE constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
    {
        if (fmod_u128_is_zero(value.mant))
        {
            value.neg = false;
            value.exp2 = 0;
            return;
        }

        const int tz = fmod_u128_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = fmod_u128_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }

    BL_FORCE_INLINE constexpr exact_dyadic_fmod exact_from_f128_fmod(const f128_s& value)
    {
        exact_dyadic_fmod hi = exact_from_double_fmod(value.hi);
        exact_dyadic_fmod lo = exact_from_double_fmod(value.lo);

        if (fmod_u128_is_zero(hi.mant))
            return lo;
        if (fmod_u128_is_zero(lo.mant))
            return hi;

        const int common_exp = (hi.exp2 < lo.exp2) ? hi.exp2 : lo.exp2;
        const fmod_u128 hi_scaled = fmod_u128_shl_bits(hi.mant, hi.exp2 - common_exp);
        const fmod_u128 lo_scaled = fmod_u128_shl_bits(lo.mant, lo.exp2 - common_exp);

        exact_dyadic_fmod out;
        out.exp2 = common_exp;

        if (hi.neg == lo.neg)
        {
            out.neg = hi.neg;
            out.mant = hi_scaled;
            fmod_u128_add_inplace(out.mant, lo_scaled);
        }
        else
        {
            const int cmp = fmod_u128_compare(hi_scaled, lo_scaled);
            if (cmp >= 0)
            {
                out.neg = hi.neg;
                out.mant = hi_scaled;
                fmod_u128_sub_inplace(out.mant, lo_scaled);
            }
            else
            {
                out.neg = lo.neg;
                out.mant = lo_scaled;
                fmod_u128_sub_inplace(out.mant, hi_scaled);
            }
        }

        normalize_exact_dyadic_fmod(out);
        return out;
    }

    // fmod kernels
    BL_FORCE_INLINE constexpr bool fmod_fast_double_divisor_abs(const f128_s& ax, double ay, f128_s& out)
    {
        if (!(ay > 0.0) || !isfinite(ay))
            return false;

        const f128_s mod{ ay, 0.0 };

        if (ax.lo == 0.0)
        {
            out = f128_s{ fmod(ax.hi, ay), 0.0 };
            return true;
        }

        const double rh = (ax.hi < ay) ? ax.hi : fmod(ax.hi, ay);
        const double rl = (absd(ax.lo) < ay) ? ax.lo : fmod(ax.lo, ay);

        f128_s r = add_inline(f128_s{ rh, 0.0 }, f128_s{ rl, 0.0 });

        if (r < 0.0)
            r = add_inline(r, mod);
        if (r >= mod)
            r = sub_inline(r, mod);

        if (r < 0.0)
            r = add_inline(r, mod);
        if (r >= mod)
            r = sub_inline(r, mod);

        if (r < 0.0 || r >= mod)
            return false;

        out = r;
        return true;
    }

    BL_FORCE_INLINE constexpr f128_s exact_dyadic_to_f128_fmod(const fmod_u128& coeff, int exp2, bool neg)
    {
        if (fmod_u128_is_zero(coeff))
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        int ratio_exp = fmod_u128_bit_length(coeff) - 1;
        fmod_u128 q = coeff;

        if (ratio_exp > 105)
        {
            const int right_shift = ratio_exp - 105;
            const bool round_bit = fmod_u128_get_bit(q, right_shift - 1);
            const bool sticky    = fmod_u128_any_low_bits_set(q, right_shift - 1);

            q = fmod_u128_shr_bits(q, right_shift);

            if (round_bit && (sticky || fmod_u128_is_odd(q)))
                fmod_u128_add_small(q, 1u);

            if (fmod_u128_bit_length(q) > 106)
            {
                q = fmod_u128_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < 105)
        {
            q = fmod_u128_shl_bits(q, 105 - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f128_s>::infinity() : std::numeric_limits<f128_s>::infinity();
        if (e2 < -1074)
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        const std::uint64_t c1 = fmod_u128_get_bits(q, 0, 53);
        const std::uint64_t c0 = fmod_u128_get_bits(q, 53, 53);
        const double hi = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = renorm(hi, lo);
        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr f128_s fmod_exact_fixed_limb(const f128_s& x, const f128_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f128_fmod(mag(x));
        const exact_dyadic_fmod dy = exact_from_f128_fmod(mag(y));

        fmod_u128 remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            const fmod_u128 denominator = fmod_u128_shl_bits(dy.mant, shift);
            remainder = fmod_u128_mod_shift_subtract(dx.mant, denominator);
            out_exp = dx.exp2;
        }
        else
        {
            remainder = fmod_u128_mod_shift_subtract(dx.mant, dy.mant);
            const int shift = dx.exp2 - dy.exp2;
            for (int i = 0; i < shift && !fmod_u128_is_zero(remainder); ++i)
                remainder = fmod_u128_double_mod(remainder, dy.mant);
            out_exp = dy.exp2;
        }

        f128_s out = exact_dyadic_to_f128_fmod(remainder, out_exp, !ispositive(x));
        if (iszero(out))
            return f128_s{ signbit(x.hi) ? -0.0 : 0.0 };
        return out;
    }

    // decimal conversion
    BL_FORCE_INLINE constexpr bool f128_try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = trunc(x);
        if (xi != x)
            return false;

        if (absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = sub_inline(xi, to_f128(hi_part));
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }

    BL_FORCE_INLINE constexpr f128_s pack_decimal_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
    {
        const std::uint64_t c1 = q.get_bits(0, 53);
        const std::uint64_t c0 = q.get_bits(53, 53);
        const double hi = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = renorm(hi, lo);
        return neg ? -out : out;
    }

    BL_MSVC_NOINLINE constexpr f128_s round_decimal_exact_to_f128(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg) noexcept
    {
        if (coeff.is_zero())
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        detail::exact_decimal::biguint numerator = coeff;
        detail::exact_decimal::biguint denominator{ 1 };
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
        detail::exact_decimal::biguint q = detail::exact_decimal::extract_rounded_significand_chunks(numerator, denominator, ratio_exp, std::numeric_limits<f128_s>::digits);
        if (q.bit_length() > std::numeric_limits<f128_s>::digits)
        {
            q.shr1();
            ++ratio_exp;
        }

        const int e2 = bin_exp + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f128_s>::infinity() : std::numeric_limits<f128_s>::infinity();
        if (e2 < -1074)
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        return pack_decimal_significand(q, e2, neg);
    }

    BL_FORCE_INLINE constexpr bool try_rounded_decimal_to_f128(const f128_s& integer_part, const char* digits, int digit_count, bool neg, f128_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!f128_try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        const detail::exact_decimal::biguint coeff = detail::fp::append_decimal_digits(
            detail::exact_decimal::biguint{ static_cast<std::uint64_t>(integer_value) },
            digits,
            digit_count);

        out = round_decimal_exact_to_f128(coeff, -digit_count, neg);
        return true;
    }

    // rounding helpers
    BL_FORCE_INLINE constexpr f128_s round_half_away_zero(const f128_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f128_s y = -floor(add_inline(-x, f128_s{ 0.5 }));
            if (iszero(y))
                return f128_s{ -0.0, 0.0 };
            return y;
        }

        return floor(add_inline(x, f128_s{ 0.5 }));
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f128_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (bl::isnan(x) || bl::isinf(x))
            return 0;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        const f128_s lo = to_f128(lo_i);
        const f128_s hi = to_f128(hi_i);

        if (x < lo || x > hi)
            return 0;

        std::int64_t out = 0;
        if (!f128_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr bool try_round_to_signed_integer(const f128_s& x, bool ties_to_even, SignedInt& out) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (bl::isnan(x) || bl::isinf(x) || absd(x.hi) >= 0x1p52)
            return false;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        if (x < to_f128(lo_i) || x > to_f128(hi_i))
            return false;

        const std::int64_t base = static_cast<std::int64_t>(x.hi);
        const f128_s frac     = sub_double_inline(x, static_cast<double>(base));
        const f128_s abs_frac = mag(frac);
        std::int64_t rounded = base;

        if (abs_frac > f128_s{ 0.5 } || (!ties_to_even && abs_frac == f128_s{ 0.5 }) ||
            (ties_to_even && abs_frac == f128_s{ 0.5 } && (base & 1ll) != 0))
        {
            rounded += (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi))) ? -1 : 1;
        }

        if (rounded < lo_i || rounded > hi_i)
            return false;

        out = static_cast<SignedInt>(rounded);
        return true;
    }

    BL_FORCE_INLINE constexpr f128_s nearest_integer_ties_even(const f128_s& q) noexcept
    {
        f128_s n = trunc(q);
        const f128_s frac = sub_inline(q, n);
        const f128_s half{ 0.5 };
        const f128_s one{ 1.0 };

        if (mag(frac) > half)
        {
            n = add_inline(n, signbit(frac) ? -one : one);
        }
        else if (mag(frac) == half)
        {
            if (detail::_f128_constexpr::fmod(n, f128_s{ 2.0 }) != f128_s{ 0.0 })
                n = add_inline(n, signbit(frac) ? -one : one);
        }

        return n;
    }


} // namespace detail::_f128

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    const int exp2 = frexp_exponent(a.hi);
    const int result_scale = exp2 / 2;
    const int input_scale = -2 * result_scale;
    const f128_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

    double y0;
    if (bl::use_constexpr_math()) {
        y0 = sqrt_seed(scaled_a.hi);
        f128_s y{ y0 };
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_math_result(y);
    }
    else {
        y0 = std::sqrt(scaled_a.hi);
        f128_s y{ y0 };
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, mul_inline(sub_mul_inline(scaled_a, y, y), f128_s{ 0.5 / y0 }));

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_math_result(y);
    }
}

// remainders
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::fmod(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    const f128_s ay = detail::_f128::mag(y);

    if (ax < ay)
        return x;

    f128_s fast{};
    if (y.lo == 0.0 && fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ signbit(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return fmod_exact_fixed_limb(x, y);
}

// decomposition and scaling
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::ldexp(const f128_s& x, int e)
{
    return canonicalize_math_result(_ldexp(x, e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::frexp(const f128_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const double lead = (x.hi != 0.0) ? x.hi : x.lo;
    int e = 0;

    if (bl::use_constexpr_math())
        e = detail::fp::frexp_exponent(lead);
    else
        (void)std::frexp(lead, &e);

    f128_s m = detail::_f128_constexpr::ldexp(x, -e);
    const f128_s am = detail::_f128::mag(m);

    if (am < f128_s{ 0.5 })
    {
        m = mul_inline(m, f128_s{ 2.0 });
        --e;
    }
    else if (am >= f128_s{ 1.0 })
    {
        m = mul_inline(m, f128_s{ 0.5 });
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}

// adjacent values
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::nextafter(const f128_s& from, const f128_s& to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (from == to)
        return to;
    if (iszero(from))
        return signbit(to)
            ? f128_s{ -std::numeric_limits<double>::denorm_min(), 0.0 }
            : f128_s{  std::numeric_limits<double>::denorm_min(), 0.0 };
    if (isinf(from))
        return signbit(from)
            ? -std::numeric_limits<f128_s>::max()
            :  std::numeric_limits<f128_s>::max();

    const double toward = (from < to)
        ? std::numeric_limits<double>::infinity()
        : -std::numeric_limits<double>::infinity();

    return renorm(
        from.hi,
        detail::fp::nextafter(from.lo, toward)
    );
}

} // namespace bl

#endif
