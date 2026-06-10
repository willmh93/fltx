/**
 * fltx/detail/f128_math_kernels.h - f128 math kernels and helper algorithms.
 *
 * Low-level f128 helper logic used by grouped math implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_MATH_KERNELS_INCLUDED
#define F128_DETAIL_MATH_KERNELS_INCLUDED
#include "fltx/detail/f128_declarations.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    using detail::fp::signbit;
    using detail::fp::fabs;
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::double_integer_is_odd;
    using detail::fp::fmod;
    using detail::fp::sqrt_seed;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::two_diff_precise;
    using detail::fp::abs_double_is_power_of_two;
    using detail::fp::frexp_exponent_limb;
    using detail::fp::ldexp_limb;

    BL_FORCE_INLINE constexpr int ilogb_finite_fast(const f128_s& x) noexcept
    {
        const double hi = x.hi != 0.0 ? x.hi : x.lo;
        const double abs_hi = absd(hi);
        int exponent = detail::fp::frexp_exponent(abs_hi) - 1;

        if (x.hi != 0.0 && x.lo != 0.0 && signbit(x.hi) != signbit(x.lo) &&
            abs_double_is_power_of_two(abs_hi))
        {
            --exponent;
        }

        return exponent;
    }

    // scaling helpers
    BL_FORCE_INLINE constexpr bool ldexp_normal_limb(double value, int exponent, double& out) noexcept
    {
        if (value == 0.0 || exponent == 0)
        {
            out = value;
            return true;
        }

        constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
        constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;
        constexpr std::uint64_t sign_mask     = 0x8000000000000000ull;

        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        const std::uint32_t exponent_bits =
            static_cast<std::uint32_t>((bits & exponent_mask) >> 52);
        if (exponent_bits == 0u || exponent_bits == 0x7ffu)
            return false;

        const int scaled_exponent = static_cast<int>(exponent_bits) + exponent;
        if (scaled_exponent <= 0 || scaled_exponent >= 0x7ff)
            return false;

        out = std::bit_cast<double>(
            (bits & (sign_mask | fraction_mask)) |
            (static_cast<std::uint64_t>(scaled_exponent) << 52));
        return true;
    }

    BL_FORCE_INLINE constexpr bool ldexp_fast_normal(const f128_s& value, int exponent, f128_s& out) noexcept
    {
        if (exponent == 0)
        {
            out = value;
            return true;
        }

        double hi{};
        double lo{};
        if (!ldexp_normal_limb(value.hi, exponent, hi) ||
            !ldexp_normal_limb(value.lo, exponent, lo))
        {
            return false;
        }

        out = f128_s{ hi, lo };
        return true;
    }

    BL_FORCE_INLINE constexpr f128_s ldexp_terms(const f128_s& value, int exponent) noexcept
    {
        return renorm(
            ldexp_limb(value.hi, exponent),
            ldexp_limb(value.lo, exponent));
    }

    BL_FORCE_INLINE constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        f128_s fast{};
        if (ldexp_fast_normal(x, e, fast))
            return fast;

        if (bl::detail::use_constexpr_math())
        {
            return renorm(
                detail::fp::ldexp(x.hi, e),
                detail::fp::ldexp(x.lo, e)
            );
        }

        return renorm(
            std::ldexp(x.hi, e),
            std::ldexp(x.lo, e)
        );
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

    BL_FORCE_INLINE constexpr bool fmod_u128_shift_exceeds_capacity(const fmod_u128& value, int bits)
    {
        return bits > 0 && !fmod_u128_is_zero(value) &&
            fmod_u128_bit_length(value) + bits > 128;
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

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_mod_shift_subtract_with_quotient_mod(
        fmod_u128 numerator,
        const fmod_u128& denominator,
        std::uint64_t& quotient_mod)
    {
        quotient_mod = 0;
        if (fmod_u128_is_zero(denominator))
            return {};
        if (fmod_u128_compare(numerator, denominator) < 0)
            return numerator;

        int shift = fmod_u128_bit_length(numerator) - fmod_u128_bit_length(denominator);
        fmod_u128 shifted = fmod_u128_shl_bits(denominator, shift);

        for (; shift >= 0; --shift)
        {
            if (fmod_u128_compare(numerator, shifted) >= 0)
            {
                fmod_u128_sub_inplace(numerator, shifted);
                if (shift < 31)
                    quotient_mod |= std::uint64_t{ 1 } << shift;
            }
            shifted = fmod_u128_shr_bits(shifted, 1);
        }

        return numerator;
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_double_mod(fmod_u128 value, const fmod_u128& modulus)
    {
        const bool overflow = (value.hi >> 63) != 0u;
        value = fmod_u128_shl1(value);
        if (overflow || fmod_u128_compare(value, modulus) >= 0)
            fmod_u128_sub_inplace(value, modulus);
        return value;
    }

    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_double_mod_with_quotient_bit(
        fmod_u128 value,
        const fmod_u128& modulus,
        std::uint64_t& bit)
    {
        const bool overflow = (value.hi >> 63) != 0u;
        value = fmod_u128_shl1(value);
        if (overflow || fmod_u128_compare(value, modulus) >= 0)
        {
            fmod_u128_sub_inplace(value, modulus);
            bit = 1;
        }
        else
        {
            bit = 0;
        }
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
    BL_FORCE_INLINE constexpr int fmod_append_expansion_term(double* terms, int count, double value) noexcept
    {
        if (value != 0.0)
            terms[count++] = value;
        return count;
    }

    BL_FORCE_INLINE constexpr int fmod_compress_expansion_zeroelim(int elen, const double* e, double* h) noexcept
    {
        if (elen <= 0)
            return 0;

        double g[16]{};
        double q = e[elen - 1];
        for (int i = elen - 2; i >= 0; --i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, e[i], q_new, low);
            q = q_new;
            g[i + 1] = low;
        }
        g[0] = q;

        int hindex = 0;
        q = g[0];
        for (int i = 1; i < elen; ++i)
        {
            double q_new{};
            double low{};
            two_sum_precise(q, g[i], q_new, low);
            if (low != 0.0)
                h[hindex++] = low;
            q = q_new;
        }

        if (q != 0.0 || hindex == 0)
            h[hindex++] = q;
        return hindex;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmod_add_scalar_precise(f128_s value, double scalar) noexcept
    {
        double s0{};
        double e0{};
        two_sum_precise(value.hi, scalar, s0, e0);

        double s1{};
        double e1{};
        two_sum_precise(value.lo, e0, s1, e1);

        return renorm(s0, s1 + e1);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmod_from_expansion_fast(const double* terms, int count) noexcept
    {
        if (count <= 0)
            return {};

        double compressed[16]{};
        const int compressed_count = fmod_compress_expansion_zeroelim(count, terms, compressed);

        f128_s sum{};
        for (int i = 0; i < compressed_count; ++i)
            sum = fmod_add_scalar_precise(sum, compressed[i]);

        return sum;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmod_from_expansion_direct(const double* terms, int count) noexcept
    {
        f128_s sum{};
        for (int i = 0; i < count; ++i)
            sum = fmod_add_scalar_precise(sum, terms[i]);
        return sum;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmod_sub_mul_scalar_compact(
        const f128_s& r,
        const f128_s& b,
        double q) noexcept
    {
        double p0{};
        double e0{};
        two_prod_precise(b.hi, q, p0, e0);

        double p1{};
        double e1{};
        two_prod_precise(b.lo, q, p1, e1);

        double s{};
        double t{};
        two_diff_precise(r.hi, p0, s, t);

        f128_s out{ s, t };
        out = fmod_add_scalar_precise(out, r.lo);
        out = fmod_add_scalar_precise(out, -e0);
        out = fmod_add_scalar_precise(out, -p1);
        out = fmod_add_scalar_precise(out, -e1);
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmod_sub_mul_scalar_expansion(
        const f128_s& r,
        const f128_s& b,
        double q) noexcept
    {
        double p0{};
        double e0{};
        two_prod_precise(b.hi, q, p0, e0);

        double p1{};
        double e1{};
        two_prod_precise(b.lo, q, p1, e1);

        double s{};
        double t{};
        two_diff_precise(r.hi, p0, s, t);

        double terms[6]{};
        int count = 0;
        count = fmod_append_expansion_term(terms, count, -e1);
        count = fmod_append_expansion_term(terms, count, -e0);
        count = fmod_append_expansion_term(terms, count, -p1);
        count = fmod_append_expansion_term(terms, count, r.lo);
        count = fmod_append_expansion_term(terms, count, t);
        count = fmod_append_expansion_term(terms, count, s);

        if (q < 0x1p47)
            return fmod_from_expansion_direct(terms, count);

        return fmod_from_expansion_fast(terms, count);
    }

    BL_FORCE_INLINE constexpr bool fmod_normalize_remainder(f128_s& r, const f128_s& modulus) noexcept
    {
        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
            {
                r = add_inline(r, modulus);
                continue;
            }

            if (r >= modulus)
            {
                r = sub_inline(r, modulus);
                continue;
            }

            return true;
        }

        return r >= 0.0 && r < modulus;
    }

    BL_FORCE_INLINE constexpr bool fmod_normalize_remainder_with_quotient(
        f128_s& r,
        const f128_s& modulus,
        std::uint64_t& quotient) noexcept
    {
        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
            {
                r = add_inline(r, modulus);
                if (quotient == 0u)
                    return false;
                --quotient;
                continue;
            }

            if (r >= modulus)
            {
                r = sub_inline(r, modulus);
                ++quotient;
                continue;
            }

            return true;
        }

        return r >= 0.0 && r < modulus;
    }

    BL_FORCE_INLINE constexpr int fmod_compare_remainder_to_half(const f128_s& r_abs, const f128_s& half) noexcept
    {
        const f128_s delta = sub_inline(r_abs, half);
        if (iszero(delta))
            return 0;
        return delta < 0.0 ? -1 : 1;
    }

    BL_FORCE_INLINE constexpr bool fmod_fast_small_quotient_abs_with_quotient(
        const f128_s& ax,
        const f128_s& ay,
        f128_s& out,
        std::uint64_t& quotient,
        bool allow_compact_residual = false) noexcept
    {
        if (!(ay.hi > 0.0) || detail::fp::isinf_or_nan(ay.hi) || !(ax >= ay))
            return false;

        const double q = detail::fp::trunc(ax.hi / ay.hi);
        if (!(q > 0.0) || q >= 0x1p53)
            return false;

        quotient = static_cast<std::uint64_t>(q);

        if (q < 0x1p48 && abs_double_is_power_of_two(q))
        {
            std::uint64_t cheap_quotient = quotient;
            f128_s r = sub_inline(ax, mul_pwr2_inline(ay, q));
            if (fmod_normalize_remainder_with_quotient(r, ay, cheap_quotient))
            {
                const f128_s edge_slack = mul_double_inline(ay, 0x1p-44);
                const f128_s half = mul_double_inline(ay, 0.5);
                const f128_s distance_to_half = mag(sub_inline(r, half));

                if (r > edge_slack &&
                    sub_inline(ay, r) > edge_slack &&
                    distance_to_half > edge_slack)
                {
                    out = r;
                    quotient = cheap_quotient;
                    return true;
                }
            }
        }

        if (allow_compact_residual)
        {
            std::uint64_t compact_quotient = quotient;
            f128_s r = fmod_sub_mul_scalar_compact(ax, ay, q);
            if (fmod_normalize_remainder_with_quotient(r, ay, compact_quotient))
            {
                out = r;
                quotient = compact_quotient;
                return true;
            }
        }

        f128_s r = fmod_sub_mul_scalar_expansion(ax, ay, q);
        if (!fmod_normalize_remainder_with_quotient(r, ay, quotient))
            return false;

        const f128_s edge_slack = mul_double_inline(ay, 0x1p-80);
        if (r <= edge_slack || sub_inline(ay, r) <= edge_slack)
            return false;

        if (allow_compact_residual)
        {
            const f128_s half = mul_double_inline(ay, 0.5);
            if (mag(sub_inline(r, half)) <= edge_slack)
                return false;
        }

        out = r;
        return true;
    }

    BL_FORCE_INLINE constexpr bool fmod_fast_small_quotient_abs(
        const f128_s& ax,
        const f128_s& ay,
        f128_s& out) noexcept
    {
        std::uint64_t quotient{};
        return fmod_fast_small_quotient_abs_with_quotient(ax, ay, out, quotient);
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
            if (fmod_u128_shift_exceeds_capacity(dy.mant, shift))
            {
                remainder = dx.mant;
            }
            else
            {
                const fmod_u128 denominator = fmod_u128_shl_bits(dy.mant, shift);
                remainder = fmod_u128_mod_shift_subtract(dx.mant, denominator);
            }
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

    BL_FORCE_INLINE constexpr f128_s fmod_reduced_or_exact(const f128_s& x, const f128_s& y)
    {
        const f128_s ay = mag(y);
        f128_s r = mag(x);

        constexpr int exact_reduction_exponent_gap = 64;
        if (frexp_exponent_limb(r.hi) - frexp_exponent_limb(ay.hi) > exact_reduction_exponent_gap)
            return fmod_exact_fixed_limb(x, y);

        for (int iteration = 0; iteration < 128 && r >= ay; ++iteration)
        {
            const int ex = frexp_exponent_limb(r.hi);
            const int ey = frexp_exponent_limb(ay.hi);
            int shift = ex - ey - 52;
            if (shift < 0)
                shift = 0;

            f128_s scaled = ldexp_terms(ay, shift);
            while (shift > 0 && scaled > r)
            {
                --shift;
                scaled = ldexp_terms(ay, shift);
            }

            if (!(scaled > 0.0) || scaled > r)
                return fmod_exact_fixed_limb(x, y);

            const double q = detail::fp::trunc(r.hi / scaled.hi);
            if (!(q > 0.0) || q >= 0x1p53)
                return fmod_exact_fixed_limb(x, y);

            r = fmod_sub_mul_scalar_expansion(r, scaled, q);
            if (!fmod_normalize_remainder(r, scaled))
                return fmod_exact_fixed_limb(x, y);
        }

        if (!fmod_normalize_remainder(r, ay))
            return fmod_exact_fixed_limb(x, y);

        if (iszero(r))
            return f128_s{ signbit(x.hi) ? -0.0 : 0.0 };

        return ispositive(x) ? r : -r;
    }

    BL_FORCE_INLINE constexpr f128_s fmod_exact_fixed_limb_abs_with_quotient_mod(
        const f128_s& ax,
        const f128_s& ay,
        std::uint64_t& quotient_mod)
    {
        constexpr std::uint64_t quotient_mask = 0x7fffffffull;

        const exact_dyadic_fmod dx = exact_from_f128_fmod(ax);
        const exact_dyadic_fmod dy = exact_from_f128_fmod(ay);

        fmod_u128 remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            if (fmod_u128_shift_exceeds_capacity(dy.mant, shift))
            {
                quotient_mod = 0;
                remainder = dx.mant;
            }
            else
            {
                const fmod_u128 denominator = fmod_u128_shl_bits(dy.mant, shift);
                remainder = fmod_u128_mod_shift_subtract_with_quotient_mod(dx.mant, denominator, quotient_mod);
            }
            out_exp = dx.exp2;
        }
        else
        {
            remainder = fmod_u128_mod_shift_subtract_with_quotient_mod(dx.mant, dy.mant, quotient_mod);
            const int shift = dx.exp2 - dy.exp2;

            int i = 0;
            for (; i < shift && !fmod_u128_is_zero(remainder); ++i)
            {
                std::uint64_t bit = 0;
                remainder = fmod_u128_double_mod_with_quotient_bit(remainder, dy.mant, bit);
                quotient_mod = ((quotient_mod << 1) | bit) & quotient_mask;
            }

            const int remaining = shift - i;
            if (remaining >= 31)
                quotient_mod = 0;
            else if (remaining > 0)
                quotient_mod = (quotient_mod << remaining) & quotient_mask;

            out_exp = dy.exp2;
        }

        f128_s out = exact_dyadic_to_f128_fmod(remainder, out_exp, false);
        if (iszero(out))
            return f128_s{ 0.0 };
        return out;
    }

    // decimal conversion
    BL_FORCE_INLINE constexpr bool try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = detail::_f128_impl::trunc(x);
        if (xi != x)
            return false;

        if (absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = sub_inline(xi, detail::_f128_impl::to_f128(hi_part));
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }

    BL_FORCE_INLINE constexpr bool is_odd_integer(const f128_s& x) noexcept
    {
        int64_t value{};
        if (try_get_int64(x, value))
            return (value & 1ll) != 0;

        if (x.lo != 0.0 || detail::fp::isinf_or_nan(x.hi))
            return false;

        return double_integer_is_odd(x.hi);
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
        if (!try_get_int64(integer_part, integer_value) || integer_value < 0)
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
        if (detail::fp::iszero_or_inf_or_nan(x.hi))
            return x;

        if (absd(x.hi) < 0x1p52)
        {
            const auto base = static_cast<long long>(x.hi);
            const double base_d = static_cast<double>(base);
            const double frac_hi = x.hi - base_d;
            const double frac_lo = x.lo;
            const double abs_frac_hi = absd(frac_hi);
            const double abs_frac_lo = absd(frac_lo);

            long long rounded = base;
            if (abs_frac_hi > 0.5 + abs_frac_lo)
            {
                rounded += (frac_hi < 0.0 || (frac_hi == 0.0 && signbit(frac_lo))) ? -1 : 1;
            }
            else if (abs_frac_hi >= 0.5 - abs_frac_lo)
            {
                const f128_s frac = sub_double_inline(x, base_d);
                if (mag(frac) >= f128_s{ 0.5 })
                    rounded += signbit(frac) ? -1 : 1;
            }

            f128_s out{ static_cast<double>(rounded), 0.0 };
            if (iszero(out))
                return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
            return out;
        }

        if (signbit(x))
        {
            f128_s y = -detail::_f128_impl::floor(add_inline(-x, f128_s{ 0.5 }));
            if (iszero(y))
                return f128_s{ -0.0, 0.0 };
            return y;
        }

        return detail::_f128_impl::floor(add_inline(x, f128_s{ 0.5 }));
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f128_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (detail::fp::isinf_or_nan(x.hi))
            return 0;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        const f128_s lo = detail::_f128_impl::to_f128(lo_i);
        const f128_s hi = detail::_f128_impl::to_f128(hi_i);

        if (x < lo || x > hi)
            return 0;

        std::int64_t out = 0;
        if (!try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr bool try_round_to_signed_integer(const f128_s& x, bool ties_to_even, SignedInt& out) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (detail::fp::isinf_or_nan(x.hi) || absd(x.hi) >= 0x1p52)
            return false;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        if (x < detail::_f128_impl::to_f128(lo_i) || x > detail::_f128_impl::to_f128(hi_i))
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

    // sqrt kernels
    [[nodiscard]] BL_FORCE_INLINE constexpr double sqrt_constexpr_head(double x) noexcept
    {
        double y = sqrt_seed(x);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        return 0.5 * (y + x / y);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double sqrt_tail_square(double c_lo, double correction) noexcept
    {
        #ifdef FMA_AVAILABLE
        if (!bl::detail::use_constexpr_math())
        {
            return std::fma(c_lo, c_lo, correction);
        }
        #endif

        return c_lo * c_lo + correction;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sqrt_compensated(const f128_s& scaled_a, double c) noexcept
    {
        const double c_hi = product_split_high(c);
        const double c_lo = c - c_hi;

        double q = c_hi * c_lo;
        q += q;

        const double p = c_hi * c_hi;
        const double u = p + q;
        const double uu = sqrt_tail_square(c_lo, (p - u) + q);
        const double cc = (((scaled_a.hi - u) - uu) + scaled_a.lo) / (c + c);

        const double y_hi = c + cc;
        return { y_hi, (c - y_hi) + cc };
    }

} // namespace detail::_f128

} // namespace bl

#endif
