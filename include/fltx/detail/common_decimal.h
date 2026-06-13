/**
 * fltx/detail/common_decimal.h - Exact decimal conversion helpers for constexpr I/O.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_COMMON_DECIMAL_INCLUDED
#define FLTX_DETAIL_COMMON_DECIMAL_INCLUDED
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "fltx/detail/biguint.h"

namespace bl::detail::exact_decimal {

[[nodiscard]] constexpr inline biguint pow5_big(int exponent) noexcept
{
    constexpr std::uint32_t pow5_chunks[] = {
        1u,
        5u,
        25u,
        125u,
        625u,
        3125u,
        15625u,
        78125u,
        390625u,
        1953125u,
        9765625u,
        48828125u,
        244140625u,
        1220703125u
    };
    constexpr int chunk_exp = 13;

    biguint out{1};
    while (exponent >= chunk_exp)
    {
        out.mul_small(pow5_chunks[chunk_exp]);
        exponent -= chunk_exp;
    }

    if (exponent > 0)
        out.mul_small(pow5_chunks[exponent]);
    return out;
}

[[nodiscard]] constexpr inline biguint pow10_big(int exponent) noexcept
{
    biguint out{1};
    while (exponent >= 9)
    {
        out.mul_small(1000000000u);
        exponent -= 9;
    }

    constexpr std::uint32_t pow10_chunks[] = {
        1u,
        10u,
        100u,
        1000u,
        10000u,
        100000u,
        1000000u,
        10000000u,
        100000000u
    };
    if (exponent > 0)
        out.mul_small(pow10_chunks[exponent]);
    return out;
}

[[nodiscard]] constexpr inline biguint rounded_decimal_places_shift(
    const biguint& magnitude,
    int shift,
    int significand_bits) noexcept
{
    if (shift >= 0)
    {
        biguint out = magnitude;
        out.shl_bits(shift);
        return out;
    }

    const int right_shift = -shift;
    biguint out = shr_bits_copy(magnitude, right_shift);
    biguint remainder = low_bits_copy(magnitude, right_shift);

    biguint twice_remainder = remainder;
    twice_remainder.shl1();

    biguint denominator;
    denominator.set_bit(right_shift);

    const int cmp = twice_remainder.compare(denominator);
    const biguint delta = abs_diff(twice_remainder, denominator);
    const int tie_window_shift = significand_bits > 15 ? significand_bits - 15 : 0;
    const bool near_decimal_tie = le_power2(delta, right_shift - tie_window_shift);

    if (near_decimal_tie)
    {
        if (out.is_odd())
            out.add_small(1);
    }
    else if (cmp > 0)
    {
        out.add_small(1);
    }

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

constexpr inline void divmod_limited_quotient(
    const biguint& numerator,
    const biguint& denominator,
    int quotient_bits,
    biguint& quotient,
    biguint& remainder) noexcept
{
    quotient.clear();
    remainder = numerator;

    if (denominator.is_zero() || quotient_bits <= 0)
        return;

    const int num_bits = numerator.bit_length();
    const int den_bits = denominator.bit_length();
    if (num_bits == 0 || den_bits == 0 || num_bits < den_bits)
    {
        remainder.trim();
        return;
    }

    int shift = num_bits - den_bits;
    if (shift >= quotient_bits)
        shift = quotient_bits - 1;

    for (int i = shift; i >= 0; --i)
    {
        if (compare_shifted(remainder, denominator, i) >= 0)
        {
            sub_shifted_inplace(remainder, denominator, i);
            quotient.set_bit(i);
        }
    }

    quotient.trim();
    remainder.trim();
}

[[nodiscard]] constexpr inline int single_bit_index(const biguint& value) noexcept
{
    int bit = -1;
    for (int i = 0; i < value.size; ++i)
    {
        const std::uint32_t word = value.words[i];
        if (word == 0)
            continue;
        if ((word & (word - 1u)) != 0 || bit >= 0)
            return -1;
        bit = i * 32 + static_cast<int>(std::countr_zero(word));
    }
    return bit;
}

[[nodiscard]] constexpr inline biguint mul_by_correction_denominator(
    const biguint& denominator,
    const biguint& value) noexcept
{
    const int shift = single_bit_index(denominator);
    if (shift >= 0)
    {
        biguint out = value;
        out.shl_bits(shift);
        return out;
    }

    if (denominator.size == 1)
        return mul_small_u64_big(value, denominator.words[0]);

    return mul_big(denominator, value);
}

[[nodiscard]] constexpr inline bool divmod_from_floor_candidate(
    const biguint& numerator,
    const biguint& denominator,
    const biguint& candidate,
    int quotient_bits,
    biguint& quotient,
    biguint& remainder) noexcept
{
    if (denominator.is_zero() || candidate.is_zero())
        return false;

    if (candidate.bit_length() > quotient_bits + 4)
        return false;

    quotient = candidate;
    biguint product = mul_by_correction_denominator(denominator, quotient);
    int corrections = 0;

    while (product.compare(numerator) > 0)
    {
        if (quotient.is_zero())
            return false;

        quotient.sub_small(1);
        product.sub_inplace(denominator);

        if (++corrections > 64)
            return false;
    }

    for (;;)
    {
        biguint next_product = product;
        next_product.add_inplace(denominator);
        if (next_product.compare(numerator) > 0)
            break;

        product = next_product;
        quotient.add_small(1);

        if (++corrections > 64)
            return false;
    }

    remainder = numerator;
    remainder.sub_inplace(product);
    return true;
}

[[nodiscard]] constexpr inline std::uint64_t div_quotient_limited_bitwise(const biguint& numerator, const biguint& denominator, int quotient_bits, biguint& remainder) noexcept
{
    remainder = numerator;

    if (denominator.is_zero() || quotient_bits <= 0)
        return 0;

    const int num_bits = numerator.bit_length();
    const int den_bits = denominator.bit_length();
    if (num_bits == 0 || den_bits == 0 || num_bits < den_bits)
    {
        remainder.trim();
        return 0;
    }

    int shift = num_bits - den_bits;
    if (shift >= quotient_bits)
        shift = quotient_bits - 1;

    std::uint64_t quotient = 0;
    for (int i = shift; i >= 0; --i)
    {
        if (compare_shifted(remainder, denominator, i) >= 0)
        {
            sub_shifted_inplace(remainder, denominator, i);
            quotient |= (std::uint64_t{ 1 } << i);
        }
    }

    remainder.trim();
    return quotient;
}

[[nodiscard]] inline double leading_value_as_double(const biguint& value) noexcept
{
    const int bits = value.bit_length();
    if (bits <= 0)
        return 0.0;

    const int keep_bits = bits < 53 ? bits : 53;
    const std::uint64_t top = value.get_bits(bits - keep_bits, keep_bits);
    return std::ldexp(static_cast<double>(top), bits - keep_bits);
}

[[nodiscard]] inline bool div_quotient_limited_estimate(
    const biguint& numerator,
    const biguint& denominator,
    double denominator_leading,
    int quotient_bits,
    std::uint64_t& quotient,
    biguint& remainder) noexcept
{
    if (denominator.is_zero() || quotient_bits <= 0 || quotient_bits > 53)
        return false;

    const int num_bits = numerator.bit_length();
    const int den_bits = denominator.bit_length();
    if (num_bits == 0 || den_bits == 0 || num_bits < den_bits)
    {
        quotient = 0;
        remainder = numerator;
        remainder.trim();
        return true;
    }

    const std::uint64_t max_quotient =
        quotient_bits == 64 ? ~std::uint64_t{ 0 } : ((std::uint64_t{ 1 } << quotient_bits) - 1u);
    if (!(denominator_leading > 0.0))
        return false;

    const double approx = leading_value_as_double(numerator) / denominator_leading;
    if (!(approx >= 1.0))
        return false;

    std::uint64_t candidate = approx >= static_cast<double>(max_quotient)
        ? max_quotient
        : static_cast<std::uint64_t>(approx);
    if (candidate == 0)
        return false;

    biguint product = mul_small_u64_big(denominator, candidate);
    int corrections = 0;

    while (product.compare(numerator) > 0)
    {
        if (candidate == 0 || ++corrections > 64)
            return false;

        --candidate;
        product.sub_inplace(denominator);
    }

    for (;;)
    {
        biguint next_product = product;
        next_product.add_inplace(denominator);
        if (next_product.compare(numerator) > 0)
            break;

        if (candidate == max_quotient || ++corrections > 64)
            return false;

        ++candidate;
        product = next_product;
    }

    remainder = numerator;
    remainder.sub_inplace(product);
    quotient = candidate;
    return true;
}

[[nodiscard]] inline bool div_quotient_limited_estimate(
    const biguint& numerator,
    const biguint& denominator,
    int quotient_bits,
    std::uint64_t& quotient,
    biguint& remainder) noexcept
{
    return div_quotient_limited_estimate(
        numerator,
        denominator,
        leading_value_as_double(denominator),
        quotient_bits,
        quotient,
        remainder);
}

[[nodiscard]] constexpr inline std::uint64_t div_quotient_limited(
    const biguint& numerator,
    const biguint& denominator,
    double denominator_leading,
    int quotient_bits,
    biguint& remainder) noexcept
{
    if (!std::is_constant_evaluated())
    {
        std::uint64_t quotient = 0;
        if (div_quotient_limited_estimate(numerator, denominator, denominator_leading, quotient_bits, quotient, remainder))
            return quotient;
    }

    return div_quotient_limited_bitwise(numerator, denominator, quotient_bits, remainder);
}

[[nodiscard]] constexpr inline std::uint64_t div_quotient_limited(const biguint& numerator, const biguint& denominator, int quotient_bits, biguint& remainder) noexcept
{
    if (!std::is_constant_evaluated())
        return div_quotient_limited(
            numerator,
            denominator,
            leading_value_as_double(denominator),
            quotient_bits,
            remainder);

    return div_quotient_limited_bitwise(numerator, denominator, quotient_bits, remainder);
}

constexpr inline void divmod_limited_quotient_chunked(
    const biguint& numerator,
    const biguint& denominator,
    int quotient_bits,
    biguint& quotient,
    biguint& remainder) noexcept
{
    quotient.clear();
    remainder = numerator;

    if (denominator.is_zero() || quotient_bits <= 0)
        return;

    const int num_bits = numerator.bit_length();
    const int den_bits = denominator.bit_length();
    if (num_bits == 0 || den_bits == 0 || num_bits < den_bits)
    {
        remainder.trim();
        return;
    }

    int highest_shift = num_bits - den_bits;
    if (highest_shift >= quotient_bits)
        highest_shift = quotient_bits - 1;

    int remaining_bits = highest_shift + 1;
    constexpr int chunk_bits = 53;
    while (remaining_bits > 0)
    {
        int take_bits = remaining_bits % chunk_bits;
        if (take_bits == 0)
            take_bits = chunk_bits;

        remaining_bits -= take_bits;

        biguint chunk_denominator = denominator;
        if (remaining_bits > 0)
            chunk_denominator.shl_bits(remaining_bits);

        biguint chunk_remainder;
        const std::uint64_t chunk = div_quotient_limited(
            remainder,
            chunk_denominator,
            take_bits,
            chunk_remainder);

        quotient.shl_bits(take_bits);
        if (chunk != 0)
            quotient.add_inplace(biguint{ chunk });

        remainder = chunk_remainder;
    }

    quotient.trim();
    remainder.trim();
}

[[nodiscard]] constexpr inline biguint extract_rounded_significand_chunks(const biguint& numerator, const biguint& denominator, int ratio_exp, int significand_bits) noexcept
{
    const int chunk_bits = 53;
    const int chunk_count = (significand_bits + chunk_bits - 1) / chunk_bits;
    const int first_chunk_bits = significand_bits - (chunk_count - 1) * chunk_bits;
    const int first_scale_bits = first_chunk_bits - 1;

    biguint normalized_num = numerator;
    biguint normalized_den = denominator;
    if (ratio_exp >= 0)
        normalized_den.shl_bits(ratio_exp);
    else
        normalized_num.shl_bits(-ratio_exp);

    biguint scaled = normalized_num;
    if (first_scale_bits > 0)
        scaled.shl_bits(first_scale_bits);

    double normalized_den_leading = 0.0;
    if (!std::is_constant_evaluated())
        normalized_den_leading = leading_value_as_double(normalized_den);

    biguint remainder;
    const std::uint64_t first_chunk = div_quotient_limited(scaled, normalized_den, normalized_den_leading, first_chunk_bits, remainder);

    biguint q{ first_chunk };
    for (int chunk_index = 1; chunk_index < chunk_count; ++chunk_index)
    {
        if (!remainder.is_zero())
            remainder.shl_bits(chunk_bits);

        const std::uint64_t chunk = div_quotient_limited(remainder, normalized_den, normalized_den_leading, chunk_bits, remainder);
        q.shl_bits(chunk_bits);
        if (chunk != 0)
            q.add_inplace(biguint{ chunk });
    }

    if (!remainder.is_zero())
        remainder.shl_bits(2);

    const std::uint64_t extra_bits = div_quotient_limited(remainder, normalized_den, normalized_den_leading, 2, remainder);
    const bool guard_bit     = (extra_bits & 0x2u) != 0;
    const bool trailing_bits = (extra_bits & 0x1u) != 0 || !remainder.is_zero();

    if (guard_bit && (trailing_bits || q.is_odd()))
        q.add_small(1);

    q.trim();
    return q;
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

[[nodiscard]] constexpr inline int compare_scaled_with_pow10exp(const biguint& mag, int bin_exp, int dec_exp) noexcept
{
    if (dec_exp >= 0)
    {
        const biguint p5 = pow5_big(dec_exp);
        const int binary_shift = bin_exp - dec_exp;
        if (binary_shift >= 0)
            return compare(shifted(mag, binary_shift), p5);
        return compare(mag, shifted(p5, -binary_shift));
    }

    const int scale = -dec_exp;
    biguint lhs = mul_big(mag, pow5_big(scale));
    const int binary_shift = bin_exp + scale;

    if (binary_shift >= 0)
    {
        lhs.shl_bits(binary_shift);
        return compare(lhs, biguint{ 1 });
    }

    return compare(lhs, shifted(biguint{ 1 }, -binary_shift));
}

[[nodiscard]] constexpr inline std::uint64_t decompose_double_mantissa(double x, int& exponent, bool& neg) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    neg = (bits >> 63) != 0;
    const std::uint64_t frac = bits & ((std::uint64_t{ 1 } << 52) - 1);
    const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
    if (exp_bits == 0)
    {
        exponent = -1074;
        return frac;
    }
    exponent = static_cast<int>(exp_bits) - 1023 - 52;
    return (std::uint64_t{ 1 } << 52) | frac;
}

[[nodiscard]] constexpr inline int floor_to_int(double x) noexcept
{
    const int truncated = static_cast<int>(x);
    return static_cast<double>(truncated) > x ? truncated - 1 : truncated;
}

[[nodiscard]] constexpr inline int decimal_exponent_from_components(const biguint& magnitude, int common_exp) noexcept
{
    const double log10_2 = 0.30102999566398119521373889472449;
    int approx = floor_to_int((magnitude.bit_length() - 1 + common_exp) * log10_2);

    int guard = 0;
    while (compare_scaled_with_pow10exp(magnitude, common_exp, approx) < 0)
    {
        --approx;
        if (++guard > 16)
            break;
    }

    guard = 0;
    while (compare_scaled_with_pow10exp(magnitude, common_exp, approx + 1) >= 0)
    {
        ++approx;
        if (++guard > 16)
            break;
    }

    return approx;
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_binary_components(
    const typename Traits::value_type& x,
    biguint& magnitude,
    int& binary_exp,
    bool& neg)
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

        common_exp = (exponent < common_exp) ? exponent : common_exp;
        have_term = true;
    }

    if (!have_term)
    {
        magnitude.clear();
        binary_exp = 0;
        neg = false;
        return false;
    }

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

    magnitude = acc.mag;
    binary_exp = common_exp;
    neg = acc.neg;
    return !magnitude.is_zero();
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_decimal_places_integer(
    const typename Traits::value_type& x,
    int decimal_places,
    biguint& coefficient,
    bool& neg)
{
    int binary_exp = 0;
    biguint magnitude;
    if (!exact_binary_components<Traits>(x, magnitude, binary_exp, neg))
    {
        coefficient.clear();
        return false;
    }

    for (int i = 0; i < decimal_places; ++i)
        magnitude.mul_small(5);

    coefficient = rounded_decimal_places_shift(
        magnitude,
        binary_exp + decimal_places,
        std::numeric_limits<typename Traits::value_type>::digits);
    return true;
}

constexpr inline void significant_decimal_ratio(
    const biguint& magnitude,
    int common_exp,
    int scale10,
    biguint& numerator,
    biguint& denominator) noexcept
{
    numerator = magnitude;
    denominator = biguint{ 1 };

    if (scale10 >= -10 && scale10 <= 10)
    {
        if (common_exp >= 0)
            numerator.shl_bits(common_exp);
        else
            denominator.shl_bits(-common_exp);

        if (scale10 >= 0)
        {
            for (int i = 0; i < scale10; ++i)
                numerator.mul_small(10);
        }
        else
        {
            for (int i = 0; i < -scale10; ++i)
                denominator.mul_small(10);
        }
        return;
    }

    if (scale10 >= 0)
        numerator = mul_big(numerator, pow5_big(scale10));
    else
        denominator = pow5_big(-scale10);

    const int binary_shift = common_exp + scale10;
    if (binary_shift >= 0)
        numerator.shl_bits(binary_shift);
    else
        denominator.shl_bits(-binary_shift);
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_significant_decimal(
    const biguint& magnitude,
    int common_exp,
    int significant_figures,
    biguint& coefficient,
    int& exp10)
{
    const int scale10 = significant_figures - 1 - exp10;
    biguint num;
    biguint den;
    significant_decimal_ratio(magnitude, common_exp, scale10, num, den);

    const biguint limit = pow10_big(significant_figures);
    const int quotient_bits = limit.bit_length() + 1;

    biguint q;
    biguint r;
    const int den_shift = single_bit_index(den);
    if (den_shift >= 0)
    {
        q = shr_bits_copy(num, den_shift);
        r = low_bits_copy(num, den_shift);
    }
    else if (quotient_bits <= 64)
        q = biguint{ div_quotient_limited(num, den, quotient_bits, r) };
    else if (!std::is_constant_evaluated())
        divmod_limited_quotient_chunked(num, den, quotient_bits, q, r);
    else if (num.bit_length() > quotient_bits)
        divmod_limited_quotient(num, den, quotient_bits, q, r);
    else
        divmod_bitwise(num, den, q, r);

    if (!r.is_zero())
    {
        biguint twice_r = r;
        twice_r.shl1();
        const int cmp = compare(twice_r, den);
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    if (compare(q, limit) >= 0)
    {
        q.div_small(10);
        ++exp10;
    }

    coefficient = q;
    return true;
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_significant_decimal(
    const typename Traits::value_type& x,
    int significant_figures,
    biguint& coefficient,
    int& exp10)
{
    biguint magnitude;
    int common_exp = 0;
    bool neg = false;
    if (!exact_binary_components<Traits>(x, magnitude, common_exp, neg) || neg)
        return false;

    exp10 = decimal_exponent_from_components(magnitude, common_exp);

    return exact_significant_decimal<Traits>(magnitude, common_exp, significant_figures, coefficient, exp10);
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_significant_decimal_from_floor_candidate(
    const biguint& magnitude,
    int common_exp,
    int significant_figures,
    const biguint& floor_candidate,
    int& exp10,
    biguint& coefficient)
{
    const int scale10 = significant_figures - 1 - exp10;
    biguint num;
    biguint den;
    significant_decimal_ratio(magnitude, common_exp, scale10, num, den);

    const biguint limit = pow10_big(significant_figures);
    const int quotient_bits = limit.bit_length() + 1;

    biguint q;
    biguint r;
    if (!divmod_from_floor_candidate(num, den, floor_candidate, quotient_bits, q, r))
        return false;

    if (!r.is_zero())
    {
        biguint twice_r = r;
        twice_r.shl1();
        const int cmp = compare(twice_r, den);
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    if (compare(q, limit) >= 0)
    {
        q.div_small(10);
        ++exp10;
    }

    coefficient = q;
    return true;
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_significant_decimal_from_floor_candidate(
    const typename Traits::value_type& x,
    int significant_figures,
    const biguint& floor_candidate,
    int& exp10,
    biguint& coefficient)
{
    biguint magnitude;
    int common_exp = 0;
    bool neg = false;
    if (!exact_binary_components<Traits>(x, magnitude, common_exp, neg) || neg)
        return false;

    return exact_significant_decimal_from_floor_candidate<Traits>(
        magnitude,
        common_exp,
        significant_figures,
        floor_candidate,
        exp10,
        coefficient);
}

template<class Traits>
[[nodiscard]] constexpr inline bool exact_decimal_exponent(
    const typename Traits::value_type& x,
    int& exp10)
{
    biguint magnitude;
    int common_exp = 0;
    bool neg = false;
    if (!exact_binary_components<Traits>(x, magnitude, common_exp, neg) || neg)
        return false;

    exp10 = decimal_exponent_from_components(magnitude, common_exp);
    return true;
}

template<typename UInt>
constexpr inline int write_unsigned_decimal_rev(char* dst, UInt value) noexcept
{
    int len = 0;
    do
    {
        dst[len++] = static_cast<char>('0' + (value % 10));
        value /= 10;
    } while (value != 0);
    return len;
}

template<typename String, typename UInt>
constexpr inline void append_unsigned_decimal(String& out, UInt value)
{
    char buf[32];
    const int len = write_unsigned_decimal_rev(buf, value);
    for (int i = len - 1; i >= 0; --i)
        out.push_back(buf[i]);
}

template<typename String>
[[nodiscard]] constexpr inline String to_decimal_string(biguint value)
{
    if (value.is_zero())
        return "0";

    constexpr std::size_t chunk_capacity = (String::static_capacity + 8u) / 9u;
    std::uint32_t chunks[chunk_capacity];
    int chunk_count = 0;
    while (!value.is_zero())
    {
        if (chunk_count >= static_cast<int>(sizeof(chunks) / sizeof(chunks[0])))
            throw "static_string capacity exceeded";

        chunks[chunk_count++] = value.div_small(1000000000u);
    }

    String out;
    append_unsigned_decimal(out, chunks[chunk_count - 1]);
    for (int i = chunk_count - 2; i >= 0; --i)
    {
        char part_buf[16];
        const int part_len = write_unsigned_decimal_rev(part_buf, chunks[i]);
        out.append(static_cast<std::size_t>(9 - part_len), '0');
        for (int j = part_len - 1; j >= 0; --j)
            out.push_back(part_buf[j]);
    }
    return out;
}

template<class Traits, typename String>
[[nodiscard]] constexpr inline bool exact_scientific_digits(const typename Traits::value_type& x, int sig, String& digits, int& exp10)
{
    biguint coefficient;
    if (!exact_significant_decimal<Traits>(x, sig, coefficient, exp10))
        return false;

    digits = to_decimal_string<String>(coefficient);
    if (static_cast<int>(digits.size()) < sig)
    {
        const std::size_t zero_pad_count = static_cast<std::size_t>(sig - static_cast<int>(digits.size()));
        digits.insert(0, zero_pad_count, '0');
    }

    return true;
}

[[nodiscard]] constexpr inline bool pow5_u64(int exponent, std::uint64_t& out) noexcept
{
    if (exponent < 0)
        return false;

    constexpr std::uint64_t max_u64 = ~std::uint64_t{ 0 };
    std::uint64_t value = 1;
    for (int i = 0; i < exponent; ++i)
    {
        if (value > max_u64 / 5)
            return false;
        value *= 5;
    }

    out = value;
    return true;
}

[[nodiscard]] constexpr inline bool pow5_u32(int exponent, std::uint32_t& out) noexcept
{
    std::uint64_t value = 0;
    if (!pow5_u64(exponent, value) || value > static_cast<std::uint64_t>(~std::uint32_t{ 0 }))
        return false;

    out = static_cast<std::uint32_t>(value);
    return true;
}

[[nodiscard]] constexpr inline int bit_length_u64(std::uint64_t value) noexcept
{
    int bits = 0;
    while (value != 0)
    {
        ++bits;
        value >>= 1;
    }
    return bits;
}

[[nodiscard]] constexpr inline int floor_log2_ratio_u64(std::uint64_t numerator, std::uint64_t denominator) noexcept
{
    int k = bit_length_u64(numerator) - bit_length_u64(denominator);

    if (k >= 0)
    {
        const std::uint64_t shifted_den = denominator << k;
        if (numerator < shifted_den)
            --k;
    }
    else
    {
        const std::uint64_t shifted_num = numerator << -k;
        if (shifted_num < denominator)
            --k;
    }

    return k;
}

template<class Traits>
[[nodiscard]] constexpr inline int decimal_conversion_significand_bits() noexcept
{
    if constexpr (requires { Traits::conversion_significand_bits; })
        return Traits::conversion_significand_bits;
    else
        return Traits::significand_bits;
}

template<class Traits>
constexpr inline bool compact_decimal_to_value(std::uint64_t coeff, int dec_exp, bool neg, typename Traits::value_type& out) noexcept
{
    if (coeff == 0)
    {
        out = Traits::zero(neg);
        return true;
    }

    std::uint64_t numerator = coeff;
    std::uint32_t denominator = 1;
    int bin_exp = 0;

    if (dec_exp >= 0)
    {
        std::uint64_t scale5 = 0;
        if (!pow5_u64(dec_exp, scale5) || numerator > (~std::uint64_t{ 0 }) / scale5)
            return false;

        numerator *= scale5;
        bin_exp = dec_exp;
    }
    else
    {
        const int original_pow10 = -dec_exp;
        int pow5 = original_pow10;
        while (pow5 > 0 && (numerator % 5) == 0)
        {
            numerator /= 5;
            --pow5;
        }

        if (!pow5_u32(pow5, denominator))
            return false;

        bin_exp = -original_pow10;
    }

    const int ratio_exp  = floor_log2_ratio_u64(numerator, denominator);
    const int conversion_bits = decimal_conversion_significand_bits<Traits>();
    const int scale_bits = conversion_bits - 1 - ratio_exp;
    if (scale_bits < 0)
        return false;

    biguint q{ numerator };
    q.shl_bits(scale_bits);

    std::uint32_t remainder = 0;
    if (denominator != 1)
        remainder = q.div_small(denominator);

    if (remainder != 0)
    {
        const std::uint64_t twice_remainder = static_cast<std::uint64_t>(remainder) << 1;
        const int cmp = twice_remainder < denominator ? -1 : twice_remainder > denominator ? 1 : 0;
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    int adjusted_ratio_exp = ratio_exp;
    if (q.bit_length() > conversion_bits)
    {
        q.shr1();
        ++adjusted_ratio_exp;
    }

    const int e2 = bin_exp + adjusted_ratio_exp;
    if (e2 > Traits::max_binary_exponent)
        out = Traits::infinity(neg);
    else if (e2 < Traits::min_binary_exponent)
        out = Traits::zero(neg);
    else
        out = Traits::pack_from_significand(q, e2, neg);

    return true;
}

template<class Traits>
constexpr inline typename Traits::value_type exact_binary_integer_to_value(biguint q, int bin_exp, bool neg) noexcept
{
    if (q.is_zero())
        return Traits::zero(neg);

    const int conversion_bits = decimal_conversion_significand_bits<Traits>();
    int ratio_exp = q.bit_length() - 1;
    if (ratio_exp > conversion_bits - 1)
    {
        const int shift = ratio_exp - (conversion_bits - 1);
        const bool round_bit = q.get_bit(shift - 1);
        const bool sticky = shift > 1 && any_low_bits_set(q, shift - 1);

        q = shr_bits_copy(q, shift);
        if (round_bit && (sticky || q.is_odd()))
            q.add_small(1);

        if (q.bit_length() > conversion_bits)
        {
            q.shr1();
            ++ratio_exp;
        }
    }
    else if (ratio_exp < conversion_bits - 1)
    {
        q.shl_bits((conversion_bits - 1) - ratio_exp);
    }

    const int e2 = bin_exp + ratio_exp;
    if (e2 > Traits::max_binary_exponent)
        return Traits::infinity(neg);
    if (e2 < Traits::min_binary_exponent)
        return Traits::zero(neg);

    return Traits::pack_from_significand(q, e2, neg);
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
        return exact_binary_integer_to_value<Traits>(numerator, dec_exp, neg);
    }
    else
    {
        denominator = pow5_big(-dec_exp);
        bin_exp = dec_exp;
    }

    int ratio_exp = floor_log2_ratio(numerator, denominator);
    const int conversion_bits = decimal_conversion_significand_bits<Traits>();

    biguint q = extract_rounded_significand_chunks(numerator, denominator, ratio_exp, conversion_bits);
    if (q.bit_length() > conversion_bits)
    {
        q.shr1();
        ++ratio_exp;
    }

    const int e2 = bin_exp + ratio_exp;
    if (e2 > Traits::max_binary_exponent)
        return Traits::infinity(neg);
    if (e2 < Traits::min_binary_exponent)
        return Traits::zero(neg);

    return Traits::pack_from_significand(q, e2, neg);
}

} // namespace bl::detail::exact_decimal

#endif
