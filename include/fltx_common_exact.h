
/**
 * fltx_common_exact.h — biguint logic needed by fltx_common_io.h
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */


#ifndef FLTX_COMMON_EXACT_INCLUDED
#define FLTX_COMMON_EXACT_INCLUDED

#include "fltx_common_math.h"

#include <algorithm>

namespace bl::detail::exact_decimal {

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
        const int writable_count = max_words - word_shift;
        const int src_count = std::min(old_size, writable_count);

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
            trim();
            return;
        }

        std::uint32_t out[max_words]{};

        std::uint32_t carry = 0;

		for (int i = 0; i < src_count; ++i)
		{
		    const int dst = word_shift + i;
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

[[nodiscard]] constexpr inline std::uint64_t div_quotient_limited(const biguint& numerator, const biguint& denominator, int quotient_bits, biguint& remainder) noexcept
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

    biguint remainder;
    const std::uint64_t first_chunk = div_quotient_limited(scaled, normalized_den, first_chunk_bits, remainder);

    biguint q{ first_chunk };
    for (int chunk_index = 1; chunk_index < chunk_count; ++chunk_index)
    {
        if (!remainder.is_zero())
            remainder.shl_bits(chunk_bits);

        const std::uint64_t chunk = div_quotient_limited(remainder, normalized_den, chunk_bits, remainder);
        q.shl_bits(chunk_bits);
        if (chunk != 0)
            q.add_inplace(biguint{ chunk });
    }

    if (!remainder.is_zero())
        remainder.shl_bits(2);

    const std::uint64_t extra_bits = div_quotient_limited(remainder, normalized_den, 2, remainder);
    const bool guard_bit = (extra_bits & 0x2u) != 0;
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

} // namespace bl::detail::exact_decimal

#endif