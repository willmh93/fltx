/**
 * fltx/detail/biguint.h - Fixed-capacity constexpr unsigned big-integer helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_BIGUINT_INCLUDED
#define FLTX_DETAIL_BIGUINT_INCLUDED
#include <cstdint>
#include <type_traits>

namespace bl::detail::exact_decimal {

struct biguint
{
    // Decimal parse/format/rounding paths stay well below this for canonical
    // f128/f256 values. The limiting use is f256 Payne-Hanek reduction: the
    // 2048-bit 2/pi table plus the low-limb span of a canonical f256 input
    // needs roughly 2208 low product bits near the large-argument threshold.
    static constexpr int max_words = 69;

    std::uint32_t words[max_words];
    int size = 0;

    constexpr biguint() noexcept
    {
    }

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

    constexpr biguint(const biguint& other) noexcept
        : size(other.size)
    {
        for (int i = 0; i < size; ++i)
            words[i] = other.words[i];

        if (std::is_constant_evaluated())
        {
            for (int i = size; i < max_words; ++i)
                words[i] = 0;
        }
    }

    constexpr biguint& operator=(const biguint& other) noexcept
    {
        if (this == &other)
            return *this;

        size = other.size;
        for (int i = 0; i < size; ++i)
            words[i] = other.words[i];

        if (std::is_constant_evaluated())
        {
            for (int i = size; i < max_words; ++i)
                words[i] = 0;
        }

        return *this;
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
        int out_shift = 0;
        while (count > 0 && out_shift < 64)
        {
            const int word_index = start >> 5;
            const int bit_offset = start & 31;
            const int available = 32 - bit_offset;
            int take = count < available ? count : available;
            if (take > 64 - out_shift)
                take = 64 - out_shift;

            if (word_index < size)
            {
                const std::uint32_t mask = take == 32
                    ? ~std::uint32_t{ 0 }
                    : ((std::uint32_t{ 1 } << take) - 1u);
                const std::uint32_t chunk = static_cast<std::uint32_t>((words[word_index] >> bit_offset) & mask);
                value |= static_cast<std::uint64_t>(chunk) << out_shift;
            }

            start += take;
            out_shift += take;
            count -= take;
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

    constexpr void sub_small(std::uint32_t value) noexcept
    {
        std::uint64_t borrow = value;
        int i = 0;

        while (borrow != 0 && i < size)
        {
            const std::uint64_t word = words[i];
            if (word < borrow)
            {
                words[i] = static_cast<std::uint32_t>((std::uint64_t{ 1 } << 32) + word - borrow);
                borrow = 1;
            }
            else
            {
                words[i] = static_cast<std::uint32_t>(word - borrow);
                borrow = 0;
            }
            ++i;
        }

        trim();
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
        const int bit_shift  = bits & 31;

        if (word_shift >= max_words)
        {
            size = 0;
            return;
        }

        const int old_size = size;
        const int writable_count = max_words - word_shift;
        const int src_count = (old_size < writable_count) ? old_size : writable_count;

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

        const std::uint32_t top_carry = static_cast<std::uint32_t>(words[src_count - 1] >> (32 - bit_shift));
        for (int i = src_count - 1; i >= 0; --i)
        {
            const int dst = word_shift + i;
            const std::uint32_t low = (i > 0)
                ? static_cast<std::uint32_t>(words[i - 1] >> (32 - bit_shift))
                : 0u;
            words[dst] = static_cast<std::uint32_t>((words[i] << bit_shift) | low);
        }

        for (int i = 0; i < word_shift; ++i)
            words[i] = 0;

        int new_size = src_count + word_shift;
        if (top_carry != 0 && new_size < max_words)
        {
            words[new_size] = top_carry;
            ++new_size;
        }

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
    const int bit_shift  = bits & 31;
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
        const std::uint64_t bi  = shifted_word_at(b, i, bits);
        const std::uint64_t sub = bi + borrow;
        const std::uint64_t ai  = a.words[i];
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

    const int product_size = a.size + b.size;
    out.size = (product_size < biguint::max_words) ? product_size : biguint::max_words;
    for (int i = 0; i < out.size; ++i)
        out.words[i] = 0;

    for (int i = 0; i < a.size; ++i)
    {
        std::uint64_t carry = 0;
        const int writable_count = biguint::max_words - i;
        const int jmax = (b.size < writable_count) ? b.size : writable_count;
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

[[nodiscard]] constexpr inline biguint mul_small_u64_big(const biguint& value, std::uint64_t multiplier) noexcept
{
    if (value.is_zero() || multiplier == 0)
        return {};

    constexpr std::uint64_t max_u32 = ~std::uint64_t{ 0 } >> 32;
    if (multiplier <= max_u32)
    {
        biguint out = value;
        out.mul_small(static_cast<std::uint32_t>(multiplier));
        return out;
    }

    biguint low = value;
    low.mul_small(static_cast<std::uint32_t>(multiplier));

    biguint high = value;
    high.mul_small(static_cast<std::uint32_t>(multiplier >> 32));
    high.shl_bits(32);

    low.add_inplace(high);
    return low;
}

[[nodiscard]] constexpr inline bool any_low_bits_set(const biguint& value, int bit_count) noexcept
{
    if (bit_count <= 0 || value.is_zero())
        return false;

    const int full_words = bit_count >> 5;
    const int rem_bits = bit_count & 31;

    const int checked_words = full_words < value.size ? full_words : value.size;
    for (int i = 0; i < checked_words; ++i)
    {
        if (value.words[i] != 0)
            return true;
    }

    if (rem_bits == 0 || full_words >= value.size)
        return false;

    const std::uint32_t mask = (std::uint32_t{ 1 } << rem_bits) - 1u;
    return (value.words[full_words] & mask) != 0;
}

[[nodiscard]] constexpr inline biguint shr_bits_copy(const biguint& value, int bits) noexcept
{
    if (bits <= 0 || value.is_zero())
        return value;

    const int word_shift = bits >> 5;
    if (word_shift >= value.size)
        return {};

    const int bit_shift = bits & 31;
    biguint out;
    out.size = value.size - word_shift;

    for (int i = 0; i < out.size; ++i)
    {
        const int src = i + word_shift;
        std::uint32_t word = value.words[src] >> bit_shift;
        if (bit_shift != 0 && src + 1 < value.size)
            word |= value.words[src + 1] << (32 - bit_shift);
        out.words[i] = word;
    }

    out.trim();
    return out;
}

[[nodiscard]] constexpr inline biguint low_bits_copy(const biguint& value, int bits) noexcept
{
    if (bits <= 0 || value.is_zero())
        return {};

    const int word_count = (bits + 31) >> 5;
    biguint out;
    out.size = value.size < word_count ? value.size : word_count;

    for (int i = 0; i < out.size; ++i)
        out.words[i] = value.words[i];

    const int rem_bits = bits & 31;
    if (rem_bits != 0 && out.size == word_count)
        out.words[out.size - 1] &= (std::uint32_t{ 1 } << rem_bits) - 1u;

    out.trim();
    return out;
}

[[nodiscard]] constexpr inline biguint abs_diff(biguint a, const biguint& b) noexcept
{
    if (a.compare(b) >= 0)
    {
        a.sub_inplace(b);
        return a;
    }

    biguint out = b;
    out.sub_inplace(a);
    return out;
}

[[nodiscard]] constexpr inline bool le_power2(const biguint& value, int bit) noexcept
{
    if (value.is_zero())
        return true;
    if (bit < 0)
        return false;

    const int highest = value.bit_length() - 1;
    if (highest != bit)
        return highest < bit;

    biguint limit;
    limit.set_bit(bit);
    return value.compare(limit) <= 0;
}

[[nodiscard]] constexpr inline biguint from_words(const std::uint32_t* words, int count) noexcept
{
    biguint out{};
    const int copy_count = count < biguint::max_words ? count : biguint::max_words;
    for (int i = 0; i < copy_count; ++i)
        out.words[i] = words[i];
    out.size = copy_count;
    out.trim();
    return out;
}

} // namespace bl::detail::exact_decimal

#endif
