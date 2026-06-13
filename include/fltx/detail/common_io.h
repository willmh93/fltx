/**
 * fltx/detail/common_io.h - Shared constexpr formatting and parsing support.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_COMMON_IO_INCLUDED
#define FLTX_DETAIL_COMMON_IO_INCLUDED
#include <cstdint>
#include <ios>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fltx/detail/ascii.h"
#include "fltx/detail/common_decimal.h"
#include "fltx/detail/common_fp.h"
#include "fltx/detail/format_flags.h"
#include "fltx/detail/static_string.h"

namespace bl::detail {

[[nodiscard]] constexpr std::uint32_t bounded_floor_to_u32(double value, std::uint32_t max_value) noexcept
{
    if (!(value > 0.0))
        return 0;
    if (value >= static_cast<double>(max_value))
        return max_value;

    if (!std::is_constant_evaluated())
        return static_cast<std::uint32_t>(value);

    std::uint32_t out = 0;
    for (std::uint32_t place = 1000000000u; place > 0; place /= 10u)
    {
        while (value >= static_cast<double>(place))
        {
            value -= static_cast<double>(place);
            out += place;
        }
    }
    return out;
}

BL_FORCE_INLINE constexpr int append_uint32_rev(char* dst, std::uint32_t value) noexcept
{
    int len = 0;
    if (value == 0)
    {
        dst[len++] = '0';
        return len;
    }

    while (value > 0)
    {
        const std::uint32_t next = value / 10u;
        const std::uint32_t digit = value - next * 10u;
        dst[len++] = static_cast<char>('0' + digit);
        value = next;
    }
    return len;
}

enum class format_kind : unsigned char { general, fixed_frac, scientific_frac, scientific_sig, hex_float };

inline constexpr int fltx_max_parse_order = 330;
inline constexpr int fltx_min_parse_order = -400;

struct hybrid_parse_token
{
    std::uint64_t coeff = 0;
    exact_decimal::biguint bounded_coeff;
    int frac_digits = 0;
    int kept_sig_digits = 0;
    int sig_digits = 0;
    int exp10 = 0;
    int first_discarded_digit = -1;
    bool any_digit = false;
    bool seen_nonzero = false;
    bool coeff_overflow = false;
    bool discarded_nonzero = false;
};

struct fltx_char_result
{
    char* ptr = nullptr;
    bool ok = false;
};

template<typename String, typename Writer>
BL_MSVC_NOINLINE constexpr void write_chars_to_string(String& out, std::size_t cap, Writer&& writer)
{
    out.resize(cap);
    char* first = out.data();
    auto r = std::forward<Writer>(writer)(first, first + out.size());
    if (!r.ok)
    {
        out.clear();
        return;
    }
    out.resize(static_cast<std::size_t>(r.ptr - first));
}

BL_FORCE_INLINE constexpr void copy_chars(char* dst, const char* src, std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i)
        dst[i] = src[i];
}

BL_FORCE_INLINE constexpr fltx_char_result emit_fixed_zero_to_chars(char* first, char* last, bool neg, int precision, bool strip_trailing_zeros) noexcept
{
    if (precision < 0)
        precision = 0;

    const int frac_len = strip_trailing_zeros ? 0 : precision;
    const std::size_t needed = static_cast<std::size_t>(neg ? 1 : 0) + 1u + (frac_len > 0 ? static_cast<std::size_t>(1 + frac_len) : 0u);
    if (static_cast<std::size_t>(last - first) < needed)
        return { first, false };

    char* p = first;
    if (neg)
        *p++ = '-';

    *p++ = '0';
    if (frac_len > 0)
    {
        *p++ = '.';
        for (int i = 0; i < frac_len; ++i)
            *p++ = '0';
    }

    return { p, true };
}

BL_FORCE_INLINE constexpr fltx_char_result append_exp10_to_chars(char* p, char* end, int e10) noexcept
{
    if (p >= end) return { p, false };
    *p++ = 'e';

    if (p >= end) return { p, false };
    if (e10 < 0) { *p++ = '-'; e10 = -e10; }
    else { *p++ = '+'; }

    char buf[8];
    int n = 0;
    do {
        buf[n++] = char('0' + (e10 % 10));
        e10 /= 10;
    } while (e10);

    if (n < 2) buf[n++] = '0';

    if (p + n > end) return { p, false };
    for (int i = n - 1; i >= 0; --i) *p++ = buf[i];

    return { p, true };
}

BL_FORCE_INLINE constexpr fltx_char_result emit_single_zero_to_chars(char* first, char* last) noexcept
{
    if (first >= last)
        return { first, false };

    *first = '0';
    return { first + 1, true };
}

BL_FORCE_INLINE constexpr fltx_char_result emit_scientific_zero_to_chars(
    char* first,
    char* last,
    bool neg,
    int frac_digits,
    bool strip_trailing_zeros) noexcept
{
    if (frac_digits < 0)
        frac_digits = 0;

    const int frac_len = strip_trailing_zeros ? 0 : frac_digits;

    const std::size_t needed =
        static_cast<std::size_t>(neg ? 1 : 0) +
        1u +
        (frac_len > 0 ? static_cast<std::size_t>(1 + frac_len) : 0u) +
        4u;

    if (static_cast<std::size_t>(last - first) < needed)
        return { first, false };

    char* p = first;
    if (neg)
        *p++ = '-';

    *p++ = '0';
    if (frac_len > 0)
    {
        *p++ = '.';
        for (int i = 0; i < frac_len; ++i)
            *p++ = '0';
    }

    copy_chars(p, "e+00", 4u);
    p += 4;
    return { p, true };
}

BL_FORCE_INLINE constexpr fltx_char_result emit_scientific_digits_to_chars(
    char* first,
    char* last,
    bool neg,
    const char* digits,
    int sig_digits,
    int exp10,
    bool strip_trailing_zeros) noexcept
{
    if (sig_digits < 1)
        sig_digits = 1;

    int last_frac = sig_digits - 1;
    if (sig_digits > 1 && strip_trailing_zeros)
    {
        while (last_frac >= 1 && digits[last_frac] == '0')
            --last_frac;
    }

    char exp_buf[16];
    auto er = append_exp10_to_chars(exp_buf, exp_buf + sizeof(exp_buf), exp10);
    if (!er.ok)
        return { first, false };

    const int exp_len = static_cast<int>(er.ptr - exp_buf);
    const bool has_frac = sig_digits > 1 && last_frac >= 1;
    const std::size_t needed =
        static_cast<std::size_t>(neg ? 1 : 0) +
        1u +
        (has_frac ? static_cast<std::size_t>(1 + last_frac) : 0u) +
        static_cast<std::size_t>(exp_len);

    if (static_cast<std::size_t>(last - first) < needed)
        return { first, false };

    char* p = first;
    if (neg)
        *p++ = '-';

    *p++ = digits[0];
    if (has_frac)
    {
        *p++ = '.';
        copy_chars(p, digits + 1, static_cast<std::size_t>(last_frac));
        p += last_frac;
    }

    copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
    p += exp_len;
    return { p, true };
}

template<class Traits>
[[nodiscard]] constexpr fltx_char_result emit_scientific_sig_for_traits(
    char* first,
    char* last,
    const typename Traits::value_type& x,
    int sig_digits,
    bool strip_trailing_zeros) noexcept
{
    if (Traits::iszero(x))
        return emit_single_zero_to_chars(first, last);

    if (sig_digits < 1)
        sig_digits = 1;

    const bool neg = Traits::is_negative(x);
    const typename Traits::value_type v = neg ? -x : x;

    typename Traits::string_type digits;
    int e10 = 0;
    if (!Traits::scientific_digits(v, sig_digits, digits, e10))
        return emit_single_zero_to_chars(first, last);

    return emit_scientific_digits_to_chars(
        first,
        last,
        neg,
        digits.data(),
        sig_digits,
        e10,
        strip_trailing_zeros);
}

template<class Traits>
[[nodiscard]] constexpr fltx_char_result emit_scientific_frac_for_traits(
    char* first,
    char* last,
    const typename Traits::value_type& x,
    int frac_digits,
    bool strip_trailing_zeros) noexcept
{
    if (frac_digits < 0)
        frac_digits = 0;

    if (Traits::iszero(x))
        return emit_scientific_zero_to_chars(
            first,
            last,
            Traits::is_negative(x),
            frac_digits,
            strip_trailing_zeros);

    return Traits::to_chars_scientific_sig(first, last, x, frac_digits + 1, strip_trailing_zeros);
}

BL_FORCE_INLINE constexpr char lower_hex_digit(unsigned value) noexcept
{
    value &= 0xfu;
    return static_cast<char>(value < 10u ? ('0' + value) : ('a' + (value - 10u)));
}

BL_FORCE_INLINE constexpr fltx_char_result append_exp2_to_chars(char* p, char* end, int e2) noexcept
{
    if (p >= end) return { p, false };
    *p++ = 'p';

    if (p >= end) return { p, false };
    if (e2 < 0) { *p++ = '-'; e2 = -e2; }
    else { *p++ = '+'; }

    char buf[8];
    int n = 0;
    do {
        buf[n++] = char('0' + (e2 % 10));
        e2 /= 10;
    } while (e2);

    if (p + n > end) return { p, false };
    for (int i = n - 1; i >= 0; --i) *p++ = buf[i];

    return { p, true };
}

[[nodiscard]] constexpr inline exact_decimal::biguint rounded_scaled_binary(
    const exact_decimal::biguint& magnitude,
    int scale) noexcept
{
    if (scale >= 0)
    {
        exact_decimal::biguint out = magnitude;
        out.shl_bits(scale);
        return out;
    }

    const int shift = -scale;
    exact_decimal::biguint out = exact_decimal::shr_bits_copy(magnitude, shift);
    const bool guard = magnitude.get_bit(shift - 1);
    const bool sticky = shift > 1 && exact_decimal::any_low_bits_set(magnitude, shift - 1);
    if (guard && (sticky || out.is_odd()))
        out.add_small(1);
    return out;
}

[[nodiscard]] constexpr inline unsigned hex_nibble_at(const exact_decimal::biguint& value, int nibble_index) noexcept
{
    return static_cast<unsigned>(value.get_bits(nibble_index * 4, 4));
}

template<class Traits>
[[nodiscard]] BL_FORCE_INLINE constexpr int hex_nominal_fraction_digits() noexcept
{
    return (Traits::significand_bits - 1 + 3) / 4;
}

template<class Traits>
[[nodiscard]] BL_FORCE_INLINE constexpr int hex_full_significand_bits() noexcept
{
    if constexpr (requires { Traits::conversion_significand_bits; })
        return Traits::conversion_significand_bits;
    else if constexpr (Traits::limb_count > 1)
        return (Traits::limb_count + 1) * 53;
    else
        return Traits::significand_bits;
}

template<class Traits>
[[nodiscard]] BL_FORCE_INLINE constexpr int hex_max_fraction_digits() noexcept
{
    return (hex_full_significand_bits<Traits>() - 1 + 3) / 4;
}

template<class Traits>
[[nodiscard]] constexpr fltx_char_result emit_hex_float_to_chars(
    char* first,
    char* last,
    const typename Traits::value_type& x,
    bool prefix,
    int precision,
    bool precision_specified,
    bool strip_trailing_zeros) noexcept
{
    int fraction_digits = precision_specified ? precision : hex_nominal_fraction_digits<Traits>();
    if (fraction_digits < 0)
        fraction_digits = 0;

    const bool zero = Traits::iszero(x);
    bool neg = Traits::is_negative(x);

    exact_decimal::biguint scaled;
    int hex_exp = 0;
    if (!zero)
    {
        exact_decimal::biguint magnitude;
        int bin_exp = 0;
        if (!exact_decimal::exact_binary_components<Traits>(x, magnitude, bin_exp, neg))
            return { first, false };

        if (!precision_specified)
        {
            const int exact_fraction_digits = (magnitude.bit_length() - 1 + 3) / 4;
            const int max_fraction_digits = hex_max_fraction_digits<Traits>();
            if (exact_fraction_digits > fraction_digits)
                fraction_digits = exact_fraction_digits < max_fraction_digits ? exact_fraction_digits : max_fraction_digits;
        }

        const int top_exp = bin_exp + magnitude.bit_length() - 1;
        hex_exp = top_exp < Traits::min_normal_binary_exponent ? Traits::min_normal_binary_exponent : top_exp;
        const int scale = bin_exp - hex_exp + 4 * fraction_digits;
        scaled = rounded_scaled_binary(magnitude, scale);
    }

    int emitted_fraction_digits = fraction_digits;
    if (strip_trailing_zeros)
    {
        while (emitted_fraction_digits > 0 &&
               hex_nibble_at(scaled, fraction_digits - emitted_fraction_digits) == 0u)
        {
            --emitted_fraction_digits;
        }
    }

    char exp_buf[16];
    auto exp_result = append_exp2_to_chars(exp_buf, exp_buf + sizeof(exp_buf), hex_exp);
    if (!exp_result.ok)
        return { first, false };
    const int exp_len = static_cast<int>(exp_result.ptr - exp_buf);

    const unsigned int_digit = zero ? 0u : hex_nibble_at(scaled, fraction_digits);
    const std::size_t needed = static_cast<std::size_t>(neg ? 1 : 0)
        + static_cast<std::size_t>(prefix ? 2 : 0)
        + 1u
        + static_cast<std::size_t>(emitted_fraction_digits > 0 ? 1 + emitted_fraction_digits : 0)
        + static_cast<std::size_t>(exp_len);
    if (static_cast<std::size_t>(last - first) < needed)
        return { last, false };

    char* p = first;
    if (neg)
        *p++ = '-';
    if (prefix)
    {
        *p++ = '0';
        *p++ = 'x';
    }

    *p++ = lower_hex_digit(int_digit);
    if (emitted_fraction_digits > 0)
    {
        *p++ = '.';
        for (int i = 0; i < emitted_fraction_digits; ++i)
        {
            const int nibble_index = fraction_digits - 1 - i;
            *p++ = lower_hex_digit(hex_nibble_at(scaled, nibble_index));
        }
    }

    copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
    p += exp_len;
    return { p, true };
}

constexpr inline std::size_t find_exponent_marker(std::string_view text) noexcept
{
    for (std::size_t index = 0; index < text.size(); ++index)
    {
        if (text[index] == 'e' || text[index] == 'E' ||
            text[index] == 'p' || text[index] == 'P')
        {
            return index;
        }
    }
    return std::string_view::npos;
}

template<typename String>
inline BL_MSVC_NOINLINE constexpr void ensure_decimal_point(String& s)
{
    const std::string_view view(s.data(), s.size());
    const std::size_t e = find_exponent_marker(view);
    const std::size_t d = view.find('.');
    if (d != std::string_view::npos && (e == std::string_view::npos || d < e))
        return;
    if (e == std::string_view::npos)
        s.push_back('.');
    else
        s.insert(e, ".");
}

template<typename String>
inline BL_MSVC_NOINLINE constexpr void apply_stream_decorations(String& s, bool showpos, bool uppercase)
{
    if (showpos && (s.empty() || s[0] != '-'))
        s.insert(0, 1, '+');
    if (!uppercase)
        return;
    for (char& c : s)
    {
        if (c == 'e')
            c = 'E';
        else if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - ('a' - 'A'));
    }
}

template<class Traits>
constexpr inline const char* special_text(const typename Traits::value_type& x, bool uppercase = false) noexcept
{
    if (Traits::isnan(x))
        return uppercase ? "NAN" : "nan";
    if (!Traits::isinf(x))
        return nullptr;
    return Traits::is_negative(x) ? (uppercase ? "-INF" : "-inf") : (uppercase ? "INF" : "inf");
}

template<class Traits, typename String>
constexpr inline bool assign_special_string(String& out, const typename Traits::value_type& x, bool uppercase = false) noexcept
{
    if (const char* text = special_text<Traits>(x, uppercase)) [[unlikely]]
    {
        out = text;
        return true;
    }
    return false;
}

template<class Traits>
[[nodiscard]] BL_FORCE_INLINE constexpr std::size_t format_capacity(int precision, format_kind kind) noexcept
{
    constexpr std::size_t sign_chars = 1u;
    constexpr std::size_t exp10_chars = 5u; // e+00 through e-400
    constexpr std::size_t exp2_chars = 6u;  // p+0 through p-1074

    switch (kind)
    {
    case format_kind::fixed_frac:
        return sign_chars
            + static_cast<std::size_t>(Traits::max_fixed_integer_digits)
            + static_cast<std::size_t>(precision > 0 ? 1 + precision : 0);

    case format_kind::scientific_frac:
        return sign_chars
            + 1u
            + static_cast<std::size_t>(precision > 0 ? 1 + precision : 0)
            + exp10_chars;

    case format_kind::scientific_sig:
    {
        const int sig_digits = precision > 1 ? precision : 1;
        return sign_chars
            + static_cast<std::size_t>(sig_digits)
            + static_cast<std::size_t>(sig_digits > 1 ? 1 : 0)
            + exp10_chars;
    }

    case format_kind::hex_float:
    {
        const int fraction_digits = precision >= 0
            ? precision
            : hex_max_fraction_digits<Traits>();
        return sign_chars
            + 2u
            + 1u
            + static_cast<std::size_t>(fraction_digits > 0 ? 1 + fraction_digits : 0)
            + exp2_chars;
    }

    case format_kind::general:
    default:
    {
        const int sig_digits = precision > 1 ? precision : 1;
        return sign_chars
            + static_cast<std::size_t>(sig_digits)
            + 6u
            + exp10_chars;
    }
    }
}

template<class Traits, typename String>
constexpr inline void format_to_string(String& out, const typename Traits::value_type& x, int precision, format_kind kind, bool strip_trailing_zeros = false)
{
    if (assign_special_string<Traits>(out, x)) [[unlikely]]
        return;

    if (kind == format_kind::scientific_sig)
    {
        if (precision < 1)
            precision = 1;
    }
    else if (kind != format_kind::hex_float && precision < 0)
    {
        precision = 0;
    }

    const std::size_t cap = format_capacity<Traits>(precision, kind);
    write_chars_to_string(out, cap, [&](char* first, char* last) {
        switch (kind)
        {
        case format_kind::general:
            return Traits::to_chars_general(first, last, x, precision, strip_trailing_zeros);
        case format_kind::fixed_frac:
            return Traits::to_chars_fixed(first, last, x, precision, strip_trailing_zeros);
        case format_kind::scientific_frac:
            return Traits::to_chars_scientific_frac(first, last, x, precision, strip_trailing_zeros);
        case format_kind::scientific_sig:
            return Traits::to_chars_scientific_sig(first, last, x, precision, strip_trailing_zeros);
        case format_kind::hex_float:
            return emit_hex_float_to_chars<Traits>(first, last, x, true, precision, false, strip_trailing_zeros);
        }
        return fltx_char_result{ first, false };
    });
}

template<class Traits, typename String>
constexpr void format_to_string(String& out, const typename Traits::value_type& x, int precision, std::ios_base::fmtflags flags)
{
    if (precision < 0)
        precision = 6;

    const float_format format = float_format_from_flags(flags);
    const bool showpoint = has_showpoint(flags);
    const bool showpos = has_showpos(flags);
    const bool uppercase = has_uppercase(flags);

    if (assign_special_string<Traits>(out, x, uppercase)) [[unlikely]]
    {
        if (showpos && (out.empty() || out[0] != '-'))
            out.insert(0, 1, '+');
        return;
    }

    if (format == float_format::hexfloat)
    {
        format_to_string<Traits>(out, x, -1, format_kind::hex_float, false);
    }
    else if (format == float_format::fixed)
    {
        format_to_string<Traits>(out, x, precision, format_kind::fixed_frac, false);
    }
    else if (format == float_format::scientific)
    {
        format_to_string<Traits>(out, x, precision, format_kind::scientific_frac, false);
    }
    else
    {
        const int sig = (precision == 0) ? 1 : precision;
        if (showpoint && Traits::iszero(x))
        {
            format_to_string<Traits>(out, x, (sig > 1) ? (sig - 1) : 0, format_kind::fixed_frac, false);
        }
        else
        {
            format_to_string<Traits>(out, x, sig, format_kind::general, !showpoint);
        }
    }

    if (showpoint)
        ensure_decimal_point(out);
    apply_stream_decorations(out, showpos, uppercase);
}

template<class Traits, typename String>
constexpr void to_string_into(String& out, const typename Traits::value_type& x, int precision, std::ios_base::fmtflags flags)
{
    format_to_string<Traits>(out, x, precision, flags);
}

template<class Traits>
[[nodiscard]] BL_FORCE_INLINE constexpr int max_bounded_decimal_digits() noexcept
{
    return ((Traits::significand_bits * 30103 + 99999) / 100000) + 8;
}

template<class Traits>
BL_FORCE_INLINE constexpr void append_hybrid_decimal_digit(
    hybrid_parse_token& token,
    int digit) noexcept
{
    if (digit == 0 && !token.seen_nonzero)
        return;

    token.seen_nonzero = true;
    ++token.sig_digits;

    if (!token.coeff_overflow)
    {
        constexpr std::uint64_t max_u64 = ~std::uint64_t{ 0 };
        const std::uint64_t udigit = static_cast<std::uint64_t>(digit);
        if (token.coeff <= (max_u64 - udigit) / 10)
        {
            token.coeff = token.coeff * 10 + udigit;
            return;
        }

        token.coeff_overflow = true;
        token.bounded_coeff = exact_decimal::biguint{ token.coeff };
        token.kept_sig_digits = token.sig_digits - 1;
    }

    if (token.kept_sig_digits < max_bounded_decimal_digits<Traits>())
    {
        token.bounded_coeff.mul_small(10);
        token.bounded_coeff.add_small(static_cast<std::uint32_t>(digit));
        ++token.kept_sig_digits;
        return;
    }

    if (token.first_discarded_digit < 0)
        token.first_discarded_digit = digit;
    else if (digit != 0)
        token.discarded_nonzero = true;
}

template<class Traits, bool bounded>
BL_MSVC_NOINLINE constexpr void scan_hybrid_decimal_digits(
    const char*& p,
    const char* last,
    hybrid_parse_token& token,
    bool fractional) noexcept
{
    while ((!bounded || p != last) && *p >= '0' && *p <= '9')
    {
        append_hybrid_decimal_digit<Traits>(token, *p - '0');
        ++p;
        token.any_digit = true;
        if (fractional)
            ++token.frac_digits;
    }
}

template<bool bounded, typename Token>
BL_MSVC_NOINLINE constexpr void scan_optional_exp10(const char*& p, const char* last, Token& token) noexcept
{
    if ((bounded && p == last) || (*p != 'e' && *p != 'E'))
        return;

    const char* pe = p + 1;
    bool neg_exp = false;
    if ((!bounded || pe != last) && (*pe == '+' || *pe == '-'))
    {
        neg_exp = (*pe == '-');
        ++pe;
    }
    if ((bounded && pe == last) || *pe < '0' || *pe > '9')
        return;

    int eacc = 0;
    while ((!bounded || pe != last) && *pe >= '0' && *pe <= '9')
    {
        const int digit = *pe - '0';
        if (eacc < 100000000)
            eacc = eacc * 10 + digit;
        ++pe;
    }

    token.exp10 = neg_exp ? -eacc : eacc;
    p = pe;
}

template<class Traits, bool bounded>
BL_MSVC_NOINLINE constexpr bool scan_hybrid_decimal_token(
    const char*& p,
    const char* last,
    hybrid_parse_token& token) noexcept
{
    scan_hybrid_decimal_digits<Traits, bounded>(p, last, token, false);
    if ((!bounded || p != last) && *p == '.')
    {
        ++p;
        scan_hybrid_decimal_digits<Traits, bounded>(p, last, token, true);
    }
    if (!token.any_digit)
        return false;

    scan_optional_exp10<bounded>(p, last, token);

    if (token.coeff_overflow &&
        (token.first_discarded_digit > 5 ||
            (token.first_discarded_digit == 5 && (token.discarded_nonzero || token.bounded_coeff.is_odd()))))
    {
        token.bounded_coeff.add_small(1);
    }

    return true;
}

BL_FORCE_INLINE constexpr bool mul_pow10_u64(std::uint64_t& value, int exp) noexcept
{
    if (exp < 0 || exp > 19)
        return false;

    constexpr std::uint64_t max_u64 = ~std::uint64_t{ 0 };
    for (int i = 0; i < exp; ++i)
    {
        if (value > max_u64 / 10)
            return false;
        value *= 10;
    }
    return true;
}

BL_FORCE_INLINE constexpr bool div_pow10_exact_u64(std::uint64_t& value, int exp, int sig_digits) noexcept
{
    if (exp < 0 || exp >= sig_digits)
        return false;

    for (int i = 0; i < exp; ++i)
    {
        if ((value % 10) != 0)
            return false;
        value /= 10;
    }
    return true;
}

template<class Traits, class Token>
BL_FORCE_INLINE constexpr bool try_parse_small_integer(const Token& token, int dec_exp, bool neg, typename Traits::value_type& out) noexcept
{
    if (token.coeff_overflow)
        return false;

    std::uint64_t value = token.coeff;
    if (dec_exp >= 0)
    {
        if (!mul_pow10_u64(value, dec_exp))
            return false;
    }
    else if (!div_pow10_exact_u64(value, -dec_exp, token.sig_digits))
    {
        return false;
    }

    out = Traits::exact_uint64_to_value(value, neg);
    return true;
}

template<class Traits, class Token>
BL_FORCE_INLINE constexpr bool try_parse_compact_decimal(const Token& token, int dec_exp, bool neg, typename Traits::value_type& out) noexcept
{
    return !token.coeff_overflow && Traits::compact_decimal_to_value(token.coeff, dec_exp, neg, out);
}

BL_FORCE_INLINE constexpr bool match_lower_token(
    const char* p,
    const char* last,
    const char* token,
    int length) noexcept
{
    for (int i = 0; i < length; ++i)
    {
        if (last != nullptr && p + i == last)
            return false;

        const char c = p[i];
        if ((last == nullptr && c == '\0') || ascii_lower(c) != static_cast<unsigned char>(token[i]))
            return false;
    }

    return true;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool special_scan_at_end(const char* p, const char* last) noexcept
{
    return last != nullptr ? p == last : *p == '\0';
}

[[nodiscard]] BL_FORCE_INLINE constexpr int special_token_length(const char* p, const char* last) noexcept
{
    if (special_scan_at_end(p, last))
        return 0;

    const unsigned char head = ascii_lower(*p);
    if (head == 'n')
        return match_lower_token(p, last, "nan", 3) ? 3 : 0;
    if (head != 'i')
        return 0;

    if (match_lower_token(p, last, "infinity", 8))
        return 8;

    return match_lower_token(p, last, "inf", 3) ? 3 : 0;
}

[[nodiscard]] BL_FORCE_INLINE constexpr int signed_special_token_length(const char* p, const char* last) noexcept
{
    int sign_length = 0;
    if (!special_scan_at_end(p, last) && (*p == '-' || *p == '+'))
    {
        ++p;
        sign_length = 1;
    }

    const int token_length = special_token_length(p, last);
    return token_length == 0 ? 0 : sign_length + token_length;
}

template<class Traits>
BL_MSVC_NOINLINE constexpr bool parse_special(
    const char*& p,
    const char* last,
    bool neg,
    typename Traits::value_type& out) noexcept
{
    const int token_length = special_token_length(p, last);
    if (token_length == 0)
        return false;

    if (ascii_lower(*p) == 'n') [[unlikely]]
    {
        out = Traits::quiet_nan();
        p += token_length;
        return true;
    }

    out = Traits::infinity(neg);
    p += token_length;
    return true;
}

template<class Traits>
BL_FORCE_INLINE constexpr void parsed_decimal_to_value(
    const hybrid_parse_token& token,
    int dec_exp,
    int approx_dec_order,
    bool neg,
    typename Traits::value_type& out) noexcept
{
    if (approx_dec_order > Traits::max_parse_order)
    {
        out = Traits::infinity(neg);
        return;
    }

    if (approx_dec_order < Traits::min_parse_order)
    {
        out = Traits::zero(neg);
        return;
    }

    if (try_parse_small_integer<Traits>(token, dec_exp, neg, out))
        return;

    if (try_parse_compact_decimal<Traits>(token, dec_exp, neg, out))
        return;

    if (!token.coeff_overflow)
    {
        out = Traits::exact_decimal_to_value(exact_decimal::biguint{ token.coeff }, dec_exp, neg);
        return;
    }

    const int skipped_sig_digits = token.sig_digits - token.kept_sig_digits;
    const int bounded_dec_exp = token.exp10 - token.frac_digits + skipped_sig_digits;
    out = Traits::exact_decimal_to_value(token.bounded_coeff, bounded_dec_exp, neg);
}

template<class Traits, bool bounded>
BL_MSVC_NOINLINE constexpr bool parse_flt(
    const char* first,
    const char* last,
    typename Traits::value_type& out,
    const char** endptr = nullptr) noexcept
{
    const char* p = first;
    while ((!bounded || p != last) && ascii_space(*p))
        ++p;

    bool neg = false;
    if ((!bounded || p != last) && (*p == '+' || *p == '-'))
    {
        neg = (*p == '-');
        ++p;
    }

    if (parse_special<Traits>(p, last, neg, out))
    {
        if (endptr)
            *endptr = p;
        return true;
    }

    hybrid_parse_token token;
    if (!scan_hybrid_decimal_token<Traits, bounded>(p, last, token))
    {
        if (endptr)
            *endptr = first;
        return false;
    }

    if (!token.seen_nonzero)
    {
        out = Traits::zero(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    const int dec_exp = token.exp10 - token.frac_digits;
    const int approx_dec_order = token.sig_digits + dec_exp - 1;
    parsed_decimal_to_value<Traits>(token, dec_exp, approx_dec_order, neg, out);
    if (endptr)
        *endptr = p;
    return true;
}

template<class Traits>
BL_MSVC_NOINLINE constexpr bool parse_flt(const char* s, typename Traits::value_type& out, const char** endptr = nullptr) noexcept
{
    return parse_flt<Traits, false>(s, nullptr, out, endptr);
}

template<class Traits>
BL_MSVC_NOINLINE constexpr bool parse_flt(
    const char* first,
    const char* last,
    typename Traits::value_type& out,
    const char** endptr = nullptr) noexcept
{
    return parse_flt<Traits, true>(first, last, out, endptr);
}

} // namespace bl::detail

namespace bl::detail
{
    [[nodiscard]] BL_FORCE_INLINE constexpr bool fixed_format_needs_exact_integer_path(
        double leading_limb,
        int max_fast_decimal_digits) noexcept
    {
        const double magnitude = fp::absd(leading_limb);
        if (!fp::isfinite(magnitude) || magnitude < 1.0)
            return false;

        constexpr double log10_2 = 0.30102999566398119521373889472449;
        const int exponent2 = fp::frexp_exponent(magnitude);
        const int conservative_decimal_digits =
            static_cast<int>(fp::floor(static_cast<double>(exponent2 - 1) * log10_2)) + 2;
        return conservative_decimal_digits > max_fast_decimal_digits;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool defaultfloat_definitely_scientific(
        double leading_limb,
        int significant_digits) noexcept
    {
        const double magnitude = fp::absd(leading_limb);
        if (!fp::isfinite(magnitude) || magnitude == 0.0)
            return false;

        constexpr double log10_2 = 0.30102999566398119521373889472449;
        const int exponent2 = fp::frexp_exponent(magnitude);
        const int lower_decimal_exponent =
            static_cast<int>(fp::floor(static_cast<double>(exponent2 - 1) * log10_2));

        return lower_decimal_exponent < -5 || lower_decimal_exponent >= significant_digits;
    }

    template<class Traits, typename String>
    [[nodiscard]] BL_FORCE_INLINE constexpr fltx_char_result emit_exact_fixed_decimal_to_chars(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int precision,
        bool strip_trailing_zeros,
        bool zero,
        bool negative)
    {
        if (precision < 0)
            precision = 0;

        if (zero)
            return emit_fixed_zero_to_chars(first, last, negative, precision, strip_trailing_zeros);

        exact_decimal::biguint coefficient;
        bool coefficient_negative = false;
        if (!exact_decimal::exact_decimal_places_integer<Traits>(
                x,
                precision,
                coefficient,
                coefficient_negative))
        {
            return { first, false };
        }

        String digits = exact_decimal::to_decimal_string<String>(coefficient);
        const int digit_count = static_cast<int>(digits.size());
        const int integer_digits = digit_count > precision ? digit_count - precision : 1;
        int fractional_digits = precision > 0 ? precision : 0;

        const auto fractional_digit =
            [&](int index) constexpr -> char
            {
                if (digit_count > precision)
                    return digits[static_cast<std::size_t>(integer_digits + index)];

                const int leading_zero_count = precision - digit_count;
                return index < leading_zero_count
                    ? '0'
                    : digits[static_cast<std::size_t>(index - leading_zero_count)];
            };

        if (strip_trailing_zeros)
        {
            while (fractional_digits > 0 && fractional_digit(fractional_digits - 1) == '0')
                --fractional_digits;
        }

        const bool emit_sign = coefficient_negative && (!coefficient.is_zero() || fractional_digits > 0);
        const std::size_t needed =
            static_cast<std::size_t>(emit_sign ? 1 : 0) +
            static_cast<std::size_t>(integer_digits) +
            (fractional_digits > 0 ? static_cast<std::size_t>(1 + fractional_digits) : 0u);
        if (static_cast<std::size_t>(last - first) < needed)
            return { first, false };

        char* p = first;
        if (emit_sign)
            *p++ = '-';

        if (digit_count > precision)
        {
            copy_chars(p, digits.data(), static_cast<std::size_t>(integer_digits));
            p += integer_digits;
        }
        else
        {
            *p++ = '0';
        }

        if (fractional_digits > 0)
        {
            *p++ = '.';
            for (int i = 0; i < fractional_digits; ++i)
                *p++ = fractional_digit(i);
        }

        return { p, true };
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr fltx_char_result emit_fixed_decimal_for_traits(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int precision,
        bool strip_trailing_zeros)
    {
        if (fixed_format_needs_exact_integer_path(
                Traits::limb(x, 0),
                std::numeric_limits<typename Traits::value_type>::digits10))
        {
            return emit_exact_fixed_decimal_to_chars<
                typename Traits::exact_decimal_traits,
                typename Traits::string_type>(
                    first,
                    last,
                    x,
                    precision,
                    strip_trailing_zeros,
                    Traits::iszero(x),
                    Traits::is_negative(x));
        }

        return Traits::to_chars_fixed_fast(first, last, x, precision, strip_trailing_zeros);
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr fltx_char_result emit_general_decimal_for_traits(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int precision,
        bool strip_trailing_zeros)
    {
        if (precision < 0)
            precision = 0;

        const int sig = precision == 0 ? 1 : precision;
        if (Traits::iszero(x))
            return emit_single_zero_to_chars(first, last);

        const auto ax = Traits::abs(x);
        if (defaultfloat_definitely_scientific(Traits::limb(ax, 0), sig))
            return Traits::to_chars_scientific_sig(first, last, x, sig, strip_trailing_zeros);

        int e10 = 0;
        if (!exact_decimal::exact_decimal_exponent<typename Traits::exact_decimal_traits>(ax, e10))
            return emit_single_zero_to_chars(first, last);

        if (e10 >= -4 && e10 < sig)
        {
            const int frac = sig > e10 + 1 ? sig - (e10 + 1) : 0;
            return Traits::to_chars_default_fixed(first, last, x, frac, sig, e10, strip_trailing_zeros);
        }

        return Traits::to_chars_scientific_sig(first, last, x, sig, strip_trailing_zeros);
    }

    template<class Traits>
    [[nodiscard]] constexpr typename Traits::string_type to_static_string_impl(
        const typename Traits::value_type& value,
        int precision,
        std::ios_base::fmtflags flags)
    {
        typename Traits::string_type out;
        to_string_into<Traits>(out, value, precision, flags);
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool should_collapse_fixed_string(precision_info precision, std::ios_base::fmtflags flags) noexcept
    {
        return float_format_from_flags(flags) == float_format::fixed &&
               precision.leading_digits > 0 &&
               precision.trailing_digits > 0;
    }

    BL_FORCE_INLINE void collapse_fixed_string(std::string& text, precision_info precision)
    {
        if (precision.leading_digits < 0)
            precision.leading_digits = 0;
        if (precision.trailing_digits < 0)
            precision.trailing_digits = 0;

        const std::size_t period = text.find('.');
        if (period == std::string::npos)
            return;

        const std::size_t fractional_begin = period + 1;
        if (fractional_begin >= text.size())
            return;

        const std::size_t fractional_digits = text.size() - fractional_begin;
        const std::size_t leading = static_cast<std::size_t>(precision.leading_digits);
        const std::size_t trailing = static_cast<std::size_t>(precision.trailing_digits);

        if (leading >= fractional_digits || trailing >= fractional_digits || leading + trailing >= fractional_digits)
            return;

        const std::size_t collapse_begin = fractional_begin + leading;
        const std::size_t collapse_end = text.size() - trailing;

        if (collapse_end <= collapse_begin + 3)
            return;

        text.replace(collapse_begin, collapse_end - collapse_begin, "...");
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE std::string to_string_impl(
        const typename Traits::value_type& value,
        precision_info precision,
        std::ios_base::fmtflags flags)
    {
        const int digits = precision.digits >= 0
            ? precision.digits
            : std::numeric_limits<typename Traits::value_type>::digits10;

        const auto text = to_static_string_impl<Traits>(value, digits, flags);
        std::string out(text.data(), text.size());
        if (should_collapse_fixed_string(precision, flags))
            collapse_fixed_string(out, precision);
        return out;
    }

} // namespace bl::detail

#endif
