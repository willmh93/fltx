#ifndef FLTX_COMMON_IO_INCLUDED
#define FLTX_COMMON_IO_INCLUDED

#include "fltx_common_exact.h"

#include <algorithm>
#include <array>
#include <ios>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace bl::fltx::common {



enum class format_kind : unsigned char { general, fixed_frac, scientific_frac, scientific_sig };


template<std::size_t capacity>
struct static_string
{
    using size_type = std::size_t;

    static constexpr size_type npos = static_cast<size_type>(-1);
    static constexpr size_type static_capacity = capacity;

    std::array<char, capacity + 1> chars{};
    size_type length = 0;

    constexpr static_string() noexcept = default;

    constexpr static_string(const char* text)
    {
        assign(text);
    }

    constexpr static_string(std::string_view text)
    {
        assign(text);
    }

    template<std::size_t size>
    constexpr static_string(const char (&text)[size])
    {
        assign(std::string_view(text, size - 1));
    }

    constexpr static_string& operator=(const char* text)
    {
        assign(text);
        return *this;
    }

    constexpr static_string& operator=(std::string_view text)
    {
        assign(text);
        return *this;
    }

    constexpr operator std::string_view() const noexcept
    {
        return view();
    }

    constexpr operator std::string() const
    {
        return std::string(chars.data(), length);
    }

    constexpr std::string_view view() const noexcept
    {
        return std::string_view(chars.data(), length);
    }

    constexpr char* data() noexcept
    {
        return chars.data();
    }

    constexpr const char* data() const noexcept
    {
        return chars.data();
    }

    constexpr const char* c_str() const noexcept
    {
        return chars.data();
    }

    constexpr size_type size() const noexcept
    {
        return length;
    }

    constexpr bool empty() const noexcept
    {
        return length == 0;
    }

    constexpr char* begin() noexcept
    {
        return chars.data();
    }

    constexpr char* end() noexcept
    {
        return chars.data() + length;
    }

    constexpr const char* begin() const noexcept
    {
        return chars.data();
    }

    constexpr const char* end() const noexcept
    {
        return chars.data() + length;
    }

    constexpr char& operator[](size_type index) noexcept
    {
        return chars[index];
    }

    constexpr const char& operator[](size_type index) const noexcept
    {
        return chars[index];
    }

    constexpr char& front() noexcept
    {
        return chars[0];
    }

    constexpr const char& front() const noexcept
    {
        return chars[0];
    }

    constexpr void clear() noexcept
    {
        length = 0;
        chars[0] = '\0';
    }

    constexpr void resize(size_type new_length)
    {
        require_capacity(new_length);
        if (new_length > length)
        {
            for (size_type index = length; index < new_length; ++index)
                chars[index] = '\0';
        }
        length = new_length;
        chars[length] = '\0';
    }

    constexpr void push_back(char value)
    {
        require_capacity(length + 1);
        chars[length++] = value;
        chars[length] = '\0';
    }

    constexpr static_string& append(size_type count, char value)
    {
        require_capacity(length + count);
        for (size_type index = 0; index < count; ++index)
            chars[length + index] = value;
        length += count;
        chars[length] = '\0';
        return *this;
    }

    constexpr static_string& append(const char* text)
    {
        return append(std::string_view(text, const_string_length(text)));
    }

    constexpr static_string& append(std::string_view text)
    {
        require_capacity(length + text.size());
        for (size_type index = 0; index < text.size(); ++index)
            chars[length + index] = text[index];
        length += text.size();
        chars[length] = '\0';
        return *this;
    }

    constexpr static_string& insert(size_type position, const char* text)
    {
        return insert(position, std::string_view(text, const_string_length(text)));
    }

    constexpr static_string& insert(size_type position, std::string_view text)
    {
        require_insert_position(position);
        require_capacity(length + text.size());
        for (size_type index = length; index > position; --index)
            chars[index + text.size() - 1] = chars[index - 1];
        for (size_type index = 0; index < text.size(); ++index)
            chars[position + index] = text[index];
        length += text.size();
        chars[length] = '\0';
        return *this;
    }

    constexpr static_string& insert(size_type position, size_type count, char value)
    {
        require_insert_position(position);
        require_capacity(length + count);
        for (size_type index = length; index > position; --index)
            chars[index + count - 1] = chars[index - 1];
        for (size_type index = 0; index < count; ++index)
            chars[position + index] = value;
        length += count;
        chars[length] = '\0';
        return *this;
    }

    constexpr void assign(const char* text)
    {
        assign(std::string_view(text, const_string_length(text)));
    }

    constexpr void assign(std::string_view text)
    {
        require_capacity(text.size());
        for (size_type index = 0; index < text.size(); ++index)
            chars[index] = text[index];
        length = text.size();
        chars[length] = '\0';
    }

private:
    static constexpr size_type const_string_length(const char* text) noexcept
    {
        size_type result = 0;
        while (text[result] != '\0')
            ++result;
        return result;
    }

    constexpr void require_capacity(size_type requested_capacity) const
    {
        if (requested_capacity > capacity)
            throw "static_string capacity exceeded";
    }

    constexpr void require_insert_position(size_type position) const
    {
        if (position > length)
            throw "static_string insert position out of range";
    }
};

template<std::size_t capacity>
inline std::ostream& operator<<(std::ostream& os, const static_string<capacity>& text)
{
    return os.write(text.data(), static_cast<std::streamsize>(text.size()));
}

using default_io_string = static_string<512>;


template<typename BigUInt>
struct parse_token
{
    using coeff_type = BigUInt;

    coeff_type coeff;
    int frac_digits = 0;
    int sig_digits = 0;
    int exp10 = 0;
    bool any_digit = false;
    bool seen_nonzero = false;
};


template<typename CharsResult, typename String, typename Writer>
constexpr BL_NO_INLINE void write_chars_to_string(String& out, std::size_t cap, Writer&& writer)
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

BL_FORCE_INLINE constexpr bool valid_float_string(const char* s) noexcept
{
    for (const char* p = s; *p; ++p)
    {
        const char c = *p;
        if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+'))
            return false;
    }
    return true;
}

BL_FORCE_INLINE constexpr unsigned char ascii_lower(char c) noexcept
{
    return static_cast<unsigned char>((c >= 'A' && c <= 'Z') ? (c | 0x20) : c);
}

BL_FORCE_INLINE constexpr const char* skip_ascii_space(const char* p) noexcept
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v')
        ++p;
    return p;
}

BL_FORCE_INLINE constexpr void copy_chars(char* dst, const char* src, std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i)
        dst[i] = src[i];
}

template<typename UInt>
BL_FORCE_INLINE constexpr int write_unsigned_decimal_rev(char* dst, UInt value) noexcept
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
BL_FORCE_INLINE constexpr void append_unsigned_decimal(String& out, UInt value)
{
    char buf[32];
    const int len = write_unsigned_decimal_rev(buf, value);
    for (int i = len - 1; i >= 0; --i)
        out.push_back(buf[i]);
}

constexpr inline std::size_t find_exponent_marker(std::string_view text) noexcept
{
    for (std::size_t index = 0; index < text.size(); ++index)
    {
        if (text[index] == 'e' || text[index] == 'E')
            return index;
    }
    return std::string_view::npos;
}

template<typename String>
inline constexpr BL_NO_INLINE void ensure_decimal_point(String& s)
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
inline constexpr BL_NO_INLINE void apply_stream_decorations(String& s, bool showpos, bool uppercase)
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
    if (const char* text = special_text<Traits>(x, uppercase))
    {
        out = text;
        return true;
    }
    return false;
}

template<class Traits>
inline bool write_stream_special(std::ostream& os, const typename Traits::value_type& x, bool showpos, bool uppercase)
{
    const char* text = special_text<Traits>(x, uppercase);
    if (!text)
        return false;
    if (showpos && text[0] != '-')
        os << '+';
    os << text;
    return true;
}

template<class Traits, typename String>
constexpr inline void format_to_string(String& out, const typename Traits::value_type& x, int precision, format_kind kind, bool strip_trailing_zeros = false)
{
    if (assign_special_string<Traits>(out, x))
        return;

    if (kind == format_kind::scientific_sig)
    {
        if (precision < 1)
            precision = 1;
    }
    else if (precision < 0)
    {
        precision = 0;
    }

    const std::size_t cap = ((kind == format_kind::general || kind == format_kind::fixed_frac) ? 1u + 309u : 1u + 1u)
        + 1u + static_cast<std::size_t>(precision) + 32u;

    write_chars_to_string<typename Traits::chars_result>(out, cap, [&](char* first, char* last) {
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
        }
        return typename Traits::chars_result{ first, false };
    });
}

template<class Traits, typename String>
BL_FORCE_INLINE constexpr void to_string_into(String& out, const typename Traits::value_type& x, int precision,
    bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    const format_kind kind = (fixed && !scientific) ? format_kind::fixed_frac : (scientific && !fixed) ? format_kind::scientific_frac : format_kind::general;
    format_to_string<Traits>(out, x, precision, kind, strip_trailing_zeros);
}

template<class Traits, typename String>
BL_FORCE_INLINE constexpr void emit_scientific(String& out, const typename Traits::value_type& x, std::streamsize prec, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, static_cast<int>(prec), format_kind::scientific_frac, strip_trailing_zeros);
}

template<class Traits, typename String>
BL_FORCE_INLINE constexpr void emit_fixed_dec(String& out, const typename Traits::value_type& x, int prec, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, prec, format_kind::fixed_frac, strip_trailing_zeros);
}

template<class Traits, typename String>
BL_FORCE_INLINE constexpr void emit_scientific_sig(String& out, const typename Traits::value_type& x, std::streamsize sig_digits, bool strip_trailing_zeros)
{
    format_to_string<Traits>(out, x, static_cast<int>(sig_digits), format_kind::scientific_sig, strip_trailing_zeros);
}

template<typename Token>
constexpr BL_NO_INLINE void scan_decimal_digits(const char*& p, Token& token, bool fractional) noexcept
{
    while (*p >= '0' && *p <= '9')
    {
        const int digit = *p - '0';
        if (digit != 0 || token.seen_nonzero)
        {
            token.coeff.mul_small(static_cast<std::uint32_t>(10));
            token.coeff.add_small(static_cast<std::uint32_t>(digit));
            ++token.sig_digits;
            token.seen_nonzero = true;
        }
        ++p;
        token.any_digit = true;
        if (fractional)
            ++token.frac_digits;
    }
}

template<typename Token>
constexpr BL_NO_INLINE void scan_optional_exp10(const char*& p, Token& token) noexcept
{
    if (*p != 'e' && *p != 'E')
        return;

    const char* pe = p + 1;
    bool neg_exp = false;
    if (*pe == '+' || *pe == '-')
    {
        neg_exp = (*pe == '-');
        ++pe;
    }
    if (*pe < '0' || *pe > '9')
        return;

    int eacc = 0;
    while (*pe >= '0' && *pe <= '9')
    {
        const int digit = *pe - '0';
        if (eacc < 100000000)
            eacc = eacc * 10 + digit;
        ++pe;
    }

    token.exp10 = neg_exp ? -eacc : eacc;
    p = pe;
}

template<typename Token>
constexpr BL_NO_INLINE bool scan_decimal_token(const char*& p, Token& token) noexcept
{
    scan_decimal_digits(p, token, false);
    if (*p == '.')
    {
        ++p;
        scan_decimal_digits(p, token, true);
    }
    if (!token.any_digit)
        return false;
    scan_optional_exp10(p, token);
    return true;
}

template<class Traits>
constexpr BL_NO_INLINE bool parse_special(const char*& p, bool neg, typename Traits::value_type& out) noexcept
{
    if (ascii_lower(p[0]) == 'n' && ascii_lower(p[1]) == 'a' && ascii_lower(p[2]) == 'n')
    {
        out = Traits::quiet_nan();
        p += 3;
        return true;
    }

    if (ascii_lower(p[0]) != 'i' || ascii_lower(p[1]) != 'n' || ascii_lower(p[2]) != 'f')
        return false;

    p += 3;
    if (ascii_lower(p[0]) == 'i' && ascii_lower(p[1]) == 'n' && ascii_lower(p[2]) == 'i' && ascii_lower(p[3]) == 't' && ascii_lower(p[4]) == 'y')
        p += 5;

    out = Traits::infinity(neg);
    return true;
}

template<class Traits>
constexpr BL_NO_INLINE bool parse_flt(const char* s, typename Traits::value_type& out, const char** endptr = nullptr) noexcept
{
    using token_type = typename Traits::parse_token;

    const char* p = skip_ascii_space(s);
    bool neg = false;
    if (*p == '+' || *p == '-')
    {
        neg = (*p == '-');
        ++p;
    }

    if (parse_special<Traits>(p, neg, out))
    {
        if (endptr)
            *endptr = p;
        return true;
    }

    token_type token;
    if (!scan_decimal_token(p, token))
    {
        if (endptr)
            *endptr = s;
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

    if (approx_dec_order > Traits::max_parse_order)
    {
        out = Traits::infinity(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    if (approx_dec_order < Traits::min_parse_order)
    {
        out = Traits::zero(neg);
        if (endptr)
            *endptr = p;
        return true;
    }

    out = Traits::exact_decimal_to_value(token.coeff, dec_exp, neg);
    if (endptr)
        *endptr = p;
    return true;
}

template<class Traits>
BL_NO_INLINE std::ostream& write_to_stream(std::ostream& os, const typename Traits::value_type& x)
{
    int prec = static_cast<int>(os.precision());
    if (prec < 0)
        prec = 6;

    const auto flags = os.flags();
    const bool fixed = (flags & std::ios_base::fixed) != 0;
    const bool scientific = (flags & std::ios_base::scientific) != 0;
    const bool showpoint = (flags & std::ios_base::showpoint) != 0;
    const bool showpos = (flags & std::ios_base::showpos) != 0;
    const bool uppercase = (flags & std::ios_base::uppercase) != 0;

    if (write_stream_special<Traits>(os, x, showpos, uppercase))
        return os;

    std::string s;
    if (fixed && !scientific)
    {
        format_to_string<Traits>(s, x, prec, format_kind::fixed_frac, false);
    }
    else if (scientific && !fixed)
    {
        format_to_string<Traits>(s, x, prec, format_kind::scientific_frac, false);
    }
    else
    {
        const int sig = (prec == 0) ? 1 : prec;
        if (Traits::iszero(x))
        {
            if (showpoint)
            {
                format_to_string<Traits>(s, x, std::max(0, sig - 1), format_kind::fixed_frac, false);
                ensure_decimal_point(s);
            }
            else
            {
                s = "0";
            }
            apply_stream_decorations(s, showpos, uppercase);
            os << s;
            return os;
        }

        typename Traits::value_type m;
        int e10 = 0;
        Traits::normalize10(Traits::abs(x), m, e10);

        if (e10 >= -4 && e10 < sig)
        {
            format_to_string<Traits>(s, x, std::max(0, sig - (e10 + 1)), format_kind::fixed_frac, !showpoint);
        }
        else if (showpoint)
        {
            format_to_string<Traits>(s, x, std::max(0, sig - 1), format_kind::scientific_frac, false);
            ensure_decimal_point(s);
        }
        else
        {
            format_to_string<Traits>(s, x, sig, format_kind::scientific_sig, true);
        }
    }

    if (showpoint)
        ensure_decimal_point(s);
    apply_stream_decorations(s, showpos, uppercase);
    os << s;
    return os;
}

} // namespace bl::fltx::common

namespace bl::fltx::common::exact_decimal {

[[nodiscard]] constexpr inline default_io_string to_decimal_string(biguint value)
{
    if (value.is_zero())
        return "0";

    std::uint32_t chunks[biguint::max_words * 2];
    int chunk_count = 0;
    while (!value.is_zero() && chunk_count < static_cast<int>(sizeof(chunks) / sizeof(chunks[0])))
        chunks[chunk_count++] = value.div_small(1000000000u);

    default_io_string out;
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

        common_exp = std::min(common_exp, exponent);
        have_term = true;
    }

    if (!have_term)
        return false;

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

    if (acc.mag.is_zero() || acc.neg)
        return false;

    const double log10_2 = 0.30102999566398119521373889472449;
    int approx = static_cast<int>(fp::floor_constexpr((acc.mag.bit_length() - 1 + common_exp) * log10_2));

    int guard = 0;
    while (compare_scaled_with_pow10exp(acc.mag, common_exp, approx) < 0)
    {
        --approx;
        if (++guard > 16)
            break;
    }

    guard = 0;
    while (compare_scaled_with_pow10exp(acc.mag, common_exp, approx + 1) >= 0)
    {
        ++approx;
        if (++guard > 16)
            break;
    }

    exp10 = approx;

    biguint num = acc.mag;
    biguint den{1};
    if (common_exp >= 0)
        num.shl_bits(common_exp);
    else
        den.shl_bits(-common_exp);

    const int scale10 = sig - 1 - exp10;
    if (scale10 >= 0)
    {
        for (int i = 0; i < scale10; ++i)
            num.mul_small(10);
    }
    else
    {
        for (int i = 0; i < -scale10; ++i)
            den.mul_small(10);
    }

    biguint q;
    biguint r;
    divmod_bitwise(num, den, q, r);
    if (!r.is_zero())
    {
        biguint twice_r = r;
        twice_r.shl1();
        const int cmp = compare(twice_r, den);
        if (cmp > 0 || (cmp == 0 && q.is_odd()))
            q.add_small(1);
    }

    const biguint limit = pow10_big(sig);
    if (compare(q, limit) >= 0)
    {
        q.div_small(10);
        ++exp10;
    }

    digits = to_decimal_string(q);
    if (static_cast<int>(digits.size()) < sig)
    {
        const std::size_t zero_pad_count = static_cast<std::size_t>(sig - static_cast<int>(digits.size()));
        digits.insert(0, zero_pad_count, '0');
    }

    return true;
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
        bin_exp = dec_exp;
    }
    else
    {
        denominator = pow5_big(-dec_exp);
        bin_exp = dec_exp;
    }

    int ratio_exp = floor_log2_ratio(numerator, denominator);

    biguint q = extract_rounded_significand_chunks(numerator, denominator, ratio_exp, Traits::significand_bits);
    if (q.bit_length() > Traits::significand_bits)
    {
        q.shr1();
        ++ratio_exp;
    }

    const int e2 = bin_exp + ratio_exp;
    if (e2 > 1023)
        return Traits::infinity(neg);
    if (e2 < -1074)
        return Traits::zero(neg);

    return Traits::pack_from_significand(q, e2, neg);
}

} // namespace bl::fltx::common::exact_decimal

#endif