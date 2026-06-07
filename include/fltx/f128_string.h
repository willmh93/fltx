/**
 * fltx/f128_string.h - Constexpr string formatting, parsing, and literals for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_STRING_INCLUDED
#define F128_STRING_INCLUDED
#include <cstddef>
#include <string>

#include "fltx/f128_limits.h"
#include "fltx/f128_math.h"
#include "fltx/detail/common_io.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    BL_FORCE_INLINE constexpr void normalize10(const f128_s& x, f128_s& m, int& exp10)
    {
        if (x.hi == 0.0 && x.lo == 0.0) { m = f128_s{ 0.0 }; exp10 = 0; return; }

        f128_s ax = abs(x);

        int e2 = detail::fp::frexp_exponent(ax.hi); // ax.hi = f * 2^(e2-1)
        int e10 = (int)detail::fp::floor((e2 - 1) * 0.30102999566398114); // ≈ log10(2)

        m = ax * pow10_128(-e10);
        while (m >= f128_s{ 10.0 }) { m = m / f128_s{ 10.0 }; ++e10; }
        while (m < f128_s{ 1.0 }) { m = m * f128_s{ 10.0 }; --e10; }
        exp10 = e10;
    }

    BL_PUSH_PRECISE;
    BL_FORCE_INLINE constexpr f128_s mul_by_double_print(f128_s a, double b) noexcept
    {
        double p, err;
        detail::_f128::two_prod_precise(a.hi, b, p, err);
        err += a.lo * b;

        double s, e;
        detail::_f128::two_sum_precise(p, err, s, e);
        return f128_s{ s, e };
    }

    BL_FORCE_INLINE constexpr f128_s sub_by_double_print(f128_s a, double b) noexcept
    {
        double s, e;
        detail::_f128::two_sum_precise(a.hi, -b, s, e);
        e += a.lo;

        double ss, ee;
        detail::_f128::two_sum_precise(s, e, ss, ee);
        return f128_s{ ss, ee };
    }
    BL_POP_PRECISE;

    BL_FORCE_INLINE constexpr int emit_uint_rev_buf(char* dst, f128_s n)
    {
        // n is a non-negative integer in f128
        const f128_s base = f128_s{ 1000000000.0 }; // 1e9

        int len = 0;

        if (n < f128_s{ 10.0 }) {
            int d = (int)n.hi;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            dst[len++] = char('0' + d);
            return len;
        }

        while (n >= base) {
            f128_s q = floor(n / base);
            f128_s r = n - q * base;

            long long chunk = (long long)detail::fp::floor(r.hi);
            if (chunk >= 1000000000LL) { chunk -= 1000000000LL; q = q + f128_s{ 1.0 }; }
            if (chunk < 0) chunk = 0;

            for (int i = 0; i < 9; ++i) {
                int d = int(chunk % 10);
                dst[len++] = char('0' + d);
                chunk /= 10;
            }

            n = q;
        }

        long long last = (long long)detail::fp::floor(n.hi);
        if (last == 0) {
            dst[len++] = '0';
        }
        else {
            while (last > 0) {
                int d = int(last % 10);
                dst[len++] = char('0' + d);
                last /= 10;
            }
        }

        return len;
    }

    struct exact_traits
    {
        using value_type = f128_s;
        static constexpr int limb_count       = 2;
        static constexpr int significand_bits = 106;

        static constexpr double limb(const value_type& x, int index) noexcept
        {
            return index == 0 ? x.hi : x.lo;
        }

        static constexpr value_type zero(bool neg = false) noexcept
        {
            return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 };
        }

        static constexpr value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }

        static constexpr value_type pack_from_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            const std::uint64_t c1 = q.get_bits(0, 53);
            const std::uint64_t c0 = q.get_bits(53, 53);
            const double hi = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
            const double lo = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
            f128_s out = detail::_f128::renorm(hi, lo);
            if (neg)
                out = -out;
            return out;
        }
    };

    template<typename String>
    constexpr inline bool exact_scientific_digits(const f128_s& x, int sig, String& digits, int& exp10)
    {
        return detail::exact_decimal::exact_scientific_digits<exact_traits>(x, sig, digits, exp10);
    }

    constexpr detail::fltx_char_result emit_fixed_dec_to_chars(char* first, char* last, f128_s x, int prec, bool strip_trailing_zeros) noexcept
    {
        if (prec < 0) prec = 0;

        if (x.hi == 0.0 && x.lo == 0.0)
            return detail::emit_fixed_zero_to_chars(first, last, detail::_f128::signbit(x.hi), prec, strip_trailing_zeros);

        const bool neg = (x.hi < 0.0);
        if (neg) x = f128_s{ -x.hi, -x.lo };
        x = detail::_f128::renorm(x.hi, x.lo);

        f128_s ip = floor(x);
        f128_s fp = sub_by_double_print(x, ip.hi);

        if (fp >= f128_s{ 1.0 }) { fp = fp - f128_s{ 1.0 }; ip = ip + f128_s{ 1.0 }; }
        else if (fp < f128_s{ 0.0 }) { fp = f128_s{ 0.0 }; }

        constexpr int kFracStack = 2048;
        char frac_stack[kFracStack];
        char* frac = frac_stack;

        std::string frac_dyn;
        if (prec > kFracStack) {
            frac_dyn.resize((size_t)prec);
            frac = (char*)frac_dyn.data();
        }

        int frac_len = (prec > 0) ? prec : 0;

        if (prec > 0) {
            constexpr double kPow10[10] = {
                1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                1000000.0, 10000000.0, 100000000.0, 1000000000.0
            };
            constexpr uint32_t kPow10u32[10] = {
                1u, 10u, 100u, 1000u, 10000u, 100000u,
                1000000u, 10000000u, 100000000u, 1000000000u
            };

            int written = 0;
            const int full = prec / 9;
            const int rem  = prec - full * 9;

            for (int c = 0; c < full; ++c) {
                fp = mul_by_double_print(fp, kPow10[9]);

                uint32_t chunk = 0;
                if (fp.hi > 0.0) {
                    const double hi_floor = detail::fp::floor(fp.hi);
                    if (hi_floor >= (double)kPow10u32[9])
                        chunk = kPow10u32[9] - 1u;
                    else
                        chunk = (uint32_t)hi_floor;
                }

                fp = sub_by_double_print(fp, (double)chunk);

                if (fp < f128_s{ 0.0 }) {
                    if (chunk > 0u) {
                        --chunk;
                        fp = sub_by_double_print(fp, -1.0);
                    }
                    else {
                        fp = f128_s{ 0.0 };
                    }
                }

                for (int i = 8; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += 9;
            }

            if (rem > 0) {
                fp = mul_by_double_print(fp, kPow10[rem]);

                uint32_t chunk = 0;
                const uint32_t chunk_limit = kPow10u32[rem] - 1u;
                if (fp.hi > 0.0) {
                    const double hi_floor = detail::fp::floor(fp.hi);
                    if (hi_floor >= (double)kPow10u32[rem])
                        chunk = chunk_limit;
                    else
                        chunk = (uint32_t)hi_floor;
                }

                fp = sub_by_double_print(fp, (double)chunk);

                if (fp < f128_s{ 0.0 }) {
                    if (chunk > 0u) {
                        --chunk;
                        fp = sub_by_double_print(fp, -1.0);
                    }
                    else {
                        fp = f128_s{ 0.0 };
                    }
                }

                for (int i = rem - 1; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += rem;
            }

            f128_s la = mul_by_double_print(fp, 10.0);
            int next = (int)la.hi;
            if (next < 0) next = 0; else if (next > 9) next = 9;
            f128_s remv = sub_by_double_print(la, (double)next);

            const int last_digit = frac[prec - 1] - '0';
            bool round_up = false;
            if (next > 5) round_up = true;
            else if (next < 5) round_up = false;
            else {
                const bool gt_half = (remv.hi > 0.0) || (remv.lo > 0.0);
                round_up = gt_half || ((last_digit & 1) != 0);
            }

            if (round_up) {
                int i = prec - 1;
                for (; i >= 0; --i) {
                    char& c = frac[i];
                    if (c == '9') c = '0';
                    else { c = char(c + 1); break; }
                }
                if (i < 0) {
                    ip = ip + f128_s{ 1.0 };
                    for (int j = 0; j < prec; ++j) frac[j] = '0';
                }
            }

            if (strip_trailing_zeros) {
                while (frac_len > 0 && frac[frac_len - 1] == '0') --frac_len;
            }
        }

        char int_rev[320];
        int int_len = emit_uint_rev_buf(int_rev, ip);

        if (neg && int_len == 1 && int_rev[0] == '0' && frac_len == 0) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        const size_t needed = (size_t)(neg ? 1 : 0) + (size_t)int_len + (frac_len ? (size_t)(1 + frac_len) : 0u);
        if ((size_t)(last - first) < needed) return { first, false };

        char* p = first;
        if (neg) *p++ = '-';

        for (int i = int_len - 1; i >= 0; --i) *p++ = int_rev[i];

        if (frac_len > 0) {
            *p++ = '.';
            detail::copy_chars(p, frac, static_cast<std::size_t>(frac_len));
            p += frac_len;
        }

        return { p, true };
    }

    BL_FORCE_INLINE constexpr detail::fltx_char_result emit_scientific_sig_to_chars_f128(char* first, char* last, const f128_s& x, int sig_digits, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        if (sig_digits < 1) sig_digits = 1;
        const bool neg = (x.hi < 0.0);
        const f128_s v = neg ? -x : x;
        const int sig = static_cast<int>(sig_digits);
        bl::default_io_string digits;
        int e = 0;
        if (!detail::_f128::exact_scientific_digits(v, sig, digits, e)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        int last_frac = sig - 1;
        if (sig > 1 && strip_trailing_zeros) {
            while (last_frac >= 1 && digits[last_frac] == '0') --last_frac;
        }
        char exp_buf[16];
        char* ep = exp_buf;
        char* eend = exp_buf + sizeof(exp_buf);
        auto er = detail::append_exp10_to_chars(ep, eend, e);
        if (!er.ok) return { first, false };
        const int exp_len = static_cast<int>(er.ptr - ep);
        const bool has_frac = (sig > 1) && (last_frac >= 1);
        const size_t needed = static_cast<size_t>(neg ? 1 : 0) + 1u + (has_frac ? static_cast<size_t>(1 + last_frac) : 0u) + static_cast<size_t>(exp_len);
        if (static_cast<size_t>(last - first) < needed) return { first, false };
        char* p = first;
        if (neg) *p++ = '-';
        *p++ = digits[0];
        if (has_frac) {
            *p++ = '.';
            detail::copy_chars(p, digits.data() + 1, static_cast<std::size_t>(last_frac));
            p += last_frac;
        }
        detail::copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
        p += exp_len;
        return { p, true };
    }

    BL_FORCE_INLINE constexpr detail::fltx_char_result emit_scientific_to_chars(char* first, char* last, const f128_s& x, int frac_digits, bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0) frac_digits = 0;
        if (iszero(x)) {
            const bool neg = detail::_f128::signbit(x.hi);
            int frac_len = strip_trailing_zeros ? 0 : static_cast<int>(frac_digits);
            char exp_buf[16];
            char* ep = exp_buf;
            char* eend = exp_buf + sizeof(exp_buf);
            auto er = detail::append_exp10_to_chars(ep, eend, 0);
            if (!er.ok) return { first, false };
            const int exp_len = static_cast<int>(er.ptr - ep);
            const size_t needed = static_cast<size_t>(neg ? 1 : 0) + 1u + (frac_len ? static_cast<size_t>(1 + frac_len) : 0u) + static_cast<size_t>(exp_len);
            if (static_cast<size_t>(last - first) < needed) return { first, false };
            char* p = first;
            if (neg) *p++ = '-';
            *p++ = '0';
            if (frac_len > 0) {
                *p++ = '.';
                for (int i = 0; i < frac_len; ++i) *p++ = '0';
            }
            detail::copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
            p += exp_len;
            return { p, true };
        }
        return emit_scientific_sig_to_chars_f128(first, last, x, frac_digits + 1, strip_trailing_zeros);
    }

    BL_FORCE_INLINE constexpr detail::fltx_char_result to_chars(char* first, char* last, const f128_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false) noexcept
    {
        if (precision < 0) precision = 0;
        if (fixed && !scientific)
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        if (scientific && !fixed)
            return emit_scientific_to_chars(first, last, x, precision, strip_trailing_zeros);
        const int sig = (precision == 0) ? 1 : precision;
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        const f128_s ax = (x.hi < 0.0) ? -x : x;
        bl::default_io_string digits;
        int e10 = 0;
        if (!detail::_f128::exact_scientific_digits(ax, 1, digits, e10)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        if (e10 >= -4 && e10 < sig) {
            const int frac = (sig > e10 + 1) ? (sig - (e10 + 1)) : 0;
            return emit_fixed_dec_to_chars(first, last, x, frac, strip_trailing_zeros);
        }
        return emit_scientific_sig_to_chars_f128(first, last, x, sig, strip_trailing_zeros);
    }

    struct f128_io_traits
    {
        using value_type = f128_s;

        static constexpr int max_parse_order = detail::fltx_max_parse_order;
        static constexpr int min_parse_order = detail::fltx_min_parse_order;

        static constexpr bool isnan(const value_type& x)       noexcept { return bl::isnan(x); }
        static constexpr bool isinf(const value_type& x)       noexcept { return bl::isinf(x); }
        static constexpr bool iszero(const value_type& x)      noexcept { return bl::iszero(x); }
        static constexpr bool is_negative(const value_type& x) noexcept { return x.hi < 0.0; }
        static constexpr value_type abs(const value_type& x)   noexcept { return (x.hi < 0.0) ? -x : x; }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 }; }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }

        static constexpr value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }
        static constexpr void normalize10(const value_type& x, value_type& m, int& e10) { detail::_f128::normalize10(x, m, e10); }
        static constexpr detail::fltx_char_result to_chars_general(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return to_chars(first, last, x, precision, false, false, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_fixed(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_scientific_frac(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_to_chars(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_scientific_sig(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_sig_to_chars_f128(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr value_type exact_uint64_to_value(std::uint64_t value, bool neg)
        {
            value_type out = detail::_f128::uint64_to_f128(value);
            return neg ? -out : out;
        }

        static constexpr bool compact_decimal_to_value(std::uint64_t coeff, int dec_exp, bool neg, value_type& out)
        {
            return detail::exact_decimal::compact_decimal_to_value<detail::_f128::exact_traits>(coeff, dec_exp, neg, out);
        }

        static constexpr value_type exact_decimal_to_value(const detail::fltx_parse_token::coeff_type& coeff, int dec_exp, bool neg)
        {
            return detail::exact_decimal::exact_decimal_to_value<detail::_f128::exact_traits>(coeff, dec_exp, neg);
        }
    };

    template<typename String>
    BL_FORCE_INLINE constexpr void to_string_into(String& out, const f128_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
    {
        detail::to_string_into<f128_io_traits>(out, x, precision, fixed, scientific, strip_trailing_zeros);
    }

} // namespace detail::_f128

BL_MSVC_NOINLINE constexpr bool parse_flt128(const char* s, f128_s& out, const char** endptr = nullptr) noexcept
{
    return detail::parse_flt<detail::_f128::f128_io_traits>(s, out, endptr);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s to_f128(const char* s) noexcept
{
    f128_s ret;
    if (parse_flt128(s, ret))
        return ret;
    return f128_s{ 0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(const std::string& s) noexcept
{
    return to_f128(s.c_str());
}

template<std::size_t capacity = bl::default_io_string::static_capacity>
[[nodiscard]] constexpr bl::static_string<capacity> to_static_string(const f128_s& x, int precision = std::numeric_limits<f128_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    bl::static_string<capacity> out;
    detail::_f128::to_string_into(out, x, precision, fixed, scientific, strip_trailing_zeros);
    return out;
}

[[nodiscard]] constexpr bl::default_io_string to_string(const f128_s& x, int precision = std::numeric_limits<f128_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    return to_static_string(x, precision, fixed, scientific, strip_trailing_zeros);
}

[[nodiscard]] inline std::string to_std_string(const f128_s& x, int precision = std::numeric_limits<f128_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    const auto text = to_string(x, precision, fixed, scientific, strip_trailing_zeros);
    return std::string(text.data(), text.size());
}

namespace detail::_f128 // primitives and kernels
{
    [[nodiscard]] consteval f128_s parse_dd_literal(const char* text, const char* expected_end)
    {
        f128_s out{};
        const char* end = text;

        if (!(parse_flt128(text, out, &end) && end == expected_end))
            throw "invalid _dd literal";

        return out;
    }

} // namespace detail::_f128

namespace literals
{
    [[nodiscard]] consteval f128 operator""_dd(const char* text, std::size_t length)
    {
        return detail::_f128::parse_dd_literal(text, text + length);
    }

    template<char... Chars>
    [[nodiscard]] consteval f128 operator""_dd()
    {
        constexpr char text[] = { Chars..., '\0' };
        return detail::_f128::parse_dd_literal(text, text + sizeof...(Chars));
    }

} // namespace literals

constexpr f128::f128(const char* text)
{
    const char* end = text;
    if (!parse_flt128(text, *this, &end))
        throw "invalid f128";
}

} // namespace bl

#endif
