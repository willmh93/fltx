#ifndef F256_IO_INCLUDED
#define F256_IO_INCLUDED

#include "f256.h"
#include "fltx_common_io.h"

#include <cstring>

namespace bl {

/// ------------------ printing helpers ------------------

namespace _f256_detail
{
    BL_PUSH_PRECISE;
    BL_PRINT_NOINLINE FORCE_INLINE f256_s mul_by_double_print(f256_s a, double b) noexcept
    {
        return a * b;
    }
    BL_PRINT_NOINLINE FORCE_INLINE f256_s sub_by_double_print(f256_s a, double b) noexcept
    {
        return a - b;
    }
    BL_POP_PRECISE;

    struct f256_chars_result
    {
        char* ptr = nullptr;
        bool ok = false;
    };
    struct f256_print_expansion
    {
        double terms[64]{}; // small -> large
        int n = 0;
    };

    inline constexpr void print_expansion_set(f256_print_expansion& st, const f256_s& x) noexcept
    {
        double tmp[4] = { x.x3, x.x2, x.x1, x.x0 };
        st.n = _f256_detail::compress_expansion_zeroelim(4, tmp, st.terms);
    }
    inline constexpr bool print_expansion_is_zero(const f256_print_expansion& st) noexcept
    {
        return st.n <= 0;
    }
    inline constexpr f256_s print_expansion_to_f256(const f256_print_expansion& st) noexcept
    {
        return from_expansion_fast(st.terms, st.n);
    }
    inline constexpr void print_expansion_scale(f256_print_expansion& st, double b) noexcept
    {
        double tmp[128]{};
        int n = _f256_detail::scale_expansion_zeroelim(st.n, st.terms, b, tmp);

        double comp[64]{};
        st.n = _f256_detail::compress_expansion_zeroelim(n, tmp, comp);
        for (int i = 0; i < st.n; ++i)
            st.terms[i] = comp[i];
    }
    inline constexpr void print_expansion_add_double(f256_print_expansion& st, double b) noexcept
    {
        double term = b;
        double tmp[128]{};
        int n = _f256_detail::fast_expansion_sum_zeroelim(st.n, st.terms, 1, &term, tmp);

        double comp[64]{};
        st.n = _f256_detail::compress_expansion_zeroelim(n, tmp, comp);
        for (int i = 0; i < st.n; ++i)
            st.terms[i] = comp[i];
    }
    inline constexpr uint32_t print_expansion_take_uint(f256_print_expansion& st, uint32_t max_value) noexcept
    {
        f256_s approx = print_expansion_to_f256(st);
        long long value = (long long)floor(approx).x0;

        if (value < 0) value = 0;
        else if (value > (long long)max_value) value = (long long)max_value;

        print_expansion_add_double(st, -(double)value);
        return (uint32_t)value;
    }

    inline constexpr void normalize10(const f256_s& x, f256_s& m, int& exp10)
    {
        if (iszero(x)) { m = f256_s{}; exp10 = 0; return; }

        f256_s ax = abs(x);

        int e2 = 0;
        (void)std::frexp(ax.x0, &e2);
        int e10 = (int)fltx::common::fp::floor_constexpr((e2 - 1) * 0.30102999566398114);

        m = ax * pow10_256(-e10);
        while (m >= f256_s{ 10.0, 0.0, 0.0, 0.0 }) { m = m / f256_s{ 10.0, 0.0, 0.0, 0.0 }; ++e10; }
        while (m < f256_s{ 1.0 }) { m = m * f256_s{ 10.0, 0.0, 0.0, 0.0 }; --e10; }
        exp10 = e10;
    }
    inline constexpr int emit_uint_rev_buf(char* dst, f256_s n)
    {
        const f256_s base = f256_s{ 1000000000.0, 0.0, 0.0, 0.0 };

        int len = 0;

        if (n < f256_s{ 10.0, 0.0, 0.0, 0.0 }) {
            int d = (int)n.x0;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            dst[len++] = char('0' + d);
            return len;
        }

        while (n >= base) {
            f256_s q = floor(n / base);
            f256_s r = n - q * base;

            long long chunk = (long long)std::floor(r.x0);
            if (chunk >= 1000000000LL) { chunk -= 1000000000LL; q += 1.0; }
            if (chunk < 0) chunk = 0;

            for (int i = 0; i < 9; ++i) {
                int d = int(chunk % 10);
                dst[len++] = char('0' + d);
                chunk /= 10;
            }

            n = q;
        }

        long long last = (long long)std::floor(n.x0);
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
    inline constexpr f256_chars_result append_exp10_to_chars_f256(char* p, char* end, int e10) noexcept
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

    using biguint = fltx::common::exact_decimal::biguint;

    struct exact_traits
    {
        using value_type = f256_s;
        static constexpr int limb_count = 4;
        static constexpr int significand_bits = 212;

        static constexpr double limb(const value_type& x, int index) noexcept
        {
            switch (index)
            {
            case 0: return x.x0;
            case 1: return x.x1;
            case 2: return x.x2;
            default: return x.x3;
            }
        }
        static constexpr value_type zero(bool neg = false) noexcept
        {
            return neg ? value_type{ -0.0, 0.0, 0.0, 0.0 } : value_type{ 0.0, 0.0, 0.0, 0.0 };
        }
        static constexpr value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static constexpr value_type pack_from_significand(const biguint& q, int e2, bool neg) noexcept
        {
            const std::uint64_t c3 = q.get_bits(0, 53);
            const std::uint64_t c2 = q.get_bits(53, 53);
            const std::uint64_t c1 = q.get_bits(106, 53);
            const std::uint64_t c0 = q.get_bits(159, 53);

            const double x0 = c0 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
            const double x1 = c1 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
            const double x2 = c2 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c2), e2 - 158) : 0.0;
            const double x3 = c3 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c3), e2 - 211) : 0.0;

            f256_s out = renorm(x0, x1, x2, x3);
            if (neg)
                out = -out;
            return out;
        }
    };

    template<typename String>
    constexpr inline bool exact_scientific_digits(const f256_s& x, int sig, String& digits, int& exp10)
    {
        return fltx::common::exact_decimal::exact_scientific_digits<exact_traits>(x, sig, digits, exp10);
    }
    constexpr inline f256_s exact_decimal_to_f256(const biguint& coeff, int dec_exp, bool neg) noexcept
    {
        return fltx::common::exact_decimal::exact_decimal_to_value<exact_traits>(coeff, dec_exp, neg);
    }

    inline constexpr double pow10_double_table[10] = {
        1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
        1000000.0, 10000000.0, 100000000.0, 1000000000.0
    };

    inline constexpr f256_chars_result emit_fixed_dec_to_chars(char* first, char* last, f256_s x, int prec, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (prec < 0) prec = 0;

        const bool neg = (x.x0 < 0.0);
        if (neg) x = -x;
        x = renorm(x.x0, x.x1, x.x2, x.x3);

        f256_s ip = floor(x);
        f256_s fp = x - ip;

        if (fp >= f256_s{ 1.0 }) { fp -= 1.0; ip += 1.0; }
        else if (fp < f256_s{}) { fp = f256_s{}; }

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

            f256_print_expansion fp_exp;
            print_expansion_set(fp_exp, fp);

            int written = 0;
            const int full = prec / 9;
            const int rem = prec - full * 9;

            for (int c = 0; c < full; ++c) {
                print_expansion_scale(fp_exp, pow10_double_table[9]);
                uint32_t chunk = print_expansion_take_uint(fp_exp, 999999999u);

                for (int i = 8; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += 9;
            }

            if (rem > 0) {
                print_expansion_scale(fp_exp, pow10_double_table[rem]);
                uint32_t chunk = print_expansion_take_uint(fp_exp, (uint32_t)pow10_double_table[rem] - 1u);

                for (int i = rem - 1; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += rem;
            }

            f256_print_expansion round_exp = fp_exp;
            print_expansion_scale(round_exp, 10.0);
            int next = (int)print_expansion_take_uint(round_exp, 9u);

            const int last_digit = frac[prec - 1] - '0';
            bool round_up = false;
            if (next > 5) round_up = true;
            else if (next < 5) round_up = false;
            else {
                const bool gt_half = !print_expansion_is_zero(round_exp);
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
                    ip += 1.0;
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
            fltx::common::copy_chars(p, frac, static_cast<std::size_t>(frac_len));
            p += frac_len;
        }

        return { p, true };
    }
    inline constexpr f256_chars_result emit_scientific_sig_to_chars(char* first, char* last, const f256_s& x, std::streamsize sig_digits, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (sig_digits < 1) sig_digits = 1;

        const bool neg = (x.x0 < 0.0);
        const f256_s v = neg ? -x : x;
        const int sig = static_cast<int>(sig_digits);

        fltx::common::default_io_string digits;
        int e = 0;
        if (!_f256_detail::exact_scientific_digits(v, sig, digits, e)) {
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
        auto er = append_exp10_to_chars_f256(ep, eend, e);
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
            fltx::common::copy_chars(p, digits.data() + 1, static_cast<std::size_t>(last_frac));
            p += last_frac;
        }
        fltx::common::copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
        p += exp_len;
        return { p, true };
    }
    inline constexpr f256_chars_result emit_scientific_to_chars(char* first, char* last, const f256_s& x, std::streamsize frac_digits, bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0) frac_digits = 0;

        if (iszero(x)) {
            const bool neg = _f256_detail::signbit_constexpr(x.x0);
            int frac_len = strip_trailing_zeros ? 0 : (int)frac_digits;

            char exp_buf[16];
            char* ep = exp_buf;
            char* eend = exp_buf + sizeof(exp_buf);
            auto er = append_exp10_to_chars_f256(ep, eend, 0);
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

            fltx::common::copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
            p += exp_len;
            return { p, true };
        }

        return emit_scientific_sig_to_chars(first, last, x, frac_digits + 1, strip_trailing_zeros);
    }
    inline constexpr f256_chars_result to_chars(char* first, char* last, const f256_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false) noexcept
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

        const f256_s ax = (x.x0 < 0.0) ? -x : x;
        fltx::common::default_io_string digits;
        int e10 = 0;
        if (!_f256_detail::exact_scientific_digits(ax, 1, digits, e10)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (e10 >= -4 && e10 < sig) {
            const int frac = std::max(0, sig - (e10 + 1));
            return emit_fixed_dec_to_chars(first, last, x, frac, strip_trailing_zeros);
        }

        return emit_scientific_sig_to_chars(first, last, x, sig, strip_trailing_zeros);
    }

    using f256_format_kind = fltx::common::format_kind;
    using f256_parse_token = fltx::common::parse_token<_f256_detail::biguint>;

    struct f256_io_traits
    {
        using value_type = f256_s;
        using chars_result = f256_chars_result;
        using parse_token = f256_parse_token;

        static constexpr int max_parse_order = 330;
        static constexpr int min_parse_order = -400;

        static constexpr bool isnan(const value_type& x) noexcept { return bl::isnan(x); }
        static constexpr bool isinf(const value_type& x) noexcept { return bl::isinf(x); }
        static constexpr bool iszero(const value_type& x) noexcept { return bl::iszero(x); }
        static constexpr bool is_negative(const value_type& x) noexcept { return x.x0 < 0.0; }
        static constexpr value_type abs(const value_type& x) noexcept { return (x.x0 < 0.0) ? -x : x; }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0, 0.0, 0.0 } : value_type{ 0.0, 0.0, 0.0, 0.0 }; }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static constexpr value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }
        static constexpr void normalize10(const value_type& x, value_type& m, int& e10) { _f256_detail::normalize10(x, m, e10); }
        static constexpr chars_result to_chars_general(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return to_chars(first, last, x, precision, false, false, strip_trailing_zeros);
        }
        static constexpr chars_result to_chars_fixed(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static constexpr chars_result to_chars_scientific_frac(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static constexpr chars_result to_chars_scientific_sig(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_sig_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static constexpr value_type exact_decimal_to_value(const parse_token::coeff_type& coeff, int dec_exp, bool neg)
        {
            return _f256_detail::exact_decimal_to_f256(coeff, dec_exp, neg);
        }
    };

    template<typename String, typename Writer>
    FORCE_INLINE constexpr void write_chars_to_string(String& out, std::size_t cap, Writer writer)
    {
        fltx::common::write_chars_to_string<f256_chars_result>(out, cap, writer);
    }
    FORCE_INLINE const char* special_text_f256(const f256_s& x, bool uppercase = false) noexcept
    {
        return fltx::common::special_text<f256_io_traits>(x, uppercase);
    }
    template<typename String>
    FORCE_INLINE constexpr bool assign_special_string(String& out, const f256_s& x, bool uppercase = false) noexcept
    {
        return fltx::common::assign_special_string<f256_io_traits>(out, x, uppercase);
    }
    template<typename String>
    FORCE_INLINE constexpr void ensure_decimal_point(String& s)
    {
        fltx::common::ensure_decimal_point(s);
    }
    template<typename String>
    FORCE_INLINE constexpr void apply_stream_decorations(String& s, bool showpos, bool uppercase)
    {
        fltx::common::apply_stream_decorations(s, showpos, uppercase);
    }
    FORCE_INLINE bool write_stream_special(std::ostream& os, const f256_s& x, bool showpos, bool uppercase)
    {
        return fltx::common::write_stream_special<f256_io_traits>(os, x, showpos, uppercase);
    }
    template<typename String>
    FORCE_INLINE constexpr void format_to_string(String& out, const f256_s& x, int precision, f256_format_kind kind, bool strip_trailing_zeros = false)
    {
        fltx::common::format_to_string<f256_io_traits>(out, x, precision, kind, strip_trailing_zeros);
    }
    template<typename String>
    FORCE_INLINE constexpr void to_string_into(String& out, const f256_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
    {
        fltx::common::to_string_into<f256_io_traits>(out, x, precision, fixed, scientific, strip_trailing_zeros);
    }
    template<typename String>
    FORCE_INLINE constexpr void emit_scientific(String& out, const f256_s& x, std::streamsize prec, bool strip_trailing_zeros)
    {
        fltx::common::emit_scientific<f256_io_traits>(out, x, prec, strip_trailing_zeros);
    }
    template<typename String>
    FORCE_INLINE constexpr void emit_fixed_dec(String& out, const f256_s& x, int prec, bool strip_trailing_zeros)
    {
        fltx::common::emit_fixed_dec<f256_io_traits>(out, x, prec, strip_trailing_zeros);
    }
    template<typename String>
    FORCE_INLINE constexpr void emit_scientific_sig(String& out, const f256_s& x, std::streamsize sig_digits, bool strip_trailing_zeros)
    {
        fltx::common::emit_scientific_sig<f256_io_traits>(out, x, sig_digits, strip_trailing_zeros);
    }

    /// ======== Parsing helpers ========

    FORCE_INLINE bool valid_flt256_string(const char* s) noexcept
    {
        return fltx::common::valid_float_string(s);
    }
    FORCE_INLINE unsigned char ascii_lower_f256(char c) noexcept
    {
        return fltx::common::ascii_lower(c);
    }
    FORCE_INLINE const char* skip_ascii_space_f256(const char* p) noexcept
    {
        return fltx::common::skip_ascii_space(p);
    }

}

/// ------------------ printing / parsing (public) ------------------

[[nodiscard]] FORCE_INLINE constexpr bool parse_flt256(const char* s, f256_s& out, const char** endptr = nullptr) noexcept
{
    return fltx::common::parse_flt<_f256_detail::f256_io_traits>(s, out, endptr);
}
[[nodiscard]] FORCE_INLINE constexpr f256_s to_f256(const char* s) noexcept
{
    f256_s ret;
    if (parse_flt256(s, ret))
        return ret;
    return f256_s{ 0.0 };
}
[[nodiscard]] FORCE_INLINE constexpr f256_s to_f256(const std::string& s) noexcept
{
    return to_f256(s.c_str());
}
template<std::size_t capacity = fltx::common::default_io_string::static_capacity>
[[nodiscard]] FORCE_INLINE constexpr fltx::common::static_string<capacity> to_static_string(const f256_s& x, int precision = std::numeric_limits<f256_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    fltx::common::static_string<capacity> out;
    _f256_detail::to_string_into(out, x, precision, fixed, scientific, strip_trailing_zeros);
    return out;
}

[[nodiscard]] FORCE_INLINE constexpr fltx::common::default_io_string to_string(const f256_s& x, int precision = std::numeric_limits<f256_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    return to_static_string(x, precision, fixed, scientific, strip_trailing_zeros);
}

[[nodiscard]] FORCE_INLINE std::string to_std_string(const f256_s& x, int precision = std::numeric_limits<f256_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    const auto text = to_string(x, precision, fixed, scientific, strip_trailing_zeros);
    return std::string(text.data(), text.size());
}

/// ------------------ stream output ------------------

inline NO_INLINE std::ostream& operator<<(std::ostream& os, const f256_s& x)
{
    return fltx::common::write_to_stream<_f256_detail::f256_io_traits>(os, x);
}

/// ------------------ literals ------------------
namespace literals
{
    [[nodiscard]] constexpr f256_s operator""_qd(unsigned long long v) noexcept {
        return to_f256(static_cast<uint64_t>(v));
    }
    [[nodiscard]] constexpr f256_s operator""_qd(long double v) noexcept {
        return f256_s{ static_cast<double>(v) };
    }
    [[nodiscard]] consteval f256_s operator""_qd(const char* text, std::size_t len) noexcept
    {
        f256_s out{};
        const char* end = text;
        if (!(parse_flt256(text, out, &end) && (static_cast<std::size_t>(end - text) == len)))
            throw "invalid _qd literal";

        return out;
    }
}

constexpr f256::f256(const char* text)
{
    const char* end = text;
    if (!parse_flt256(text, *this, &end))
        throw "invalid f256";
}

#define QD(x) bl::to_f256(#x)

} // namespace bl

#endif