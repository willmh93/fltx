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
#include <ios>
#include <string>

#include "fltx/f128_limits.h"
#include "fltx/detail/f128_math_basic.h"
#include "fltx/detail/common_io.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    struct f128_io_traits;

    BL_FORCE_INLINE constexpr void normalize10(const f128_s& x, f128_s& m, int& exp10)
    {
        if (x.hi == 0.0 && x.lo == 0.0) { m = f128_s{ 0.0 }; exp10 = 0; return; }

        f128_s ax = abs(x);

        int e2 = detail::fp::frexp_exponent(ax.hi); // ax.hi = f * 2^(e2-1)
        int e10 = (int)detail::fp::floor((e2 - 1) * 0.30102999566398114); // ≈ log10(2)

        m = ax * bl::detail::_f128_impl::pow10_128(-e10);
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

    constexpr inline bool approximate_scientific_digits(const f128_s& x, int sig, char* digits, int capacity, int& exp10) noexcept
    {
        if (sig < 1)
            sig = 1;

        f128_s scaled{};
        normalize10(x, scaled, exp10);
        if (!(scaled >= f128_s{ 1.0 }) || !(scaled < f128_s{ 10.0 }))
            return false;

        const int digit_count = sig + 1;
        if (digit_count > capacity)
            return false;

        for (int i = 0; i < digit_count; ++i)
        {
            const int digit = static_cast<int>(scaled.hi);
            if (digit < -9 || digit > 19)
                return false;

            digits[i] = static_cast<char>('0' + digit);
            scaled = mul_by_double_print(sub_by_double_print(scaled, static_cast<double>(digit)), 10.0);
        }

        for (int i = digit_count - 1; i > 0; --i)
        {
            if (digits[i] < '0')
            {
                --digits[i - 1];
                digits[i] = static_cast<char>(digits[i] + 10);
            }
            else if (digits[i] > '9')
            {
                ++digits[i - 1];
                digits[i] = static_cast<char>(digits[i] - 10);
            }
        }

        if (digits[0] <= '0' || digits[0] > '9')
            return false;

        if (digits[digit_count - 1] >= '5')
        {
            ++digits[digit_count - 2];
            int i = digit_count - 2;
            while (i > 0 && digits[i] > '9')
            {
                digits[i] = static_cast<char>(digits[i] - 10);
                ++digits[--i];
            }
        }

        if (digits[0] > '9')
        {
            ++exp10;
            for (int i = sig - 1; i >= 2; --i)
                digits[i] = digits[i - 1];
            digits[0] = '1';
            if (sig > 1)
                digits[1] = '0';
        }

        return true;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool should_try_approximate_scientific_digits(const f128_s& x, int sig) noexcept
    {
        if (sig <= 16)
            return true;

        constexpr int max_meaningful_sig = std::numeric_limits<f128_s>::digits10 + 1;
        return sig < max_meaningful_sig && x.hi >= 1.0e-4 && x.hi < 1.0e5;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool should_accept_approximate_scientific_digits(int sig, int exp10) noexcept
    {
        if (sig <= 16)
            return true;

        constexpr int max_meaningful_sig = std::numeric_limits<f128_s>::digits10 + 1;
        return sig < max_meaningful_sig && exp10 >= -4 && exp10 <= 4;
    }

    BL_FORCE_INLINE constexpr int emit_uint_rev_buf(char* dst, f128_s n)
    {
        // n is a non-negative integer in f128
        const f128_s base = f128_s{ 1000000000.0 }; // 1e9

        int len = 0;

        if (n < f128_s{ 10.0 }) {
            const std::uint32_t d = detail::bounded_floor_to_u32(detail::fp::floor(n.hi), 9u);
            dst[len++] = static_cast<char>('0' + d);
            return len;
        }

        while (n >= base) {
            f128_s q = detail::_f128_impl::floor(n / base);
            f128_s r = n - q * base;

            std::uint32_t chunk = detail::bounded_floor_to_u32(detail::fp::floor(r.hi), 1000000000u);
            if (chunk >= 1000000000u) { chunk = 0; q = q + f128_s{ 1.0 }; }

            for (int i = 0; i < 9; ++i) {
                const std::uint32_t next = chunk / 10u;
                const std::uint32_t d = chunk - next * 10u;
                dst[len++] = static_cast<char>('0' + d);
                chunk = next;
            }

            n = q;
        }

        len += detail::append_uint32_rev(
            dst + len,
            detail::bounded_floor_to_u32(detail::fp::floor(n.hi), 999999999u));

        return len;
    }

    struct exact_traits
    {
        using value_type = f128_s;
        static constexpr int limb_count       = 2;
        static constexpr int significand_bits = 106;
        static constexpr int max_binary_exponent = 1023;
        static constexpr int min_binary_exponent = -1074;

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
            if (q.bit_length() > significand_bits)
            {
                const std::uint64_t c2 = q.get_bits(0, 53);
                const std::uint64_t c1 = q.get_bits(53, 53);
                const std::uint64_t c0 = q.get_bits(106, 53);

                const double hi = c0 ? detail::fp::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
                const double mid = c1 ? detail::fp::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
                const double lo = c2 ? detail::fp::ldexp(static_cast<double>(c2), e2 - 158) : 0.0;

                const f128_s tail = detail::_f128::renorm(mid, lo);
                f128_s out = detail::_f128::renorm(hi, tail.hi);
                out = detail::_f128::renorm(out.hi, out.lo + tail.lo);
                if (neg)
                    out = -out;
                return out;
            }

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
        if (sig < 1)
            sig = 1;

        detail::exact_decimal::biguint magnitude;
        int common_exp = 0;
        bool exact_neg = false;
        if (sig <= 32 &&
            detail::exact_decimal::exact_binary_components<exact_traits>(x, magnitude, common_exp, exact_neg) &&
            !exact_neg)
        {
            exp10 = detail::exact_decimal::decimal_exponent_from_components(magnitude, common_exp);
            if (exp10 >= 10 && exp10 <= 32)
            {
                detail::exact_decimal::biguint coefficient;
                int exact_exp10 = exp10;
                if (detail::exact_decimal::exact_significant_decimal<exact_traits>(
                        magnitude,
                        common_exp,
                        sig,
                        coefficient,
                        exact_exp10))
                {
                    exp10 = exact_exp10;
                    digits = detail::exact_decimal::to_decimal_string<String>(coefficient);
                    if (static_cast<int>(digits.size()) < sig)
                    {
                        const std::size_t zero_pad_count = static_cast<std::size_t>(sig - static_cast<int>(digits.size()));
                        digits.insert(0, zero_pad_count, '0');
                    }
                    return true;
                }
            }

            f128_s m = x * bl::detail::_f128_impl::pow10_128(-exp10);
            if (detail::fp::isfinite(m.hi))
            {
                detail::exact_decimal::biguint candidate;
                for (int i = 0; i < sig; ++i)
                {
                    int digit = static_cast<int>(detail::fp::floor(m.hi));
                    if (digit < 0) digit = 0;
                    else if (digit > 9) digit = 9;

                    candidate.mul_small(10);
                    candidate.add_small(static_cast<std::uint32_t>(digit));

                    m = mul_by_double_print(sub_by_double_print(m, static_cast<double>(digit)), 10.0);
                }

                detail::exact_decimal::biguint coefficient;
                int corrected_exp10 = exp10;
                if (detail::exact_decimal::exact_significant_decimal_from_floor_candidate<exact_traits>(
                        magnitude,
                        common_exp,
                        sig,
                        candidate,
                        corrected_exp10,
                        coefficient))
                {
                    exp10 = corrected_exp10;
                    digits = detail::exact_decimal::to_decimal_string<String>(coefficient);
                    if (static_cast<int>(digits.size()) < sig)
                    {
                        const std::size_t zero_pad_count = static_cast<std::size_t>(sig - static_cast<int>(digits.size()));
                        digits.insert(0, zero_pad_count, '0');
                    }
                    return true;
                }
            }
        }

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

        f128_s ip = detail::_f128_impl::floor(x);
        f128_s fp = x - ip;

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
        if (iszero(x))
            return detail::emit_single_zero_to_chars(first, last);

        if (sig_digits < 1) sig_digits = 1;
        const bool neg = (x.hi < 0.0);
        const f128_s v = neg ? -x : x;
        const int sig = static_cast<int>(sig_digits);
        char approximate_digits[40]{};
        bl::f128_io_string exact_digits;
        const char* digit_data = approximate_digits;
        int e = 0;
        bool have_digits = false;
        if (detail::_f128::should_try_approximate_scientific_digits(v, sig) &&
            detail::_f128::approximate_scientific_digits(
                v,
                sig,
                approximate_digits,
                static_cast<int>(sizeof(approximate_digits)),
                e) &&
            detail::_f128::should_accept_approximate_scientific_digits(sig, e))
        {
            have_digits = true;
        }

        if (!have_digits) {
            if (!detail::_f128::exact_scientific_digits(v, sig, exact_digits, e))
            {
                if (first >= last) return { first, false };
                *first = '0';
                return { first + 1, true };
            }
            digit_data = exact_digits.data();
        }

        return detail::emit_scientific_digits_to_chars(
            first,
            last,
            neg,
            digit_data,
            sig,
            e,
            strip_trailing_zeros);
    }

    struct f128_io_traits
    {
        using value_type = f128_s;
        using string_type = bl::f128_io_string;
        using exact_decimal_traits = detail::_f128::exact_traits;

        static constexpr int max_parse_order = detail::fltx_max_parse_order;
        static constexpr int min_parse_order = detail::fltx_min_parse_order;
        static constexpr int limb_count = detail::_f128::exact_traits::limb_count;
        static constexpr int significand_bits = detail::_f128::exact_traits::significand_bits;
        static constexpr int max_binary_exponent = 1023;
        static constexpr int min_normal_binary_exponent = -1022;
        static constexpr int min_binary_exponent = -1074;
        static constexpr int max_fixed_integer_digits = 309;

        static constexpr double limb(const value_type& x, int index) noexcept { return detail::_f128::exact_traits::limb(x, index); }
        static constexpr bool isnan(const value_type& x)       noexcept { return bl::isnan(x); }
        static constexpr bool isinf(const value_type& x)       noexcept { return bl::isinf(x); }
        static constexpr bool iszero(const value_type& x)      noexcept { return bl::iszero(x); }
        static constexpr bool is_negative(const value_type& x) noexcept { return detail::_f128::signbit(x.hi); }
        static constexpr value_type abs(const value_type& x)   noexcept { return (x.hi < 0.0) ? -x : x; }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 }; }
        static constexpr value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }

        static constexpr value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }
        static constexpr detail::fltx_char_result to_chars_general(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_general_decimal_for_traits<f128_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_fixed(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_fixed_decimal_for_traits<f128_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_fixed_fast(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_default_fixed(
            char* first,
            char* last,
            const value_type& x,
            int precision,
            int significant_digits,
            int exp10,
            bool strip_trailing_zeros)
        {
            if (significant_digits <= 16 && exp10 >= 10)
            {
                return detail::emit_exact_fixed_decimal_to_chars<exact_decimal_traits, string_type>(
                    first,
                    last,
                    x,
                    precision,
                    strip_trailing_zeros,
                    iszero(x),
                    is_negative(x));
            }

            return to_chars_fixed_fast(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr detail::fltx_char_result to_chars_scientific_frac(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_scientific_frac_for_traits<f128_io_traits>(first, last, x, precision, strip_trailing_zeros);
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

        static constexpr value_type exact_decimal_to_value(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg)
        {
            return detail::exact_decimal::exact_decimal_to_value<detail::_f128::exact_traits>(coeff, dec_exp, neg);
        }

        static constexpr value_type pack_from_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return detail::_f128::exact_traits::pack_from_significand(q, e2, neg);
        }
    };

    [[nodiscard]] BL_MSVC_NOINLINE constexpr bool parse(const char* s, f128_s& out, const char** endptr = nullptr) noexcept
    {
        return detail::parse_flt<f128_io_traits>(s, out, endptr);
    }

} // namespace detail::_f128

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s to_f128(const char* s) noexcept
{
    f128_s ret;
    if (detail::_f128::parse(s, ret))
        return ret;
    return f128_s{ 0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(const std::string& s) noexcept
{
    return to_f128(s.c_str());
}

[[nodiscard]] constexpr bl::f128_io_string to_static_string(
    const f128_s& value,
    int precision = std::numeric_limits<f128_s>::digits10,
    std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
{
    return detail::to_static_string_impl<detail::_f128::f128_io_traits>(value, precision, flags);
}

[[nodiscard]] inline std::string to_string(
    const f128_s& value,
    precision_info precision = std::numeric_limits<f128_s>::digits10,
    std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
{
    return detail::to_string_impl<detail::_f128::f128_io_traits>(value, precision, flags);
}

namespace detail::_f128 // primitives and kernels
{
    [[nodiscard]] consteval f128_s parse_dd_literal(const char* text, const char* expected_end)
    {
        f128_s out{};
        const char* end = text;

        if (!(parse(text, out, &end) && end == expected_end))
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
    if (!detail::_f128::parse(text, *this, &end))
        throw "invalid f128";
}

} // namespace bl

#endif
