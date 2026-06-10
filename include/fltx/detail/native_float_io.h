/**
 * fltx/detail/native_float_io.h - Native f32/f64 constexpr string formatting traits.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_NATIVE_FLOAT_IO_INCLUDED
#define FLTX_DETAIL_NATIVE_FLOAT_IO_INCLUDED

#include <bit>
#include <cstdint>
#include <limits>

#include "fltx/aliases.h"
#include "fltx/detail/common_io.h"
#include "fltx/detail/f32_math_basic.h"

namespace bl::detail::_native_float_io
{
    template<class Traits>
    [[nodiscard]] constexpr bool exact_scaled_integer_digits(
        const typename Traits::value_type& x,
        int decimal_places,
        bl::default_io_string& digits,
        bool& neg) noexcept
    {
        if (decimal_places < 0)
            decimal_places = 0;

        neg = Traits::is_negative(x);
        int common_exp = std::numeric_limits<int>::max();
        bool have_term = false;

        for (int i = 0; i < Traits::limb_count; ++i)
        {
            const double limb = Traits::limb(x, i);
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = exact_decimal::decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            common_exp = (exponent < common_exp) ? exponent : common_exp;
            have_term = true;
        }

        if (!have_term)
        {
            digits = "0";
            return true;
        }

        exact_decimal::signed_biguint acc{};
        for (int i = 0; i < Traits::limb_count; ++i)
        {
            const double limb = Traits::limb(x, i);
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = exact_decimal::decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            exact_decimal::biguint term{ mantissa };
            term.shl_bits(exponent - common_exp);
            exact_decimal::add_signed(acc, term, limb_neg);
        }

        if (acc.mag.is_zero())
        {
            digits = "0";
            return true;
        }

        neg = acc.neg;

        exact_decimal::biguint num = acc.mag;
        exact_decimal::biguint den{ 1 };
        if (common_exp >= 0)
            num.shl_bits(common_exp);
        else
            den.shl_bits(-common_exp);

        for (int i = 0; i < decimal_places; ++i)
            num.mul_small(10);

        exact_decimal::biguint q;
        exact_decimal::biguint r;
        exact_decimal::divmod_bitwise(num, den, q, r);
        if (!r.is_zero())
        {
            exact_decimal::biguint twice_r = r;
            twice_r.shl1();
            const int cmp = exact_decimal::compare(twice_r, den);
            if (cmp > 0 || (cmp == 0 && q.is_odd()))
                q.add_small(1);
        }

        digits = exact_decimal::to_decimal_string(q);
        return true;
    }

    template<class Traits>
    [[nodiscard]] constexpr fltx_char_result emit_fixed_to_chars(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int precision,
        bool strip_trailing_zeros) noexcept
    {
        if (precision < 0)
            precision = 0;

        bl::default_io_string scaled_digits;
        bool neg = false;
        if (!exact_scaled_integer_digits<Traits>(x, precision, scaled_digits, neg))
            return { first, false };

        const bool zero_value = Traits::iszero(x);
        const int scaled_len = static_cast<int>(scaled_digits.size());
        const int integer_digits = (scaled_len > precision) ? (scaled_len - precision) : 0;
        const int leading_fraction_zeros = (scaled_len < precision) ? (precision - scaled_len) : 0;

        int fraction_len = precision;
        if (strip_trailing_zeros)
        {
            while (fraction_len > 0)
            {
                char c = '0';
                const int fraction_index = fraction_len - 1;
                if (scaled_len > precision)
                {
                    c = scaled_digits[integer_digits + fraction_index];
                }
                else if (fraction_index >= leading_fraction_zeros)
                {
                    c = scaled_digits[fraction_index - leading_fraction_zeros];
                }

                if (c != '0')
                    break;
                --fraction_len;
            }
        }

        const bool rounded_to_plain_zero = scaled_len == 1 && scaled_digits[0] == '0' && fraction_len == 0 && !zero_value;
        const bool emit_sign = neg && !rounded_to_plain_zero;
        const std::size_t needed = static_cast<std::size_t>(emit_sign ? 1 : 0)
            + static_cast<std::size_t>(integer_digits > 0 ? integer_digits : 1)
            + static_cast<std::size_t>(fraction_len > 0 ? 1 + fraction_len : 0);
        if (static_cast<std::size_t>(last - first) < needed)
            return { first, false };

        char* p = first;
        if (emit_sign)
            *p++ = '-';

        if (integer_digits > 0)
        {
            copy_chars(p, scaled_digits.data(), static_cast<std::size_t>(integer_digits));
            p += integer_digits;
        }
        else
        {
            *p++ = '0';
        }

        if (fraction_len > 0)
        {
            *p++ = '.';
            for (int i = 0; i < fraction_len; ++i)
            {
                char c = '0';
                if (scaled_len > precision)
                {
                    c = scaled_digits[integer_digits + i];
                }
                else if (i >= leading_fraction_zeros)
                {
                    c = scaled_digits[i - leading_fraction_zeros];
                }
                *p++ = c;
            }
        }

        return { p, true };
    }

    template<class Traits>
    [[nodiscard]] constexpr fltx_char_result emit_scientific_sig_to_chars(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int sig_digits,
        bool strip_trailing_zeros) noexcept
    {
        if (Traits::iszero(x))
        {
            if (first >= last)
                return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (sig_digits < 1)
            sig_digits = 1;

        const bool neg = Traits::is_negative(x);
        const typename Traits::value_type v = neg ? -x : x;

        bl::default_io_string digits;
        int e10 = 0;
        if (!exact_decimal::exact_scientific_digits<Traits>(v, sig_digits, digits, e10))
        {
            if (first >= last)
                return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        int last_frac = sig_digits - 1;
        if (sig_digits > 1 && strip_trailing_zeros)
        {
            while (last_frac >= 1 && digits[last_frac] == '0')
                --last_frac;
        }

        char exp_buf[16];
        char* ep = exp_buf;
        char* eend = exp_buf + sizeof(exp_buf);
        auto er = append_exp10_to_chars(ep, eend, e10);
        if (!er.ok)
            return { first, false };

        const int exp_len = static_cast<int>(er.ptr - ep);
        const bool has_frac = (sig_digits > 1) && (last_frac >= 1);
        const std::size_t needed = static_cast<std::size_t>(neg ? 1 : 0)
            + 1u
            + static_cast<std::size_t>(has_frac ? 1 + last_frac : 0)
            + static_cast<std::size_t>(exp_len);
        if (static_cast<std::size_t>(last - first) < needed)
            return { first, false };

        char* p = first;
        if (neg)
            *p++ = '-';

        *p++ = digits[0];
        if (has_frac)
        {
            *p++ = '.';
            copy_chars(p, digits.data() + 1, static_cast<std::size_t>(last_frac));
            p += last_frac;
        }

        copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
        p += exp_len;
        return { p, true };
    }

    template<class Traits>
    [[nodiscard]] constexpr fltx_char_result emit_scientific_frac_to_chars(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int frac_digits,
        bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0)
            frac_digits = 0;

        if (Traits::iszero(x))
        {
            const bool neg = Traits::is_negative(x);
            const int frac_len = strip_trailing_zeros ? 0 : frac_digits;

            char exp_buf[16];
            char* ep = exp_buf;
            char* eend = exp_buf + sizeof(exp_buf);
            auto er = append_exp10_to_chars(ep, eend, 0);
            if (!er.ok)
                return { first, false };

            const int exp_len = static_cast<int>(er.ptr - ep);
            const std::size_t needed = static_cast<std::size_t>(neg ? 1 : 0)
                + 1u
                + static_cast<std::size_t>(frac_len > 0 ? 1 + frac_len : 0)
                + static_cast<std::size_t>(exp_len);
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

            copy_chars(p, exp_buf, static_cast<std::size_t>(exp_len));
            p += exp_len;
            return { p, true };
        }

        return emit_scientific_sig_to_chars<Traits>(first, last, x, frac_digits + 1, strip_trailing_zeros);
    }

    template<class Traits>
    [[nodiscard]] constexpr fltx_char_result emit_general_to_chars(
        char* first,
        char* last,
        const typename Traits::value_type& x,
        int precision,
        bool strip_trailing_zeros) noexcept
    {
        if (precision < 0)
            precision = 0;

        const int sig = (precision == 0) ? 1 : precision;
        if (Traits::iszero(x))
        {
            if (first >= last)
                return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        bl::default_io_string digits;
        int e10 = 0;
        const typename Traits::value_type ax = Traits::abs(x);
        if (!exact_decimal::exact_scientific_digits<Traits>(ax, 1, digits, e10))
        {
            if (first >= last)
                return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (e10 >= -4 && e10 < sig)
        {
            const int frac = (sig > e10 + 1) ? (sig - (e10 + 1)) : 0;
            return emit_fixed_to_chars<Traits>(first, last, x, frac, strip_trailing_zeros);
        }

        return emit_scientific_sig_to_chars<Traits>(first, last, x, sig, strip_trailing_zeros);
    }

    struct f64_io_traits
    {
        using value_type = f64;

        static constexpr int max_parse_order = 308;
        static constexpr int min_parse_order = -324;
        static constexpr int limb_count = 1;
        static constexpr int significand_bits = 53;
        static constexpr int max_binary_exponent = 1023;
        static constexpr int min_binary_exponent = -1074;

        static constexpr double limb(value_type x, int) noexcept { return x; }
        static constexpr bool isnan(value_type x) noexcept { return fp::isnan(x); }
        static constexpr bool isinf(value_type x) noexcept { return fp::isinf(x); }
        static constexpr bool iszero(value_type x) noexcept { return x == 0.0; }
        static constexpr bool is_negative(value_type x) noexcept { return fp::signbit(x); }
        static constexpr value_type abs(value_type x) noexcept { return fp::fabs(x); }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? -0.0 : 0.0; }
        static constexpr value_type infinity(bool neg = false) noexcept { return neg ? -std::numeric_limits<value_type>::infinity() : std::numeric_limits<value_type>::infinity(); }
        static constexpr value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }

        static constexpr fltx_char_result to_chars_general(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_general_to_chars<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_fixed(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_to_chars<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_frac(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_frac_to_chars<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_sig(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_sig_to_chars<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr value_type exact_uint64_to_value(std::uint64_t value, bool neg) noexcept
        {
            const value_type out = static_cast<value_type>(value);
            return neg ? -out : out;
        }

        static constexpr bool compact_decimal_to_value(std::uint64_t coeff, int dec_exp, bool neg, value_type& out) noexcept
        {
            return exact_decimal::compact_decimal_to_value<f64_io_traits>(coeff, dec_exp, neg, out);
        }

        static constexpr value_type exact_decimal_to_value(const fltx_parse_token::coeff_type& coeff, int dec_exp, bool neg) noexcept
        {
            return detail::_f64_impl::exact_decimal_to_native_value<detail::_f64_impl::f64_decimal_round_traits>(coeff, dec_exp, neg);
        }

        static constexpr value_type pack_from_significand(const exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return detail::_f64_impl::exact_dyadic_to_double(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

    struct f32_io_traits
    {
        using value_type = f32;

        static constexpr int max_parse_order = 38;
        static constexpr int min_parse_order = -46;
        static constexpr int limb_count = 1;
        static constexpr int significand_bits = 24;
        static constexpr int max_binary_exponent = 127;
        static constexpr int min_binary_exponent = -149;

        static constexpr double limb(value_type x, int) noexcept { return static_cast<double>(x); }
        static constexpr bool isnan(value_type x) noexcept { return fp::isnan(x); }
        static constexpr bool isinf(value_type x) noexcept { return fp::isinf(x); }
        static constexpr bool iszero(value_type x) noexcept { return x == 0.0f; }
        static constexpr bool is_negative(value_type x) noexcept { return fp::signbit(x); }
        static constexpr value_type abs(value_type x) noexcept { return fp::fabs(x); }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? -0.0f : 0.0f; }
        static constexpr value_type infinity(bool neg = false) noexcept { return neg ? -std::numeric_limits<value_type>::infinity() : std::numeric_limits<value_type>::infinity(); }
        static constexpr value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }

        static constexpr fltx_char_result to_chars_general(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_general_to_chars<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_fixed(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_to_chars<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_frac(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_frac_to_chars<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_sig(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_sig_to_chars<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr value_type exact_uint64_to_value(std::uint64_t value, bool neg) noexcept
        {
            const value_type out = static_cast<value_type>(value);
            return neg ? -out : out;
        }

        static constexpr bool compact_decimal_to_value(std::uint64_t coeff, int dec_exp, bool neg, value_type& out) noexcept
        {
            return exact_decimal::compact_decimal_to_value<f32_io_traits>(coeff, dec_exp, neg, out);
        }

        static constexpr value_type exact_decimal_to_value(const fltx_parse_token::coeff_type& coeff, int dec_exp, bool neg) noexcept
        {
            return detail::_f64_impl::exact_decimal_to_native_value<detail::_f32_impl::f32_decimal_round_traits>(coeff, dec_exp, neg);
        }

        static constexpr value_type pack_from_significand(const exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return detail::_f32_impl::exact_dyadic_to_float(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

} // namespace bl::detail::_native_float_io

#endif
