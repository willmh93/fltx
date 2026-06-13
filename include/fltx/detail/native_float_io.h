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

#include <cstdint>
#include <limits>

#include "fltx/aliases.h"
#include "fltx/detail/common_io.h"
#include "fltx/detail/native_float_decimal.h"

namespace bl::detail::_native_float_io
{
    struct f64_io_traits
    {
        using value_type = f64;
        using string_type = bl::f64_io_string;
        using exact_decimal_traits = f64_io_traits;

        static constexpr int max_parse_order = 308;
        static constexpr int min_parse_order = -324;
        static constexpr int limb_count = 1;
        static constexpr int significand_bits = 53;
        static constexpr int max_binary_exponent = 1023;
        static constexpr int min_normal_binary_exponent = -1022;
        static constexpr int min_binary_exponent = -1074;
        static constexpr int max_fixed_integer_digits = 309;

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
            return detail::emit_general_decimal_for_traits<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_fixed(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_exact_fixed_decimal_to_chars<f64_io_traits, string_type>(
                first,
                last,
                x,
                precision,
                strip_trailing_zeros,
                iszero(x),
                is_negative(x));
        }

        static constexpr fltx_char_result to_chars_default_fixed(
            char* first,
            char* last,
            value_type x,
            int precision,
            int,
            int,
            bool strip_trailing_zeros)
        {
            return to_chars_fixed(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_frac(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_scientific_frac_for_traits<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_sig(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_scientific_sig_for_traits<f64_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr bool scientific_digits(value_type x, int sig, string_type& digits, int& exp10)
        {
            return exact_decimal::exact_scientific_digits<f64_io_traits>(x, sig, digits, exp10);
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

        static constexpr value_type exact_decimal_to_value(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg) noexcept
        {
            return detail::_native_float_decimal::exact_decimal_to_native_value<detail::_native_float_decimal::f64_decimal_round_traits>(coeff, dec_exp, neg);
        }

        static constexpr value_type pack_from_significand(const exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return detail::_native_float_decimal::exact_dyadic_to_double(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

    struct f32_io_traits
    {
        using value_type = f32;
        using string_type = bl::f32_io_string;
        using exact_decimal_traits = f32_io_traits;

        static constexpr int max_parse_order = 38;
        static constexpr int min_parse_order = -46;
        static constexpr int limb_count = 1;
        static constexpr int significand_bits = 24;
        static constexpr int max_binary_exponent = 127;
        static constexpr int min_normal_binary_exponent = -126;
        static constexpr int min_binary_exponent = -149;
        static constexpr int max_fixed_integer_digits = 39;

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
            return detail::emit_general_decimal_for_traits<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_fixed(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_exact_fixed_decimal_to_chars<f32_io_traits, string_type>(
                first,
                last,
                x,
                precision,
                strip_trailing_zeros,
                iszero(x),
                is_negative(x));
        }

        static constexpr fltx_char_result to_chars_default_fixed(
            char* first,
            char* last,
            value_type x,
            int precision,
            int,
            int,
            bool strip_trailing_zeros)
        {
            return to_chars_fixed(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_frac(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_scientific_frac_for_traits<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr fltx_char_result to_chars_scientific_sig(char* first, char* last, value_type x, int precision, bool strip_trailing_zeros)
        {
            return detail::emit_scientific_sig_for_traits<f32_io_traits>(first, last, x, precision, strip_trailing_zeros);
        }

        static constexpr bool scientific_digits(value_type x, int sig, string_type& digits, int& exp10)
        {
            return exact_decimal::exact_scientific_digits<f32_io_traits>(x, sig, digits, exp10);
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

        static constexpr value_type exact_decimal_to_value(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg) noexcept
        {
            return detail::_native_float_decimal::exact_decimal_to_native_value<detail::_native_float_decimal::f32_decimal_round_traits>(coeff, dec_exp, neg);
        }

        static constexpr value_type pack_from_significand(const exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return detail::_native_float_decimal::exact_dyadic_to_float(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

} // namespace bl::detail::_native_float_io

#endif
