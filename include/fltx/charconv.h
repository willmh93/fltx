/**
 * fltx/charconv.h - std::charconv-shaped parsing and formatting helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_CHARCONV_INCLUDED
#define FLTX_CHARCONV_INCLUDED

#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include "fltx/detail/native_float_io.h"
#include "fltx/f128_string.h"
#include "fltx/f256_string.h"

namespace bl::detail::charconv
{
    [[nodiscard]] BL_FORCE_INLINE constexpr bool supported_output_format(std::chars_format fmt) noexcept
    {
        return fmt == std::chars_format::general ||
               fmt == std::chars_format::fixed ||
               fmt == std::chars_format::scientific;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool supported_input_format(std::chars_format fmt) noexcept
    {
        return supported_output_format(fmt) || fmt == std::chars_format::hex;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool is_ascii_space(char c) noexcept
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr int hex_digit_value(char ch) noexcept
    {
        if ('0' <= ch && ch <= '9')
            return ch - '0';
        if ('a' <= ch && ch <= 'f')
            return 10 + (ch - 'a');
        if ('A' <= ch && ch <= 'F')
            return 10 + (ch - 'A');
        return -1;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool begins_special(const char* first) noexcept
    {
        if (*first == '-')
            ++first;

        return (ascii_lower(first[0]) == 'n' && ascii_lower(first[1]) == 'a' && ascii_lower(first[2]) == 'n') ||
               (ascii_lower(first[0]) == 'i' && ascii_lower(first[1]) == 'n' && ascii_lower(first[2]) == 'f');
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_exponent_marker(const char* first, const char* last) noexcept
    {
        for (const char* p = first; p != last; ++p)
        {
            if (*p == 'e' || *p == 'E')
                return true;
        }
        return false;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr std::size_t exponent_marker_offset(const char* first, std::size_t length) noexcept
    {
        for (std::size_t index = 0; index < length; ++index)
        {
            if (first[index] == 'e' || first[index] == 'E')
                return index;
        }
        return length;
    }

    template<class Traits>
    [[nodiscard]] constexpr typename Traits::value_type exact_binary_to_value(
        detail::exact_decimal::biguint coeff,
        int bin_exp,
        bool neg) noexcept
    {
        if (coeff.is_zero())
            return Traits::zero(neg);

        int ratio_exp = coeff.bit_length() - 1;
        detail::exact_decimal::biguint denominator{ 1 };
        detail::exact_decimal::biguint q = detail::exact_decimal::extract_rounded_significand_chunks(
            coeff,
            denominator,
            ratio_exp,
            Traits::significand_bits);

        if (q.bit_length() > Traits::significand_bits)
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

    template<class Traits>
    [[nodiscard]] constexpr bool parse_hex_float(
        const char* first,
        const char* last,
        typename Traits::value_type& value,
        const char** endptr) noexcept
    {
        const char* p = first;
        bool neg = false;
        if (*p == '-')
        {
            neg = true;
            ++p;
        }

        if (last - p >= 3 && detail::parse_special<Traits>(p, neg, value))
        {
            *endptr = p;
            return true;
        }

        detail::exact_decimal::biguint coeff;
        bool any_digit = false;
        bool fractional = false;
        int frac_hex_digits = 0;

        while (p != last)
        {
            if (*p == '.' && !fractional)
            {
                fractional = true;
                ++p;
                continue;
            }

            const int digit = hex_digit_value(*p);
            if (digit < 0)
                break;

            coeff.mul_small(16);
            if (digit != 0)
                coeff.add_small(static_cast<std::uint32_t>(digit));

            any_digit = true;
            if (fractional)
                ++frac_hex_digits;
            ++p;
        }

        if (!any_digit)
        {
            *endptr = first;
            return false;
        }

        int exp2 = 0;
        if (p != last && (*p == 'p' || *p == 'P'))
        {
            const char* exponent_marker = p;
            ++p;

            bool exp_neg = false;
            if (p != last && (*p == '+' || *p == '-'))
            {
                exp_neg = (*p == '-');
                ++p;
            }

            int parsed_exp = 0;
            bool any_exp_digit = false;
            while (p != last && '0' <= *p && *p <= '9')
            {
                any_exp_digit = true;
                if (parsed_exp < 100000000)
                    parsed_exp = parsed_exp * 10 + (*p - '0');
                ++p;
            }

            if (any_exp_digit)
                exp2 = exp_neg ? -parsed_exp : parsed_exp;
            else
                p = exponent_marker;
        }

        value = exact_binary_to_value<Traits>(coeff, exp2 - 4 * frac_hex_digits, neg);
        *endptr = p;
        return true;
    }

    template<class Traits>
    [[nodiscard]] constexpr std::to_chars_result emit_special(
        char* first,
        char* last,
        const typename Traits::value_type& value) noexcept
    {
        const char* text = special_text<Traits>(value);
        if (!text)
            return { first, std::errc::invalid_argument };

        std::size_t length = 0;
        while (text[length] != '\0')
            ++length;

        if (static_cast<std::size_t>(last - first) < length)
            return { last, std::errc::value_too_large };

        copy_chars(first, text, length);
        return { first + length, std::errc{} };
    }

    template<class Traits>
    [[nodiscard]] constexpr std::to_chars_result to_chars_impl(
        char* first,
        char* last,
        const typename Traits::value_type& value,
        std::chars_format fmt,
        int precision) noexcept
    {
        if (!supported_output_format(fmt))
            return { first, std::errc::invalid_argument };

        if (special_text<Traits>(value) != nullptr)
            return emit_special<Traits>(first, last, value);

        detail::fltx_char_result result{};
        if (fmt == std::chars_format::fixed)
            result = Traits::to_chars_fixed(first, last, value, precision, false);
        else if (fmt == std::chars_format::scientific)
            result = Traits::to_chars_scientific_frac(first, last, value, precision, false);
        else
            result = Traits::to_chars_general(first, last, value, precision, true);

        return result.ok
            ? std::to_chars_result{ result.ptr, std::errc{} }
            : std::to_chars_result{ last, std::errc::value_too_large };
    }

    template<class Traits>
    [[nodiscard]] std::from_chars_result from_chars_impl(
        const char* first,
        const char* last,
        typename Traits::value_type& value,
        std::chars_format fmt) noexcept
    {
        if (!supported_input_format(fmt))
            return { first, std::errc::invalid_argument };

        if (first == last || *first == '+' || is_ascii_space(*first))
            return { first, std::errc::invalid_argument };

        const std::size_t length = static_cast<std::size_t>(last - first);
        constexpr std::size_t stack_capacity = 1024;
        char stack[stack_capacity + 1]{};
        std::string dynamic;
        char* buffer = stack;

        if (length > stack_capacity)
        {
            dynamic.assign(first, last);
            dynamic.push_back('\0');
            buffer = dynamic.data();
        }
        else
        {
            copy_chars(buffer, first, length);
            buffer[length] = '\0';
        }

        if (fmt == std::chars_format::fixed)
        {
            const std::size_t exponent_offset = exponent_marker_offset(buffer, length);
            if (exponent_offset != length)
                buffer[exponent_offset] = '\0';
        }

        typename Traits::value_type parsed{};
        const char* end = nullptr;
        const bool parsed_ok = fmt == std::chars_format::hex
            ? parse_hex_float<Traits>(buffer, buffer + length, parsed, &end)
            : detail::parse_flt<Traits>(buffer, parsed, &end);

        if (!parsed_ok || end == buffer)
            return { first, std::errc::invalid_argument };

        if (fmt == std::chars_format::scientific &&
            !begins_special(buffer) &&
            !has_exponent_marker(buffer, end))
        {
            return { first, std::errc::invalid_argument };
        }

        value = parsed;
        return { first + (end - buffer), std::errc{} };
    }

} // namespace bl::detail::charconv

namespace bl
{
    template<class T>
    struct parse_result
    {
        T value{};
        std::size_t consumed = 0;
        std::errc ec{};

        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return ec == std::errc{};
        }
    };

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f32 value) noexcept
    {
        return std::to_chars(first, last, value);
    }

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f32 value,
        std::chars_format fmt) noexcept
    {
        return std::to_chars(first, last, value, fmt);
    }

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f32 value,
        std::chars_format fmt,
        int precision) noexcept
    {
        return std::to_chars(first, last, value, fmt, precision);
    }

    [[nodiscard]] inline std::from_chars_result from_chars(
        const char* first,
        const char* last,
        f32& value,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        return std::from_chars(first, last, value, fmt);
    }

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f64 value) noexcept
    {
        return std::to_chars(first, last, value);
    }

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f64 value,
        std::chars_format fmt) noexcept
    {
        return std::to_chars(first, last, value, fmt);
    }

    [[nodiscard]] inline std::to_chars_result to_chars(
        char* first,
        char* last,
        f64 value,
        std::chars_format fmt,
        int precision) noexcept
    {
        return std::to_chars(first, last, value, fmt, precision);
    }

    [[nodiscard]] inline std::from_chars_result from_chars(
        const char* first,
        const char* last,
        f64& value,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        return std::from_chars(first, last, value, fmt);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f128_s& value) noexcept
    {
        return detail::charconv::to_chars_impl<detail::_f128::f128_io_traits>(
            first,
            last,
            value,
            std::chars_format::general,
            std::numeric_limits<f128_s>::max_digits10);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f128_s& value,
        std::chars_format fmt) noexcept
    {
        const int precision = fmt == std::chars_format::general
            ? std::numeric_limits<f128_s>::max_digits10
            : 6;
        return detail::charconv::to_chars_impl<detail::_f128::f128_io_traits>(
            first,
            last,
            value,
            fmt,
            precision);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f128_s& value,
        std::chars_format fmt,
        int precision) noexcept
    {
        return detail::charconv::to_chars_impl<detail::_f128::f128_io_traits>(
            first,
            last,
            value,
            fmt,
            precision);
    }

    [[nodiscard]] inline std::from_chars_result from_chars(
        const char* first,
        const char* last,
        f128_s& value,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        return detail::charconv::from_chars_impl<detail::_f128::f128_io_traits>(first, last, value, fmt);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f256_s& value) noexcept
    {
        return detail::charconv::to_chars_impl<detail::_f256::f256_io_traits>(
            first,
            last,
            value,
            std::chars_format::general,
            std::numeric_limits<f256_s>::max_digits10);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f256_s& value,
        std::chars_format fmt) noexcept
    {
        const int precision = fmt == std::chars_format::general
            ? std::numeric_limits<f256_s>::max_digits10
            : 6;

        return detail::charconv::to_chars_impl<detail::_f256::f256_io_traits>(
            first,
            last,
            value,
            fmt,
            precision);
    }

    [[nodiscard]] constexpr std::to_chars_result to_chars(
        char* first,
        char* last,
        const f256_s& value,
        std::chars_format fmt,
        int precision) noexcept
    {
        return detail::charconv::to_chars_impl<detail::_f256::f256_io_traits>(
            first,
            last,
            value,
            fmt,
            precision);
    }

    [[nodiscard]] inline std::from_chars_result from_chars(
        const char* first,
        const char* last,
        f256_s& value,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        return detail::charconv::from_chars_impl<detail::_f256::f256_io_traits>(first, last, value, fmt);
    }

    namespace detail::charconv
    {
        template<class>
        struct dependent_false : std::false_type {};

        template<class T>
        struct parse_traits_for
        {
            static constexpr bool supported = false;
        };

        template<>
        struct parse_traits_for<f32>
        {
            static constexpr bool supported = true;
            using traits = detail::_native_float_io::f32_io_traits;
        };

        template<>
        struct parse_traits_for<f64>
        {
            static constexpr bool supported = true;
            using traits = detail::_native_float_io::f64_io_traits;
        };

        template<>
        struct parse_traits_for<f128_s>
        {
            static constexpr bool supported = true;
            using traits = detail::_f128::f128_io_traits;
        };

        template<>
        struct parse_traits_for<f128>
        {
            static constexpr bool supported = true;
            using traits = detail::_f128::f128_io_traits;
        };

        template<>
        struct parse_traits_for<f256_s>
        {
            static constexpr bool supported = true;
            using traits = detail::_f256::f256_io_traits;
        };

        template<>
        struct parse_traits_for<f256>
        {
            static constexpr bool supported = true;
            using traits = detail::_f256::f256_io_traits;
        };

        template<class Traits>
        [[nodiscard]] constexpr std::from_chars_result from_chars_constexpr_buffer(
            char* first,
            char* last,
            typename Traits::value_type& value,
            std::chars_format fmt) noexcept
        {
            if (!supported_input_format(fmt))
                return { first, std::errc::invalid_argument };

            if (first == last || *first == '+' || is_ascii_space(*first))
                return { first, std::errc::invalid_argument };

            if (fmt == std::chars_format::fixed)
            {
                const std::size_t length = static_cast<std::size_t>(last - first);
                const std::size_t exponent_offset = exponent_marker_offset(first, length);
                if (exponent_offset != length)
                    first[exponent_offset] = '\0';
            }

            typename Traits::value_type parsed{};
            const char* end = nullptr;
            const bool parsed_ok = fmt == std::chars_format::hex
                ? parse_hex_float<Traits>(first, last, parsed, &end)
                : detail::parse_flt<Traits>(first, parsed, &end);

            if (!parsed_ok || end == first)
                return { first, std::errc::invalid_argument };

            if (fmt == std::chars_format::scientific &&
                !begins_special(first) &&
                !has_exponent_marker(first, end))
            {
                return { first, std::errc::invalid_argument };
            }

            value = parsed;
            return { end, std::errc{} };
        }

        template<class T>
        [[nodiscard]] constexpr parse_result<T> parse_constexpr(
            std::string_view text,
            std::chars_format fmt) noexcept
        {
            using Traits = typename parse_traits_for<T>::traits;

            parse_result<T> result{};

            if (text.size() > default_io_string::static_capacity)
            {
                result.ec = std::errc::invalid_argument;
                return result;
            }

            default_io_string buffer{ text };
            typename Traits::value_type parsed{};
            const auto parsed_result = from_chars_constexpr_buffer<Traits>(
                buffer.data(),
                buffer.data() + buffer.size(),
                parsed,
                fmt);

            result.consumed = static_cast<std::size_t>(parsed_result.ptr - buffer.data());
            result.ec = parsed_result.ec;
            if (result.ec == std::errc{} && result.consumed != text.size())
                result.ec = std::errc::invalid_argument;

            if (result.ec == std::errc{})
                result.value = T{ parsed };

            return result;
        }

        template<class T>
        [[nodiscard]] parse_result<T> parse_runtime(
            std::string_view text,
            std::chars_format fmt) noexcept
        {
            parse_result<T> result{};

            if (text.empty())
            {
                result.ec = std::errc::invalid_argument;
                return result;
            }

            T parsed{};
            const auto parsed_result = bl::from_chars(text.data(), text.data() + text.size(), parsed, fmt);
            result.consumed = static_cast<std::size_t>(parsed_result.ptr - text.data());
            result.ec = parsed_result.ec;
            if (result.ec == std::errc{} && result.consumed != text.size())
                result.ec = std::errc::invalid_argument;

            if (result.ec == std::errc{})
                result.value = parsed;

            return result;
        }

    } // namespace detail::charconv

    template<class T>
    [[nodiscard]] constexpr parse_result<std::remove_cvref_t<T>> try_parse(
        std::string_view text,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        using value_type = std::remove_cvref_t<T>;

        if constexpr (!detail::charconv::parse_traits_for<value_type>::supported)
        {
            static_assert(
                detail::charconv::dependent_false<value_type>::value,
                "bl::parse<T> and bl::try_parse<T> support bl::f32, bl::f64, bl::f128, bl::f128_s, bl::f256, and bl::f256_s."
            );
        }
        else
        {
            if consteval
            {
                return detail::charconv::parse_constexpr<value_type>(text, fmt);
            }
            else
            {
                return detail::charconv::parse_runtime<value_type>(text, fmt);
            }
        }
    }

    template<class T>
    [[nodiscard]] constexpr bool try_parse(
        std::string_view text,
        T& out,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        using value_type = std::remove_cvref_t<T>;
        const auto parsed = try_parse<value_type>(text, fmt);
        if (!parsed)
            return false;

        out = parsed.value;
        return true;
    }

    template<class T>
    [[nodiscard]] constexpr std::remove_cvref_t<T> parse(
        std::string_view text,
        std::remove_cvref_t<T> fallback,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        const auto parsed = try_parse<T>(text, fmt);
        return parsed ? parsed.value : fallback;
    }

    template<class T>
    [[nodiscard]] constexpr std::remove_cvref_t<T> parse(
        std::string_view text,
        std::chars_format fmt = std::chars_format::general)
    {
        const auto parsed = try_parse<T>(text, fmt);
        if (!parsed)
            throw "invalid bl::parse input";
        return parsed.value;
    }

} // namespace bl

#endif
