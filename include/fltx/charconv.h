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
#include <system_error>

#include "fltx/f128_string.h"
#include "fltx/f256_string.h"

namespace bl::detail::charconv
{
    [[nodiscard]] BL_FORCE_INLINE constexpr bool supported_format(std::chars_format fmt) noexcept
    {
        return fmt == std::chars_format::general ||
               fmt == std::chars_format::fixed ||
               fmt == std::chars_format::scientific;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool is_ascii_space(char c) noexcept
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
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
        if (!supported_format(fmt))
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
        if (!supported_format(fmt))
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
        if (!detail::parse_flt<Traits>(buffer, parsed, &end) || end == buffer)
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

    [[nodiscard]] std::from_chars_result from_chars(
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

    [[nodiscard]] std::from_chars_result from_chars(
        const char* first,
        const char* last,
        f256_s& value,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        return detail::charconv::from_chars_impl<detail::_f256::f256_io_traits>(first, last, value, fmt);
    }

} // namespace bl

#endif
