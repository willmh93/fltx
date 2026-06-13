/**
 * fltx/format.h - std::format integration for fltx types.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_FORMAT_INCLUDED
#define FLTX_FORMAT_INCLUDED

#include <cstddef>
#include <ios>
#include <iterator>
#include <limits>
#include <string>

#include "fltx/string.h"

#if defined(__has_include)
#  if __has_include(<format>)
#    include <format>
#  endif
#endif

#if defined(__cpp_lib_format)
#  define FLTX_HAS_STD_FORMAT 1
#else
#  define FLTX_HAS_STD_FORMAT 0
#endif

#if FLTX_HAS_STD_FORMAT

namespace bl::detail::format
{
    struct standard_format_spec
    {
        char fill = ' ';
        char align = '>';
        char sign = '-';
        char type = '\0';
        int width = 0;
        int precision = -1;
        bool alternate = false;
        bool zero_pad = false;

        template<class ParseContext>
        constexpr auto parse(ParseContext& ctx)
        {
            auto it = ctx.begin();
            const auto end = ctx.end();

            if (it == end || *it == '}')
                return it;

            auto next = it;
            ++next;
            if (next != end && (*next == '<' || *next == '>' || *next == '^'))
            {
                fill = *it;
                align = *next;
                it = next;
                ++it;
            }
            else if (*it == '<' || *it == '>' || *it == '^')
            {
                align = *it;
                ++it;
            }

            if (it != end && (*it == '+' || *it == '-' || *it == ' '))
            {
                sign = *it;
                ++it;
            }

            if (it != end && *it == '#')
            {
                alternate = true;
                ++it;
            }

            if (it != end && *it == '0')
            {
                zero_pad = true;
                fill = '0';
                align = '>';
                ++it;
            }

            while (it != end && *it >= '0' && *it <= '9')
            {
                width = width * 10 + (*it - '0');
                ++it;
            }

            if (it != end && *it == '.')
            {
                ++it;
                if (it == end || *it < '0' || *it > '9')
                    throw std::format_error("invalid fltx format precision");

                precision = 0;
                while (it != end && *it >= '0' && *it <= '9')
                {
                    precision = precision * 10 + (*it - '0');
                    ++it;
                }
            }

            if (it != end && *it != '}')
            {
                type = *it;
                ++it;
            }

            if (it != end && *it != '}')
                throw std::format_error("invalid fltx format specifier");

            switch (type)
            {
            case '\0':
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
            case 'a':
            case 'A':
                return it;
            default:
                throw std::format_error("unsupported fltx format type");
            }
        }
    };

    template<class Value>
    struct io_traits_for;

    template<>
    struct io_traits_for<bl::f128_s>
    {
        using type = bl::detail::_f128::f128_io_traits;
    };

    template<>
    struct io_traits_for<bl::f256_s>
    {
        using type = bl::detail::_f256::f256_io_traits;
    };

    [[nodiscard]] inline bool has_sign_prefix(const std::string& text) noexcept
    {
        return !text.empty() && (text[0] == '-' || text[0] == '+' || text[0] == ' ');
    }

    inline void apply_padding(std::string& text, const standard_format_spec& spec)
    {
        if (spec.width <= 0 || text.size() >= static_cast<std::size_t>(spec.width))
            return;

        const std::size_t pad = static_cast<std::size_t>(spec.width) - text.size();
        if (spec.align == '<')
        {
            text.append(pad, spec.fill);
            return;
        }

        if (spec.align == '^')
        {
            const std::size_t left = pad / 2;
            const std::size_t right = pad - left;
            text.insert(0, left, spec.fill);
            text.append(right, spec.fill);
            return;
        }

        if (spec.zero_pad && spec.fill == '0' && has_sign_prefix(text))
        {
            text.insert(1, pad, '0');
            return;
        }

        text.insert(0, pad, spec.fill);
    }

    [[nodiscard]] inline bool is_uppercase_type(char type) noexcept
    {
        return type >= 'A' && type <= 'Z';
    }

    template<class Value>
    [[nodiscard]] inline std::string format_hex_value_to_string(const Value& value, const standard_format_spec& spec)
    {
        using Traits = typename io_traits_for<Value>::type;

        std::string text;
        const bool uppercase = is_uppercase_type(spec.type);
        if (const char* special = special_text<Traits>(value, uppercase)) [[unlikely]]
        {
            text = special;
        }
        else
        {
            const int precision = spec.precision >= 0 ? spec.precision : -1;
            const bool precision_specified = spec.precision >= 0;
            text.resize(bl::detail::format_capacity<Traits>(
                precision,
                bl::detail::format_kind::hex_float));

            const auto result = bl::detail::emit_hex_float_to_chars<Traits>(
                text.data(),
                text.data() + text.size(),
                value,
                true,
                precision,
                precision_specified,
                !precision_specified);

            if (!result.ok)
                throw std::format_error("unable to format fltx hexfloat value");

            text.resize(static_cast<std::size_t>(result.ptr - text.data()));
            if (spec.alternate)
                bl::detail::ensure_decimal_point(text);
        }

        bl::detail::apply_stream_decorations(text, spec.sign == '+', uppercase);
        if (spec.sign == ' ' && !has_sign_prefix(text))
            text.insert(0, 1, ' ');
        return text;
    }

    template<class Value>
    [[nodiscard]] inline std::string format_value_to_string(const Value& value, const standard_format_spec& spec)
    {
        const char lower_type = is_uppercase_type(spec.type)
            ? static_cast<char>(spec.type + ('a' - 'A'))
            : spec.type;

        if (lower_type == 'a')
        {
            std::string text = format_hex_value_to_string(value, spec);
            apply_padding(text, spec);
            return text;
        }

        std::ios_base::fmtflags flags{};
        if (lower_type == 'f')
            flags |= std::ios_base::fixed;
        else if (lower_type == 'e')
            flags |= std::ios_base::scientific;

        if (spec.alternate)
            flags |= std::ios_base::showpoint;
        if (spec.sign == '+')
            flags |= std::ios_base::showpos;
        if (is_uppercase_type(spec.type))
            flags |= std::ios_base::uppercase;

        const bool explicit_general = lower_type == 'g';
        const bool explicit_float_format = lower_type == 'f' || lower_type == 'e';
        const bool precision_defaults_to_six = explicit_general || explicit_float_format;
        const int precision = spec.precision >= 0
            ? spec.precision
            : (precision_defaults_to_six ? 6 : std::numeric_limits<Value>::max_digits10);

        std::string text = bl::to_string(value, precision, flags);
        if (spec.sign == ' ' && (text.empty() || text[0] != '-'))
            text.insert(0, 1, ' ');

        apply_padding(text, spec);
        return text;
    }

    template<class Value, class FormatContext>
    typename FormatContext::iterator format_value(
        const Value& value,
        const standard_format_spec& spec,
        FormatContext& ctx)
    {
        const std::string text = format_value_to_string(value, spec);
        return std::copy(text.begin(), text.end(), ctx.out());
    }

} // namespace bl::detail::format

template<>
struct std::formatter<bl::f128_s, char>
{
    bl::detail::format::standard_format_spec spec{};

    constexpr auto parse(std::format_parse_context& ctx)
    {
        return spec.parse(ctx);
    }

    template<class FormatContext>
    auto format(const bl::f128_s& value, FormatContext& ctx) const
    {
        return bl::detail::format::format_value(value, spec, ctx);
    }
};

template<>
struct std::formatter<bl::f128, char>
{
    bl::detail::format::standard_format_spec spec{};

    constexpr auto parse(std::format_parse_context& ctx)
    {
        return spec.parse(ctx);
    }

    template<class FormatContext>
    auto format(const bl::f128& value, FormatContext& ctx) const
    {
        return bl::detail::format::format_value(static_cast<const bl::f128_s&>(value), spec, ctx);
    }
};

template<>
struct std::formatter<bl::f256_s, char>
{
    bl::detail::format::standard_format_spec spec{};

    constexpr auto parse(std::format_parse_context& ctx)
    {
        return spec.parse(ctx);
    }

    template<class FormatContext>
    auto format(const bl::f256_s& value, FormatContext& ctx) const
    {
        return bl::detail::format::format_value(value, spec, ctx);
    }
};

template<>
struct std::formatter<bl::f256, char>
{
    bl::detail::format::standard_format_spec spec{};

    constexpr auto parse(std::format_parse_context& ctx)
    {
        return spec.parse(ctx);
    }

    template<class FormatContext>
    auto format(const bl::f256& value, FormatContext& ctx) const
    {
        return bl::detail::format::format_value(static_cast<const bl::f256_s&>(value), spec, ctx);
    }
};

#endif

#endif
