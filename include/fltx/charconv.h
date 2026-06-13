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
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include "fltx/detail/ascii.h"
#include "fltx/detail/native_float_io.h"
#include "fltx/f128_string.h"
#include "fltx/f256_string.h"

namespace bl::detail::charconv
{
    [[nodiscard]] BL_FORCE_INLINE constexpr bool supported_output_format(std::chars_format fmt) noexcept
    {
        return fmt == std::chars_format::general ||
               fmt == std::chars_format::fixed ||
               fmt == std::chars_format::scientific ||
               fmt == std::chars_format::hex;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool supported_input_format(std::chars_format fmt) noexcept
    {
        return supported_output_format(fmt);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool begins_special(const char* first, const char* last = nullptr) noexcept
    {
        return signed_special_token_length(first, last) != 0;
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr std::from_chars_result parse_special_from_chars(
        const char* first,
        const char* last,
        typename Traits::value_type& value) noexcept
    {
        const char* p = first;
        bool neg = false;
        if (p != last && *p == '-')
        {
            neg = true;
            ++p;
        }

        if (!detail::parse_special<Traits>(p, last, neg, value))
            return { first, std::errc::invalid_argument };

        return { p, std::errc{} };
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE std::from_chars_result native_from_chars(
        const char* first,
        const char* last,
        typename Traits::value_type& value,
        std::chars_format fmt) noexcept
    {
        const auto result = std::from_chars(first, last, value, fmt);
        if (result.ec == std::errc::invalid_argument && begins_special(first, last)) [[unlikely]]
            return parse_special_from_chars<Traits>(first, last, value);
        return result;
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

    [[nodiscard]] BL_FORCE_INLINE constexpr int hex_digit_bit_width(int digit) noexcept
    {
        int width = 0;
        while (digit != 0)
        {
            ++width;
            digit >>= 1;
        }
        return width;
    }

    BL_FORCE_INLINE constexpr void append_bounded_hex_bit(
        detail::exact_decimal::biguint& coeff,
        int& kept_bits,
        int& total_bits,
        int max_kept_bits,
        bool bit,
        bool& sticky) noexcept
    {
        ++total_bits;
        if (kept_bits < max_kept_bits)
        {
            coeff.shl1();
            if (bit)
                coeff.add_small(1);
            ++kept_bits;
            return;
        }

        if (bit)
            sticky = true;
    }

    template<class Traits>
    BL_FORCE_INLINE constexpr void append_bounded_hex_digit(
        detail::exact_decimal::biguint& coeff,
        int digit,
        int& kept_bits,
        int& total_bits,
        bool& seen_nonzero,
        bool& sticky) noexcept
    {
        constexpr int max_kept_bits = detail::hex_full_significand_bits<Traits>() + 2;

        if (!seen_nonzero)
        {
            if (digit == 0)
                return;

            seen_nonzero = true;
            const int width = hex_digit_bit_width(digit);
            for (int bit_index = width - 1; bit_index >= 0; --bit_index)
            {
                append_bounded_hex_bit(
                    coeff,
                    kept_bits,
                    total_bits,
                    max_kept_bits,
                    ((digit >> bit_index) & 1) != 0,
                    sticky);
            }
            return;
        }

        for (int bit_index = 3; bit_index >= 0; --bit_index)
        {
            append_bounded_hex_bit(
                coeff,
                kept_bits,
                total_bits,
                max_kept_bits,
                ((digit >> bit_index) & 1) != 0,
                sticky);
        }
    }

    template<class Traits>
    [[nodiscard]] constexpr typename Traits::value_type exact_binary_to_value(
        detail::exact_decimal::biguint coeff,
        int bin_exp,
        bool neg) noexcept
    {
        if (coeff.is_zero())
            return Traits::zero(neg);

        constexpr int target_bits = detail::hex_full_significand_bits<Traits>();
        int ratio_exp = coeff.bit_length() - 1;
        detail::exact_decimal::biguint q = coeff;
        if (ratio_exp > target_bits - 1)
        {
            const int shift = ratio_exp - (target_bits - 1);
            const bool round_bit = coeff.get_bit(shift - 1);
            const bool sticky = shift > 1 && detail::exact_decimal::any_low_bits_set(coeff, shift - 1);

            q = detail::exact_decimal::shr_bits_copy(coeff, shift);
            if (round_bit && (sticky || q.is_odd()))
                q.add_small(1);

            if (q.bit_length() > target_bits)
            {
                q.shr1();
                ++ratio_exp;
            }
        }
        else if (ratio_exp < target_bits - 1)
        {
            q.shl_bits((target_bits - 1) - ratio_exp);
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
        const char** endptr,
        bool allow_prefix = false,
        bool allow_leading_plus = false) noexcept
    {
        const char* p = first;
        bool neg = false;
        if (*p == '-' || (allow_leading_plus && *p == '+'))
        {
            neg = *p == '-';
            ++p;
        }

        if (detail::parse_special<Traits>(p, last, neg, value)) [[unlikely]]
        {
            *endptr = p;
            return true;
        }

        if (allow_prefix && last - p >= 2 && p[0] == '0' && ascii_lower(p[1]) == 'x')
            p += 2;

        detail::exact_decimal::biguint coeff;
        bool any_digit = false;
        bool seen_nonzero = false;
        bool fractional = false;
        bool sticky = false;
        int kept_bits = 0;
        int total_bits = 0;
        int frac_hex_digits = 0;

        while (p != last)
        {
            if (*p == '.' && !fractional)
            {
                fractional = true;
                ++p;
                continue;
            }

            const int digit = ascii_hex_digit_value(*p);
            if (digit < 0)
                break;

            append_bounded_hex_digit<Traits>(
                coeff,
                digit,
                kept_bits,
                total_bits,
                seen_nonzero,
                sticky);

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

        const int discarded_bits = total_bits - kept_bits;
        if (sticky)
            coeff.set_bit(0);

        value = exact_binary_to_value<Traits>(coeff, exp2 - 4 * frac_hex_digits + discarded_bits, neg);
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

        if (special_text<Traits>(value) != nullptr) [[unlikely]]
            return emit_special<Traits>(first, last, value);

        detail::fltx_char_result result{};
        if (fmt == std::chars_format::fixed)
            result = Traits::to_chars_fixed(first, last, value, precision, false);
        else if (fmt == std::chars_format::scientific)
            result = Traits::to_chars_scientific_frac(first, last, value, precision, false);
        else if (fmt == std::chars_format::hex)
            result = detail::emit_hex_float_to_chars<Traits>(
                first,
                last,
                value,
                false,
                precision,
                precision >= 0,
                precision < 0);
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

        if (first == last || *first == '+' || ascii_space(*first))
            return { first, std::errc::invalid_argument };

        typename Traits::value_type parsed{};
        const char* end = nullptr;
        bool parsed_ok = false;
        if (fmt == std::chars_format::hex)
        {
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

            parsed_ok = parse_hex_float<Traits>(buffer, buffer + length, parsed, &end);
            end = first + (end - buffer);
        }
        else
        {
            const char* parse_last = last;
            if (fmt == std::chars_format::fixed)
            {
                const std::size_t length = static_cast<std::size_t>(last - first);
                const std::size_t exponent_offset = exponent_marker_offset(first, length);
                parse_last = first + exponent_offset;
            }
            parsed_ok = detail::parse_flt<Traits>(first, parse_last, parsed, &end);
        }

        if (!parsed_ok || end == first)
            return { first, std::errc::invalid_argument };

        if (fmt == std::chars_format::scientific &&
            !begins_special(first, last) &&
            !has_exponent_marker(first, end))
        {
            return { first, std::errc::invalid_argument };
        }

        value = parsed;
        return { end, std::errc{} };
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
        return detail::charconv::native_from_chars<detail::_native_float_io::f32_io_traits>(first, last, value, fmt);
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
        return detail::charconv::native_from_chars<detail::_native_float_io::f64_io_traits>(first, last, value, fmt);
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
            : fmt == std::chars_format::hex
            ? -1
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
            : fmt == std::chars_format::hex
            ? -1
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
        inline constexpr bool unqualified_parse_value =
            std::is_same_v<T, std::remove_cvref_t<T>>;

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

        [[nodiscard]] BL_FORCE_INLINE constexpr bool has_hex_float_prefix(std::string_view text) noexcept
        {
            std::size_t index = 0;
            if (!text.empty() && (text[0] == '-' || text[0] == '+'))
                index = 1;

            return text.size() >= index + 2 &&
                   text[index] == '0' &&
                   ascii_lower(text[index + 1]) == 'x';
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr std::chars_format effective_parse_format(
            std::string_view text,
            std::chars_format fmt) noexcept
        {
            return fmt == std::chars_format::general && has_hex_float_prefix(text)
                ? std::chars_format::hex
                : fmt;
        }

        template<class Traits>
        [[nodiscard]] constexpr std::from_chars_result from_chars_constexpr_buffer(
            char* first,
            char* last,
            typename Traits::value_type& value,
            std::chars_format fmt,
            bool allow_hex_prefix = false,
            bool allow_leading_plus = false) noexcept
        {
            if (!supported_input_format(fmt))
                return { first, std::errc::invalid_argument };

            if (first == last || ascii_space(*first))
                return { first, std::errc::invalid_argument };

            if (*first == '+' && !allow_leading_plus)
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
                ? parse_hex_float<Traits>(first, last, parsed, &end, allow_hex_prefix, allow_leading_plus)
                : detail::parse_flt<Traits>(first, parsed, &end);

            if (!parsed_ok || end == first)
                return { first, std::errc::invalid_argument };

            if (fmt == std::chars_format::scientific &&
                !begins_special(first, last) &&
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
            using buffer_type = typename Traits::string_type;
            const std::chars_format effective_fmt = effective_parse_format(text, fmt);
            const bool allow_hex_prefix = effective_fmt == std::chars_format::hex;

            parse_result<T> result{};

            if (text.size() > buffer_type::static_capacity)
            {
                result.ec = std::errc::invalid_argument;
                return result;
            }

            buffer_type buffer{ text };
            typename Traits::value_type parsed{};
            const auto parsed_result = from_chars_constexpr_buffer<Traits>(
                buffer.data(),
                buffer.data() + buffer.size(),
                parsed,
                effective_fmt,
                allow_hex_prefix,
                true);

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

            const std::chars_format effective_fmt = effective_parse_format(text, fmt);
            const bool use_flexible_hex = effective_fmt == std::chars_format::hex;
            const bool allow_leading_plus = !text.empty() && text[0] == '+';

            T parsed{};
            std::from_chars_result parsed_result{};
            if (use_flexible_hex || allow_leading_plus)
            {
                using Traits = typename parse_traits_for<T>::traits;
                constexpr std::size_t stack_capacity = 1024;
                char stack[stack_capacity + 1]{};
                std::string dynamic;
                char* buffer = stack;

                if (text.size() > stack_capacity)
                {
                    dynamic.assign(text.data(), text.data() + text.size());
                    dynamic.push_back('\0');
                    buffer = dynamic.data();
                }
                else
                {
                    copy_chars(buffer, text.data(), text.size());
                    buffer[text.size()] = '\0';
                }

                typename Traits::value_type traits_value{};
                const auto traits_result = from_chars_constexpr_buffer<Traits>(
                    buffer,
                    buffer + text.size(),
                    traits_value,
                    effective_fmt,
                    use_flexible_hex,
                    allow_leading_plus);
                parsed = T{ traits_value };
                parsed_result = { text.data() + (traits_result.ptr - buffer), traits_result.ec };
            }
            else
            {
                parsed_result = bl::from_chars(text.data(), text.data() + text.size(), parsed, effective_fmt);
            }
            result.consumed = static_cast<std::size_t>(parsed_result.ptr - text.data());
            result.ec = parsed_result.ec;
            if (result.ec == std::errc{} && result.consumed != text.size())
                result.ec = std::errc::invalid_argument;

            if (result.ec == std::errc{})
                result.value = parsed;

            return result;
        }

    } // namespace detail::charconv

    template<class Value>
    [[nodiscard]] constexpr parse_result<Value> try_parse(
        std::string_view text,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        static_assert(
            detail::charconv::unqualified_parse_value<Value>,
            "bl::parse<T> and bl::try_parse<T> expect T to be an unqualified value type, such as bl::f256."
        );

        if constexpr (!detail::charconv::parse_traits_for<Value>::supported)
        {
            static_assert(
                detail::charconv::dependent_false<Value>::value,
                "bl::parse<T> and bl::try_parse<T> support bl::f32, bl::f64, bl::f128, bl::f128_s, bl::f256, and bl::f256_s."
            );
        }
        else
        {
            if consteval
            {
                return detail::charconv::parse_constexpr<Value>(text, fmt);
            }
            else
            {
                return detail::charconv::parse_runtime<Value>(text, fmt);
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

    template<class Value>
    [[nodiscard]] constexpr Value parse(
        std::string_view text,
        Value fallback,
        std::chars_format fmt = std::chars_format::general) noexcept
    {
        const auto parsed = try_parse<Value>(text, fmt);
        return parsed ? parsed.value : fallback;
    }

    template<class Value>
    [[nodiscard]] constexpr Value parse(
        std::string_view text,
        std::chars_format fmt = std::chars_format::general)
    {
        const auto parsed = try_parse<Value>(text, fmt);
        if (!parsed)
        {
            switch (parsed.ec)
            {
            case std::errc::invalid_argument:
                throw std::invalid_argument("bl::parse invalid input");
            case std::errc::result_out_of_range:
                throw std::out_of_range("bl::parse input out of range");
            default:
                throw std::logic_error("bl::parse unexpected error code");
            }
        }
        return parsed.value;
    }

} // namespace bl

#endif
