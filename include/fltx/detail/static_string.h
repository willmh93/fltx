/**
 * fltx/detail/static_string.h - Fixed-capacity constexpr string helpers for fltx I/O.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_STATIC_STRING_INCLUDED
#define FLTX_DETAIL_STATIC_STRING_INCLUDED
#include <cstddef>
#include <string>
#include <string_view>

namespace bl
{
    struct precision_info
    {
        int digits = -1;
        int leading_digits = 0;
        int trailing_digits = 0;

        constexpr precision_info() noexcept = default;

        constexpr precision_info(int precision_digits) noexcept :
            digits(precision_digits)
        {
        }

        constexpr precision_info(int precision_digits, int leading, int trailing) noexcept :
            digits(precision_digits),
            leading_digits(leading),
            trailing_digits(trailing)
        {
        }
    };

    template<std::size_t capacity>
    struct static_string
    {
        using size_type = std::size_t;

        static constexpr size_type npos            = static_cast<size_type>(-1);
        static constexpr size_type static_capacity = capacity;

        char chars[capacity + 1]{};
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
        constexpr static_string(const char(&text)[size])
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
            return std::string(chars, length);
        }

        constexpr std::string_view view() const noexcept
        {
            return std::string_view(chars, length);
        }

        constexpr char* data() noexcept
        {
            return chars;
        }

        constexpr const char* data() const noexcept
        {
            return chars;
        }

        constexpr const char* c_str() const noexcept
        {
            return chars;
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
            return chars;
        }

        constexpr char* end() noexcept
        {
            return chars + length;
        }

        constexpr const char* begin() const noexcept
        {
            return chars;
        }

        constexpr const char* end() const noexcept
        {
            return chars + length;
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

    using f32_io_string  = static_string<64>;
    using f64_io_string  = static_string<352>;
    using f128_io_string = static_string<352>;
    using f256_io_string = static_string<384>;

} // namespace bl

#endif
