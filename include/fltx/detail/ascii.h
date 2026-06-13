/**
 * fltx/detail/ascii.h - Small constexpr ASCII helpers shared by I/O code.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_ASCII_INCLUDED
#define FLTX_DETAIL_ASCII_INCLUDED
#include "fltx/config.h"

namespace bl::detail {

BL_FORCE_INLINE constexpr unsigned char ascii_lower(char c) noexcept
{
    return static_cast<unsigned char>((c >= 'A' && c <= 'Z') ? (c | 0x20) : c);
}

BL_FORCE_INLINE constexpr bool ascii_space(char c) noexcept
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ascii_hex_digit_value(char ch) noexcept
{
    if ('0' <= ch && ch <= '9')
        return ch - '0';
    if ('a' <= ch && ch <= 'f')
        return 10 + (ch - 'a');
    if ('A' <= ch && ch <= 'F')
        return 10 + (ch - 'A');
    return -1;
}

} // namespace bl::detail

#endif
