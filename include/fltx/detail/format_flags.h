/**
 * fltx/detail/format_flags.h - Internal helpers for std::ios_base::fmtflags.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_FORMAT_FLAGS_INCLUDED
#define FLTX_DETAIL_FORMAT_FLAGS_INCLUDED

#include <ios>

#include "fltx/config.h"

namespace bl::detail
{
    enum class float_format : unsigned char
    {
        defaultfloat,
        fixed,
        scientific,
        hexfloat
    };

    [[nodiscard]] BL_FORCE_INLINE constexpr float_format float_format_from_flags(std::ios_base::fmtflags flags) noexcept
    {
        const auto field = flags & std::ios_base::floatfield;
        if (field == std::ios_base::fixed)
            return float_format::fixed;
        if (field == std::ios_base::scientific)
            return float_format::scientific;
        if (field == (std::ios_base::fixed | std::ios_base::scientific))
            return float_format::hexfloat;
        return float_format::defaultfloat;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_showpoint(std::ios_base::fmtflags flags) noexcept
    {
        return (flags & std::ios_base::showpoint) != std::ios_base::fmtflags{};
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_showpos(std::ios_base::fmtflags flags) noexcept
    {
        return (flags & std::ios_base::showpos) != std::ios_base::fmtflags{};
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_uppercase(std::ios_base::fmtflags flags) noexcept
    {
        return (flags & std::ios_base::uppercase) != std::ios_base::fmtflags{};
    }

} // namespace bl::detail

#endif
