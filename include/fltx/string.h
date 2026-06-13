/**
 * fltx/string.h - Unified string formatting and parsing interface.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_STRING_INCLUDED
#define FLTX_STRING_INCLUDED

#include <ios>
#include <limits>
#include <string>

#include "fltx/aliases.h"
#include "fltx/detail/native_float_io.h"
#include "fltx/f128_string.h"
#include "fltx/f256_string.h"

namespace bl
{
    [[nodiscard]] BL_FORCE_INLINE constexpr bl::f32_io_string to_static_string(
        f32 value,
        int precision = std::numeric_limits<f32>::digits10,
        std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
    {
        bl::f32_io_string out;
        detail::to_string_into<detail::_native_float_io::f32_io_traits>(
            out,
            value,
            precision,
            flags
        );
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bl::f64_io_string to_static_string(
        f64 value,
        int precision = std::numeric_limits<f64>::digits10,
        std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
    {
        bl::f64_io_string out;
        detail::to_string_into<detail::_native_float_io::f64_io_traits>(
            out,
            value,
            precision,
            flags
        );
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE std::string to_string(
        f32 value,
        precision_info precision = std::numeric_limits<f32>::digits10,
        std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
    {
        const int digits = precision.digits >= 0 ? precision.digits : std::numeric_limits<f32>::digits10;
        std::string out;
        detail::to_string_into<detail::_native_float_io::f32_io_traits>(
            out,
            value,
            digits,
            flags
        );
        if (detail::should_collapse_fixed_string(precision, flags))
            detail::collapse_fixed_string(out, precision);
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE std::string to_string(
        f64 value,
        precision_info precision = std::numeric_limits<f64>::digits10,
        std::ios_base::fmtflags flags = std::ios_base::fmtflags{})
    {
        std::string out;
        const int digits = precision.digits >= 0 ? precision.digits : std::numeric_limits<f64>::digits10;
        detail::to_string_into<detail::_native_float_io::f64_io_traits>(
            out,
            value,
            digits,
            flags
        );
        if (detail::should_collapse_fixed_string(precision, flags))
            detail::collapse_fixed_string(out, precision);
        return out;
    }

} // namespace bl

#endif
