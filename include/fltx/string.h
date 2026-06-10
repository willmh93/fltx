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

#include <limits>
#include <string>

#include "fltx/aliases.h"
#include "fltx/detail/native_float_io.h"
#include "fltx/f128_string.h"
#include "fltx/f256_string.h"

namespace bl
{
    BL_FORCE_INLINE constexpr bl::default_io_string to_static_string(f32 value,
        int precision = std::numeric_limits<f32>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        bl::default_io_string out;
        detail::to_string_into<detail::_native_float_io::f32_io_traits>(
            out,
            value,
            precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    BL_FORCE_INLINE constexpr bl::default_io_string to_static_string(f64 value,
        int precision = std::numeric_limits<f64>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        bl::default_io_string out;
        detail::to_string_into<detail::_native_float_io::f64_io_traits>(
            out,
            value,
            precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    BL_FORCE_INLINE std::string to_string(f32 a,
        precision_info precision,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        const int digits = precision.digits >= 0 ? precision.digits : std::numeric_limits<f32>::digits10;
        std::string out;
        detail::to_string_into<detail::_native_float_io::f32_io_traits>(
            out,
            a,
            digits,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        if (detail::should_collapse_fixed_string(precision, fixed, scientific))
            detail::collapse_fixed_string(out, precision);
        return out;
    }

    BL_FORCE_INLINE std::string to_string(f32 a,
        int precision = std::numeric_limits<f32>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        return to_string(a, precision_info{ precision }, fixed, scientific, strip_trailing_zeros);
    }

    BL_FORCE_INLINE std::string to_string(f64 a,
        precision_info precision,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        std::string out;
        const int digits = precision.digits >= 0 ? precision.digits : std::numeric_limits<f64>::digits10;
        detail::to_string_into<detail::_native_float_io::f64_io_traits>(
            out,
            a,
            digits,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        if (detail::should_collapse_fixed_string(precision, fixed, scientific))
            detail::collapse_fixed_string(out, precision);
        return out;
    }

    BL_FORCE_INLINE std::string to_string(f64 a,
        int precision = std::numeric_limits<f64>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        return to_string(a, precision_info{ precision }, fixed, scientific, strip_trailing_zeros);
    }

} // namespace bl

#endif
