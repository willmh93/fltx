
/**
 * fltx_common_base.h — Provides a single constexpr interface for f32/f64/f128/f256 printing/parsing
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */


#ifndef FLTX_IO_INCLUDED
#define FLTX_IO_INCLUDED

#include <algorithm>
#include <limits>
#include <string>

#include "fltx_core.h"

#include "f128_io.h"
#include "f256_io.h"

namespace bl
{
    BL_FORCE_INLINE constexpr detail::default_io_string to_string(f32 value,
        int precision = std::numeric_limits<f32>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        return to_static_string((f128)value, precision, fixed, scientific, strip_trailing_zeros);
    }

    BL_FORCE_INLINE constexpr detail::default_io_string to_string(f64 value,
        int precision = std::numeric_limits<f64>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        return to_static_string((f128)value, precision, fixed, scientific, strip_trailing_zeros);
    }

    BL_FORCE_INLINE std::string to_std_string(f32 a,
        int precision = std::numeric_limits<f32>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        std::string out;
        detail::_f128::to_string_into(
            out,
            f128_s{ static_cast<f64>(a), 0.0 },
            precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    BL_FORCE_INLINE std::string to_std_string(f64 a,
        int precision = std::numeric_limits<f64>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        std::string out;
        detail::_f128::to_string_into(
            out,
            f128_s{ a, 0.0 },
            precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    template<typename T>
    std::string to_string_collapsed(T value, int max_precision, int peek_front = 4, int peek_back = 10)
    {
        if (max_precision < 0) max_precision = 0;
        if (peek_front < 0) peek_front = 0;
        if (peek_back < 0) peek_back = 0;

        peek_front = std::min(peek_front, max_precision);
        peek_back = std::min(peek_back, max_precision);

        std::string s = bl::to_string(value, max_precision, true, false);
        size_t period_i = s.find('.');

        if (period_i == std::string::npos)
            return s;

        int front_end = (int)period_i + 1 + peek_front;
        int back_start = (int)period_i + 1 + (max_precision - peek_back);

        // ensure we're collapsing more than 3 characters, otherwise there's little point
        if (back_start <= front_end + 3)
            return s;

        std::string ret;
        ret += s.substr(0, front_end);
        ret += "...";
        ret += s.substr(back_start, peek_back);

        return ret;
    }
}

#endif