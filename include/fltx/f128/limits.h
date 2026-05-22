/**
 * fltx/f128/limits.h - std::numeric_limits specializations for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_LIMITS_INCLUDED
#define FLTX_F128_LIMITS_INCLUDED
#include <limits>

#include "fltx/f128/type.h"

template<>
struct std::numeric_limits<bl::f128_s>
{
    static constexpr bool is_specialized = true;

    static constexpr bl::f128_s min()    noexcept { return { numeric_limits<double>::min(), 0.0 }; }
    static constexpr bl::f128_s max()    noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon() }; }
    static constexpr bl::f128_s lowest() noexcept { return { -numeric_limits<double>::max(), numeric_limits<double>::epsilon() }; }

    static constexpr bl::f128_s epsilon()       noexcept { return { 1.232595164407831e-32, 0.0 }; } // ~2^-106, a single ulp of double-double
    static constexpr bl::f128_s round_error()   noexcept { return { 0.5, 0.0 }; }
    static constexpr bl::f128_s infinity()      noexcept { return { numeric_limits<double>::infinity(), 0.0 }; }
    static constexpr bl::f128_s quiet_NaN()     noexcept { return { numeric_limits<double>::quiet_NaN(), 0.0 }; }
    static constexpr bl::f128_s signaling_NaN() noexcept { return { numeric_limits<double>::signaling_NaN(), 0.0 }; }
    static constexpr bl::f128_s denorm_min()    noexcept { return { numeric_limits<double>::denorm_min(), 0.0 }; }

    static constexpr bool has_infinity      = true;
    static constexpr bool has_quiet_NaN     = true;
    static constexpr bool has_signaling_NaN = true;

    static constexpr int digits       = 106; // ~53 bits * 2
    static constexpr int digits10     = 31;  // log10(2^106) ~= 31.9
    static constexpr int max_digits10 = 33;
    static constexpr bool is_signed  = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact   = false;
    static constexpr int radix        = 2;

    static constexpr int min_exponent   = numeric_limits<double>::min_exponent;
    static constexpr int max_exponent   = numeric_limits<double>::max_exponent;
    static constexpr int min_exponent10 = numeric_limits<double>::min_exponent10;
    static constexpr int max_exponent10 = numeric_limits<double>::max_exponent10;

    static constexpr bool is_iec559  = false; // not IEEE-754 compliant
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo  = false;

    static constexpr bool traps           = false;
    static constexpr bool tinyness_before = false;

    static constexpr float_round_style round_style = round_to_nearest;
};

BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS(bl::f128, bl::f128_s)

#ifndef FLTX_LIMITS_ALIAS_DEFINED
#define FLTX_LIMITS_ALIAS_DEFINED
namespace bl
{
    using std::numeric_limits;
} // namespace bl
#endif

#endif
