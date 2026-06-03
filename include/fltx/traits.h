/**
 * fltx/traits.h - Compile-time FLTX type traits and precision metadata.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_TRAITS_INCLUDED
#define FLTX_TRAITS_INCLUDED
#include <concepts>
#include <type_traits>

#include "fltx/aliases.h"

namespace bl
{
    template<class T> concept is_f32_v  = std::same_as<T, f32>;
    template<class T> concept is_f64_v  = std::same_as<T, f64>;
    template<class T> concept is_f128_v = std::same_as<T, f128_s> || std::same_as<T, f128>;
    template<class T> concept is_f256_v = std::same_as<T, f256_s> || std::same_as<T, f256>;

    template<class T> concept is_fltx_v = std::same_as<T, f128_s> || std::same_as<T, f256_s> ||
                                          std::same_as<T, f128>   || std::same_as<T, f256>;

    template<class T> concept is_floating_point_v = std::is_floating_point_v<T> || is_fltx_v<T>;
    template<class T> concept is_arithmetic_v     = std::is_arithmetic_v<T>     || is_fltx_v<T>;
    template<class T> concept is_integral_v       = std::is_integral_v<T>;

    enum struct FloatType : int { F32, F64, F128, F256, COUNT };
    static inline const char* FloatTypeNames[(int)FloatType::COUNT] = {
        "F32",
        "F64",
        "F128",
        "F256"
    };

} // namespace bl

#endif
