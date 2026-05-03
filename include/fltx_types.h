/**
 * fltx_types.h - Convenience aliases and compile-time type traits.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_TYPES_INCLUDED
#define FLTX_TYPES_INCLUDED

#include "fltx_common_base.h"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace bl
{
    using i8  = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;
          
    using u8  = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using f32 = float;
    using f64 = double;

    struct f128;
    struct f128_s;
    struct f256;
    struct f256_s;

    template<class T> concept is_f32_v  = std::is_same_v<T, f32>;
    template<class T> concept is_f64_v  = std::is_same_v<T, f64>;
    template<class T> concept is_f128_v = std::is_same_v<T, f128_s> || std::is_same_v<T, f128>;
    template<class T> concept is_f256_v = std::is_same_v<T, f256_s> || std::is_same_v<T, f256>;

    template<class T> concept is_fltx_v = std::same_as<T, f128_s> || std::same_as<T, f256_s> ||
                                          std::same_as<T, f128>   || std::same_as<T, f256>;

    // match std style
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
}

#endif
