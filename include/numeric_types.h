#pragma once

#include <cstdint>
#include <type_traits>

#include "fltx_core.h"

namespace bl
{
    typedef int8_t   i8;
    typedef int16_t  i16;
    typedef int32_t  i32;
    typedef int64_t  i64;

    typedef uint8_t  u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;

    typedef float  f32;
    typedef double f64;

    template<class T> concept is_f32_v  = std::is_same_v<T, f32>;
    template<class T> concept is_f64_v  = std::is_same_v<T, f64>;
    template<class T> concept is_f128_v = std::is_same_v<T, f128> || std::is_same_v<T, f128_t>;
    template<class T> concept is_f256_v = std::is_same_v<T, f256> || std::is_same_v<T, f256_t>;

    template<class T> concept is_fltx_v           = std::same_as<T, f128>   || std::same_as<T, f256> ||
                                                    std::same_as<T, f128_t> || std::same_as<T, f256_t>;

    // match std style
    template<class T> concept is_floating_point_v = std::is_floating_point_v<T> || is_fltx_v<T>;
    template<class T> concept is_arithmetic_v     = std::is_arithmetic_v<T> || is_fltx_v<T>;
    template<class T> concept is_integral_v       = std::is_integral_v<T>;
}
