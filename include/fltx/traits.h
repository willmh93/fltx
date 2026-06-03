/**
 * fltx/traits.h - Compile-time fltx type traits and precision metadata.
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
    template<class T> concept fltx_f32  = std::same_as<std::remove_cv_t<T>, f32>;
    template<class T> concept fltx_f64  = std::same_as<std::remove_cv_t<T>, f64>;
    template<class T> concept fltx_f128 = std::same_as<std::remove_cv_t<T>, f128_s> ||
                                          std::same_as<std::remove_cv_t<T>, f128>;
    template<class T> concept fltx_f256 = std::same_as<std::remove_cv_t<T>, f256_s> ||
                                          std::same_as<std::remove_cv_t<T>, f256>;

    template<class T> concept fltx_extended_float = fltx_f128<T> || fltx_f256<T>;
    template<class T> concept fltx_float          = fltx_f32<T>  || fltx_f64<T> || fltx_extended_float<T>;

    template<class T> concept fltx_floating_point = std::is_floating_point_v<T> || fltx_extended_float<T>;
    template<class T> concept fltx_arithmetic     = std::is_arithmetic_v<T>     || fltx_extended_float<T>;

    template<class T> inline constexpr bool is_f32_v  = fltx_f32<T>;
    template<class T> inline constexpr bool is_f64_v  = fltx_f64<T>;
    template<class T> inline constexpr bool is_f128_v = fltx_f128<T>;
    template<class T> inline constexpr bool is_f256_v = fltx_f256<T>;

    template<class T> inline constexpr bool is_fltx_extended_float_v = fltx_extended_float<T>;
    template<class T> inline constexpr bool is_fltx_float_v          = fltx_float<T>;

    template<class T> inline constexpr bool is_floating_point_v = fltx_floating_point<T>;
    template<class T> inline constexpr bool is_arithmetic_v     = fltx_arithmetic<T>;
    template<class T> inline constexpr bool is_integral_v       = std::is_integral_v<T>;

    enum struct FloatType : int { F32, F64, F128, F256, COUNT };
    static inline const char* FloatTypeNames[(int)FloatType::COUNT] = {
        "F32",
        "F64",
        "F128",
        "F256"
    };

} // namespace bl

#endif
