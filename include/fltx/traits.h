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
#include <string_view>
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

    template<class T> inline constexpr int fltx_precision_rank_v = 0;
    template<> inline constexpr int fltx_precision_rank_v<f32>    = 32;
    template<> inline constexpr int fltx_precision_rank_v<f64>    = 64;
    template<> inline constexpr int fltx_precision_rank_v<long double> = 128;
    template<> inline constexpr int fltx_precision_rank_v<f128_s> = 128;
    template<> inline constexpr int fltx_precision_rank_v<f128>   = 128;
    template<> inline constexpr int fltx_precision_rank_v<f256_s> = 256;
    template<> inline constexpr int fltx_precision_rank_v<f256>   = 256;

    template<class T>
    inline constexpr int fltx_type_precision_rank_v =
        fltx_precision_rank_v<std::remove_cvref_t<T>>;

    template<class Base, class Exp>
    concept fltx_pow_wider_floating_exponent =
        fltx_floating_point<Base> &&
        fltx_floating_point<Exp> &&
        (fltx_type_precision_rank_v<Exp> > fltx_type_precision_rank_v<Base>);

    enum struct FloatType : int
    {
        F32,
        F64,
        F128,
        F256,
        COUNT
    };

    [[nodiscard]] constexpr std::string_view to_string(FloatType type) noexcept
    {
        switch (type)
        {
        case FloatType::F32:  return "f32";
        case FloatType::F64:  return "f64";
        case FloatType::F128: return "f128";
        case FloatType::F256: return "f256";
        default:              return "unknown";
        }
    }

} // namespace bl

#endif
