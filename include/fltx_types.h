#ifndef FLTX_TYPES_INCLUDED
#define FLTX_TYPES_INCLUDED

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

    // Convenience macro to enable bitops for one enum type
    #define bl_enable_enum_bitops(E) \
    template<> struct enable_enum_bitops<E> : std::true_type {}

    template<class E>
    struct enable_enum_bitops : std::false_type {};

    template<class E>
    concept bitmask_enum = std::is_enum_v<E> && enable_enum_bitops<E>::value;

    template<bitmask_enum E>
    using enum_underlying_t = std::underlying_type_t<E>;

    enum struct FloatType : int { F32, F64, F128, F256, COUNT };
    bl_enable_enum_bitops(FloatType);

    static inline const char* FloatTypeNames[(int)FloatType::COUNT] = { 
        "F32",
        "F64",
        "F128",
        "F256" 
    };
}

#endif