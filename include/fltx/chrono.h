/**
 * fltx/chrono.h - std::chrono integration for fltx duration reps.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_CHRONO_INCLUDED
#define FLTX_CHRONO_INCLUDED

#include <chrono>
#include <type_traits>

#include "fltx/f128.h"
#include "fltx/f256.h"

namespace bl::detail
{
    template<class T>
    inline constexpr bool fltx_chrono_scalar_v =
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>;

    template<class T>
    struct fltx_chrono_scalar_common_rep
    {
        using clean_t = std::remove_cv_t<T>;

        using type = std::conditional_t<
            std::is_same_v<clean_t, f128_s> || std::is_same_v<clean_t, f128>,
            f128,
            f256>;
    };

} // namespace bl::detail

#define FLTX_DEFINE_COMMON_TYPE(lhs, rhs, result) \
template<>                                        \
struct std::common_type<lhs, rhs>                 \
{                                                 \
    using type = result;                          \
};

FLTX_DEFINE_COMMON_TYPE(bl::f128_s, bl::f128_s, bl::f128_s)
FLTX_DEFINE_COMMON_TYPE(bl::f128_s, bl::f128,   bl::f128)
FLTX_DEFINE_COMMON_TYPE(bl::f128,   bl::f128_s, bl::f128)
FLTX_DEFINE_COMMON_TYPE(bl::f128,   bl::f128,   bl::f128)

FLTX_DEFINE_COMMON_TYPE(bl::f256_s, bl::f256_s, bl::f256_s)
FLTX_DEFINE_COMMON_TYPE(bl::f256_s, bl::f256,   bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f256,   bl::f256_s, bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f256,   bl::f256,   bl::f256)

FLTX_DEFINE_COMMON_TYPE(bl::f128_s, bl::f256_s, bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f128_s, bl::f256,   bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f128,   bl::f256_s, bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f128,   bl::f256,   bl::f256)

FLTX_DEFINE_COMMON_TYPE(bl::f256_s, bl::f128_s, bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f256_s, bl::f128,   bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f256,   bl::f128_s, bl::f256)
FLTX_DEFINE_COMMON_TYPE(bl::f256,   bl::f128,   bl::f256)

#undef FLTX_DEFINE_COMMON_TYPE

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<bl::f128_s, T>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f128_s>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<T, bl::f128_s>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f128_s>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<bl::f128, T>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f128>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<T, bl::f128>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f128>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<bl::f256_s, T>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f256_s>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<T, bl::f256_s>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f256_s>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<bl::f256, T>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f256>::type;
};

template<class T>
    requires bl::detail::fltx_chrono_scalar_v<T>
struct std::common_type<T, bl::f256>
{
    using type = typename bl::detail::fltx_chrono_scalar_common_rep<bl::f256>::type;
};

template<>
struct std::chrono::treat_as_floating_point<bl::f128_s> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f128> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f256_s> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f256> : std::true_type
{
};

#endif
