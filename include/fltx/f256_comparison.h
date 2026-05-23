/**
 * fltx/f256_comparison.h - Comparison operators for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_COMPARISON_INCLUDED
#define F256_COMPARISON_INCLUDED
#include <type_traits>

#include "fltx/f256_type.h"

namespace bl {

namespace detail::_f256 // primitives and kernels
{
    template<class T>
    concept compare_scalar =
        !std::is_same_v<std::remove_cvref_t<T>, bool> &&
        (std::is_same_v<std::remove_cvref_t<T>, float> ||
         std::is_same_v<std::remove_cvref_t<T>, double> ||
         detail::fp::is_integer_scalar_v<std::remove_cvref_t<T>>);

    BL_FORCE_INLINE constexpr bool compare_less(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        if (ax0 < bx0) return true;
        if (ax0 > bx0) return false;
        if (ax1 < bx1) return true;
        if (ax1 > bx1) return false;
        if (ax2 < bx2) return true;
        if (ax2 > bx2) return false;
        return ax3 < bx3;
    }

    BL_FORCE_INLINE constexpr bool compare_less_equal(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        if (ax0 < bx0) return true;
        if (ax0 > bx0) return false;
        if (ax1 < bx1) return true;
        if (ax1 > bx1) return false;
        if (ax2 < bx2) return true;
        if (ax2 > bx2) return false;
        return ax3 <= bx3;
    }

    BL_FORCE_INLINE constexpr bool compare_equal(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        return ax0 == bx0 && ax1 == bx1 && ax2 == bx2 && ax3 == bx3;
    }

    template<class T>
    BL_FORCE_INLINE constexpr void compare_terms(T value, double& x0, double& x1, double& x2, double& x3) noexcept
    {
        using clean_t = std::remove_cvref_t<T>;

        if constexpr (std::is_same_v<clean_t, float> ||
                      std::is_same_v<clean_t, double> ||
                      detail::fp::integer_type_fits_exact_double_v<clean_t>)
        {
            x0 = static_cast<double>(value);
            x1 = 0.0;
        }
        else if constexpr (std::is_signed_v<clean_t>)
        {
            detail::fp::int64_to_exact_double_pair(static_cast<int64_t>(value), x0, x1);
        }
        else
        {
            detail::fp::uint64_to_exact_double_pair(static_cast<uint64_t>(value), x0, x1);
        }

        x2 = 0.0;
        x3 = 0.0;
    }

} // namespace detail::_f256


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, const f256_s& b)
{
    return detail::_f256::compare_less(a.x0, a.x1, a.x2, a.x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, const f256_s& b)
{
    return b < a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, const f256_s& b)
{
    return detail::_f256::compare_less_equal(a.x0, a.x1, a.x2, a.x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, const f256_s& b)
{
    return b <= a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, const f256_s& b)
{
    return detail::_f256::compare_equal(a.x0, a.x1, a.x2, a.x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, const f256_s& b)
{
    return !(a == b);
}


template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, T b)
{
    double bx0{}, bx1{}, bx2{}, bx3{};
    detail::_f256::compare_terms(b, bx0, bx1, bx2, bx3);

    return detail::_f256::compare_less(a.x0, a.x1, a.x2, a.x3, bx0, bx1, bx2, bx3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(T a, const f256_s& b)
{
    double ax0{}, ax1{}, ax2{}, ax3{};
    detail::_f256::compare_terms(a, ax0, ax1, ax2, ax3);

    return detail::_f256::compare_less(ax0, ax1, ax2, ax3, b.x0, b.x1, b.x2, b.x3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, T b)
{
    return b < a;
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(T a, const f256_s& b)
{
    return b < a;
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, T b)
{
    double bx0{}, bx1{}, bx2{}, bx3{};
    detail::_f256::compare_terms(b, bx0, bx1, bx2, bx3);

    return detail::_f256::compare_less_equal(a.x0, a.x1, a.x2, a.x3, bx0, bx1, bx2, bx3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(T a, const f256_s& b)
{
    double ax0{}, ax1{}, ax2{}, ax3{};
    detail::_f256::compare_terms(a, ax0, ax1, ax2, ax3);

    return detail::_f256::compare_less_equal(ax0, ax1, ax2, ax3, b.x0, b.x1, b.x2, b.x3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, T b)
{
    return b <= a;
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(T a, const f256_s& b)
{
    return b <= a;
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, T b)
{
    double bx0{}, bx1{}, bx2{}, bx3{};
    detail::_f256::compare_terms(b, bx0, bx1, bx2, bx3);

    return detail::_f256::compare_equal(a.x0, a.x1, a.x2, a.x3, bx0, bx1, bx2, bx3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(T a, const f256_s& b)
{
    double ax0{}, ax1{}, ax2{}, ax3{};
    detail::_f256::compare_terms(a, ax0, ax1, ax2, ax3);

    return detail::_f256::compare_equal(ax0, ax1, ax2, ax3, b.x0, b.x1, b.x2, b.x3);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, T b)
{
    return !(a == b);
}

template<detail::_f256::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(T a, const f256_s& b)
{
    return !(a == b);
}

} // namespace bl

#endif
