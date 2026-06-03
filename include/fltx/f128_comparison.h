/**
 * fltx/f128_comparison.h - Comparison operators for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_COMPARISON_INCLUDED
#define F128_COMPARISON_INCLUDED
#include <compare>
#include <type_traits>

#include "fltx/f128_type.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    template<class T>
    concept compare_scalar =
        !std::is_same_v<std::remove_cvref_t<T>, bool> &&
        (std::is_same_v<std::remove_cvref_t<T>, float> ||
         std::is_same_v<std::remove_cvref_t<T>, double> ||
         detail::fp::is_integer_scalar_v<std::remove_cvref_t<T>>);

    BL_FORCE_INLINE constexpr bool compare_less(double ahi, double alo, double bhi, double blo) noexcept
    {
        return (ahi < bhi) || (ahi == bhi && alo < blo);
    }

    BL_FORCE_INLINE constexpr bool compare_less_equal(double ahi, double alo, double bhi, double blo) noexcept
    {
        return (ahi < bhi) || (ahi == bhi && alo <= blo);
    }

    BL_FORCE_INLINE constexpr bool compare_equal(double ahi, double alo, double bhi, double blo) noexcept
    {
        return ahi == bhi && alo == blo;
    }

    BL_FORCE_INLINE constexpr bool compare_unordered(double ahi, double alo, double bhi, double blo) noexcept
    {
        return detail::fp::isnan(ahi) || detail::fp::isnan(alo) ||
               detail::fp::isnan(bhi) || detail::fp::isnan(blo);
    }

    BL_FORCE_INLINE constexpr std::partial_ordering compare_three_way(
        double ahi,
        double alo,
        double bhi,
        double blo) noexcept
    {
        if (compare_unordered(ahi, alo, bhi, blo))
            return std::partial_ordering::unordered;
        if (compare_less(ahi, alo, bhi, blo))
            return std::partial_ordering::less;
        if (compare_less(bhi, blo, ahi, alo))
            return std::partial_ordering::greater;
        return std::partial_ordering::equivalent;
    }

    template<class T>
    BL_FORCE_INLINE constexpr void compare_terms(T value, double& hi, double& lo) noexcept
    {
        using clean_t = std::remove_cvref_t<T>;

        if constexpr (std::is_same_v<clean_t, float> ||
                      std::is_same_v<clean_t, double> ||
                      detail::fp::integer_type_fits_exact_double_v<clean_t>)
        {
            hi = static_cast<double>(value);
            lo = 0.0;
        }
        else if constexpr (std::is_signed_v<clean_t>)
        {
            detail::fp::int64_to_exact_double_pair(static_cast<int64_t>(value), hi, lo);
        }
        else
        {
            detail::fp::uint64_to_exact_double_pair(static_cast<uint64_t>(value), hi, lo);
        }
    }

} // namespace detail::_f128


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, const f128_s& b)
{
    return detail::_f128::compare_less(a.hi, a.lo, b.hi, b.lo);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, const f128_s& b)
{
    return b < a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, const f128_s& b)
{
    return detail::_f128::compare_less_equal(a.hi, a.lo, b.hi, b.lo);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, const f128_s& b)
{
    return b <= a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, const f128_s& b)
{
    return detail::_f128::compare_equal(a.hi, a.lo, b.hi, b.lo);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, const f128_s& b)
{
    return !(a == b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr std::partial_ordering operator<=>(const f128_s& a, const f128_s& b) noexcept
{
    return detail::_f128::compare_three_way(a.hi, a.lo, b.hi, b.lo);
}


template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, T b)
{
    double bhi{}, blo{};
    detail::_f128::compare_terms(b, bhi, blo);

    return detail::_f128::compare_less(a.hi, a.lo, bhi, blo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(T a, const f128_s& b)
{
    double ahi{}, alo{};
    detail::_f128::compare_terms(a, ahi, alo);

    return detail::_f128::compare_less(ahi, alo, b.hi, b.lo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, T b)
{
    return b < a;
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(T a, const f128_s& b)
{
    return b < a;
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, T b)
{
    double bhi{}, blo{};
    detail::_f128::compare_terms(b, bhi, blo);

    return detail::_f128::compare_less_equal(a.hi, a.lo, bhi, blo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(T a, const f128_s& b)
{
    double ahi{}, alo{};
    detail::_f128::compare_terms(a, ahi, alo);

    return detail::_f128::compare_less_equal(ahi, alo, b.hi, b.lo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, T b)
{
    return b <= a;
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(T a, const f128_s& b)
{
    return b <= a;
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, T b)
{
    double bhi{}, blo{};
    detail::_f128::compare_terms(b, bhi, blo);

    return detail::_f128::compare_equal(a.hi, a.lo, bhi, blo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(T a, const f128_s& b)
{
    double ahi{}, alo{};
    detail::_f128::compare_terms(a, ahi, alo);

    return detail::_f128::compare_equal(ahi, alo, b.hi, b.lo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, T b)
{
    return !(a == b);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(T a, const f128_s& b)
{
    return !(a == b);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr std::partial_ordering operator<=>(const f128_s& a, T b) noexcept
{
    double bhi{}, blo{};
    detail::_f128::compare_terms(b, bhi, blo);

    return detail::_f128::compare_three_way(a.hi, a.lo, bhi, blo);
}

template<detail::_f128::compare_scalar T>
[[nodiscard]] BL_FORCE_INLINE constexpr std::partial_ordering operator<=>(T a, const f128_s& b) noexcept
{
    double ahi{}, alo{};
    detail::_f128::compare_terms(a, ahi, alo);

    return detail::_f128::compare_three_way(ahi, alo, b.hi, b.lo);
}

} // namespace bl

#endif
