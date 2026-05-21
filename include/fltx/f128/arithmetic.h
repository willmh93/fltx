/**
 * fltx/f128/arithmetic.h - Public arithmetic operators for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_ARITHMETIC_INCLUDED
#define FLTX_F128_ARITHMETIC_INCLUDED
#include "fltx/detail/f128/arithmetic.h"

namespace bl {

BL_PUSH_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator+(const f128_s& a, const f128_s& b) noexcept
{
    return detail::_f128::add_inline(a, b);
}

[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator-(const f128_s& a, const f128_s& b) noexcept
{
    return detail::_f128::sub_inline(a, b);
}
BL_POP_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept
{
    return detail::_f128::mul_inline(a, b);
}

[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept
{
    return detail::_f128::div_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept
{
    return detail::_f128::add_double_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept
{
    return detail::_f128::sub_double_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept
{
    return detail::_f128::mul_double_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept
{
    return detail::_f128::div_double_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(double a, const f128_s& b) noexcept { return b + a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(double a, const f128_s& b) noexcept { return -(b - a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(double a, const f128_s& b) noexcept { return b * a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(double a, const f128_s& b) noexcept { return f128_s{ a } / b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, float b) noexcept { return a + (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, float b) noexcept { return a - (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, float b) noexcept { return a * (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, float b) noexcept { return a / (double)b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(float a, const f128_s& b) noexcept { return (double)a + b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(float a, const f128_s& b) noexcept { return (double)a - b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(float a, const f128_s& b) noexcept { return (double)a * b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(float a, const f128_s& b) noexcept { return (double)a / b; }

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a + static_cast<double>(b);

    return a + detail::_f128::integer_to_f128(b);
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a - static_cast<double>(b);

    return a - detail::_f128::integer_to_f128(b);
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a * static_cast<double>(b);

    return a * detail::_f128::integer_to_f128(b);
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a / static_cast<double>(b);

    return a / detail::_f128::integer_to_f128(b);
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(T a, const f128_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) + b;

    return detail::_f128::integer_to_f128(a) + b;
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(T a, const f128_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) - b;

    return detail::_f128::integer_to_f128(a) - b;
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(T a, const f128_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) * b;

    return detail::_f128::integer_to_f128(a) * b;
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(T a, const f128_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) / b;

    return detail::_f128::integer_to_f128(a) / b;
}

} // namespace bl

#endif
