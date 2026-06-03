/**
 * fltx/f256_arithmetic.h - Public arithmetic operators for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_ARITHMETIC_INCLUDED
#define F256_ARITHMETIC_INCLUDED
#include "fltx/f256_conversions.h"
#include "fltx/detail/f256_arithmetic.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, const f256_s& b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::add_checked_inline(a, b),
        detail::_f256_runtime::add(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, const f256_s& b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::sub_checked_inline(a, b),
        detail::_f256_runtime::sub(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, const f256_s& b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::mul_checked_inline(a, b),
        detail::_f256_runtime::mul(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, const f256_s& b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::div_checked_inline(a, b),
        detail::_f256_runtime::div(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::add_double_checked_inline(a, b),
        detail::_f256_runtime::add_double(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::sub_double_checked_inline(a, b),
        detail::_f256_runtime::sub_double(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::mul_double_checked_inline(a, b),
        detail::_f256_runtime::mul_double(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::div_double_checked_inline(a, b),
        detail::_f256_runtime::div_double(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(double a, const f256_s& b) noexcept { return b + a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(double a, const f256_s& b) noexcept { return -(b - a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(double a, const f256_s& b) noexcept { return b * a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(double a, const f256_s& b) noexcept { return f256_s{ a } / b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, float b) noexcept { return a + (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, float b) noexcept { return a - (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, float b) noexcept { return a * (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, float b) noexcept { return a / (double)b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(float a, const f256_s& b) noexcept { return (double)a + b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(float a, const f256_s& b) noexcept { return (double)a - b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(float a, const f256_s& b) noexcept { return (double)a * b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(float a, const f256_s& b) noexcept { return (double)a / b; }

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a + static_cast<double>(b);

    return detail::_f256::add_dd(a, detail::_f256::integer_to_double_double(b));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a - static_cast<double>(b);

    return detail::_f256::sub_dd(a, detail::_f256::integer_to_double_double(b));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a * static_cast<double>(b);

    return detail::_f256::mul_dd(a, detail::_f256::integer_to_double_double(b));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, T b) noexcept
{
    if (detail::fp::integer_fits_exact_double(b))
        return a / static_cast<double>(b);

    return detail::_f256::div_dd(a, detail::_f256::integer_to_double_double(b));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(T a, const f256_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) + b;

    return detail::_f256::add_dd(b, detail::_f256::integer_to_double_double(a));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(T a, const f256_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) - b;

    return detail::_f256::sub_dd(detail::_f256::integer_to_double_double(a), b);
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(T a, const f256_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) * b;

    return detail::_f256::mul_dd(b, detail::_f256::integer_to_double_double(a));
}

template<class T, std::enable_if_t<detail::fp::is_integer_scalar_v<T>, int> = 0>
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(T a, const f256_s& b) noexcept
{
    if (detail::fp::integer_fits_exact_double(a))
        return static_cast<double>(a) / b;

    return detail::_f256::div_dd(detail::_f256::integer_to_double_double(a), b);
}

} // namespace bl

#endif
