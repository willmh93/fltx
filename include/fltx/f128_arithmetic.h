/**
 * fltx/f128_arithmetic.h - Public arithmetic operators for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_ARITHMETIC_INCLUDED
#define F128_ARITHMETIC_INCLUDED
#include "fltx/detail/f128_arithmetic.h"

namespace bl {

BL_PUSH_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator+(const f128_s& a, const f128_s& b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b.hi)) [[unlikely]]
        return detail::_f128::add_special(a, b);

    const f128_s out = detail::_f128::add_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a));
    return out;
}

[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator-(const f128_s& a, const f128_s& b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b.hi)) [[unlikely]]
        return detail::_f128::sub_special(a, b);

    const f128_s out = detail::_f128::sub_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a));
    return out;
}
BL_POP_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b.hi)) [[unlikely]]
        return detail::_f128::mul_special(a, b);

    const f128_s out = detail::_f128::mul_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a) != signbit(b));
    return out;
}

[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept
{
    if (!detail::fp::isfinite(b.hi) || b.hi == 0.0) [[unlikely]]
        return detail::_f128::div_special(a, b);

    return detail::_f128::div_inline(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b)) [[unlikely]]
        return detail::_f128::add_special(a, f128_s{ b, 0.0 });

    const f128_s out = detail::_f128::add_double_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a));
    return out;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b)) [[unlikely]]
        return detail::_f128::sub_special(a, f128_s{ b, 0.0 });

    const f128_s out = detail::_f128::sub_double_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a));
    return out;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept
{
    if (!detail::fp::isfinite(a.hi) || !detail::fp::isfinite(b)) [[unlikely]]
        return detail::_f128::mul_special(a, f128_s{ b, 0.0 });

    const f128_s out = detail::_f128::mul_double_inline(a, b);
    if (!detail::fp::isfinite(out.hi)) [[unlikely]]
        return detail::_f128::signed_infinity(signbit(a) != detail::fp::signbit(b));
    return out;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept
{
    if (!detail::fp::isfinite(b) || b == 0.0) [[unlikely]]
        return detail::_f128::div_special(a, f128_s{ b, 0.0 });

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
