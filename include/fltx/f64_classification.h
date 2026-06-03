/**
 * fltx/f64_classification.h - constexpr <cmath>-style classification functions for f64.
 *
 * f64 classification, predicates, and comparison helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_CLASSIFICATION_INCLUDED
#define F64_CLASSIFICATION_INCLUDED

#include "fltx/aliases.h"
#include "fltx/detail/common_math.h"
#include "fltx/detail/math_utils.h"


namespace bl {

namespace detail::_f64_impl
{
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::signbit;
    using detail::fp::fabs;

    BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
    {
        return x == 0.0;
    }

    BL_FORCE_INLINE constexpr double abs(double x) noexcept
    {
        return fabs(x);
    }

} // namespace detail::_f64_impl

[[nodiscard]] BL_FORCE_INLINE constexpr double abs(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::abs(x),
        std::abs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fabs(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::abs(x),
        std::fabs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::signbit(x),
        std::signbit(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::isnan(x),
        std::isnan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::isinf(x),
        std::isinf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::isfinite(x),
        std::isfinite(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
{
    return x == 0.0;
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(double x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<double>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(double x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(double a, double b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(double a, double b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a != b;
}

} // namespace bl

#endif
