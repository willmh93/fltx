/**
 * fltx/f32_classification.h - constexpr <cmath>-style classification functions for f32.
 *
 * f32 classification, predicates, and comparison helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F32_CLASSIFICATION_INCLUDED
#define F32_CLASSIFICATION_INCLUDED

#include "fltx/aliases.h"
#include "fltx/detail/common_math.h"
#include "fltx/detail/math_utils.h"


namespace bl {

namespace detail::_f32_impl
{
    using detail::fp::fabs;
    using detail::fp::isfinite;
    using detail::fp::isinf;
    using detail::fp::isnan;
    using detail::fp::signbit;

    BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
    {
        return x == 0.0f;
    }

} // namespace detail::_f32_impl

[[nodiscard]] BL_FORCE_INLINE constexpr float abs(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::fabs(x),
        std::abs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr float fabs(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::fabs(x),
        std::fabs(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::signbit(x),
        std::signbit(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::isnan(x),
        std::isnan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::isinf(x),
        std::isinf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(float x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f32_impl::isfinite(x),
        std::isfinite(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(float x) noexcept
{
    return detail::_f32_impl::iszero(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(float x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<float>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(float x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(float a, float b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(float a, float b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(float a, float b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(float a, float b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(float a, float b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(float a, float b) noexcept
{
    return !isunordered(a, b) && a != b;
}

} // namespace bl

#endif
