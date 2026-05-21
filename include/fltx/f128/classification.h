/**
 * fltx/f128/classification.h - Value utilities and classification predicates for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_CLASSIFICATION_INCLUDED
#define FLTX_F128_CLASSIFICATION_INCLUDED
#include "fltx/f128/comparison.h"
#include "fltx/f128/stl.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s abs(const f128_s& a) noexcept
{
    return (a.hi < 0.0) ? -a : a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s clamp(const f128_s& v, const f128_s& lo, const f128_s& hi) noexcept
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}


[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(const f128_s& x)      noexcept { return detail::fp::isnan(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(const f128_s& x)      noexcept { return detail::fp::isinf(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(const f128_s& x)   noexcept { return detail::fp::isfinite(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(const f128_s& x)     noexcept { return x.hi == 0.0 && x.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool ispositive(const f128_s& x) noexcept { return x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0); }

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(const f128_s& x) noexcept
{
    return detail::_f128::signbit(x.hi) || (x.hi == 0.0 && detail::_f128::signbit(x.lo));
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(const f128_s& x) noexcept
{
    if (isnan(x))  [[unlikely]] return FP_NAN;
    if (isinf(x))  [[unlikely]] return FP_INFINITE;
    if (iszero(x)) [[unlikely]] return FP_ZERO;

    return abs(x) < std::numeric_limits<f128_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(const f128_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(const f128_s& a, const f128_s& b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

} // namespace bl

#endif
