/**
 * fltx/f256/classification.h - value utilities and classification predicates for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_CLASSIFICATION_INCLUDED
#define FLTX_F256_CLASSIFICATION_INCLUDED
#include "fltx/f256/comparison.h"
#include "fltx/f256/stl.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s abs(const f256_s& a) noexcept
{
    return (a.x0 < 0.0) ? -a : a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s clamp(const f256_s& v, const f256_s& lo, const f256_s& hi) noexcept
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}


[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(const f256_s& a)      noexcept { return detail::_f256::isnan(a.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(const f256_s& a)      noexcept { return detail::_f256::isinf(a.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(const f256_s& x)   noexcept { return detail::_f256::isfinite(x.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(const f256_s& a)     noexcept { return a.x0 == 0 && a.x1 == 0 && a.x2 == 0 && a.x3 == 0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool ispositive(const f256_s& x) noexcept { return x.x0 > 0 || (x.x0 == 0 && (x.x1 > 0 || (x.x1 == 0 && (x.x2 > 0 || (x.x2 == 0 && x.x3 > 0))))); }

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(const f256_s& x) noexcept
{
    return detail::_f256::signbit_constexpr(x.x0)
        || (x.x0 == 0.0 && (detail::_f256::signbit_constexpr(x.x1)
        || (x.x1 == 0.0 && (detail::_f256::signbit_constexpr(x.x2)
        || (x.x2 == 0.0 && detail::_f256::signbit_constexpr(x.x3))))));
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(const f256_s& x) noexcept
{
    if (isnan(x))  [[unlikely]] return FP_NAN;
    if (isinf(x))  [[unlikely]] return FP_INFINITE;
    if (iszero(x)) [[unlikely]] return FP_ZERO;

    return abs(x) < std::numeric_limits<f256_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(const f256_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(const f256_s& a, const f256_s& b) noexcept
{
    return isnan(a) || isnan(b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

} // namespace bl

#endif
