/**
 * fltx/detail/f128_math_sqrt.h - f128 square-root implementation details.
 *
 * Inline f128 sqrt implementation shared by public math and dependent detail math headers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_SQRT_INCLUDED
#define FLTX_F128_DETAIL_SQRT_INCLUDED
#include "fltx/detail/f128_math_shared.h"

namespace bl {

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    constexpr double fast_min = 0x1p-900;
    constexpr double fast_max = 0x1p900;
    if (!bl::use_constexpr_math() && a.hi >= fast_min && a.hi <= fast_max)
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128::sqrt_compensated(a, std::sqrt(a.hi)));

    const int exp2 = frexp_exponent(a.hi);
    const int result_scale = exp2 / 2;
    const int input_scale = -2 * result_scale;
    const f128_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

    double seed{};
    if (bl::use_constexpr_math())
    {
        seed = detail::_f128::sqrt_constexpr_head(scaled_a.hi);
    }
    else
    {
        seed = std::sqrt(scaled_a.hi);
    }

    f128_s y = detail::_f128::sqrt_compensated(scaled_a, seed);

    if (result_scale != 0)
        y = ldexp_terms(y, result_scale);

    return F128_CANONICALIZE_MATH_RESULT(y);
}

} // namespace bl

#endif // FLTX_F128_DETAIL_SQRT_INCLUDED
