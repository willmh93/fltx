/**
 * fltx/f128/math/pow.h - f128 power functions.
 *
 * Power wrappers using integer/dyadic shortcuts where available, otherwise the exp/log path.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_POW_INCLUDED
#define FLTX_F128_POW_INCLUDED
#include "fltx/detail/f128/math/pow.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::pow(x, y),
        detail::_f128_runtime::pow(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow(const f128_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::pow(x, y),
        detail::_f128_runtime::pow(x, y)
    );
}

} // namespace bl

#endif
