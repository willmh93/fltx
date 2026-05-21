/**
 * fltx/f128/math/erf.h - f128 error functions.
 *
 * Error-function wrappers backed by series, Chebyshev, and continued-fraction regions.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_ERF_INCLUDED
#define FLTX_F128_ERF_INCLUDED
#include "fltx/detail/f128/math/erf.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 erf(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::erf(x),
        detail::_f128_runtime::erf(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 erfc(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::erfc(x),
        detail::_f128_runtime::erfc(x)
    );
}

} // namespace bl

#endif
