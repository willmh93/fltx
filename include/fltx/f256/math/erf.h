/**
 * fltx/f256/math/erf.h - f256 error functions.
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

#ifndef FLTX_F256_ERF_INCLUDED
#define FLTX_F256_ERF_INCLUDED
#include "fltx/detail/f256/math/erf.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 erf(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::erf(x),
        detail::_f256_runtime::erf(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 erfc(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::erfc(x),
        detail::_f256_runtime::erfc(x)
    );
}

} // namespace bl

#endif
