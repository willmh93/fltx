/**
 * fltx/f256/math/gamma.h - f256 gamma functions.
 *
 * Gamma/log-gamma wrappers using recurrence, near-one/two series, and asymptotic regions.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_GAMMA_INCLUDED
#define FLTX_F256_GAMMA_INCLUDED
#include "fltx/detail/f256/math/gamma.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 lgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::lgamma(x),
        detail::_f256_runtime::lgamma(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 tgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::tgamma(x),
        detail::_f256_runtime::tgamma(x)
    );
}

} // namespace bl

#endif
