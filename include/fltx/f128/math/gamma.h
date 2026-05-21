/**
 * fltx/f128/math/gamma.h - f128 gamma functions.
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

#ifndef FLTX_F128_GAMMA_INCLUDED
#define FLTX_F128_GAMMA_INCLUDED
#include "fltx/detail/f128/math/gamma.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 lgamma(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::lgamma(x),
        detail::_f128_runtime::lgamma(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 tgamma(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::tgamma(x),
        detail::_f128_runtime::tgamma(x)
    );
}

} // namespace bl

#endif
