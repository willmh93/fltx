/**
 * fltx/f128/math/hyperbolic.h - f128 hyperbolic functions.
 *
 * Hyperbolic and inverse-hyperbolic wrappers built from the shared exp/log/sqrt kernels.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_HYPERBOLIC_INCLUDED
#define FLTX_F128_HYPERBOLIC_INCLUDED
#include "fltx/detail/f128/math/hyperbolic.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 sinh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sinh(x),
        detail::_f128_runtime::sinh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 cosh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::cosh(x),
        detail::_f128_runtime::cosh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 tanh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::tanh(x),
        detail::_f128_runtime::tanh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 asinh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::asinh(x),
        detail::_f128_runtime::asinh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 acosh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::acosh(x),
        detail::_f128_runtime::acosh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 atanh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::atanh(x),
        detail::_f128_runtime::atanh(x)
    );
}

} // namespace bl

#endif
