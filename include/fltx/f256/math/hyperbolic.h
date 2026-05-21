/**
 * fltx/f256/math/hyperbolic.h - f256 hyperbolic functions.
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

#ifndef FLTX_F256_HYPERBOLIC_INCLUDED
#define FLTX_F256_HYPERBOLIC_INCLUDED
#include "fltx/detail/f256/math/hyperbolic.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 sinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sinh(x),
        detail::_f256_runtime::sinh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 cosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::cosh(x),
        detail::_f256_runtime::cosh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 tanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::tanh(x),
        detail::_f256_runtime::tanh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 asinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::asinh(x),
        detail::_f256_runtime::asinh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 acosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::acosh(x),
        detail::_f256_runtime::acosh(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 atanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::atanh(x),
        detail::_f256_runtime::atanh(x)
    );
}

} // namespace bl

#endif
