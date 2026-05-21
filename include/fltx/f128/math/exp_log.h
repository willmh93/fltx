/**
 * fltx/f128/math/exp_log.h - f128 exponential and logarithm functions.
 *
 * Range-reduced exp/log wrappers, including base-2/base-10, one-plus variants, and log-as-double helpers.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_EXP_LOG_INCLUDED
#define FLTX_F128_EXP_LOG_INCLUDED
#include "fltx/detail/f128/math/exp_log.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f128_s a)
{
    return detail::_f128::log_as_double_impl(a);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 exp(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::exp(x),
        detail::_f128_runtime::exp(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 exp2(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::exp2(x),
        detail::_f128_runtime::exp2(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 log(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::log(a),
        detail::_f128_runtime::log(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 log2(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::log2(a),
        detail::_f128_runtime::log2(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 log10(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::log10(x),
        detail::_f128_runtime::log10(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 expm1(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::expm1(x),
        detail::_f128_runtime::expm1(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 log1p(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::log1p(x),
        detail::_f128_runtime::log1p(x)
    );
}

} // namespace bl

#endif
