/**
 * fltx/f256/math/exp_log.h - f256 exponential and logarithm functions.
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

#ifndef FLTX_F256_EXP_LOG_INCLUDED
#define FLTX_F256_EXP_LOG_INCLUDED
#include "fltx/detail/f256/math/exp_log.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept
{
    return detail::_f256::log_as_double_impl(a);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 exp(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::exp(x),
        detail::_f256_runtime::exp(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 exp2(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::exp2(x),
        detail::_f256_runtime::exp2(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 log(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::log(a),
        detail::_f256_runtime::log(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 log2(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::log2(a),
        detail::_f256_runtime::log2(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 log10(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::log10(a),
        detail::_f256_runtime::log10(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 expm1(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::expm1(x),
        detail::_f256_runtime::expm1(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 log1p(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::log1p(x),
        detail::_f256_runtime::log1p(x)
    );
}

} // namespace bl

#endif
