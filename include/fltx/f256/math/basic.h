/**
 * fltx/f256/math/basic.h - Basic f256 operations.
 *
 * Rounding, decomposition, square root, remainders, scaling, cbrt, hypot, fused arithmetic helpers, and adjacent-value operations.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_BASIC_INCLUDED
#define FLTX_F256_BASIC_INCLUDED
#include "fltx/detail/f256/math/basic.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fabs(const f256_s& a) noexcept
{
    return detail::_f256::fabs_impl(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 rint(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::rint(x),
        detail::_f256_runtime::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::lround(x),
        detail::_f256_runtime::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::llround(x),
        detail::_f256_runtime::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::lrint(x),
        detail::_f256_runtime::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::llrint(x),
        detail::_f256_runtime::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 hypot(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::hypot(x, y),
        detail::_f256_runtime::hypot(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    return detail::_f256::fma_impl(x, y, z);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmin(const f256_s& a, const f256_s& b)
{
    return detail::_f256::fmin_impl(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmax(const f256_s& a, const f256_s& b)
{
    return detail::_f256::fmax_impl(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fdim(const f256_s& x, const f256_s& y)
{
    return detail::_f256::fdim_impl(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 copysign(const f256_s& x, const f256_s& y)
{
    return detail::_f256::copysign_impl(x, y);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 fmod(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::fmod(x, y),
        detail::_f256_runtime::fmod(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 round(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::round(a),
        detail::_f256_runtime::round(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 round_to_decimals(f256_s v, int prec)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::round_to_decimals(v, prec),
        detail::_f256_runtime::round_to_decimals(v, prec)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 sqrt(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sqrt(a),
        detail::_f256_runtime::sqrt(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 nearbyint(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::nearbyint(a),
        detail::_f256_runtime::nearbyint(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 ldexp(const f256_s& a, int e)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::ldexp(a, e),
        detail::_f256_runtime::ldexp(a, e)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 cbrt(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::cbrt(x),
        detail::_f256_runtime::cbrt(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 remquo(const f256_s& x, const f256_s& y, int* quo)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::remquo(x, y, quo),
        detail::_f256_runtime::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 remainder(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::remainder(x, y),
        detail::_f256_runtime::remainder(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 frexp(const f256_s& x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::frexp(x, exp),
        detail::_f256_runtime::frexp(x, exp)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 modf(const f256_s& x, f256_s* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::modf(x, iptr),
        detail::_f256_runtime::modf(x, iptr)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr int ilogb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::ilogb(x),
        detail::_f256_runtime::ilogb(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 logb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::logb(x),
        detail::_f256_runtime::logb(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 scalbn(const f256_s& x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::scalbn(x, e),
        detail::_f256_runtime::scalbn(x, e)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 scalbln(const f256_s& x, long e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::scalbln(x, e),
        detail::_f256_runtime::scalbln(x, e)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 nextafter(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::nextafter(from, to),
        detail::_f256_runtime::nextafter(from, to)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 nexttoward(const f256_s& from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::nexttoward(from, to),
        detail::_f256_runtime::nexttoward(from, to)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::nexttoward(from, to),
        detail::_f256_runtime::nexttoward(from, to)
    );
}

} // namespace bl

#endif
