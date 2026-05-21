/**
 * fltx/f128/math/basic.h - basic f128 operations.
 *
 * Rounding, decomposition, square root, remainders, hypot, cbrt, fused arithmetic helpers, and adjacent-value operations.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_BASIC_INCLUDED
#define FLTX_F128_BASIC_INCLUDED
#include "fltx/detail/f128/math/basic.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fabs(const f128_s& a) noexcept
{
    return detail::_f128::fabs_impl(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 round(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::round(a),
        detail::_f128_runtime::round(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 nearbyint(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::nearbyint(a),
        detail::_f128_runtime::nearbyint(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 rint(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::rint(x),
        detail::_f128_runtime::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::lround(x),
        detail::_f128_runtime::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::llround(x),
        detail::_f128_runtime::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::lrint(x),
        detail::_f128_runtime::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::llrint(x),
        detail::_f128_runtime::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    return detail::_f128::fma_impl(x, y, z);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fmin(const f128_s& a, const f128_s& b)
{
    return detail::_f128::fmin_impl(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fmax(const f128_s& a, const f128_s& b)
{
    return detail::_f128::fmax_impl(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fdim(const f128_s& x, const f128_s& y)
{
    return detail::_f128::fdim_impl(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 copysign(const f128_s& x, const f128_s& y)
{
    return detail::_f128::copysign_impl(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept
{
    return detail::_f128::ilogb_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 logb(const f128_s& x) noexcept
{
    return detail::_f128::logb_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 scalbn(const f128_s& x, int e) noexcept
{
    return detail::_f128::scalbn_impl(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 scalbln(const f128_s& x, long e) noexcept
{
    return detail::_f128::scalbln_impl(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 nexttoward(const f128_s& from, long double to) noexcept
{
    return detail::_f128::nexttoward_impl(from, to);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 nexttoward(const f128_s& from, const f128_s& to) noexcept
{
    return detail::_f128::nexttoward_impl(from, to);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 fmod(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::fmod(x, y),
        detail::_f128_runtime::fmod(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 round_to_decimals(f128_s v, int prec)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::round_to_decimals(v, prec),
        detail::_f128_runtime::round_to_decimals(v, prec)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 remainder(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::remainder(x, y),
        detail::_f128_runtime::remainder(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 sqrt(f128_s a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sqrt(a),
        detail::_f128_runtime::sqrt(a)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 ldexp(const f128_s& x, int e)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::ldexp(x, e),
        detail::_f128_runtime::ldexp(x, e)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 cbrt(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::cbrt(x),
        detail::_f128_runtime::cbrt(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 hypot(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::hypot(x, y),
        detail::_f128_runtime::hypot(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 remquo(const f128_s& x, const f128_s& y, int* quo)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::remquo(x, y, quo),
        detail::_f128_runtime::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 frexp(const f128_s& x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::frexp(x, exp),
        detail::_f128_runtime::frexp(x, exp)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 modf(const f128_s& x, f128_s* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::modf(x, iptr),
        detail::_f128_runtime::modf(x, iptr)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 nextafter(const f128_s& from, const f128_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::nextafter(from, to),
        detail::_f128_runtime::nextafter(from, to)
    );
}

} // namespace bl

#endif
