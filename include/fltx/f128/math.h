/**
 * fltx/f128/math.h - constexpr <cmath>-style functions for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_MATH_INCLUDED
#define FLTX_F128_MATH_INCLUDED

#include "fltx/detail/f128/math/basic.h"
#include "fltx/detail/f128/math/exp_log.h"
#include "fltx/detail/f128/math/pow.h"
#include "fltx/detail/f128/math/trig.h"
#include "fltx/detail/f128/math/hyperbolic.h"
#include "fltx/detail/f128/math/erf.h"
#include "fltx/detail/f128/math/gamma.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fabs(const f128_s& a) noexcept
{
    return abs(a);
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
    return detail::_f128_constexpr::fma(x, y, z);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fmin(const f128_s& a, const f128_s& b)
{
    return detail::_f128_constexpr::fmin(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fmax(const f128_s& a, const f128_s& b)
{
    return detail::_f128_constexpr::fmax(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 fdim(const f128_s& x, const f128_s& y)
{
    return detail::_f128_constexpr::fdim(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 copysign(const f128_s& x, const f128_s& y)
{
    return detail::_f128_constexpr::copysign(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept
{
    return detail::_f128_constexpr::ilogb(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 logb(const f128_s& x) noexcept
{
    return detail::_f128_constexpr::logb(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 scalbn(const f128_s& x, int e) noexcept
{
    return detail::_f128_constexpr::scalbn(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 scalbln(const f128_s& x, long e) noexcept
{
    return detail::_f128_constexpr::scalbln(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 nexttoward(const f128_s& from, long double to) noexcept
{
    return detail::_f128_constexpr::nexttoward(from, to);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 nexttoward(const f128_s& from, const f128_s& to) noexcept
{
    return detail::_f128_constexpr::nexttoward(from, to);
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

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f128_s a)
{
    return detail::_f128_constexpr::log_as_double(a);
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

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::pow(x, y),
        detail::_f128_runtime::pow(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow(const f128_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::pow(x, y),
        detail::_f128_runtime::pow(x, y)
    );
}

} // namespace bl

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128 atan(const f128_s& x)
{
    return detail::_f128_constexpr::atan(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 asin(const f128_s& x)
{
    return detail::_f128_constexpr::asin(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 acos(const f128_s& x)
{
    return detail::_f128_constexpr::acos(x);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sincos(x, s_out, c_out),
        detail::_f128_runtime::sincos(x, s_out, c_out)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 sin(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sin(x),
        detail::_f128_runtime::sin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 cos(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::cos(x),
        detail::_f128_runtime::cos(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 tan(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::tan(x),
        detail::_f128_runtime::tan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 atan2(const f128_s& y, const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::atan2(y, x),
        detail::_f128_runtime::atan2(y, x)
    );
}

template<class Vec>
    requires detail::fp::sincos_vector_assignable<Vec, f128_s>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(const f128_s& x, Vec& out)
{
    f128_s s_out{};
    f128_s c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    if (!ok)
    {
        s_out = bl::sin(x);
        c_out = bl::cos(x);
    }
    detail::fp::assign_sincos_vector(out, s_out, c_out);
    return ok;
}

template<class Value>
    requires (std::same_as<std::remove_cvref_t<Value>, f128> || std::same_as<std::remove_cvref_t<Value>, f128_s>)
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<std::remove_cvref_t<Value>> sincos(const f128_s& x)
{
    using Result = std::remove_cvref_t<Value>;

    Result s_out{};
    Result c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    if (!ok)
    {
        s_out = bl::sin(x);
        c_out = bl::cos(x);
    }
    return detail::fp::make_sincos_result(s_out, c_out, ok);
}

} // namespace bl

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
