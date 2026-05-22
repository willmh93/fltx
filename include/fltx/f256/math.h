/**
 * fltx/f256/math.h - constexpr <cmath>-style functions for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_MATH_INCLUDED
#define FLTX_F256_MATH_INCLUDED

#include "fltx/detail/f256/math/basic.h"
#include "fltx/detail/f256/math/exp_log.h"
#include "fltx/detail/f256/math/pow.h"
#include "fltx/detail/f256/math/trig.h"
#include "fltx/detail/f256/math/hyperbolic.h"
#include "fltx/detail/f256/math/erf.h"
#include "fltx/detail/f256/math/gamma.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fabs(const f256_s& a) noexcept
{
    return abs(a);
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
    return detail::_f256_constexpr::fma(x, y, z);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmin(const f256_s& a, const f256_s& b)
{
    return detail::_f256_constexpr::fmin(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmax(const f256_s& a, const f256_s& b)
{
    return detail::_f256_constexpr::fmax(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fdim(const f256_s& x, const f256_s& y)
{
    return detail::_f256_constexpr::fdim(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 copysign(const f256_s& x, const f256_s& y)
{
    return detail::_f256_constexpr::copysign(x, y);
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

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept
{
    return detail::_f256_constexpr::log_as_double(a);
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

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 pow(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::pow(x, y),
        detail::_f256_runtime::pow(x, y)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 pow(const f256_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::pow(x, y),
        detail::_f256_runtime::pow(x, y)
    );
}

} // namespace bl

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sincos(x, s_out, c_out),
        detail::_f256_runtime::sincos(x, s_out, c_out)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 sin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sin(x),
        detail::_f256_runtime::sin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 cos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::cos(x),
        detail::_f256_runtime::cos(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 tan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::tan(x),
        detail::_f256_runtime::tan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 atan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::atan(x),
        detail::_f256_runtime::atan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 atan2(const f256_s& y, const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::atan2(y, x),
        detail::_f256_runtime::atan2(y, x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 asin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::asin(x),
        detail::_f256_runtime::asin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 acos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::acos(x),
        detail::_f256_runtime::acos(x)
    );
}

template<class Vec>
    requires detail::fp::sincos_vector_assignable<Vec, f256_s>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(const f256_s& x, Vec& out)
{
    f256_s s_out{};
    f256_s c_out{};
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
    requires (std::same_as<std::remove_cvref_t<Value>, f256> || std::same_as<std::remove_cvref_t<Value>, f256_s>)
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<std::remove_cvref_t<Value>> sincos(const f256_s& x)
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
