/**
 * fltx/f256_math.h - constexpr <cmath>-style functions for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_MATH_INCLUDED
#define F256_MATH_INCLUDED

#include "fltx/detail/f256_math_basic.h"
#include "fltx/detail/f256_math_transcendental.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fabs(const f256_s& a) noexcept
{
    return abs(a);
}

// rounding
[[nodiscard]] BL_FORCE_INLINE constexpr f256 floor(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::floor(a),
        detail::_f256_runtime::floor(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 ceil(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::ceil(a),
        detail::_f256_runtime::ceil(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 trunc(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::trunc(a),
        detail::_f256_runtime::trunc(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 rint(const f256_s& x)
{
    return detail::_f256_impl::rint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::lround(x),
        detail::_f256_runtime::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::llround(x),
        detail::_f256_runtime::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::lrint(x),
        detail::_f256_runtime::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::llrint(x),
        detail::_f256_runtime::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 hypot(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::hypot(x, y),
        detail::_f256_runtime::hypot(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    return detail::_f256_impl::fma(x, y, z);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmin(const f256_s& a, const f256_s& b)
{
    return detail::_f256_impl::fmin(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmax(const f256_s& a, const f256_s& b)
{
    return detail::_f256_impl::fmax(a, b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fdim(const f256_s& x, const f256_s& y)
{
    return detail::_f256_impl::fdim(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 copysign(const f256_s& x, const f256_s& y)
{
    return detail::_f256_impl::copysign(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 fmod(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::fmod(x, y),
        detail::_f256_runtime::fmod(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 round(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::round(a),
        detail::_f256_runtime::round(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 round_to_decimals(f256_s v, int prec)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::round_to_decimals(v, prec),
        detail::_f256_runtime::round_to_decimals(v, prec)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 sqrt(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::sqrt(a),
        detail::_f256_runtime::sqrt(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 nearbyint(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::nearbyint(a),
        detail::_f256_runtime::nearbyint(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 ldexp(const f256_s& a, int e)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::ldexp(a, e),
        detail::_f256_runtime::ldexp(a, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 cbrt(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::cbrt(x),
        detail::_f256_runtime::cbrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 remquo(const f256_s& x, const f256_s& y, int* quo)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::remquo(x, y, quo),
        detail::_f256_runtime::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 remainder(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::remainder(x, y),
        detail::_f256_runtime::remainder(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 frexp(const f256_s& x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::frexp(x, exp),
        detail::_f256_runtime::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 modf(const f256_s& x, f256_s* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::modf(x, iptr),
        detail::_f256_runtime::modf(x, iptr)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::ilogb(x),
        detail::_f256_runtime::ilogb(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 logb(const f256_s& x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::logb(x),
        detail::_f256_runtime::logb(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 scalbn(const f256_s& x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::scalbn(x, e),
        detail::_f256_runtime::scalbn(x, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 scalbln(const f256_s& x, long e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::scalbln(x, e),
        detail::_f256_runtime::scalbln(x, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 nextafter(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::nextafter(from, to),
        detail::_f256_runtime::nextafter(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 nexttoward(const f256_s& from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::nexttoward(from, to),
        detail::_f256_runtime::nexttoward(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::nexttoward(from, to),
        detail::_f256_runtime::nexttoward(from, to)
    );
}

// exp / log
[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept
{
    return detail::_f256_impl::log_as_double(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 exp(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::exp(x),
        detail::_f256_runtime::exp(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 exp2(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::exp2(x),
        detail::_f256_runtime::exp2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 log(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::log(a),
        detail::_f256_runtime::log(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 log2(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::log2(a),
        detail::_f256_runtime::log2(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 log10(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::log10(a),
        detail::_f256_runtime::log10(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 expm1(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::expm1(x),
        detail::_f256_runtime::expm1(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 log1p(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::log1p(x),
        detail::_f256_runtime::log1p(x)
    );
}

// pow
[[nodiscard]] BL_FORCE_INLINE constexpr f256 pow10_256(int k)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::pow10_256(k),
        detail::_f256_runtime::pow10_256(k)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 pow(const f256_s& x, const f256_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::pow(x, y),
        detail::_f256_runtime::pow(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 pow(const f256_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::pow(x, y),
        detail::_f256_runtime::pow(x, y)
    );
}

// trig
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::sincos(x, s_out, c_out),
        detail::_f256_runtime::sincos(x, s_out, c_out)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 sin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::sin(x),
        detail::_f256_runtime::sin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 cos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::cos(x),
        detail::_f256_runtime::cos(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 tan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::tan(x),
        detail::_f256_runtime::tan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 atan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::atan(x),
        detail::_f256_runtime::atan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 atan2(const f256_s& y, const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::atan2(y, x),
        detail::_f256_runtime::atan2(y, x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 asin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::asin(x),
        detail::_f256_runtime::asin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 acos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::acos(x),
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

template<class Value> requires (std::same_as<std::remove_cvref_t<Value>, f256> || std::same_as<std::remove_cvref_t<Value>, f256_s>)
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

// hyperbolic
[[nodiscard]] BL_FORCE_INLINE constexpr f256 sinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::sinh(x),
        detail::_f256_runtime::sinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 cosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::cosh(x),
        detail::_f256_runtime::cosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 tanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::tanh(x),
        detail::_f256_runtime::tanh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 asinh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::asinh(x),
        detail::_f256_runtime::asinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 acosh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::acosh(x),
        detail::_f256_runtime::acosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 atanh(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::atanh(x),
        detail::_f256_runtime::atanh(x)
    );
}

// erf / erfc
[[nodiscard]] BL_FORCE_INLINE constexpr f256 erf(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::erf(x),
        detail::_f256_runtime::erf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 erfc(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::erfc(x),
        detail::_f256_runtime::erfc(x)
    );
}

// gamma
[[nodiscard]] BL_FORCE_INLINE constexpr f256 lgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::lgamma(x),
        detail::_f256_runtime::lgamma(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256 tgamma(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_impl::tgamma(x),
        detail::_f256_runtime::tgamma(x)
    );
}

} // namespace bl

#endif
