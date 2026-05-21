/**
 * fltx/f256/detail/declarations.h - shared f256 math declarations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_DECLARATIONS_INCLUDED
#define FLTX_F256_DETAIL_DECLARATIONS_INCLUDED
#include <cstddef>

#include "fltx/detail/f256/expressions.h"
#include "fltx/detail/f256/consts.h"
#include "fltx/detail/common/math.h"
#include "fltx/detail/decimal.h"
#include "fltx/detail/math_utils.h"

namespace bl {

namespace detail::_f256_runtime
{
    BL_NO_INLINE f256_s fmod(const f256_s& x, const f256_s& y);
    BL_NO_INLINE f256_s round(const f256_s& a);
    BL_NO_INLINE f256_s round_to_decimals(f256_s v, int prec);
    BL_NO_INLINE f256_s sqrt(const f256_s& a);
    BL_NO_INLINE f256_s nearbyint(const f256_s& a);
    BL_NO_INLINE f256_s rint(const f256_s& x);
    BL_NO_INLINE long lround(const f256_s& x);
    BL_NO_INLINE long long llround(const f256_s& x);
    BL_NO_INLINE long lrint(const f256_s& x);
    BL_NO_INLINE long long llrint(const f256_s& x);
    BL_NO_INLINE f256_s ldexp(const f256_s& a, int e);

    BL_NO_INLINE f256_s exp(const f256_s& x);
    BL_NO_INLINE f256_s exp2(const f256_s& x);
    BL_NO_INLINE f256_s log(const f256_s& a);
    BL_NO_INLINE f256_s log2(const f256_s& a);
    BL_NO_INLINE f256_s log10(const f256_s& a);
    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y);
    BL_NO_INLINE f256_s pow(const f256_s& x, double y);

    BL_NO_INLINE bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out);
    BL_NO_INLINE f256_s sin(const f256_s& x);
    BL_NO_INLINE f256_s cos(const f256_s& x);
    BL_NO_INLINE f256_s tan(const f256_s& x);
    BL_NO_INLINE f256_s atan(const f256_s& x);
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x);
    BL_NO_INLINE f256_s asin(const f256_s& x);
    BL_NO_INLINE f256_s acos(const f256_s& x);

    BL_NO_INLINE f256_s expm1(const f256_s& x);
    BL_NO_INLINE f256_s log1p(const f256_s& x);
    BL_NO_INLINE f256_s sinh(const f256_s& x);
    BL_NO_INLINE f256_s cosh(const f256_s& x);
    BL_NO_INLINE f256_s tanh(const f256_s& x);
    BL_NO_INLINE f256_s asinh(const f256_s& x);
    BL_NO_INLINE f256_s acosh(const f256_s& x);
    BL_NO_INLINE f256_s atanh(const f256_s& x);

    BL_NO_INLINE f256_s cbrt(const f256_s& x);
    BL_NO_INLINE f256_s hypot(const f256_s& x, const f256_s& y);
    BL_NO_INLINE f256_s remquo(const f256_s& x, const f256_s& y, int* quo);
    BL_NO_INLINE f256_s remainder(const f256_s& x, const f256_s& y);
    BL_NO_INLINE f256_s frexp(const f256_s& x, int* exp) noexcept;
    BL_NO_INLINE f256_s modf(const f256_s& x, f256_s* iptr) noexcept;
    BL_NO_INLINE int ilogb(const f256_s& x) noexcept;
    BL_NO_INLINE f256_s logb(const f256_s& x) noexcept;
    BL_NO_INLINE f256_s scalbn(const f256_s& x, int e) noexcept;
    BL_NO_INLINE f256_s scalbln(const f256_s& x, long e) noexcept;
    BL_NO_INLINE f256_s nextafter(const f256_s& from, const f256_s& to) noexcept;
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, long double to) noexcept;
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept;

    BL_NO_INLINE f256_s erf(const f256_s& x);
    BL_NO_INLINE f256_s erfc(const f256_s& x);
    BL_NO_INLINE f256_s lgamma(const f256_s& x);
    BL_NO_INLINE f256_s tgamma(const f256_s& x);
    BL_NO_INLINE f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept;
    BL_NO_INLINE f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept;
    BL_NO_INLINE f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept;
    BL_NO_INLINE void horner_pair_forward(const f256_s* left_coeffs, const f256_s* right_coeffs, std::size_t count, const f256_s& x, f256_s& left_out, f256_s& right_out) noexcept;
    BL_NO_INLINE f256_s cheb_eval(const f256_s& x, const f256_s* coeffs, std::size_t count, double shift) noexcept;
    BL_NO_INLINE f256_s log1p_series_reduced(const f256_s& x) noexcept;

} // namespace detail::_f256_runtime

namespace detail::_f256_constexpr
{
    using namespace detail::_f256;

    BL_FORCE_INLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y);
    BL_FORCE_INLINE constexpr f256_s round(const f256_s& a);
    BL_FORCE_INLINE constexpr f256_s round_to_decimals(f256_s v, int prec);
    BL_FORCE_INLINE constexpr f256_s sqrt(const f256_s& a);
    BL_FORCE_INLINE constexpr f256_s nearbyint(const f256_s& a);
    BL_FORCE_INLINE constexpr f256_s rint(const f256_s& x);
    BL_FORCE_INLINE constexpr long lround(const f256_s& x);
    BL_FORCE_INLINE constexpr long long llround(const f256_s& x);
    BL_FORCE_INLINE constexpr long lrint(const f256_s& x);
    BL_FORCE_INLINE constexpr long long llrint(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s ldexp(const f256_s& a, int e);

    BL_FORCE_INLINE constexpr f256_s exp(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s exp2(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s log(const f256_s& a);
    BL_FORCE_INLINE constexpr f256_s log2(const f256_s& a);
    BL_FORCE_INLINE constexpr f256_s log10(const f256_s& a);
    BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, const f256_s& y);
    BL_MSVC_NOINLINE constexpr f256_s pow(const f256_s& x, double y);

    BL_FORCE_INLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out);
    BL_FORCE_INLINE constexpr f256_s sin(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s cos(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s tan(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s atan(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s asin(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s acos(const f256_s& x);

    BL_FORCE_INLINE constexpr f256_s expm1(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s log1p(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s sinh(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s cosh(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s tanh(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s asinh(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s acosh(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s atanh(const f256_s& x);

    BL_FORCE_INLINE constexpr f256_s cbrt(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s hypot(const f256_s& x, const f256_s& y);
    BL_FORCE_INLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo);
    BL_FORCE_INLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y);
    BL_FORCE_INLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept;
    BL_FORCE_INLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept;
    BL_FORCE_INLINE constexpr int ilogb(const f256_s& x) noexcept;
    BL_FORCE_INLINE constexpr f256_s logb(const f256_s& x) noexcept;
    BL_FORCE_INLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept;
    BL_FORCE_INLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept;
    BL_FORCE_INLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept;
    BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept;
    BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept;

    BL_FORCE_INLINE constexpr f256_s erf(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s erfc(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s lgamma(const f256_s& x);
    BL_FORCE_INLINE constexpr f256_s tgamma(const f256_s& x);

} // namespace detail::_f256_constexpr

} // namespace bl

#endif
