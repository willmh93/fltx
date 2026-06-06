/**
 * fltx/detail/f128_declarations.h - Shared f128 math declarations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_DECLARATIONS_INCLUDED
#define F128_DETAIL_DECLARATIONS_INCLUDED
#include <cstddef>

#include "fltx/f128_arithmetic.h"
#include "fltx/f128_classification.h"
#include "fltx/detail/f128_consts.h"
#include "fltx/detail/common_math.h"
#include "fltx/detail/common_decimal.h"
#include "fltx/detail/math_utils.h"

namespace bl {

namespace detail::_f128_runtime
{
    // roots
    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y);

    // rounding and decimals
    BL_NO_INLINE f128_s trunc(const f128_s& a);
    BL_NO_INLINE f128_s round(const f128_s& a);
    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec);
    BL_NO_INLINE f128_s nearbyint(const f128_s& a);
    BL_NO_INLINE f128_s rint(const f128_s& x);
    BL_NO_INLINE long lround(const f128_s& x);
    BL_NO_INLINE long long llround(const f128_s& x);
    BL_NO_INLINE long lrint(const f128_s& x);
    BL_NO_INLINE long long llrint(const f128_s& x);

    // remainders
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo);

    // fractional decomposition
    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept;

    // decomposition and scaling
    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e);

    // exp / log
    BL_NO_INLINE f128_s exp(const f128_s& x);
    BL_NO_INLINE f128_s exp2(const f128_s& x);
    BL_NO_INLINE f128_s log(const f128_s& a);
    BL_NO_INLINE f128_s log2(const f128_s& a);
    BL_NO_INLINE f128_s log10(const f128_s& x);
    BL_NO_INLINE f128_s expm1(const f128_s& x);
    BL_NO_INLINE f128_s log1p(const f128_s& x);

    // pow
    BL_NO_INLINE f128_s pow10_128(int k);
    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s pow(const f128_s& x, double y);

    // trig
    BL_NO_INLINE bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out);
    BL_NO_INLINE f128_s sin(const f128_s& x);
    BL_NO_INLINE f128_s cos(const f128_s& x);
    BL_NO_INLINE f128_s tan(const f128_s& x);
    BL_NO_INLINE f128_s atan(const f128_s& x);
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x);
    BL_NO_INLINE f128_s asin(const f128_s& x);
    BL_NO_INLINE f128_s acos(const f128_s& x);

    // hyperbolic
    BL_NO_INLINE f128_s sinh(const f128_s& x);
    BL_NO_INLINE f128_s cosh(const f128_s& x);
    BL_NO_INLINE f128_s tanh(const f128_s& x);
    BL_NO_INLINE f128_s asinh(const f128_s& x);
    BL_NO_INLINE f128_s acosh(const f128_s& x);
    BL_NO_INLINE f128_s atanh(const f128_s& x);

    // erf/gamma and polynomials
    BL_NO_INLINE f128_s erf(const f128_s& x);
    BL_NO_INLINE f128_s erfc(const f128_s& x);
    BL_NO_INLINE f128_s lgamma(const f128_s& x);
    BL_NO_INLINE f128_s tgamma(const f128_s& x);
    BL_NO_INLINE f128_s horner_forward(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept;
    BL_NO_INLINE f128_s horner_reverse(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept;
    BL_NO_INLINE void horner_pair_forward(const f128_s* left_coeffs, const f128_s* right_coeffs, std::size_t count, const f128_s& x, f128_s& left_out, f128_s& right_out) noexcept;

} // namespace detail::_f128_runtime

namespace detail::_f128_impl
{
    using namespace detail::_f128;

    // roots
    BL_FORCE_INLINE constexpr f128_s sqrt(f128_s a);
    BL_FORCE_INLINE constexpr f128_s cbrt(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y);

    // rounding and decimals
    BL_FORCE_INLINE constexpr double floor_limb(double x) noexcept;
    BL_FORCE_INLINE constexpr double ceil_limb(double x) noexcept;
    BL_FORCE_INLINE constexpr f128_s floor(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s ceil(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s trunc(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s round(const f128_s& a);
    BL_FORCE_INLINE f128_s round_runtime(const f128_s& a) noexcept;
    BL_FORCE_INLINE constexpr f128_s round_to_decimals(f128_s v, int prec);
    BL_FORCE_INLINE constexpr f128_s nearbyint(const f128_s& a);
    BL_FORCE_INLINE f128_s nearbyint_runtime(const f128_s& a) noexcept;
    BL_FORCE_INLINE constexpr f128_s rint(const f128_s& x);
    BL_FORCE_INLINE constexpr long lround(const f128_s& x);
    BL_FORCE_INLINE constexpr long long llround(const f128_s& x);
    BL_FORCE_INLINE constexpr long lrint(const f128_s& x);
    BL_FORCE_INLINE constexpr long long llrint(const f128_s& x);

    // arithmetic and comparisons
    BL_FORCE_INLINE constexpr f128_s fma(const f128_s& x, const f128_s& y, const f128_s& z);
    BL_FORCE_INLINE constexpr f128_s fmin(const f128_s& a, const f128_s& b);
    BL_FORCE_INLINE constexpr f128_s fmax(const f128_s& a, const f128_s& b);
    BL_FORCE_INLINE constexpr f128_s fdim(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s copysign(const f128_s& x, const f128_s& y);

    // remainders
    BL_FORCE_INLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo);

    // fractional decomposition
    BL_FORCE_INLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept;

    // decomposition and scaling
    BL_FORCE_INLINE constexpr f128_s ldexp(const f128_s& x, int e);
    BL_FORCE_INLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept;
    BL_FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept;
    BL_FORCE_INLINE constexpr f128_s logb(const f128_s& x) noexcept;
    BL_FORCE_INLINE constexpr f128_s scalbn(const f128_s& x, int e) noexcept;
    BL_FORCE_INLINE constexpr f128_s scalbln(const f128_s& x, long e) noexcept;

    // adjacent values
    BL_FORCE_INLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept;
    BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, long double to) noexcept;
    BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, const f128_s& to) noexcept;

    // exp / log
    BL_FORCE_INLINE constexpr double log_as_double(f128_s a);
    BL_FORCE_INLINE constexpr f128_s exp(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s exp2(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s log(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s log2(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s log10(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s expm1(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s log1p(const f128_s& x);

    // pow
    BL_FORCE_INLINE constexpr f128_s pow10_128(int k);
    BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, const f128_s& y);
    BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, double y);

    // trig
    BL_FORCE_INLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out);
    BL_FORCE_INLINE constexpr f128_s sin(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s cos(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tan(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s atan(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s asin(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s acos(const f128_s& x);

    // hyperbolic
    BL_FORCE_INLINE constexpr f128_s sinh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s cosh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tanh(const f128_s& x);
    BL_MSVC_NOINLINE constexpr f128_s asinh(const f128_s& x);
    BL_MSVC_NOINLINE constexpr f128_s acosh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s atanh(const f128_s& x);

    // erf/gamma
    BL_FORCE_INLINE constexpr f128_s erf(const f128_s& x);
    BL_MSVC_NOINLINE constexpr f128_s erfc(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s lgamma(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tgamma(const f128_s& x);

} // namespace detail::_f128_impl

} // namespace bl

#endif
