/**
 * fltx/f256_math.cpp - Core f256 runtime helpers and common math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256_math_basic.h"
#include "fltx/detail/f256_math_erf.h"
#include "fltx/detail/f256_math_hyperbolic.h"
#include "fltx/detail/f256_math_trig.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::mul_add_horner_step_inline(a, b, c);
    }

    BL_NO_INLINE f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f256_s{};
            right_out = f256_s{};
            return;
        }

        f256_s left  = left_coeffs[0];
        f256_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            left = detail::_f256::mul_add_inline(left, x, left_coeffs[i]);
            right = detail::_f256::mul_add_inline(right, x, right_coeffs[i]);
        }

        left_out = left;
        right_out = right;
    }

    BL_NO_INLINE f256_s cheb_eval(const f256_s& x, const f256_s* coeffs, std::size_t count, double shift) noexcept
    {
        if (count == 0)
            return {};

        const f256_s t = detail::_f256::sub_inline(
            detail::_f256::mul_double_inline(x, 2.0),
            f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (std::size_t i = count - 1; i >= 1; --i)
        {
            const f256_s b0 = detail::_f256::add_inline(
                detail::_f256::mul_double_sub_inline(detail::_f256::mul_inline(t, b1), 2.0, b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return detail::_f256::add_inline(detail::_f256::mul_sub_inline(t, b1, b2), coeffs[0]);
    }

    BL_NO_INLINE f256_s log1p_series_reduced(const f256_s& x) noexcept
    {
        const f256_s z = detail::_f256::div_add_double_inline(x, x, 2.0);
        const f256_s z2 = detail::_f256::sqr_inline(z);

        f256_s term = z;
        f256_s sum  = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term = detail::_f256::mul_inline(term, z2);
            const f256_s add = detail::_f256::div_double_inline(term, static_cast<double>(k));
            sum = detail::_f256::add_inline(sum, add);

            const f256_s asum  = detail::_f256::mag(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (detail::_f256::mag(add) <= detail::_f256::mul_inline(f256_s::eps(), scale))
                break;
        }

        return detail::_f256::add_inline(sum, sum);
    }

    BL_NO_INLINE f256_s fmod(const f256_s& x, const f256_s& y) { return detail::_f256_impl::fmod(x, y); }
    BL_NO_INLINE f256_s round(const f256_s& a) { return detail::_f256_impl::round(a); }
    BL_NO_INLINE f256_s round_to_decimals(f256_s v, int prec) { return detail::_f256_impl::round_to_decimals(v, prec); }
    BL_NO_INLINE f256_s sqrt(const f256_s& a) { return detail::_f256_impl::sqrt(a); }
    BL_NO_INLINE f256_s nearbyint(const f256_s& a) { return detail::_f256_impl::nearbyint(a); }

    BL_NO_INLINE long lround(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
    }

    BL_NO_INLINE long long llround(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
    }

    BL_NO_INLINE long lrint(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_impl::nearbyint(x));
    }

    BL_NO_INLINE long long llrint(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_impl::nearbyint(x));
    }

    BL_NO_INLINE f256_s ldexp(const f256_s& a, int e) { return detail::_f256_impl::ldexp(a, e); }

    BL_NO_INLINE f256_s cbrt(const f256_s& x) { return detail::_f256_impl::cbrt(x); }
    BL_NO_INLINE f256_s hypot(const f256_s& x, const f256_s& y) { return detail::_f256_impl::hypot(x, y); }
    BL_NO_INLINE f256_s remquo(const f256_s& x, const f256_s& y, int* quo) { return detail::_f256_impl::remquo(x, y, quo); }
    BL_NO_INLINE f256_s remainder(const f256_s& x, const f256_s& y) { return detail::_f256_impl::remainder(x, y); }
    BL_NO_INLINE f256_s frexp(const f256_s& x, int* exp) noexcept { return detail::_f256_impl::frexp(x, exp); }
    BL_NO_INLINE f256_s modf(const f256_s& x, f256_s* iptr) noexcept { return detail::_f256_impl::modf(x, iptr); }
    BL_NO_INLINE int ilogb(const f256_s& x) noexcept { return detail::_f256_impl::ilogb(x); }
    BL_NO_INLINE f256_s logb(const f256_s& x) noexcept { return detail::_f256_impl::logb(x); }
    BL_NO_INLINE f256_s scalbn(const f256_s& x, int e) noexcept { return detail::_f256_impl::scalbn(x, e); }
    BL_NO_INLINE f256_s scalbln(const f256_s& x, long e) noexcept { return detail::_f256_impl::scalbln(x, e); }
    BL_NO_INLINE f256_s nextafter(const f256_s& from, const f256_s& to) noexcept { return detail::_f256_impl::nextafter(from, to); }
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, long double to) noexcept { return detail::_f256_impl::nexttoward(from, to); }
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept { return detail::_f256_impl::nexttoward(from, to); }

    BL_NO_INLINE f256_s erf(const f256_s& x) { return detail::_f256_impl::erf(x); }
    BL_NO_INLINE f256_s erfc(const f256_s& x) { return detail::_f256_impl::erfc(x); }

    BL_NO_INLINE f256_s sinh(const f256_s& x) { return detail::_f256_impl::sinh(x); }
    BL_NO_INLINE f256_s cosh(const f256_s& x) { return detail::_f256_impl::cosh(x); }
    BL_NO_INLINE f256_s tanh(const f256_s& x) { return detail::_f256_impl::tanh(x); }
    BL_NO_INLINE f256_s asinh(const f256_s& x) { return detail::_f256_impl::asinh(x); }
    BL_NO_INLINE f256_s acosh(const f256_s& x) { return detail::_f256_impl::acosh(x); }
    BL_NO_INLINE f256_s atanh(const f256_s& x) { return detail::_f256_impl::atanh(x); }

    BL_NO_INLINE bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out) { return detail::_f256_impl::sincos(x, s_out, c_out); }
    BL_NO_INLINE f256_s sin(const f256_s& x) { return detail::_f256_impl::sin(x); }
    BL_NO_INLINE f256_s cos(const f256_s& x) { return detail::_f256_impl::cos(x); }
    BL_NO_INLINE f256_s tan(const f256_s& x) { return detail::_f256_impl::tan(x); }
    BL_NO_INLINE f256_s atan(const f256_s& x) { return detail::_f256_impl::atan(x); }
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x) { return detail::_f256_impl::atan2(y, x); }
    BL_NO_INLINE f256_s asin(const f256_s& x) { return detail::_f256_impl::asin(x); }
    BL_NO_INLINE f256_s acos(const f256_s& x) { return detail::_f256_impl::acos(x); }

} // namespace bl::detail::_f256_runtime
