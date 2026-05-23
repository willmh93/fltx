/**
 * fltx/f128_transcendental.cpp - Runtime f128 transcendental math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f128_math_transcendental.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s horner_forward(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f128_s horner_reverse(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f128_s* left_coeffs,
        const f128_s* right_coeffs,
        std::size_t count,
        const f128_s& x,
        f128_s& left_out,
        f128_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f128_s{};
            right_out = f128_s{};
            return;
        }

        f128_s left  = left_coeffs[0];
        f128_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            detail::_f128::mul_add_pair_same_rhs_inline(left, right, x, left_coeffs[i], right_coeffs[i], left, right);

        left_out = left;
        right_out = right;
    }

    BL_NO_INLINE f128_s exp(const f128_s& x) { return detail::_f128_impl::exp(x); }
    BL_NO_INLINE f128_s exp2(const f128_s& x) { return detail::_f128_impl::exp2(x); }
    BL_NO_INLINE f128_s expm1(const f128_s& x) { return detail::_f128_impl::expm1(x); }

    BL_NO_INLINE f128_s log(const f128_s& a) { return detail::_f128_impl::log(a); }
    BL_NO_INLINE f128_s log2(const f128_s& a) { return detail::_f128_impl::log2(a); }
    BL_NO_INLINE f128_s log10(const f128_s& x) { return detail::_f128_impl::log10(x); }
    BL_NO_INLINE f128_s log1p(const f128_s& x) { return detail::_f128_impl::log1p(x); }

    BL_NO_INLINE f128_s cbrt(const f128_s& x) { return detail::_f128_impl::cbrt(x); }
    BL_NO_INLINE f128_s sinh(const f128_s& x) { return detail::_f128_impl::sinh(x); }
    BL_NO_INLINE f128_s cosh(const f128_s& x) { return detail::_f128_impl::cosh(x); }
    BL_NO_INLINE f128_s tanh(const f128_s& x) { return detail::_f128_impl::tanh(x); }
    BL_NO_INLINE f128_s asinh(const f128_s& x) { return detail::_f128_impl::asinh(x); }
    BL_NO_INLINE f128_s acosh(const f128_s& x) { return detail::_f128_impl::acosh(x); }
    BL_NO_INLINE f128_s atanh(const f128_s& x) { return detail::_f128_impl::atanh(x); }

    BL_NO_INLINE f128_s pow10_128(int k) { return detail::_f128_impl::pow10_128(k); }
    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y) { return detail::_f128_impl::pow(x, y); }
    BL_NO_INLINE f128_s pow(const f128_s& x, double y) { return detail::_f128_impl::pow(x, y); }

    BL_NO_INLINE bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out) { return detail::_f128_impl::sincos(x, s_out, c_out); }
    BL_NO_INLINE f128_s sin(const f128_s& x) { return detail::_f128_impl::sin(x); }
    BL_NO_INLINE f128_s cos(const f128_s& x) { return detail::_f128_impl::cos(x); }
    BL_NO_INLINE f128_s tan(const f128_s& x) { return detail::_f128_impl::tan(x); }
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x) { return detail::_f128_impl::atan2(y, x); }

    BL_NO_INLINE f128_s erf(const f128_s& x) { return detail::_f128_impl::erf(x); }
    BL_NO_INLINE f128_s erfc(const f128_s& x) { return detail::_f128_impl::erfc(x); }
    BL_NO_INLINE f128_s lgamma(const f128_s& x) { return detail::_f128_impl::lgamma(x); }
    BL_NO_INLINE f128_s tgamma(const f128_s& x) { return detail::_f128_impl::tgamma(x); }

} // namespace bl::detail::_f128_runtime
