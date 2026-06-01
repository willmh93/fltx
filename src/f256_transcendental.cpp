/**
 * fltx/f256_transcendental.cpp - Runtime f256 transcendental math functions.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/detail/f256_math_transcendental.h"

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

    // roots
    BL_NO_INLINE f256_s cbrt(const f256_s& x)
    {
        return detail::_f256_impl::cbrt(x);
    }

    // exponential and logarithmic
    BL_NO_INLINE f256_s exp(const f256_s& x)
    {
        return detail::_f256_impl::exp(x);
    }

    BL_NO_INLINE f256_s exp2(const f256_s& x)
    {
        return detail::_f256_impl::exp2(x);
    }

    BL_NO_INLINE f256_s log(const f256_s& a)
    {
        return detail::_f256_impl::log(a);
    }

    BL_NO_INLINE f256_s log2(const f256_s& a)
    {
        return detail::_f256_impl::log2(a);
    }

    BL_NO_INLINE f256_s log10(const f256_s& a)
    {
        return detail::_f256_impl::log10(a);
    }

    BL_NO_INLINE f256_s expm1(const f256_s& x)
    {
        return detail::_f256_impl::expm1(x);
    }

    BL_NO_INLINE f256_s log1p(const f256_s& x)
    {
        return detail::_f256_impl::log1p(x);
    }

    // powers
    BL_NO_INLINE f256_s pow10_256(int k)
    {
        return detail::_f256_impl::pow10_256(k);
    }

    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y)
    {
        return detail::_f256_impl::pow(x, y);
    }

    BL_NO_INLINE f256_s pow(const f256_s& x, double y)
    {
        return detail::_f256_impl::pow(x, y);
    }

    // trigonometric
    BL_NO_INLINE bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
    {
        return detail::_f256_impl::sincos(x, s_out, c_out);
    }

    BL_NO_INLINE f256_s sin(const f256_s& x)
    {
        return detail::_f256_impl::sin(x);
    }

    BL_NO_INLINE f256_s cos(const f256_s& x)
    {
        return detail::_f256_impl::cos(x);
    }

    BL_NO_INLINE f256_s tan(const f256_s& x)
    {
        return detail::_f256_impl::tan(x);
    }

    BL_NO_INLINE f256_s atan(const f256_s& x)
    {
        return detail::_f256_impl::atan(x);
    }

    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x)
    {
        return detail::_f256_impl::atan2(y, x);
    }

    BL_NO_INLINE f256_s asin(const f256_s& x)
    {
        return detail::_f256_impl::asin(x);
    }

    BL_NO_INLINE f256_s acos(const f256_s& x)
    {
        return detail::_f256_impl::acos(x);
    }

    // hyperbolic
    BL_NO_INLINE f256_s sinh(const f256_s& x)
    {
        return detail::_f256_impl::sinh(x);
    }

    BL_NO_INLINE f256_s cosh(const f256_s& x)
    {
        return detail::_f256_impl::cosh(x);
    }

    BL_NO_INLINE f256_s tanh(const f256_s& x)
    {
        return detail::_f256_impl::tanh(x);
    }

    BL_NO_INLINE f256_s asinh(const f256_s& x)
    {
        return detail::_f256_impl::asinh(x);
    }

    BL_NO_INLINE f256_s acosh(const f256_s& x)
    {
        return detail::_f256_impl::acosh(x);
    }

    BL_NO_INLINE f256_s atanh(const f256_s& x)
    {
        return detail::_f256_impl::atanh(x);
    }

    // special functions
    BL_NO_INLINE f256_s erf(const f256_s& x)
    {
        return detail::_f256_impl::erf(x);
    }

    BL_NO_INLINE f256_s erfc(const f256_s& x)
    {
        return detail::_f256_impl::erfc(x);
    }

    BL_NO_INLINE f256_s lgamma(const f256_s& x)
    {
        return detail::_f256_impl::lgamma(x);
    }

    BL_NO_INLINE f256_s tgamma(const f256_s& x)
    {
        return detail::_f256_impl::tgamma(x);
    }

} // namespace bl::detail::_f256_runtime
