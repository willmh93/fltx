/**
 * fltx/detail/f128_math_trig.h - Trigonometry implementation details.
 *
 * f128 angle reduction plus sin/cos/tan/atan2 constexpr implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_TRIG_INCLUDED
#define FLTX_F128_DETAIL_TRIG_INCLUDED
#include "fltx/detail/f128_math_sqrt.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    // argument reduction
    BL_FORCE_INLINE constexpr bool remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
	{
	    const double ax = fabs(x.hi);
	    if (!isfinite(ax))
	        return false;

	    if (ax > 7.0e15)
	        return false;

	    const f128_s t = mul_inline(x, invpi2);

	    double qd = nearbyint_ties_even(t.hi);
	    if (!isfinite(qd) ||
	        qd < static_cast<double>(std::numeric_limits<long long>::min()) ||
	        qd > static_cast<double>(std::numeric_limits<long long>::max()))
	    {
	        return false;
	    }

	    const f128_s delta = sub_inline(t, f128_s{ qd });
	    if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
	        qd += 1.0;
	    else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
	        qd -= 1.0;

	    if (qd < static_cast<double>(std::numeric_limits<long long>::min()) ||
	        qd > static_cast<double>(std::numeric_limits<long long>::max()))
	    {
	        return false;
	    }

	    constexpr f128_s pi_2_hi{ pi_2_hi_d };
	    constexpr f128_s pi_2_mid{ pi_2_mid_d };
	    constexpr f128_s pi_2_lo{ pi_2_lo_d };
	    constexpr f128_s pi_4{ pi_4_hi };

	    long long n = static_cast<long long>(qd);
	    const f128_s q{ static_cast<double>(n) };

	    f128_s r = x;
	    r = sub_inline(r, mul_inline(q, pi_2_hi));
	    r = sub_inline(r, mul_inline(q, pi_2_mid));
	    r = sub_inline(r, mul_inline(q, pi_2_lo));

	    if (r > pi_4)
	    {
	        ++n;
	        r = sub_inline(r, pi_2_hi);
	        r = sub_inline(r, pi_2_mid);
	        r = sub_inline(r, pi_2_lo);
	    }
	    else if (r < -pi_4)
	    {
	        --n;
	        r = add_inline(r, pi_2_hi);
	        r = add_inline(r, pi_2_mid);
	        r = add_inline(r, pi_2_lo);
	    }

	    n_out = n;
	    r_out = r;
	    return true;
	}

    // polynomial evaluation
    BL_FORCE_INLINE constexpr f128_s horner_forward_inline(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_FORCE_INLINE constexpr f128_s horner_reverse_inline(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward_inline(const f128_s* left_coeffs, const f128_s* right_coeffs, std::size_t count, const f128_s& x, f128_s& left_out, f128_s& right_out) noexcept
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
            mul_add_pair_same_rhs_inline(left, right, x, left_coeffs[i], right_coeffs[i], left, right);

        left_out = left;
        right_out = right;
    }

    BL_FORCE_INLINE constexpr f128_s horner_forward(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_forward_inline(coeffs, count, x);
        }

        return detail::_f128_runtime::horner_forward(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr f128_s horner_reverse(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_reverse_inline(coeffs, count, x);
        }

        return detail::_f128_runtime::horner_reverse(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward(const f128_s* left_coeffs, const f128_s* right_coeffs, std::size_t count, const f128_s& x, f128_s& left_out, f128_s& right_out) noexcept
    {
        if (bl::use_constexpr_math())
        {
            horner_pair_forward_inline(left_coeffs, right_coeffs, count, x, left_out, right_out);
            return;
        }

        detail::_f128_runtime::horner_pair_forward(left_coeffs, right_coeffs, count, x, left_out, right_out);
    }

    // sine/cosine kernels
    BL_FORCE_INLINE constexpr f128_s sin_kernel_small(const f128_s& x)
    {
        using namespace detail::_f128;

        const f128_s t = mul_inline(x, x);

        const f128_s ps = horner_forward(
            f128_sin_coeffs_pi4 + f128_trig_small_coeff_offset,
            f128_trig_small_coeff_count,
            t);

        return mul_add_inline(mul_inline(x, t), ps, x);
    }

    BL_FORCE_INLINE constexpr f128_s cos_kernel_small(const f128_s& x)
    {
        using namespace detail::_f128;

        const f128_s t = mul_inline(x, x);

        const f128_s pc = horner_forward(
            f128_cos_coeffs_pi4 + f128_trig_small_coeff_offset,
            f128_trig_small_coeff_count,
            t);

        return mul_add_inline(t, pc, f128_s{ 1.0 });
    }

    BL_FORCE_INLINE constexpr void sincos_kernel_small(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        using namespace detail::_f128;

        const f128_s t = mul_inline(x, x);

        f128_s ps{};
        f128_s pc{};
        horner_pair_forward(
            f128_sin_coeffs_pi4 + f128_trig_small_coeff_offset,
            f128_cos_coeffs_pi4 + f128_trig_small_coeff_offset,
            f128_trig_small_coeff_count,
            t,
            ps,
            pc);

        const f128_s xt = mul_inline(x, t);
        s_out = mul_add_inline(xt, ps, x);
        c_out = mul_add_inline(t, pc, f128_s{ 1.0 });
    }

    BL_FORCE_INLINE constexpr void sincos_kernel_pi64_reduced(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        int k = static_cast<int>(nearbyint_ties_even(x.hi * 20.371832715762604));
        if (k < -16)
            k = -16;
        else if (k > 16)
            k = 16;

        if (k == 0)
        {
            sincos_kernel_small(x, s_out, c_out);
            return;
        }

        const f128_s a = mul_double_inline(std::numbers::pi_v<f128_s>, static_cast<double>(k) * 0.015625);
        const f128_s u = sub_inline(x, a);

        f128_s su{}, cu{};
        sincos_kernel_small(u, su, cu);

        const int table_index = k < 0 ? -k : k;
        const f128_s sa = k < 0 ? -f128_sin_table_pi64[table_index] : f128_sin_table_pi64[table_index];
        const f128_s ca = f128_cos_table_pi64[table_index];

        s_out = add_inline(mul_inline(ca, su), mul_inline(sa, cu));
        c_out = sub_inline(mul_inline(ca, cu), mul_inline(sa, su));
    }

    BL_FORCE_INLINE constexpr f128_s sin_kernel_pi4(const f128_s& x)
    {
        const f128_s t = mul_inline(x, x);

        const f128_s ps = horner_forward(f128_sin_coeffs_pi4, f128_trig_coeff_count_pi4, t);

        return mul_add_inline(mul_inline(x, t), ps, x);
    }

    BL_FORCE_INLINE constexpr f128_s cos_kernel_pi4(const f128_s& x)
    {
        const f128_s t = mul_inline(x, x);

        const f128_s pc = horner_forward(f128_cos_coeffs_pi4, f128_trig_coeff_count_pi4, t);

        return mul_add_inline(t, pc, f128_s{ 1.0 });
    }

    BL_FORCE_INLINE constexpr void sincos_kernel_pi4(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        sincos_kernel_pi64_reduced(x, s_out, c_out);
    }

    // arctangent kernels
    BL_FORCE_INLINE constexpr f128_s atan_series_reduced(const f128_s& z)
    {
        constexpr int count = static_cast<int>(sizeof(f128_atan_reduced_coeffs) / sizeof(f128_atan_reduced_coeffs[0]));

        const f128_s z2 = mul_inline(z, z);
        const f128_s p = horner_reverse(f128_atan_reduced_coeffs, static_cast<std::size_t>(count), z2);

        return mul_inline(z, p);
    }

    BL_FORCE_INLINE constexpr f128_s atan_core_unit(const f128_s& z)
    {
        int k = static_cast<int>(nearbyint_ties_even(z.hi * 16.0));
        if (k <= 0)
            return atan_series_reduced(z);
        if (k > 16)
            k = 16;

        const double a = static_cast<double>(k) * 0.0625;
        const f128_s u = div_inline(
            sub_double_inline(z, a),
            add_double_inline(mul_double_inline(z, a), 1.0));

        return add_inline(f128_atan_reduced_table_16[k], atan_series_reduced(u));
    }

    BL_FORCE_INLINE constexpr f128_s _atan(const f128_s& x)
    {
        if (isnan(x))  return x;
        if (iszero(x)) return x;
        if (isinf(x))  return signbit(x.hi) ? -pi_2 : pi_2;

        const bool neg = x.hi < 0.0;
        const f128_s ax = neg ? -x : x;

        if (ax > f128_s{ 1.0 })
        {
            const f128_s core = atan_core_unit(div_inline(f128_s{ 1.0 }, ax));
            const f128_s out  = sub_inline(pi_2, core);
            return neg ? -out : out;
        }

        const f128_s out = atan_core_unit(ax);
        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr f128_s _asin(const f128_s& x)
    {
        if (isnan(x))
            return x;

        const f128_s ax = detail::_f128::mag(x);
        if (ax > f128_s{ 1.0 })
            return std::numeric_limits<f128_s>::quiet_NaN();
        if (ax == f128_s{ 1.0 })
            return (x.hi < 0.0) ? -pi_2 : pi_2;

        if (ax <= f128_s{ 0.5 })
            return _atan(div_inline(x, detail::_f128_impl::sqrt(sub_inline(f128_s{ 1.0 }, mul_inline(x, x)))));

        const f128_s t = detail::_f128_impl::sqrt(div_inline(sub_double_inline(1.0, ax), add_double_inline(ax, 1.0)));
        const f128_s a = sub_inline(pi_2, mul_double_inline(_atan(t), 2.0));
        return (x.hi < 0.0) ? -a : a;
    }

} // namespace detail::_f128

// inverse trig functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::atan(const f128_s& x)
{
    return F128_CANONICALIZE_MATH_RESULT(detail::_f128::_atan(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::asin(const f128_s& x)
{
    return F128_CANONICALIZE_MATH_RESULT(detail::_f128::_asin(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::acos(const f128_s& x)
{
    if (isnan(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    if (ax > f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (x == f128_s{ 1.0 })
        return f128_s{ 0.0 };
    if (x == f128_s{ -1.0 })
        return std::numbers::pi_v<f128_s>;

    return F128_CANONICALIZE_MATH_RESULT(sub_inline(pi_2, detail::_f128::_asin(x)));
}

// sine/cosine functions
[[nodiscard]] BL_MSVC_NOINLINE constexpr bool detail::_f128_impl::sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    const double ax = detail::_f128::fabs(x.hi);
    if (!isfinite(ax))
    {
        s_out = f128_s{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= pi_4_hi)
    {
        sincos_kernel_pi4(x, s_out, c_out);
        s_out = F128_CANONICALIZE_MATH_RESULT(s_out);
        c_out = F128_CANONICALIZE_MATH_RESULT(c_out);
        return true;
    }

    long long n = 0;
    f128_s r{};
    if (!remainder_pio2(x, n, r))
        return false;

    f128_s sr{}, cr{};
    sincos_kernel_pi4(r, sr, cr);

    switch ((int)(n & 3))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    s_out = F128_CANONICALIZE_MATH_RESULT(s_out);
    c_out = F128_CANONICALIZE_MATH_RESULT(c_out);
    return true;
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::sin(const f128_s& x)
{
    const double ax = detail::_f128::fabs(x.hi);
    if (!isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= pi_4_hi)
        return F128_CANONICALIZE_MATH_RESULT(sin_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return F128_CANONICALIZE_MATH_RESULT(f128_s{ detail::fp::sin(static_cast<double>(x)) });
        }
        else
        {
            return F128_CANONICALIZE_MATH_RESULT(f128_s{ std::sin((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return F128_CANONICALIZE_MATH_RESULT(sin_kernel_pi4(r));
    case 1: return F128_CANONICALIZE_MATH_RESULT(cos_kernel_pi4(r));
    case 2: return F128_CANONICALIZE_MATH_RESULT(-sin_kernel_pi4(r));
    default: return F128_CANONICALIZE_MATH_RESULT(-cos_kernel_pi4(r));
    }
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::cos(const f128_s& x)
{
    const double ax = detail::_f128::fabs(x.hi);
    if (!isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= pi_4_hi)
        return F128_CANONICALIZE_MATH_RESULT(cos_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return F128_CANONICALIZE_MATH_RESULT(f128_s{ detail::fp::cos(static_cast<double>(x)) });
        }
        else
        {
            return F128_CANONICALIZE_MATH_RESULT(f128_s{ std::cos((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return F128_CANONICALIZE_MATH_RESULT(cos_kernel_pi4(r));
    case 1: return F128_CANONICALIZE_MATH_RESULT(-sin_kernel_pi4(r));
    case 2: return F128_CANONICALIZE_MATH_RESULT(-cos_kernel_pi4(r));
    default: return F128_CANONICALIZE_MATH_RESULT(sin_kernel_pi4(r));
    }
}

// tangent and atan2
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::tan(const f128_s& x)
{
    f128_s s{}, c{};
    if (detail::_f128_impl::sincos(x, s, c))
        return div_inline(s, c);
    const double xd = (double)x;
    if (bl::use_constexpr_math()) {
        return f128_s{ detail::fp::tan(xd) };
    } else {
        return f128_s{ std::tan(xd) };
    }
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::atan2(const f128_s& y, const f128_s& x)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (iszero(x))
    {
        if (iszero(y))
            return f128_s{ std::numeric_limits<double>::quiet_NaN() };
        return ispositive(y) ? detail::_f128::pi_2 : -detail::_f128::pi_2;
    }

    if (iszero(y))
    {
        if (x.hi < 0.0)
            return signbit(y.hi) ? -std::numbers::pi_v<f128_s> : std::numbers::pi_v<f128_s>;
        return y;
    }

    const f128_s ax = detail::_f128::mag(x);
    const f128_s ay = detail::_f128::mag(y);

    if (ax == ay)
    {
        if (x.hi < 0.0)
        {
            return F128_CANONICALIZE_MATH_RESULT(
                (y.hi < 0.0) ? -detail::_f128::pi_3_4 : detail::_f128::pi_3_4);
        }

        return F128_CANONICALIZE_MATH_RESULT(
            (y.hi < 0.0) ? -detail::_f128::pi_4 : detail::_f128::pi_4);
    }

    if (ax >= ay)
    {
        f128_s a = detail::_f128::_atan(detail::_f128::div_inline(y, x));

        if (x.hi < 0.0)
            a = detail::_f128::add_inline(a, (y.hi < 0.0) ? -std::numbers::pi_v<f128_s> : std::numbers::pi_v<f128_s>);
        return F128_CANONICALIZE_MATH_RESULT(a);
    }

    const f128_s a = detail::_f128::_atan(detail::_f128::div_inline(x, y));
    return F128_CANONICALIZE_MATH_RESULT(
        (y.hi < 0.0) ? detail::_f128::sub_inline(-detail::_f128::pi_2, a) : detail::_f128::sub_inline(detail::_f128::pi_2, a));
}

} // namespace bl

#endif
