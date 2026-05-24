/**
 * fltx/detail/f128_math_transcendental.h - f128 transcendental math implementation details.
 *
 * f128 cbrt, exp/log, pow, trig, hyperbolic, erf, and gamma implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_MATH_TRANSCENDENTAL_INCLUDED
#define F128_DETAIL_MATH_TRANSCENDENTAL_INCLUDED
#include "fltx/detail/f128_math_basic.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::log_as_double(f128_s a);

namespace detail::_f128 // primitives and kernels
{
    // expm1/log1p functions
    BL_FORCE_INLINE constexpr f128_s log1p_double_seed_residual(const f128_s& r) noexcept
    {
        const f128_s r2 = mul_inline(r, r);
        const f128_s r3 = mul_inline(r2, r);
        const f128_s r4 = mul_inline(r2, r2);
        const f128_s r5 = mul_inline(r4, r);

        f128_s correction = r;
        correction = sub_inline(correction, r2 * 0.5);
        correction = add_inline(correction, r3 / 3.0);
        correction = sub_inline(correction, r4 * 0.25);
        correction = add_inline(correction, r5 * 0.2);
        return correction;
    }

    BL_MSVC_NOINLINE constexpr f128_s log1p_series_reduced(const f128_s& x)
    {
        const f128_s z = div_inline(x, add_inline(f128_s{ 2.0 }, x));
        const f128_s z2 = mul_inline(z, z);

        f128_s term = z;
        f128_s sum  = z;

        for (int k = 3; k <= 81; k += 2)
        {
            term = mul_inline(term, z2);
            const f128_s add = div_inline(term, f128_s{ static_cast<double>(k) });
            sum = add_inline(sum, add);

            const f128_s asum  = mag(sum);
            const f128_s scale = (asum > f128_s{ 1.0 }) ? asum : f128_s{ 1.0 };
            if (mag(add) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return add_inline(sum, sum);
    }

    // exponential functions
    BL_MSVC_NOINLINE constexpr f128_s expm1_tiny(const f128_s& r)
    {
        f128_s p = exp_inv_fact[(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 1];
        for (int i = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 2; i >= 0; --i)
            p = mul_add_inline(p, r, exp_inv_fact[i]);
        p = mul_add_inline(p, r, f128_s{0.5});
        return mul_add_inline(mul_inline(r, r), p, r);
    }

    BL_MSVC_NOINLINE constexpr f128_s _exp(const f128_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.hi < 0.0) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();

        if (x.hi > 709.782712893384)
            return std::numeric_limits<f128_s>::infinity();

        if (x.hi < -745.133219101941)
            return f128_s{ 0.0 };

        if (iszero(x))
            return f128_s{ 1.0 };

        const f128_s t = mul_inline(x, std::numbers::log2e_v<f128_s>);

        double kd = nearbyint_ties_even(t.hi);
        const f128_s delta = sub_inline(t, f128_s{ kd });
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s r = mul_inline(sub_inline(x, mul_inline(f128_s{ kd }, std::numbers::ln2_v<f128_s>)), f128_s{ 0.001953125 });

        f128_s e = expm1_tiny(r);
        for (int i = 0; i < 9; ++i)
            e = mul_add_inline(e, e, e * 2.0);

        return _ldexp(add_inline(e, f128_s{ 1.0 }), k);
    }

    BL_MSVC_NOINLINE constexpr f128_s _exp2(const f128_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.hi < 0.0) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();

        if (x.hi > 1023.0 || x.hi < -1074.0)
            return _exp(mul_inline(x, std::numbers::ln2_v<f128_s>));

        if (iszero(x))
            return f128_s{ 1.0 };

        double kd = nearbyint_ties_even(x.hi);
        const f128_s delta = sub_inline(x, f128_s{ kd });
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s reduced = sub_inline(x, f128_s{ kd });
        const f128_s r = mul_inline(mul_inline(reduced, std::numbers::ln2_v<f128_s>), f128_s{ 0.001953125 });

        f128_s e = expm1_tiny(r);
        for (int i = 0; i < 9; ++i)
            e = mul_add_inline(e, e, e * 2.0);

        return _ldexp(add_inline(e, f128_s{ 1.0 }), k);
    }

    // logarithm functions
    BL_MSVC_NOINLINE constexpr f128_s _log(const f128_s& a)
    {
        if (isnan(a))
            return a;
        if (iszero(a))
            return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
        if (a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0))
            return std::numeric_limits<f128_s>::quiet_NaN();
        if (isinf(a))
            return a;

        int exp2 = 0;
        if (bl::use_constexpr_math()) {
            exp2 = detail::fp::frexp_exponent(a.hi);
        }
        else {
            (void)std::frexp(a.hi, &exp2);
        }

        f128_s m = _ldexp(a, -exp2);
        if (m < sqrt_half)
        {
            m = mul_inline(m, f128_s{ 2.0 });
            --exp2;
        }

        const f128_s exp2_ln2 = mul_inline(f128_s{ static_cast<double>(exp2) }, std::numbers::ln2_v<f128_s>);
        f128_s y = add_inline(exp2_ln2, f128_s{ detail::_f128_impl::log_as_double(m) });
        if (bl::use_constexpr_math())
        {
            y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
            y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
            y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
        }
        else
        {
            const f128_s residual = mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 });
            y = add_inline(y, log1p_double_seed_residual(residual));
        }
        return y;
    }

    BL_FORCE_INLINE constexpr bool f128_try_exact_binary_log2(const f128_s& x, int& out) noexcept
    {
        if (!(x.hi > 0.0) || x.lo != 0.0)
            return false;

        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x.hi);
        const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
        const std::uint64_t frac_bits = bits & ((std::uint64_t{ 1 } << 52) - 1);

        if (exp_bits == 0 || frac_bits != 0)
            return false;

        out = static_cast<int>(exp_bits) - 1023;
        return true;
    }

    // power functions
    BL_FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

    // sine/cosine functions
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

    // inverse trig functions
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

    // inverse hyperbolic functions
    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum   = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (mag(term) <= f128_s::eps())
                break;
        }

        return sum;
    }

    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series_runtime(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum   = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (mag(term) <= f128_s::eps())
                break;
        }

        return sum;
    }

    // erf/erfc functions
    BL_FORCE_INLINE constexpr f128_s erf_positive_series(const f128_s& x)
    {
        const f128_s xx = mul_inline(x, x);
        f128_s power = x;
        f128_s sum   = x;

        for (int n = 1; n < 256; ++n)
        {
            power = mul_inline(power, div_inline(-xx, f128_s{ static_cast<double>(n) }));
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * n + 1) });
            sum = add_inline(sum, term);
            if (mag(term) < f128_s::eps())
                break;
        }

        return F128_CANONICALIZE_MATH_RESULT(mul_inline(mul_inline(f128_s{ 2.0 }, std::numbers::inv_sqrtpi_v<f128_s>), sum));
    }

    BL_FORCE_INLINE constexpr f128_s erfc_positive_cf(const f128_s& x)
    {
        const f128_s z = mul_inline(x, x);
        constexpr f128_s a = f128_s{ 0.5 };
        constexpr f128_s tiny = f128_s{ 1.0e-300 };

        f128_s b = sub_inline(add_inline(z, f128_s{ 1.0 }), a);
        f128_s c = div_inline(f128_s{ 1.0 }, tiny);
        f128_s d = div_inline(f128_s{ 1.0 }, b);
        f128_s h = d;

        for (int i = 1; i <= 96; ++i)
        {
            const f128_s ii = f128_s{ static_cast<double>(i) };
            const f128_s an = -mul_inline(ii, sub_inline(ii, a));

            b = add_inline(b, f128_s{ 2.0 });

            d = mul_add_inline(an, d, b);
            if (mag(d) < tiny)
                d = tiny;

            c = add_inline(b, div_inline(an, c));
            if (mag(c) < tiny)
                c = tiny;

            d = div_inline(f128_s{ 1.0 }, d);
            const f128_s delta = mul_inline(d, c);
            h = mul_inline(h, delta);

            if (mag(sub_inline(delta, f128_s{ 1.0 })) <= mul_inline(f128_s{ 32.0 }, f128_s::eps()))
                break;
        }

        const f128_s out = mul_inline(mul_inline(mul_inline(detail::_f128_impl::exp(-z), x), std::numbers::inv_sqrtpi_v<f128_s>), h);
        return F128_CANONICALIZE_MATH_RESULT(out);
    }

    // gamma functions
    BL_MSVC_NOINLINE constexpr f128_s lgamma1p_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        const f128_s p = horner_reverse(lgamma1p_coeff, static_cast<std::size_t>(count), y);

        return mul_inline(y, mul_add_inline(y, p, -std::numbers::egamma_v<f128_s>));
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma1p5_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p5_coeff) / sizeof(lgamma1p5_coeff[0]));

        const f128_s p = horner_reverse(lgamma1p5_coeff, static_cast<std::size_t>(count), y);

        const f128_s constant = sub_inline(half_log_two_pi, mul_inline(f128_s{ 1.5 }, std::numbers::ln2_v<f128_s>));
        const f128_s linear   = sub_inline(sub_inline(f128_s{ 2.0 }, std::numbers::egamma_v<f128_s>), mul_inline(f128_s{ 2.0 }, std::numbers::ln2_v<f128_s>));
        return mul_add_inline(y, mul_add_inline(y, p, linear), constant);
    }

    BL_MSVC_NOINLINE constexpr bool try_lgamma_near_one_or_two(const f128_s& x, f128_s& out) noexcept
    {
        const f128_s y1 = sub_inline(x, f128_s{ 1.0 });
        if (mag(y1) <= f128_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f128_s y15 = sub_inline(x, f128_s{ 1.5 });
        if (mag(y15) <= f128_s{ 0.25 })
        {
            out = lgamma1p5_series(y15);
            return true;
        }

        const f128_s y2 = sub_inline(x, f128_s{ 2.0 });
        if (mag(y2) <= f128_s{ 0.25 })
        {
            out = add_inline(log1p_series_reduced(y2), lgamma1p_series(y2));
            return true;
        }

        return false;
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma_stirling_asymptotic(const f128_s& z) noexcept
    {
        const f128_s inv    = div_inline(f128_s{ 1.0 }, z);
        const f128_s inv2   = mul_inline(inv, inv);
        const f128_s series = mul_inline(inv, horner_reverse(
            lgamma_stirling_coeffs,
            sizeof(lgamma_stirling_coeffs) / sizeof(lgamma_stirling_coeffs[0]),
            inv2));

        return add_inline(add_inline(sub_inline(mul_inline(sub_inline(z, f128_s{ 0.5 }), detail::_f128_impl::log(z)), z), half_log_two_pi), series);
    }

    BL_MSVC_NOINLINE constexpr void positive_recurrence_product(const f128_s& x, const f128_s& asymptotic_min, f128_s& z, f128_s& product, int& product_scale2) noexcept
    {
        z = x;
        product = f128_s{ 1.0 };
        product_scale2 = 0;

        while (z < asymptotic_min)
        {
            product = mul_inline(product, z);

            const double hi = product.hi;
            if (hi != 0.0)
            {
                const int exponent = frexp_exponent(hi);
                if (exponent > 512 || exponent < -512)
                {
                    product = detail::_f128_impl::ldexp(product, -exponent);
                    product_scale2 += exponent;
                }
            }

            z = add_inline(z, f128_s{ 1.0 });
        }
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma_positive_low_range(const f128_s& x) noexcept
    {
        f128_s y = x;
        f128_s product{ 1.0 };
        bool shifted_up = false;

        if (y < f128_s{ 0.75 })
        {
            shifted_up = true;
            do
            {
                product = mul_inline(product, y);
                y = add_inline(y, f128_s{ 1.0 });
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y = sub_inline(y, f128_s{ 1.0 });
                product = mul_inline(product, y);
            }
        }

        f128_s local{};
        try_lgamma_near_one_or_two(y, local);

        if (product == f128_s{ 1.0 })
            return local;

        const f128_s correction = detail::_f128_impl::log(product);
        return shifted_up ? sub_inline(local, correction) : add_inline(local, correction);
    }

    BL_MSVC_NOINLINE constexpr f128_s gamma_positive_low_range(const f128_s& x) noexcept
    {
        f128_s y = x;
        f128_s product{ 1.0 };
        bool shifted_up = false;

        if (y < f128_s{ 0.75 })
        {
            shifted_up = true;
            do
            {
                product = mul_inline(product, y);
                y = add_inline(y, f128_s{ 1.0 });
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y = sub_inline(y, f128_s{ 1.0 });
                product = mul_inline(product, y);
            }
        }

        f128_s local_lgamma{};
        try_lgamma_near_one_or_two(y, local_lgamma);
        const f128_s local_gamma = detail::_f128_impl::exp(local_lgamma);
        return shifted_up ? div_inline(local_gamma, product) : mul_inline(local_gamma, product);
    }

    BL_MSVC_NOINLINE constexpr f128_s lgamma_positive_recurrence(const f128_s& x) noexcept
    {
        f128_s near_value{};
        if (try_lgamma_near_one_or_two(x, near_value))
            return near_value;

        if (x <= f128_s{ 16.0 })
            return lgamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z{};
        f128_s product{};
        int product_scale2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_scale2);

        return sub_inline(
            sub_inline(lgamma_stirling_asymptotic(z), detail::_f128_impl::log(product)),
            mul_inline(f128_s{ static_cast<double>(product_scale2) }, std::numbers::ln2_v<f128_s>));
    }

    BL_MSVC_NOINLINE constexpr f128_s gamma_positive_recurrence(const f128_s& x) noexcept
    {
        f128_s near_lgamma{};
        if (try_lgamma_near_one_or_two(x, near_lgamma))
            return detail::_f128_impl::exp(near_lgamma);

        if (x <= f128_s{ 16.0 })
            return gamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z{};
        f128_s product{};
        int product_scale2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_scale2);

        f128_s out = div_inline(detail::_f128_impl::exp(lgamma_stirling_asymptotic(z)), product);
        if (product_scale2 != 0)
            out = detail::_f128_impl::ldexp(out, -product_scale2);

        return out;
    }

} // namespace detail::_f128

// exponential functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::exp(const f128_s& x)
{
    return F128_CANONICALIZE_MATH_RESULT(_exp(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::exp2(const f128_s& x)
{
    return F128_CANONICALIZE_MATH_RESULT(_exp2(x));
}

// logarithm functions
[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::log_as_double(f128_s a)
{
    const double hi = a.hi;
    if (hi <= 0.0)
        return detail::fp::log(static_cast<double>(a));

    return detail::fp::log(hi) + detail::fp::log1p(a.lo / hi);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::log(const f128_s& a)
{
    return F128_CANONICALIZE_MATH_RESULT(_log(a));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::log2(const f128_s& a)
{
    int exact_exp2{};
    if (f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(mul_inline(_log(a), std::numbers::log2e_v<f128_s>));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::log10(const f128_s& x)
{
    if (x.hi > 0.0)
    {
        const int exp2 =
            detail::fp::frexp_exponent(x.hi);
        const int k0 =
            static_cast<int>(detail::fp::floor((exp2 - 1) * 0.30102999566398114));

        for (int k = k0 - 2; k <= k0 + 2; ++k)
        {
            if (x == detail::_f128_impl::pow10_128(k))
                return f128_s{ static_cast<double>(k), 0.0 };
        }
    }

    return F128_CANONICALIZE_MATH_RESULT(mul_inline(_log(x), std::numbers::log10e_v<f128_s>));
}

// expm1/log1p functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::expm1(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return x;
    if (isinf(x))
        return signbit(x)
            ? f128_s{ -1.0, 0.0 }
            : std::numeric_limits<f128_s>::infinity();

    const f128_s ax = detail::_f128::mag(x);
    if (ax <= f128_s{ 0.5 })
    {
        f128_s term = x;
        f128_s sum  = x;

        for (int n = 2; n <= 80; ++n)
        {
            term = div_inline(mul_inline(term, x), f128_s{ static_cast<double>(n) });
            sum = add_inline(sum, term);

            const f128_s abs_sum = detail::_f128::mag(sum);
            const f128_s scale = (abs_sum < f128_s{ 1.0 }) ? f128_s{ 1.0 } : abs_sum;
            if (detail::_f128::mag(term) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return F128_CANONICALIZE_MATH_RESULT(sum);
    }

    return F128_CANONICALIZE_MATH_RESULT(sub_inline(detail::_f128_impl::exp(x), f128_s{ 1.0 }));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::log1p(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x == f128_s{ -1.0 })
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (x < f128_s{ -1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(x))
        return x;
    if (iszero(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    if (ax <= f128_s{ 0.5 })
        return F128_CANONICALIZE_MATH_RESULT(log1p_series_reduced(x));

    const f128_s u = add_inline(f128_s{ 1.0 }, x);
    if (sub_inline(u, f128_s{ 1.0 }) == x)
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128_impl::log(u));

    if (x > f128_s{ 0.0 } && x <= f128_s{ 1.0 })
    {
        const f128_s t = div_inline(x, add_inline(f128_s{ 1.0 }, detail::_f128_impl::sqrt(add_inline(f128_s{ 1.0 }, x))));
        return F128_CANONICALIZE_MATH_RESULT(mul_inline(log1p_series_reduced(t), f128_s{ 2.0 }));
    }

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128_impl::log(u));

    const f128_s y = sub_inline(u, f128_s{ 1.0 });
    if (iszero(y))
        return x;

    return F128_CANONICALIZE_MATH_RESULT(mul_inline(detail::_f128_impl::log(u), div_inline(x, y)));
}


// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::cbrt(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y{};
    if (bl::use_constexpr_math())
    {
        y = detail::_f128_impl::exp(div_inline(detail::_f128_impl::log(ax), f128_s{ 3.0 }));
    }
    else
    {
        int exp2 = 0;
        double mantissa = std::frexp(ax.hi, &exp2);
        int rem  = exp2 % 3;
        if (rem < 0)
            rem += 3;
        if (rem != 0)
        {
            mantissa = std::ldexp(mantissa, rem);
            exp2 -= rem;
        }

        y = f128_s{ std::cbrt(mantissa), 0.0 };
        if (exp2 != 0)
            y = _ldexp(y, exp2 / 3);
    }

    y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });
    y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });

    if (bl::use_constexpr_math())
        y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });

    if (neg)
        y = -y;

    return F128_CANONICALIZE_MATH_RESULT(y);
}

// power functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::pow10_128(int k)
{
    if (k == 0) [[unlikely]]
        return f128_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    // fast small-exponent path
    if (n <= 16) {
        f128_s r = f128_s{ 1.0 };
        const f128_s ten = f128_s{ 10.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f128_s{ 1.0 } / r);
    }

    f128_s r = f128_s{ 1.0 };
    f128_s base = f128_s{ 10.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f128_s{ 1.0 } / r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s yi = detail::_f128_impl::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && try_get_int64(yi, yi64))
        return powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return F128_CANONICALIZE_MATH_RESULT(_exp(mul_inline(y, _log(x))));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::pow(const f128_s& x, double y)
{
    if (y == 0.0)
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (y == 1.0)  return x;
    if (y == 2.0)  return F128_CANONICALIZE_MATH_RESULT(x * x);
    if (y == -1.0) return F128_CANONICALIZE_MATH_RESULT(f128_s{ 1.0 } / x);
    if (y == 0.5)  return F128_CANONICALIZE_MATH_RESULT(detail::_f128_impl::sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? detail::fp::ceil(y)
            : detail::fp::floor(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && absd(yi) < 0x1p63)
        return powi(x, static_cast<int64_t>(yi));

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(_log(-x) * y);
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return F128_CANONICALIZE_MATH_RESULT(y_is_odd ? -magnitude : magnitude);
    }

    return F128_CANONICALIZE_MATH_RESULT(_exp(_log(x) * y));
}

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

// sinh/cosh/tanh
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::sinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    if (ax <= f128_s{ 0.5 })
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s term = x;
        f128_s sum  = x;

        for (int n = 1; n <= 40; ++n)
        {
            const double denom = static_cast<double>((n * 2) * (n * 2 + 1));
            term = div_inline(mul_inline(term, x2), f128_s{ denom });
            sum = add_inline(sum, term);

            const f128_s abs_sum = detail::_f128::mag(sum);
            const f128_s scale = (abs_sum < f128_s{ 1.0 }) ? f128_s{ 1.0 } : abs_sum;
            if (detail::_f128::mag(term) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return F128_CANONICALIZE_MATH_RESULT(sum);
    }

    const f128_s ex = detail::_f128_impl::exp(ax);
    f128_s out = mul_inline(sub_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 });
    if (signbit(x))
        out = -out;
    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::cosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = detail::_f128::mag(x);
    const f128_s ex = detail::_f128_impl::exp(ax);
    return F128_CANONICALIZE_MATH_RESULT(mul_inline(add_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 }));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::tanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = detail::_f128::mag(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = detail::_f128_impl::expm1(add_inline(ax, ax));
    f128_s out = div_inline(em1, add_inline(em1, f128_s{ 2.0 }));
    if (signbit(x))
        out = -out;
    return F128_CANONICALIZE_MATH_RESULT(out);
}

// inverse hyperbolic functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::asinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_impl::log(ax), std::numbers::ln2_v<f128_s>);
    else
        out = detail::_f128_impl::log(add_inline(ax, detail::_f128_impl::sqrt(add_inline(mul_inline(ax, ax), f128_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::acosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x < f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (x == f128_s{ 1.0 })
        return f128_s{ 0.0 };
    if (isinf(x))
        return x;

    f128_s out{};
    if (x > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_impl::log(x), std::numbers::ln2_v<f128_s>);
    else
        out = detail::_f128_impl::log(add_inline(x, detail::_f128_impl::sqrt(mul_inline(sub_inline(x, f128_s{ 1.0 }), add_inline(x, f128_s{ 1.0 })))));

    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::atanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    if (ax > f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (ax == f128_s{ 1.0 })
        return signbit(x)
            ? f128_s{ -std::numeric_limits<double>::infinity(), 0.0 }
            : f128_s{  std::numeric_limits<double>::infinity(), 0.0 };

    if (ax <= f128_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
        {
            return F128_CANONICALIZE_MATH_RESULT(atanh_small_series(x));
        }

        return F128_CANONICALIZE_MATH_RESULT(atanh_small_series_runtime(x));
    }

    const f128_s out = mul_inline(detail::_f128_impl::log(div_inline(add_inline(f128_s{ 1.0 }, x), sub_inline(f128_s{ 1.0 }, x))), f128_s{ 0.5 });
    return F128_CANONICALIZE_MATH_RESULT(out);
}

// erf/erfc functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::erf(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s out = ax < f128_s{ 2.0 }
        ? erf_positive_series(ax)
        : (ax > f128_s{ 27.0 } ? f128_s{ 1.0 } : sub_inline(f128_s{ 1.0 }, erfc_positive_cf(ax)));

    if (neg)
        out = -out;

    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::erfc(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return f128_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f128_s{ 2.0 } : f128_s{ 0.0 };

    if (signbit(x))
    {
        const f128_s ax = -x;
        if (ax < f128_s{ 2.0 })
            return F128_CANONICALIZE_MATH_RESULT(add_inline(f128_s{ 1.0 }, erf_positive_series(ax)));
        if (ax > f128_s{ 27.0 })
            return f128_s{ 2.0 };
        return F128_CANONICALIZE_MATH_RESULT(sub_inline(f128_s{ 2.0 }, erfc_positive_cf(ax)));
    }

    // use the existing high-quality erf series throughout the region where it is stable
    if (x < f128_s{ 2.0 })
        return F128_CANONICALIZE_MATH_RESULT(sub_inline(f128_s{ 1.0 }, erf_positive_series(x)));

    if (x > f128_s{ 27.0 })
        return f128_s{ 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(erfc_positive_cf(x));
}

// gamma functions
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::lgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(lgamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = detail::_f128_impl::sin(mul_inline(std::numbers::pi_v<f128_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        sub_inline(
            sub_inline(detail::_f128_impl::log(std::numbers::pi_v<f128_s>), detail::_f128_impl::log(detail::_f128::mag(sinpix))),
            lgamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x)));

    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::tgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(gamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = detail::_f128_impl::sin(mul_inline(std::numbers::pi_v<f128_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = div_inline(std::numbers::pi_v<f128_s>, mul_inline(sinpix, gamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x))));
    return F128_CANONICALIZE_MATH_RESULT(out);
}

} // namespace bl

#endif // F128_DETAIL_MATH_TRANSCENDENTAL_INCLUDED
