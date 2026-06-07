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
#include "fltx/f128_numbers.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::log_as_double(f128_s a);

namespace detail::_f128 // primitives and kernels
{
    // expm1/log1p functions
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

    BL_MSVC_NOINLINE constexpr f128_s log_normalized_series(const f128_s& m)
    {
        constexpr f128_s inv_odd[] = {
            { 0x1.5555555555555p-2, 0x1.5555555555555p-56 }, // 1/3
            { 0x1.999999999999ap-3, -0x1.999999999999ap-57 }, // 1/5
            { 0x1.2492492492492p-3, 0x1.2492492492492p-57 }, // 1/7
            { 0x1.c71c71c71c71cp-4, 0x1.c71c71c71c71cp-58 }, // 1/9
            { 0x1.745d1745d1746p-4, -0x1.745d1745d1746p-59 }, // 1/11
            { 0x1.3b13b13b13b14p-4, -0x1.3b13b13b13b14p-58 }, // 1/13
            { 0x1.1111111111111p-4, 0x1.1111111111111p-60 }, // 1/15
            { 0x1.e1e1e1e1e1e1ep-5, 0x1.e1e1e1e1e1e1ep-61 }, // 1/17
            { 0x1.af286bca1af28p-5, 0x1.af286bca1af28p-59 }, // 1/19
            { 0x1.8618618618618p-5, 0x1.8618618618618p-59 }, // 1/21
            { 0x1.642c8590b2164p-5, 0x1.642c8590b2164p-60 }, // 1/23
            { 0x1.47ae147ae147bp-5, -0x1.eb851eb851eb8p-61 }, // 1/25
            { 0x1.2f684bda12f68p-5, 0x1.2f684bda12f68p-59 }, // 1/27
            { 0x1.1a7b9611a7b96p-5, 0x1.1a7b9611a7b96p-61 }, // 1/29
            { 0x1.0842108421084p-5, 0x1.0842108421084p-60 }, // 1/31
            { 0x1.f07c1f07c1f08p-6, -0x1.f07c1f07c1f08p-61 }, // 1/33
            { 0x1.d41d41d41d41dp-6, 0x1.0750750750750p-60 }, // 1/35
            { 0x1.bacf914c1bad0p-6, -0x1.bacf914c1bad0p-60 }, // 1/37
            { 0x1.a41a41a41a41ap-6, 0x1.0690690690690p-60 }, // 1/39
            { 0x1.8f9c18f9c18fap-6, -0x1.f3831f3831f38p-61 }, // 1/41
            { 0x1.7d05f417d05f4p-6, 0x1.7d05f417d05f4p-62 }, // 1/43
        };

        const f128_s z = div_inline(sub_double_inline(m, 1.0), add_double_inline(m, 1.0));
        const f128_s z2 = mul_inline(z, z);

        f128_s p = inv_odd[sizeof(inv_odd) / sizeof(inv_odd[0]) - 1];
        for (int i = static_cast<int>(sizeof(inv_odd) / sizeof(inv_odd[0])) - 2; i >= 0; --i)
            p = mul_add_inline(p, z2, inv_odd[i]);

        p = mul_add_inline(p, z2, f128_s{ 1.0 });
        return mul_double_inline(mul_inline(z, p), 2.0);
    }

    // exponential functions
    BL_MSVC_NOINLINE constexpr f128_s expm1_tiny(const f128_s& r)
    {
        constexpr int coeff_count = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0]));

        f128_s p = mul_inline(r, r);
        f128_s sum = add_inline(r, mul_pwr2_inline(p, 0.5));
        const double threshold = absd(r.hi) * f128_s::eps().hi;

        p = mul_inline(p, r);
        for (int i = 0; i < coeff_count; ++i)
        {
            const f128_s term = mul_inline(p, exp_inv_fact[i]);
            sum = add_inline(sum, term);
            if (absd(term.hi) <= threshold)
                break;
            p = mul_inline(p, r);
        }

        return sum;
    }

    BL_FORCE_INLINE constexpr f128_s exp_integer_factor(int n) noexcept
    {
        if (n == 0)
            return f128_s{ 1.0 };

        const bool negative = n < 0;
        std::uint32_t exponent = static_cast<std::uint32_t>(negative ? -n : n);
        const f128_s* table = negative ? exp_integer_inv_table : exp_integer_table;
        f128_s factor{ 1.0 };

        for (std::size_t i = 0; exponent != 0 && i < (sizeof(exp_integer_table) / sizeof(exp_integer_table[0])); ++i)
        {
            if ((exponent & 1u) != 0)
                factor = mul_inline(factor, table[i]);
            exponent >>= 1u;
        }

        return factor;
    }

    BL_FORCE_INLINE constexpr double exp_nearest_integer(double x) noexcept
    {
        if (bl::use_constexpr_math())
            return nearbyint_ties_even(x);

        return static_cast<double>(
            x >= 0.0
                ? static_cast<int>(x + 0.5)
                : static_cast<int>(x - 0.5));
    }

    BL_MSVC_NOINLINE constexpr f128_s exp_general_scaled(const f128_s& x, bool sub_one) noexcept
    {
        const double nd = exp_nearest_integer(x.hi);
        const int n = static_cast<int>(nd);
        const f128_s reduced = sub_inline(x, f128_s{ nd });
        const f128_s r = mul_pwr2_inline(reduced, 0.0078125);

        f128_s e = expm1_tiny(r);
        for (int i = 0; i < 7; ++i)
            e = mul_add_inline(e, e, mul_pwr2_inline(e, 2.0));

        if (n == 0)
            return sub_one ? e : add_inline(e, f128_s{ 1.0 });

        const f128_s factor = exp_integer_factor(n);
        const f128_s scaled = mul_add_inline(factor, e, factor);
        return sub_one ? sub_inline(scaled, f128_s{ 1.0 }) : scaled;
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

        return exp_general_scaled(x, false);
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

        const double kd = exp_nearest_integer(x.hi);
        const int k = static_cast<int>(kd);
        const f128_s reduced = sub_inline(x, f128_s{ kd });
        const f128_s r = mul_pwr2_inline(mul_dd_inline(reduced, std::numbers::ln2_v<f128_s>), 0.0078125);

        f128_s e = expm1_tiny(r);
        for (int i = 0; i < 7; ++i)
            e = mul_add_inline(e, e, mul_pwr2_inline(e, 2.0));

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
        return add_inline(exp2_ln2, log_normalized_series(m));
    }

    BL_FORCE_INLINE constexpr bool f128_try_exact_binary_log2(const f128_s& x, int& out) noexcept
    {
        if (!(x.hi > 0.0) || x.lo != 0.0)
            return false;

        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x.hi);
        const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
        const std::uint64_t frac_bits = bits & ((std::uint64_t{ 1 } << 52) - 1);

        if (exp_bits == 0 || exp_bits == 0x7ffu || frac_bits != 0)
            return false;

        out = static_cast<int>(exp_bits) - 1023;
        return true;
    }

    // power functions
    BL_FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

    BL_FORCE_INLINE constexpr f128_s polish_eighth_root(const f128_s& x, const f128_s& y)
    {
        if (iszero(y))
            return y;

        const f128_s y2 = mul_inline(y, y);
        const f128_s y4 = mul_inline(y2, y2);
        const f128_s y7 = mul_inline(mul_inline(y4, y2), y);
        const f128_s y8 = mul_inline(y4, y4);
        const f128_s correction = div_double_inline(div_inline(sub_inline(x, y8), y7), 8.0);

        return add_inline(y, correction);
    }

    BL_FORCE_INLINE constexpr f128_s pow_positive_eighth_fraction(const f128_s& x, int numerator)
    {
        const f128_s r2 = detail::_f128_impl::sqrt(x);
        if (numerator == 4)
            return r2;

        const f128_s r4 = detail::_f128_impl::sqrt(r2);
        if (numerator == 2)
            return r4;

        f128_s out{ 1.0 };
        if ((numerator & 4) != 0)
            out = mul_inline(out, r2);
        if ((numerator & 2) != 0)
            out = mul_inline(out, r4);
        if ((numerator & 1) != 0)
        {
            const f128_s r8 = polish_eighth_root(x, detail::_f128_impl::sqrt(r4));
            if (numerator == 1)
                return r8;
            out = mul_inline(out, r8);
        }
        return out;
    }

    BL_FORCE_INLINE constexpr bool pow_dyadic_eighth_exponent_in_range(int64_t n) noexcept
    {
        if (n == std::numeric_limits<int64_t>::min())
            return false;

        const bool neg = n < 0;
        const std::uint64_t magnitude = neg ? static_cast<std::uint64_t>(-n) : static_cast<std::uint64_t>(n);
        return magnitude <= 1024;
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f128_s& x, const f128_s& y, int64_t& n)
    {
        if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
            return false;

        if (!try_get_int64(mul_double_inline(y, 8.0), n))
            return false;

        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f128_s& x, double y, int64_t& n) noexcept
    {
        if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
            return false;

        const double scaled = y * 8.0;
        if (detail::fp::isinf_or_nan(scaled) || absd(scaled) >= 0x1p63)
            return false;

        const double rounded = detail::fp::trunc(scaled);
        if (rounded != scaled)
            return false;

        n = static_cast<int64_t>(rounded);
        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_NO_INLINE constexpr f128_s pow_dyadic_eighth_unchecked(const f128_s& x, int64_t n)
    {
        if (n == 0)
            return f128_s{ 1.0 };

        const bool neg = n < 0;
        const std::uint64_t magnitude = neg ? static_cast<std::uint64_t>(-n) : static_cast<std::uint64_t>(n);
        const std::uint64_t whole = magnitude / 8u;
        const int rem = static_cast<int>(magnitude & 7u);

        f128_s result = (whole == 0u) ? f128_s{ 1.0 } : powi(x, static_cast<int64_t>(whole));
        if (rem != 0)
            result = mul_inline(result, pow_positive_eighth_fraction(x, rem));
        if (neg)
            result = recip(result);

        return result;
    }

    BL_FORCE_INLINE constexpr f128_s pow_from_log_product(const f128_s& product)
    {
        if (detail::_f128::mag(product) <= f128_s{ 0.125 })
            return add_inline(f128_s{ 1.0 }, expm1_tiny(product));

        return _exp(product);
    }

    BL_FORCE_INLINE constexpr f128_s log_for_pow_positive(const f128_s& x)
    {
        const f128_s xm1 = sub_double_inline(x, 1.0);
        if (detail::_f128::mag(xm1) <= f128_s{ 0.5 })
            return log1p_series_reduced(xm1);

        return _log(x);
    }

    // sine/cosine functions
    BL_FORCE_INLINE constexpr bool remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
	{
	    const double ax = fabs(x.hi);
	    if (detail::fp::isinf_or_nan(ax))
	        return false;

	    if (ax > 7.0e15)
	        return false;

	    const f128_s t = mul_inline(x, invpi2);

	    double qd = nearbyint_ties_even(t.hi);
	    if (detail::fp::isinf_or_nan(qd) ||
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

    BL_FORCE_INLINE constexpr f128_s _acos(const f128_s& x)
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

        return sub_inline(pi_2, detail::_f128::_asin(x));
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

    // erf/erfc functions
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erf_cheb_eval(
        const f128_s& x,
        const f128_s* coeffs,
        std::size_t count,
        double shift)
    {
        const f128_s t = sub_inline(mul_double_inline(x, 2.0), f128_s{ shift });
        f128_s b1{ 0.0 };
        f128_s b2{ 0.0 };

        for (int i = static_cast<int>(count) - 1; i >= 1; --i)
        {
            const f128_s b0 = add_inline(
                sub_inline(mul_double_inline(mul_inline(t, b1), 2.0), b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return F128_CANONICALIZE_MATH_RESULT(add_inline(mul_sub_inline(t, b1, b2), coeffs[0]));
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erf_positive_cheb_1_2(const f128_s& x)
    {
        return erf_cheb_eval(x, f128_erf_cheb_1_2, f128_erf_cheb_1_2_coeff_count, 3.0);
    }

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
                const int exponent = detail::fp::frexp_exponent_limb(hi);
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

    BL_MSVC_NOINLINE constexpr f128_s sinpi_reduced(const f128_s& x) noexcept
    {
        const f128_s n = detail::_f128_impl::nearbyint(x);
        const f128_s r = sub_inline(x, n);
        f128_s out = detail::_f128_impl::sin(mul_inline(std::numbers::pi_v<f128_s>, r));
        if (is_odd_integer(n))
            out = -out;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s cbrt_constexpr_seed(const f128_s& ax)
    {
        return detail::_f128_impl::exp(div_inline(detail::_f128_impl::log(ax), f128_s{ 3.0 }));
    }
}

namespace detail::_f128_runtime
{
    [[nodiscard]] BL_FORCE_INLINE f128_s cbrt_seed(const f128_s& ax)
    {
        int exp2 = 0;
        double mantissa = std::frexp(ax.hi, &exp2);
        int rem = exp2 % 3;
        if (rem < 0)
            rem += 3;
        if (rem != 0)
        {
            mantissa = std::ldexp(mantissa, rem);
            exp2 -= rem;
        }

        f128_s y{ std::cbrt(mantissa), 0.0 };
        if (exp2 != 0)
            y = detail::_f128::_ldexp(y, exp2 / 3);
        return y;
    }
}

namespace detail::_f128
{
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s cbrt_seed(const f128_s& ax)
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            cbrt_constexpr_seed(ax),
            detail::_f128_runtime::cbrt_seed(ax)
        );
    }

    BL_PUSH_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s cbrt_compensated(const f128_s& ax, double c) noexcept
    {
        double c2_hi{}, c2_lo{};
        two_prod_precise(c, c, c2_hi, c2_lo);

        double c3_hi{}, c3_lo{};
        two_prod_precise(c2_hi, c, c3_hi, c3_lo);
        c3_lo += c2_lo * c;

        double residual_hi{}, residual_lo{};
        two_sum_precise(ax.hi, -c3_hi, residual_hi, residual_lo);
        residual_lo += ax.lo - c3_lo;

        const f128_s residual = renorm(residual_hi, residual_lo);
        const double inv_derivative = 1.0 / (3.0 * c2_hi);
        const double cc = residual.hi * inv_derivative + residual.lo * inv_derivative;

        const double y_hi = c + cc;
        return { y_hi, (c - y_hi) + cc };
    }
    BL_POP_PRECISE;

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
    if (isnan(a))
        return a;
    if (iszero(a))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (signbit(a))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(a))
        return a;

    int exact_exp2{};
    if (f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(mul_inline(_log(a), std::numbers::log2e_v<f128_s>));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::log10(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (signbit(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(x))
        return x;

    if (detail::fp::isfinite(x.hi) && x.hi > 0.0 && x.lo == 0.0)
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

    if (x.hi > 709.782712893384)
        return std::numeric_limits<f128_s>::infinity();

    if (x.hi < -745.133219101941)
        return f128_s{ -1.0, 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(exp_general_scaled(x, true));
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
    if (ax <= f128_s{ 0.25 })
        return F128_CANONICALIZE_MATH_RESULT(log1p_series_reduced(x));

    const f128_s u = add_inline(f128_s{ 1.0 }, x);
    return F128_CANONICALIZE_MATH_RESULT(_log(u));
}


// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::cbrt(const f128_s& x)
{
    using namespace detail::_f128;

    if (detail::fp::iszero_or_inf_or_nan(x.hi))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y = cbrt_compensated(ax, cbrt_seed(ax).hi);

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

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (x == f128_s{ 1.0 } || (x == f128_s{ -1.0 } && detail::fp::isinf(y.hi)))
        return f128_s{ 1.0 };

    if (detail::fp::isnan(x.hi) || detail::fp::isnan(y.hi))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (detail::fp::isinf(y.hi))
    {
        const f128_s ax = detail::_f128::mag(x);
        if (ax == f128_s{ 1.0 })
            return f128_s{ 1.0 };
        if (ax < f128_s{ 1.0 })
            return signbit(y) ? std::numeric_limits<f128_s>::infinity() : f128_s{ 0.0 };
        return signbit(y) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();
    }

    const f128_s yi = detail::_f128_impl::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && try_get_int64(yi, yi64))
        return powi(x, yi64);

    int64_t dyadic_exponent{};
    if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
        return F128_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = pow_from_log_product(mul_inline(y, log_for_pow_positive(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return F128_CANONICALIZE_MATH_RESULT(pow_from_log_product(mul_inline(y, log_for_pow_positive(x))));
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::pow(const f128_s& x, double y)
{
    if (y == 0.0)
        return f128_s{ 1.0 };

    if (x == f128_s{ 1.0 } || (x == f128_s{ -1.0 } && detail::fp::isinf(y)))
        return f128_s{ 1.0 };

    if (detail::fp::isnan(x.hi) || detail::fp::isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (detail::fp::isinf(y))
    {
        const f128_s ax = detail::_f128::mag(x);
        if (ax == f128_s{ 1.0 })
            return f128_s{ 1.0 };
        if (ax < f128_s{ 1.0 })
            return detail::fp::signbit(y) ? std::numeric_limits<f128_s>::infinity() : f128_s{ 0.0 };
        return detail::fp::signbit(y) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();
    }

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

    int64_t dyadic_exponent{};
    if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
        return F128_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = pow_from_log_product(log_for_pow_positive(-x) * y);
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return F128_CANONICALIZE_MATH_RESULT(y_is_odd ? -magnitude : magnitude);
    }

    return F128_CANONICALIZE_MATH_RESULT(pow_from_log_product(log_for_pow_positive(x) * y));
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
    return F128_CANONICALIZE_MATH_RESULT(detail::_f128::_acos(x));
}

// sine/cosine functions
[[nodiscard]] BL_MSVC_NOINLINE constexpr bool detail::_f128_impl::sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    if (iszero(x))
    {
        s_out = x;
        c_out = f128_s{ 1.0 };
        return true;
    }

    const double ax = detail::_f128::fabs(x.hi);
    if (detail::fp::isinf_or_nan(ax))
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
    if (iszero(x))
        return x;

    const double ax = detail::_f128::fabs(x.hi);
    if (detail::fp::isinf_or_nan(ax))
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
    if (detail::fp::isinf_or_nan(ax))
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
    if (iszero(x))
        return x;

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
    if (detail::fp::isnan(x.hi) || detail::fp::isnan(y.hi))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (isinf(y))
    {
        if (isinf(x))
        {
            if (x.hi < 0.0)
                return signbit(y) ? -detail::_f128::pi_3_4 : detail::_f128::pi_3_4;
            return signbit(y) ? -detail::_f128::pi_4 : detail::_f128::pi_4;
        }

        return signbit(y) ? -detail::_f128::pi_2 : detail::_f128::pi_2;
    }
    if (isinf(x))
    {
        if (x.hi < 0.0)
            return signbit(y) ? -std::numbers::pi_v<f128_s> : std::numbers::pi_v<f128_s>;
        return f128_s{ signbit(y) ? -0.0 : 0.0, 0.0 };
    }

    if (iszero(x))
    {
        if (iszero(y))
        {
            if (signbit(x))
                return signbit(y) ? -std::numbers::pi_v<f128_s> : std::numbers::pi_v<f128_s>;
            return y;
        }
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

    if (detail::fp::iszero_or_inf_or_nan(x.hi))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    if (ax <= f128_s{ 0.5 })
    {
        const f128_s e = detail::_f128_impl::expm1(x);
        return F128_CANONICALIZE_MATH_RESULT(
            mul_double_inline(div_inline(mul_inline(e, add_double_inline(e, 2.0)), add_double_inline(e, 1.0)), 0.5));
    }

    const f128_s ex = detail::_f128_impl::exp(ax);
    f128_s out = mul_double_inline(sub_inline(ex, div_inline(f128_s{ 1.0 }, ex)), 0.5);
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
    return F128_CANONICALIZE_MATH_RESULT(mul_double_inline(add_inline(ex, div_inline(f128_s{ 1.0 }, ex)), 0.5));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::tanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (detail::fp::iszero_or_nan(x.hi))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = detail::_f128::mag(x);
    if (ax > f128_s{ 40.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    if (ax >= f128_s{ 0.5 })
    {
        const f128_s e = detail::_f128_impl::exp(mul_double_inline(ax, -2.0));
        f128_s out = sub_inline(f128_s{ 1.0 }, div_inline(mul_double_inline(e, 2.0), add_double_inline(e, 1.0)));
        if (signbit(x))
            out = -out;
        return F128_CANONICALIZE_MATH_RESULT(out);
    }

    const f128_s e = detail::_f128_impl::expm1(x);
    const f128_s ep1 = add_double_inline(e, 1.0);
    return F128_CANONICALIZE_MATH_RESULT(
        div_inline(
            mul_inline(e, add_double_inline(e, 2.0)),
            add_double_inline(mul_inline(ep1, ep1), 1.0)));
}

// inverse hyperbolic functions
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::asinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (detail::fp::iszero_or_inf_or_nan(x.hi))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_impl::log(ax), std::numbers::ln2_v<f128_s>);
    else if (ax <= f128_s{ 0.5 })
    {
        const f128_s ax2 = mul_inline(ax, ax);
        out = detail::_f128_impl::log1p(add_inline(
            ax,
            div_inline(ax2, add_double_inline(detail::_f128_impl::sqrt(add_double_inline(ax2, 1.0)), 1.0))));
    }
    else
        out = detail::_f128_impl::log(add_inline(ax, detail::_f128_impl::sqrt(add_double_inline(mul_inline(ax, ax), 1.0))));

    if (signbit(x))
        out = -out;
    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::acosh(const f128_s& x)
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
    else if (x < f128_s{ 1.25 })
    {
        const f128_s xm1 = sub_double_inline(x, 1.0);
        out = detail::_f128_impl::log1p(add_inline(
            xm1,
            detail::_f128_impl::sqrt(mul_inline(xm1, add_double_inline(x, 1.0)))));
    }
    else
        out = detail::_f128_impl::log(add_inline(
            x,
            detail::_f128_impl::sqrt(mul_inline(sub_double_inline(x, 1.0), add_double_inline(x, 1.0)))));

    return F128_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::atanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (detail::fp::iszero_or_nan(x.hi))
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
        return F128_CANONICALIZE_MATH_RESULT(atanh_small_series(x));
    }

    if (ax < f128_s{ 0.25 })
    {
        const f128_s r = div_inline(mul_double_inline(x, 2.0), sub_double_inline(1.0, x));
        return F128_CANONICALIZE_MATH_RESULT(mul_double_inline(detail::_f128_impl::log1p(r), 0.5));
    }

    const f128_s out = mul_double_inline(
        detail::_f128_impl::log(div_inline(add_double_inline(x, 1.0), sub_double_inline(1.0, x))),
        0.5);
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

    f128_s out{};
    if (ax < f128_s{ 1.0 })
        out = erf_positive_series(ax);
    else if (ax < f128_s{ 2.0 })
        out = erf_positive_cheb_1_2(ax);
    else
        out = ax > f128_s{ 27.0 } ? f128_s{ 1.0 } : sub_inline(f128_s{ 1.0 }, erfc_positive_cf(ax));

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
        if (ax < f128_s{ 1.0 })
            return F128_CANONICALIZE_MATH_RESULT(add_inline(f128_s{ 1.0 }, erf_positive_series(ax)));
        if (ax < f128_s{ 2.0 })
            return F128_CANONICALIZE_MATH_RESULT(add_inline(f128_s{ 1.0 }, erf_positive_cheb_1_2(ax)));
        if (ax > f128_s{ 27.0 })
            return f128_s{ 2.0 };
        return F128_CANONICALIZE_MATH_RESULT(sub_inline(f128_s{ 2.0 }, erfc_positive_cf(ax)));
    }

    // Keep the short series near zero and switch to the fixed midrange approximation before it gets expensive.
    if (x < f128_s{ 1.0 })
        return F128_CANONICALIZE_MATH_RESULT(sub_inline(f128_s{ 1.0 }, erf_positive_series(x)));
    if (x < f128_s{ 2.0 })
        return F128_CANONICALIZE_MATH_RESULT(sub_inline(f128_s{ 1.0 }, erf_positive_cheb_1_2(x)));

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
        return std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(lgamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = sinpi_reduced(x);
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
    if (iszero(x))
        return f128_s{ signbit(x) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity(), 0.0 };

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(gamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = sinpi_reduced(x);
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = div_inline(std::numbers::pi_v<f128_s>, mul_inline(sinpix, gamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x))));
    return F128_CANONICALIZE_MATH_RESULT(out);
}

} // namespace bl

#endif // F128_DETAIL_MATH_TRANSCENDENTAL_INCLUDED
