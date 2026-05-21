/**
 * fltx/detail/f256/math/trig.h - trigonometry implementation details.
 *
 * f256 angle reduction, trig kernels, and inverse trig implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_TRIG_IMPL_INCLUDED
#define FLTX_F256_DETAIL_TRIG_IMPL_INCLUDED
#include "fltx/detail/f256/math_support.h"

namespace bl {

namespace detail::_f256
{
    BL_FORCE_INLINE constexpr biguint biguint_from_words(const std::uint32_t* words, int count)
    {
        biguint out{};
        const int copy_count = count < biguint::max_words ? count : biguint::max_words;
        for (int i = 0; i < copy_count; ++i)
            out.words[i] = words[i];
        out.size = copy_count;
        out.trim();
        return out;
    }

    BL_FORCE_INLINE constexpr biguint biguint_low_bits(const biguint& value, int bits)
    {
        biguint out{};
        if (bits <= 0 || value.is_zero())
            return out;

        const int full_words = bits >> 5;
        const int rem_bits   = bits & 31;
        const int copy_words = full_words < value.size ? full_words : value.size;

        for (int i = 0; i < copy_words; ++i)
            out.words[i] = value.words[i];

        out.size = copy_words;
        if (rem_bits != 0 && full_words < value.size && out.size < biguint::max_words)
        {
            out.words[out.size++] = value.words[full_words] & ((std::uint32_t{ 1 } << rem_bits) - 1u);
        }

        out.trim();
        return out;
    }

    BL_FORCE_INLINE constexpr biguint biguint_pow2(int bit)
    {
        biguint out{};
        out.set_bit(bit);
        return out;
    }

    BL_FORCE_INLINE constexpr bool remainder_pi2_payne_hanek(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        const bool neg = signbit_constexpr(x.x0);
        const exact_dyadic_fmod dx = exact_from_f256_fmod(abs(x));
        if (dx.mant.is_zero())
        {
            n_out = 0;
            r_out = neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };
            return true;
        }

        const biguint two_over_pi = biguint_from_words(
            two_over_pi_fixed_words,
            static_cast<int>(sizeof(two_over_pi_fixed_words) / sizeof(two_over_pi_fixed_words[0])));
        const biguint product = mul_big(dx.mant, two_over_pi);
        const int scale_bits = two_over_pi_fixed_bits - dx.exp2;
        if (scale_bits <= 0)
            return false;

        const biguint rem = biguint_low_bits(product, scale_bits);
        const bool half_bit = rem.get_bit(scale_bits - 1);
        const bool sticky = biguint_any_low_bits_set(rem, scale_bits - 1);
        const bool integer_odd = product.get_bit(scale_bits);
        const bool round_up = half_bit && (sticky || integer_odd);

        unsigned n_mod4 =
            (product.get_bit(scale_bits) ? 1u : 0u) |
            (product.get_bit(scale_bits + 1) ? 2u : 0u);
        if (round_up)
            n_mod4 = (n_mod4 + 1u) & 3u;

        biguint y_coeff = rem;
        bool y_neg = false;
        if (round_up)
        {
            y_coeff = biguint_pow2(scale_bits);
            y_coeff.sub_inplace(rem);
            y_neg = !y_coeff.is_zero();
        }

        f256_s r = exact_dyadic_to_f256_fmod(y_coeff, -scale_bits, y_neg);
        r = mul_inline(r, pi_2);

        if (r > pi_4)
        {
            r = sub_inline(r, pi_2);
            n_mod4 = (n_mod4 + 1u) & 3u;
        }
        else if (r < -pi_4)
        {
            r = add_inline(r, pi_2);
            n_mod4 = (n_mod4 + 3u) & 3u;
        }

        if (neg)
        {
            r = -r;
            n_mod4 = (4u - n_mod4) & 3u;
        }

        n_out = static_cast<long long>(n_mod4);
        r_out = r;
        return true;
    }

    BL_FORCE_INLINE constexpr bool remainder_pi2(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        if (!isfinite(x.x0))
            return false;

        if (abs(x) <= pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256_s q = detail::_f256_constexpr::nearbyint(mul_inline(x, invpi2));
        const double qd = q.x0;

        if (!detail::fp::isfinite(qd) || detail::fp::absd(qd) > 9.0e15)
        {
            return remainder_pi2_payne_hanek(x, n_out, r_out);
        }

        long long n = (long long)qd;
        f256_s r = x;
        r = sub_mul_double_inline(r, q, pi_2.x0);
        r = sub_mul_double_inline(r, q, pi_2.x1);
        r = sub_mul_double_inline(r, q, pi_2.x2);
        r = sub_mul_double_inline(r, q, pi_2.x3);

        if (r > pi_4)
        {
            r = sub_inline(r, pi_2);
            ++n;
        }
        else if (r < -pi_4)
        {
            r = add_inline(r, pi_2);
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }

    #if BL_F256_ENABLE_SIMD
    BL_FORCE_INLINE constexpr f256_s mul_from_two_prod_terms(
        double p0, double p1, double p2, double p3, double p4, double p5,
        double p6, double p7, double p8, double p9,
        double q0, double q1, double q2, double q3, double q4, double q5,
        double q6, double q7, double q8, double q9,
        double tail_mul0, double tail_mul1, double tail_mul2) noexcept
    {
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);
        t1 += (q3 + q5);

        two_sum_precise(p6, p8, r0, r1);
        r1 += (p7 + p9);

        two_sum_precise(t0, r0, q3, q4);
        q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;

        t1 += tail_mul0 + tail_mul1 + tail_mul2
            + q6 + q7 + q8 + q9 + s2;

        return renorm5(p0, p1, s0, t0, t1);
    }

    BL_FORCE_INLINE void mul_pair_simd(
        const f256_s& a0, const f256_s& b0,
        const f256_s& a1, const f256_s& b1,
        f256_s& out0, f256_s& out1) noexcept
    {
        double p00{}, p10{}, p20{}, p30{}, p40{}, p50{};
        double q00{}, q10{}, q20{}, q30{}, q40{}, q50{};

        double p01{}, p11{}, p21{}, p31{}, p41{}, p51{};
        double q01{}, q11{}, q21{}, q31{}, q41{}, q51{};

        two_prod_precise(a0.x0, b0.x0, p00, q00);
        two_prod_precise(a0.x0, b0.x1, p10, q10);
        two_prod_precise(a0.x1, b0.x0, p20, q20);
        two_prod_precise(a0.x0, b0.x2, p30, q30);
        two_prod_precise(a0.x1, b0.x1, p40, q40);
        two_prod_precise(a0.x2, b0.x0, p50, q50);

        two_prod_precise(a1.x0, b1.x0, p01, q01);
        two_prod_precise(a1.x0, b1.x1, p11, q11);
        two_prod_precise(a1.x1, b1.x0, p21, q21);
        two_prod_precise(a1.x0, b1.x2, p31, q31);
        two_prod_precise(a1.x1, b1.x1, p41, q41);
        two_prod_precise(a1.x2, b1.x0, p51, q51);

        const simd::f64x2 ax0 = simd::f64x2_set(a0.x0, a1.x0);
        const simd::f64x2 ax1 = simd::f64x2_set(a0.x1, a1.x1);
        const simd::f64x2 ax2 = simd::f64x2_set(a0.x2, a1.x2);
        const simd::f64x2 ax3 = simd::f64x2_set(a0.x3, a1.x3);

        const simd::f64x2 bx0 = simd::f64x2_set(b0.x0, b1.x0);
        const simd::f64x2 bx1 = simd::f64x2_set(b0.x1, b1.x1);
        const simd::f64x2 bx2 = simd::f64x2_set(b0.x2, b1.x2);
        const simd::f64x2 bx3 = simd::f64x2_set(b0.x3, b1.x3);

        simd::f64x2 p6{}, p7{}, p8{}, p9{};
        simd::f64x2 q6{}, q7{}, q8{}, q9{};

        simd::f64x2_two_prod_precise(ax0, bx3, p6, q6);
        simd::f64x2_two_prod_precise(ax1, bx2, p7, q7);
        simd::f64x2_two_prod_precise(ax2, bx1, p8, q8);
        simd::f64x2_two_prod_precise(ax3, bx0, p9, q9);

        alignas(16) double p6v[2], p7v[2], p8v[2], p9v[2];
        alignas(16) double q6v[2], q7v[2], q8v[2], q9v[2];

        simd::f64x2_store_array(p6, p6v);
        simd::f64x2_store_array(p7, p7v);
        simd::f64x2_store_array(p8, p8v);
        simd::f64x2_store_array(p9, p9v);
        simd::f64x2_store_array(q6, q6v);
        simd::f64x2_store_array(q7, q7v);
        simd::f64x2_store_array(q8, q8v);
        simd::f64x2_store_array(q9, q9v);

        out0 = mul_from_two_prod_terms(
            p00, p10, p20, p30, p40, p50,
            p6v[0], p7v[0], p8v[0], p9v[0],
            q00, q10, q20, q30, q40, q50,
            q6v[0], q7v[0], q8v[0], q9v[0],
            a0.x1 * b0.x3, a0.x2 * b0.x2, a0.x3 * b0.x1
        );

        out1 = mul_from_two_prod_terms(
            p01, p11, p21, p31, p41, p51,
            p6v[1], p7v[1], p8v[1], p9v[1],
            q01, q11, q21, q31, q41, q51,
            q6v[1], q7v[1], q8v[1], q9v[1],
            a1.x1 * b1.x3, a1.x2 * b1.x2, a1.x3 * b1.x1
        );
    }
    #endif
    BL_MSVC_NOINLINE constexpr f256_s sin_kernel_pi4_constexpr(const f256_s& r)
    {
        const f256_s t = sqr_inline(r);

        const f256_s ps = horner_forward_constexpr(f256_sin_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        return mul_add_inline(mul_inline(r, t), ps, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s cos_kernel_pi4_constexpr(const f256_s& r)
    {
        const f256_s t = sqr_inline(r);

        const f256_s pc = horner_forward_constexpr(f256_cos_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        return mul_add_inline(t, pc, f256_s{ 1.0 });
    }

    BL_MSVC_NOINLINE constexpr void sincos_kernel_pi4_constexpr(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        const f256_s t = sqr_inline(r);

        f256_s ps{};
        f256_s pc{};
        horner_pair_forward_constexpr(f256_sin_coeffs_pi4, f256_cos_coeffs_pi4, f256_trig_coeff_count_pi4, t, ps, pc);

        s_out = mul_add_inline(mul_inline(r, t), ps, r);
        c_out = mul_add_inline(t, pc, f256_s{ 1.0 });
    }

    BL_MSVC_NOINLINE constexpr void sincos_kernel_small(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        const f256_s t = sqr_inline(r);

        f256_s ps{};
        f256_s pc{};
        horner_pair_forward(
            f256_sin_coeffs_pi4 + f256_trig_small_coeff_offset,
            f256_cos_coeffs_pi4 + f256_trig_small_coeff_offset,
            f256_trig_small_coeff_count,
            t,
            ps,
            pc);

        const f256_s rt = mul_inline(r, t);
        s_out = mul_add_horner_step(rt, ps, r);
        c_out = mul_add_horner_step(t, pc, f256_s{ 1.0 });
    }

    BL_MSVC_NOINLINE constexpr void sincos_kernel_pi64_reduced(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        int k = static_cast<int>(nearbyint_ties_even(r.x0 * 20.371832715762604));
        if (k < -16)
            k = -16;
        else if (k > 16)
            k = 16;

        if (k == 0)
        {
            sincos_kernel_small(r, s_out, c_out);
            return;
        }

        const f256_s a = mul_double_inline(std::numbers::pi_v<f256_s>, static_cast<double>(k) * 0.015625);
        const f256_s u = sub_inline(r, a);

        f256_s su{};
        f256_s cu{};
        sincos_kernel_small(u, su, cu);

        const int table_index = k < 0 ? -k : k;
        const f256_s sa = k < 0 ? -f256_sin_table_pi64[table_index] : f256_sin_table_pi64[table_index];
        const f256_s ca = f256_cos_table_pi64[table_index];

        s_out = mul_add_inline(ca, su, mul_inline(sa, cu));
        c_out = mul_sub_inline(ca, cu, mul_inline(sa, su));
    }

    BL_MSVC_NOINLINE constexpr f256_s sin_kernel_pi4(const f256_s& r)
    {
        if (bl::use_constexpr_math())
            return sin_kernel_pi4_constexpr(r);

        const f256_s t = sqr_inline(r);
        const f256_s ps = horner_forward(f256_sin_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        const f256_s rt = mul_inline(r, t);
        return mul_add_horner_step(rt, ps, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s cos_kernel_pi4(const f256_s& r)
    {
        if (bl::use_constexpr_math())
            return cos_kernel_pi4_constexpr(r);

        const f256_s t = sqr_inline(r);
        const f256_s pc = horner_forward(f256_cos_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        return mul_add_horner_step(t, pc, f256_s{ 1.0 });
    }

    BL_MSVC_NOINLINE constexpr void sincos_kernel_pi4(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        sincos_kernel_pi64_reduced(r, s_out, c_out);
    }

    BL_MSVC_NOINLINE constexpr bool _sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
    {
        const double ax = fabs_constexpr(x.x0);
        if (!isfinite(ax))
        {
            s_out = f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
            c_out = s_out;
            return false;
        }

        if (ax <= static_cast<double>(pi_4))
        {
            sincos_kernel_pi4(x, s_out, c_out);
            return true;
        }

        long long n = 0;
        f256_s r{};
        if (!remainder_pi2(x, n, r))
        {
            s_out = f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
            c_out = s_out;
            return false;
        }

        f256_s sr{}, cr{};
        sincos_kernel_pi4(r, sr, cr);

        switch ((int)(n & 3LL))
        {
        case 0: s_out = sr;  c_out = cr;  break;
        case 1: s_out = cr;  c_out = -sr; break;
        case 2: s_out = -sr; c_out = -cr; break;
        default: s_out = -cr; c_out = sr;  break;
        }

        return true;
    }

    BL_MSVC_NOINLINE constexpr f256_s atan_series_reduced(const f256_s& z)
    {
        const f256_s z2 = sqr_inline(z);
        const f256_s p = horner_reverse(f256_atan_reduced_coeffs, f256_atan_reduced_coeff_count, z2);
        return mul_inline(z, p);
    }

    BL_MSVC_NOINLINE constexpr f256_s atan_core_unit(const f256_s& z)
    {
        using namespace detail::_f256;

        int k = static_cast<int>(nearbyint_ties_even(z.x0 * 16.0));
        if (k <= 0)
            return atan_series_reduced(z);
        if (k > 16)
            k = 16;

        const double a = static_cast<double>(k) * 0.0625;
        const f256_s u = div_inline(
            sub_double_inline(z, a),
            add_scalar_precise(mul_double_inline(z, a), 1.0));

        return add_inline(f256_atan_reduced_table_16[k], atan_series_reduced(u));
    }

    BL_MSVC_NOINLINE constexpr f256_s _atan(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))  return x;
        if (iszero(x)) return x;
        if (isinf(x))  return signbit_constexpr(x.x0) ? -pi_2 : pi_2;

        const bool neg = x.x0 < 0.0;
        const f256_s ax = neg ? -x : x;

        if (ax > f256_s{ 1.0 })
        {
            const f256_s core = atan_core_unit(recip(ax));
            const f256_s out  = sub_inline(pi_2, core);
            return neg ? -out : out;
        }

        const f256_s out = atan_core_unit(ax);
        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr f256_s _asin(const f256_s& x)
    {
        using namespace detail::_f256;

        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (ax == f256_s{ 1.0 })
            return (x.x0 < 0.0) ? -pi_2 : pi_2;

        if (ax <= f256_s{ 0.5 })
            return _atan(div_inline(x, detail::_f256_constexpr::sqrt(add_raw5_value_inline(neg_raw5(sqr_raw5_inline(x)), f256_s{ 1.0 }))));

        const f256_s t = detail::_f256_constexpr::sqrt(div_inline(sub_double_inline(1.0, ax), add_double_inline(ax, 1.0)));
        const f256_s a = sub_mul_double_inline(pi_2, _atan(t), 2.0);
        return (x.x0 < 0.0) ? -a : a;
    }

    BL_FORCE_INLINE constexpr f256_s _acos(const f256_s& x)
    {
        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (x == f256_s{ 1.0 })
            return f256_s{ 0.0 };
        if (x == f256_s{ -1.0 })
            return std::numbers::pi_v<f256_s>;

        return pi_2 - _asin(x);
    }

} // namespace detail::_f256

[[nodiscard]] BL_FORCE_INLINE constexpr bool   detail::_f256_constexpr::sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    bool ret = _sincos(x, s_out, c_out);
    s_out = canonicalize_math_result(s_out);
    c_out = canonicalize_math_result(c_out);
    return ret;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::sin(const f256_s& x)
{
    const double ax = fabs_constexpr(x.x0);
    if (!isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(pi_4))
        return sin_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!remainder_pi2(x, n, r))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    switch ((int)(n & 3LL))
    {
    case 0: return sin_kernel_pi4(r);
    case 1: return cos_kernel_pi4(r);
    case 2: return -sin_kernel_pi4(r);
    default: return -cos_kernel_pi4(r);
    }
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::cos(const f256_s& x)
{
    const double ax = fabs_constexpr(x.x0);
    if (!isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(pi_4))
        return cos_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!remainder_pi2(x, n, r))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    switch ((int)(n & 3LL))
    {
    case 0: return cos_kernel_pi4(r);
    case 1: return -sin_kernel_pi4(r);
    case 2: return -cos_kernel_pi4(r);
    default: return sin_kernel_pi4(r);
    }
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::tan(const f256_s& x)
{
    f256_s s{}, c{};
    if (_sincos(x, s, c))
        return canonicalize_math_result(s / c);

    return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::atan(const f256_s& x)
{
    return canonicalize_math_result(_atan(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::atan2(const f256_s& y, const f256_s& x)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (iszero(x))
    {
        if (iszero(y))
            return f256_s{ std::numeric_limits<double>::quiet_NaN() };
        return ispositive(y) ? pi_2 : -pi_2;
    }

    if (iszero(y))
    {
        if (x.x0 < 0.0)
            return signbit_constexpr(y.x0) ? -std::numbers::pi_v<f256_s> : std::numbers::pi_v<f256_s>;
        return y;
    }

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax == ay)
    {
        if (x.x0 < 0.0)
        {
            return canonicalize_math_result(
                (y.x0 < 0.0) ? -pi_3_4 : pi_3_4);
        }

        return canonicalize_math_result(
            (y.x0 < 0.0) ? -pi_4 : pi_4);
    }

    if (ax >= ay)
    {
        f256_s a = _atan(y / x);

        if (x.x0 < 0.0)
            a += (y.x0 < 0.0) ? -std::numbers::pi_v<f256_s> : std::numbers::pi_v<f256_s>;
        return canonicalize_math_result(a);
    }

    f256_s a = _atan(x / y);
    return canonicalize_math_result((y.x0 < 0.0) ? (-pi_2 - a) : (pi_2 - a));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::asin(const f256_s& x)
{
    return canonicalize_math_result(_asin(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::acos(const f256_s& x)
{
    return detail::_f256::canonicalize_math_result(detail::_f256::_acos(x));
}

} // namespace bl

#endif
