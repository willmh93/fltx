/**
 * fltx/detail/f256_transcendental.h - f256 transcendental math implementation details.
 *
 * f256 cbrt, exp/log, pow, trig, hyperbolic, erf, and gamma implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_TRANSCENDENTAL_INCLUDED
#define FLTX_F256_DETAIL_TRANSCENDENTAL_INCLUDED
#include "fltx/detail/f256_math_basic.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f256_impl::log_as_double(f256_s a) noexcept;

namespace detail::_f256 // primitives and kernels
{
    BL_FORCE_INLINE constexpr f256_s mul_add_horner_step_inline(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return mul_add_inline(a, b, c);
    }

    BL_FORCE_INLINE constexpr f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return mul_add_horner_step_inline(a, b, c);
        }

        return detail::_f256_runtime::mul_add_horner_step(a, b, c);
    }

    BL_FORCE_INLINE constexpr f256_s horner_forward_inline(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_FORCE_INLINE constexpr f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_forward_inline(coeffs, count, x);
        }

        return detail::_f256_runtime::horner_forward(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr f256_s horner_reverse_inline(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_FORCE_INLINE constexpr f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (bl::use_constexpr_math())
        {
            return horner_reverse_inline(coeffs, count, x);
        }

        return detail::_f256_runtime::horner_reverse(coeffs, count, x);
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward_inline(
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
            left = mul_add_inline(left, x, left_coeffs[i]);
            right = mul_add_inline(right, x, right_coeffs[i]);
        }

        left_out = left;
        right_out = right;
    }

    BL_FORCE_INLINE constexpr void horner_pair_forward(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (bl::use_constexpr_math())
        {
            horner_pair_forward_inline(left_coeffs, right_coeffs, count, x, left_out, right_out);
            return;
        }

        detail::_f256_runtime::horner_pair_forward(left_coeffs, right_coeffs, count, x, left_out, right_out);
    }

    // expm1/log1p functions
    BL_FORCE_INLINE constexpr f256_s log1p_double_seed_residual(const f256_s& r) noexcept
    {
        const f256_s r2 = sqr_inline(r);
        const f256_s r3 = mul_inline(r2, r);
        const f256_s r4 = sqr_inline(r2);
        const f256_s r5 = mul_inline(r4, r);

        f256_s correction = r;
        correction = sub_mul_double_inline(correction, r2, 0.5);
        correction = add_inline(correction, div_double_inline(r3, 3.0));
        correction = sub_mul_double_inline(correction, r4, 0.25);
        correction = add_mul_double_inline(correction, r5, 0.2);
        return correction;
    }

    BL_MSVC_NOINLINE constexpr f256_s expm1_tiny(const f256_s& r)
    {
        f256_s p = horner_reverse(exp_inv_fact, 15, r);
        p = mul_add_horner_step(p, r, f256_s{ 0.5 });
        return mul_add_horner_step(sqr_inline(r), p, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s expm1_reduced(const f256_s& x)
    {
        const f256_s t = mul_inline(x, std::numbers::log2e_v<f256_s>);

        double kd = nearbyint_ties_even(t.x0);
        const f256_s delta = sub_double_inline(t, kd);
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s kd_ln2 = mul_double_inline(std::numbers::ln2_v<f256_s>, kd);
        const f256_s r = mul_double_inline(sub_inline(x, kd_ln2), 0.0009765625);

        f256_s e = expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = mul_add_inline(e, e, mul_double_inline(e, 2.0));

        if (k == 0)
            return e;

        return add_scalar_precise(_ldexp(add_scalar_precise(e, 1.0), k), -1.0);
    }

    BL_MSVC_NOINLINE constexpr f256_s log1p_series_reduced(const f256_s& x)
    {
        if (!bl::use_constexpr_math())
        {
            return detail::_f256_runtime::log1p_series_reduced(x);
        }

        const f256_s z = div_add_double_inline(x, x, 2.0);
        const f256_s z2 = sqr_inline(z);

        f256_s term = z;
        f256_s sum  = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term = mul_inline(term, z2);
            const f256_s add = div_double_inline(term, static_cast<double>(k));
            sum = add_inline(sum, add);

            const f256_s asum  = mag(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (mag(add) <= mul_inline(f256_s::eps(), scale))
                break;
        }

        return add_inline(sum, sum);
    }

    // exponential functions
    BL_FORCE_INLINE constexpr f256_s expm1_tiny_fast_13(const f256_s& r) noexcept
    {
        f256_s p = exp_inv_fact[12];
        for (std::size_t i = 12; i > 0; --i)
            p = mul_add_inline(p, r, exp_inv_fact[i - 1]);

        p = mul_add_inline(p, r, f256_s{ 0.5 });
        return mul_add_inline(sqr_inline(r), p, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s exp_from_reduced_64(const f256_s& x, bool base2) noexcept
    {
        const f256_s t = base2 ? x : mul_inline(x, std::numbers::log2e_v<f256_s>);
        const int m = static_cast<int>(nearbyint_ties_even(t.x0 * 64.0));
        int n = m / 64;
        int j = m - n * 64;
        if (j < 0)
        {
            j += 64;
            --n;
        }

        const f256_s reduced = base2
            ? sub_double_inline(x, static_cast<double>(n) + static_cast<double>(j) / 64.0)
            : sub_inline(x, mul_double_inline(std::numbers::ln2_v<f256_s>, static_cast<double>(n) + static_cast<double>(j) / 64.0));

        const f256_s r = mul_double_inline(base2 ? mul_inline(reduced, std::numbers::ln2_v<f256_s>) : reduced, 0.125);
        f256_s e = expm1_tiny_fast_13(r);
        e = mul_add_inline(e, e, mul_double_inline(e, 2.0));
        e = mul_add_inline(e, e, mul_double_inline(e, 2.0));
        e = mul_add_inline(e, e, mul_double_inline(e, 2.0));

        return _ldexp(mul_inline(exp2_table_64[j], add_scalar_precise(e, 1.0)), n);
    }

    // logarithm functions
    BL_MSVC_NOINLINE constexpr f256_s log_with_fast_exp_correction(const f256_s& a) noexcept
    {
        if (isnan(a))
            return a;
        if (iszero(a))
            return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
        if (a.x0 < 0.0 || (a.x0 == 0.0 && (a.x1 < 0.0 || (a.x1 == 0.0 && (a.x2 < 0.0 || (a.x2 == 0.0 && a.x3 < 0.0))))))
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (isinf(a))
            return a;

        int exp2 = 0;
        if (bl::use_constexpr_math())
        {
            exp2 = frexp_exponent(a.x0);
        }
        else
        {
            (void)std::frexp(a.x0, &exp2);
        }

        f256_s m = _ldexp(a, -exp2);
        if (m < sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }
        if (m < f256_s{ 1.0 })
        {
            m *= 2.0;
            --exp2;
        }

        double log2_m{};
        if (bl::use_constexpr_math())
        {
            log2_m = detail::fp::log(m.x0) * 1.4426950408889634074;
        }
        else
        {
            log2_m = std::log2(m.x0);
        }

        int j = static_cast<int>(nearbyint_ties_even(log2_m * 64.0));
        if (j < 0)
            j = 0;
        else if (j > 64)
            j = 64;

        const f256_s c = (j == 64) ? f256_s{ 2.0 } : exp2_table_64[j];
        const f256_s u = div_inline(sub_inline(m, c), add_inline(m, c));
        const f256_s u2 = sqr_inline(u);

        f256_s p = log_atanh_coeffs[11];
        for (std::size_t i = 11; i > 0; --i)
            p = mul_add_inline(p, u2, log_atanh_coeffs[i - 1]);

        const f256_s table_log = mul_double_inline(std::numbers::ln2_v<f256_s>, static_cast<double>(exp2) + static_cast<double>(j) / 64.0);
        return add_inline(table_log, mul_inline(u, p));
    }

    BL_NO_INLINE constexpr f256_s exp_for_pow(const f256_s& x) noexcept
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();
        if (x.x0 > 709.782712893384)
            return std::numeric_limits<f256_s>::infinity();
        if (x.x0 < -745.133219101941)
            return f256_s{ 0.0 };
        if (iszero(x))
            return f256_s{ 1.0 };

        return F256_CANONICALIZE_MATH_RESULT(exp_from_reduced_64(x, false));
    }

    BL_MSVC_NOINLINE constexpr f256_s _exp(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 709.782712893384)
            return std::numeric_limits<f256_s>::infinity();

        if (x.x0 < -745.133219101941)
            return f256_s{ 0.0 };

        if (iszero(x))
            return f256_s{ 1.0 };

        return exp_from_reduced_64(x, false);
    }

    BL_MSVC_NOINLINE constexpr f256_s _exp2(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.x0 < 0.0) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 1023.0 || x.x0 < -1074.0)
            return _exp(mul_inline(x, std::numbers::ln2_v<f256_s>));

        if (iszero(x))
            return f256_s{ 1.0 };

        return exp_from_reduced_64(x, true);
    }

    BL_MSVC_NOINLINE constexpr f256_s _log(const f256_s& a)
    {
        return log_with_fast_exp_correction(a);
    }

    BL_MSVC_NOINLINE constexpr f256_s _expm1(const f256_s& x)
    {
        if (isnan(x))
            return x;
        if (x == f256_s{ 0.0 })
            return x;
        if (isinf(x))
            return signbit(x.x0)
                ? f256_s{ -1.0, 0.0, 0.0, 0.0 }
                : std::numeric_limits<f256_s>::infinity();

        if (x.x0 > 709.782712893384)
            return std::numeric_limits<f256_s>::infinity();

        if (x.x0 < -745.133219101941)
            return f256_s{ -1.0, 0.0, 0.0, 0.0 };

        return expm1_reduced(x);
    }

    // power functions
    BL_FORCE_INLINE constexpr f256_s powi(f256_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }

    BL_FORCE_INLINE constexpr f256_s polish_eighth_root(const f256_s& x, const f256_s& y)
    {
        if (iszero(y))
            return y;

        const f256_s y2 = sqr_inline(y);
        const f256_s y4 = sqr_inline(y2);
        const f256_s y7 = mul_inline(mul_inline(y4, y2), y);
        const f256_s y8 = sqr_inline(y4);
        const f256_s correction = div_double_inline(div_inline(sub_inline(x, y8), y7), 8.0);

        return add_inline(y, correction);
    }

    BL_FORCE_INLINE constexpr f256_s pow_positive_eighth_fraction(const f256_s& x, int numerator)
    {
        const f256_s r2 = detail::_f256_impl::sqrt(x);
        if (numerator == 4)
            return r2;

        const f256_s r4 = detail::_f256_impl::sqrt(r2);
        if (numerator == 2)
            return r4;

        f256_s out{ 1.0 };
        if ((numerator & 4) != 0)
            out = mul_inline(out, r2);
        if ((numerator & 2) != 0)
            out = mul_inline(out, r4);
        if ((numerator & 1) != 0)
        {
            const f256_s r8 = polish_eighth_root(x, detail::_f256_impl::sqrt(r4));
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
        const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
        return magnitude <= 1024;
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f256_s& x, const f256_s& y, int64_t& n)
    {
        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
            return false;

        if (!try_get_int64(mul_double_inline(y, 8.0), n))
            return false;

        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_FORCE_INLINE constexpr bool try_get_pow_dyadic_eighth_exponent(const f256_s& x, double y, int64_t& n) noexcept
    {
        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
            return false;

        const double scaled = y * 8.0;
        if (!isfinite(scaled) || absd(scaled) >= 0x1p63)
            return false;

        const double rounded = trunc(scaled);
        if (rounded != scaled)
            return false;

        n = static_cast<int64_t>(rounded);
        return pow_dyadic_eighth_exponent_in_range(n);
    }

    BL_NO_INLINE constexpr f256_s pow_dyadic_eighth_unchecked(const f256_s& x, int64_t n)
    {
        if (n == 0)
            return f256_s{ 1.0 };

        const bool neg = n < 0;
        const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
        const uint64_t whole     = magnitude / 8u;
        const int rem = static_cast<int>(magnitude & 7u);

        f256_s result = (whole == 0u) ? f256_s{ 1.0 } : powi(x, static_cast<int64_t>(whole));
        if (rem != 0)
            result = mul_inline(result, pow_positive_eighth_fraction(x, rem));
        if (neg)
            result = recip(result);

        return result;
    }

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
        const bool neg = signbit(x.x0);
        const exact_dyadic_fmod dx = exact_from_f256_fmod(mag(x));
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

    // sine/cosine functions
    BL_FORCE_INLINE constexpr bool remainder_pi2(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        if (!isfinite(x.x0))
            return false;

        if (mag(x) <= pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256_s q = detail::_f256_impl::nearbyint(mul_inline(x, invpi2));
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

    BL_MSVC_NOINLINE constexpr f256_s sin_kernel_pi4_inline(const f256_s& r)
    {
        const f256_s t = sqr_inline(r);

        const f256_s ps = horner_forward_inline(f256_sin_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        return mul_add_inline(mul_inline(r, t), ps, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s cos_kernel_pi4_inline(const f256_s& r)
    {
        const f256_s t = sqr_inline(r);

        const f256_s pc = horner_forward_inline(f256_cos_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        return mul_add_inline(t, pc, f256_s{ 1.0 });
    }

    BL_MSVC_NOINLINE constexpr void sincos_kernel_pi4_inline(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        const f256_s t = sqr_inline(r);

        f256_s ps{};
        f256_s pc{};
        horner_pair_forward_inline(f256_sin_coeffs_pi4, f256_cos_coeffs_pi4, f256_trig_coeff_count_pi4, t, ps, pc);

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
        {
            return sin_kernel_pi4_inline(r);
        }

        const f256_s t = sqr_inline(r);
        const f256_s ps = horner_forward(f256_sin_coeffs_pi4, f256_trig_coeff_count_pi4, t);

        const f256_s rt = mul_inline(r, t);
        return mul_add_horner_step(rt, ps, r);
    }

    BL_MSVC_NOINLINE constexpr f256_s cos_kernel_pi4(const f256_s& r)
    {
        if (bl::use_constexpr_math())
        {
            return cos_kernel_pi4_inline(r);
        }

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
        const double ax = fabs(x.x0);
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

    // inverse trig functions
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
        if (isinf(x))  return signbit(x.x0) ? -pi_2 : pi_2;

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

        const f256_s ax = detail::_f256::mag(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (ax == f256_s{ 1.0 })
            return (x.x0 < 0.0) ? -pi_2 : pi_2;

        if (ax <= f256_s{ 0.5 })
            return _atan(div_inline(x, detail::_f256_impl::sqrt(add_raw5_value_inline(neg_raw5(sqr_raw5_inline(x)), f256_s{ 1.0 }))));

        const f256_s t = detail::_f256_impl::sqrt(div_inline(sub_double_inline(1.0, ax), add_double_inline(ax, 1.0)));
        const f256_s a = sub_mul_double_inline(pi_2, _atan(t), 2.0);
        return (x.x0 < 0.0) ? -a : a;
    }

    // inverse hyperbolic functions
    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series(const f256_s& x)
    {
        const f256_s x2 = sqr_inline(x);
        f256_s sum   = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f256_s term = div_double_inline(power, static_cast<double>(2 * k + 1));
            sum = add_inline(sum, term);

            if (mag(term) <= f256_s::eps())
                break;
        }

        return sum;
    }

    BL_MSVC_NOINLINE constexpr f256_s atanh_small_series_runtime(const f256_s& x)
    {
        const f256_s x2 = sqr_inline(x);
        f256_s sum   = x;
        f256_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f256_s term = div_double_inline(power, static_cast<double>(2 * k + 1));
            sum = add_inline(sum, term);

            if (mag(term) <= f256_s::eps())
                break;
        }

        return sum;
    }

    // erf/erfc functions
    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_cheb_eval(const f256_s& x, const f256_s* coeffs, double shift)
    {
        if (!bl::use_constexpr_math())
        {
            return detail::_f256_runtime::cheb_eval(x, coeffs, f256_erf_cheb_coeff_count, shift);
        }

        const f256_s t = sub_inline(mul_double_inline(x, 2.0), f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (int i = static_cast<int>(f256_erf_cheb_coeff_count) - 1; i >= 1; --i)
        {
            const f256_s b0 = add_inline(
                mul_double_sub_inline(mul_inline(t, b1), 2.0, b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return add_inline(mul_sub_inline(t, b1, b2), coeffs[0]);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_cheb(const f256_s& x)
    {
        if (x < f256_s{ 1.0 })
            return erf_cheb_eval(x, f256_erf_cheb_0_1, 1.0);
        if (x < f256_s{ 2.0 })
            return erf_cheb_eval(x, f256_erf_cheb_1_2, 3.0);
        return erf_cheb_eval(x, f256_erf_cheb_2_3, 5.0);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erf_positive_series(const f256_s& x)
    {
        const f256_s xx = mul_inline(x, x);
        f256_s power = x;
        f256_s sum   = x;

        for (int n = 1; n < 512; ++n)
        {
            power = mul_inline(
                power,
                div_inline(-xx, f256_s{ static_cast<double>(n) }));

            const f256_s term = div_inline(
                power,
                f256_s{ static_cast<double>(2 * n + 1) });

            sum = add_inline(sum, term);
            if (mag(term) < f256_s::eps())
                break;
        }

        return mul_inline(
            mul_inline(f256_s{ 2.0 }, std::numbers::inv_sqrtpi_v<f256_s>),
            sum);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cheb_3_4(const f256_s& x)
    {
        return erf_cheb_eval(x, f256_erfc_cheb_3_4, 7.0);
    }

    [[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s erfc_positive_cf(const f256_s& x)
    {
        const f256_s z = sqr_inline(x);
        constexpr f256_s a = f256_s{ 0.5 };
        constexpr f256_s tiny = f256_s{ 1.0e-300, 0.0, 0.0, 0.0 };

        f256_s b = add_double_inline(z, 0.5);
        f256_s c = f256_s{ 1.0 } / tiny;
        f256_s d = f256_s{ 1.0 } / b;
        f256_s h = d;

        for (int i = 1; i <= 160; ++i)
        {
            const f256_s ii = f256_s{ static_cast<double>(i) };
            const f256_s an = -mul_inline(ii, sub_inline(ii, a));

            b = add_double_inline(b, 2.0);

            d = mul_add_inline(an, d, b);
            if (mag(d) < tiny)
                d = tiny;

            c = add_inline(b, div_inline(an, c));
            if (mag(c) < tiny)
                c = tiny;

            d = f256_s{ 1.0 } / d;
            const f256_s delta = mul_inline(d, c);
            h = mul_inline(h, delta);

            if (mag(sub_double_inline(delta, 1.0)) <= mul_double_inline(f256_s::eps(), 64.0))
                break;
        }

        return mul_inline(
            mul_inline(_exp(-z), x),
            mul_inline(std::numbers::inv_sqrtpi_v<f256_s>, h));
    }

    // gamma functions
    BL_NO_INLINE constexpr f256_s lgamma1p_series(const f256_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f256_s p = horner_reverse(lgamma1p_coeff, static_cast<std::size_t>(count), y);

        return mul_inline(y, mul_add_inline(y, p, -std::numbers::egamma_v<f256_s>));
    }

    BL_NO_INLINE constexpr bool try_lgamma_near_one_or_two(const f256_s& x, f256_s& out) noexcept
    {
        const f256_s y1 = sub_double_inline(x, 1.0);
        if (mag(y1) <= f256_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f256_s y2 = sub_double_inline(x, 2.0);
        if (mag(y2) <= f256_s{ 0.25 })
        {
            out = add_inline(log1p_series_reduced(y2), lgamma1p_series(y2));
            return true;
        }

        return false;
    }

    BL_NO_INLINE constexpr bool try_lgamma_short_recurrence(const f256_s& x, f256_s& out) noexcept
    {
        if (!(x > f256_s{ 0.0 }) || !(x < f256_s{ 32.0 }))
            return false;

        f256_s z = x;
        f256_s product{ 1.0 };
        bool shifted_up = false;

        while (z < f256_s{ 1.0 })
        {
            product = mul_inline(product, z);
            z = add_double_inline(z, 1.0);
            shifted_up = true;
        }
        while (z > f256_s{ 2.25 })
        {
            z = sub_double_inline(z, 1.0);
            product = mul_inline(product, z);
        }

        f256_s near_value{};
        if (!try_lgamma_near_one_or_two(z, near_value))
            return false;

        const f256_s log_product = _log(product);
        out = shifted_up ? sub_inline(near_value, log_product) : add_inline(near_value, log_product);
        return true;
    }

    BL_NO_INLINE constexpr void positive_recurrence_product(const f256_s& x, const f256_s& asymptotic_min, f256_s& z, f256_s& product, int& product_exp2) noexcept
    {
        z = x;
        product = f256_s{ 1.0 };
        product_exp2 = 0;

        while (z < asymptotic_min)
        {
            product = mul_inline(product, z);

            const double hi = product.x0;
            if (hi != 0.0)
            {
                const int e = frexp_exponent(hi);
                if (e > 512 || e < -512)
                {
                    product = detail::_f256_impl::ldexp(product, -e);
                    product_exp2 += e;
                }
            }

            z = add_double_inline(z, 1.0);
        }
    }

    BL_NO_INLINE constexpr f256_s lgamma_stirling_asymptotic(const f256_s& z) noexcept
    {
        const f256_s inv    = f256_s{ 1.0 } / z;
        const f256_s inv2   = sqr_eval(inv);
        const f256_s series = mul_eval(inv, horner_reverse(
            lgamma_stirling_coeffs,
            sizeof(lgamma_stirling_coeffs) / sizeof(lgamma_stirling_coeffs[0]),
            inv2));

        return add_eval(
            add_eval(mul_sub_eval(sub_double_eval(z, 0.5), detail::_f256_impl::log(z), z), half_log_two_pi),
            series);
    }

    BL_NO_INLINE constexpr f256_s lgamma_positive_recurrence(const f256_s& x) noexcept
    {
        f256_s near_value{};
        if (try_lgamma_near_one_or_two(x, near_value))
            return near_value;
        if (try_lgamma_short_recurrence(x, near_value))
            return near_value;

        constexpr f256_s asymptotic_min = f256_s{ 128.0 };

        f256_s z{};
        f256_s product{};
        int product_exp2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_exp2);

        return sub_mul_double_eval(
            sub_eval(lgamma_stirling_asymptotic(z), detail::_f256_impl::log(product)),
            std::numbers::ln2_v<f256_s>,
            static_cast<double>(product_exp2));
    }

} // namespace detail::_f256

// exponential functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::exp(const f256_s& x)
{
    return F256_CANONICALIZE_MATH_RESULT(_exp(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::exp2(const f256_s& x)
{
    return F256_CANONICALIZE_MATH_RESULT(_exp2(x));
}

// logarithm functions
[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f256_impl::log_as_double(f256_s a) noexcept
{
    const double hi = a.x0;
    if (hi <= 0.0)
        return detail::fp::log(static_cast<double>(a));

    const double lo = (a.x1 + a.x2) + a.x3;
    if (!bl::use_constexpr_math())
    {
        return std::log(hi) + std::log1p(lo / hi);
    }

    return detail::fp::log(hi) + detail::fp::log1p(lo / hi);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::log(const f256_s& a)
{
    return F256_CANONICALIZE_MATH_RESULT(_log(a));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::log2(const f256_s& a)
{
    return F256_CANONICALIZE_MATH_RESULT(mul_inline(_log(a), std::numbers::log2e_v<f256_s>));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::log10(const f256_s& a)
{
    return F256_CANONICALIZE_MATH_RESULT(_log(a) / std::numbers::ln10_v<f256_s>);
}

// expm1/log1p functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::expm1(const f256_s& x)
{
    return F256_CANONICALIZE_MATH_RESULT(_expm1(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::log1p(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ -1.0 })
        return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (x < f256_s{ -1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(x))
        return x;
    if (iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (ax <= f256_s{ 0.5 })
        return F256_CANONICALIZE_MATH_RESULT(log1p_series_reduced(x));

    const f256_s u = add_double_inline(x, 1.0);
    if (sub_double_inline(u, 1.0) == x)
        return F256_CANONICALIZE_MATH_RESULT(detail::_f256_impl::log(u));

    if (x > f256_s{ 0.0 } && x <= f256_s{ 1.0 })
    {
        const f256_s t = div_inline(x, add_double_inline(detail::_f256_impl::sqrt(add_double_inline(x, 1.0)), 1.0));
        return F256_CANONICALIZE_MATH_RESULT(mul_double_inline(log1p_series_reduced(t), 2.0));
    }

    if (x > f256_s{ 0.0 })
        return F256_CANONICALIZE_MATH_RESULT(detail::_f256_impl::log(u));

    const f256_s y = sub_double_inline(u, 1.0);
    if (iszero(y))
        return x;

    return F256_CANONICALIZE_MATH_RESULT(mul_inline(detail::_f256_impl::log(u), div_inline(x, y)));
}


// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::cbrt(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s y{};
    if (bl::use_constexpr_math())
    {
        y = detail::_f256_impl::exp(detail::_f256_impl::log(ax) / f256_s{ 3.0 });
    }
    else
    {
        int exp2 = 0;
        double mantissa = std::frexp(ax.x0, &exp2);
        int rem  = exp2 % 3;
        if (rem < 0)
            rem += 3;
        if (rem != 0)
        {
            mantissa = std::ldexp(mantissa, rem);
            exp2 -= rem;
        }

        y = f256_s{ std::cbrt(mantissa), 0.0, 0.0, 0.0 };
        if (exp2 != 0)
            y = detail::_f256_impl::ldexp(y, exp2 / 3);
    }

    const auto cbrt_newton_step = [&](const f256_s& current) constexpr -> f256_s
    {
        const f256_s current_squared = sqr_inline(current);
        const f256_s quotient = div_inline(ax, current_squared);
        return div_double_inline(add_scaled_inline<2, 1>(current, quotient), 3.0);
    };
    const auto cbrt_tail_step = [&](const f256_s& current) constexpr -> f256_s
    {
        const double inv_derivative = 1.0 / (3.0 * current.x0 * current.x0);
        const f256_s current_squared = sqr_inline(current);
        const f256_s residual = value_sub_mul_inline(ax, current_squared, current);
        return add_inline(current, mul_double_inline(residual, inv_derivative));
    };

    if (bl::use_constexpr_math())
    {
        y = cbrt_newton_step(y);
        y = cbrt_newton_step(y);
        y = cbrt_newton_step(y);
    }
    else
    {
        y = cbrt_tail_step(y);
        y = cbrt_tail_step(y);
        y = cbrt_tail_step(y);
        y = cbrt_tail_step(y);
    }

    if (neg)
        y = -y;

    return F256_CANONICALIZE_MATH_RESULT(y);
}

// power functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::pow10_256(int k)
{
    if (k == 0) [[unlikely]]
        return f256_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    if (n <= 16) {
        f256_s r = f256_s{ 1.0 };
        const f256_s ten = f256_s{ 10.0, 0.0, 0.0, 0.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f256_s{ 1.0 } / r);
    }

    f256_s r = f256_s{ 1.0 };
    f256_s base = f256_s{ 10.0, 0.0, 0.0, 0.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f256_s{ 1.0 } / r);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::pow(const f256_s& x, const f256_s& y)
{
    if (iszero(y))
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s yi = detail::_f256_impl::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && try_get_int64(yi, yi64))
        return powi(x, yi64);

    int64_t dyadic_exponent{};
    if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
        return F256_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = exp_for_pow(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return F256_CANONICALIZE_MATH_RESULT(exp_for_pow(mul_inline(y, _log(x))));
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::pow(const f256_s& x, double y)
{
    if (y == 0.0)
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (y == 1.0) return x;
    if (y == 2.0) return F256_CANONICALIZE_MATH_RESULT(sqr_inline(x));
    if (y == -1.0) return F256_CANONICALIZE_MATH_RESULT(f256_s{ 1.0 } / x);
    if (y == 0.5) return F256_CANONICALIZE_MATH_RESULT(detail::_f256_impl::sqrt(x));

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
        return F256_CANONICALIZE_MATH_RESULT(pow_dyadic_eighth_unchecked(x, dyadic_exponent));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = exp_for_pow(mul_double_inline(_log(-x), y));
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return F256_CANONICALIZE_MATH_RESULT(y_is_odd ? -magnitude : magnitude);
    }

    return F256_CANONICALIZE_MATH_RESULT(exp_for_pow(mul_double_inline(_log(x), y)));
}

// inverse trig functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::atan(const f256_s& x)
{
    return F256_CANONICALIZE_MATH_RESULT(_atan(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::asin(const f256_s& x)
{
    return F256_CANONICALIZE_MATH_RESULT(_asin(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::acos(const f256_s& x)
{
    if (isnan(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (ax > f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (x == f256_s{ 1.0 })
        return f256_s{ 0.0 };
    if (x == f256_s{ -1.0 })
        return std::numbers::pi_v<f256_s>;

    return F256_CANONICALIZE_MATH_RESULT(pi_2 - _asin(x));
}

// sine/cosine functions
[[nodiscard]] BL_FORCE_INLINE constexpr bool detail::_f256_impl::sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    bool ret = _sincos(x, s_out, c_out);
    s_out = F256_CANONICALIZE_MATH_RESULT(s_out);
    c_out = F256_CANONICALIZE_MATH_RESULT(c_out);
    return ret;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sin(const f256_s& x)
{
    const double ax = detail::_f256::fabs(x.x0);
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

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::cos(const f256_s& x)
{
    const double ax = detail::_f256::fabs(x.x0);
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

// tangent and atan2
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::tan(const f256_s& x)
{
    f256_s s{}, c{};
    if (_sincos(x, s, c))
        return F256_CANONICALIZE_MATH_RESULT(s / c);

    return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::atan2(const f256_s& y, const f256_s& x)
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
            return signbit(y.x0) ? -std::numbers::pi_v<f256_s> : std::numbers::pi_v<f256_s>;
        return y;
    }

    const f256_s ax = detail::_f256::mag(x);
    const f256_s ay = detail::_f256::mag(y);

    if (ax == ay)
    {
        if (x.x0 < 0.0)
        {
            return F256_CANONICALIZE_MATH_RESULT(
                (y.x0 < 0.0) ? -pi_3_4 : pi_3_4);
        }

        return F256_CANONICALIZE_MATH_RESULT(
            (y.x0 < 0.0) ? -pi_4 : pi_4);
    }

    if (ax >= ay)
    {
        f256_s a = _atan(y / x);

        if (x.x0 < 0.0)
            a += (y.x0 < 0.0) ? -std::numbers::pi_v<f256_s> : std::numbers::pi_v<f256_s>;
        return F256_CANONICALIZE_MATH_RESULT(a);
    }

    f256_s a = _atan(x / y);
    return F256_CANONICALIZE_MATH_RESULT((y.x0 < 0.0) ? (-pi_2 - a) : (pi_2 - a));
}

// sinh/cosh/tanh
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sinh(const f256_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (!bl::use_constexpr_math())
    {
        const f256_s em1 = _expm1(ax);
        f256_s out = div_inline(
            mul_add_inline(em1, em1, mul_double_inline(em1, 2.0)),
            mul_double_inline(add_scalar_precise(em1, 1.0), 2.0));
        if (signbit(x))
            out = -out;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    if (ax <= f256_s{ 0.5 })
    {
        const f256_s x2 = mul_inline(x, x);
        f256_s term = x;
        f256_s sum  = x;

        for (int n = 1; n <= 256; ++n)
        {
            term = div_double_inline(
                mul_inline(term, x2),
                static_cast<double>((2 * n) * (2 * n + 1)));
            sum = add_inline(sum, term);

            const f256_s asum  = detail::_f256::mag(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (detail::_f256::mag(term) <= mul_inline(f256_s::eps(), scale))
                break;
        }

        return F256_CANONICALIZE_MATH_RESULT(sum);
    }

    const f256_s ex     = _exp(ax);
    const f256_s inv_ex = recip(ex);
    f256_s out = mul_double_inline(sub_inline(ex, inv_ex), 0.5);
    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::cosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s ax     = detail::_f256::mag(x);
    const f256_s ex     = _exp(ax);
    const f256_s inv_ex = recip(ex);
    return F256_CANONICALIZE_MATH_RESULT(
        mul_double_inline(add_inline(ex, inv_ex), 0.5));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::tanh(const f256_s& x)
{
    using namespace detail::_f256;
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s ax = detail::_f256::mag(x);
    if (ax > f256_s{ 20.0 })
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s em1   = _expm1(mul_double_inline(ax, 2.0));
    const f256_s denom = add_scalar_precise(em1, 2.0);

    f256_s out = div_inline(em1, denom);

    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

// inverse hyperbolic functions
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::asinh(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    f256_s out{};
    if (ax > f256_s{ 0x1p500 })
        out = add_inline(detail::_f256_impl::log(ax), std::numbers::ln2_v<f256_s>);
    else
        out = detail::_f256_impl::log(add_inline(ax, detail::_f256_impl::sqrt(add_raw5_value_inline(sqr_raw5_inline(ax), f256_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::acosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x < f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (x == f256_s{ 1.0 })
        return f256_s{ 0.0 };
    if (isinf(x))
        return x;

    f256_s out{};
    if (x > f256_s{ 0x1p500 })
        out = add_inline(detail::_f256_impl::log(x), std::numbers::ln2_v<f256_s>);
    else
        out = detail::_f256_impl::log(add_inline(
            x,
            detail::_f256_impl::sqrt(mul_inline(sub_double_inline(x, 1.0), add_double_inline(x, 1.0)))));

    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::atanh(const f256_s& x)
{
    if (isnan(x) || iszero(x))
        return x;

    const f256_s ax = detail::_f256::mag(x);
    if (ax > f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (ax == f256_s{ 1.0 })
        return signbit(x)
        ? f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }
        : f256_s{ std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };

    if (ax <= f256_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
        {
            return F256_CANONICALIZE_MATH_RESULT(atanh_small_series(x));
        }

        return F256_CANONICALIZE_MATH_RESULT(atanh_small_series_runtime(x));
    }

    const f256_s out = mul_double_inline(
        detail::_f256_impl::log(div_inline(add_double_inline(x, 1.0), sub_double_inline(1.0, x))),
        0.5);
    return F256_CANONICALIZE_MATH_RESULT(out);
}

// erf/erfc functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::erf(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s out{};

    if (ax < f256_s{ 1.0 })
    {
        out = erf_positive_series(ax);
    }
    else if (ax < f256_s{ 3.0 })
    {
        out = erf_positive_cheb(ax);
    }
    else if (ax < f256_s{ 4.0 })
    {
        out = f256_s{ 1.0 } - erfc_positive_cheb_3_4(ax);
    }
    else
    {
        out = f256_s{ 1.0 } - erfc_positive_cf(ax);
    }

    if (neg)
        out = -out;

    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s detail::_f256_impl::erfc(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return f256_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f256_s{ 2.0 } : f256_s{ 0.0 };

    if (signbit(x))
        return F256_CANONICALIZE_MATH_RESULT(add_double_inline(detail::_f256_impl::erf(-x), 1.0));

    if (x < f256_s{ 1.0 })
        return F256_CANONICALIZE_MATH_RESULT(sub_double_inline(1.0, erf_positive_series(x)));

    if (x < f256_s{ 3.0 })
        return F256_CANONICALIZE_MATH_RESULT(sub_double_inline(1.0, erf_positive_cheb(x)));

    if (x < f256_s{ 4.0 })
        return F256_CANONICALIZE_MATH_RESULT(erfc_positive_cheb_3_4(x));

    if (x > f256_s{ 40.0 })
        return f256_s{ 0.0 };

    return F256_CANONICALIZE_MATH_RESULT(erfc_positive_cf(x));
}

// gamma functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::lgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return F256_CANONICALIZE_MATH_RESULT(lgamma_positive_recurrence(x));

    const f256_s xi = detail::_f256_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::infinity();

    const f256_s sinpix = detail::_f256_impl::sin(mul_inline(std::numbers::pi_v<f256_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s out =
        mul_double_eval(half_log_pi, 2.0)
        - detail::_f256_impl::log(detail::_f256::mag(sinpix))
        - lgamma_positive_recurrence(f256_s{ 1.0 } - x);

    return F256_CANONICALIZE_MATH_RESULT(out);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::tgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return F256_CANONICALIZE_MATH_RESULT(_exp(lgamma_positive_recurrence(x)));

    const f256_s xi = detail::_f256_impl::trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s sinpix = detail::_f256_impl::sin(mul_inline(std::numbers::pi_v<f256_s>, x));
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s log_abs = sub_eval(
        sub_eval(mul_double_eval(half_log_pi, 2.0), detail::_f256_impl::log(detail::_f256::mag(sinpix))),
        lgamma_positive_recurrence(sub_double_inline(1.0, x)));
    f256_s out = _exp(log_abs);
    if (signbit(sinpix))
        out = -out;
    return F256_CANONICALIZE_MATH_RESULT(out);
}

} // namespace bl

#endif // FLTX_F256_DETAIL_TRANSCENDENTAL_INCLUDED
