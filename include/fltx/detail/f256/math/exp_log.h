/**
 * fltx/detail/f256/math/exp_log.h - exp/log implementation details.
 *
 * f256 exponential and logarithm helpers and kernels.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_EXP_LOG_INCLUDED
#define FLTX_F256_DETAIL_EXP_LOG_INCLUDED
#include "fltx/detail/f256/math_shared.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f256_constexpr::log_as_double(f256_s a) noexcept;

namespace detail::_f256
{

    // log1p helpers
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
            return detail::_f256_runtime::log1p_series_reduced(x);

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

    // exponential kernels
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

    // logarithm kernels
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

        const double log2_m = bl::use_constexpr_math()
            ? detail::fp::log(m.x0) * 1.4426950408889634074
            : std::log2(m.x0);

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

        return canonicalize_math_result(exp_from_reduced_64(x, false));
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

} // namespace detail::_f256

// exponential functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::exp(const f256_s& x)
{
    return canonicalize_math_result(_exp(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::exp2(const f256_s& x)
{
    return canonicalize_math_result(_exp2(x));
}

// logarithm functions
[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f256_constexpr::log_as_double(f256_s a) noexcept
{
    const double hi = a.x0;
    if (hi <= 0.0)
        return detail::fp::log(static_cast<double>(a));

    const double lo = (a.x1 + a.x2) + a.x3;
    if (!bl::use_constexpr_math())
        return std::log(hi) + std::log1p(lo / hi);

    return detail::fp::log(hi) + detail::fp::log1p(lo / hi);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::log(const f256_s& a)
{
    return canonicalize_math_result(_log(a));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::log2(const f256_s& a)
{
    return canonicalize_math_result(mul_inline(_log(a), std::numbers::log2e_v<f256_s>));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::log10(const f256_s& a)
{
    return canonicalize_math_result(_log(a) / std::numbers::ln10_v<f256_s>);
}

// expm1/log1p functions
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::expm1(const f256_s& x)
{
    return canonicalize_math_result(_expm1(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::log1p(const f256_s& x)
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
        return canonicalize_math_result(log1p_series_reduced(x));

    const f256_s u = add_double_inline(x, 1.0);
    if (sub_double_inline(u, 1.0) == x)
        return canonicalize_math_result(detail::_f256_constexpr::log(u));

    if (x > f256_s{ 0.0 } && x <= f256_s{ 1.0 })
    {
        const f256_s t = div_inline(x, add_double_inline(detail::_f256_constexpr::sqrt(add_double_inline(x, 1.0)), 1.0));
        return canonicalize_math_result(mul_double_inline(log1p_series_reduced(t), 2.0));
    }

    if (x > f256_s{ 0.0 })
        return canonicalize_math_result(detail::_f256_constexpr::log(u));

    const f256_s y = sub_double_inline(u, 1.0);
    if (iszero(y))
        return x;

    return canonicalize_math_result(mul_inline(detail::_f256_constexpr::log(u), div_inline(x, y)));
}

} // namespace bl

#endif
