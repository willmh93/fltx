/**
 * fltx/detail/f128_math_exp_log.h - exp/log implementation details.
 *
 * Range-reduced f128 exponential and logarithm cores, including base-2/base-10 and one-plus variants.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_EXP_LOG_INCLUDED
#define FLTX_F128_DETAIL_EXP_LOG_INCLUDED
#include "fltx/detail/f128_math_sqrt.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::log_as_double(f128_s a);

namespace detail::_f128 // primitives and kernels
{

    // log1p helpers
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

    BL_MSVC_NOINLINE constexpr f128_s f128_log1p_series_reduced(const f128_s& x)
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

    // exponential kernels
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

    // logarithm kernels
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

    // exact logarithm cases
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
        return F128_CANONICALIZE_MATH_RESULT(f128_log1p_series_reduced(x));

    const f128_s u = add_inline(f128_s{ 1.0 }, x);
    if (sub_inline(u, f128_s{ 1.0 }) == x)
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128_impl::log(u));

    if (x > f128_s{ 0.0 } && x <= f128_s{ 1.0 })
    {
        const f128_s t = div_inline(x, add_inline(f128_s{ 1.0 }, detail::_f128_impl::sqrt(add_inline(f128_s{ 1.0 }, x))));
        return F128_CANONICALIZE_MATH_RESULT(mul_inline(f128_log1p_series_reduced(t), f128_s{ 2.0 }));
    }

    if (x > f128_s{ 0.0 })
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128_impl::log(u));

    const f128_s y = sub_inline(u, f128_s{ 1.0 });
    if (iszero(y))
        return x;

    return F128_CANONICALIZE_MATH_RESULT(mul_inline(detail::_f128_impl::log(u), div_inline(x, y)));
}

} // namespace bl

#endif
