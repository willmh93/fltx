/**
 * fltx/detail/f128/math/basic.h - Basic math implementation details.
 *
 * f128 rounding, decomposition, remainder, cbrt, hypot, and adjacent-value implementations.
 * This core includes exp/log details because cbrt uses them in constant evaluation.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_DETAIL_BASIC_INCLUDED
#define FLTX_F128_DETAIL_BASIC_INCLUDED
#include "fltx/detail/f128/math/exp_log.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::round(const f128_s& a)
{
    return detail::_f128::round_impl(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::nearbyint(const f128_s& a)
{
    return detail::_f128::nearbyint_impl(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::rint(const f128_s& x)
{
    return detail::_f128_constexpr::nearbyint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_constexpr::lround(const f128_s& x)
{
    return detail::_f128::lround_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_constexpr::llround(const f128_s& x)
{
    return detail::_f128::llround_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_constexpr::lrint(const f128_s& x)
{
    return detail::_f128::lrint_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_constexpr::llrint(const f128_s& x)
{
    return detail::_f128::llrint_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::round_to_decimals(f128_s v, int prec)
{
    constexpr int local_capacity = std::numeric_limits<f128_s>::max_digits10;

    if (prec <= 0) return v;
    if (prec > local_capacity) prec = local_capacity;

    constexpr f128_s INV10_DD
    {
        0.1000000000000000055511151231257827021181583404541015625,
       -0.0000000000000000055511151231257827021181583404541015625
    };

    char digits[local_capacity];

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f128_s ip   = detail::_f128_constexpr::floor(v);
    f128_s frac = sub_inline(v, ip);

    f128_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = mul_inline(w, f128_s{ 10.0 });

        int di = static_cast<int>(detail::_f128_constexpr::floor(w).hi);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = sub_inline(w, f128_s{ static_cast<double>(di) });
    }

    f128_s la = mul_inline(w, f128_s{ 10.0 });

    const f128_s tie_slop = mul_inline(f128_s::eps(), f128_s{ 65536.0 });
    int next = static_cast<int>(detail::_f128_constexpr::floor(la).hi);
    if (next < 0) next = 0;

    f128_s rem = sub_inline(la, f128_s{ static_cast<double>(next) });
    if (next < 10 && rem >= sub_inline(f128_s{ 1.0 }, tie_slop))
    {
        ++next;
        rem = sub_inline(rem, f128_s{ 1.0 });
    }

    const int last = digits[prec - 1] - '0';
    const bool beyond_half = rem > tie_slop;
    const bool round_up    =
        (next > 5) ||
        (next == 5 && (beyond_half || (last & 1)));

    if (round_up)
    {
        int i = prec - 1;
        for (; i >= 0; --i)
        {
            if (digits[i] == '9')
            {
                digits[i] = '0';
            }
            else
            {
                ++digits[i];
                break;
            }
        }

        if (i < 0)
            ip = add_inline(ip, f128_s{ 1.0 });
    }

    f128_s exact_out{};
    if (try_rounded_decimal_to_f128(ip, digits, prec, neg, exact_out))
        return exact_out;

    f128_s frac_val{ 0.0, 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = add_inline(
            frac_val,
            f128_s{ static_cast<double>(digits[i] - '0') });

        frac_val = mul_inline(frac_val, INV10_DD);
    }

    f128_s out = add_inline(ip, frac_val);
    return neg ? -out : out;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::remainder(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s ay   = abs(y);
    f128_s r = detail::_f128_constexpr::fmod(x, y);
    const f128_s ar   = abs(r);
    const f128_s half = mul_inline(ay, f128_s{ 0.5 });

    if (ar > half)
    {
        r = add_inline(r, signbit(r) ? ay : -ay);
    }
    else if (ar == half)
    {
        const f128_s q = detail::_f128_constexpr::trunc(div_inline(x, y));
        const f128_s q_mod2 = abs(detail::_f128_constexpr::fmod(q, f128_s{ 2.0 }));
        if (q_mod2 != f128_s{ 0.0 })
            r = add_inline(r, signbit(r) ? ay : -ay);
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::cbrt(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y{};
    if (bl::use_constexpr_math())
    {
        y = detail::_f128_constexpr::exp(div_inline(detail::_f128_constexpr::log(ax), f128_s{ 3.0 }));
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

    return canonicalize_math_result(y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::hypot(const f128_s& x, const f128_s& y)
{
    using namespace detail::_f128;

    if (isinf(x) || isinf(y))
        return std::numeric_limits<f128_s>::infinity();
    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    f128_s ax = abs(x);
    f128_s ay = abs(y);
    if (ax < ay)
        std::swap(ax, ay);

    if (iszero(ax))
        return f128_s{ 0.0 };
    if (iszero(ay))
        return canonicalize_math_result(ax);

    int ex = 0;
    int ey = 0;
    if (bl::use_constexpr_math())
    {
        ex = detail::fp::frexp_exponent(ax.hi);
        ey = detail::fp::frexp_exponent(ay.hi);
    }
    else
    {
        (void)std::frexp(ax.hi, &ex);
        (void)std::frexp(ay.hi, &ey);
    }

    if ((ex - ey) > 55)
        return canonicalize_math_result(ax);

    const f128_s r = div_inline(ay, ax);
    return canonicalize_math_result(mul_inline(ax, detail::_f128_constexpr::sqrt(add_inline(f128_s{ 1.0 }, mul_inline(r, r)))));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::remquo(const f128_s& x, const f128_s& y, int* quo)
{
    using namespace detail::_f128;

    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s n = nearest_integer_ties_even(div_inline(x, y));
    f128_s r = sub_inline(x, mul_inline(n, y));

    if (quo)
    {
        const f128_s qbits = detail::_f128_constexpr::fmod(abs(n), f128_s{ 2147483648.0 });
        int bits = static_cast<int>(detail::_f128_constexpr::trunc(qbits).hi);
        if (signbit(n))
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::modf(const f128_s& x, f128_s* iptr) noexcept
{
    const f128_s i = detail::_f128_constexpr::trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
}

} // namespace bl

#endif
