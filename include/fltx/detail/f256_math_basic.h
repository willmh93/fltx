/**
 * fltx/detail/f256_math_basic.h - Basic math implementation details.
 *
 * f256 rounding, decomposition, remainder, cbrt, hypot, and adjacent-value implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_DETAIL_BASIC_INCLUDED
#define FLTX_F256_DETAIL_BASIC_INCLUDED
#include "fltx/detail/f256_math_exp_log.h"

namespace bl {

// norms
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::hypot(const f256_s& x, const f256_s& y)
{
    if (isinf(x) || isinf(y))
        return std::numeric_limits<f256_s>::infinity();
    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    f256_s ax = detail::_f256::mag(x);
    f256_s ay = detail::_f256::mag(y);
    if (ax < ay)
    {
        const f256_s tmp = ax;
        ax = ay;
        ay = tmp;
    }

    if (iszero(ax))
        return f256_s{ 0.0 };
    if (iszero(ay))
        return F256_CANONICALIZE_MATH_RESULT(ax);

    int ex = 0;
    int ey = 0;
    if (bl::use_constexpr_math())
    {
        ex = frexp_exponent(ax.x0);
        ey = frexp_exponent(ay.x0);
    }
    else
    {
        (void)std::frexp(ax.x0, &ex);
        (void)std::frexp(ay.x0, &ey);
    }

    if ((ex - ey) > 110)
        return F256_CANONICALIZE_MATH_RESULT(ax);

    if (!bl::use_constexpr_math())
    {
        if (ex > -450 && ex < 450)
            return F256_CANONICALIZE_MATH_RESULT(detail::_f256_impl::sqrt(add_raw5_raw5_inline(sqr_raw5_inline(ax), sqr_raw5_inline(ay))));
    }

    const f256_s r = div_inline(ay, ax);
    return F256_CANONICALIZE_MATH_RESULT(mul_inline(ax, detail::_f256_impl::sqrt(add_raw5_value_inline(sqr_raw5_inline(r), f256_s{ 1.0 }))));
}

// integer rounding
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::rint(const f256_s& x)
{
    return detail::_f256_impl::nearbyint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f256_impl::lround(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f256_impl::llround(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f256_impl::lrint(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_impl::nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f256_impl::llrint(const f256_s& x)
{
    return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_impl::nearbyint(x));
}

// arithmetic and comparisons
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    return F256_CANONICALIZE_MATH_RESULT(mul_add_inline(x, y, z));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmin(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return bl::signbit(a) ? a : b;
    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmax(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return bl::signbit(a) ? b : a;
    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fdim(const f256_s& x, const f256_s& y)
{
    return (x > y) ? F256_CANONICALIZE_MATH_RESULT(x - y) : f256_s{ 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::copysign(const f256_s& x, const f256_s& y)
{
    return bl::signbit(x) == bl::signbit(y) ? x : -x;
}

// rounding and decimals
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::round(const f256_s& a)
{
    return round_half_away_zero(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::round_to_decimals(f256_s v, int prec)
{
    constexpr int local_capacity = std::numeric_limits<f256_s>::max_digits10;

    if (prec <= 0) return v;
    if (prec > local_capacity) prec = local_capacity;

    constexpr f256_s inv10_qd{
         0x1.999999999999ap-4,
        -0x1.999999999999ap-58,
         0x1.999999999999ap-112,
        -0x1.999999999999ap-166
    };

    char digits[local_capacity];

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f256_s ip   = detail::_f256_impl::floor(v);
    f256_s frac = sub_inline(v, ip);

    f256_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = mul_double_inline(w, 10.0);

        int di = static_cast<int>(detail::_f256_impl::floor(w).x0);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = sub_double_inline(w, static_cast<double>(di));
    }

    f256_s la = mul_double_inline(w, 10.0);

    const f256_s tie_slop = mul_double_inline(f256_s::eps(), 65536.0);
    int next = static_cast<int>(detail::_f256_impl::floor(la).x0);
    if (next < 0) next = 0;

    f256_s rem = sub_double_inline(la, static_cast<double>(next));
    if (next < 10 && rem >= sub_inline(f256_s{ 1.0 }, tie_slop))
    {
        ++next;
        rem = sub_double_inline(rem, 1.0);
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
            ip = add_double_inline(ip, 1.0);
    }

    f256_s exact_out{};
    if (try_rounded_decimal_to_f256(ip, digits, prec, neg, exact_out))
        return exact_out;

    f256_s frac_val{ 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = add_double_inline(frac_val, static_cast<double>(digits[i] - '0'));
        frac_val = mul_inline(frac_val, inv10_qd);
    }

    f256_s out = add_inline(ip, frac_val);
    return neg ? -out : out;
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

// remainders
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::remquo(const f256_s& x, const f256_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = nearest_integer_ties_even(x / y);
    f256_s r = value_sub_mul_inline(x, n, y);

    if (quo)
        *quo = low_quotient_bits(n);

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return F256_CANONICALIZE_MATH_RESULT(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::remainder(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = nearest_integer_ties_even(x / y);
    f256_s r = value_sub_mul_inline(x, n, y);

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return F256_CANONICALIZE_MATH_RESULT(r);
}

// fractional decomposition
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::modf(const f256_s& x, f256_s* iptr) noexcept
{
    const f256_s i = detail::_f256_impl::trunc(x);
    if (iptr)
        *iptr = i;

    f256_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
    return frac;
}

// decomposition and scaling
[[nodiscard]] BL_FORCE_INLINE constexpr int detail::_f256_impl::ilogb(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_ILOGBNAN;
    if (iszero(x)) return FP_ILOGB0;
    if (isinf(x))  return std::numeric_limits<int>::max();

    int e = 0;
    (void)detail::_f256_impl::frexp(detail::_f256::mag(x), &e);
    return e - 1;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::logb(const f256_s& x) noexcept
{
    if (isnan(x))  return x;
    if (iszero(x)) return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (isinf(x))  return std::numeric_limits<f256_s>::infinity();

    return f256_s{ static_cast<double>(detail::_f256_impl::ilogb(x)), 0.0, 0.0, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::scalbn(const f256_s& x, int e) noexcept
{
    return detail::_f256_impl::ldexp(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::scalbln(const f256_s& x, long e) noexcept
{
    return detail::_f256_impl::ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nexttoward(const f256_s& from, long double to) noexcept
{
    return detail::_f256_impl::nextafter(from, f256_s{ static_cast<double>(to), 0.0, 0.0, 0.0 });
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    return detail::_f256_impl::nextafter(from, to);
}

} // namespace bl

#endif
