/**
 * fltx/detail/f128_math_basic.h - Basic math implementation details.
 *
 * f128 rounding, decomposition, remainder, hypot, and adjacent-value implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_MATH_BASIC_INCLUDED
#define F128_DETAIL_MATH_BASIC_INCLUDED
#include "fltx/detail/f128_math_kernels.h"

namespace bl {

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    constexpr double fast_min = 0x1p-900;
    constexpr double fast_max = 0x1p900;
    if (!bl::use_constexpr_math() && a.hi >= fast_min && a.hi <= fast_max)
        return F128_CANONICALIZE_MATH_RESULT(detail::_f128::sqrt_compensated(a, std::sqrt(a.hi)));

    const int exp2 = frexp_exponent(a.hi);
    const int result_scale = exp2 / 2;
    const int input_scale = -2 * result_scale;
    const f128_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

    double seed{};
    if (bl::use_constexpr_math())
    {
        seed = detail::_f128::sqrt_constexpr_head(scaled_a.hi);
    }
    else
    {
        seed = std::sqrt(scaled_a.hi);
    }

    f128_s y = detail::_f128::sqrt_compensated(scaled_a, seed);

    if (result_scale != 0)
        y = ldexp_terms(y, result_scale);

    return F128_CANONICALIZE_MATH_RESULT(y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::hypot(const f128_s& x, const f128_s& y)
{
    using namespace detail::_f128;

    if (isinf(x) || isinf(y))
        return std::numeric_limits<f128_s>::infinity();
    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    f128_s ax = detail::_f128::mag(x);
    f128_s ay = detail::_f128::mag(y);
    if (ax < ay)
    {
        const f128_s tmp = ax;
        ax = ay;
        ay = tmp;
    }

    if (iszero(ax))
        return f128_s{ 0.0 };
    if (iszero(ay))
        return F128_CANONICALIZE_MATH_RESULT(ax);

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
        return F128_CANONICALIZE_MATH_RESULT(ax);

    const f128_s r = div_inline(ay, ax);
    return F128_CANONICALIZE_MATH_RESULT(mul_inline(ax, detail::_f128_impl::sqrt(add_inline(f128_s{ 1.0 }, mul_inline(r, r)))));
}

// rounding
[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::floor_limb(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128::floor(x),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::ceil_limb(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128::ceil(x),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::floor(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128_impl::floor_limb(a.lo));
    }

    double hi = detail::_f128_impl::floor_limb(a.hi);
    if (hi == a.hi && a.lo < 0.0)
        hi -= 1.0;
    return f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::ceil(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = detail::fp::double_integer_threshold;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128_impl::ceil_limb(a.lo));
    }

    double hi = detail::_f128_impl::ceil_limb(a.hi);
    if (hi == a.hi && a.lo > 0.0)
        hi += 1.0;
    return f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::trunc(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    return (a.hi < 0.0) ? detail::_f128_impl::ceil(a) : detail::_f128_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::round(const f128_s& a)
{
    return round_half_away_zero(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::nearbyint(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f128_s t = detail::_f128_impl::floor(a);
    f128_s frac = sub_inline(a, t);

    if (frac < f128_s{ 0.5 })
        return t;

    if (frac > f128_s{ 0.5 })
    {
        t = add_inline(t, f128_s{ 1.0 });
        if (iszero(t))
            return f128_s{ signbit(a.hi) ? -0.0 : 0.0 };
        return t;
    }

    if (detail::_f128_impl::fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t = add_inline(t, f128_s{ 1.0 });

    if (iszero(t))
        return f128_s{ signbit(a.hi) ? -0.0 : 0.0 };

    return t;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::rint(const f128_s& x)
{
    return detail::_f128_impl::nearbyint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_impl::lround(const f128_s& x)
{
    return to_signed_integer_or_zero<long>(round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_impl::llround(const f128_s& x)
{
    return to_signed_integer_or_zero<long long>(round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_impl::lrint(const f128_s& x)
{
    return to_signed_integer_or_zero<long>(detail::_f128_impl::nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_impl::llrint(const f128_s& x)
{
    return to_signed_integer_or_zero<long long>(detail::_f128_impl::nearbyint(x));
}

// arithmetic and comparisons
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    return F128_CANONICALIZE_MATH_RESULT(add_inline(mul_inline(x, y), z));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fmin(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fmax(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fdim(const f128_s& x, const f128_s& y)
{
    return (x > y) ? F128_CANONICALIZE_MATH_RESULT(sub_inline(x, y)) : f128_s{ 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::copysign(const f128_s& x, const f128_s& y)
{
    return signbit(x) == signbit(y) ? x : -x;
}

// decomposition and scaling
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::ldexp(const f128_s& x, int e)
{
    return F128_CANONICALIZE_MATH_RESULT(_ldexp(x, e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::frexp(const f128_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (!detail::fp::isfinite(x.hi) || (x.hi == 0.0 && x.lo == 0.0))
        return x;

    int e = 0;
    double scaled_hi = 0.0;
    double scaled_lo = 0.0;

    if (bl::use_constexpr_math())
    {
        const double lead = (x.hi != 0.0) ? x.hi : x.lo;
        e = detail::fp::frexp_exponent(lead);
        scaled_hi = detail::fp::ldexp(x.hi, -e);
        scaled_lo = detail::fp::ldexp(x.lo, -e);
    }
    else
    {
        if (x.hi != 0.0)
        {
            scaled_hi = std::frexp(x.hi, &e);
            scaled_lo = std::ldexp(x.lo, -e);
        }
        else
        {
            scaled_hi = std::frexp(x.lo, &e);
        }
    }

    const double hi = scaled_hi + scaled_lo;
    f128_s m{hi, (scaled_hi - hi) + scaled_lo};
    f128_s am = (m.hi < 0.0) ? -m : m;

    if (am.hi == 0.5 && am.lo < 0.0)
    {
        m.hi *= 2.0;
        m.lo *= 2.0;
        --e;
    }
    else if (am.hi == 1.0 && am.lo >= 0.0)
    {
        m.hi *= 0.5;
        m.lo *= 0.5;
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}

[[nodiscard]] BL_FORCE_INLINE constexpr int detail::_f128_impl::ilogb(const f128_s& x) noexcept
{
    if (isnan(x))
        return FP_ILOGBNAN;
    if (iszero(x))
        return FP_ILOGB0;
    if (isinf(x))
        return std::numeric_limits<int>::max();

    int e = 0;
    (void)detail::_f128_impl::frexp(detail::_f128::mag(x), &e);
    return e - 1;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::logb(const f128_s& x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    return f128_s{ static_cast<double>(detail::_f128_impl::ilogb(x)), 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::scalbn(const f128_s& x, int e) noexcept
{
    return detail::_f128_impl::ldexp(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::scalbln(const f128_s& x, long e) noexcept
{
    return detail::_f128_impl::ldexp(x, static_cast<int>(e));
}

// adjacent values
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::nextafter(const f128_s& from, const f128_s& to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (from == to)
        return to;
    if (iszero(from))
        return signbit(to)
            ? f128_s{ -std::numeric_limits<double>::denorm_min(), 0.0 }
            : f128_s{  std::numeric_limits<double>::denorm_min(), 0.0 };
    if (isinf(from))
        return signbit(from)
            ? -std::numeric_limits<f128_s>::max()
            :  std::numeric_limits<f128_s>::max();

    const double toward = (from < to)
        ? std::numeric_limits<double>::infinity()
        : -std::numeric_limits<double>::infinity();

    return renorm(
        from.hi,
        detail::fp::nextafter(from.lo, toward)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::nexttoward(const f128_s& from, long double to) noexcept
{
    return detail::_f128_impl::nextafter(from, f128_s{ static_cast<double>(to) });
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::nexttoward(const f128_s& from, const f128_s& to) noexcept
{
    return detail::_f128_impl::nextafter(from, to);
}

// decimal rounding
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::round_to_decimals(f128_s v, int prec)
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

    f128_s ip   = detail::_f128_impl::floor(v);
    f128_s frac = sub_inline(v, ip);

    f128_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = mul_inline(w, f128_s{ 10.0 });

        int di = static_cast<int>(detail::_f128_impl::floor(w).hi);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = sub_inline(w, f128_s{ static_cast<double>(di) });
    }

    f128_s la = mul_inline(w, f128_s{ 10.0 });

    const f128_s tie_slop = mul_inline(f128_s::eps(), f128_s{ 65536.0 });
    int next = static_cast<int>(detail::_f128_impl::floor(la).hi);
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

// remainders
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_impl::fmod(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f128_s ax = detail::_f128::mag(x);
    const f128_s ay = detail::_f128::mag(y);

    if (ax < ay)
        return x;

    f128_s fast{};
    if (y.lo == 0.0 && fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ signbit(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return fmod_exact_fixed_limb(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::remainder(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s ay   = detail::_f128::mag(y);
    f128_s r = detail::_f128_impl::fmod(x, y);
    const f128_s ar   = detail::_f128::mag(r);
    const f128_s half = mul_inline(ay, f128_s{ 0.5 });

    if (ar > half)
    {
        r = add_inline(r, signbit(r) ? ay : -ay);
    }
    else if (ar == half)
    {
        const f128_s q = detail::_f128_impl::trunc(div_inline(x, y));
        const f128_s q_mod2 = detail::_f128::mag(detail::_f128_impl::fmod(q, f128_s{ 2.0 }));
        if (q_mod2 != f128_s{ 0.0 })
            r = add_inline(r, signbit(r) ? ay : -ay);
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::remquo(const f128_s& x, const f128_s& y, int* quo)
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
        const f128_s qbits = detail::_f128_impl::fmod(detail::_f128::mag(n), f128_s{ 2147483648.0 });
        int bits = static_cast<int>(detail::_f128_impl::trunc(qbits).hi);
        if (signbit(n))
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(r);
}

// fractional decomposition
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::modf(const f128_s& x, f128_s* iptr) noexcept
{
    const f128_s i = detail::_f128_impl::trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
}

} // namespace bl

#endif
