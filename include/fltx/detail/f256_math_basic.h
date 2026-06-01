/**
 * fltx/detail/f256_math_basic.h - Basic math implementation details.
 *
 * f256 rounding, decomposition, remainder, hypot, and adjacent-value implementations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_DETAIL_MATH_BASIC_INCLUDED
#define F256_DETAIL_MATH_BASIC_INCLUDED
#include "fltx/detail/f256_math_kernels.h"
#include "fltx/detail/simd.h"

namespace bl {

namespace detail::_f256
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double floor_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::fp::floor(x),
            std::floor(x)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double ceil_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::fp::ceil(x),
            -std::floor(-x)
        );
    }
}

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_FORCE_INLINE double trunc_limb(double x) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(x))
            return x;

        const double ax = detail::fp::absd(x);
        if (ax >= detail::fp::double_integer_threshold)
            return x;

        const double out = static_cast<double>(static_cast<long long>(x));
        return out == 0.0 ? (detail::fp::signbit(x) ? -0.0 : 0.0) : out;
    }
}

namespace detail::_f256
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double trunc_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::fp::trunc(x),
            detail::_f256_runtime::trunc_limb(x)
        );
    }
}

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_FORCE_INLINE double nearbyint_limb(double x) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(x))
            return x;

        const double t = detail::_f256::floor_limb(x);
        const double frac = x - t;
        double out = t;
        if (frac > 0.5 || (frac == 0.5 && detail::fp::double_integer_is_odd(t)))
            out = t + 1.0;

        return out == 0.0 ? (detail::fp::signbit(x) ? -0.0 : 0.0) : out;
    }
}

namespace detail::_f256
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double nearbyint_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::fp::nearbyint(x),
            detail::_f256_runtime::nearbyint_limb(x)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE double nearbyint_limb_finite_small(double x) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return static_cast<double>(_mm_cvtsd_si64(_mm_set_sd(x)));
        #else
        return nearbyint_limb(x);
        #endif
    }
}

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_FORCE_INLINE double round_half_away_zero_limb(double x) noexcept
    {
        if (detail::fp::iszero_or_inf_or_nan(x))
            return x;

        const double ax = detail::fp::absd(x);
        if (ax >= detail::fp::double_integer_threshold)
            return x;

        double out = static_cast<double>(static_cast<long long>(ax + 0.5));
        if (detail::fp::signbit(x))
            out = -out;
        return out == 0.0 ? (detail::fp::signbit(x) ? -0.0 : 0.0) : out;
    }
}

namespace detail::_f256
{
    [[nodiscard]] BL_FORCE_INLINE constexpr double round_half_away_zero_limb(double x) noexcept
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::fp::round_half_away_zero(x),
            detail::_f256_runtime::round_half_away_zero_limb(x)
        );
    }

    [[nodiscard]] BL_FORCE_INLINE double round_half_away_zero_limb_finite_small(double x) noexcept
    {
        double out = static_cast<double>(static_cast<long long>(detail::fp::absd(x) + 0.5));
        if (detail::fp::signbit(x))
            out = -out;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_negative_tail(double x1, double x2, double x3) noexcept
    {
        return x1 < 0.0 || (x1 == 0.0 && (x2 < 0.0 || (x2 == 0.0 && x3 < 0.0)));
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool has_positive_tail(double x1, double x2, double x3) noexcept
    {
        return x1 > 0.0 || (x1 == 0.0 && (x2 > 0.0 || (x2 == 0.0 && x3 > 0.0)));
    }

    BL_FORCE_INLINE constexpr void adjust_rounded_limb_for_tail(
        double& rounded,
        double value,
        double next1,
        double next2,
        double next3) noexcept
    {
        const double delta = rounded - value;
        if (detail::fp::absd(delta) != 0.5)
            return;

        if (delta > 0.0)
        {
            if (has_negative_tail(next1, next2, next3))
                rounded -= 1.0;
        }
        else if (has_positive_tail(next1, next2, next3))
        {
            rounded += 1.0;
        }
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s signed_zero_like(const f256_s& a) noexcept
    {
        return signed_zero_from(a.x0);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s canonicalize_rounded_zero(f256_s out, const f256_s& a) noexcept
    {
        return out.x0 == 0.0 ? signed_zero_like(a) : out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s floor_limbwise(const f256_s& a) noexcept
    {
        double x0 = floor_limb(a.x0);
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;

        if (!detail::fp::isfinite(x0))
            return f256_s{ x0, 0.0, 0.0, 0.0 };

        if (x0 == a.x0)
        {
            x1 = floor_limb(a.x1);
            if (x1 == a.x1)
            {
                x2 = floor_limb(a.x2);
                if (x2 == a.x2)
                    x3 = floor_limb(a.x3);
            }

            return canonicalize_rounded_zero(renorm(x0, x1, x2, x3), a);
        }

        return f256_s{ x0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s ceil_limbwise(const f256_s& a) noexcept
    {
        double x0 = ceil_limb(a.x0);
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;

        if (!detail::fp::isfinite(x0))
            return f256_s{ x0, 0.0, 0.0, 0.0 };

        if (x0 == a.x0)
        {
            x1 = ceil_limb(a.x1);
            if (x1 == a.x1)
            {
                x2 = ceil_limb(a.x2);
                if (x2 == a.x2)
                    x3 = ceil_limb(a.x3);
            }

            return canonicalize_rounded_zero(renorm(x0, x1, x2, x3), a);
        }

        return x0 == 0.0 ? signed_zero_like(a) : f256_s{ x0 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s trunc_limbwise(const f256_s& a) noexcept
    {
        const double x0 = trunc_limb(a.x0);

        if (!detail::fp::isfinite(x0))
            return f256_s{ x0, 0.0, 0.0, 0.0 };

        if (x0 != a.x0)
            return x0 == 0.0 ? signed_zero_like(a) : f256_s{ x0 };

        if (a.x0 == 0.0)
            return signed_zero_like(a);

        return a.x0 < 0.0 ? ceil_limbwise(a) : floor_limbwise(a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s nearbyint_limbwise(const f256_s& a) noexcept
    {
        double x0 = nearbyint_limb(a.x0);
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;

        if (x0 == a.x0)
        {
            x1 = nearbyint_limb(a.x1);
            if (x1 == a.x1)
            {
                x2 = nearbyint_limb(a.x2);
                if (x2 == a.x2)
                    x3 = nearbyint_limb(a.x3);
                else
                    adjust_rounded_limb_for_tail(x2, a.x2, a.x3, 0.0, 0.0);
            }
            else
                adjust_rounded_limb_for_tail(x1, a.x1, a.x2, a.x3, 0.0);
        }
        else
        {
            adjust_rounded_limb_for_tail(x0, a.x0, a.x1, a.x2, a.x3);
            return x0 == 0.0 ? signed_zero_like(a) : f256_s{ x0 };
        }

        return canonicalize_rounded_zero(renorm(x0, x1, x2, x3), a);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s round_half_away_zero_limbwise(const f256_s& a) noexcept
    {
        double x0 = round_half_away_zero_limb(a.x0);
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;

        if (x0 == a.x0)
        {
            x1 = round_half_away_zero_limb(a.x1);
            if (x1 == a.x1)
            {
                x2 = round_half_away_zero_limb(a.x2);
                if (x2 == a.x2)
                    x3 = round_half_away_zero_limb(a.x3);
                else
                    adjust_rounded_limb_for_tail(x2, a.x2, a.x3, 0.0, 0.0);
            }
            else
                adjust_rounded_limb_for_tail(x1, a.x1, a.x2, a.x3, 0.0);
        }
        else
        {
            adjust_rounded_limb_for_tail(x0, a.x0, a.x1, a.x2, a.x3);
            return f256_s{ x0 };
        }

        return canonicalize_rounded_zero(renorm(x0, x1, x2, x3), a);
    }

}

namespace detail::_f256_impl
{
    [[nodiscard]] BL_FORCE_INLINE f256_s round_runtime(const f256_s& a) noexcept
    {
        if (detail::fp::isinf_or_nan(a.x0)) [[unlikely]]
            return a;

        if (detail::fp::absd(a.x0) < detail::fp::double_integer_threshold)
        {
            double x0 = detail::_f256::round_half_away_zero_limb_finite_small(a.x0);
            detail::_f256::adjust_rounded_limb_for_tail(x0, a.x0, a.x1, a.x2, a.x3);
            return x0 == 0.0 ? detail::_f256::signed_zero_like(a) : f256_s{ x0 };
        }

        return detail::_f256::round_half_away_zero_limbwise(a);
    }

    [[nodiscard]] BL_FORCE_INLINE f256_s nearbyint_runtime(const f256_s& a) noexcept
    {
        if (detail::fp::absd(a.x0) < detail::fp::double_integer_threshold)
        {
            double x0 = detail::_f256::nearbyint_limb_finite_small(a.x0);

            const double delta = x0 - a.x0;
            if (delta == 0.5)
            {
                if (detail::_f256::has_negative_tail(a.x1, a.x2, a.x3))
                    x0 -= 1.0;
            }
            else if (delta == -0.5 && detail::_f256::has_positive_tail(a.x1, a.x2, a.x3))
            {
                x0 += 1.0;
            }

            return x0 == 0.0 ? detail::_f256::signed_zero_like(a) : f256_s{ x0 };
        }

        if (detail::fp::iszero_or_inf_or_nan(a.x0))
            return a;

        return detail::_f256::nearbyint_limbwise(a);
    }
}

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sqrt(const f256_s& a)
{
    using namespace detail::_f256;

    if (a.x0 <= 0.0)
    {
        if (iszero(a))
            return a;
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    if (isinf(a))
        return a;

    return sqrt_impl_fast(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::sqrt_accurate(const f256_s& a)
{
    using namespace detail::_f256;

    if (a.x0 <= 0.0)
    {
        if (iszero(a))
            return a;
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    if (isinf(a))
        return a;

    return sqrt_impl(a);
}

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

    const int ex = detail::fp::frexp_exponent_limb(ax.x0);
    const int ey = detail::fp::frexp_exponent_limb(ay.x0);

    if ((ex - ey) > 110)
        return F256_CANONICALIZE_MATH_RESULT(ax);

    if (ex > -450 && ex < 450)
        return F256_CANONICALIZE_MATH_RESULT(detail::_f256_impl::sqrt_accurate(add_raw5_raw5_inline(sqr_raw5_inline(ax), sqr_raw5_inline(ay))));

    const f256_s r = div_inline(ay, ax);
    return F256_CANONICALIZE_MATH_RESULT(mul_inline(ax, detail::_f256_impl::sqrt_accurate(add_raw5_value_inline(sqr_raw5_inline(r), f256_s{ 1.0 }))));
}

// rounding and decimals
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::floor(const f256_s& a)
{
    return detail::_f256::floor_limbwise(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::ceil(const f256_s& a)
{
    return -detail::_f256_impl::floor(-a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::trunc(const f256_s& a)
{
    if (detail::fp::isinf_or_nan(a.x0)) [[unlikely]]
        return a;

    return detail::fp::signbit(a.x0) ? detail::_f256_impl::ceil(a) : detail::_f256_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::round(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::round_half_away_zero_limbwise(a),
        detail::_f256_impl::round_runtime(a)
    );
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

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nearbyint(const f256_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256::nearbyint_limbwise(a),
        detail::_f256_impl::nearbyint_runtime(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::rint(const f256_s& x)
{
    return detail::_f256_impl::nearbyint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f256_impl::lround(const f256_s& x)
{
    long out = 0;
    if (detail::_f256::try_round_to_signed_integer(x, false, out))
        return out;

    return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f256_impl::llround(const f256_s& x)
{
    long long out = 0;
    if (detail::_f256::try_round_to_signed_integer(x, false, out))
        return out;

    return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f256_impl::lrint(const f256_s& x)
{
    long out = 0;
    if (detail::_f256::try_round_to_signed_integer(x, true, out))
        return out;

    return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_impl::nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f256_impl::llrint(const f256_s& x)
{
    long long out = 0;
    if (detail::_f256::try_round_to_signed_integer(x, true, out))
        return out;

    return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_impl::nearbyint(x));
}

// arithmetic and comparisons
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    if (detail::fp::isinf_or_nan(x.x0) || detail::fp::isinf_or_nan(y.x0) || detail::fp::isinf_or_nan(z.x0)) [[unlikely]]
        return f256_s{ std::fma(x.x0, y.x0, z.x0), 0.0, 0.0, 0.0 };

    return F256_CANONICALIZE_MATH_RESULT(mul_add_inline(x, y, z));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmin(const f256_s& a, const f256_s& b)
{
    if (a.x0 != b.x0)
    {
        if (detail::fp::isnan(a.x0)) [[unlikely]]
            return b;
        if (detail::fp::isnan(b.x0)) [[unlikely]]
            return a;
        return a.x0 < b.x0 ? a : b;
    }

    if (a.x1 != b.x1)
        return a.x1 < b.x1 ? a : b;
    if (a.x2 != b.x2)
        return a.x2 < b.x2 ? a : b;
    if (a.x3 != b.x3)
        return a.x3 < b.x3 ? a : b;

    if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
        return detail::fp::signbit(a.x0) ? a : b;

    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmax(const f256_s& a, const f256_s& b)
{
    if (a.x0 != b.x0)
    {
        if (detail::fp::isnan(a.x0)) [[unlikely]]
            return b;
        if (detail::fp::isnan(b.x0)) [[unlikely]]
            return a;
        return a.x0 > b.x0 ? a : b;
    }

    if (a.x1 != b.x1)
        return a.x1 > b.x1 ? a : b;
    if (a.x2 != b.x2)
        return a.x2 > b.x2 ? a : b;
    if (a.x3 != b.x3)
        return a.x3 > b.x3 ? a : b;

    if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
        return detail::fp::signbit(a.x0) ? b : a;

    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fdim(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y)) [[unlikely]]
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(x)) [[unlikely]]
    {
        if (signbit(x))
            return f256_s{ 0.0 };
        return (isinf(y) && !signbit(y)) ? f256_s{ 0.0 } : std::numeric_limits<f256_s>::infinity();
    }
    if (isinf(y)) [[unlikely]]
        return signbit(y) ? std::numeric_limits<f256_s>::infinity() : f256_s{ 0.0 };

    return (x > y) ? F256_CANONICALIZE_MATH_RESULT(x - y) : f256_s{ 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::copysign(const f256_s& x, const f256_s& y)
{
    return bl::signbit(x) == bl::signbit(y) ? x : -x;
}

// remainders
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::fmod(const f256_s& x, const f256_s& y)
{
    if (detail::fp::isinf_or_nan(x.x0) || detail::fp::iszero_or_nan(y.x0))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (detail::fp::isinf(y.x0) || x.x0 == 0.0)
        return x;

    const f256_s ax = detail::_f256::mag(x);
    const f256_s ay = detail::_f256::mag(y);

    if (ax < ay)
        return x;

    f256_s fast{};
    if (fmod_fast_small_quotient_abs(ax, ay, fast))
    {
        if (iszero(fast))
            return detail::_f256::signed_zero_like(x);
        const f256_s out = ispositive(x) ? fast : -fast;
        return F256_CANONICALIZE_MATH_RESULT(out);
    }

    return F256_CANONICALIZE_MATH_RESULT(fmod_reduced_or_exact(x, y));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::remainder(const f256_s& x, const f256_s& y)
{
    return detail::_f256_impl::remquo(x, y, nullptr);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::remquo(const f256_s& x, const f256_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (detail::fp::isinf_or_nan(x.x0) || detail::fp::iszero_or_nan(y.x0))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (detail::fp::isinf(y.x0) || x.x0 == 0.0)
        return x;

    const bool x_negative = signbit(x);
    const bool quotient_negative = x_negative != signbit(y);
    const f256_s ax = detail::_f256::mag(x);
    const f256_s ay = detail::_f256::mag(y);

    f256_s r_abs{};
    std::uint64_t quotient_abs = 0;
    bool fast = false;

    if (ax < ay)
    {
        r_abs = ax;
        fast = true;
    }
    else
    {
        fast = fmod_fast_small_quotient_abs_with_quotient(ax, ay, r_abs, quotient_abs);
    }

    if (fast)
    {
        const f256_s half = mul_double_inline(ay, 0.5);
        const int half_cmp = detail::_f256::fmod_compare_remainder_to_half(r_abs, half);
        if (half_cmp > 0 || (half_cmp == 0 && ((quotient_abs & 1u) != 0u)))
        {
            r_abs = sub_inline(r_abs, ay);
            ++quotient_abs;
        }

        if (quo)
            *quo = detail::fp::remquo_low_quotient_bits(quotient_abs, quotient_negative);

        f256_s r = x_negative ? -r_abs : r_abs;
        if (iszero(r))
            return detail::_f256::signed_zero_like(x);

        return F256_CANONICALIZE_MATH_RESULT(r);
    }

    std::uint64_t quotient_mod = 0;
    r_abs = fmod_exact_abs_with_quotient_mod(ax, ay, quotient_mod);
    const f256_s half = mul_double_inline(ay, 0.5);
    const int half_cmp = detail::_f256::fmod_compare_remainder_to_half(r_abs, half);

    if (half_cmp > 0)
    {
        r_abs = sub_inline(r_abs, ay);
        ++quotient_mod;
    }
    else if (half_cmp == 0 && ((quotient_mod & 1u) != 0u))
    {
        r_abs = sub_inline(r_abs, ay);
        ++quotient_mod;
    }

    if (quo)
        *quo = detail::fp::remquo_low_quotient_bits(quotient_mod, quotient_negative);

    f256_s r = x_negative ? -r_abs : r_abs;
    if (iszero(r))
        return detail::_f256::signed_zero_like(x);

    return F256_CANONICALIZE_MATH_RESULT(r);
}

// fractional decomposition
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::modf(const f256_s& x, f256_s* iptr) noexcept
{
    if (isnan(x))
    {
        if (iptr)
            *iptr = x;
        return x;
    }
    if (isinf(x))
    {
        if (iptr)
            *iptr = x;
        return detail::_f256::signed_zero_like(x);
    }

    const f256_s i = detail::_f256_impl::trunc(x);
    if (iptr)
        *iptr = i;

    f256_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = detail::_f256::signed_zero_like(x);
    return frac;
}

// decomposition and scaling
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::ldexp(const f256_s& a, int e)
{
    if (detail::fp::iszero_or_inf_or_nan(a.x0)) [[unlikely]]
        return a;

    return F256_CANONICALIZE_MATH_RESULT(_ldexp(a, e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::frexp(const f256_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (detail::fp::iszero_or_inf_or_nan(x.x0))
        return x;

    int e = 0;

    if (bl::use_constexpr_math())
    {
        e = detail::fp::frexp_exponent(x.x0);
    }
    else
    {
        (void)std::frexp(x.x0, &e);
    }

    const bool safe_fast_scale =
        detail::fp::absd(x.x0) >= std::numeric_limits<double>::min();
    f256_s m = safe_fast_scale
        ? detail::_f256::_ldexp(x, -e)
        : detail::_f256::ldexp_terms(x, -e);
    if ((m.x0 == 0.5 && detail::_f256::has_negative_tail(m.x1, m.x2, m.x3))
        || (m.x0 == -0.5 && detail::_f256::has_positive_tail(m.x1, m.x2, m.x3)))
    {
        m = safe_fast_scale
            ? detail::_f256::_ldexp(m, 1)
            : detail::_f256::ldexp_terms(m, 1);
        --e;
    }
    else if ((m.x0 == 1.0 && !detail::_f256::has_negative_tail(m.x1, m.x2, m.x3))
        || (m.x0 == -1.0 && !detail::_f256::has_positive_tail(m.x1, m.x2, m.x3)))
    {
        m = safe_fast_scale
            ? detail::_f256::_ldexp(m, -1)
            : detail::_f256::ldexp_terms(m, -1);
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}

[[nodiscard]] BL_FORCE_INLINE constexpr int detail::_f256_impl::ilogb(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_ILOGBNAN;
    if (iszero(x)) return FP_ILOGB0;
    if (isinf(x))  return std::numeric_limits<int>::max();

    const double lead =
        (x.x0 != 0.0) ? x.x0 :
        (x.x1 != 0.0) ? x.x1 :
        (x.x2 != 0.0) ? x.x2 : x.x3;
    return detail::fp::frexp_exponent(detail::fp::absd(lead)) - 1;
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

// adjacent values
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::nextafter(const f256_s& from, const f256_s& to) noexcept
{
    if (detail::fp::isnan(from.x0) || detail::fp::isnan(to.x0))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (from == to)
        return to;
    if (iszero(from))
        return signbit(to)
        ? f256_s{ -std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 }
        : f256_s{ std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 };
    if (isinf(from))
        return signbit(from)
        ? -std::numeric_limits<f256_s>::max()
        : std::numeric_limits<f256_s>::max();

    const double toward = (from < to)
        ? std::numeric_limits<double>::infinity()
        : -std::numeric_limits<double>::infinity();

    return normalize_nextafter_tail(
        from,
        detail::fp::nextafter(from.x3, toward));
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
