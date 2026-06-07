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
#include "fltx/detail/simd.h"

namespace bl {

namespace detail::_f128
{
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s hypot_sqrt_sum(const f128_s& sum)
    {
        BL_CONSTEXPR_RUNTIME_DISPATCH(
            detail::_f128_impl::sqrt(sum),
            detail::_f128::sqrt_compensated(sum, std::sqrt(sum.hi))
        );
    }
}

namespace detail::_f128
{
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s nearbyint_generic(const f128_s& a)
    {
        if (detail::fp::iszero_or_inf_or_nan(a.hi))
            return a;

        if (detail::_f128::absd(a.hi) < 0x1p52)
        {
            const auto base = static_cast<long long>(a.hi);
            const double base_d = static_cast<double>(base);
            const double frac_hi = a.hi - base_d;
            const double frac_lo = a.lo;
            const double abs_frac_hi = detail::_f128::absd(frac_hi);
            const double abs_frac_lo = detail::_f128::absd(frac_lo);

            long long rounded = base;
            if (abs_frac_hi > 0.5 + abs_frac_lo)
            {
                rounded += (frac_hi < 0.0 || (frac_hi == 0.0 && detail::_f128::signbit(frac_lo))) ? -1 : 1;
            }
            else if (abs_frac_hi >= 0.5 - abs_frac_lo)
            {
                const f128_s frac = sub_double_inline(a, base_d);
                const f128_s abs_frac = detail::_f128::mag(frac);
                if (abs_frac > f128_s{ 0.5 } || (abs_frac == f128_s{ 0.5 } && (base & 1ll) != 0))
                    rounded += signbit(frac) ? -1 : 1;
            }

            f128_s out{ static_cast<double>(rounded), 0.0 };
            if (iszero(out))
                return f128_s{ signbit(a) ? -0.0 : 0.0, 0.0 };
            return out;
        }

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
}

namespace detail::_f128_impl
{
    [[nodiscard]] BL_FORCE_INLINE f128_s round_runtime(const f128_s& a) noexcept
    {
        #if defined(__EMSCRIPTEN__)
        if (detail::fp::isinf_or_nan(a.hi)) [[unlikely]]
            return a;

        if (detail::_f128::absd(a.hi) < detail::fp::double_integer_threshold)
        {
            double rounded = static_cast<double>(static_cast<long long>(detail::fp::absd(a.hi) + 0.5));
            if (detail::fp::signbit(a.hi))
                rounded = -rounded;

            const double delta = rounded - a.hi;
            if ((delta == 0.5 && a.lo < 0.0) || (delta == -0.5 && a.lo > 0.0))
                rounded += (rounded < 0.0) ? 1.0 : -1.0;

            if (rounded == 0.0)
                return f128_s{ bl::signbit(a) ? -0.0 : 0.0, 0.0 };
            return f128_s{ rounded, 0.0 };
        }
        #endif

        double rounded = std::round(a.hi);
        if (rounded == a.hi)
        {
            const double rounded_lo = std::round(a.lo);
            if (rounded_lo != 0.0)
            {
                double hi{};
                double lo{};
                detail::fp::quick_two_sum_precise(rounded, rounded_lo, hi, lo);
                return f128_s{ hi, lo };
            }
            return f128_s{ rounded, 0.0 };
        }

        const double delta = rounded - a.hi;
        if ((delta == 0.5 && a.lo < 0.0) || (delta == -0.5 && a.lo > 0.0))
            rounded += (rounded < 0.0) ? 1.0 : -1.0;

        if (rounded == 0.0)
            return f128_s{ bl::signbit(a) ? -0.0 : 0.0, 0.0 };
        return f128_s{ rounded, 0.0 };
    }

    [[nodiscard]] BL_FORCE_INLINE f128_s nearbyint_runtime(const f128_s& a) noexcept
    {
        if (detail::_f128::absd(a.hi) < detail::fp::double_integer_threshold)
        {
            #if BL_FLTX_HAS_SSE2
            double rounded = static_cast<double>(_mm_cvtsd_si64(_mm_set_sd(a.hi)));
            #elif defined(__EMSCRIPTEN__)
            double rounded = std::nearbyint(a.hi);
            #else
            double rounded = std::round(a.hi);
            #endif

            const double delta = rounded - a.hi;

            #if !BL_FLTX_HAS_SSE2 && !defined(__EMSCRIPTEN__)
            if (delta == 0.5)
            {
                if (a.lo == 0.0 && detail::fp::double_integer_is_odd(rounded))
                    rounded -= 1.0;
            }
            else if (delta == -0.5)
            {
                if (a.lo == 0.0 && detail::fp::double_integer_is_odd(rounded))
                    rounded += 1.0;
            }
            #endif

            if (delta == 0.5 && a.lo < 0.0)
                rounded -= 1.0;
            else if (delta == -0.5 && a.lo > 0.0)
                rounded += 1.0;

            if (rounded == 0.0)
                return f128_s{ bl::signbit(a) ? -0.0 : 0.0, 0.0 };
            return f128_s{ rounded, 0.0 };
        }

        return detail::_f128_runtime::nearbyint_slow(a);
    }
}

// roots
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::sqrt(f128_s a)
{
    if (detail::fp::iszero_or_negative_or_inf_or_nan(a.hi)) [[unlikely]]
    {
        if (a.hi == 0.0 && a.lo == 0.0)
            return a;

        if (detail::fp::isposinf(a.hi))
            return a;

        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    constexpr double fast_min = 0x1p-900;
    constexpr double fast_max = 0x1p900;

    if (bl::use_constexpr_math() || a.hi < fast_min || a.hi > fast_max)
    {
        const int exp2 = detail::fp::frexp_exponent_limb(a.hi);
        const int result_scale = exp2 / 2;
        const int input_scale = -2 * result_scale;
        const f128_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

        double seed{};
        if (bl::use_constexpr_math())
            seed = detail::_f128::sqrt_constexpr_head(scaled_a.hi);
        else
            seed = std::sqrt(scaled_a.hi);

        f128_s y = detail::_f128::sqrt_compensated(scaled_a, seed);

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return F128_CANONICALIZE_MATH_RESULT(y);
    }

    return F128_CANONICALIZE_MATH_RESULT(detail::_f128::sqrt_compensated(a, std::sqrt(a.hi)));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::hypot(const f128_s& x, const f128_s& y)
{
    using namespace detail::_f128;

    if (detail::fp::isinf_or_nan(x.hi, y.hi)) [[unlikely]]
    {
        if (detail::fp::isinf(x.hi, y.hi))
            return std::numeric_limits<f128_s>::infinity();
        return detail::fp::isnan(x.hi) ? x : y;
    }

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

    if (ay.hi <= ax.hi * 0x1p-55)
        return F128_CANONICALIZE_MATH_RESULT(ax);

    if (ax.hi > 0x1p-500 && ax.hi < 0x1p500)
    {
        const f128_s sum = add_inline(sqr_dd_inline(ax), sqr_dd_inline(ay));
        return F128_CANONICALIZE_MATH_RESULT(hypot_sqrt_sum(sum));
    }

    const f128_s r = div_inline(ay, ax);
    return F128_CANONICALIZE_MATH_RESULT(mul_inline(ax, detail::_f128_impl::sqrt(add_inline(f128_s{ 1.0 }, mul_inline(r, r)))));
}

// rounding and decimals
[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::floor_limb(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::fp::floor(x),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double detail::_f128_impl::ceil_limb(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::fp::ceil(x),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::floor(const f128_s& a)
{
    double hi = detail::_f128_impl::floor_limb(a.hi);
    double lo = 0.0;

    if (!detail::fp::isfinite(hi))
        return f128_s{ hi, 0.0 };

    if (hi == a.hi)
    {
        lo = detail::_f128_impl::floor_limb(a.lo);
        const f128_s out = detail::_f128::renorm(hi, lo);
        return out.hi == 0.0 ? detail::_f128::signed_zero(detail::fp::signbit(a.hi)) : out;
    }

    return f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::ceil(const f128_s& a)
{
    double hi = detail::_f128_impl::ceil_limb(a.hi);
    double lo = 0.0;

    if (!detail::fp::isfinite(hi))
        return f128_s{ hi, 0.0 };

    if (hi == a.hi)
    {
        lo = detail::_f128_impl::ceil_limb(a.lo);
        const f128_s out = detail::_f128::renorm(hi, lo);
        return out.hi == 0.0 ? detail::_f128::signed_zero(detail::fp::signbit(a.hi)) : out;
    }

    return hi == 0.0 ? detail::_f128::signed_zero(detail::fp::signbit(a.hi)) : f128_s{ hi, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::trunc(const f128_s& a)
{
    if (detail::fp::iszero_or_inf_or_nan(a.hi)) [[unlikely]]
        return a;

    if (detail::_f128::absd(a.hi) < detail::fp::double_integer_threshold)
    {
        double hi = static_cast<double>(static_cast<long long>(a.hi));
        if (hi == a.hi)
        {
            if (a.hi > 0.0 && a.lo < 0.0)
                hi -= 1.0;
            else if (a.hi < 0.0 && a.lo > 0.0)
                hi += 1.0;
        }

        if (hi == 0.0)
            return f128_s{ signbit(a) ? -0.0 : 0.0, 0.0 };
        return f128_s{ hi, 0.0 };
    }

    return (a.hi < 0.0) ? detail::_f128_impl::ceil(a) : detail::_f128_impl::floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::round(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        round_half_away_zero(a),
        detail::_f128_impl::round_runtime(a)
    );
}

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

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::nearbyint(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128::nearbyint_generic(a),
        detail::_f128_impl::nearbyint_runtime(a)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::rint(const f128_s& x)
{
    return detail::_f128_impl::nearbyint(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_impl::lround(const f128_s& x)
{
    long out = 0;
    if (detail::_f128::try_round_to_signed_integer(x, false, out))
        return out;

    return to_signed_integer_or_zero<long>(round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_impl::llround(const f128_s& x)
{
    long long out = 0;
    if (detail::_f128::try_round_to_signed_integer(x, false, out))
        return out;

    return to_signed_integer_or_zero<long long>(round_half_away_zero(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long detail::_f128_impl::lrint(const f128_s& x)
{
    long out = 0;
    if (detail::_f128::try_round_to_signed_integer(x, true, out))
        return out;

    return to_signed_integer_or_zero<long>(detail::_f128_impl::nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long detail::_f128_impl::llrint(const f128_s& x)
{
    long long out = 0;
    if (detail::_f128::try_round_to_signed_integer(x, true, out))
        return out;

    return to_signed_integer_or_zero<long long>(detail::_f128_impl::nearbyint(x));
}

// arithmetic and comparisons
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    if (detail::fp::isinf_or_nan(x.hi) || detail::fp::isinf_or_nan(y.hi) || detail::fp::isinf_or_nan(z.hi)) [[unlikely]]
        return f128_s{ std::fma(x.hi, y.hi, z.hi), 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(add_inline(mul_dd_inline(x, y), z));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fmin(const f128_s& a, const f128_s& b)
{
    if (a.hi != b.hi)
    {
        if (detail::fp::isnan(b.hi)) [[unlikely]]
            return a;
        return a.hi < b.hi ? a : b;
    }

    if (a.lo != b.lo)
        return a.lo < b.lo ? a : b;

    if (a.hi == 0.0 && a.lo == 0.0)
        return detail::_f128::signbit(a.hi) ? a : b;

    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fmax(const f128_s& a, const f128_s& b)
{
    if (a.hi != b.hi)
    {
        if (detail::fp::isnan(b.hi)) [[unlikely]]
            return a;
        return a.hi > b.hi ? a : b;
    }

    if (a.lo != b.lo)
        return a.lo > b.lo ? a : b;

    if (a.hi == 0.0 && a.lo == 0.0)
        return detail::_f128::signbit(a.hi) ? b : a;

    return a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fdim(const f128_s& x, const f128_s& y)
{
    if (!detail::fp::isinf_or_nan(x.hi) && !detail::fp::isinf_or_nan(y.hi))
    {
        const bool x_greater_y = (x.hi > y.hi) || (x.hi == y.hi && x.lo > y.lo);
        return x_greater_y ? F128_CANONICALIZE_MATH_RESULT(sub_inline(x, y)) : f128_s{ 0.0 };
    }

    if (detail::fp::isnan(x.hi) || detail::fp::isnan(y.hi))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(x))
    {
        if (signbit(x))
            return f128_s{ 0.0 };
        return (isinf(y) && !signbit(y)) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();
    }
    if (isinf(y))
        return signbit(y) ? std::numeric_limits<f128_s>::infinity() : f128_s{ 0.0 };

    return (x > y) ? F128_CANONICALIZE_MATH_RESULT(sub_inline(x, y)) : f128_s{ 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::copysign(const f128_s& x, const f128_s& y)
{
    return detail::_f128::signbit(x.hi) == detail::_f128::signbit(y.hi) ? x : -x;
}

// remainders
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::fmod(const f128_s& x, const f128_s& y)
{
    if (detail::fp::isinf_or_nan(x.hi) || detail::fp::iszero_or_nan(y.hi))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (detail::fp::isinf(y.hi) || x.hi == 0.0)
        return x;

    const f128_s ax = detail::_f128::mag(x);
    const f128_s ay = detail::_f128::mag(y);

    if (ax < ay)
        return x;

    f128_s fast{};
    if (fmod_fast_small_quotient_abs(ax, ay, fast))
    {
        if (iszero(fast))
            return f128_s{ signbit(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return fmod_reduced_or_exact(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::remainder(const f128_s& x, const f128_s& y)
{
    return detail::_f128_impl::remquo(x, y, nullptr);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::remquo(const f128_s& x, const f128_s& y, int* quo)
{
    using namespace detail::_f128;

    if (quo)
        *quo = 0;

    if (detail::fp::isinf_or_nan(x.hi) || detail::fp::iszero_or_nan(y.hi))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (detail::fp::isinf(y.hi) || x.hi == 0.0)
        return x;

    const bool x_negative = signbit(x);
    const bool quotient_negative = x_negative != signbit(y);
    const f128_s ax = mag(x);
    const f128_s ay = mag(y);

    f128_s r_abs{};
    std::uint64_t quotient_abs = 0;
    bool fast = false;

    if (ax < ay)
    {
        r_abs = ax;
        fast = true;
    }
    else
    {
        fast = fmod_fast_small_quotient_abs_with_quotient(ax, ay, r_abs, quotient_abs, true);
    }

    if (fast)
    {
        const f128_s half = mul_double_inline(ay, 0.5);
        const int half_cmp = detail::_f128::fmod_compare_remainder_to_half(r_abs, half);
        if (half_cmp > 0 || (half_cmp == 0 && ((quotient_abs & 1u) != 0u)))
        {
            r_abs = sub_inline(r_abs, ay);
            ++quotient_abs;
        }

        if (quo)
            *quo = detail::fp::remquo_low_quotient_bits(quotient_abs, quotient_negative);

        f128_s r = x_negative ? -r_abs : r_abs;
        if (iszero(r))
            return f128_s{ x_negative ? -0.0 : 0.0, 0.0 };

        return F128_CANONICALIZE_MATH_RESULT(r);
    }

    std::uint64_t quotient_mod = 0;
    r_abs = fmod_exact_fixed_limb_abs_with_quotient_mod(ax, ay, quotient_mod);
    const f128_s half = mul_double_inline(ay, 0.5);
    const int half_cmp = detail::_f128::fmod_compare_remainder_to_half(r_abs, half);

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

    f128_s r = x_negative ? -r_abs : r_abs;
    if (iszero(r))
        return f128_s{ x_negative ? -0.0 : 0.0, 0.0 };

    return F128_CANONICALIZE_MATH_RESULT(r);
}

// fractional decomposition
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::modf(const f128_s& x, f128_s* iptr) noexcept
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
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    }

    const f128_s i = detail::_f128_impl::trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
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

    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ull;
    constexpr std::uint64_t fraction_mask = 0x000fffffffffffffull;
    constexpr std::uint64_t sign_mask     = 0x8000000000000000ull;

    const std::uint64_t hi_bits = std::bit_cast<std::uint64_t>(x.hi);
    const std::uint32_t hi_exponent =
        static_cast<std::uint32_t>((hi_bits & exponent_mask) >> 52);

    if (hi_exponent == 0x7ffu || ((hi_bits & ~sign_mask) == 0u && x.lo == 0.0))
        return x;

    int e = 0;
    double scaled_hi = 0.0;
    double scaled_lo = 0.0;

    if (hi_exponent != 0u && hi_exponent != 0x7ffu)
    {
        e = static_cast<int>(hi_exponent) - 1022;
        scaled_hi = std::bit_cast<double>(
            (hi_bits & (sign_mask | fraction_mask)) |
            (std::uint64_t{ 1022 } << 52));

        const std::uint64_t lo_bits = std::bit_cast<std::uint64_t>(x.lo);
        const std::uint32_t lo_exponent =
            static_cast<std::uint32_t>((lo_bits & exponent_mask) >> 52);
        const int scaled_lo_exponent = static_cast<int>(lo_exponent) - e;

        if (x.lo == 0.0)
        {
            scaled_lo = x.lo;
        }
        else if (lo_exponent != 0u && lo_exponent != 0x7ffu &&
                 scaled_lo_exponent > 0 && scaled_lo_exponent < 0x7ff)
        {
            scaled_lo = std::bit_cast<double>(
                (lo_bits & (sign_mask | fraction_mask)) |
                (static_cast<std::uint64_t>(scaled_lo_exponent) << 52));
        }
        else
        {
            scaled_lo = detail::fp::ldexp(x.lo, -e);
        }

        if ((scaled_hi == 0.5 && scaled_lo < 0.0) ||
            (scaled_hi == -0.5 && scaled_lo > 0.0))
        {
            scaled_hi *= 2.0;
            scaled_lo *= 2.0;
            --e;
        }

        if (exp)
            *exp = e;

        return f128_s{ scaled_hi, scaled_lo };
    }
    else
    {
        const double lead = (x.hi != 0.0) ? x.hi : x.lo;
        e = detail::fp::frexp_exponent(lead);
        scaled_hi = detail::fp::ldexp(x.hi, -e);
        scaled_lo = detail::fp::ldexp(x.lo, -e);
    }

    const double hi = scaled_hi + scaled_lo;
    f128_s m{hi, (scaled_hi - hi) + scaled_lo};
    const double abs_hi = (m.hi < 0.0) ? -m.hi : m.hi;
    const double abs_lo = (m.hi < 0.0) ? -m.lo : m.lo;

    if (abs_hi == 0.5 && abs_lo < 0.0)
    {
        m.hi *= 2.0;
        m.lo *= 2.0;
        --e;
    }
    else if (abs_hi == 1.0 && abs_lo >= 0.0)
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

    return ilogb_finite_fast(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::logb(const f128_s& x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    return f128_s{ static_cast<double>(ilogb_finite_fast(x)), 0.0 };
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
    if (detail::fp::isnan(from.hi) || detail::fp::isnan(to.hi))
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

} // namespace bl

#endif
