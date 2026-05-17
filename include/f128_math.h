/**
 * f128_math.h - constexpr <cmath>-style functions for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_MATH_INCLUDED
#define F128_MATH_INCLUDED

#include "f128.h"
#include "f128_consts.h"
#include "fltx_common_math.h"
#include "fltx_decimal.h"
#include "fltx_math_utils.h"

namespace bl {

namespace detail::_f128_runtime
{
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec);
    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s sqrt(f128_s a);
    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e);

    BL_NO_INLINE f128_s exp(const f128_s& x);
    BL_NO_INLINE f128_s exp2(const f128_s& x);
    BL_NO_INLINE f128_s log(const f128_s& a);
    BL_NO_INLINE f128_s log2(const f128_s& a);
    BL_NO_INLINE f128_s log10(const f128_s& x);
    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s pow(const f128_s& x, double y);

    BL_NO_INLINE bool   sincos(const f128_s& x, f128_s& s_out, f128_s& c_out);
    BL_NO_INLINE f128_s sin(const f128_s& x);
    BL_NO_INLINE f128_s cos(const f128_s& x);
    BL_NO_INLINE f128_s tan(const f128_s& x);
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x);

    BL_NO_INLINE f128_s expm1(const f128_s& x);
    BL_NO_INLINE f128_s log1p(const f128_s& x);
    BL_NO_INLINE f128_s sinh(const f128_s& x);
    BL_NO_INLINE f128_s cosh(const f128_s& x);
    BL_NO_INLINE f128_s tanh(const f128_s& x);
    BL_NO_INLINE f128_s asinh(const f128_s& x);
    BL_NO_INLINE f128_s acosh(const f128_s& x);
    BL_NO_INLINE f128_s atanh(const f128_s& x);

    BL_NO_INLINE f128_s cbrt(const f128_s& x);
    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y);
    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo);
    BL_NO_INLINE f128_s frexp(const f128_s& x, int* exp) noexcept;
    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept;
    BL_NO_INLINE f128_s nextafter(const f128_s& from, const f128_s& to) noexcept;

    BL_NO_INLINE f128_s erf(const f128_s& x);
    BL_NO_INLINE f128_s erfc(const f128_s& x);
    BL_NO_INLINE f128_s lgamma(const f128_s& x);
    BL_NO_INLINE f128_s tgamma(const f128_s& x);
}
namespace detail::_f128_constexpr
{
    using namespace detail::_f128;

    BL_FORCE_INLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s round_to_decimals(f128_s v, int prec);
    BL_FORCE_INLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s sqrt(f128_s a);
    BL_FORCE_INLINE constexpr f128_s ldexp(const f128_s& x, int e);

    BL_FORCE_INLINE constexpr f128_s exp(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s exp2(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s log(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s log2(const f128_s& a);
    BL_FORCE_INLINE constexpr f128_s log10(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s pow(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s pow(const f128_s& x, double y);

    BL_FORCE_INLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out);
    BL_FORCE_INLINE constexpr f128_s sin(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s cos(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tan(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x);

    BL_FORCE_INLINE constexpr f128_s expm1(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s log1p(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s sinh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s cosh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tanh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s asinh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s acosh(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s atanh(const f128_s& x);

    BL_FORCE_INLINE constexpr f128_s cbrt(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y);
    BL_FORCE_INLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo);
    BL_FORCE_INLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept;
    BL_FORCE_INLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept;
    BL_FORCE_INLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept;

    BL_FORCE_INLINE constexpr f128_s erf(const f128_s& x);
    BL_MSVC_NOINLINE constexpr f128_s erfc(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s lgamma(const f128_s& x);
    BL_FORCE_INLINE constexpr f128_s tgamma(const f128_s& x);
}

// forward declare wrappers to runtime/constexpr calls
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s round_to_decimals(f128_s v, int prec);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sqrt(f128_s a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s ldexp(const f128_s& x, int e);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s exp(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s exp2(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log(const f128_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log2(const f128_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log10(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, const f128_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, double y);

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sin(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cos(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tan(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s expm1(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log1p(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sinh(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cosh(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tanh(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s asinh(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s acosh(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s atanh(const f128_s& x);

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cbrt(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept;
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept;

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erf(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erfc(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s lgamma(const f128_s& x);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tgamma(const f128_s& x);

/// ============= Math =============

namespace detail::_f128
{
    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::double_integer_is_odd;
    using detail::fp::fmod_constexpr;
    using detail::fp::sqrt_seed_constexpr;
    using detail::fp::nearbyint_ties_even;

    // frexp / ldexp
    BL_FORCE_INLINE  constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::frexp_exponent_constexpr(value);

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }
    BL_FORCE_INLINE  constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::ldexp_constexpr2(value, exponent);

        return std::ldexp(value, exponent);
    }
    BL_FORCE_INLINE  constexpr f128_s ldexp_terms(const f128_s& value, int exponent) noexcept
    {
        return renorm(
            ldexp_limb(value.hi, exponent),
            ldexp_limb(value.lo, exponent));
    }
    BL_FORCE_INLINE  constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        if (bl::use_constexpr_math())
        {
            return renorm(
                detail::fp::ldexp_constexpr2(x.hi, e),
                detail::fp::ldexp_constexpr2(x.lo, e)
            );
        }
        else
        {
            return renorm(
                std::ldexp(x.hi, e),
                std::ldexp(x.lo, e)
            );
        }
    }

    // fmod
    struct fmod_u128
    {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
    };
    struct exact_dyadic_fmod
    {
        bool neg = false;
        int exp2 = 0;
        fmod_u128 mant{};
    };
    BL_FORCE_INLINE  constexpr bool fmod_u128_is_zero(const fmod_u128& value)
    {
        return value.lo == 0 && value.hi == 0;
    }
    BL_FORCE_INLINE  constexpr bool fmod_u128_is_odd(const fmod_u128& value)
    {
        return (value.lo & 1u) != 0;
    }
    BL_FORCE_INLINE  constexpr int  fmod_u128_compare(const fmod_u128& a, const fmod_u128& b)
    {
        if (a.hi < b.hi) return -1;
        if (a.hi > b.hi) return 1;
        if (a.lo < b.lo) return -1;
        if (a.lo > b.lo) return 1;
        return 0;
    }
    BL_FORCE_INLINE  constexpr int  fmod_u128_bit_length(const fmod_u128& value)
    {
        if (value.hi != 0)
            return 128 - static_cast<int>(std::countl_zero(value.hi));
        if (value.lo != 0)
            return 64 - static_cast<int>(std::countl_zero(value.lo));
        return 0;
    }
    BL_FORCE_INLINE  constexpr int  fmod_u128_trailing_zero_bits(const fmod_u128& value)
    {
        if (value.lo != 0)
            return static_cast<int>(std::countr_zero(value.lo));
        if (value.hi != 0)
            return 64 + static_cast<int>(std::countr_zero(value.hi));
        return 0;
    }
    BL_FORCE_INLINE  constexpr bool fmod_u128_get_bit(const fmod_u128& value, int index)
    {
        if (index < 0 || index >= 128)
            return false;
        if (index < 64)
            return ((value.lo >> index) & 1u) != 0;
        return ((value.hi >> (index - 64)) & 1u) != 0;
    }
    BL_FORCE_INLINE  constexpr std::uint64_t fmod_u128_get_bits(const fmod_u128& value, int start, int count)
    {
        std::uint64_t out = 0;
        for (int i = 0; i < count; ++i)
        {
            if (fmod_u128_get_bit(value, start + i))
                out |= (std::uint64_t{ 1 } << i);
        }
        return out;
    }
    BL_FORCE_INLINE  constexpr bool fmod_u128_any_low_bits_set(const fmod_u128& value, int count)
    {
        if (count <= 0)
            return false;

        if (count >= 64)
        {
            if (value.lo != 0)
                return true;
            count -= 64;
            if (count >= 64)
                return value.hi != 0;
            return (value.hi & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
        }

        return (value.lo & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
    }
    BL_FORCE_INLINE  constexpr void fmod_u128_add_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += b.lo;
        a.hi += b.hi + (a.lo < old_lo ? 1u : 0u);
    }
    BL_FORCE_INLINE  constexpr void fmod_u128_add_small(fmod_u128& a, std::uint32_t value)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += value;
        if (a.lo < old_lo)
            ++a.hi;
    }
    BL_FORCE_INLINE  constexpr void fmod_u128_sub_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t borrow = (a.lo < b.lo) ? 1u : 0u;
        a.lo -= b.lo;
        a.hi -= b.hi + borrow;
    }
    BL_FORCE_INLINE  constexpr fmod_u128 fmod_u128_shl_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { 0, value.lo << (bits - 64) };

        return {
            value.lo << bits,
            (value.hi << bits) | (value.lo >> (64 - bits))
        };
    }
    BL_FORCE_INLINE  constexpr fmod_u128 fmod_u128_shr_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { value.hi >> (bits - 64), 0 };

        return {
            (value.lo >> bits) | (value.hi << (64 - bits)),
            value.hi >> bits
        };
    }
    BL_FORCE_INLINE  constexpr fmod_u128 fmod_u128_shl1(fmod_u128 value)
    {
        return { value.lo << 1, (value.hi << 1) | (value.lo >> 63) };
    }
    BL_FORCE_INLINE  constexpr fmod_u128 fmod_u128_mod_shift_subtract(fmod_u128 numerator, const fmod_u128& denominator)
    {
        if (fmod_u128_is_zero(denominator))
            return {};
        if (fmod_u128_compare(numerator, denominator) < 0)
            return numerator;

        int shift = fmod_u128_bit_length(numerator) - fmod_u128_bit_length(denominator);
        fmod_u128 shifted = fmod_u128_shl_bits(denominator, shift);

        for (; shift >= 0; --shift)
        {
            if (fmod_u128_compare(numerator, shifted) >= 0)
                fmod_u128_sub_inplace(numerator, shifted);
            shifted = fmod_u128_shr_bits(shifted, 1);
        }

        return numerator;
    }
    BL_FORCE_INLINE  constexpr fmod_u128 fmod_u128_double_mod(fmod_u128 value, const fmod_u128& modulus)
    {
        value = fmod_u128_shl1(value);
        if (fmod_u128_compare(value, modulus) >= 0)
            fmod_u128_sub_inplace(value, modulus);
        return value;
    }
    BL_FORCE_INLINE  constexpr exact_dyadic_fmod exact_from_double_fmod(double value)
    {
        exact_dyadic_fmod out;
        if (value == 0.0)
            return out;

        int exponent = 0;
        bool neg = false;
        const std::uint64_t mantissa = detail::exact_decimal::decompose_double_mantissa(value, exponent, neg);
        if (mantissa == 0)
            return out;

        out.neg = neg;
        out.exp2 = exponent;
        out.mant.lo = mantissa;
        return out;
    }
    BL_FORCE_INLINE  constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
    {
        if (fmod_u128_is_zero(value.mant))
        {
            value.neg = false;
            value.exp2 = 0;
            return;
        }

        const int tz = fmod_u128_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = fmod_u128_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }
    BL_FORCE_INLINE  constexpr exact_dyadic_fmod exact_from_f128_fmod(const f128_s& value)
    {
        exact_dyadic_fmod hi = exact_from_double_fmod(value.hi);
        exact_dyadic_fmod lo = exact_from_double_fmod(value.lo);

        if (fmod_u128_is_zero(hi.mant))
            return lo;
        if (fmod_u128_is_zero(lo.mant))
            return hi;

        const int common_exp = std::min(hi.exp2, lo.exp2);
        const fmod_u128 hi_scaled = fmod_u128_shl_bits(hi.mant, hi.exp2 - common_exp);
        const fmod_u128 lo_scaled = fmod_u128_shl_bits(lo.mant, lo.exp2 - common_exp);

        exact_dyadic_fmod out;
        out.exp2 = common_exp;

        if (hi.neg == lo.neg)
        {
            out.neg = hi.neg;
            out.mant = hi_scaled;
            fmod_u128_add_inplace(out.mant, lo_scaled);
        }
        else
        {
            const int cmp = fmod_u128_compare(hi_scaled, lo_scaled);
            if (cmp >= 0)
            {
                out.neg = hi.neg;
                out.mant = hi_scaled;
                fmod_u128_sub_inplace(out.mant, lo_scaled);
            }
            else
            {
                out.neg = lo.neg;
                out.mant = lo_scaled;
                fmod_u128_sub_inplace(out.mant, hi_scaled);
            }
        }

        normalize_exact_dyadic_fmod(out);
        return out;
    }
    BL_FORCE_INLINE  constexpr bool fmod_fast_double_divisor_abs(const f128_s& ax, double ay, f128_s& out)
    {
        if (!(ay > 0.0) || !isfinite(ay))
            return false;

        const f128_s mod{ ay, 0.0 };

        if (ax.lo == 0.0)
        {
            out = f128_s{ fmod_constexpr(ax.hi, ay), 0.0 };
            return true;
        }

        const double rh = (ax.hi < ay) ? ax.hi : fmod_constexpr(ax.hi, ay);
        const double rl = (absd(ax.lo) < ay) ? ax.lo : fmod_constexpr(ax.lo, ay);

        f128_s r = add_inline(f128_s{ rh, 0.0 }, f128_s{ rl, 0.0 });

        if (r < 0.0)
            r = add_inline(r, mod);
        if (r >= mod)
            r = sub_inline(r, mod);

        if (r < 0.0)
            r = add_inline(r, mod);
        if (r >= mod)
            r = sub_inline(r, mod);

        if (r < 0.0 || r >= mod)
            return false;

        out = r;
        return true;
    }
    BL_FORCE_INLINE  constexpr f128_s exact_dyadic_to_f128_fmod(const fmod_u128& coeff, int exp2, bool neg)
    {
        if (fmod_u128_is_zero(coeff))
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        int ratio_exp = fmod_u128_bit_length(coeff) - 1;
        fmod_u128 q = coeff;

        if (ratio_exp > 105)
        {
            const int right_shift = ratio_exp - 105;
            const bool round_bit = fmod_u128_get_bit(q, right_shift - 1);
            const bool sticky = fmod_u128_any_low_bits_set(q, right_shift - 1);

            q = fmod_u128_shr_bits(q, right_shift);

            if (round_bit && (sticky || fmod_u128_is_odd(q)))
                fmod_u128_add_small(q, 1u);

            if (fmod_u128_bit_length(q) > 106)
            {
                q = fmod_u128_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < 105)
        {
            q = fmod_u128_shl_bits(q, 105 - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f128_s>::infinity() : std::numeric_limits<f128_s>::infinity();
        if (e2 < -1074)
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        const std::uint64_t c1 = fmod_u128_get_bits(q, 0, 53);
        const std::uint64_t c0 = fmod_u128_get_bits(q, 53, 53);
        const double hi = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = renorm(hi, lo);
        return neg ? -out : out;
    }
    BL_FORCE_INLINE  constexpr f128_s fmod_exact_fixed_limb(const f128_s& x, const f128_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f128_fmod(abs(x));
        const exact_dyadic_fmod dy = exact_from_f128_fmod(abs(y));

        fmod_u128 remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            const fmod_u128 denominator = fmod_u128_shl_bits(dy.mant, shift);
            remainder = fmod_u128_mod_shift_subtract(dx.mant, denominator);
            out_exp = dx.exp2;
        }
        else
        {
            remainder = fmod_u128_mod_shift_subtract(dx.mant, dy.mant);
            const int shift = dx.exp2 - dy.exp2;
            for (int i = 0; i < shift && !fmod_u128_is_zero(remainder); ++i)
                remainder = fmod_u128_double_mod(remainder, dy.mant);
            out_exp = dy.exp2;
        }

        f128_s out = exact_dyadic_to_f128_fmod(remainder, out_exp, !ispositive(x));
        if (iszero(out))
            return f128_s{ signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return out;
    }

    // integer conversion
    BL_FORCE_INLINE  constexpr bool f128_try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = trunc(x);
        if (xi != x)
            return false;

        if (absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = sub_inline(xi, to_f128(hi_part));
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }

    // decimal conversion
    BL_FORCE_INLINE  constexpr f128_s pack_decimal_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
    {
        const std::uint64_t c1 = q.get_bits(0, 53);
        const std::uint64_t c0 = q.get_bits(53, 53);
        const double hi = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = renorm(hi, lo);
        return neg ? -out : out;
    }
    BL_MSVC_NOINLINE constexpr f128_s round_decimal_exact_to_f128(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg) noexcept
    {
        if (coeff.is_zero())
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        detail::exact_decimal::biguint numerator = coeff;
        detail::exact_decimal::biguint denominator{ 1 };
        int bin_exp = 0;

        if (dec_exp >= 0)
        {
            numerator = detail::exact_decimal::mul_big(coeff, detail::exact_decimal::pow5_big(dec_exp));
            bin_exp = dec_exp;
        }
        else
        {
            denominator = detail::exact_decimal::pow5_big(-dec_exp);
            bin_exp = dec_exp;
        }

        int ratio_exp = detail::exact_decimal::floor_log2_ratio(numerator, denominator);
        detail::exact_decimal::biguint q = detail::exact_decimal::extract_rounded_significand_chunks(numerator, denominator, ratio_exp, std::numeric_limits<f128_s>::digits);
        if (q.bit_length() > std::numeric_limits<f128_s>::digits)
        {
            q.shr1();
            ++ratio_exp;
        }

        const int e2 = bin_exp + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f128_s>::infinity() : std::numeric_limits<f128_s>::infinity();
        if (e2 < -1074)
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        return pack_decimal_significand(q, e2, neg);
    }
    BL_FORCE_INLINE  constexpr bool try_rounded_decimal_to_f128(const f128_s& integer_part, const char* digits, int digit_count, bool neg, f128_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!f128_try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        const detail::exact_decimal::biguint coeff = detail::fp::append_decimal_digits(
            detail::exact_decimal::biguint{ static_cast<std::uint64_t>(integer_value) },
            digits,
            digit_count);

        out = round_decimal_exact_to_f128(coeff, -digit_count, neg);
        return true;
    }

    // pow
    BL_FORCE_INLINE  constexpr bool is_odd_integer(const f128_s& x) noexcept
    {
        int64_t value{};
        if (f128_try_get_int64(x, value))
            return (value & 1ll) != 0;

        if (x.lo != 0.0 || !isfinite(x.hi))
            return false;

        return double_integer_is_odd(x.hi);
    }
    BL_FORCE_INLINE  constexpr f128_s powi(f128_s base, int64_t exp)
    {
        return detail::fp::powi_by_squaring(base, exp);
    }
    BL_FORCE_INLINE  constexpr bool f128_try_exact_binary_log2(const f128_s& x, int& out) noexcept
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

    // exp / log
    BL_FORCE_INLINE constexpr double log_as_double_impl(f128_s a)
    {
        const double hi = a.hi;
        if (hi <= 0.0)
            return detail::fp::log_constexpr(static_cast<double>(a));

        return detail::fp::log_constexpr(hi) + detail::fp::log1p_constexpr(a.lo / hi);
    }
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
        f128_s sum = z;

        for (int k = 3; k <= 81; k += 2)
        {
            term = mul_inline(term, z2);
            const f128_s add = div_inline(term, f128_s{ static_cast<double>(k) });
            sum = add_inline(sum, add);

            const f128_s asum = abs(sum);
            const f128_s scale = (asum > f128_s{ 1.0 }) ? asum : f128_s{ 1.0 };
            if (abs(add) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return add_inline(sum, sum);
    }
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

        const f128_s t = mul_inline(x, inv_ln2);

        double kd = nearbyint_ties_even(t.hi);
        const f128_s delta = sub_inline(t, f128_s{ kd });
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s r = mul_inline(sub_inline(x, mul_inline(f128_s{ kd }, ln2)), f128_s{ 0.001953125 });

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
            return _exp(mul_inline(x, ln2));

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
        const f128_s r = mul_inline(mul_inline(reduced, ln2), f128_s{ 0.001953125 });

        f128_s e = expm1_tiny(r);
        for (int i = 0; i < 9; ++i)
            e = mul_add_inline(e, e, e * 2.0);

        return _ldexp(add_inline(e, f128_s{ 1.0 }), k);
    }
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
            exp2 = detail::fp::frexp_exponent_constexpr(a.hi);
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

        const f128_s exp2_ln2 = mul_inline(f128_s{ static_cast<double>(exp2) }, ln2);
        f128_s y = add_inline(exp2_ln2, f128_s{ log_as_double_impl(m) });
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

    // trig
    BL_FORCE_INLINE constexpr bool    remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
	{
	    const double ax = fabs_constexpr(x.hi);
	    if (!isfinite(ax))
	        return false;

	    if (ax > 7.0e15)
	        return false;

	    const f128_s t = mul_inline(x, invpi2);

	    double qd = nearbyint_ties_even(t.hi);
	    if (!isfinite(qd) ||
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
    BL_FORCE_INLINE constexpr f128_s  sin_kernel_pi4(const f128_s& x)
    {
        using namespace detail::_f128;

        const f128_s t = mul_inline(x, x);

        f128_s ps = f128_s{  1.13099628864477159e-31,  1.04980154129595057e-47 };
        ps = mul_add_inline(ps, t, f128_s{ -9.18368986379554615e-29, -1.43031503967873224e-45 });
        ps = mul_add_inline(ps, t, f128_s{  6.44695028438447391e-26, -1.93304042337034642e-42 });
        ps = mul_add_inline(ps, t, f128_s{ -3.86817017063068404e-23,  8.84317765548234382e-40 });
        ps = mul_add_inline(ps, t, f128_s{  1.95729410633912625e-20, -1.36435038300879076e-36 });
        ps = mul_add_inline(ps, t, f128_s{ -8.22063524662432972e-18, -2.21418941196042654e-34 });
        ps = mul_add_inline(ps, t, f128_s{  2.81145725434552060e-15,  1.65088427308614330e-31 });
        ps = mul_add_inline(ps, t, f128_s{ -7.64716373181981648e-13, -7.03872877733452971e-30 });
        ps = mul_add_inline(ps, t, f128_s{  1.60590438368216146e-10,  1.25852945887520981e-26 });
        ps = mul_add_inline(ps, t, f128_s{ -2.50521083854417188e-08,  1.44881407093591197e-24 });
        ps = mul_add_inline(ps, t, f128_s{  2.75573192239858907e-06, -1.85839327404647208e-22 });
        ps = mul_add_inline(ps, t, f128_s{ -1.98412698412698413e-04, -1.72095582934207053e-22 });
        ps = mul_add_inline(ps, t, f128_s{  8.33333333333333322e-03,  1.15648231731787140e-19 });
        ps = mul_add_inline(ps, t, f128_s{ -1.66666666666666657e-01, -9.25185853854297066e-18 });

        return mul_add_inline(mul_inline(x, t), ps, x);
    }
    BL_FORCE_INLINE constexpr f128_s  cos_kernel_pi4(const f128_s& x)
    {
        using namespace detail::_f128;

        const f128_s t = mul_inline(x, x);

        f128_s pc =   f128_s{  3.27988923706983791e-30,  1.51175427440298786e-46 };
        pc = mul_add_inline(pc, t, f128_s{ -2.47959626322479746e-27,  1.29537309647652292e-43 });
        pc = mul_add_inline(pc, t, f128_s{  1.61173757109611835e-24, -3.68465735645097656e-41 });
        pc = mul_add_inline(pc, t, f128_s{ -8.89679139245057329e-22,  7.91140261487237594e-38 });
        pc = mul_add_inline(pc, t, f128_s{  4.11031762331216486e-19,  1.44129733786595266e-36 });
        pc = mul_add_inline(pc, t, f128_s{ -1.56192069685862265e-16, -1.19106796602737541e-32 });
        pc = mul_add_inline(pc, t, f128_s{  4.77947733238738530e-14,  4.39920548583408094e-31 });
        pc = mul_add_inline(pc, t, f128_s{ -1.14707455977297247e-11, -2.06555127528307454e-28 });
        pc = mul_add_inline(pc, t, f128_s{  2.08767569878680990e-09, -1.20734505911325997e-25 });
        pc = mul_add_inline(pc, t, f128_s{ -2.75573192239858907e-07, -2.37677146222502973e-23 });
        pc = mul_add_inline(pc, t, f128_s{  2.48015873015873016e-05,  2.15119478667758816e-23 });
        pc = mul_add_inline(pc, t, f128_s{ -1.38888888888888894e-03,  5.30054395437357706e-20 });
        pc = mul_add_inline(pc, t, f128_s{  4.16666666666666644e-02,  2.31296463463574269e-18 });
        pc = mul_add_inline(pc, t, f128_s{ -5.00000000000000000e-01,  0.0                     });

        return mul_add_inline(t, pc, f128_s{ 1.0 });
    }
    BL_FORCE_INLINE constexpr void sincos_kernel_pi4(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        s_out = sin_kernel_pi4(x);
        c_out = cos_kernel_pi4(x);
    }
    BL_FORCE_INLINE constexpr f128_s atan_impl(const f128_s& x)
    {
        return atan2(x, f128_s{ 1.0 });
    }
    BL_FORCE_INLINE constexpr f128_s asin_impl(const f128_s& x)
    {
        return atan2(x, sqrt(sub_inline(f128_s{ 1.0 }, mul_inline(x, x))));
    }
    BL_FORCE_INLINE constexpr f128_s acos_impl(const f128_s& x)
    {
        return atan2(sqrt(sub_inline(f128_s{ 1.0 }, mul_inline(x, x))), x);
    }

    // classification / signs
    BL_FORCE_INLINE constexpr f128_s fabs_impl(const f128_s& a) noexcept
    {
        return abs(a);
    }

    // rounding
    using detail::fp::nextafter_double_constexpr;
    BL_FORCE_INLINE constexpr f128_s round_half_away_zero(const f128_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f128_s y = -floor(add_inline(-x, f128_s{ 0.5 }));
            if (iszero(y))
                return f128_s{ -0.0, 0.0 };
            return y;
        }

        return floor(add_inline(x, f128_s{ 0.5 }));
    }
    BL_FORCE_INLINE  constexpr f128_s round_impl(const f128_s& a)
    {
        return round_half_away_zero(a);
    }
    BL_FORCE_INLINE  constexpr f128_s nearbyint_impl(const f128_s& a)
    {
        if (isnan(a) || isinf(a) || iszero(a))
            return a;

        f128_s t = floor(a);
        f128_s frac = sub_inline(a, t);

        if (frac < f128_s{ 0.5 })
            return t;

        if (frac > f128_s{ 0.5 })
        {
            t = add_inline(t, f128_s{ 1.0 });
            if (iszero(t))
                return f128_s{ signbit_constexpr(a.hi) ? -0.0 : 0.0 };
            return t;
        }

        if (fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
            t = add_inline(t, f128_s{ 1.0 });

        if (iszero(t))
            return f128_s{ signbit_constexpr(a.hi) ? -0.0 : 0.0 };

        return t;
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f128_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        static_assert(sizeof(SignedInt) <= sizeof(std::int64_t));

        if (bl::isnan(x) || bl::isinf(x))
            return 0;

        constexpr auto lo_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::lowest());
        constexpr auto hi_i = static_cast<std::int64_t>(std::numeric_limits<SignedInt>::max());
        const f128_s lo = to_f128(lo_i);
        const f128_s hi = to_f128(hi_i);

        if (x < lo || x > hi)
            return 0;

        std::int64_t out = 0;
        if (!f128_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }
    BL_FORCE_INLINE constexpr f128_s nearest_integer_ties_even(const f128_s& q) noexcept
    {
        f128_s n = trunc(q);
        const f128_s frac = sub_inline(q, n);
        const f128_s half{ 0.5 };
        const f128_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n = add_inline(n, signbit(frac) ? -one : one);
        }
        else if (abs(frac) == half)
        {
            if (fmod(n, f128_s{ 2.0 }) != f128_s{ 0.0 })
                n = add_inline(n, signbit(frac) ? -one : one);
        }

        return n;
    }

    // gamma
    BL_MSVC_NOINLINE constexpr f128_s lgamma1p_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f128_s p = lgamma1p_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = mul_add_inline(p, y, lgamma1p_coeff[i]);

        return mul_inline(y, mul_add_inline(y, p, -egamma));
    }
    BL_MSVC_NOINLINE constexpr f128_s lgamma1p5_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p5_coeff) / sizeof(lgamma1p5_coeff[0]));

        f128_s p = lgamma1p5_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = mul_add_inline(p, y, lgamma1p5_coeff[i]);

        const f128_s constant = sub_inline(half_log_two_pi, mul_inline(f128_s{ 1.5 }, ln2));
        const f128_s linear = sub_inline(sub_inline(f128_s{ 2.0 }, egamma), mul_inline(f128_s{ 2.0 }, ln2));
        return mul_add_inline(y, mul_add_inline(y, p, linear), constant);
    }
    BL_MSVC_NOINLINE constexpr bool   try_lgamma_near_one_or_two(const f128_s& x, f128_s& out) noexcept
    {
        const f128_s y1 = sub_inline(x, f128_s{ 1.0 });
        if (abs(y1) <= f128_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f128_s y15 = sub_inline(x, f128_s{ 1.5 });
        if (abs(y15) <= f128_s{ 0.25 })
        {
            out = lgamma1p5_series(y15);
            return true;
        }

        const f128_s y2 = sub_inline(x, f128_s{ 2.0 });
        if (abs(y2) <= f128_s{ 0.25 })
        {
            out = add_inline(f128_log1p_series_reduced(y2), lgamma1p_series(y2));
            return true;
        }

        return false;
    }
    BL_MSVC_NOINLINE constexpr f128_s lgamma_stirling_asymptotic(const f128_s& z) noexcept
    {
        const f128_s inv = div_inline(f128_s{ 1.0 }, z);
        const f128_s inv2 = mul_inline(inv, inv);

        f128_s series = div_inline(inv, f128_s{ 12.0 });
        f128_s invpow = mul_inline(inv, inv2);

        series = sub_inline(series, div_inline(invpow, f128_s{ 360.0 }));
        invpow = mul_inline(invpow, inv2);
        series = add_inline(series, div_inline(invpow, f128_s{ 1260.0 }));
        invpow = mul_inline(invpow, inv2);
        series = sub_inline(series, div_inline(invpow, f128_s{ 1680.0 }));
        invpow = mul_inline(invpow, inv2);
        series = add_inline(series, div_inline(invpow, f128_s{ 1188.0 }));
        invpow = mul_inline(invpow, inv2);
        series = sub_inline(series, mul_inline(invpow, div_inline(f128_s{ 691.0 }, f128_s{ 360360.0 })));
        invpow = mul_inline(invpow, inv2);
        series = add_inline(series, div_inline(invpow, f128_s{ 156.0 }));
        invpow = mul_inline(invpow, inv2);
        series = sub_inline(series, mul_inline(invpow, div_inline(f128_s{ 3617.0 }, f128_s{ 122400.0 })));
        invpow = mul_inline(invpow, inv2);
        series = add_inline(series, mul_inline(invpow, div_inline(f128_s{ 43867.0 }, f128_s{ 244188.0 })));
        invpow = mul_inline(invpow, inv2);
        series = sub_inline(series, mul_inline(invpow, div_inline(f128_s{ 174611.0 }, f128_s{ 125400.0 })));
        invpow = mul_inline(invpow, inv2);
        series = add_inline(series, mul_inline(invpow, div_inline(f128_s{ 77683.0 }, f128_s{ 5796.0 })));
        invpow = mul_inline(invpow, inv2);
        series = sub_inline(series, mul_inline(invpow, div_inline(f128_s{ 236364091.0 }, f128_s{ 1506960.0 })));

        return add_inline(add_inline(sub_inline(mul_inline(sub_inline(z, f128_s{ 0.5 }), log(z)), z), half_log_two_pi), series);
    }
    BL_MSVC_NOINLINE constexpr void   positive_recurrence_product(const f128_s& x, const f128_s& asymptotic_min, f128_s& z, f128_s& product, int& product_scale2) noexcept
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
                const int exponent = frexp_exponent(hi);
                if (exponent > 512 || exponent < -512)
                {
                    product = ldexp(product, -exponent);
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

        const f128_s correction = log(product);
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
        const f128_s local_gamma = exp(local_lgamma);
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
            sub_inline(lgamma_stirling_asymptotic(z), log(product)),
            mul_inline(f128_s{ static_cast<double>(product_scale2) }, ln2));
    }
    BL_MSVC_NOINLINE constexpr f128_s gamma_positive_recurrence(const f128_s& x) noexcept
    {
        f128_s near_lgamma{};
        if (try_lgamma_near_one_or_two(x, near_lgamma))
            return exp(near_lgamma);

        if (x <= f128_s{ 16.0 })
            return gamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z{};
        f128_s product{};
        int product_scale2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_scale2);

        f128_s out = div_inline(exp(lgamma_stirling_asymptotic(z)), product);
        if (product_scale2 != 0)
            out = ldexp(out, -product_scale2);

        return out;
    }

    // hyperbolic
    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series_constexpr(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (abs(term) <= f128_s::eps())
                break;
        }

        return sum;
    }
    BL_MSVC_NOINLINE constexpr f128_s atanh_small_series_runtime(const f128_s& x)
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s sum = x;
        f128_s power = x;

        for (int k = 1; k <= 32; ++k)
        {
            power = mul_inline(power, x2);
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * k + 1) });
            sum = add_inline(sum, term);

            if (abs(term) <= f128_s::eps())
                break;
        }

        return sum;
    }

    // integer rounding
    BL_FORCE_INLINE constexpr f128_s rint_impl(const f128_s& x)
    {
        return nearbyint_impl(x);
    }
    BL_FORCE_INLINE constexpr long lround_impl(const f128_s& x)
    {
        return to_signed_integer_or_zero<long>(round_half_away_zero(x));
    }
    BL_FORCE_INLINE constexpr long long llround_impl(const f128_s& x)
    {
        return to_signed_integer_or_zero<long long>(round_half_away_zero(x));
    }
    BL_FORCE_INLINE constexpr long lrint_impl(const f128_s& x)
    {
        return to_signed_integer_or_zero<long>(nearbyint_impl(x));
    }
    BL_FORCE_INLINE constexpr long long llrint_impl(const f128_s& x)
    {
        return to_signed_integer_or_zero<long long>(nearbyint_impl(x));
    }

    // fused arithmetic / min-max
    BL_FORCE_INLINE constexpr f128_s fma_impl(const f128_s& x, const f128_s& y, const f128_s& z)
    {
        return canonicalize_math_result(add_inline(mul_inline(x, y), z));
    }
    BL_FORCE_INLINE constexpr f128_s fmin_impl(const f128_s& a, const f128_s& b)
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a < b) return a;
        if (b < a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? a : b;
        return a;
    }
    BL_FORCE_INLINE constexpr f128_s fmax_impl(const f128_s& a, const f128_s& b)
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a > b) return a;
        if (b > a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? b : a;
        return a;
    }
    BL_FORCE_INLINE constexpr f128_s fdim_impl(const f128_s& x, const f128_s& y)
    {
        return (x > y) ? canonicalize_math_result(sub_inline(x, y)) : f128_s{ 0.0 };
    }
    BL_FORCE_INLINE constexpr f128_s copysign_impl(const f128_s& x, const f128_s& y)
    {
        return signbit(x) == signbit(y) ? x : -x;
    }

    // logb / scalbn
    BL_FORCE_INLINE constexpr int    ilogb_impl(const f128_s& x) noexcept
    {
        if (isnan(x))
            return FP_ILOGBNAN;
        if (iszero(x))
            return FP_ILOGB0;
        if (isinf(x))
            return std::numeric_limits<int>::max();

        int e = 0;
        (void)frexp(abs(x), &e);
        return e - 1;
    }
    BL_FORCE_INLINE constexpr f128_s logb_impl(const f128_s& x) noexcept
    {
        if (isnan(x))
            return x;
        if (iszero(x))
            return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
        if (isinf(x))
            return std::numeric_limits<f128_s>::infinity();

        return f128_s{ static_cast<double>(ilogb_impl(x)), 0.0 };
    }
    BL_FORCE_INLINE constexpr f128_s scalbn_impl(const f128_s& x, int e) noexcept
    {
        return ldexp(x, e);
    }
    BL_FORCE_INLINE constexpr f128_s scalbln_impl(const f128_s& x, long e) noexcept
    {
        return ldexp(x, static_cast<int>(e));
    }

    // nextafter / nexttoward
    BL_FORCE_INLINE constexpr f128_s nexttoward_impl(const f128_s& from, long double to) noexcept
    {
        return nextafter(from, f128_s{ static_cast<double>(to) });
    }
    BL_FORCE_INLINE constexpr f128_s nexttoward_impl(const f128_s& from, const f128_s& to) noexcept
    {
        return nextafter(from, to);
    }

    // erf / erfc
    BL_FORCE_INLINE constexpr f128_s erf_positive_series(const f128_s& x)
    {
        const f128_s xx = mul_inline(x, x);
        f128_s power = x;
        f128_s sum = x;

        for (int n = 1; n < 256; ++n)
        {
            power = mul_inline(power, div_inline(-xx, f128_s{ static_cast<double>(n) }));
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * n + 1) });
            sum = add_inline(sum, term);
            if (abs(term) < f128_s::eps())
                break;
        }

        return canonicalize_math_result(mul_inline(mul_inline(f128_s{ 2.0 }, inv_sqrtpi), sum));
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
            if (abs(d) < tiny)
                d = tiny;

            c = add_inline(b, div_inline(an, c));
            if (abs(c) < tiny)
                c = tiny;

            d = div_inline(f128_s{ 1.0 }, d);
            const f128_s delta = mul_inline(d, c);
            h = mul_inline(h, delta);

            if (abs(sub_inline(delta, f128_s{ 1.0 })) <= mul_inline(f128_s{ 32.0 }, f128_s::eps()))
                break;
        }

        const f128_s out = mul_inline(mul_inline(mul_inline(detail::_f128_constexpr::exp(-z), x), inv_sqrtpi), h);
        return canonicalize_math_result(out);
    }
}

/// ============= f128 inline public wrappers =============

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f128_s a) { return detail::_f128::log_as_double_impl(a); }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fabs(const f128_s& a) noexcept { return detail::_f128::fabs_impl(a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s round(const f128_s& a) { return detail::_f128::round_impl(a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nearbyint(const f128_s& a) { return detail::_f128::nearbyint_impl(a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s rint(const f128_s& x) { return detail::_f128::rint_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f128_s& x) { return detail::_f128::lround_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f128_s& x) { return detail::_f128::llround_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f128_s& x) { return detail::_f128::lrint_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f128_s& x) { return detail::_f128::llrint_impl(x); }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s atan(const f128_s& x) { return detail::_f128::atan_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s asin(const f128_s& x) { return detail::_f128::asin_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s acos(const f128_s& x) { return detail::_f128::acos_impl(x); }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fma(const f128_s& x, const f128_s& y, const f128_s& z) { return detail::_f128::fma_impl(x, y, z); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmin(const f128_s& a, const f128_s& b) { return detail::_f128::fmin_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmax(const f128_s& a, const f128_s& b) { return detail::_f128::fmax_impl(a, b); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fdim(const f128_s& x, const f128_s& y) { return detail::_f128::fdim_impl(x, y); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s copysign(const f128_s& x, const f128_s& y) { return detail::_f128::copysign_impl(x, y); }

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept { return detail::_f128::ilogb_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s logb(const f128_s& x) noexcept { return detail::_f128::logb_impl(x); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s scalbn(const f128_s& x, int e) noexcept { return detail::_f128::scalbn_impl(x, e); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s scalbln(const f128_s& x, long e) noexcept { return detail::_f128::scalbln_impl(x, e); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, long double to) noexcept { return detail::_f128::nexttoward_impl(from, to); }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, const f128_s& to) noexcept { return detail::_f128::nexttoward_impl(from, to); }

/// ============= f128 constexpr implementations =============

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::fmod(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    const f128_s ay = abs(y);

    if (ax < ay)
        return x;

    f128_s fast{};
    if (y.lo == 0.0 && fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return fmod_exact_fixed_limb(x, y);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::round_to_decimals(f128_s v, int prec)
{
    constexpr int local_capacity = std::numeric_limits<f128_s>::max_digits10;

    if (prec <= 0) return v;
    if (prec > local_capacity) prec = local_capacity;

    constexpr f128_s INV10_DD{
        0.1000000000000000055511151231257827021181583404541015625,
       -0.0000000000000000055511151231257827021181583404541015625
    };

    char digits[local_capacity];

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f128_s ip = detail::_f128_constexpr::floor(v);
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
    const bool round_up =
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
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::remainder(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s ay = abs(y);
    f128_s r = detail::_f128_constexpr::fmod(x, y);
    const f128_s ar = abs(r);
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
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    const int exp2 = frexp_exponent(a.hi);
    const int result_scale = exp2 / 2;
    const int input_scale = -2 * result_scale;
    const f128_s scaled_a = input_scale == 0 ? a : ldexp_terms(a, input_scale);

    double y0;
    if (bl::use_constexpr_math()) {
        y0 = sqrt_seed_constexpr(scaled_a.hi);
        f128_s y{ y0 };
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_math_result(y);
    }
    else {
        y0 = std::sqrt(scaled_a.hi);
        f128_s y{ y0 };
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, div_inline(sub_mul_inline(scaled_a, y, y), add_inline(y, y)));
        y = add_inline(y, mul_inline(sub_mul_inline(scaled_a, y, y), f128_s{ 0.5 / y0 }));

        if (result_scale != 0)
            y = ldexp_terms(y, result_scale);

        return canonicalize_math_result(y);
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::ldexp(const f128_s& x, int e)
{
    return canonicalize_math_result(_ldexp(x, e));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::exp(const f128_s& x)
{
    return canonicalize_math_result(_exp(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::exp2(const f128_s& x)
{
    return canonicalize_math_result(_exp2(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::log(const f128_s& a)
{
    return canonicalize_math_result(_log(a));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::log2(const f128_s& a)
{
    int exact_exp2{};
    if (f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return canonicalize_math_result(mul_inline(_log(a), inv_ln2));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::log10(const f128_s& x)
{
    if (x.hi > 0.0)
    {
        const int exp2 =
            detail::fp::frexp_exponent_constexpr(x.hi);
        const int k0 =
            static_cast<int>(detail::fp::floor_constexpr((exp2 - 1) * 0.30102999566398114));

        for (int k = k0 - 2; k <= k0 + 2; ++k)
        {
            if (x == detail::_f128_constexpr::pow10_128(k))
                return f128_s{ static_cast<double>(k), 0.0 };
        }
    }

    return canonicalize_math_result(mul_inline(_log(x), inv_ln10));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s yi = detail::_f128_constexpr::trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && f128_try_get_int64(yi, yi64))
        return powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(mul_inline(y, _log(-x)));
        return is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return canonicalize_math_result(_exp(mul_inline(y, _log(x))));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::pow(const f128_s& x, double y)
{
    if (y == 0.0)
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    if (y == 1.0)  return x;
    if (y == 2.0)  return canonicalize_math_result(x * x);
    if (y == -1.0) return canonicalize_math_result(f128_s{ 1.0 } / x);
    if (y == 0.5)  return canonicalize_math_result(detail::_f128_constexpr::sqrt(x));

    double yi{};
    if (bl::use_constexpr_math())
    {
        yi = (y < 0.0)
            ? ceil_constexpr(y)
            : floor_constexpr(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && absd(yi) < 0x1p63)
        return powi(x, static_cast<int64_t>(yi));

    if (x.hi < 0.0 || (x.hi == 0.0 && signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _exp(_log(-x) * y);
        const bool y_is_odd =
            (absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return canonicalize_math_result(_exp(_log(x) * y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr bool   detail::_f128_constexpr::sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    const double ax = fabs_constexpr(x.hi);
    if (!isfinite(ax))
    {
        s_out = f128_s{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= pi_4_hi)
    {
        sincos_kernel_pi4(x, s_out, c_out);
        s_out = canonicalize_math_result(s_out);
        c_out = canonicalize_math_result(c_out);
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

    s_out = canonicalize_math_result(s_out);
    c_out = canonicalize_math_result(c_out);
    return true;
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::sin(const f128_s& x)
{
    const double ax = fabs_constexpr(x.hi);
    if (!isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= pi_4_hi)
        return canonicalize_math_result(sin_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return canonicalize_math_result(f128_s{ detail::fp::sin_constexpr(static_cast<double>(x)) });
        }
        else
        {
            return canonicalize_math_result(f128_s{ std::sin((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return canonicalize_math_result(sin_kernel_pi4(r));
    case 1: return canonicalize_math_result(cos_kernel_pi4(r));
    case 2: return canonicalize_math_result(-sin_kernel_pi4(r));
    default: return canonicalize_math_result(-cos_kernel_pi4(r));
    }
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::cos(const f128_s& x)
{
    const double ax = fabs_constexpr(x.hi);
    if (!isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= pi_4_hi)
        return canonicalize_math_result(cos_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return canonicalize_math_result(f128_s{ detail::fp::cos_constexpr(static_cast<double>(x)) });
        }
        else
        {
            return canonicalize_math_result(f128_s{ std::cos((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return canonicalize_math_result(cos_kernel_pi4(r));
    case 1: return canonicalize_math_result(-sin_kernel_pi4(r));
    case 2: return canonicalize_math_result(-cos_kernel_pi4(r));
    default: return canonicalize_math_result(sin_kernel_pi4(r));
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::tan(const f128_s& x)
{
    f128_s s{}, c{};
    if (detail::_f128_constexpr::sincos(x, s, c))
        return div_inline(s, c);
    const double xd = (double)x;
    if (bl::use_constexpr_math()) {
        return f128_s{ detail::fp::tan_constexpr(xd) };
    } else {
        return f128_s{ std::tan(xd) };
    }
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::atan2(const f128_s& y, const f128_s& x)
{
    if (iszero(x)) [[unlikely]]
    {
        if (iszero(y)) [[unlikely]]
            return f128_s{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? detail::_f128::pi_2 : -detail::_f128::pi_2;
    }

    const f128_s scale = std::max(abs(x), abs(y));
    const f128_s xs = detail::_f128::div_inline(x, scale);
    const f128_s ys = detail::_f128::div_inline(y, scale);

    f128_s v{ detail::fp::atan2_constexpr(y.hi, x.hi) };

    for (int i = 0; i < 2; ++i)
    {
        f128_s sv{}, cv{};
        if (!detail::_f128_constexpr::sincos(v, sv, cv))
        {
            const double vd = (double)v;
            if (bl::use_constexpr_math()) {
                double sd{}, cd{};
                detail::fp::sincos_constexpr(vd, sd, cd);
                sv = f128_s{ sd };
                cv = f128_s{ cd };
            } else {
                sv = f128_s{ std::sin(vd) };
                cv = f128_s{ std::cos(vd) };
            }
        }

        const f128_s f  = detail::_f128::diff_products_inline(xs, sv, ys, cv);
        const f128_s fp = detail::_f128::sum_products_inline(xs, cv, ys, sv);

        v = detail::_f128::sub_inline(v, detail::_f128::div_inline(f, fp));
    }

    return detail::_f128::canonicalize_math_result(v);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::expm1(const f128_s& x)
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

    const f128_s ax = abs(x);
    if (ax <= f128_s{ 0.5 })
    {
        f128_s term = x;
        f128_s sum = x;

        for (int n = 2; n <= 80; ++n)
        {
            term = div_inline(mul_inline(term, x), f128_s{ static_cast<double>(n) });
            sum = add_inline(sum, term);

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return canonicalize_math_result(sum);
    }

    return canonicalize_math_result(sub_inline(detail::_f128_constexpr::exp(x), f128_s{ 1.0 }));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::log1p(const f128_s& x)
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

    const f128_s ax = abs(x);
    if (ax <= f128_s{ 0.5 })
        return canonicalize_math_result(f128_log1p_series_reduced(x));

    const f128_s u = add_inline(f128_s{ 1.0 }, x);
    if (sub_inline(u, f128_s{ 1.0 }) == x)
        return canonicalize_math_result(detail::_f128_constexpr::log(u));

    if (x > f128_s{ 0.0 } && x <= f128_s{ 1.0 })
    {
        const f128_s t = div_inline(x, add_inline(f128_s{ 1.0 }, detail::_f128_constexpr::sqrt(add_inline(f128_s{ 1.0 }, x))));
        return canonicalize_math_result(mul_inline(f128_log1p_series_reduced(t), f128_s{ 2.0 }));
    }

    if (x > f128_s{ 0.0 })
        return canonicalize_math_result(detail::_f128_constexpr::log(u));

    const f128_s y = sub_inline(u, f128_s{ 1.0 });
    if (iszero(y))
        return x;

    return canonicalize_math_result(mul_inline(detail::_f128_constexpr::log(u), div_inline(x, y)));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::sinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax <= f128_s{ 0.5 })
    {
        const f128_s x2 = mul_inline(x, x);
        f128_s term = x;
        f128_s sum = x;

        for (int n = 1; n <= 40; ++n)
        {
            const double denom = static_cast<double>((n * 2) * (n * 2 + 1));
            term = div_inline(mul_inline(term, x2), f128_s{ denom });
            sum = add_inline(sum, term);

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= mul_inline(f128_s::eps(), scale))
                break;
        }

        return canonicalize_math_result(sum);
    }

    const f128_s ex = detail::_f128_constexpr::exp(ax);
    f128_s out = mul_inline(sub_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 });
    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::cosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = abs(x);
    const f128_s ex = detail::_f128_constexpr::exp(ax);
    return canonicalize_math_result(mul_inline(add_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 }));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::tanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = abs(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = detail::_f128_constexpr::expm1(add_inline(ax, ax));
    f128_s out = div_inline(em1, add_inline(em1, f128_s{ 2.0 }));
    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::asinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = abs(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = add_inline(detail::_f128_constexpr::log(ax), ln2);
    else
        out = detail::_f128_constexpr::log(add_inline(ax, detail::_f128_constexpr::sqrt(add_inline(mul_inline(ax, ax), f128_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::acosh(const f128_s& x)
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
        out = add_inline(detail::_f128_constexpr::log(x), ln2);
    else
        out = detail::_f128_constexpr::log(add_inline(x, detail::_f128_constexpr::sqrt(mul_inline(sub_inline(x, f128_s{ 1.0 }), add_inline(x, f128_s{ 1.0 })))));

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::atanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax > f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (ax == f128_s{ 1.0 })
        return signbit(x)
            ? f128_s{ -std::numeric_limits<double>::infinity(), 0.0 }
            : f128_s{  std::numeric_limits<double>::infinity(), 0.0 };

    if (ax <= f128_s{ 0.125 })
    {
        if (bl::use_constexpr_math())
            return canonicalize_math_result(atanh_small_series_constexpr(x));

        return canonicalize_math_result(atanh_small_series_runtime(x));
    }

    const f128_s out = mul_inline(detail::_f128_constexpr::log(div_inline(add_inline(f128_s{ 1.0 }, x), sub_inline(f128_s{ 1.0 }, x))), f128_s{ 0.5 });
    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::cbrt(const f128_s& x)
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
        int rem = exp2 % 3;
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
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::hypot(const f128_s& x, const f128_s& y)
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
        ex = detail::fp::frexp_exponent_constexpr(ax.hi);
        ey = detail::fp::frexp_exponent_constexpr(ay.hi);
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
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::remquo(const f128_s& x, const f128_s& y, int* quo)
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
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::frexp(const f128_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const double lead = (x.hi != 0.0) ? x.hi : x.lo;
    int e = 0;

    if (bl::use_constexpr_math())
        e = detail::fp::frexp_exponent_constexpr(lead);
    else
        (void)std::frexp(lead, &e);

    f128_s m = detail::_f128_constexpr::ldexp(x, -e);
    const f128_s am = abs(m);

    if (am < f128_s{ 0.5 })
    {
        m = mul_inline(m, f128_s{ 2.0 });
        --e;
    }
    else if (am >= f128_s{ 1.0 })
    {
        m = mul_inline(m, f128_s{ 0.5 });
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::modf(const f128_s& x, f128_s* iptr) noexcept
{
    const f128_s i = detail::_f128_constexpr::trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = sub_inline(x, i);
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::nextafter(const f128_s& from, const f128_s& to) noexcept
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
        nextafter_double_constexpr(from.lo, toward)
    );
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::erf(const f128_s& x)
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

    f128_s out = ax < f128_s{ 2.0 }
        ? erf_positive_series(ax)
        : (ax > f128_s{ 27.0 } ? f128_s{ 1.0 } : sub_inline(f128_s{ 1.0 }, erfc_positive_cf(ax)));

    if (neg)
        out = -out;

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s detail::_f128_constexpr::erfc(const f128_s& x)
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
        if (ax < f128_s{ 2.0 })
            return canonicalize_math_result(add_inline(f128_s{ 1.0 }, erf_positive_series(ax)));
        if (ax > f128_s{ 27.0 })
            return f128_s{ 2.0 };
        return canonicalize_math_result(sub_inline(f128_s{ 2.0 }, erfc_positive_cf(ax)));
    }

    // use the existing high-quality erf series throughout the region where it is stable
    if (x < f128_s{ 2.0 })
        return canonicalize_math_result(sub_inline(f128_s{ 1.0 }, erf_positive_series(x)));

    if (x > f128_s{ 27.0 })
        return f128_s{ 0.0 };

    return canonicalize_math_result(erfc_positive_cf(x));
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::lgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return canonicalize_math_result(lgamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_constexpr::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = detail::_f128_constexpr::sin(mul_inline(pi, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        sub_inline(
            sub_inline(detail::_f128_constexpr::log(pi), detail::_f128_constexpr::log(abs(sinpix))),
            lgamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x)));

    return canonicalize_math_result(out);
}
[[nodiscard]] BL_FORCE_INLINE  constexpr f128_s detail::_f128_constexpr::tgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return canonicalize_math_result(gamma_positive_recurrence(x));

    const f128_s xi = detail::_f128_constexpr::trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = detail::_f128_constexpr::sin(mul_inline(pi, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = div_inline(pi, mul_inline(sinpix, gamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x))));
    return canonicalize_math_result(out);
}

/// ============= f128 public wrappers =============

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::fmod(x, y), detail::_f128_runtime::fmod(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s round_to_decimals(f128_s v, int prec)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::round_to_decimals(v, prec), detail::_f128_runtime::round_to_decimals(v, prec));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::remainder(x, y), detail::_f128_runtime::remainder(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sqrt(f128_s a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::sqrt(a), detail::_f128_runtime::sqrt(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s ldexp(const f128_s& x, int e)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::ldexp(x, e), detail::_f128_runtime::ldexp(x, e));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s exp(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::exp(x), detail::_f128_runtime::exp(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s exp2(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::exp2(x), detail::_f128_runtime::exp2(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::log(a), detail::_f128_runtime::log(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log2(const f128_s& a)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::log2(a), detail::_f128_runtime::log2(a));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log10(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::log10(x), detail::_f128_runtime::log10(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::pow(x, y), detail::_f128_runtime::pow(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s pow(const f128_s& x, double y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::pow(x, y), detail::_f128_runtime::pow(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr bool   sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::sincos(x, s_out, c_out), detail::_f128_runtime::sincos(x, s_out, c_out));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sin(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::sin(x), detail::_f128_runtime::sin(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cos(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::cos(x), detail::_f128_runtime::cos(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tan(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::tan(x), detail::_f128_runtime::tan(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::atan2(y, x), detail::_f128_runtime::atan2(y, x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s expm1(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::expm1(x), detail::_f128_runtime::expm1(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s log1p(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::log1p(x), detail::_f128_runtime::log1p(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s sinh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::sinh(x), detail::_f128_runtime::sinh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cosh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::cosh(x), detail::_f128_runtime::cosh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tanh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::tanh(x), detail::_f128_runtime::tanh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s asinh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::asinh(x), detail::_f128_runtime::asinh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s acosh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::acosh(x), detail::_f128_runtime::acosh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s atanh(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::atanh(x), detail::_f128_runtime::atanh(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s cbrt(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::cbrt(x), detail::_f128_runtime::cbrt(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::hypot(x, y), detail::_f128_runtime::hypot(x, y));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::remquo(x, y, quo), detail::_f128_runtime::remquo(x, y, quo));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::frexp(x, exp), detail::_f128_runtime::frexp(x, exp));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::modf(x, iptr), detail::_f128_runtime::modf(x, iptr));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::nextafter(from, to), detail::_f128_runtime::nextafter(from, to));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erf(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::erf(x), detail::_f128_runtime::erf(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s erfc(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::erfc(x), detail::_f128_runtime::erfc(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s lgamma(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::lgamma(x), detail::_f128_runtime::lgamma(x));
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s tgamma(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(detail::_f128_constexpr::tgamma(x), detail::_f128_runtime::tgamma(x));
}

}

#endif
