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
#include "fltx_common_exact.h"
#include "fltx_math_utils.h"

namespace bl {

/// ------------------ math ------------------

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr bool f128_try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = trunc(x);
        if (xi != x)
            return false;

        if (detail::_f128::absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = sub_inline(xi, to_f128(hi_part));
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }
    BL_FORCE_INLINE constexpr bool is_odd_integer(const f128_s& x) noexcept
    {
        int64_t value{};
        if (detail::_f128::f128_try_get_int64(x, value))
            return (value & 1ll) != 0;

        if (x.lo != 0.0 || !detail::_f128::isfinite(x.hi))
            return false;

        return detail::_f128::double_integer_is_odd(x.hi);
    }
    BL_FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        if (exp == 0)
            return f128_s{ 1.0 };

        const bool invert = exp < 0;
        uint64_t n = invert ? detail::_f128::magnitude_u64(exp) : static_cast<uint64_t>(exp);
        f128_s result{ 1.0 };

        while (n != 0)
        {
            if ((n & 1u) != 0)
                result = mul_inline(result, base);

            n >>= 1;
            if (n != 0)
                base = mul_inline(base, base);
        }

        return invert ? div_inline(f128_s{ 1.0 }, result) : result;
    }
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

    BL_FORCE_INLINE constexpr bool fmod_u128_is_zero(const fmod_u128& value)
    {
        return value.lo == 0 && value.hi == 0;
    }
    BL_FORCE_INLINE constexpr bool fmod_u128_is_odd(const fmod_u128& value)
    {
        return (value.lo & 1u) != 0;
    }
    BL_FORCE_INLINE constexpr int fmod_u128_compare(const fmod_u128& a, const fmod_u128& b)
    {
        if (a.hi < b.hi) return -1;
        if (a.hi > b.hi) return 1;
        if (a.lo < b.lo) return -1;
        if (a.lo > b.lo) return 1;
        return 0;
    }
    BL_FORCE_INLINE constexpr int fmod_u128_bit_length(const fmod_u128& value)
    {
        if (value.hi != 0)
            return 128 - static_cast<int>(std::countl_zero(value.hi));
        if (value.lo != 0)
            return 64 - static_cast<int>(std::countl_zero(value.lo));
        return 0;
    }
    BL_FORCE_INLINE constexpr int fmod_u128_trailing_zero_bits(const fmod_u128& value)
    {
        if (value.lo != 0)
            return static_cast<int>(std::countr_zero(value.lo));
        if (value.hi != 0)
            return 64 + static_cast<int>(std::countr_zero(value.hi));
        return 0;
    }
    BL_FORCE_INLINE constexpr bool fmod_u128_get_bit(const fmod_u128& value, int index)
    {
        if (index < 0 || index >= 128)
            return false;
        if (index < 64)
            return ((value.lo >> index) & 1u) != 0;
        return ((value.hi >> (index - 64)) & 1u) != 0;
    }
    BL_FORCE_INLINE constexpr std::uint64_t fmod_u128_get_bits(const fmod_u128& value, int start, int count)
    {
        std::uint64_t out = 0;
        for (int i = 0; i < count; ++i)
        {
            if (fmod_u128_get_bit(value, start + i))
                out |= (std::uint64_t{ 1 } << i);
        }
        return out;
    }
    BL_FORCE_INLINE constexpr bool fmod_u128_any_low_bits_set(const fmod_u128& value, int count)
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
    BL_FORCE_INLINE constexpr void fmod_u128_add_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += b.lo;
        a.hi += b.hi + (a.lo < old_lo ? 1u : 0u);
    }
    BL_FORCE_INLINE constexpr void fmod_u128_add_small(fmod_u128& a, std::uint32_t value)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += value;
        if (a.lo < old_lo)
            ++a.hi;
    }
    BL_FORCE_INLINE constexpr void fmod_u128_sub_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t borrow = (a.lo < b.lo) ? 1u : 0u;
        a.lo -= b.lo;
        a.hi -= b.hi + borrow;
    }
    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shl_bits(fmod_u128 value, int bits)
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
    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shr_bits(fmod_u128 value, int bits)
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
    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_shl1(fmod_u128 value)
    {
        return { value.lo << 1, (value.hi << 1) | (value.lo >> 63) };
    }
    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_mod_shift_subtract(fmod_u128 numerator, const fmod_u128& denominator)
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
    BL_FORCE_INLINE constexpr fmod_u128 fmod_u128_double_mod(fmod_u128 value, const fmod_u128& modulus)
    {
        value = fmod_u128_shl1(value);
        if (fmod_u128_compare(value, modulus) >= 0)
            fmod_u128_sub_inplace(value, modulus);
        return value;
    }
    BL_FORCE_INLINE constexpr exact_dyadic_fmod exact_from_double_fmod(double value)
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
    BL_FORCE_INLINE constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
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
    BL_FORCE_INLINE constexpr exact_dyadic_fmod exact_from_f128_fmod(const f128_s& value)
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
    BL_FORCE_INLINE constexpr bool fmod_fast_double_divisor_abs(const f128_s& ax, double ay, f128_s& out)
    {
        if (!(ay > 0.0) || !detail::_f128::isfinite(ay))
            return false;

        const f128_s mod{ ay, 0.0 };

        if (ax.lo == 0.0)
        {
            out = f128_s{ detail::_f128::fmod_constexpr(ax.hi, ay), 0.0 };
            return true;
        }

        const double rh = (ax.hi < ay) ? ax.hi : detail::_f128::fmod_constexpr(ax.hi, ay);
        const double rl = (detail::_f128::absd(ax.lo) < ay) ? ax.lo : detail::_f128::fmod_constexpr(ax.lo, ay);

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
    BL_FORCE_INLINE constexpr f128_s exact_dyadic_to_f128_fmod(const fmod_u128& coeff, int exp2, bool neg)
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

        f128_s out = detail::_f128::renorm(hi, lo);
        return neg ? -out : out;
    }
    BL_FORCE_INLINE constexpr f128_s fmod_exact_fixed_limb(const f128_s& x, const f128_s& y)
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
            return f128_s{ detail::_f128::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return out;
    }
    BL_FORCE_INLINE constexpr int frexp_exponent(double value) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::frexp_exponent_constexpr(value);

        int exponent = 0;
        (void)std::frexp(value, &exponent);
        return exponent;
    }
    BL_FORCE_INLINE constexpr double ldexp_limb(double value, int exponent) noexcept
    {
        if (bl::use_constexpr_math())
            return detail::fp::ldexp_constexpr2(value, exponent);

        return std::ldexp(value, exponent);
    }
    BL_FORCE_INLINE constexpr f128_s ldexp_terms(const f128_s& value, int exponent) noexcept
    {
        return detail::_f128::renorm(
            detail::_f128::ldexp_limb(value.hi, exponent),
            detail::_f128::ldexp_limb(value.lo, exponent));
    }
    BL_FORCE_INLINE constexpr f128_s pack_decimal_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
    {
        const std::uint64_t c1 = q.get_bits(0, 53);
        const std::uint64_t c0 = q.get_bits(53, 53);
        const double hi = c0 ? detail::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? detail::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = detail::_f128::renorm(hi, lo);
        return neg ? -out : out;
    }
    BL_NO_INLINE constexpr f128_s round_decimal_exact_to_f128(const detail::exact_decimal::biguint& coeff, int dec_exp, bool neg) noexcept
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

        return detail::_f128::pack_decimal_significand(q, e2, neg);
    }
    BL_FORCE_INLINE constexpr bool try_rounded_decimal_to_f128(const f128_s& integer_part, const char* digits, int digit_count, bool neg, f128_s& out) noexcept
    {
        int64_t integer_value = 0;
        if (!detail::_f128::f128_try_get_int64(integer_part, integer_value) || integer_value < 0)
            return false;

        detail::exact_decimal::biguint coeff{ static_cast<std::uint64_t>(integer_value) };
        for (int i = 0; i < digit_count; ++i)
        {
            coeff.mul_small(10);
            coeff.add_small(static_cast<std::uint32_t>(digits[i] - '0'));
        }

        out = detail::_f128::round_decimal_exact_to_f128(coeff, -digit_count, neg);
        return true;
    }
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y)
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
    if (y.lo == 0.0 && detail::_f128::fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ detail::_f128::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return detail::_f128::fmod_exact_fixed_limb(x, y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s round(const f128_s& a)
{
    using namespace detail::_f128;

    f128_s t = floor(add_inline(a, f128_s{ 0.5 }));
    if (sub_inline(t, a) == f128_s{ 0.5 } && fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t = sub_inline(t, f128_s{ 1.0 });
    return t;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s round_to_decimals(f128_s v, int prec)
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

    f128_s ip = floor(v);
    f128_s frac = detail::_f128::sub_inline(v, ip);

    f128_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = detail::_f128::mul_inline(w, f128_s{ 10.0 });

        int di = static_cast<int>(floor(w).hi);
        if (di < 0) di = 0;
        else if (di > 9) di = 9;

        digits[i] = static_cast<char>('0' + di);
        w = detail::_f128::sub_inline(w, f128_s{ static_cast<double>(di) });
    }

    f128_s la = detail::_f128::mul_inline(w, f128_s{ 10.0 });

    const f128_s tie_slop = detail::_f128::mul_inline(f128_s::eps(), f128_s{ 65536.0 });
    int next = static_cast<int>(floor(la).hi);
    if (next < 0) next = 0;

    f128_s rem = detail::_f128::sub_inline(la, f128_s{ static_cast<double>(next) });
    if (next < 10 && rem >= detail::_f128::sub_inline(f128_s{ 1.0 }, tie_slop))
    {
        ++next;
        rem = detail::_f128::sub_inline(rem, f128_s{ 1.0 });
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
            ip = detail::_f128::add_inline(ip, f128_s{ 1.0 });
    }

    f128_s exact_out{};
    if (detail::_f128::try_rounded_decimal_to_f128(ip, digits, prec, neg, exact_out))
        return exact_out;

    f128_s frac_val{ 0.0, 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = detail::_f128::add_inline(
            frac_val,
            f128_s{ static_cast<double>(digits[i] - '0') });

        frac_val = detail::_f128::mul_inline(frac_val, INV10_DD);
    }

    f128_s out = detail::_f128::add_inline(ip, frac_val);
    return neg ? -out : out;
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s ay = abs(y);
    f128_s r = fmod(x, y);
    const f128_s ar = abs(r);
    const f128_s half = detail::_f128::mul_inline(ay, f128_s{ 0.5 });

    if (ar > half)
    {
        r = detail::_f128::add_inline(r, signbit(r) ? ay : -ay);
    }
    else if (ar == half)
    {
        const f128_s q = trunc(detail::_f128::div_inline(x, y));
        const f128_s q_mod2 = abs(fmod(q, f128_s{ 2.0 }));
        if (q_mod2 != f128_s{ 0.0 })
            r = detail::_f128::add_inline(r, signbit(r) ? ay : -ay);
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return detail::_f128::canonicalize_math_result(r);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    const int exp2 = detail::_f128::frexp_exponent(a.hi);
    const int result_scale = exp2 / 2;
    const int input_scale = -2 * result_scale;
    const f128_s scaled_a = input_scale == 0 ? a : detail::_f128::ldexp_terms(a, input_scale);

    double y0;
    if (bl::use_constexpr_math()) {
        y0 = detail::_f128::sqrt_seed_constexpr(scaled_a.hi);
        f128_s y{ y0 };
        y = detail::_f128::add_inline(y, detail::_f128::div_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), detail::_f128::add_inline(y, y)));
        y = detail::_f128::add_inline(y, detail::_f128::div_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), detail::_f128::add_inline(y, y)));
        y = detail::_f128::add_inline(y, detail::_f128::div_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), detail::_f128::add_inline(y, y)));

        if (result_scale != 0)
            y = detail::_f128::ldexp_terms(y, result_scale);

        return detail::_f128::canonicalize_math_result(y);
    }
    else {
        y0 = std::sqrt(scaled_a.hi);
        f128_s y{ y0 };
        y = detail::_f128::add_inline(y, detail::_f128::div_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), detail::_f128::add_inline(y, y)));
        y = detail::_f128::add_inline(y, detail::_f128::div_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), detail::_f128::add_inline(y, y)));
        y = detail::_f128::add_inline(y, detail::_f128::mul_inline(detail::_f128::sub_mul_inline(scaled_a, y, y), f128_s{ 0.5 / y0 }));

        if (result_scale != 0)
            y = detail::_f128::ldexp_terms(y, result_scale);

        return detail::_f128::canonicalize_math_result(y);
    }
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nearbyint(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f128_s t = floor(a);
    f128_s frac = detail::_f128::sub_inline(a, t);

    if (frac < f128_s{ 0.5 })
        return t;

    if (frac > f128_s{ 0.5 })
    {
        t = detail::_f128::add_inline(t, f128_s{ 1.0 });
        if (iszero(t))
            return f128_s{ detail::_f128::signbit_constexpr(a.hi) ? -0.0 : 0.0 };
        return t;
    }

    if (fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t = detail::_f128::add_inline(t, f128_s{ 1.0 });

    if (iszero(t))
        return f128_s{ detail::_f128::signbit_constexpr(a.hi) ? -0.0 : 0.0 };

    return t;
}

/// ------------------ transcendentals ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f128_s a)
{
    const double hi = a.hi;
    if (hi <= 0.0)
        return bl::detail::fp::log_constexpr(static_cast<double>(a));

    return bl::detail::fp::log_constexpr(hi) + bl::detail::fp::log1p_constexpr(a.lo / hi);
}

namespace detail::_f128
{
    inline constexpr f128_s e          = std::numbers::e_v<f128_s>;
    inline constexpr f128_s log2e      = std::numbers::log2e_v<f128_s>;
    inline constexpr f128_s log10e     = std::numbers::log10e_v<f128_s>;
    inline constexpr f128_s pi         = std::numbers::pi_v<f128_s>;
    inline constexpr f128_s inv_pi     = std::numbers::inv_pi_v<f128_s>;
    inline constexpr f128_s inv_sqrtpi = std::numbers::inv_sqrtpi_v<f128_s>;
    inline constexpr f128_s ln2        = std::numbers::ln2_v<f128_s>;
    inline constexpr f128_s ln10       = std::numbers::ln10_v<f128_s>;
    inline constexpr f128_s sqrt2      = std::numbers::sqrt2_v<f128_s>;
    inline constexpr f128_s sqrt3      = std::numbers::sqrt3_v<f128_s>;
    inline constexpr f128_s inv_sqrt3  = std::numbers::inv_sqrt3_v<f128_s>;
    inline constexpr f128_s egamma     = std::numbers::egamma_v<f128_s>;
    inline constexpr f128_s phi        = std::numbers::phi_v<f128_s>;

    inline constexpr f128_s pi_2      = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54 };
    inline constexpr f128_s pi_4      = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55 };
    inline constexpr f128_s invpi2    = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55 };

    inline constexpr f128_s inv_ln2   = log2e;
    inline constexpr f128_s inv_ln10  = log10e;
    inline constexpr f128_s sqrt_half = { 0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55 };
    inline constexpr f128_s half_log_two_pi = { 0x1.d67f1c864beb5p-1, -0x1.65b5a1b7ff5dfp-55 };

    inline constexpr double pi_4_hi = detail::_f128::pi_4.hi;
    inline constexpr double pi_2_hi_d = 0x1.921fb54442d18p+0;
    inline constexpr double pi_2_mid_d = 0x1.1a62633145c07p-54;
    inline constexpr double pi_2_lo_d = -0x1.f1976b7ed8fbcp-110;

    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::double_integer_is_odd;
    using detail::fp::fmod_constexpr;
    using detail::fp::sqrt_seed_constexpr;
    using detail::fp::nearbyint_ties_even;

    inline constexpr f128_s exp_inv_fact[] = {
        f128_s{ 1.66666666666666657e-01,  9.25185853854297066e-18 },
        f128_s{ 4.16666666666666644e-02,  2.31296463463574266e-18 },
        f128_s{ 8.33333333333333322e-03,  1.15648231731787138e-19 },
        f128_s{ 1.38888888888888894e-03, -5.30054395437357706e-20 },
        f128_s{ 1.98412698412698413e-04,  1.72095582934207053e-22 },
        f128_s{ 2.48015873015873016e-05,  2.15119478667758816e-23 },
        f128_s{ 2.75573192239858925e-06, -1.85839327404647208e-22 },
        f128_s{ 2.75573192239858883e-07,  2.37677146222502973e-23 },
        f128_s{ 2.50521083854417202e-08, -1.44881407093591197e-24 },
        f128_s{ 2.08767569878681002e-09, -1.20734505911325997e-25 },
        f128_s{ 1.60590438368216133e-10,  1.25852945887520981e-26 },
        f128_s{ 1.14707455977297245e-11,  2.06555127528307454e-28 },
        f128_s{ 7.64716373181981641e-13,  7.03872877733453001e-30 },
        f128_s{ 4.77947733238738525e-14,  4.39920548583408126e-31 },
        f128_s{ 2.81145725434552060e-15,  1.65088427308614326e-31 }
    };

    BL_NO_INLINE constexpr f128_s f128_log1p_series_reduced(const f128_s& x)
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



    inline constexpr f128_s lgamma1p_coeff[] = {
        f128_s{ 0x1.a51a6625307d3p-1, 0x1.1873d8912200cp-56 },
        f128_s{ -0x1.9a4d55beab2d7p-2, 0x1.4c26d1b465993p-59 },
        f128_s{ 0x1.151322ac7d848p-2, 0x1.b5f91211196e5p-57 },
        f128_s{ -0x1.a8b9c17aa6149p-3, -0x1.2e826a4fdae1ap-58 },
        f128_s{ 0x1.5b40cb100c306p-3, 0x1.4a79940f15696p-59 },
        f128_s{ -0x1.2703a1dcea3aep-3, -0x1.6307fd0794ac4p-57 },
        f128_s{ 0x1.010b36af86397p-3, -0x1.741a635b224a6p-59 },
        f128_s{ -0x1.c806706d57db4p-4, -0x1.56aa806fdd3eep-58 },
        f128_s{ 0x1.9a01e385d5f8fp-4, 0x1.813418f3768cdp-59 },
        f128_s{ -0x1.748c33114c6d6p-4, -0x1.ea57624080720p-61 },
        f128_s{ 0x1.556ad63243bc4p-4, 0x1.5de8580fae81dp-62 },
        f128_s{ -0x1.3b1d971fc5985p-4, 0x1.e58607e493dfdp-59 },
        f128_s{ 0x1.2496df8320c5fp-4, 0x1.cf4b4ae040be8p-58 },
        f128_s{ -0x1.11133476e7fe0p-4, -0x1.dc9a4ff396ee3p-59 },
        f128_s{ 0x1.00010064cdeb2p-4, 0x1.7879d0156affep-59 },
        f128_s{ -0x1.e1e2d311e8abdp-5, 0x1.8d2a110ce956bp-59 },
        f128_s{ 0x1.c71ce3a20b419p-5, -0x1.be9617d035b06p-59 },
        f128_s{ -0x1.af28a1b5688a0p-5, -0x1.74741e885fefbp-59 },
        f128_s{ 0x1.9999b3352d5bap-5, 0x1.4951b4c6be56dp-62 },
        f128_s{ -0x1.86186db77bfbfp-5, -0x1.6dedef1f58778p-59 },
        f128_s{ 0x1.745d1d1778df9p-5, 0x1.02b8fe0a898e7p-61 },
        f128_s{ -0x1.642c88591b66dp-5, 0x1.1074551cafc60p-59 },
        f128_s{ 0x1.555556aaafdcdp-5, 0x1.54a05fce04ef6p-59 },
        f128_s{ -0x1.47ae151eb9fb7p-5, -0x1.d038d4d4653c2p-59 },
        f128_s{ 0x1.3b13b189d925ep-5, 0x1.f4ad5a89f860cp-59 },
        f128_s{ -0x1.2f684c00002bcp-5, -0x1.055a3ba5e6a12p-59 },
        f128_s{ 0x1.24924936db7bcp-5, 0x1.f2631c34f2cbcp-59 },
        f128_s{ -0x1.1a7b961a7b9aap-5, 0x1.e116d2f11b9bcp-59 },
        f128_s{ 0x1.111111155556dp-5, -0x1.527ce242d7c8fp-59 },
        f128_s{ -0x1.08421086318cep-5, 0x1.1db4d8fcae8c6p-59 },
        f128_s{ 0x1.0000000100002p-5, 0x1.b8fd913d3546ap-59 },
        f128_s{ -0x1.f07c1f08ba2eap-6, -0x1.31bb2e9036633p-60 },
        f128_s{ 0x1.e1e1e1e25a5a6p-6, 0x1.3e46eaa03f9ccp-61 },
        f128_s{ -0x1.d41d41d457c58p-6, 0x1.0600661f0f0e3p-62 },
        f128_s{ 0x1.c71c71c738e39p-6, -0x1.d93a55599cf57p-63 },
        f128_s{ -0x1.bacf914c29837p-6, -0x1.797fe7c73f29ap-60 },
        f128_s{ 0x1.af286bca21af3p-6, -0x1.df4d835f028bdp-60 },
        f128_s{ -0x1.a41a41a41d89ep-6, 0x1.d6bf77cbc25c7p-60 },
        f128_s{ 0x1.999999999b333p-6, 0x1.9ad0584412591p-61 },
        f128_s{ -0x1.8f9c18f9c2577p-6, 0x1.766fd061292d7p-60 },
        f128_s{ 0x1.8618618618c31p-6, -0x1.e77d97e1c5a45p-61 },
        f128_s{ -0x1.7d05f417d08eep-6, -0x1.1dcf2bd1488c1p-61 },
        f128_s{ 0x1.745d1745d18bap-6, 0x1.7460941753bf5p-61 },
        f128_s{ -0x1.6c16c16c16ccdp-6, 0x1.9998769b89af0p-61 },
        f128_s{ 0x1.642c8590b21bdp-6, 0x1.bd3805d865a75p-61 },
        f128_s{ -0x1.5c9882b931083p-6, 0x1.1b3bdabc05a8dp-60 },
        f128_s{ 0x1.555555555556bp-6, -0x1.555550480911cp-60 },
        f128_s{ -0x1.4e5e0a72f0544p-6, 0x1.4e5e03d9bbd88p-62 },
        f128_s{ 0x1.47ae147ae1480p-6, 0x1.13e7474dcd9a5p-85 },
        f128_s{ -0x1.4141414141417p-6, 0x1.a5a5a57890971p-60 },
        f128_s{ 0x1.3b13b13b13b15p-6, -0x1.3b13b1001f8aep-62 },
        f128_s{ -0x1.3521cfb2b78c2p-6, 0x1.826a4395c1891p-61 },
        f128_s{ 0x1.2f684bda12f69p-6, -0x1.a12f684a465ffp-60 },
        f128_s{ -0x1.29e4129e4129ep-6, -0x1.9999999a1db84p-60 },
        f128_s{ 0x1.2492492492492p-6, 0x1.6db6db6de21c5p-60 }
    };

    inline constexpr f128_s lgamma1p5_coeff[] = {
        f128_s{ 0x1.de9e64df22ef3p-2, -0x1.6d48ec9933fbap-57 },
        f128_s{ -0x1.1ae55b180726cp-3, -0x1.959aeebbe37a9p-59 },
        f128_s{ 0x1.e0f840dad61dap-5, -0x1.599fc3fe0a24cp-59 },
        f128_s{ -0x1.da59d5374a543p-6, -0x1.0628c23cf6fdcp-63 },
        f128_s{ 0x1.f9ca39daa929cp-7, -0x1.69e59f1067e8fp-67 },
        f128_s{ -0x1.1a8ba4f0ea597p-7, -0x1.7d1f4799cdd85p-61 },
        f128_s{ 0x1.456f1ad666a3bp-8, -0x1.3247be39407adp-62 },
        f128_s{ -0x1.7edb812f6426ep-9, -0x1.5cb4446f39441p-64 },
        f128_s{ 0x1.c9735ae9db2c1p-10, -0x1.00df931d99976p-65 },
        f128_s{ -0x1.148a319eec639p-10, 0x1.8b9481cc9d8c5p-66 },
        f128_s{ 0x1.517c5a1579f10p-11, -0x1.6de593e736460p-65 },
        f128_s{ -0x1.9eff1d1c8bdc2p-12, -0x1.b074a9d2f567bp-68 },
        f128_s{ 0x1.00c41c13e4c1cp-12, 0x1.23a0bd176970bp-66 },
        f128_s{ -0x1.3f6dff22ac1c2p-13, 0x1.c991818178cf5p-68 },
        f128_s{ 0x1.8f3619541742cp-14, -0x1.9194563cfb41ap-69 },
        f128_s{ -0x1.f4ea079c9c87ap-15, -0x1.f4d6e8d46504bp-71 },
        f128_s{ 0x1.3b5e73f18d398p-15, 0x1.c67a3cfb2c122p-70 },
        f128_s{ -0x1.8e583480fb843p-16, 0x1.f69ef26bd25fap-75 },
        f128_s{ 0x1.f88eb43555368p-17, -0x1.9028d0c42fa7ep-74 },
        f128_s{ -0x1.4059677eed115p-17, 0x1.7b1825e75d8d6p-73 },
        f128_s{ 0x1.97b6b03fa7446p-18, -0x1.6f01f1c1c4be3p-72 },
        f128_s{ -0x1.03fd6bf0808efp-18, -0x1.fd918e6fbcaebp-72 },
        f128_s{ 0x1.4c355353d5241p-19, -0x1.8b80de6628253p-76 },
        f128_s{ -0x1.a939cf6ab6697p-20, -0x1.4c9d10090b8a2p-74 },
        f128_s{ 0x1.109491756a3f0p-20, 0x1.76bf49056fc2fp-74 },
        f128_s{ -0x1.5dfabe1235651p-21, 0x1.cdba663360cb3p-76 },
        f128_s{ 0x1.c1f93171f89d3p-22, -0x1.e4e55b85afbc8p-76 },
        f128_s{ -0x1.21a3531259833p-22, 0x1.bd691063eee70p-77 },
        f128_s{ 0x1.754fa60ab8b7ap-23, -0x1.ac1c40cde5565p-77 },
        f128_s{ -0x1.e1b1158537c59p-24, -0x1.a038cdbafc436p-79 },
        f128_s{ 0x1.3717b2266f892p-24, 0x1.4c81741efeb59p-79 },
        f128_s{ -0x1.92387e0fdf9f7p-25, -0x1.49d2a7a71b29dp-79 },
        f128_s{ 0x1.0442ab98bfc68p-25, 0x1.92553e7d00a21p-79 },
        f128_s{ -0x1.51196689e7eeep-26, 0x1.a18474ec60c9dp-81 },
        f128_s{ 0x1.b4fafffb9d100p-27, 0x1.a3302fc538f03p-83 },
        f128_s{ -0x1.1b7260c6f0a98p-27, -0x1.e26f947be4327p-82 },
        f128_s{ 0x1.6ffbc9ee7fb98p-28, -0x1.e26e0739956f6p-83 },
        f128_s{ -0x1.de1068c0e801bp-29, -0x1.6cd7c1b959661p-83 },
        f128_s{ 0x1.36bdddabf16d9p-29, -0x1.d926441c8b4c4p-83 },
        f128_s{ -0x1.943780143c19dp-30, -0x1.100cc8f4aa6e9p-84 },
        f128_s{ 0x1.070fcd4094009p-30, 0x1.f83a3a445b423p-86 },
        f128_s{ -0x1.56978e4716a05p-31, -0x1.9e4fdaf5a2dabp-85 },
        f128_s{ 0x1.be68640e30872p-32, -0x1.5a5aae6418194p-89 },
        f128_s{ -0x1.22fde267b9daep-32, -0x1.9c6f37412f9cdp-87 },
        f128_s{ 0x1.7b8defa86bdb7p-33, 0x1.7abe7975fce0dp-91 },
        f128_s{ -0x1.ef4e19e105fa4p-34, 0x1.8f318f9909851p-91 },
        f128_s{ 0x1.4352fb8f40e4ep-34, -0x1.33c691a77efa8p-88 },
        f128_s{ -0x1.a64d09df9d496p-35, 0x1.c17c3d74dc13ap-89 },
        f128_s{ 0x1.13e73d105cf63p-35, 0x1.f93f22cf4af99p-89 },
        f128_s{ -0x1.68a86a97e144dp-36, 0x1.b5c3a7b89ba21p-90 },
        f128_s{ 0x1.d7a128edfb44ap-37, -0x1.3be7aba9d356bp-93 },
        f128_s{ -0x1.347cbbc72064fp-37, 0x1.71403302e085ap-91 },
        f128_s{ 0x1.93b308b268643p-38, 0x1.195a1b5637c85p-92 },
        f128_s{ -0x1.083d54d1dd741p-38, -0x1.a77a9ad2416c5p-94 },
        f128_s{ 0x1.5a072c06a19ffp-39, 0x1.a277f2aa37865p-93 },
        f128_s{ -0x1.c546c66581161p-40, -0x1.52266404da246p-94 },
        f128_s{ 0x1.28f967804ba17p-40, 0x1.e507a5793f98cp-94 },
        f128_s{ -0x1.85411e14a2adfp-41, 0x1.e9f2f01cda3ecp-99 },
        f128_s{ 0x1.fe5b10aef75a5p-42, -0x1.be0c224692c7fp-97 },
        f128_s{ -0x1.4ea8d461f1e88p-42, 0x1.6971a22ddceddp-96 },
        f128_s{ 0x1.b7040357324f2p-43, -0x1.fb0047ded483fp-97 },
        f128_s{ -0x1.20080d0717845p-43, 0x1.92ff4fa145300p-97 },
        f128_s{ 0x1.7a0a91194edbep-44, -0x1.31aefa4aa973fp-101 },
        f128_s{ -0x1.f04ce33f6b75fp-45, 0x1.50d6b1d688eddp-99 },
        f128_s{ 0x1.45da900805e80p-45, 0x1.bbbca6a9ab3ecp-99 },
        f128_s{ -0x1.abfcade4542c8p-46, -0x1.ed48fc6d40a25p-104 },
        f128_s{ 0x1.1920f4bba0b3ap-46, 0x1.a51e5ef7eafe1p-100 },
        f128_s{ -0x1.7167e74d1d5dbp-47, -0x1.424703463268fp-101 },
        f128_s{ 0x1.e5813e9fdd73cp-48, -0x1.77856885e6f00p-103 },
        f128_s{ -0x1.3f1c7614f1b4ap-48, 0x1.f8e0424563d70p-102 },
        f128_s{ 0x1.a39275546d348p-49, 0x1.aeebff90f4bf0p-103 },
        f128_s{ -0x1.13e20e066adfep-49, -0x1.4458bd2819a7ep-103 },
        f128_s{ 0x1.6adf8811aa8e4p-50, 0x1.03b5e8b6a3a9fp-104 },
        f128_s{ -0x1.dd613b8a280e3p-51, -0x1.55e677244df9dp-105 },
        f128_s{ 0x1.3a10cf9786245p-51, -0x1.53120b9a883f2p-105 },
        f128_s{ -0x1.9d50dbffed7c4p-52, 0x1.8ea0d2722d1cbp-106 },
        f128_s{ 0x1.1002e3ee72b89p-52, 0x1.e2e040801e860p-106 },
        f128_s{ -0x1.66173f813291dp-53, 0x1.5d465546aa135p-107 },
        f128_s{ 0x1.d77c7a03b5c88p-54, -0x1.2375a930ce7ddp-109 },
        f128_s{ -0x1.3671909a25851p-54, -0x1.64135363de411p-112 },
        f128_s{ 0x1.98e0800337a90p-55, 0x1.60030a839fc44p-110 },
        f128_s{ -0x1.0d4cec7961518p-55, -0x1.ff3f70e2ec811p-114 },
        f128_s{ 0x1.62caee67064efp-56, -0x1.f8e5727c66691p-112 },
        f128_s{ -0x1.d37dd6bdf63ddp-57, -0x1.8f1ce56e5d21ap-111 },
        f128_s{ 0x1.34097d9ee7b5ap-57, 0x1.3e3d23b7c0e31p-111 },
        f128_s{ -0x1.95fec6eaf0a8dp-58, 0x1.90c9907868b95p-112 },
        f128_s{ 0x1.0b967777f0122p-58, 0x1.f7bf8ce33021dp-112 },
        f128_s{ -0x1.60c65e387cbcep-59, 0x1.a17ca2202219cp-113 },
        f128_s{ 0x1.d123e4872935bp-60, -0x1.22a19e8d18b94p-116 }
    };

    BL_NO_INLINE constexpr f128_s lgamma1p_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f128_s p = lgamma1p_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = mul_add_inline(p, y, lgamma1p_coeff[i]);

        return mul_inline(y, mul_add_inline(y, p, -detail::_f128::egamma));
    }

    BL_NO_INLINE constexpr f128_s lgamma1p5_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p5_coeff) / sizeof(lgamma1p5_coeff[0]));

        f128_s p = lgamma1p5_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = mul_add_inline(p, y, lgamma1p5_coeff[i]);

        const f128_s constant = sub_inline(detail::_f128::half_log_two_pi, mul_inline(f128_s{ 1.5 }, detail::_f128::ln2));
        const f128_s linear = sub_inline(sub_inline(f128_s{ 2.0 }, detail::_f128::egamma), mul_inline(f128_s{ 2.0 }, detail::_f128::ln2));
        return mul_add_inline(y, mul_add_inline(y, p, linear), constant);
    }

    BL_NO_INLINE constexpr bool try_lgamma_near_one_or_two(const f128_s& x, f128_s& out) noexcept
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

    BL_NO_INLINE constexpr f128_s f128_expm1_tiny(const f128_s& r)
    {
        f128_s p = exp_inv_fact[(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 1];
        for (int i = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 2; i >= 0; --i)
            p = mul_add_inline(p, r, exp_inv_fact[i]);
        p = mul_add_inline(p, r, f128_s{0.5});
        return mul_add_inline(mul_inline(r, r), p, r);
    }

    BL_FORCE_INLINE constexpr bool f128_remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
	{
	    const double ax = detail::_f128::fabs_constexpr(x.hi);
	    if (!detail::_f128::isfinite(ax))
	        return false;

	    if (ax > 7.0e15)
	        return false;

	    const f128_s t = mul_inline(x, detail::_f128::invpi2);

	    double qd = detail::_f128::nearbyint_ties_even(t.hi);
	    if (!detail::_f128::isfinite(qd) ||
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

	    constexpr f128_s pi_2_hi{ detail::_f128::pi_2_hi_d };
	    constexpr f128_s pi_2_mid{ detail::_f128::pi_2_mid_d };
	    constexpr f128_s pi_2_lo{ detail::_f128::pi_2_lo_d };
	    constexpr f128_s pi_4{ detail::_f128::pi_4_hi };

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
    BL_FORCE_INLINE constexpr f128_s f128_sin_kernel_pi4(const f128_s& x)
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
    BL_FORCE_INLINE constexpr f128_s f128_cos_kernel_pi4(const f128_s& x)
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
    BL_FORCE_INLINE constexpr void f128_sincos_kernel_pi4(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        s_out = f128_sin_kernel_pi4(x);
        c_out = f128_cos_kernel_pi4(x);
    }


    BL_FORCE_INLINE constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        if (bl::use_constexpr_math())
        {
            return detail::_f128::renorm(
                detail::fp::ldexp_constexpr2(x.hi, e),
                detail::fp::ldexp_constexpr2(x.lo, e)
            );
        }
        else
        {
            return detail::_f128::renorm(
                std::ldexp(x.hi, e),
                std::ldexp(x.lo, e)
            );
        }
    }
    BL_NO_INLINE constexpr f128_s _exp(const f128_s& x)
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

        const f128_s t = mul_inline(x, detail::_f128::inv_ln2);

        double kd = detail::_f128::nearbyint_ties_even(t.hi);
        const f128_s delta = sub_inline(t, f128_s{ kd });
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s r = mul_inline(sub_inline(x, mul_inline(f128_s{ kd }, detail::_f128::ln2)), f128_s{ 0.0009765625 });

        f128_s e = detail::_f128::f128_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = mul_inline(e, add_inline(e, f128_s{ 2.0 }));

        return _ldexp(add_inline(e, f128_s{ 1.0 }), k);
    }
    BL_NO_INLINE constexpr f128_s _log(const f128_s& a)
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
        if (m < detail::_f128::sqrt_half)
        {
            m = mul_inline(m, f128_s{ 2.0 });
            --exp2;
        }

        const f128_s exp2_ln2 = mul_inline(f128_s{ static_cast<double>(exp2) }, detail::_f128::ln2);
        f128_s y = add_inline(exp2_ln2, f128_s{ log_as_double(m) });
        y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
        y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
        y = add_inline(y, mul_sub_inline(m, _exp(sub_inline(exp2_ln2, y)), f128_s{ 1.0 }));
        return y;
    }
}

BL_NO_INLINE    constexpr f128_s pow10_128(int k);

// exp
[[nodiscard]] BL_NO_INLINE constexpr f128_s ldexp(const f128_s& x, int e)
{
    return detail::_f128::canonicalize_math_result(detail::_f128::_ldexp(x, e));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s exp(const f128_s& x)
{
    return detail::_f128::canonicalize_math_result(detail::_f128::_exp(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s exp2(const f128_s& x)
{
    return detail::_f128::canonicalize_math_result(detail::_f128::_exp(detail::_f128::mul_inline(x, detail::_f128::ln2)));
}

// log
[[nodiscard]] BL_NO_INLINE constexpr f128_s log(const f128_s& a)
{
    return detail::_f128::canonicalize_math_result(detail::_f128::_log(a));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log2(const f128_s& a)
{
    int exact_exp2{};
    if (detail::_f128::f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return detail::_f128::canonicalize_math_result(detail::_f128::mul_inline(detail::_f128::_log(a), detail::_f128::inv_ln2));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log10(const f128_s& x)
{
    if (x.hi > 0.0)
    {
        const int exp2 =
            detail::fp::frexp_exponent_constexpr(x.hi);
        const int k0 =
            static_cast<int>(detail::fp::floor_constexpr((exp2 - 1) * 0.30102999566398114));

        for (int k = k0 - 2; k <= k0 + 2; ++k)
        {
            if (x == pow10_128(k))
                return f128_s{ static_cast<double>(k), 0.0 };
        }
    }

    return detail::_f128::canonicalize_math_result(detail::_f128::mul_inline(detail::_f128::_log(x), detail::_f128::inv_ln10));
}

// pow
[[nodiscard]] BL_NO_INLINE constexpr f128_s pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s yi = trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && detail::_f128::f128_try_get_int64(yi, yi64))
        return detail::_f128::powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && detail::_f128::signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = detail::_f128::_exp(detail::_f128::mul_inline(y, detail::_f128::_log(-x)));
        return detail::_f128::is_odd_integer(yi) ? -magnitude : magnitude;
    }

    return detail::_f128::canonicalize_math_result(detail::_f128::_exp(detail::_f128::mul_inline(y, detail::_f128::_log(x))));
}


// trig
[[nodiscard]] BL_NO_INLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    const double ax = detail::_f128::fabs_constexpr(x.hi);
    if (!detail::_f128::isfinite(ax))
    {
        s_out = f128_s{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= detail::_f128::pi_4_hi)
    {
        detail::_f128::f128_sincos_kernel_pi4(x, s_out, c_out);
        s_out = detail::_f128::canonicalize_math_result(s_out);
        c_out = detail::_f128::canonicalize_math_result(c_out);
        return true;
    }

    long long n = 0;
    f128_s r{};
    if (!detail::_f128::f128_remainder_pio2(x, n, r))
        return false;

    f128_s sr{}, cr{};
    detail::_f128::f128_sincos_kernel_pi4(r, sr, cr);

    switch ((int)(n & 3))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    s_out = detail::_f128::canonicalize_math_result(s_out);
    c_out = detail::_f128::canonicalize_math_result(c_out);
    return true;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s sin(const f128_s& x)
{
    const double ax = detail::_f128::fabs_constexpr(x.hi);
    if (!detail::_f128::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= detail::_f128::pi_4_hi)
        return detail::_f128::canonicalize_math_result(detail::_f128::f128_sin_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!detail::_f128::f128_remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return detail::_f128::canonicalize_math_result(f128_s{ detail::fp::sin_constexpr(static_cast<double>(x)) });
        }
        else
        {
            return detail::_f128::canonicalize_math_result(f128_s{ std::sin((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return detail::_f128::canonicalize_math_result(detail::_f128::f128_sin_kernel_pi4(r));
    case 1: return detail::_f128::canonicalize_math_result(detail::_f128::f128_cos_kernel_pi4(r));
    case 2: return detail::_f128::canonicalize_math_result(-detail::_f128::f128_sin_kernel_pi4(r));
    default: return detail::_f128::canonicalize_math_result(-detail::_f128::f128_cos_kernel_pi4(r));
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s cos(const f128_s& x)
{
    const double ax = detail::_f128::fabs_constexpr(x.hi);
    if (!detail::_f128::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= detail::_f128::pi_4_hi)
        return detail::_f128::canonicalize_math_result(detail::_f128::f128_cos_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!detail::_f128::f128_remainder_pio2(x, n, r))
    {
        if (bl::use_constexpr_math())
        {
            return detail::_f128::canonicalize_math_result(f128_s{ detail::fp::cos_constexpr(static_cast<double>(x)) });
        }
        else 
        {
            return detail::_f128::canonicalize_math_result(f128_s{ std::cos((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return detail::_f128::canonicalize_math_result(detail::_f128::f128_cos_kernel_pi4(r));
    case 1: return detail::_f128::canonicalize_math_result(-detail::_f128::f128_sin_kernel_pi4(r));
    case 2: return detail::_f128::canonicalize_math_result(-detail::_f128::f128_cos_kernel_pi4(r));
    default: return detail::_f128::canonicalize_math_result(detail::_f128::f128_sin_kernel_pi4(r));
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tan(const f128_s& x)
{
    f128_s s{}, c{};
    if (sincos(x, s, c))
        return detail::_f128::div_inline(s, c);
    const double xd = (double)x;
    if (bl::use_constexpr_math()) {
        return f128_s{ detail::fp::tan_constexpr(xd) };
    } else {
        return f128_s{ std::tan(xd) };
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x)
{
    using namespace detail::_f128;

    if (iszero(x)) [[unlikely]]
    {
        if (iszero(y)) [[unlikely]]
            return f128_s{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? pi_2 : -pi_2;
    }

    const f128_s scale = std::max(abs(x), abs(y));
    //const f128_s xs = x / scale;
    //const f128_s ys = y / scale;
    const f128_s xs = div_inline(x, scale);
    const f128_s ys = div_inline(y, scale);

    f128_s v{ detail::fp::atan2_constexpr(y.hi, x.hi) };

    for (int i = 0; i < 2; ++i)
    {
        f128_s sv{}, cv{};
        if (!sincos(v, sv, cv))
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

        //const f128_s f  = xs * sv - ys * cv;
        //const f128_s fp = xs * cv + ys * sv;
        //v = v - f / fp;
        const f128_s f  = diff_products_inline(xs, sv, ys, cv);
        const f128_s fp = sum_products_inline(xs, cv, ys, sv);

        v = sub_inline(v, div_inline(f, fp));
    }

    return detail::_f128::canonicalize_math_result(v);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s atan(const f128_s& x)
{
    return atan2(x, f128_s{ 1.0 });
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s asin(const f128_s& x)
{
    using namespace detail::_f128;
    return atan2(x, sqrt(sub_inline(f128_s{ 1.0 }, mul_inline(x, x))));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s acos(const f128_s& x)
{
    using namespace detail::_f128;
    return atan2(sqrt(sub_inline(f128_s{ 1.0 }, mul_inline(x, x))), x);
}


[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fabs(const f128_s& a) noexcept
{
    return abs(a);
}


namespace detail::_f128
{
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
    BL_FORCE_INLINE constexpr double nextafter_double_constexpr(double from, double to) noexcept
    {
        if (detail::fp::isnan(from) || detail::fp::isnan(to))
            return std::numeric_limits<double>::quiet_NaN();

        if (from == to)
            return to;

        if (from == 0.0)
            return detail::fp::signbit_constexpr(to)
                ? -std::numeric_limits<double>::denorm_min()
                :  std::numeric_limits<double>::denorm_min();

        std::uint64_t bits = std::bit_cast<std::uint64_t>(from);
        if ((from > 0.0) == (from < to))
            ++bits;
        else
            --bits;

        return std::bit_cast<double>(bits);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f128_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        if (isnan(x) || isinf(x))
            return 0;

        const f128_s lo = to_f128(static_cast<int64_t>(std::numeric_limits<SignedInt>::lowest()));
        const f128_s hi = to_f128(static_cast<int64_t>(std::numeric_limits<SignedInt>::max()));
        if (x < lo || x > hi)
            return 0;

        int64_t out = 0;
        if (!detail::_f128::f128_try_get_int64(x, out))
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
    
    BL_NO_INLINE constexpr f128_s lgamma_stirling_asymptotic(const f128_s& z) noexcept
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

        return add_inline(add_inline(sub_inline(mul_inline(sub_inline(z, f128_s{ 0.5 }), log(z)), z), detail::_f128::half_log_two_pi), series);
    }
    BL_NO_INLINE constexpr void positive_recurrence_product(const f128_s& x, const f128_s& asymptotic_min, f128_s& z, f128_s& product, int& product_scale2) noexcept
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
    BL_NO_INLINE constexpr f128_s lgamma_positive_low_range(const f128_s& x) noexcept
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

    BL_NO_INLINE constexpr f128_s gamma_positive_low_range(const f128_s& x) noexcept
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

    BL_NO_INLINE constexpr f128_s lgamma_positive_recurrence(const f128_s& x) noexcept
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
            mul_inline(f128_s{ static_cast<double>(product_scale2) }, detail::_f128::ln2));
    }
    BL_NO_INLINE constexpr f128_s gamma_positive_recurrence(const f128_s& x) noexcept
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

    [[nodiscard]] BL_NO_INLINE constexpr f128_s atanh_small_series_constexpr(const f128_s& x)
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
    [[nodiscard]] BL_NO_INLINE inline f128_s atanh_small_series_runtime(const f128_s& x)
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
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s expm1(const f128_s& x)
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

        return detail::_f128::canonicalize_math_result(sum);
    }

    return detail::_f128::canonicalize_math_result(sub_inline(exp(x), f128_s{ 1.0 }));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log1p(const f128_s& x)
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
        return detail::_f128::canonicalize_math_result(detail::_f128::f128_log1p_series_reduced(x));

    const f128_s u = add_inline(f128_s{ 1.0 }, x);
    if (sub_inline(u, f128_s{ 1.0 }) == x)
        return detail::_f128::canonicalize_math_result(log(u));

    if (x > f128_s{ 0.0 } && x <= f128_s{ 1.0 })
    {
        const f128_s t = div_inline(x, add_inline(f128_s{ 1.0 }, sqrt(add_inline(f128_s{ 1.0 }, x))));
        return detail::_f128::canonicalize_math_result(mul_inline(detail::_f128::f128_log1p_series_reduced(t), f128_s{ 2.0 }));
    }

    if (x > f128_s{ 0.0 })
        return detail::_f128::canonicalize_math_result(log(u));

    const f128_s y = sub_inline(u, f128_s{ 1.0 });
    if (iszero(y))
        return x;

    return detail::_f128::canonicalize_math_result(mul_inline(log(u), div_inline(x, y)));
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s sinh(const f128_s& x)
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

        return detail::_f128::canonicalize_math_result(sum);
    }

    const f128_s ex = exp(ax);
    f128_s out = mul_inline(sub_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 });
    if (signbit(x))
        out = -out;
    return detail::_f128::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s cosh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = abs(x);
    const f128_s ex = exp(ax);
    return detail::_f128::canonicalize_math_result(mul_inline(add_inline(ex, div_inline(f128_s{ 1.0 }, ex)), f128_s{ 0.5 }));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tanh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = abs(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = expm1(add_inline(ax, ax));
    f128_s out = div_inline(em1, add_inline(em1, f128_s{ 2.0 }));
    if (signbit(x))
        out = -out;
    return detail::_f128::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s asinh(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = abs(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = add_inline(log(ax), detail::_f128::ln2);
    else
        out = log(add_inline(ax, sqrt(add_inline(mul_inline(ax, ax), f128_s{ 1.0 }))));

    if (signbit(x))
        out = -out;
    return detail::_f128::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s acosh(const f128_s& x)
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
        out = add_inline(log(x), detail::_f128::ln2);
    else
        out = log(add_inline(x, sqrt(mul_inline(sub_inline(x, f128_s{ 1.0 }), add_inline(x, f128_s{ 1.0 })))));

    return detail::_f128::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s atanh(const f128_s& x)
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
            return detail::_f128::canonicalize_math_result(detail::_f128::atanh_small_series_constexpr(x));

        return detail::_f128::canonicalize_math_result(detail::_f128::atanh_small_series_runtime(x));
    }

    const f128_s out = mul_inline(log(div_inline(add_inline(f128_s{ 1.0 }, x), sub_inline(f128_s{ 1.0 }, x))), f128_s{ 0.5 });
    return detail::_f128::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s cbrt(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y{};
    if (bl::use_constexpr_math())
    {
        y = exp(div_inline(log(ax), f128_s{ 3.0 }));
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
            y = detail::_f128::_ldexp(y, exp2 / 3);
    }

    y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });
    y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });

    if (bl::use_constexpr_math())
        y = div_inline(add_inline(add_inline(y, y), div_inline(ax, mul_inline(y, y))), f128_s{ 3.0 });

    if (neg)
        y = -y;

    return detail::_f128::canonicalize_math_result(y);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y)
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
        return detail::_f128::canonicalize_math_result(ax);

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
        return detail::_f128::canonicalize_math_result(ax);

    const f128_s r = div_inline(ay, ax);
    return detail::_f128::canonicalize_math_result(mul_inline(ax, sqrt(add_inline(f128_s{ 1.0 }, mul_inline(r, r)))));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s rint(const f128_s& x)
{
    return nearbyint(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f128_s& x)
{
    return detail::_f128::to_signed_integer_or_zero<long>(detail::_f128::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f128_s& x)
{
    return detail::_f128::to_signed_integer_or_zero<long long>(detail::_f128::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f128_s& x)
{
    return detail::_f128::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f128_s& x)
{
    return detail::_f128::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo)
{
    using namespace detail::_f128;

    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s n = detail::_f128::nearest_integer_ties_even(div_inline(x, y));
    f128_s r = sub_inline(x, mul_inline(n, y));

    if (quo)
    {
        const f128_s qbits = fmod(abs(n), f128_s{ 2147483648.0 });
        int bits = static_cast<int>(trunc(qbits).hi);
        if (signbit(n))
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return detail::_f128::canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    return detail::_f128::canonicalize_math_result(detail::_f128::add_inline(detail::_f128::mul_inline(x, y), z));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmin(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fmax(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fdim(const f128_s& x, const f128_s& y)
{
    return (x > y) ? detail::_f128::canonicalize_math_result(detail::_f128::sub_inline(x, y)) : f128_s{ 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s copysign(const f128_s& x, const f128_s& y)
{
    return signbit(x) == signbit(y) ? x : -x;
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept
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

    f128_s m = ldexp(x, -e);
    const f128_s am = abs(m);

    if (am < f128_s{ 0.5 })
    {
        m = detail::_f128::mul_inline(m, f128_s{ 2.0 });
        --e;
    }
    else if (am >= f128_s{ 1.0 })
    {
        m = detail::_f128::mul_inline(m, f128_s{ 0.5 });
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept
{
    const f128_s i = trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = detail::_f128::sub_inline(x, i);
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
}
[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept
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
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s logb(const f128_s& x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    return f128_s{ static_cast<double>(ilogb(x)), 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s scalbn(const f128_s& x, int e) noexcept
{
    return ldexp(x, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s scalbln(const f128_s& x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept
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

    return detail::_f128::renorm(
        from.hi,
        detail::_f128::nextafter_double_constexpr(from.lo, toward)
    );
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, long double to) noexcept
{
    return nextafter(from, f128_s{ static_cast<double>(to) });
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, const f128_s& to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s erfc(const f128_s& x);
[[nodiscard]] BL_NO_INLINE constexpr f128_s erf(const f128_s& x)
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

    f128_s out{ 0.0 };

    if (ax < f128_s{ 2.0 })
    {
        const f128_s xx = mul_inline(ax, ax);
        f128_s power = ax;
        f128_s sum = ax;

        for (int n = 1; n < 256; ++n)
        {
            power = mul_inline(power, div_inline(-xx, f128_s{ static_cast<double>(n) }));
            const f128_s term = div_inline(power, f128_s{ static_cast<double>(2 * n + 1) });
            sum = add_inline(sum, term);
            if (abs(term) < f128_s::eps())
                break;
        }

        out = mul_inline(mul_inline(f128_s{ 2.0 }, detail::_f128::inv_sqrtpi), sum);
    }
    else
    {
        out = sub_inline(f128_s{ 1.0 }, erfc(ax));
    }

    if (neg)
        out = -out;

    return detail::_f128::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s erfc(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return f128_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f128_s{ 2.0 } : f128_s{ 0.0 };

    if (signbit(x))
        return detail::_f128::canonicalize_math_result(add_inline(f128_s{ 1.0 }, erf(-x)));

    // use the existing high-quality erf series throughout the region where it is stable
    if (x < f128_s{ 2.0 })
        return detail::_f128::canonicalize_math_result(sub_inline(f128_s{ 1.0 }, erf(x)));

    if (x > f128_s{ 27.0 })
        return f128_s{ 0.0 };

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

    const f128_s out = mul_inline(mul_inline(mul_inline(exp(-z), x), detail::_f128::inv_sqrtpi), h);
    return detail::_f128::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s lgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return detail::_f128::canonicalize_math_result(detail::_f128::lgamma_positive_recurrence(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = sin(mul_inline(detail::_f128::pi, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        sub_inline(
            sub_inline(log(detail::_f128::pi), log(abs(sinpix))),
            detail::_f128::lgamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x)));

    return detail::_f128::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tgamma(const f128_s& x)
{
    using namespace detail::_f128;

    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return detail::_f128::canonicalize_math_result(detail::_f128::gamma_positive_recurrence(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = sin(mul_inline(detail::_f128::pi, x));
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = div_inline(detail::_f128::pi, mul_inline(sinpix, detail::_f128::gamma_positive_recurrence(sub_inline(f128_s{ 1.0 }, x))));
    return detail::_f128::canonicalize_math_result(out);
}

}

#endif
