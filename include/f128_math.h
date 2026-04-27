#ifndef F128_MATH_INCLUDED
#define F128_MATH_INCLUDED

#include "f128.h"
#include "fltx_common_exact.h"

namespace bl {

/// ------------------ math ------------------

namespace _f128_detail
{
    BL_FORCE_INLINE constexpr bool f128_try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = trunc(x);
        if (xi != x)
            return false;

        if (_f128_detail::absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = xi - to_f128(hi_part);
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }
    BL_FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        if (exp == 0)
            return f128_s{ 1.0 };

        const bool invert = exp < 0;
        uint64_t n = invert ? _f128_detail::magnitude_u64(exp) : static_cast<uint64_t>(exp);
        f128_s result{ 1.0 };

        while (n != 0)
        {
            if ((n & 1u) != 0)
                result *= base;

            n >>= 1;
            if (n != 0)
                base *= base;
        }

        return invert ? (f128_s{ 1.0 } / result) : result;
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
        const std::uint64_t mantissa = fltx::common::exact_decimal::decompose_double_mantissa(value, exponent, neg);
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
        if (!(ay > 0.0) || !_f128_detail::isfinite(ay))
            return false;

        const f128_s mod{ ay, 0.0 };

        if (ax.lo == 0.0)
        {
            out = f128_s{ _f128_detail::fmod_constexpr(ax.hi, ay), 0.0 };
            return true;
        }

        const double rh = (ax.hi < ay) ? ax.hi : _f128_detail::fmod_constexpr(ax.hi, ay);
        const double rl = (_f128_detail::absd(ax.lo) < ay) ? ax.lo : _f128_detail::fmod_constexpr(ax.lo, ay);

        f128_s r = f128_s{ rh, 0.0 } + f128_s{ rl, 0.0 };

        if (r < 0.0)
            r += mod;
        if (r >= mod)
            r -= mod;

        if (r < 0.0)
            r += mod;
        if (r >= mod)
            r -= mod;

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
        const double hi = c0 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = _f128_detail::renorm(hi, lo);
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
            return f128_s{ _f128_detail::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return out;
    }
}

[[nodiscard]] inline BL_NO_INLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y)
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
    if (y.lo == 0.0 && _f128_detail::fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ _f128_detail::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return _f128_detail::fmod_exact_fixed_limb(x, y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s round(const f128_s& a)
{
    f128_s t = floor(a + f128_s{ 0.5 });
    if ((t - a) == f128_s{ 0.5 } && fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t -= f128_s{ 1.0 };
    return t;
}
[[nodiscard]] inline BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec)
{
    if (prec <= 0) return v;

    static constexpr f128_s INV10_DD{
        0.1000000000000000055511151231257827021181583404541015625,  // hi (double rounded)
       -0.0000000000000000055511151231257827021181583404541015625   // lo = 0.1 - hi
    };

    // Sign
    const bool neg = v < 0.0;
    if (neg) v = -v;

    // Split
    f128_s ip = floor(v);
    f128_s frac = v - ip;

    // Extract digits with one look-ahead
    std::string dig; dig.reserve((size_t)prec);
    f128_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;
        int di = (int)floor(w).hi;
        if (di < 0) di = 0; else if (di > 9) di = 9;
        dig.push_back(char('0' + di));
        w = w - f128_s{ (double)di };
    }

    // Look-ahead digit
    f128_s la = w * 10.0;
    int next = (int)floor(la).hi;
    if (next < 0) next = 0; else if (next > 9) next = 9;
    f128_s rem = la - f128_s{ (double)next };

    // ties-to-even on last printed digit
    const int last = dig.empty() ? 0 : (dig.back() - '0');
    const bool round_up =
        (next > 5) ||
        (next == 5 && (rem.hi > 0.0 || rem.lo > 0.0 || (last & 1)));

    if (round_up) {
        // propagate carry over fractional digits; if overflow, bump integer part
        int i = prec - 1;
        for (; i >= 0; --i) {
            if (dig[(size_t)i] == '9') dig[(size_t)i] = '0';
            else { ++dig[(size_t)i]; break; }
        }
        if (i < 0) ip = ip + 1.0;
    }

    // Rebuild fractional value backward
    f128_s frac_val{ 0.0, 0.0 };
    for (int i = prec - 1; i >= 0; --i) {
        frac_val = frac_val + f128_s{ (double)(dig[(size_t)i] - '0') };
        frac_val = frac_val * INV10_DD;
    }

    f128_s out = ip + frac_val;
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
    const f128_s half = ay * f128_s{ 0.5 };

    if (ar > half)
    {
        r += signbit(r) ? ay : -ay;
    }
    else if (ar == half)
    {
        const f128_s q = trunc(x / y);
        const f128_s q_mod2 = abs(fmod(q, f128_s{ 2.0 }));
        if (q_mod2 != f128_s{ 0.0 })
            r += signbit(r) ? ay : -ay;
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return _f128_detail::canonicalize_math_result(r);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    double y0;
    if (bl::is_constant_evaluated()) {
        y0 = _f128_detail::sqrt_seed_constexpr(a.hi);
    }
    else {
        y0 = std::sqrt(a.hi);
    }
    f128_s y{ y0 };

    // Newton refinements
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    return _f128_detail::canonicalize_math_result(y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s nearbyint(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f128_s t = floor(a);
    f128_s frac = a - t;

    if (frac < f128_s{ 0.5 })
        return t;

    if (frac > f128_s{ 0.5 })
    {
        t += f128_s{ 1.0 };
        if (iszero(t))
            return f128_s{ _f128_detail::signbit_constexpr(a.hi) ? -0.0 : 0.0 };
        return t;
    }

    if (fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t += f128_s{ 1.0 };

    if (iszero(t))
        return f128_s{ _f128_detail::signbit_constexpr(a.hi) ? -0.0 : 0.0 };

    return t;
}

/// ------------------ transcendentals ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f128_s a)
{
    const double hi = a.hi;
    if (hi <= 0.0)
        return bl::fltx::common::fp::log_constexpr(static_cast<double>(a));

    return bl::fltx::common::fp::log_constexpr(hi) + bl::fltx::common::fp::log1p_constexpr(a.lo / hi);
}

namespace _f128_const
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
}
namespace _f128_detail
{
    inline constexpr double pi_4_hi = _f128_const::pi_4.hi;
    inline constexpr double pi_2_hi_d = 0x1.921fb54442d18p+0;
    inline constexpr double pi_2_mid_d = 0x1.1a62633145c07p-54;
    inline constexpr double pi_2_lo_d = -0x1.f1976b7ed8fbcp-110;

    using fltx::common::fp::signbit_constexpr;
    using fltx::common::fp::fabs_constexpr;
    using fltx::common::fp::floor_constexpr;
    using fltx::common::fp::ceil_constexpr;
    using fltx::common::fp::double_integer_is_odd;
    using fltx::common::fp::fmod_constexpr;
    using fltx::common::fp::sqrt_seed_constexpr;
    using fltx::common::fp::nearbyint_ties_even;

    BL_FORCE_INLINE constexpr f128_s f128_log1p_series_reduced(const f128_s& x)
    {
        const f128_s z = x / (f128_s{ 2.0 } + x);
        const f128_s z2 = z * z;

        f128_s term = z;
        f128_s sum = z;

        for (int k = 3; k <= 81; k += 2)
        {
            term *= z2;
            const f128_s add = term / f128_s{ static_cast<double>(k) };
            sum += add;

            const f128_s asum = abs(sum);
            const f128_s scale = (asum > f128_s{ 1.0 }) ? asum : f128_s{ 1.0 };
            if (abs(add) <= f128_s::eps() * scale)
                break;
        }

        return sum + sum;
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
            p = p * y + lgamma1p_coeff[i];

        return y * (-_f128_const::egamma + y * p);
    }

    BL_NO_INLINE constexpr f128_s lgamma1p5_series(const f128_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p5_coeff) / sizeof(lgamma1p5_coeff[0]));

        f128_s p = lgamma1p5_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = p * y + lgamma1p5_coeff[i];

        const f128_s constant = _f128_const::half_log_two_pi - f128_s{ 1.5 } * _f128_const::ln2;
        const f128_s linear = f128_s{ 2.0 } - _f128_const::egamma - f128_s{ 2.0 } * _f128_const::ln2;
        return constant + y * (linear + y * p);
    }

    BL_FORCE_INLINE constexpr bool try_lgamma_near_one_or_two(const f128_s& x, f128_s& out) noexcept
    {
        const f128_s y1 = x - f128_s{ 1.0 };
        if (abs(y1) <= f128_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f128_s y15 = x - f128_s{ 1.5 };
        if (abs(y15) <= f128_s{ 0.25 })
        {
            out = lgamma1p5_series(y15);
            return true;
        }

        const f128_s y2 = x - f128_s{ 2.0 };
        if (abs(y2) <= f128_s{ 0.25 })
        {
            out = f128_log1p_series_reduced(y2) + lgamma1p_series(y2);
            return true;
        }

        return false;
    }

    BL_FORCE_INLINE constexpr f128_s f128_exp_kernel_ln2_half(const f128_s& r)
    {
        f128_s p = f128_s{ 8.89679139245057408e-22 };
        p *= r + f128_s{ 1.95729410633912626e-20 };
        p *= r + f128_s{ 4.11031762331216484e-19 };
        p *= r + f128_s{ 8.22063524662432950e-18 };
        p *= r + f128_s{ 1.56192069685862253e-16 };
        p *= r + f128_s{ 2.81145725434552060e-15 };
        p *= r + f128_s{ 4.77947733238738525e-14 };
        p *= r + f128_s{ 7.64716373181981641e-13 };
        p *= r + f128_s{ 1.14707455977297245e-11 };
        p *= r + f128_s{ 1.60590438368216133e-10 };
        p *= r + f128_s{ 2.08767569878681002e-09 };
        p *= r + f128_s{ 2.50521083854417202e-08 };
        p *= r + f128_s{ 2.75573192239858883e-07 };
        p *= r + f128_s{ 2.75573192239858925e-06 };
        p *= r + f128_s{ 2.48015873015873016e-05 };
        p *= r + f128_s{ 1.98412698412698413e-04 };
        p *= r + f128_s{ 1.38888888888888894e-03 };
        p *= r + f128_s{ 8.33333333333333322e-03 };
        p *= r + f128_s{ 4.16666666666666644e-02 };
        p *= r + f128_s{ 1.66666666666666657e-01 };
        p *= r + f128_s{ 5.00000000000000000e-01 };
        p *= r + f128_s{ 1.0 };
        return (p * r) + f128_s{ 1.0 };
    }
    BL_FORCE_INLINE constexpr f128_s f128_expm1_tiny(const f128_s& r)
    {
        f128_s p =    f128_s{1.0} / f128_s{6227020800.0};
        p = p * r + f128_s{1.0} / f128_s{479001600.0};
        p = p * r + f128_s{1.0} / f128_s{39916800.0};
        p = p * r + f128_s{1.0} / f128_s{3628800.0};
        p = p * r + f128_s{1.0} / f128_s{362880.0};
        p = p * r + f128_s{1.0} / f128_s{40320.0};
        p = p * r + f128_s{1.0} / f128_s{5040.0};
        p = p * r + f128_s{1.0} / f128_s{720.0};
        p = p * r + f128_s{1.0} / f128_s{120.0};
        p = p * r + f128_s{1.0} / f128_s{24.0};
        p = p * r + f128_s{1.0} / f128_s{6.0};
        p = p * r + f128_s{0.5};
        return r + (r * r) * p;
    }

    BL_FORCE_INLINE constexpr bool f128_remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
	{
	    const double ax = _f128_detail::fabs_constexpr(x.hi);
	    if (!_f128_detail::isfinite(ax))
	        return false;

	    if (ax > 7.0e15)
	        return false;

	    const f128_s t = x * _f128_const::invpi2;

	    double qd = _f128_detail::nearbyint_ties_even(t.hi);
	    if (!_f128_detail::isfinite(qd) ||
	        qd < static_cast<double>(std::numeric_limits<long long>::min()) ||
	        qd > static_cast<double>(std::numeric_limits<long long>::max()))
	    {
	        return false;
	    }

	    const f128_s delta = t - f128_s{ qd };
	    if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
	        qd += 1.0;
	    else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
	        qd -= 1.0;

	    if (qd < static_cast<double>(std::numeric_limits<long long>::min()) ||
	        qd > static_cast<double>(std::numeric_limits<long long>::max()))
	    {
	        return false;
	    }

	    const f128_s pi_2_hi{ _f128_detail::pi_2_hi_d };
	    const f128_s pi_2_mid{ _f128_detail::pi_2_mid_d };
	    const f128_s pi_2_lo{ _f128_detail::pi_2_lo_d };
	    const f128_s pi_4{ _f128_detail::pi_4_hi };

	    long long n = static_cast<long long>(qd);
	    const f128_s q{ static_cast<double>(n) };

	    f128_s r = x;
	    r -= q * pi_2_hi;
	    r -= q * pi_2_mid;
	    r -= q * pi_2_lo;

	    if (r > pi_4)
	    {
	        ++n;
	        r -= pi_2_hi;
	        r -= pi_2_mid;
	        r -= pi_2_lo;
	    }
	    else if (r < -pi_4)
	    {
	        --n;
	        r += pi_2_hi;
	        r += pi_2_mid;
	        r += pi_2_lo;
	    }

	    n_out = n;
	    r_out = r;
	    return true;
	}
    BL_FORCE_INLINE constexpr f128_s f128_sin_kernel_pi4(const f128_s& x)
    {
        const f128_s t = x * x;

        f128_s ps = f128_s{  1.13099628864477159e-31,  1.04980154129595057e-47 };
        ps = ps * t + f128_s{ -9.18368986379554615e-29, -1.43031503967873224e-45 };
        ps = ps * t + f128_s{  6.44695028438447391e-26, -1.93304042337034642e-42 };
        ps = ps * t + f128_s{ -3.86817017063068404e-23,  8.84317765548234382e-40 };
        ps = ps * t + f128_s{  1.95729410633912625e-20, -1.36435038300879076e-36 };
        ps = ps * t + f128_s{ -8.22063524662432972e-18, -2.21418941196042654e-34 };
        ps = ps * t + f128_s{  2.81145725434552060e-15,  1.65088427308614330e-31 };
        ps = ps * t + f128_s{ -7.64716373181981648e-13, -7.03872877733452971e-30 };
        ps = ps * t + f128_s{  1.60590438368216146e-10,  1.25852945887520981e-26 };
        ps = ps * t + f128_s{ -2.50521083854417188e-08,  1.44881407093591197e-24 };
        ps = ps * t + f128_s{  2.75573192239858907e-06, -1.85839327404647208e-22 };
        ps = ps * t + f128_s{ -1.98412698412698413e-04, -1.72095582934207053e-22 };
        ps = ps * t + f128_s{  8.33333333333333322e-03,  1.15648231731787140e-19 };
        ps = ps * t + f128_s{ -1.66666666666666657e-01, -9.25185853854297066e-18 };
        return x + x * t * ps;
    }
    BL_FORCE_INLINE constexpr f128_s f128_cos_kernel_pi4(const f128_s& x)
    {
        const f128_s t = x * x;

        f128_s pc = f128_s{  3.27988923706983791e-30,  1.51175427440298786e-46 };
        pc = pc * t + f128_s{ -2.47959626322479746e-27,  1.29537309647652292e-43 };
        pc = pc * t + f128_s{  1.61173757109611835e-24, -3.68465735645097656e-41 };
        pc = pc * t + f128_s{ -8.89679139245057329e-22,  7.91140261487237594e-38 };
        pc = pc * t + f128_s{  4.11031762331216486e-19,  1.44129733786595266e-36 };
        pc = pc * t + f128_s{ -1.56192069685862265e-16, -1.19106796602737541e-32 };
        pc = pc * t + f128_s{  4.77947733238738530e-14,  4.39920548583408094e-31 };
        pc = pc * t + f128_s{ -1.14707455977297247e-11, -2.06555127528307454e-28 };
        pc = pc * t + f128_s{  2.08767569878680990e-09, -1.20734505911325997e-25 };
        pc = pc * t + f128_s{ -2.75573192239858907e-07, -2.37677146222502973e-23 };
        pc = pc * t + f128_s{  2.48015873015873016e-05,  2.15119478667758816e-23 };
        pc = pc * t + f128_s{ -1.38888888888888894e-03,  5.30054395437357706e-20 };
        pc = pc * t + f128_s{  4.16666666666666644e-02,  2.31296463463574269e-18 };
        pc = pc * t + f128_s{ -5.00000000000000000e-01,  0.0 };
        return f128_s{ 1.0 } + t * pc;
    }
    BL_FORCE_INLINE constexpr void f128_sincos_kernel_pi4(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        s_out = f128_sin_kernel_pi4(x);
        c_out = f128_cos_kernel_pi4(x);
    }

    BL_FORCE_INLINE constexpr f128_s canonicalize_exp_result(f128_s value) noexcept
    {
        value.lo = fltx::common::fp::zero_low_fraction_bits_finite<6>(value.lo);
        return value;
    }

    BL_FORCE_INLINE constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        if (bl::is_constant_evaluated())
        {
            return canonicalize_exp_result(_f128_detail::renorm(
                fltx::common::fp::ldexp_constexpr2(x.hi, e),
                fltx::common::fp::ldexp_constexpr2(x.lo, e)
            ));
        }
        else
        {
            return canonicalize_exp_result(_f128_detail::renorm(
                std::ldexp(x.hi, e),
                std::ldexp(x.lo, e)
            ));
        }
    }
    BL_FORCE_INLINE constexpr f128_s _exp(const f128_s& x)
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

        const f128_s t = x * _f128_const::inv_ln2;

        double kd = _f128_detail::nearbyint_ties_even(t.hi);
        const f128_s delta = t - f128_s{ kd };
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s r = (x - f128_s{ kd } * _f128_const::ln2) * f128_s{ 0.0009765625 };

        f128_s e = _f128_detail::f128_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = e * (e + 2.0);

        return _ldexp(e + 1.0, k);
    }
    BL_FORCE_INLINE constexpr f128_s _log(const f128_s& a)
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
        if (bl::is_constant_evaluated()) {
            exp2 = fltx::common::fp::frexp_exponent_constexpr(a.hi);
        }
        else {
            (void)std::frexp(a.hi, &exp2);
        }

        f128_s m = _ldexp(a, -exp2);
        if (m < _f128_const::sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }

        const f128_s exp2_ln2 = f128_s{ static_cast<double>(exp2) } * _f128_const::ln2;
        f128_s y = exp2_ln2 + f128_s{ log_as_double(m) };
        y += m * _exp(exp2_ln2 - y) - 1.0;
        y += m * _exp(exp2_ln2 - y) - 1.0;
        y += m * _exp(exp2_ln2 - y) - 1.0;
        return y;
    }
}

BL_NO_INLINE    constexpr f128_s pow10_128(int k);

// exp
[[nodiscard]] BL_NO_INLINE constexpr f128_s ldexp(const f128_s& x, int e)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_ldexp(x, e));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s exp(const f128_s& x)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_exp(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s exp2(const f128_s& x)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_exp(x * _f128_const::ln2));
}

// log
[[nodiscard]] BL_NO_INLINE constexpr f128_s log(const f128_s& a)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_log(a));
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log2(const f128_s& a)
{
    int exact_exp2{};
    if (_f128_detail::f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return _f128_detail::canonicalize_math_result(_f128_detail::_log(a) * _f128_const::inv_ln2);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log10(const f128_s& x)
{
    if (x.hi > 0.0)
    {
        const int exp2 =
            fltx::common::fp::frexp_exponent_constexpr(x.hi);
        const int k0 =
            static_cast<int>(fltx::common::fp::floor_constexpr((exp2 - 1) * 0.30102999566398114));

        for (int k = k0 - 2; k <= k0 + 2; ++k)
        {
            if (x == pow10_128(k))
                return f128_s{ static_cast<double>(k), 0.0 };
        }
    }

    return _f128_detail::canonicalize_math_result(_f128_detail::_log(x) * _f128_const::inv_ln10);
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
    if (y_is_int && _f128_detail::f128_try_get_int64(yi, yi64))
        return _f128_detail::powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && _f128_detail::signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = _f128_detail::_exp(y * _f128_detail::_log(-x));
        const f128_s parity = fmod(abs(yi), f128_s{ 2.0 });
        return (parity == f128_s{ 1.0 }) ? -magnitude : magnitude;
    }

    return _f128_detail::_exp(y * _f128_detail::_log(x));
}


// trig
[[nodiscard]] BL_NO_INLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
    {
        s_out = f128_s{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= _f128_detail::pi_4_hi)
    {
        _f128_detail::f128_sincos_kernel_pi4(x, s_out, c_out);
        s_out = _f128_detail::canonicalize_math_result(s_out);
        c_out = _f128_detail::canonicalize_math_result(c_out);
        return true;
    }

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
        return false;

    f128_s sr{}, cr{};
    _f128_detail::f128_sincos_kernel_pi4(r, sr, cr);

    switch ((int)(n & 3))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    s_out = _f128_detail::canonicalize_math_result(s_out);
    c_out = _f128_detail::canonicalize_math_result(c_out);
    return true;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s sin(const f128_s& x)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= _f128_detail::pi_4_hi)
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
    {
        if (bl::is_constant_evaluated()) 
        {
            return _f128_detail::canonicalize_math_result(f128_s{ fltx::common::fp::sin_constexpr(static_cast<double>(x)) });
        }
        else 
        {
            return _f128_detail::canonicalize_math_result(f128_s{ std::sin((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(r));
    case 1: return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(r));
    case 2: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_sin_kernel_pi4(r));
    default: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_cos_kernel_pi4(r));
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s cos(const f128_s& x)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= _f128_detail::pi_4_hi)
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
    {
        if (bl::is_constant_evaluated())
        {
            return _f128_detail::canonicalize_math_result(f128_s{ fltx::common::fp::cos_constexpr(static_cast<double>(x)) });
        }
        else 
        {
            return _f128_detail::canonicalize_math_result(f128_s{ std::cos((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(r));
    case 1: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_sin_kernel_pi4(r));
    case 2: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_cos_kernel_pi4(r));
    default: return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(r));
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tan(const f128_s& x)
{
    f128_s s{}, c{};
    if (sincos(x, s, c))
        return s / c;
    const double xd = (double)x;
    if (bl::is_constant_evaluated()) {
        return f128_s{ fltx::common::fp::tan_constexpr(xd) };
    } else {
        return f128_s{ std::tan(xd) };
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x)
{
    if (iszero(x))
    {
        if (iszero(y))
            return f128_s{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? _f128_const::pi_2 : -_f128_const::pi_2;
    }

    const f128_s scale = std::max(abs(x), abs(y));
    const f128_s xs = x / scale;
    const f128_s ys = y / scale;

    f128_s v{ fltx::common::fp::atan2_constexpr(y.hi, x.hi) };

    for (int i = 0; i < 2; ++i)
    {
        f128_s sv{}, cv{};
        if (!sincos(v, sv, cv))
        {
            const double vd = (double)v;
            if (bl::is_constant_evaluated()) {
                double sd{}, cd{};
                fltx::common::fp::sincos_constexpr(vd, sd, cd);
                sv = f128_s{ sd };
                cv = f128_s{ cd };
            } else {
                sv = f128_s{ std::sin(vd) };
                cv = f128_s{ std::cos(vd) };
            }
        }

        const f128_s f = xs * sv - ys * cv;
        const f128_s fp = xs * cv + ys * sv;

        v = v - f / fp;
    }

    return _f128_detail::canonicalize_math_result(v);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s atan(const f128_s& x)
{
    return atan2(x, f128_s{ 1.0 });
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s asin(const f128_s& x)
{
    return atan2(x, sqrt(f128_s{ 1.0 } - x * x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s acos(const f128_s& x)
{
    return atan2(sqrt(f128_s{ 1.0 } - x * x), x);
}


[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fabs(const f128_s& a) noexcept
{
    return abs(a);
}


namespace _f128_detail
{
    BL_FORCE_INLINE constexpr f128_s round_half_away_zero(const f128_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f128_s y = -floor((-x) + f128_s{ 0.5 });
            if (iszero(y))
                return f128_s{ -0.0, 0.0 };
            return y;
        }

        return floor(x + f128_s{ 0.5 });
    }
    BL_FORCE_INLINE constexpr double nextafter_double_constexpr(double from, double to) noexcept
    {
        if (fltx::common::fp::isnan(from) || fltx::common::fp::isnan(to))
            return std::numeric_limits<double>::quiet_NaN();

        if (from == to)
            return to;

        if (from == 0.0)
            return fltx::common::fp::signbit_constexpr(to)
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
        if (!_f128_detail::f128_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }
    BL_FORCE_INLINE constexpr f128_s nearest_integer_ties_even(const f128_s& q) noexcept
    {
        f128_s n = trunc(q);
        const f128_s frac = q - n;
        const f128_s half{ 0.5 };
        const f128_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n += signbit(frac) ? -one : one;
        }
        else if (abs(frac) == half)
        {
            if (fmod(n, f128_s{ 2.0 }) != f128_s{ 0.0 })
                n += signbit(frac) ? -one : one;
        }

        return n;
    }
    
    BL_NO_INLINE constexpr f128_s lgamma_stirling_asymptotic(const f128_s& z) noexcept
    {
        const f128_s inv = f128_s{ 1.0 } / z;
        const f128_s inv2 = inv * inv;

        f128_s series = inv / f128_s{ 12.0 };
        f128_s invpow = inv * inv2;

        series -= invpow / f128_s{ 360.0 };
        invpow *= inv2;
        series += invpow / f128_s{ 1260.0 };
        invpow *= inv2;
        series -= invpow / f128_s{ 1680.0 };
        invpow *= inv2;
        series += invpow / f128_s{ 1188.0 };
        invpow *= inv2;
        series -= invpow * (f128_s{ 691.0 } / f128_s{ 360360.0 });
        invpow *= inv2;
        series += invpow / f128_s{ 156.0 };
        invpow *= inv2;
        series -= invpow * (f128_s{ 3617.0 } / f128_s{ 122400.0 });
        invpow *= inv2;
        series += invpow * (f128_s{ 43867.0 } / f128_s{ 244188.0 });
        invpow *= inv2;
        series -= invpow * (f128_s{ 174611.0 } / f128_s{ 125400.0 });
        invpow *= inv2;
        series += invpow * (f128_s{ 77683.0 } / f128_s{ 5796.0 });
        invpow *= inv2;
        series -= invpow * (f128_s{ 236364091.0 } / f128_s{ 1506960.0 });

        return (z - f128_s{ 0.5 }) * log(z) - z + _f128_const::half_log_two_pi + series;
    }
    BL_NO_INLINE constexpr f128_s lgamma_positive_low_range(const f128_s& x) noexcept
    {
        f128_s y = x;
        f128_s correction{ 0.0 };

        if (y < f128_s{ 0.75 })
        {
            do
            {
                correction -= log(y);
                y += f128_s{ 1.0 };
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y -= f128_s{ 1.0 };
                correction += log(y);
            }
        }

        f128_s local{};
        try_lgamma_near_one_or_two(y, local);
        return local + correction;
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
                product *= y;
                y += f128_s{ 1.0 };
            }
            while (y < f128_s{ 0.75 });
        }
        else
        {
            while (y > f128_s{ 2.25 })
            {
                y -= f128_s{ 1.0 };
                product *= y;
            }
        }

        f128_s local_lgamma{};
        try_lgamma_near_one_or_two(y, local_lgamma);
        const f128_s local_gamma = exp(local_lgamma);
        return shifted_up ? (local_gamma / product) : (local_gamma * product);
    }

    BL_NO_INLINE constexpr f128_s lgamma_positive_recurrence(const f128_s& x) noexcept
    {
        if (x <= f128_s{ 16.0 })
            return lgamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z = x;
        f128_s product{ 1.0 };
        int product_scale2 = 0;

        while (z < asymptotic_min)
        {
            product *= z;

            const double hi = product.hi;
            if (hi != 0.0)
            {
                const int exponent = fltx::common::fp::frexp_exponent_constexpr(hi);
                if (exponent > 512 || exponent < -512)
                {
                    product = ldexp(product, -exponent);
                    product_scale2 += exponent;
                }
            }

            z += f128_s{ 1.0 };
        }

        return lgamma_stirling_asymptotic(z)
            - log(product)
            - f128_s{ static_cast<double>(product_scale2) } * _f128_const::ln2;
    }
    BL_NO_INLINE constexpr f128_s gamma_positive_recurrence(const f128_s& x) noexcept
    {
        if (x <= f128_s{ 16.0 })
            return gamma_positive_low_range(x);

        constexpr f128_s asymptotic_min = f128_s{ 40.0 };

        f128_s z = x;
        f128_s product{ 1.0 };
        int product_scale2 = 0;

        while (z < asymptotic_min)
        {
            product *= z;

            const double hi = product.hi;
            if (hi != 0.0)
            {
                const int exponent = fltx::common::fp::frexp_exponent_constexpr(hi);
                if (exponent > 512 || exponent < -512)
                {
                    product = ldexp(product, -exponent);
                    product_scale2 += exponent;
                }
            }

            z += f128_s{ 1.0 };
        }

        f128_s out = exp(lgamma_stirling_asymptotic(z)) / product;
        if (product_scale2 != 0)
            out = ldexp(out, -product_scale2);

        return out;
    }
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s expm1(const f128_s& x)
{
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
            term = (term * x) / f128_s{ static_cast<double>(n) };
            sum += term;

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= f128_s::eps() * scale)
                break;
        }

        return _f128_detail::canonicalize_math_result(sum);
    }

    return _f128_detail::canonicalize_math_result(exp(x) - f128_s{ 1.0 });
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s log1p(const f128_s& x)
{
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
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_log1p_series_reduced(x));

    const f128_s u = f128_s{ 1.0 } + x;
    if ((u - f128_s{ 1.0 }) == x)
        return _f128_detail::canonicalize_math_result(log(u));

    if (x > f128_s{ 0.0 } && x <= f128_s{ 1.0 })
    {
        const f128_s t = x / (f128_s{ 1.0 } + sqrt(f128_s{ 1.0 } + x));
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_log1p_series_reduced(t) * f128_s{ 2.0 });
    }

    if (x > f128_s{ 0.0 })
        return _f128_detail::canonicalize_math_result(log(u));

    const f128_s y = u - f128_s{ 1.0 };
    if (iszero(y))
        return x;

    return _f128_detail::canonicalize_math_result(log(u) * (x / y));
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s sinh(const f128_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax <= f128_s{ 0.5 })
    {
        const f128_s x2 = x * x;
        f128_s term = x;
        f128_s sum = x;

        for (int n = 1; n <= 40; ++n)
        {
            const double denom = static_cast<double>((n * 2) * (n * 2 + 1));
            term = (term * x2) / f128_s{ denom };
            sum += term;

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= f128_s::eps() * scale)
                break;
        }

        return _f128_detail::canonicalize_math_result(sum);
    }

    const f128_s ex = exp(ax);
    f128_s out = (ex - f128_s{ 1.0 } / ex) * f128_s{ 0.5 };
    if (signbit(x))
        out = -out;
    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s cosh(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = abs(x);
    const f128_s ex = exp(ax);
    return _f128_detail::canonicalize_math_result((ex + f128_s{ 1.0 } / ex) * f128_s{ 0.5 });
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tanh(const f128_s& x)
{
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = abs(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = expm1(ax + ax);
    f128_s out = em1 / (em1 + f128_s{ 2.0 });
    if (signbit(x))
        out = -out;
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s asinh(const f128_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = abs(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = log(ax) + _f128_const::ln2;
    else
        out = log(ax + sqrt(ax * ax + f128_s{ 1.0 }));

    if (signbit(x))
        out = -out;
    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s acosh(const f128_s& x)
{
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
        out = log(x) + _f128_const::ln2;
    else
        out = log(x + sqrt((x - f128_s{ 1.0 }) * (x + f128_s{ 1.0 })));

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s atanh(const f128_s& x)
{
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
        const f128_s x2 = x * x;
        f128_s sum = x;
        f128_s power = x;
        for (int k = 1; k <= 80; ++k)
        {
            power *= x2;
            const f128_s term = power / f128_s{ static_cast<double>(2 * k + 1) };
            sum += term;

            const f128_s scale = std::max(abs(sum), f128_s{ 1.0 });
            if (abs(term) <= f128_s::eps() * scale)
                break;
        }
        return _f128_detail::canonicalize_math_result(sum);
    }

    const f128_s out = log1p((x + x) / (f128_s{ 1.0 } - x)) * f128_s{ 0.5 };
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s cbrt(const f128_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y{};
    if (bl::is_constant_evaluated())
    {
        y = exp(log(ax) / f128_s{ 3.0 });
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
            y = _f128_detail::_ldexp(y, exp2 / 3);
    }

    y = (y + y + ax / (y * y)) / f128_s{ 3.0 };
    y = (y + y + ax / (y * y)) / f128_s{ 3.0 };

    if (bl::is_constant_evaluated())
        y = (y + y + ax / (y * y)) / f128_s{ 3.0 };

    if (neg)
        y = -y;

    return _f128_detail::canonicalize_math_result(y);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y)
{
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
        return _f128_detail::canonicalize_math_result(ax);

    int ex = 0;
    int ey = 0;
    if (bl::is_constant_evaluated())
    {
        ex = fltx::common::fp::frexp_exponent_constexpr(ax.hi);
        ey = fltx::common::fp::frexp_exponent_constexpr(ay.hi);
    }
    else
    {
        (void)std::frexp(ax.hi, &ex);
        (void)std::frexp(ay.hi, &ey);
    }

    if ((ex - ey) > 55)
        return _f128_detail::canonicalize_math_result(ax);

    const f128_s r = ay / ax;
    return _f128_detail::canonicalize_math_result(ax * sqrt(f128_s{ 1.0 } + r * r));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s rint(const f128_s& x)
{
    return nearbyint(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long>(_f128_detail::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long long>(_f128_detail::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s n = _f128_detail::nearest_integer_ties_even(x / y);
    f128_s r = x - n * y;

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

    return _f128_detail::canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    return _f128_detail::canonicalize_math_result(x * y + z);
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
    return (x > y) ? _f128_detail::canonicalize_math_result(x - y) : f128_s{ 0.0 };
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

    if (bl::is_constant_evaluated())
        e = fltx::common::fp::frexp_exponent_constexpr(lead);
    else
        (void)std::frexp(lead, &e);

    f128_s m = ldexp(x, -e);
    const f128_s am = abs(m);

    if (am < f128_s{ 0.5 })
    {
        m *= f128_s{ 2.0 };
        --e;
    }
    else if (am >= f128_s{ 1.0 })
    {
        m *= f128_s{ 0.5 };
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

    f128_s frac = x - i;
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

    return _f128_detail::renorm(
        from.hi,
        _f128_detail::nextafter_double_constexpr(from.lo, toward)
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
        const f128_s xx = ax * ax;
        f128_s power = ax;
        f128_s sum = ax;

        for (int n = 1; n < 256; ++n)
        {
            power *= -xx / f128_s{ static_cast<double>(n) };
            const f128_s term = power / f128_s{ static_cast<double>(2 * n + 1) };
            sum += term;
            if (abs(term) < f128_s::eps())
                break;
        }

        out = f128_s{ 2.0 } * _f128_const::inv_sqrtpi * sum;
    }
    else
    {
        out = f128_s{ 1.0 } - erfc(ax);
    }

    if (neg)
        out = -out;

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s erfc(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return f128_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f128_s{ 2.0 } : f128_s{ 0.0 };

    if (signbit(x))
        return _f128_detail::canonicalize_math_result(f128_s{ 1.0 } + erf(-x));

    // use the existing high-quality erf series throughout the region where it is stable
    if (x < f128_s{ 2.0 })
        return _f128_detail::canonicalize_math_result(f128_s{ 1.0 } - erf(x));

    if (x > f128_s{ 27.0 })
        return f128_s{ 0.0 };

    const f128_s z = x * x;
    constexpr f128_s a = f128_s{ 0.5 };
    constexpr f128_s tiny = f128_s{ 1.0e-300 };

    f128_s b = z + f128_s{ 1.0 } - a;
    f128_s c = f128_s{ 1.0 } / tiny;
    f128_s d = f128_s{ 1.0 } / b;
    f128_s h = d;

    for (int i = 1; i <= 96; ++i)
    {
        const f128_s ii = f128_s{ static_cast<double>(i) };
        const f128_s an = -(ii * (ii - a));

        b += f128_s{ 2.0 };

        d = an * d + b;
        if (abs(d) < tiny)
            d = tiny;

        c = b + an / c;
        if (abs(c) < tiny)
            c = tiny;

        d = f128_s{ 1.0 } / d;
        const f128_s delta = d * c;
        h *= delta;

        if (abs(delta - f128_s{ 1.0 }) <= f128_s{ 32.0 } * f128_s::eps())
            break;
    }

    const f128_s out = exp(-z) * x * _f128_const::inv_sqrtpi * h;
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f128_s lgamma(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return _f128_detail::canonicalize_math_result(_f128_detail::lgamma_positive_recurrence(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = sin(_f128_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        log(_f128_const::pi)
        - log(abs(sinpix))
        - _f128_detail::lgamma_positive_recurrence(f128_s{ 1.0 } - x);

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s tgamma(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return _f128_detail::canonicalize_math_result(_f128_detail::gamma_positive_recurrence(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = sin(_f128_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = _f128_const::pi / (sinpix * _f128_detail::gamma_positive_recurrence(f128_s{ 1.0 } - x));
    return _f128_detail::canonicalize_math_result(out);
}

}

#endif