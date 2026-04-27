#ifndef F256_MATH_INCLUDED
#define F256_MATH_INCLUDED
#include "f256.h"

namespace bl {

/// ------------------ math ------------------

namespace _f256_detail
{
    struct exact_dyadic_fmod
    {
        int exp2 = 0;
        biguint mant{};
    };

    BL_FORCE_INLINE constexpr bool biguint_is_odd(const biguint& value)
    {
        return !value.is_zero() && (value.words[0] & 1u) != 0;
    }
    BL_FORCE_INLINE constexpr bool biguint_any_low_bits_set(const biguint& value, int bit_count)
    {
        if (bit_count <= 0)
            return false;

        const int full_words = bit_count >> 5;
        const int rem_bits = bit_count & 31;

        for (int i = 0; i < full_words && i < value.size; ++i)
        {
            if (value.words[i] != 0)
                return true;
        }

        if (rem_bits != 0 && full_words < value.size)
        {
            const std::uint32_t mask = (std::uint32_t{ 1 } << rem_bits) - 1u;
            if ((value.words[full_words] & mask) != 0)
                return true;
        }

        return false;
    }
    BL_FORCE_INLINE constexpr int biguint_trailing_zero_bits(const biguint& value)
    {
        int count = 0;
        for (int i = 0; i < value.size; ++i)
        {
            const std::uint32_t word = value.words[i];
            if (word == 0)
            {
                count += 32;
                continue;
            }

            std::uint32_t bits = word;
            while ((bits & 1u) == 0u)
            {
                bits >>= 1;
                ++count;
            }
            break;
        }
        return count;
    }
    BL_FORCE_INLINE constexpr biguint biguint_shr_bits(biguint value, int bits)
    {
        if (bits <= 0 || value.is_zero())
            return value;

        const int word_shift = bits >> 5;
        const int bit_shift = bits & 31;

        if (word_shift >= value.size)
        {
            value.clear();
            return value;
        }

        if (word_shift > 0)
        {
            for (int i = 0; i + word_shift < value.size; ++i)
                value.words[i] = value.words[i + word_shift];
            value.size -= word_shift;
        }

        if (bit_shift != 0)
        {
            std::uint32_t carry = 0;
            for (int i = value.size - 1; i >= 0; --i)
            {
                const std::uint32_t next_carry = static_cast<std::uint32_t>(value.words[i] << (32 - bit_shift));
                value.words[i] = static_cast<std::uint32_t>((value.words[i] >> bit_shift) | carry);
                carry = next_carry;
            }
        }

        value.trim();
        return value;
    }
    BL_FORCE_INLINE constexpr biguint biguint_double_mod(biguint remainder, const biguint& modulus)
    {
        remainder.shl1();
        if (remainder.compare(modulus) >= 0)
            remainder.sub_inplace(modulus);
        return remainder;
    }
    BL_FORCE_INLINE constexpr biguint biguint_mod(const biguint& numerator, const biguint& modulus)
    {
        biguint remainder{};
        mod_shift_subtract(numerator, modulus, remainder);
        return remainder;
    }
    BL_FORCE_INLINE constexpr biguint biguint_mul_mod(const biguint& a, const biguint& b, const biguint& modulus)
    {
        if (a.is_zero() || b.is_zero())
            return {};

        return biguint_mod(mul_big(a, b), modulus);
    }
    BL_FORCE_INLINE constexpr biguint biguint_pow2_mod(int exponent, const biguint& modulus)
    {
        if (modulus.is_zero())
            return {};
        if (exponent <= 0)
            return biguint_mod(biguint{ 1u }, modulus);

        biguint result = biguint_mod(biguint{ 1u }, modulus);
        biguint base = biguint_mod(biguint{ 2u }, modulus);

        while (exponent > 0)
        {
            if ((exponent & 1) != 0)
                result = biguint_mul_mod(result, base, modulus);

            exponent >>= 1;
            if (exponent != 0)
                base = biguint_mul_mod(base, base, modulus);
        }

        return result;
    }
    BL_FORCE_INLINE constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
    {
        if (value.mant.is_zero())
        {
            value.exp2 = 0;
            return;
        }

        const int tz = biguint_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = biguint_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }
    BL_FORCE_INLINE constexpr exact_dyadic_fmod exact_from_f256_fmod(const f256_s& x)
    {
        int common_exp = std::numeric_limits<int>::max();
        const double limbs[4] = { x.x0, x.x1, x.x2, x.x3 };

        for (double limb : limbs)
        {
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            if (exponent < common_exp)
                common_exp = exponent;
        }

        exact_dyadic_fmod out{};
        if (common_exp == std::numeric_limits<int>::max())
            return out;

        signed_biguint acc{};
        for (double limb : limbs)
        {
            if (limb == 0.0)
                continue;

            int exponent = 0;
            bool limb_neg = false;
            const std::uint64_t mantissa = decompose_double_mantissa(limb, exponent, limb_neg);
            if (mantissa == 0)
                continue;

            biguint term{ mantissa };
            term.shl_bits(exponent - common_exp);
            add_signed(acc, term, limb_neg);
        }

        if (acc.neg || acc.mag.is_zero())
            return out;

        out.exp2 = common_exp;
        out.mant = acc.mag;
        normalize_exact_dyadic_fmod(out);
        return out;
    }
    BL_FORCE_INLINE constexpr f256_s exact_dyadic_to_f256_fmod(const biguint& coeff, int exp2, bool neg)
    {
        if (coeff.is_zero())
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        constexpr int kept_bits = 53 * 5;
        int ratio_exp = coeff.bit_length() - 1;
        biguint q = coeff;

        if (ratio_exp > (kept_bits - 1))
        {
            const int right_shift = ratio_exp - (kept_bits - 1);
            const bool round_bit = q.get_bit(right_shift - 1);
            const bool sticky = biguint_any_low_bits_set(q, right_shift - 1);

            q = biguint_shr_bits(q, right_shift);

            if (round_bit && (sticky || biguint_is_odd(q)))
                q.add_small(1u);

            if (q.bit_length() > kept_bits)
            {
                q = biguint_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < (kept_bits - 1))
        {
            q.shl_bits((kept_bits - 1) - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f256_s>::infinity() : std::numeric_limits<f256_s>::infinity();
        if (e2 < -1074)
            return neg ? f256_s{ -0.0, 0.0, 0.0, 0.0 } : f256_s{ 0.0, 0.0, 0.0, 0.0 };

        const std::uint64_t c4 = q.get_bits(0, 53);
        const std::uint64_t c3 = q.get_bits(53, 53);
        const std::uint64_t c2 = q.get_bits(106, 53);
        const std::uint64_t c1 = q.get_bits(159, 53);
        const std::uint64_t c0 = q.get_bits(212, 53);

        const double x0 = c0 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double x1 = c1 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
        const double x2 = c2 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c2), e2 - 158) : 0.0;
        const double x3 = c3 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c3), e2 - 211) : 0.0;
        const double x4 = c4 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c4), e2 - 264) : 0.0;

        f256_s out = _f256_detail::renorm5(x0, x1, x2, x3, x4);
        return neg ? -out : out;
    }
    BL_FORCE_INLINE constexpr f256_s fmod_exact(const f256_s& x, const f256_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f256_fmod(abs(x));
        const exact_dyadic_fmod dy = exact_from_f256_fmod(abs(y));

        if (dx.mant.is_zero() || dy.mant.is_zero())
            return f256_s{ _f256_detail::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

        biguint remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            biguint denominator = dy.mant;
            denominator.shl_bits(shift);
            mod_shift_subtract(dx.mant, denominator, remainder);
            out_exp = dx.exp2;
        }
        else
        {
            remainder = biguint_mod(dx.mant, dy.mant);
            const int shift = dx.exp2 - dy.exp2;
            if (!remainder.is_zero() && shift != 0)
            {
                const biguint scale = biguint_pow2_mod(shift, dy.mant);
                remainder = biguint_mul_mod(remainder, scale, dy.mant);
            }
            out_exp = dy.exp2;
        }

        f256_s out = exact_dyadic_to_f256_fmod(remainder, out_exp, !ispositive(x));
        if (iszero(out))
            return f256_s{ _f256_detail::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return out;
    }
    BL_FORCE_INLINE constexpr bool fmod_fast_double_divisor_abs(const f256_s& ax, double ay, f256_s& out)
    {
        if (!(ay > 0.0) || !_f256_detail::isfinite(ay))
            return false;

        const f256_s mod{ ay, 0.0, 0.0, 0.0 };

        if (ax.x1 == 0.0 && ax.x2 == 0.0 && ax.x3 == 0.0)
        {
            out = f256_s{ _f256_detail::fmod_constexpr(ax.x0, ay), 0.0, 0.0, 0.0 };
            return true;
        }

        const double r0 = (ax.x0 < ay) ? ax.x0 : _f256_detail::fmod_constexpr(ax.x0, ay);
        const double r1 = (_f256_detail::absd(ax.x1) < ay) ? ax.x1 : _f256_detail::fmod_constexpr(ax.x1, ay);
        const double r2 = (_f256_detail::absd(ax.x2) < ay) ? ax.x2 : _f256_detail::fmod_constexpr(ax.x2, ay);
        const double r3 = (_f256_detail::absd(ax.x3) < ay) ? ax.x3 : _f256_detail::fmod_constexpr(ax.x3, ay);

        f256_s r = f256_s{ r0, 0.0, 0.0, 0.0 } +
            f256_s{ r1, 0.0, 0.0, 0.0 } +
            f256_s{ r2, 0.0, 0.0, 0.0 } +
            f256_s{ r3, 0.0, 0.0, 0.0 };

        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
                r += mod;
            if (r >= mod)
                r -= mod;
        }

        if (r < 0.0 || r >= mod)
            return false;

        // reject boundary-adjacent results so the exact fallback handles the
        // cases where double-limb modular reduction is not strong enough
        const f256_s ar = abs(r);
        const f256_s slack = mod * f256_s{ 0x1p-160 };
        if (ar <= slack || ar >= mod - slack)
            return false;

        out = r;
        return true;
    }
    BL_FORCE_INLINE constexpr bool fmod_fast_qd_divisor_abs(const f256_s& ax, const f256_s& ay, f256_s& out)
    {
        if (!(ay > 0.0))
            return false;

        const f256_s q_floor = floor(ax / ay);
        if (q_floor.x1 != 0.0 || q_floor.x2 != 0.0 || q_floor.x3 != 0.0)
            return false;
        if (_f256_detail::absd(q_floor.x0) >= 0x1p53)
            return false;

        const double q = q_floor.x0;
        f256_s r = _f256_detail::sub_mul_scalar_exact(ax, ay, q);

        for (int i = 0; i < 4; ++i)
        {
            if (r < 0.0)
            {
                r += ay;
                continue;
            }

            if (r >= ay)
            {
                r -= ay;
                continue;
            }

            out = r;
            return true;
        }

        if (r < 0.0 || r >= ay)
            return false;

        out = r;
        return true;
    }
    BL_FORCE_INLINE constexpr bool f256_try_get_int64(const f256_s& x, int64_t& out)
    {
        const f256_s xi = trunc(x);
        if (xi != x)
            return false;

        if (_f256_detail::absd(xi.x0) >= 0x1p63)
            return false;

        const int64_t p0 = static_cast<int64_t>(xi.x0);
        const f256_s r0 = xi - to_f256(p0);
        const int64_t p1 = static_cast<int64_t>(r0.x0);
        const f256_s r1 = r0 - to_f256(p1);
        const int64_t p2 = static_cast<int64_t>(r1.x0);
        const f256_s r2 = r1 - to_f256(p2);
        const int64_t p3 = static_cast<int64_t>(r2.x0 + r2.x1 + r2.x2 + r2.x3);

        out = p0 + p1 + p2 + p3;
        return true;
    }
    BL_FORCE_INLINE constexpr f256_s powi(f256_s base, int64_t exp)
    {
        if (exp == 0)
            return f256_s{ 1.0 };

        const bool invert = exp < 0;
        uint64_t n = invert ? _f256_detail::magnitude_u64(exp) : static_cast<uint64_t>(exp);
        f256_s result{ 1.0 };

        while (n != 0)
        {
            if ((n & 1u) != 0)
                result *= base;

            n >>= 1;
            if (n != 0)
                base *= base;
        }

        return invert ? (f256_s{ 1.0 } / result) : result;
    }
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s fmod(const f256_s& x, const f256_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax < ay)
        return x;

    f256_s fast{};
    if (y.x1 == 0.0 && y.x2 == 0.0 && y.x3 == 0.0 && _f256_detail::fmod_fast_double_divisor_abs(ax, ay.x0, fast))
    {
        if (iszero(fast))
            return f256_s{ _f256_detail::signbit_constexpr(x.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return _f256_detail::fmod_exact(x, y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s round(const f256_s& a)
{
    f256_s t = floor(a + f256_s{ 0.5 });
    if ((t - a) == f256_s{ 0.5 } && fmod(t, f256_s{ 2.0 }) != f256_s{ 0.0 })
        t -= f256_s{ 1.0 };
    return t;
}
[[nodiscard]] BL_FORCE_INLINE f256_s round_to_decimals(f256_s v, int prec)
{
    if (prec <= 0) return v;

    static constexpr f256_s inv10_qd{
         0x1.999999999999ap-4,
        -0x1.999999999999ap-58,
         0x1.999999999999ap-112,
        -0x1.999999999999ap-166
    };

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f256_s ip = floor(v);
    f256_s frac = v - ip;

    std::string dig;
    dig.reserve((size_t)prec);

    f256_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;
        int di = (int)floor(w).x0;
        if (di < 0) di = 0;
        else if (di > 9) di = 9;
        dig.push_back(char('0' + di));
        w = w - f256_s{ (double)di };
    }

    f256_s la = w * 10.0;
    int next = (int)floor(la).x0;
    if (next < 0) next = 0;
    else if (next > 9) next = 9;
    f256_s rem = la - f256_s{ (double)next };

    const int last = dig.empty() ? 0 : (dig.back() - '0');
    const bool round_up =
        (next > 5) ||
        (next == 5 && (rem.x0 > 0.0 || rem.x1 > 0.0 || rem.x2 > 0.0 || rem.x3 > 0.0 || (last & 1)));

    if (round_up)
    {
        int i = prec - 1;
        for (; i >= 0; --i)
        {
            if (dig[(size_t)i] == '9') dig[(size_t)i] = '0';
            else
            {
                ++dig[(size_t)i];
                break;
            }
        }

        if (i < 0)
            ip = ip + 1.0;
    }

    f256_s frac_val{ 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = frac_val + f256_s{ (double)(dig[(size_t)i] - '0') };
        frac_val = frac_val * inv10_qd;
    }

    f256_s out = ip + frac_val;
    return neg ? -out : out;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqrt(const f256_s& a)
{
    if (a.x0 <= 0.0)
    {
        if (iszero(a))
            return a;
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    if (isinf(a))
        return a;

    double y0 = _f256_detail::sqrt_seed_constexpr(a.x0);

    f256_s y{ y0, 0.0, 0.0, 0.0 };
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);

    return y;
}
// y.x3 = fltx::common::fp::zero_low_fraction_bits_finite<9>(y.x3);
// return y;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s nearbyint(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f256_s t = floor(a);
    const f256_s frac = a - t;

    if (frac < f256_s{ 0.5 })
        return t;

    if (frac > f256_s{ 0.5 })
    {
        t += f256_s{ 1.0 };
        if (iszero(t))
            return f256_s{ _f256_detail::signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        return t;
    }

    if (fmod(t, f256_s{ 2.0 }) != f256_s{ 0.0 })
        t += f256_s{ 1.0 };

    if (iszero(t))
        return f256_s{ _f256_detail::signbit_constexpr(a.x0) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return t;
}

/// ------------------ transcendentals ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(f256_s a) noexcept
{
    const double hi = a.x0;
    if (hi <= 0.0)
        return fltx::common::fp::log_constexpr(static_cast<double>(a));

    const double lo = (a.x1 + a.x2) + a.x3;
    return fltx::common::fp::log_constexpr(hi) + fltx::common::fp::log1p_constexpr(lo / hi);
}

namespace _f256_const
{
    inline constexpr f256_s e = std::numbers::e_v<f256_s>;
    inline constexpr f256_s log2e = std::numbers::log2e_v<f256_s>;
    inline constexpr f256_s log10e = std::numbers::log10e_v<f256_s>;
    inline constexpr f256_s pi = std::numbers::pi_v<f256_s>;
    inline constexpr f256_s inv_pi = std::numbers::inv_pi_v<f256_s>;
    inline constexpr f256_s inv_sqrtpi = std::numbers::inv_sqrtpi_v<f256_s>;
    inline constexpr f256_s ln2 = std::numbers::ln2_v<f256_s>;
    inline constexpr f256_s ln10 = std::numbers::ln10_v<f256_s>;
    inline constexpr f256_s sqrt2 = std::numbers::sqrt2_v<f256_s>;
    inline constexpr f256_s sqrt3 = std::numbers::sqrt3_v<f256_s>;
    inline constexpr f256_s inv_sqrt3 = std::numbers::inv_sqrt3_v<f256_s>;
    inline constexpr f256_s egamma = std::numbers::egamma_v<f256_s>;
    inline constexpr f256_s phi = std::numbers::phi_v<f256_s>;

    inline constexpr f256_s pi_2 = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54, -0x1.f1976b7ed8fbcp-110,  0x1.4cf98e804177dp-164 };
    inline constexpr f256_s pi_4 = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55, -0x1.f1976b7ed8fbcp-111,  0x1.4cf98e804177dp-165 };
    inline constexpr f256_s invpi2 = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55, -0x1.6447e493ad4cep-109,  0x1.e21c820ff28b2p-163 };
    inline constexpr f256_s pi_3_4 = pi_2 + pi_4;
    inline constexpr f256_s inv_ln2 = log2e;
    inline constexpr f256_s inv_ln10 = log10e;
    inline constexpr f256_s sqrt_half = { 0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55,  0x1.57d3e3adec175p-109,  0x1.2775099da2f59p-165 };
    inline constexpr f256_s half_log_two_pi = { 0x1.d67f1c864beb5p-1, -0x1.65b5a1b7ff5dfp-55, -0x1.b7f70c13dc1ccp-110, 0x1.3458b4ddec6a3p-164 };
}
namespace _f256_detail
{
    inline constexpr f256_s exp_inv_fact[] = {
        f256_s{ 1.66666666666666657e-01,  9.25185853854297066e-18,  5.13581318503262866e-34,  2.85094902409834186e-50 },
        f256_s{ 4.16666666666666644e-02,  2.31296463463574266e-18,  1.28395329625815716e-34,  7.12737256024585466e-51 },
        f256_s{ 8.33333333333333322e-03,  1.15648231731787138e-19,  1.60494162032269652e-36,  2.22730392507682967e-53 },
        f256_s{ 1.38888888888888894e-03, -5.30054395437357706e-20, -1.73868675534958776e-36, -1.63335621172300840e-52 },
        f256_s{ 1.98412698412698413e-04,  1.72095582934207053e-22,  1.49269123913941271e-40,  1.29470326746002471e-58 },
        f256_s{ 2.48015873015873016e-05,  2.15119478667758816e-23,  1.86586404892426588e-41,  1.61837908432503088e-59 },
        f256_s{ 2.75573192239858925e-06, -1.85839327404647208e-22,  8.49175460488199287e-39, -5.72661640789429621e-55 },
        f256_s{ 2.75573192239858883e-07,  2.37677146222502973e-23, -3.26318890334088294e-40,  1.61435111860404415e-56 },
        f256_s{ 2.50521083854417202e-08, -1.44881407093591197e-24,  2.04267351467144546e-41, -8.49632672007163175e-58 },
        f256_s{ 2.08767569878681002e-09, -1.20734505911325997e-25,  1.70222792889287100e-42,  1.41609532150396700e-58 },
        f256_s{ 1.60590438368216133e-10,  1.25852945887520981e-26, -5.31334602762985031e-43,  3.54021472597605528e-59 },
        f256_s{ 1.14707455977297245e-11,  2.06555127528307454e-28,  6.88907923246664603e-45,  5.72920002655109095e-61 },
        f256_s{ 7.64716373181981641e-13,  7.03872877733453001e-30, -7.82753927716258345e-48,  1.92138649443790242e-64 },
        f256_s{ 4.77947733238738525e-14,  4.39920548583408126e-31, -4.89221204822661465e-49,  1.20086655902368901e-65 },
        f256_s{ 2.81145725434552060e-15,  1.65088427308614326e-31, -2.87777179307447918e-50,  4.27110689256293549e-67 }
    };

    BL_FORCE_INLINE constexpr f256_s f256_expm1_tiny(const f256_s& r)
    {
        f256_s p = exp_inv_fact[(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 1];
        for (int i = static_cast<int>(sizeof(exp_inv_fact) / sizeof(exp_inv_fact[0])) - 2; i >= 0; --i)
            p = p * r + exp_inv_fact[i];
        p = p * r + f256_s{ 0.5 };
        return r + (r * r) * p;
    }
    BL_FORCE_INLINE constexpr f256_s f256_log1p_series_reduced(const f256_s& x)
    {
        const f256_s z = x / (f256_s{ 2.0 } + x);
        const f256_s z2 = z * z;

        f256_s term = z;
        f256_s sum = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term *= z2;
            const f256_s add = term / f256_s{ static_cast<double>(k) };
            sum += add;

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(add) <= f256_s::eps() * scale)
                break;
        }

        return sum + sum;
    }


    inline constexpr f256_s lgamma1p_coeff[] = {
        f256_s{ 0x1.a51a6625307d3p-1, 0x1.1873d8912200cp-56, -0x1.4c68528ddc956p-110, 0x1.162d8b33582c0p-168 },
        f256_s{ -0x1.9a4d55beab2d7p-2, 0x1.4c26d1b465993p-59, -0x1.aa121007a9210p-113, 0x1.e77545b273b41p-167 },
        f256_s{ 0x1.151322ac7d848p-2, 0x1.b5f91211196e5p-57, 0x1.1afde2c358986p-112, -0x1.ff5e9b485c055p-167 },
        f256_s{ -0x1.a8b9c17aa6149p-3, -0x1.2e826a4fdae1ap-58, 0x1.bec8fd99e4b23p-112, 0x1.c4d2077d59b06p-166 },
        f256_s{ 0x1.5b40cb100c306p-3, 0x1.4a79940f15696p-59, -0x1.38825ea888f47p-113, -0x1.465029c2b0433p-167 },
        f256_s{ -0x1.2703a1dcea3aep-3, -0x1.6307fd0794ac4p-57, 0x1.fcb7807245585p-111, 0x1.f0664358361f4p-166 },
        f256_s{ 0x1.010b36af86397p-3, -0x1.741a635b224a6p-59, 0x1.9336e1bce5c27p-113, 0x1.49afd8254897cp-167 },
        f256_s{ -0x1.c806706d57db4p-4, -0x1.56aa806fdd3eep-58, 0x1.bbb9c2de4a62ap-112, -0x1.3011b58722bd7p-167 },
        f256_s{ 0x1.9a01e385d5f8fp-4, 0x1.813418f3768cdp-59, 0x1.9ac3b8f78d2dbp-113, 0x1.f9d6fac7bc2bep-167 },
        f256_s{ -0x1.748c33114c6d6p-4, -0x1.ea57624080720p-61, 0x1.d4f09980d4de7p-116, -0x1.ecb44a07a7c5dp-170 },
        f256_s{ 0x1.556ad63243bc4p-4, 0x1.5de8580fae81dp-62, 0x1.cccd6abe647edp-119, 0x1.1be9a3144317ap-173 },
        f256_s{ -0x1.3b1d971fc5985p-4, 0x1.e58607e493dfdp-59, -0x1.abfc7225b8175p-113, -0x1.2ecb61bf48473p-169 },
        f256_s{ 0x1.2496df8320c5fp-4, 0x1.cf4b4ae040be8p-58, 0x1.4c882cc4762e8p-112, 0x1.7476f52945b0fp-166 },
        f256_s{ -0x1.11133476e7fe0p-4, -0x1.dc9a4ff396ee3p-59, -0x1.3b08c41a7a8b6p-113, 0x1.cf5c5597a8f3ep-168 },
        f256_s{ 0x1.00010064cdeb2p-4, 0x1.7879d0156affep-59, -0x1.0fbd29f2ffe91p-113, 0x1.89bce2341cdd7p-167 },
        f256_s{ -0x1.e1e2d311e8abdp-5, 0x1.8d2a110ce956bp-59, 0x1.63ee8a858cae0p-113, -0x1.e39f4153afc89p-167 },
        f256_s{ 0x1.c71ce3a20b419p-5, -0x1.be9617d035b06p-59, 0x1.89baba83cec5cp-115, -0x1.dcfb4b6decc54p-169 },
        f256_s{ -0x1.af28a1b5688a0p-5, -0x1.74741e885fefbp-59, -0x1.1b477c35fac2fp-113, 0x1.d7c330022aca2p-168 },
        f256_s{ 0x1.9999b3352d5bap-5, 0x1.4951b4c6be56dp-62, -0x1.ab8f8f67d6af8p-118, 0x1.986c8bd56b1b5p-172 },
        f256_s{ -0x1.86186db77bfbfp-5, -0x1.6dedef1f58778p-59, 0x1.f067254ca5106p-114, -0x1.fba80968fedbfp-168 },
        f256_s{ 0x1.745d1d1778df9p-5, 0x1.02b8fe0a898e7p-61, 0x1.c5b2f696e8978p-115, 0x1.8192959451748p-169 },
        f256_s{ -0x1.642c88591b66dp-5, 0x1.1074551cafc60p-59, -0x1.d738a4a5cbb56p-116, 0x1.1369f37f73d94p-170 },
        f256_s{ 0x1.555556aaafdcdp-5, 0x1.54a05fce04ef6p-59, 0x1.7588d216e6fc9p-114, -0x1.c8c5449ed87ecp-170 },
        f256_s{ -0x1.47ae151eb9fb7p-5, -0x1.d038d4d4653c2p-59, -0x1.7ebe42832f35dp-113, -0x1.28f5805cfb70ap-167 },
        f256_s{ 0x1.3b13b189d925ep-5, 0x1.f4ad5a89f860cp-59, 0x1.c3ba6ba46072cp-113, 0x1.dbc96b627a7eap-167 },
        f256_s{ -0x1.2f684c00002bcp-5, -0x1.055a3ba5e6a12p-59, 0x1.15f6f174f29fcp-114, -0x1.a4d95a3ce102dp-168 },
        f256_s{ 0x1.24924936db7bcp-5, 0x1.f2631c34f2cbcp-59, 0x1.6d33663f9d067p-116, -0x1.8e49cb0d4669dp-172 },
        f256_s{ -0x1.1a7b961a7b9aap-5, 0x1.e116d2f11b9bcp-59, -0x1.7ae84ff57e3e8p-113, 0x1.48411ad6c2f3cp-168 },
        f256_s{ 0x1.111111155556dp-5, -0x1.527ce242d7c8fp-59, -0x1.c6b75ac93ef74p-116, -0x1.59689b24da75ap-170 },
        f256_s{ -0x1.08421086318cep-5, 0x1.1db4d8fcae8c6p-59, -0x1.8aff5794eab89p-114, 0x1.29680715cc491p-169 },
        f256_s{ 0x1.0000000100002p-5, 0x1.b8fd913d3546ap-59, 0x1.815322257c298p-113, -0x1.6a18198001fe7p-168 },
        f256_s{ -0x1.f07c1f08ba2eap-6, -0x1.31bb2e9036633p-60, 0x1.151f1a77d048fp-115, -0x1.19b08afd4a13dp-171 },
        f256_s{ 0x1.e1e1e1e25a5a6p-6, 0x1.3e46eaa03f9ccp-61, 0x1.2f952ffc8aa9bp-115, 0x1.b1cb7727dbed5p-172 },
        f256_s{ -0x1.d41d41d457c58p-6, 0x1.0600661f0f0e3p-62, 0x1.8207e60d9c037p-116, 0x1.aa0de68815246p-170 },
        f256_s{ 0x1.c71c71c738e39p-6, -0x1.d93a55599cf57p-63, 0x1.4c492654868edp-117, 0x1.710b595a6ccadp-171 },
        f256_s{ -0x1.bacf914c29837p-6, -0x1.797fe7c73f29ap-60, 0x1.9a39cef32197ap-116, -0x1.a972d739b9af9p-171 },
        f256_s{ 0x1.af286bca21af3p-6, -0x1.df4d835f028bdp-60, 0x1.a8048d2fde67cp-114, -0x1.762e068021af3p-168 },
        f256_s{ -0x1.a41a41a41d89ep-6, 0x1.d6bf77cbc25c7p-60, 0x1.aa5ec7c867b9bp-114, 0x1.aec81d4fdf930p-169 },
        f256_s{ 0x1.999999999b333p-6, 0x1.9ad0584412591p-61, -0x1.5494ee7aa6acbp-115, 0x1.017dce762d13fp-170 },
        f256_s{ -0x1.8f9c18f9c2577p-6, 0x1.766fd061292d7p-60, 0x1.8f5129cb00964p-114, 0x1.c84bbfd5adbcep-170 },
        f256_s{ 0x1.8618618618c31p-6, -0x1.e77d97e1c5a45p-61, -0x1.917b2ef884989p-116, -0x1.2370e4fefbcf8p-171 },
        f256_s{ -0x1.7d05f417d08eep-6, -0x1.1dcf2bd1488c1p-61, 0x1.653c4b9cdc9cbp-119, 0x1.ce66a9ae95282p-173 },
        f256_s{ 0x1.745d1745d18bap-6, 0x1.7460941753bf5p-61, -0x1.7ff16da4c5b6dp-115, -0x1.a4622b9427056p-169 },
        f256_s{ -0x1.6c16c16c16ccdp-6, 0x1.9998769b89af0p-61, 0x1.e206e60394ebap-115, 0x1.da1d89108d728p-174 },
        f256_s{ 0x1.642c8590b21bdp-6, 0x1.bd3805d865a75p-61, 0x1.4fa64231f4873p-116, 0x1.48681770a7db2p-172 },
        f256_s{ -0x1.5c9882b931083p-6, 0x1.1b3bdabc05a8dp-60, -0x1.e180a2e6d8d8ap-115, -0x1.adf0f5b6511bdp-169 },
        f256_s{ 0x1.555555555556bp-6, -0x1.555550480911cp-60, -0x1.7e33c2f812c41p-115, 0x1.fc4f9b927ed2fp-173 },
        f256_s{ -0x1.4e5e0a72f0544p-6, 0x1.4e5e03d9bbd88p-62, 0x1.8ad74f7d864c5p-118, 0x1.de83badc29713p-174 },
        f256_s{ 0x1.47ae147ae1480p-6, 0x1.13e7474dcd9a5p-85, 0x1.f485e7aa34c2bp-140, 0x1.03954ca6cce4bp-194 },
        f256_s{ -0x1.4141414141417p-6, 0x1.a5a5a57890971p-60, 0x1.683bc59ca465cp-116, -0x1.bec1a58fd95fbp-170 },
        f256_s{ 0x1.3b13b13b13b15p-6, -0x1.3b13b1001f8aep-62, -0x1.204bb43c1d47fp-117, -0x1.0119b57ca7085p-171 },
        f256_s{ -0x1.3521cfb2b78c2p-6, 0x1.826a4395c1891p-61, 0x1.a4b38e003722bp-116, -0x1.cce6803948c97p-170 },
        f256_s{ 0x1.2f684bda12f69p-6, -0x1.a12f684a465ffp-60, 0x1.bad2fc9ce5f0bp-114, 0x1.bfc93a89e124bp-168 },
        f256_s{ -0x1.29e4129e4129ep-6, -0x1.9999999a1db84p-60, -0x1.4b1b19906daf9p-114, 0x1.dc501d5c7a0cep-168 },
        f256_s{ 0x1.2492492492492p-6, 0x1.6db6db6de21c5p-60, 0x1.c06a19ff7e3e7p-115, -0x1.c1d6ed17c4a8ep-173 },
        f256_s{ -0x1.1f7047dc11f70p-6, -0x1.435e50d7a2602p-60, 0x1.2b58ebdccf2bcp-114, 0x1.3df12156c2cb2p-168 },
        f256_s{ 0x1.1a7b9611a7b96p-6, 0x1.611a7b9624375p-62, -0x1.53910f155f6e2p-122, 0x1.cdf4e01c19e6fp-176 },
        f256_s{ -0x1.15b1e5f75270dp-6, -0x1.a08ad8f313fd5p-64, 0x1.694e002134230p-118, -0x1.19c40b553edeep-172 },
        f256_s{ 0x1.1111111111111p-6, 0x1.2222222224208p-62, -0x1.660eeab940148p-117, 0x1.3f841c542be73p-175 },
        f256_s{ -0x1.0c9714fbcda3bp-6, 0x1.f368eb043208bp-61, -0x1.9180e02b1ebb4p-115, -0x1.ae60aa3d03da9p-169 },
        f256_s{ 0x1.0842108421084p-6, 0x1.0a5294a52965cp-61, 0x1.6a716f3fe41bbp-116, -0x1.3966f9e59e4cep-171 },
        f256_s{ -0x1.0410410410410p-6, -0x1.04924924924dap-60, -0x1.2c9f914026d3ep-114, 0x1.1aa441e1febb5p-170 },
        f256_s{ 0x1.0000000000000p-6, 0x1.0000000005e83p-70, -0x1.6ea6b12420976p-124, 0x1.46b6024a71bf4p-179 },
        f256_s{ -0x1.f81f81f81f820p-7, 0x1.f7e07e07e07d1p-61, -0x1.17cedd5cd4340p-119, -0x1.231cb51f5fa1ap-175 },
        f256_s{ 0x1.f07c1f07c1f08p-7, -0x1.f03e0f83e0f7ap-62, 0x1.37710ca3a5bb1p-116, -0x1.c5ed76f77cd34p-171 },
        f256_s{ -0x1.e9131abf0b767p-7, -0x1.505bb39503d26p-62, 0x1.12bb0a2d9bbe6p-116, -0x1.08b90d7460356p-173 },
        f256_s{ 0x1.e1e1e1e1e1e1ep-7, 0x1.e200000000002p-63, 0x1.920f4caafc2b6p-118, -0x1.71978c07e1374p-172 },
        f256_s{ -0x1.dae6076b981dbp-7, 0x1.9f7a6f4de9bd3p-63, -0x1.f2252e21e87a9p-118, 0x1.83d4b1dbeca59p-172 },
        f256_s{ 0x1.d41d41d41d41dp-7, 0x1.0752492492492p-61, 0x1.614270fa6214ep-115, -0x1.f77a00fa73fe6p-171 },
        f256_s{ -0x1.cd85689039b0bp-7, 0x1.76fa976fc64f5p-62, 0x1.2735476e24ed0p-117, -0x1.f8b4320455699p-173 },
        f256_s{ 0x1.c71c71c71c71cp-7, 0x1.c71ce38e38e39p-61, -0x1.ace34a7163d76p-117, -0x1.1f77c846a312ap-172 },
        f256_s{ -0x1.c0e070381c0e0p-7, -0x1.c0e0a8542a151p-61, 0x1.5c87938fc211bp-115, 0x1.f7dd13bd0c614p-170 },
        f256_s{ 0x1.bacf914c1bad0p-7, -0x1.bacf759f22983p-61, -0x1.d5c71a9cd2d7bp-115, -0x1.f2ba4ed0c813cp-171 },
        f256_s{ -0x1.b4e81b4e81b4fp-7, 0x1.f92c51eb851ecp-61, -0x1.ebc0cadfc39dcp-115, -0x1.d86dcb378da2fp-172 },
        f256_s{ 0x1.af286bca1af28p-7, 0x1.af287286bca1bp-61, -0x1.ae8b63624e57ep-118, -0x1.858b16f233d73p-176 },
        f256_s{ -0x1.a98ef606a63bep-7, 0x1.f959c0d4c77b0p-61, 0x1.a9820b7fc62f2p-116, -0x1.f72b8dbc6b5d5p-175 },
        f256_s{ 0x1.a41a41a41a41ap-7, 0x1.06906aaaaaaabp-61, -0x1.5553354f8d76dp-115, 0x1.0d86d971be33fp-169 },
        f256_s{ -0x1.9ec8e951033d9p-7, -0x1.d2a209b8b577ep-63, -0x1.84df26ea720dcp-117, 0x1.af04d98dfad02p-172 },
        f256_s{ 0x1.999999999999ap-7, -0x1.9999993333333p-61, -0x1.999923ba7b188p-116, 0x1.c2ee41f7b15dcp-170 },
        f256_s{ -0x1.948b0fcd6e9e0p-7, -0x1.948b100000000p-61, -0x1.3671909a8aa7dp-135, -0x1.13181f3d7138fp-189 },
        f256_s{ 0x1.8f9c18f9c18fap-7, -0x1.f3831f063e706p-62, -0x1.f38305aa29f35p-117, 0x1.a672177cf91d8p-171 },
        f256_s{ -0x1.8acb90f6bf3aap-7, 0x1.721ed7dafcea7p-61, -0x1.c87b61b7e6f4bp-115, -0x1.5d61cf8be7a0cp-169 },
        f256_s{ 0x1.8618618618618p-7, 0x1.8618618c30c31p-61, -0x1.e79e7884d38b8p-116, -0x1.9d9fb59a71375p-174 },
        f256_s{ -0x1.8181818181818p-7, -0x1.8181818d8d8d9p-63, 0x1.39393765bb628p-118, -0x1.3b4d334cdaa33p-172 },
        f256_s{ 0x1.7d05f417d05f4p-7, 0x1.7d05f41dc4771p-63, 0x1.dc477251cdf4bp-119, 0x1.b79aeab06c17dp-176 },
        f256_s{ -0x1.78a4c8178a4c8p-7, -0x1.78a4c81a7b961p-63, -0x1.a7b9617ffb47dp-119, 0x1.cb1cecaaf7e8ap-174 },
        f256_s{ 0x1.745d1745d1746p-7, -0x1.745d17451745dp-62, -0x1.745d1735180cep-118, -0x1.65e2a01ab292dp-172 },
        f256_s{ -0x1.702e05c0b8170p-7, -0x1.702e05c114228p-62, -0x1.14228451ead7ap-116, -0x1.83351baaf0d6fp-174 },
        f256_s{ 0x1.6c16c16c16c17p-7, -0x1.f49f49f471c72p-62, 0x1.c71c71c80503cp-117, -0x1.a7fcfb6c46765p-171 },
        f256_s{ -0x1.6816816816817p-7, 0x1.fa5fa5fa54654p-61, 0x1.951951950626ap-115, -0x1.09210a392fa45p-169 },
        f256_s{ 0x1.642c8590b2164p-7, 0x1.642c8590bd37ap-62, 0x1.bd37a6f4eb3f9p-116, 0x1.60eb1b25b5ad2p-170 },
        f256_s{ -0x1.6058160581606p-7, 0x1.fa7e9fa7e739dp-61, -0x1.8c6318c639e26p-117, 0x1.7351b5a7b933ap-172 },
        f256_s{ 0x1.5c9882b931057p-7, 0x1.310572620d9dfp-62, 0x1.46cefa8d9f550p-116, -0x1.db2f84890bedcp-171 },
        f256_s{ -0x1.58ed2308158edp-7, -0x1.1840ac7692dcfp-62, -0x1.fa9c4b73e01ddp-116, -0x1.d473788da23dbp-171 },
        f256_s{ 0x1.5555555555555p-7, 0x1.5555555555aabp-61, -0x1.5555555555423p-115, -0x1.174105986442cp-171 },
        f256_s{ -0x1.51d07eae2f815p-7, -0x1.d07eae2f81facp-63, 0x1.d07eae2f81389p-117, -0x1.cdf5daadab33dp-173 },
        f256_s{ 0x1.4e5e0a72f0539p-7, 0x1.e0a72f05398d1p-61, -0x1.4e5e0a72f0324p-119, -0x1.c12afec73670cp-175 },
        f256_s{ -0x1.4afd6a052bf5bp-7, 0x1.fad40a57eb45dp-61, 0x1.745d1745d171ap-117, -0x1.6fa0eb9403245p-172 },
        f256_s{ 0x1.47ae147ae147bp-7, -0x1.eb851eb851d71p-63, 0x1.70a3d70a3d719p-117, -0x1.ed1522e63fd11p-172 },
        f256_s{ -0x1.446f86562d9fbp-7, 0x1.1be1958b67e19p-63, 0x1.62d9faee41e66p-117, -0x1.58747c5a6bc0cp-171 },
        f256_s{ 0x1.4141414141414p-7, 0x1.4141414141464p-63, 0x1.919191919191bp-117, -0x1.480866d9ca729p-171 },
        f256_s{ -0x1.3e22cbce4a902p-7, -0x1.f1165e725481ep-61, 0x1.65e7254813e23p-116, -0x1.dc15cf8adce0bp-170 },
        f256_s{ 0x1.3b13b13b13b14p-7, -0x1.3b13b13b13b0fp-61, 0x1.d89d89d89d89ep-116, -0x1.805e8b12fa608p-170 },
        f256_s{ -0x1.3813813813814p-7, 0x1.fb1fb1fb1fb1dp-61, 0x1.0750750750750p-115, 0x1.c58bf33315042p-169 },
        f256_s{ 0x1.3521cfb2b78c1p-7, 0x1.a90e7d95bc60cp-62, 0x1.3521cfb2b78c1p-118, 0x1.f6047d965bd53p-173 },
        f256_s{ -0x1.323e34a2b10bfp-7, -0x1.9b8396ba9de82p-61, 0x1.a515885fb3707p-116, 0x1.654f614bb639ep-171 },
        f256_s{ 0x1.2f684bda12f68p-7, 0x1.2f684bda12f69p-61, -0x1.a12f684bda12fp-115, -0x1.a0a91f24de414p-169 },
        f256_s{ -0x1.2c9fb4d812ca0p-7, 0x1.2c9fb4d812ca0p-61, -0x1.c2ef8f441c2f0p-115, 0x1.c2c335581d907p-169 },
        f256_s{ 0x1.29e4129e4129ep-7, 0x1.04a7904a7904bp-61, -0x1.d1745d1745d17p-115, -0x1.17372b38419eap-169 },
        f256_s{ -0x1.27350b8812735p-7, -0x1.71024e6a17103p-64, 0x1.9f22983759f23p-118, -0x1.9f494ebf2163bp-172 },
        f256_s{ 0x1.2492492492492p-7, 0x1.2492492492492p-61, 0x1.36db6db6db6dbp-115, 0x1.b6dd06f7523c6p-169 }
    };

    BL_NO_INLINE constexpr f256_s lgamma1p_series(const f256_s& y) noexcept
    {
        constexpr int count = static_cast<int>(sizeof(lgamma1p_coeff) / sizeof(lgamma1p_coeff[0]));

        f256_s p = lgamma1p_coeff[count - 1];
        for (int i = count - 2; i >= 0; --i)
            p = p * y + lgamma1p_coeff[i];

        return y * (-_f256_const::egamma + y * p);
    }

    BL_FORCE_INLINE constexpr bool try_lgamma_near_one_or_two(const f256_s& x, f256_s& out) noexcept
    {
        const f256_s y1 = x - f256_s{ 1.0 };
        if (abs(y1) <= f256_s{ 0.25 })
        {
            out = lgamma1p_series(y1);
            return true;
        }

        const f256_s y2 = x - f256_s{ 2.0 };
        if (abs(y2) <= f256_s{ 0.25 })
        {
            out = f256_log1p_series_reduced(y2) + lgamma1p_series(y2);
            return true;
        }

        return false;
    }

    BL_FORCE_INLINE constexpr bool f256_remainder_pi2(const f256_s& x, long long& n_out, f256_s& r_out)
    {
        if (!_f256_detail::isfinite(x.x0))
            return false;

        if (abs(x) <= _f256_const::pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256_s q = nearbyint(x * _f256_const::invpi2);
        const double qd = q.x0;

        if (!fltx::common::fp::isfinite(qd) || fltx::common::fp::absd(qd) > 9.0e15)
        {
            const double xd = static_cast<double>(x);
            const double fallback_qd = (double)fltx::common::fp::llround_constexpr(xd * static_cast<double>(_f256_const::invpi2));

            if (!fltx::common::fp::isfinite(fallback_qd) || fltx::common::fp::absd(fallback_qd) > 9.0e15)
                return false;

            const long long n = (long long)fallback_qd;
            const f256_s qf{ (double)n };

            f256_s r = x;
            r -= qf * _f256_const::pi_2.x0;
            r -= qf * _f256_const::pi_2.x1;
            r -= qf * _f256_const::pi_2.x2;
            r -= qf * _f256_const::pi_2.x3;

            if (r > _f256_const::pi_4)
            {
                r -= _f256_const::pi_2;
                n_out = n + 1;
            }
            else if (r < -_f256_const::pi_4)
            {
                r += _f256_const::pi_2;
                n_out = n - 1;
            }
            else
            {
                n_out = n;
            }

            r_out = r;
            return true;
        }

        long long n = (long long)qd;
        f256_s r = x;
        r -= q * _f256_const::pi_2.x0;
        r -= q * _f256_const::pi_2.x1;
        r -= q * _f256_const::pi_2.x2;
        r -= q * _f256_const::pi_2.x3;

        if (r > _f256_const::pi_4)
        {
            r -= _f256_const::pi_2;
            ++n;
        }
        else if (r < -_f256_const::pi_4)
        {
            r += _f256_const::pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }
    inline constexpr f256_s f256_sin_coeffs_pi4[] = {
        {  0x1.5a42f0dfeb086p-209, -0x1.35ae015f78f6ep-264, -0x1.c71a521ce2e79p-318,  0x1.6a300230ce998p-372 },
        { -0x1.8da8e0a127ebap-198,  0x1.21d2eac9d275cp-252,  0x1.ad541d26964afp-306,  0x1.1c066ebdf95dep-360 },
        {  0x1.a3cb872220648p-187, -0x1.c7f4e85b8e6cdp-241, -0x1.413a0bc5fc28ap-295, -0x1.16ae534063fabp-352 },
        { -0x1.95db45257e512p-176, -0x1.6e5d72b6f79b9p-231, -0x1.b830cf0b5b5c6p-291, -0x1.29276833f5728p-345 },
        {  0x1.65e61c39d0241p-165, -0x1.c0ed181727269p-220, -0x1.abbd2f56bbc2fp-276, -0x1.18ff57fdc2e4ep-330 },
        { -0x1.1e99449a4bacep-154,  0x1.fefbb89514b3cp-210,  0x1.53433f743a2d9p-264, -0x1.25f70d1395dd7p-320 },
        {  0x1.9ec8d1c94e85bp-144, -0x1.670e9d4784ec6p-201,  0x1.79fe5954939a2p-255,  0x1.82e418d9b0c9ep-311 },
        { -0x1.0dc59c716d91fp-133, -0x1.419e3fad3f031p-188, -0x1.d9d7ed1981ffcp-244,  0x1.345ea5d66a84bp-300 },
        {  0x1.3981254dd0d52p-123, -0x1.2b1f4c8015a2fp-177, -0x1.d82af23edb6dbp-231,  0x1.a1cd20123a99bp-285 },
        { -0x1.434d2e783f5bcp-113, -0x1.0b87b91be9affp-167, -0x1.c89db1796db75p-224,  0x1.8923b7699c8bep-278 },
        {  0x1.259f98b4358adp-103,  0x1.eaf8c39dd9bc5p-157, -0x1.6e29990a26fb6p-211, -0x1.2d867809b5568p-267 },
        { -0x1.d1ab1c2dccea3p-94,  -0x1.054d0c78aea14p-149,  0x1.196bf16c33a56p-203, -0x1.f0e65ed04d346p-257 },
        {  0x1.3f3ccdd165fa9p-84,  -0x1.58ddadf344487p-139, -0x1.e8ed8001ad67ep-193,  0x1.80a5edffcced7p-247 },
        { -0x1.761b41316381ap-75,   0x1.3423c7d91404fp-130, -0x1.e6135bfc1194ap-185,  0x1.ba7b1a3077b39p-239 },
        {  0x1.71b8ef6dcf572p-66,  -0x1.d043ae40c4647p-120,  0x1.486121e81d5fep-176, -0x1.2d4ba8e1e64c7p-230 },
        { -0x1.2f49b46814157p-57,  -0x1.2650f61dbdcb4p-112,  0x1.69502917cbf3bp-166, -0x1.e35fbddac4553p-223 },
        {  0x1.952c77030ad4ap-49,   0x1.ac981465ddc6cp-103, -0x1.588b72e53bc5fp-165,  0x1.7079e8909271ap-221 },
        { -0x1.ae7f3e733b81fp-41,  -0x1.1d8656b0ee8cbp-97,   0x1.6e142a138f825p-157, -0x1.43c0c38ccdcc6p-212 },
        {  0x1.6124613a86d09p-33,   0x1.f28e0cc748ebep-87,  -0x1.7b2c4c8a840bcp-141,  0x1.c71cca1034c07p-195 },
        { -0x1.ae64567f544e4p-26,   0x1.c062e06d1f209p-80,  -0x1.c7880adcbc46ep-136,  0x1.5553a6f0fed60p-190 },
        {  0x1.71de3a556c734p-19,  -0x1.c154f8ddc6c00p-73,   0x1.71de3a556c734p-127, -0x1.c154f8ddc6c00p-181 },
        { -0x1.a01a01a01a01ap-13,  -0x1.a01a01a01a01ap-73,  -0x1.a01a01a01a01ap-133, -0x1.a01a01a01a01ap-193 },
        {  0x1.1111111111111p-7,    0x1.1111111111111p-63,   0x1.1111111111111p-119,  0x1.1111111111111p-175 },
        { -0x1.5555555555555p-3,   -0x1.5555555555555p-57,  -0x1.5555555555555p-111, -0x1.5555555555555p-165 }
    };
    inline constexpr f256_s f256_cos_coeffs_pi4[] = {
        {  0x1.091b406b6ff26p-203,  0x1.e973637973b18p-257, -0x1.1e38136f0edcap-311, -0x1.7ab33e52a1d28p-366 },
        { -0x1.240804f659510p-192, -0x1.8b291b93c9718p-246, -0x1.096c752f5341fp-301,  0x1.c12972a70641ep-355 },
        {  0x1.272b1b03fec6ap-181,  0x1.3f67cc9f9fdb8p-235, -0x1.71dcd047354c9p-289, -0x1.c3f29289464c4p-346 },
        { -0x1.10af527530de8p-170, -0x1.b626c912ee5c8p-225, -0x1.349f032c6e859p-279,  0x1.ec616617f45c6p-333 },
        {  0x1.ca8ed42a12ae3p-160,  0x1.a07244abad2abp-224,  0x1.facdac6fb71b7p-278, -0x1.ca2f486d514e1p-339 },
        { -0x1.5d4acb9c0c3abp-149,  0x1.6ec2c8f5b13b2p-205, -0x1.e2860aaa59188p-259,  0x1.866eba0408569p-313 },
        {  0x1.df983290c2ca9p-139,  0x1.5835c6895393bp-194, -0x1.0578f45b1aaaep-249, -0x1.281508688972dp-303 },
        { -0x1.2710231c0fd7ap-128, -0x1.3f8a2b4af9d6bp-184, -0x1.c32215a9f317ep-238,  0x1.d451e158a1205p-293 },
        {  0x1.434d2e783f5bcp-118,  0x1.0b87b91be9affp-172,  0x1.c89db1796db75p-229, -0x1.8923b7699c8bep-283 },
        { -0x1.3932c5047d60ep-108, -0x1.832b7b530a627p-162, -0x1.5d2c61f6d124cp-218, -0x1.f192b328d82c4p-272 },
        {  0x1.0a18a2635085dp-98,   0x1.b9e2e28e1aa54p-153,  0x1.a8549a9d99586p-207, -0x1.141dcc8cc5668p-266 },
        { -0x1.88e85fc6a4e5ap-89,   0x1.71c37ebd16540p-143, -0x1.494676265a364p-197,  0x1.397b40007db79p-253 },
        {  0x1.f2cf01972f578p-80,  -0x1.9ada5fcc1ab14p-135,  0x1.440ce7fd610dcp-189, -0x1.26fcbc204fcd1p-243 },
        { -0x1.0ce396db7f853p-70,   0x1.aebcdbd20331cp-124,  0x1.38a88578b4d75p-178, -0x1.c0fbc29694fb8p-233 },
        {  0x1.e542ba4020225p-62,   0x1.ea72b4afe3c2fp-120, -0x1.44020dfd65c8cp-174, -0x1.6e69b50fc88abp-231 },
        { -0x1.6827863b97d97p-53,  -0x1.eec01221a8b0bp-107,  0x1.568798662118bp-161, -0x1.f00d8b9e49291p-222 },
        {  0x1.ae7f3e733b81fp-45,   0x1.1d8656b0ee8cbp-101, -0x1.6e142a138f825p-161,  0x1.43c0c38ccdcc6p-216 },
        { -0x1.93974a8c07c9dp-37,  -0x1.05d6f8a2efd1fp-92,  -0x1.3aa3346236a5dp-147, -0x1.d75f096ea801ep-201 },
        {  0x1.1eed8eff8d898p-29,  -0x1.2aec959e14c06p-83,   0x1.2fb0073dd2d9ep-139,  0x1.c71d90b4ab715p-193 },
        { -0x1.27e4fb7789f5cp-22,  -0x1.cbbc05b4fa99ap-76,   0x1.c6d278883e8f5p-132, -0x1.95567d3a50ccep-186 },
        {  0x1.a01a01a01a01ap-16,   0x1.a01a01a01a01ap-76,   0x1.a01a01a01a01ap-136,  0x1.a01a01a01a01ap-196 },
        { -0x1.6c16c16c16c17p-10,   0x1.f49f49f49f49fp-65,   0x1.27d27d27d27d2p-119,  0x1.f49f49f49f49fp-173 },
        {  0x1.5555555555555p-5,    0x1.5555555555555p-59,   0x1.5555555555555p-113,  0x1.5555555555555p-167 },
        { -0x1.0000000000000p-1,    0x0.0p+0,                0x0.0p+0,                0x0.0p+0 }
    };
    inline constexpr std::size_t f256_trig_coeff_count_pi4 = sizeof(f256_sin_coeffs_pi4) / sizeof(f256_sin_coeffs_pi4[0]);

    #if BL_F256_ENABLE_SIMD
    BL_FORCE_INLINE __m128d f256_trig_simd_set(double lane0, double lane1) noexcept
    {
        return _mm_set_pd(lane1, lane0);
    }
    BL_FORCE_INLINE __m128d f256_trig_simd_splat(double value) noexcept
    {
        return _mm_set1_pd(value);
    }
    BL_FORCE_INLINE void f256_trig_simd_store(__m128d value, double& lane0, double& lane1) noexcept
    {
        alignas(16) double lanes[2];
        _mm_storeu_pd(lanes, value);
        lane0 = lanes[0];
        lane1 = lanes[1];
    }
    BL_FORCE_INLINE void f256_trig_simd_two_sum(__m128d a, __m128d b, __m128d& s, __m128d& e) noexcept
    {
        s = _mm_add_pd(a, b);
        const __m128d bb = _mm_sub_pd(s, a);
        e = _mm_add_pd(_mm_sub_pd(a, _mm_sub_pd(s, bb)), _mm_sub_pd(b, bb));
    }
    BL_FORCE_INLINE void f256_trig_simd_quick_two_sum(__m128d a, __m128d b, __m128d& s, __m128d& e) noexcept
    {
        s = _mm_add_pd(a, b);
        e = _mm_sub_pd(b, _mm_sub_pd(s, a));
    }
    BL_FORCE_INLINE void f256_trig_simd_two_prod(__m128d a, __m128d b, __m128d& p, __m128d& e) noexcept
    {
        p = _mm_mul_pd(a, b);

        const __m128d split = _mm_set1_pd(134217729.0);
        const __m128d a_scaled = _mm_mul_pd(a, split);
        const __m128d b_scaled = _mm_mul_pd(b, split);

        const __m128d a_hi = _mm_sub_pd(a_scaled, _mm_sub_pd(a_scaled, a));
        const __m128d b_hi = _mm_sub_pd(b_scaled, _mm_sub_pd(b_scaled, b));
        const __m128d a_lo = _mm_sub_pd(a, a_hi);
        const __m128d b_lo = _mm_sub_pd(b, b_hi);

        e = _mm_add_pd(
            _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a_hi, b_hi), p), _mm_mul_pd(a_hi, b_lo)),
            _mm_add_pd(_mm_mul_pd(a_lo, b_hi), _mm_mul_pd(a_lo, b_lo))
        );
    }
    BL_FORCE_INLINE void f256_trig_simd_three_sum(__m128d& a, __m128d& b, __m128d& c) noexcept
    {
        __m128d t1{}, t2{}, t3{};
        f256_trig_simd_two_sum(a, b, t1, t2);
        f256_trig_simd_two_sum(c, t1, a, t3);
        f256_trig_simd_two_sum(t2, t3, b, c);
    }
    BL_FORCE_INLINE void f256_trig_simd_three_sum2(__m128d& a, __m128d& b, __m128d& c) noexcept
    {
        __m128d t1{}, t2{}, t3{};
        f256_trig_simd_two_sum(a, b, t1, t2);
        f256_trig_simd_two_sum(c, t1, a, t3);
        b = _mm_add_pd(t2, t3);
    }
    BL_FORCE_INLINE constexpr f256_s f256_mul_from_two_prod_terms(
        double p0, double p1, double p2, double p3, double p4, double p5,
        double p6, double p7, double p8, double p9,
        double q0, double q1, double q2, double q3, double q4, double q5,
        double q6, double q7, double q8, double q9,
        double tail_mul0, double tail_mul1, double tail_mul2) noexcept
    {
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        _f256_detail::three_sum(p1, p2, q0);
        _f256_detail::three_sum(p2, q1, q2);
        _f256_detail::three_sum(p3, p4, p5);

        _f256_detail::two_sum_precise(p2, p3, s0, t0);
        _f256_detail::two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        _f256_detail::two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        _f256_detail::two_sum_precise(q0, q3, q0, q3);
        _f256_detail::two_sum_precise(q4, q5, q4, q5);
        _f256_detail::two_sum_precise(p6, p7, p6, p7);
        _f256_detail::two_sum_precise(p8, p9, p8, p9);

        _f256_detail::two_sum_precise(q0, q4, t0, t1);
        t1 += (q3 + q5);

        _f256_detail::two_sum_precise(p6, p8, r0, r1);
        r1 += (p7 + p9);

        _f256_detail::two_sum_precise(t0, r0, q3, q4);
        q4 += (t1 + r1);

        _f256_detail::two_sum_precise(q3, s1, t0, t1);
        t1 += q4;

        t1 += tail_mul0 + tail_mul1 + tail_mul2
            + q6 + q7 + q8 + q9 + s2;

        return _f256_detail::renorm5(p0, p1, s0, t0, t1);
    }

    BL_FORCE_INLINE void f256_mul_pair_simd(
        const f256_s& a0, const f256_s& b0,
        const f256_s& a1, const f256_s& b1,
        f256_s& out0, f256_s& out1) noexcept
    {
        double p00{}, p10{}, p20{}, p30{}, p40{}, p50{};
        double q00{}, q10{}, q20{}, q30{}, q40{}, q50{};

        double p01{}, p11{}, p21{}, p31{}, p41{}, p51{};
        double q01{}, q11{}, q21{}, q31{}, q41{}, q51{};

        _f256_detail::two_prod_precise(a0.x0, b0.x0, p00, q00);
        _f256_detail::two_prod_precise(a0.x0, b0.x1, p10, q10);
        _f256_detail::two_prod_precise(a0.x1, b0.x0, p20, q20);
        _f256_detail::two_prod_precise(a0.x0, b0.x2, p30, q30);
        _f256_detail::two_prod_precise(a0.x1, b0.x1, p40, q40);
        _f256_detail::two_prod_precise(a0.x2, b0.x0, p50, q50);

        _f256_detail::two_prod_precise(a1.x0, b1.x0, p01, q01);
        _f256_detail::two_prod_precise(a1.x0, b1.x1, p11, q11);
        _f256_detail::two_prod_precise(a1.x1, b1.x0, p21, q21);
        _f256_detail::two_prod_precise(a1.x0, b1.x2, p31, q31);
        _f256_detail::two_prod_precise(a1.x1, b1.x1, p41, q41);
        _f256_detail::two_prod_precise(a1.x2, b1.x0, p51, q51);

        const __m128d ax0 = f256_trig_simd_set(a0.x0, a1.x0);
        const __m128d ax1 = f256_trig_simd_set(a0.x1, a1.x1);
        const __m128d ax2 = f256_trig_simd_set(a0.x2, a1.x2);
        const __m128d ax3 = f256_trig_simd_set(a0.x3, a1.x3);

        const __m128d bx0 = f256_trig_simd_set(b0.x0, b1.x0);
        const __m128d bx1 = f256_trig_simd_set(b0.x1, b1.x1);
        const __m128d bx2 = f256_trig_simd_set(b0.x2, b1.x2);
        const __m128d bx3 = f256_trig_simd_set(b0.x3, b1.x3);

        __m128d p6{}, p7{}, p8{}, p9{};
        __m128d q6{}, q7{}, q8{}, q9{};

        f256_trig_simd_two_prod(ax0, bx3, p6, q6);
        f256_trig_simd_two_prod(ax1, bx2, p7, q7);
        f256_trig_simd_two_prod(ax2, bx1, p8, q8);
        f256_trig_simd_two_prod(ax3, bx0, p9, q9);

        alignas(16) double p6v[2], p7v[2], p8v[2], p9v[2];
        alignas(16) double q6v[2], q7v[2], q8v[2], q9v[2];

        _mm_storeu_pd(p6v, p6);
        _mm_storeu_pd(p7v, p7);
        _mm_storeu_pd(p8v, p8);
        _mm_storeu_pd(p9v, p9);
        _mm_storeu_pd(q6v, q6);
        _mm_storeu_pd(q7v, q7);
        _mm_storeu_pd(q8v, q8);
        _mm_storeu_pd(q9v, q9);

        out0 = _f256_detail::f256_mul_from_two_prod_terms(
            p00, p10, p20, p30, p40, p50,
            p6v[0], p7v[0], p8v[0], p9v[0],
            q00, q10, q20, q30, q40, q50,
            q6v[0], q7v[0], q8v[0], q9v[0],
            a0.x1 * b0.x3, a0.x2 * b0.x2, a0.x3 * b0.x1
        );

        out1 = _f256_detail::f256_mul_from_two_prod_terms(
            p01, p11, p21, p31, p41, p51,
            p6v[1], p7v[1], p8v[1], p9v[1],
            q01, q11, q21, q31, q41, q51,
            q6v[1], q7v[1], q8v[1], q9v[1],
            a1.x1 * b1.x3, a1.x2 * b1.x2, a1.x3 * b1.x1
        );
    }
    #endif

    BL_FORCE_INLINE constexpr f256_s f256_sin_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s ps = f256_sin_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            ps = ps * t + f256_sin_coeffs_pi4[i];
        return r + r * t * ps;
    }
    BL_FORCE_INLINE constexpr f256_s f256_cos_kernel_pi4(const f256_s& r)
    {
        const f256_s t = r * r;

        f256_s pc = f256_cos_coeffs_pi4[0];
        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            pc = pc * t + f256_cos_coeffs_pi4[i];
        return f256_s{ 1.0 } + t * pc;
    }
    BL_FORCE_INLINE constexpr void f256_sincos_kernel_pi4(const f256_s& r, f256_s& s_out, f256_s& c_out)
    {
        const f256_s t = r * r;

        f256_s ps = f256_sin_coeffs_pi4[0];
        f256_s pc = f256_cos_coeffs_pi4[0];

        #if BL_F256_ENABLE_SIMD
        if (_f256_detail::f256_runtime_trig_simd_enabled())
        {
            for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
            {
                f256_s next_ps{}, next_pc{};
                f256_mul_pair_simd(ps, t, pc, t, next_ps, next_pc);
                ps = next_ps + f256_sin_coeffs_pi4[i];
                pc = next_pc + f256_cos_coeffs_pi4[i];
            }

            const f256_s rt = r * t;
            f256_s sin_tail{}, cos_tail{};
            f256_mul_pair_simd(ps, rt, pc, t, sin_tail, cos_tail);
            s_out = r + sin_tail;
            c_out = f256_s{ 1.0 } + cos_tail;
            return;
        }
        #endif

        for (std::size_t i = 1; i < f256_trig_coeff_count_pi4; ++i)
        {
            ps = ps * t + f256_sin_coeffs_pi4[i];
            pc = pc * t + f256_cos_coeffs_pi4[i];
        }

        const f256_s rt = r * t;
        s_out = r + rt * ps;
        c_out = f256_s{ 1.0 } + t * pc;
    }

    BL_FORCE_INLINE constexpr f256_s canonicalize_exp_result(f256_s value) noexcept
    {
        value.x3 = fltx::common::fp::zero_low_fraction_bits_finite<8>(value.x3);
        return value;
    }

    BL_FORCE_INLINE constexpr f256_s _ldexp(const f256_s& a, int e)
    {
        double s;
        if (bl::is_constant_evaluated())
        {
            s = bl::fltx::common::fp::ldexp_constexpr2(1.0, e);
        }
        else
        {
            s = std::ldexp(1.0, e);
        }

        if (bl::is_constant_evaluated())
        {
            return canonicalize_exp_result(_f256_detail::renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s));
        }
        else
        {
            #if BL_F256_ENABLE_SIMD
            if (_f256_detail::f256_runtime_simd_enabled())
            {
                const __m128d scale = _f256_detail::f256_simd_splat(s);
                __m128d lo = _mm_mul_pd(_f256_detail::f256_simd_set(a.x0, a.x1), scale);
                __m128d hi = _mm_mul_pd(_f256_detail::f256_simd_set(a.x2, a.x3), scale);
                double x0{}, x1{}, x2{}, x3{};
                _f256_detail::f256_simd_store(lo, x0, x1);
                _f256_detail::f256_simd_store(hi, x2, x3);
                return canonicalize_exp_result(_f256_detail::renorm(x0, x1, x2, x3));
            }
            else
            #endif
            {
                return canonicalize_exp_result(_f256_detail::renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s));
            }
        }
    }
    BL_FORCE_INLINE constexpr f256_s _exp(const f256_s& x)
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

        const f256_s t = x * _f256_const::inv_ln2;

        double kd = _f256_detail::nearbyint_ties_even(t.x0);
        const f256_s delta = t - f256_s{ kd };
        if (delta.x0 > 0.5 || (delta.x0 == 0.5 && (delta.x1 > 0.0 || (delta.x1 == 0.0 && (delta.x2 > 0.0 || (delta.x2 == 0.0 && delta.x3 > 0.0))))))
            kd += 1.0;
        else if (delta.x0 < -0.5 || (delta.x0 == -0.5 && (delta.x1 < 0.0 || (delta.x1 == 0.0 && (delta.x2 < 0.0 || (delta.x2 == 0.0 && delta.x3 < 0.0))))))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f256_s r = (x - f256_s{ kd } * _f256_const::ln2) * f256_s{ 0.0009765625 };

        f256_s e = _f256_detail::f256_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = e * (e + 2.0);

        return _ldexp(e + 1.0, k);
    }
    BL_FORCE_INLINE constexpr f256_s _log(const f256_s& a)
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
        if (bl::is_constant_evaluated()) {
            exp2 = _f256_detail::frexp_exponent_constexpr(a.x0);
        }
        else {
            (void)std::frexp(a.x0, &exp2);
        }

        f256_s m = _ldexp(a, -exp2);
        if (m < _f256_const::sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }

        f256_s y = f256_s{ (double)exp2 } * _f256_const::ln2 + f256_s{ log_as_double(m), 0.0, 0.0, 0.0 };
        y += m * _exp(-y + f256_s{ (double)exp2 } * _f256_const::ln2) - 1.0;
        y += m * _exp(-y + f256_s{ (double)exp2 } * _f256_const::ln2) - 1.0;
        return y;
    }
}

// exp
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s ldexp(const f256_s& a, int e)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_ldexp(a, e));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s exp(const f256_s& x)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_exp(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s exp2(const f256_s& x)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_exp(x * _f256_const::ln2));
}

// log
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s log(const f256_s& a)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_log(a));
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s log2(const f256_s& a)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_log(a) * _f256_const::inv_ln2);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s log10(const f256_s& a)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_log(a) / _f256_const::ln10);
}

// pow
[[nodiscard]] BL_NO_INLINE constexpr f256_s pow(const f256_s& x, const f256_s& y)
{
    if (iszero(y))
        return f256_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s yi = trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && _f256_detail::f256_try_get_int64(yi, yi64))
        return _f256_detail::powi(x, yi64);

    if (x.x0 < 0.0 || (x.x0 == 0.0 && _f256_detail::signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = _f256_detail::_exp(y * _f256_detail::_log(-x));
        const f256_s parity = fmod(abs(yi), f256_s{ 2.0 });
        return (parity == f256_s{ 1.0 }) ? -magnitude : magnitude;
    }

    return _f256_detail::canonicalize_math_result(_f256_detail::_exp(y * _f256_detail::_log(x)));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s pow(const f256_s& x, double y)
{
    if (y == 0.0)
        return f256_s{ 1.0 };

    if (isnan(x) || _f256_detail::isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (y == 1.0) return x;
    if (y == 2.0) return _f256_detail::canonicalize_math_result(x * x);
    if (y == -1.0) return _f256_detail::canonicalize_math_result(f256_s{ 1.0 } / x);
    if (y == 0.5) return _f256_detail::canonicalize_math_result(sqrt(x));

    double yi{};
    if (bl::is_constant_evaluated())
    {
        yi = (y < 0.0)
            ? _f256_detail::ceil_constexpr(y)
            : _f256_detail::floor_constexpr(y);
    }
    else
    {
        yi = std::trunc(y);
    }

    const bool y_is_int = (yi == y);

    if (y_is_int && _f256_detail::absd(yi) < 0x1p63)
        return _f256_detail::powi(x, static_cast<int64_t>(yi));

    if (x.x0 < 0.0 || (x.x0 == 0.0 && _f256_detail::signbit_constexpr(x.x0)))
    {
        if (!y_is_int)
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s magnitude = exp(f256_s{ y } * log(-x));
        const bool y_is_odd =
            (_f256_detail::absd(yi) < 0x1p53) &&
            ((static_cast<int64_t>(yi) & 1ll) != 0);

        return _f256_detail::canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
    }

    return _f256_detail::canonicalize_math_result(exp(f256_s{ y } * log(x)));
}


// trig
namespace _f256_detail
{
    BL_FORCE_INLINE constexpr bool _sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
    {
        const double ax = _f256_detail::fabs_constexpr(x.x0);
        if (!_f256_detail::isfinite(ax))
        {
            s_out = f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
            c_out = s_out;
            return false;
        }

        if (ax <= static_cast<double>(_f256_const::pi_4))
        {
            _f256_detail::f256_sincos_kernel_pi4(x, s_out, c_out);
            return true;
        }

        long long n = 0;
        f256_s r{};
        if (!_f256_detail::f256_remainder_pi2(x, n, r))
            return false;

        f256_s sr{}, cr{};
        _f256_detail::f256_sincos_kernel_pi4(r, sr, cr);

        switch ((int)(n & 3LL))
        {
        case 0: s_out = sr;  c_out = cr;  break;
        case 1: s_out = cr;  c_out = -sr; break;
        case 2: s_out = -sr; c_out = -cr; break;
        default: s_out = -cr; c_out = sr;  break;
        }

        return true;
    }

    BL_NO_INLINE constexpr f256_s atan_core_unit(const f256_s& z)
    {
        f256_s v = f256_s{ fltx::common::fp::atan_constexpr(static_cast<double>(z)) };

        for (int i = 0; i < 2; ++i)
        {
            f256_s sv{}, cv{};
            if (!_sincos(v, sv, cv))
            {
                const double vd = static_cast<double>(v);
                double sd{}, cd{};
                fltx::common::fp::sincos_constexpr(vd, sd, cd);
                sv = f256_s{ sd };
                cv = f256_s{ cd };
            }

            #if BL_F256_ENABLE_SIMD
            if (_f256_detail::f256_runtime_trig_simd_enabled())
            {
                f256_s zcv{}, zsv{};
                _f256_detail::f256_mul_pair_simd(z, cv, z, sv, zcv, zsv);
                const f256_s f = sv - zcv;
                const f256_s fp = cv + zsv;
                v = v - f / fp;
                continue;
            }
            #endif

            const f256_s f = sv - z * cv;
            const f256_s fp = cv + z * sv;
            v = v - f / fp;
        }

        return v;
    }
    BL_NO_INLINE constexpr f256_s _atan(const f256_s& x)
    {
        if (isnan(x))  return x;
        if (iszero(x)) return x;
        if (isinf(x))  return _f256_detail::signbit_constexpr(x.x0) ? -_f256_const::pi_2 : _f256_const::pi_2;

        const bool neg = x.x0 < 0.0;
        const f256_s ax = neg ? -x : x;

        if (ax > f256_s{ 1.0 })
        {
            const f256_s core = _f256_detail::atan_core_unit(recip(ax));
            const f256_s out = _f256_const::pi_2 - core;
            return neg ? -out : out;
        }

        const f256_s out = _f256_detail::atan_core_unit(ax);
        return neg ? -out : out;
    }
    BL_FORCE_INLINE constexpr f256_s _asin(const f256_s& x)
    {
        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (ax == f256_s{ 1.0 })
            return (x.x0 < 0.0) ? -_f256_const::pi_2 : _f256_const::pi_2;

        if (ax <= f256_s{ 0.5 })
            return _atan(x / sqrt(f256_s{ 1.0 } - x * x));

        const f256_s t = sqrt((f256_s{ 1.0 } - ax) / (f256_s{ 1.0 } + ax));
        const f256_s a = _f256_const::pi_2 - (_atan(t) + _atan(t));
        return (x.x0 < 0.0) ? -a : a;
    }
    BL_FORCE_INLINE constexpr f256_s _acos(const f256_s& x)
    {
        if (isnan(x))
            return x;

        const f256_s ax = abs(x);
        if (ax > f256_s{ 1.0 })
            return std::numeric_limits<f256_s>::quiet_NaN();
        if (x == f256_s{ 1.0 })
            return f256_s{ 0.0 };
        if (x == f256_s{ -1.0 })
            return _f256_const::pi;

        return _f256_const::pi_2 - _asin(x);
    }
}

[[nodiscard]] BL_NO_INLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    bool ret = _f256_detail::_sincos(x, s_out, c_out);
    s_out = _f256_detail::canonicalize_math_result(s_out);
    c_out = _f256_detail::canonicalize_math_result(c_out);
    return ret;
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s sin(const f256_s& x)
{
    const double ax = _f256_detail::fabs_constexpr(x.x0);
    if (!_f256_detail::isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(_f256_const::pi_4))
        return _f256_detail::f256_sin_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!_f256_detail::f256_remainder_pi2(x, n, r))
    {
        if (bl::is_constant_evaluated())
        {
            return f256_s{ fltx::common::fp::sin_constexpr(static_cast<double>(x)) };
        }
        else
        {
            return f256_s{ std::sin(static_cast<double>(x)) };
        }
    }
    switch ((int)(n & 3LL))
    {
    case 0: return _f256_detail::f256_sin_kernel_pi4(r);
    case 1: return _f256_detail::f256_cos_kernel_pi4(r);
    case 2: return -_f256_detail::f256_sin_kernel_pi4(r);
    default: return -_f256_detail::f256_cos_kernel_pi4(r);
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s cos(const f256_s& x)
{
    const double ax = _f256_detail::fabs_constexpr(x.x0);
    if (!_f256_detail::isfinite(ax))
        return f256_s{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };

    if (ax <= static_cast<double>(_f256_const::pi_4))
        return _f256_detail::f256_cos_kernel_pi4(x);

    long long n = 0;
    f256_s r{};
    if (!_f256_detail::f256_remainder_pi2(x, n, r))
    {
        if (bl::is_constant_evaluated())
        {
            return f256_s{ fltx::common::fp::cos_constexpr(static_cast<double>(x)) };
        }
        else
        {
            return f256_s{ std::cos(static_cast<double>(x)) };
        }
    }

    switch ((int)(n & 3LL))
    {
    case 0: return _f256_detail::f256_cos_kernel_pi4(r);
    case 1: return -_f256_detail::f256_sin_kernel_pi4(r);
    case 2: return -_f256_detail::f256_cos_kernel_pi4(r);
    default: return _f256_detail::f256_sin_kernel_pi4(r);
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tan(const f256_s& x)
{
    f256_s s{}, c{};
    if (_f256_detail::_sincos(x, s, c))
        return _f256_detail::canonicalize_math_result(s / c);

    if (bl::is_constant_evaluated())
    {
        return _f256_detail::canonicalize_math_result(f256_s{ fltx::common::fp::tan_constexpr(static_cast<double>(x)) });
    }
    else
    {
        return _f256_detail::canonicalize_math_result(f256_s{ std::tan(static_cast<double>(x)) });
    }
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atan(const f256_s& x)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_atan(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atan2(const f256_s& y, const f256_s& x)
{
    if (isnan(x) || isnan(y))
        return std::numeric_limits<f256_s>::quiet_NaN();

    if (iszero(x))
    {
        if (iszero(y))
            return f256_s{ std::numeric_limits<double>::quiet_NaN() };
        return ispositive(y) ? _f256_const::pi_2 : -_f256_const::pi_2;
    }

    if (iszero(y))
    {
        if (x.x0 < 0.0)
            return _f256_detail::signbit_constexpr(y.x0) ? -_f256_const::pi : _f256_const::pi;
        return y;
    }

    const f256_s ax = abs(x);
    const f256_s ay = abs(y);

    if (ax == ay)
    {
        if (x.x0 < 0.0)
        {
            return _f256_detail::canonicalize_math_result(
                (y.x0 < 0.0) ? -_f256_const::pi_3_4 : _f256_const::pi_3_4);
        }

        return _f256_detail::canonicalize_math_result(
            (y.x0 < 0.0) ? -_f256_const::pi_4 : _f256_const::pi_4);
    }

    if (ax >= ay)
    {
        f256_s a = _f256_detail::_atan(y / x);

        if (x.x0 < 0.0)
            a += (y.x0 < 0.0) ? -_f256_const::pi : _f256_const::pi;
        return _f256_detail::canonicalize_math_result(a);
    }

    f256_s a = _f256_detail::_atan(x / y);
    return _f256_detail::canonicalize_math_result((y.x0 < 0.0) ? (-_f256_const::pi_2 - a) : (_f256_const::pi_2 - a));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s asin(const f256_s& x)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_asin(x));
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s acos(const f256_s& x)
{
    return _f256_detail::canonicalize_math_result(_f256_detail::_acos(x));
}


[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fabs(const f256_s& a) noexcept
{
    return abs(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(const f256_s& x) noexcept
{
    return _f256_detail::signbit_constexpr(x.x0)
        || (x.x0 == 0.0 && (_f256_detail::signbit_constexpr(x.x1)
        || (x.x1 == 0.0 && (_f256_detail::signbit_constexpr(x.x2)
        || (x.x2 == 0.0 && _f256_detail::signbit_constexpr(x.x3))))));
}
[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<f256_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(const f256_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(const f256_s& a, const f256_s& b) noexcept
{
    return isnan(a) || isnan(b);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(const f256_s& a, const f256_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

namespace _f256_detail
{
    BL_FORCE_INLINE constexpr f256_s round_half_away_zero(const f256_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f256_s y = -floor((-x) + f256_s{ 0.5 });
            if (iszero(y))
                return f256_s{ -0.0, 0.0, 0.0, 0.0 };
            return y;
        }

        return floor(x + f256_s{ 0.5 });
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
            : std::numeric_limits<double>::denorm_min();

        std::uint64_t bits = std::bit_cast<std::uint64_t>(from);
        if ((from > 0.0) == (from < to))
            ++bits;
        else
            --bits;

        return std::bit_cast<double>(bits);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f256_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        if (isnan(x) || isinf(x))
            return 0;

        const f256_s lo = to_f256(static_cast<int64_t>(std::numeric_limits<SignedInt>::lowest()));
        const f256_s hi = to_f256(static_cast<int64_t>(std::numeric_limits<SignedInt>::max()));
        if (x < lo || x > hi)
            return 0;

        int64_t out = 0;
        if (!_f256_detail::f256_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }

    BL_FORCE_INLINE constexpr f256_s nearest_integer_ties_even(const f256_s& q) noexcept
    {
        f256_s n = trunc(q);
        const f256_s frac = q - n;
        const f256_s half{ 0.5 };
        const f256_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n += signbit(frac) ? -one : one;
        }
        else if (abs(frac) == half)
        {
            if (fmod(n, f256_s{ 2.0 }) != f256_s{ 0.0 })
                n += signbit(frac) ? -one : one;
        }

        return n;
    }

    BL_FORCE_INLINE constexpr f256_s lgamma_stirling_asymptotic(const f256_s& z) noexcept
    {
        const f256_s inv = f256_s{ 1.0 } / z;
        const f256_s inv2 = inv * inv;

        f256_s series = inv / f256_s{ 12.0 };
        f256_s invpow = inv * inv2;

        series -= invpow / f256_s{ 360.0 };
        invpow *= inv2;
        series += invpow / f256_s{ 1260.0 };
        invpow *= inv2;
        series -= invpow / f256_s{ 1680.0 };
        invpow *= inv2;
        series += invpow / f256_s{ 1188.0 };
        invpow *= inv2;
        series -= invpow * (f256_s{ 691.0 } / f256_s{ 360360.0 });
        invpow *= inv2;
        series += invpow / f256_s{ 156.0 };
        invpow *= inv2;
        series -= invpow * (f256_s{ 3617.0 } / f256_s{ 122400.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 43867.0 } / f256_s{ 244188.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 174611.0 } / f256_s{ 125400.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 77683.0 } / f256_s{ 5796.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 236364091.0 } / f256_s{ 1506960.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 657931.0 } / f256_s{ 300.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 3392780147.0 } / f256_s{ 93960.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 1723168255201.0 } / f256_s{ 2492028.0 });
        invpow *= inv2;
        series -= invpow * (f256_s{ 7709321041217.0 } / f256_s{ 505920.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 151628697551.0 } / f256_s{ 3960.0 });
        invpow *= inv2;
		
        const f256_s b28_num = to_f256(std::uint64_t{ 2631527155305347737 }) * f256_s{ 10.0 } + f256_s{ 3.0 };
        series -= invpow * (b28_num / f256_s{ 5609403360.0 });
        invpow *= inv2;
        series += invpow * (f256_s{ 154210205991661.0 } / f256_s{ 444.0 });

        return (z - f256_s{ 0.5 }) * log(z) - z + _f256_const::half_log_two_pi + series;
    }

    BL_FORCE_INLINE constexpr void positive_recurrence_product(
        const f256_s& x,
        const f256_s& asymptotic_min,
        f256_s& z,
        f256_s& product,
        int& product_exp2) noexcept
    {
        z = x;
        product = f256_s{ 1.0 };
        product_exp2 = 0;

        while (z < asymptotic_min)
        {
            product *= z;

            const double hi = product.x0;
            if (hi != 0.0)
            {
                const int e = frexp_exponent_constexpr(hi);
                if (e > 512 || e < -512)
                {
                    product = ldexp(product, -e);
                    product_exp2 += e;
                }
            }

            z += f256_s{ 1.0 };
        }
    }

    BL_NO_INLINE constexpr f256_s lgamma_positive_recurrence(const f256_s& x) noexcept
    {
        f256_s near_value{};
        if (try_lgamma_near_one_or_two(x, near_value))
            return near_value;

        constexpr f256_s asymptotic_min = f256_s{ 128.0 };

        f256_s z{};
        f256_s product{};
        int product_exp2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_exp2);

        return lgamma_stirling_asymptotic(z)
            - log(product)
            - f256_s{ static_cast<double>(product_exp2) } * _f256_const::ln2;
    }

    BL_NO_INLINE constexpr f256_s gamma_positive_recurrence(const f256_s& x) noexcept
    {
        f256_s near_lgamma{};
        if (try_lgamma_near_one_or_two(x, near_lgamma))
            return exp(near_lgamma);

        constexpr f256_s asymptotic_min = f256_s{ 128.0 };

        f256_s z{};
        f256_s product{};
        int product_exp2 = 0;
        positive_recurrence_product(x, asymptotic_min, z, product, product_exp2);

        f256_s result = exp(lgamma_stirling_asymptotic(z)) / product;
        if (product_exp2 != 0)
            result = ldexp(result, -product_exp2);

        return result;
    }
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s expm1(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return x;
    if (isinf(x))
        return signbit(x)
        ? f256_s{ -1.0, 0.0, 0.0, 0.0 }
    : std::numeric_limits<f256_s>::infinity();

    const f256_s ax = abs(x);
    if (ax <= f256_s{ 0.5 })
    {
        f256_s term = x;
        f256_s sum = x;

        for (int n = 2; n <= 256; ++n)
        {
            term = (term * x) / f256_s{ static_cast<double>(n) };
            sum += term;

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }

        return _f256_detail::canonicalize_math_result(sum);
    }

    return _f256_detail::canonicalize_math_result(exp(x) - f256_s{ 1.0 });
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s log1p(const f256_s& x)
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

    const f256_s ax = abs(x);
    if (ax <= f256_s{ 0.5 })
        return _f256_detail::canonicalize_math_result(_f256_detail::f256_log1p_series_reduced(x));

    const f256_s u = f256_s{ 1.0 } + x;
    if ((u - f256_s{ 1.0 }) == x)
        return _f256_detail::canonicalize_math_result(log(u));

    if (x > f256_s{ 0.0 } && x <= f256_s{ 1.0 })
    {
        const f256_s t = x / (f256_s{ 1.0 } + sqrt(f256_s{ 1.0 } + x));
        return _f256_detail::canonicalize_math_result(_f256_detail::f256_log1p_series_reduced(t) * f256_s { 2.0 });
    }

    if (x > f256_s{ 0.0 })
        return _f256_detail::canonicalize_math_result(log(u));

    const f256_s y = u - f256_s{ 1.0 };
    if (iszero(y))
        return x;

    return _f256_detail::canonicalize_math_result(log(u) * (x / y));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s sinh(const f256_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (ax <= f256_s{ 0.5 })
    {
        const f256_s x2 = x * x;
        f256_s term = x;
        f256_s sum = x;

        for (int n = 1; n <= 256; ++n)
        {
            term = (term * x2) / f256_s{ static_cast<double>((2 * n) * (2 * n + 1)) };
            sum += term;

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }

        return _f256_detail::canonicalize_math_result(sum);
    }

    const f256_s ex = exp(ax);
    f256_s out = (ex - f256_s{ 1.0 } / ex) * f256_s{ 0.5 };
    if (signbit(x))
        out = -out;
    return _f256_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s cosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s ax = abs(x);
    const f256_s ex = exp(ax);
    return _f256_detail::canonicalize_math_result((ex + f256_s{ 1.0 } / ex) * f256_s { 0.5 });
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tanh(const f256_s& x)
{
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s ax = abs(x);
    if (ax > f256_s{ 20.0 })
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };

    const f256_s em1 = expm1(ax + ax);
    f256_s out = em1 / (em1 + f256_s{ 2.0 });
    if (signbit(x))
        out = -out;
    return _f256_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s asinh(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f256_s ax = abs(x);
    f256_s out{};
    if (ax > f256_s{ 0x1p500 })
        out = log(ax) + _f256_const::ln2;
    else
        out = log(ax + sqrt(ax * ax + f256_s{ 1.0 }));

    if (signbit(x))
        out = -out;
    return _f256_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s acosh(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x < f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (x == f256_s{ 1.0 })
        return f256_s{ 0.0 };
    if (isinf(x))
        return x;

    f256_s out{};
    if (x > f256_s{ 0x1p500 })
        out = log(x) + _f256_const::ln2;
    else
        out = log(x + sqrt((x - f256_s{ 1.0 }) * (x + f256_s{ 1.0 })));

    return _f256_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s atanh(const f256_s& x)
{
    if (isnan(x) || iszero(x))
        return x;

    const f256_s ax = abs(x);
    if (ax > f256_s{ 1.0 })
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (ax == f256_s{ 1.0 })
        return signbit(x)
        ? f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }
    : f256_s{ std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };

    if (ax <= f256_s{ 0.125 })
    {
        const f256_s x2 = x * x;
        f256_s sum = x;
        f256_s power = x;
        for (int k = 1; k <= 256; ++k)
        {
            power *= x2;
            const f256_s term = power / f256_s{ static_cast<double>(2 * k + 1) };
            sum += term;

            const f256_s asum = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }
        return _f256_detail::canonicalize_math_result(sum);
    }

    const f256_s out = log1p((x + x) / (f256_s{ 1.0 } - x)) * f256_s { 0.5 };
    return _f256_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s cbrt(const f256_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s y{};
    if (bl::is_constant_evaluated())
    {
        y = exp(log(ax) / f256_s{ 3.0 });
    }
    else
    {
        int exp2 = 0;
        double mantissa = std::frexp(ax.x0, &exp2);
        int rem = exp2 % 3;
        if (rem < 0)
            rem += 3;
        if (rem != 0)
        {
            mantissa = std::ldexp(mantissa, rem);
            exp2 -= rem;
        }

        y = f256_s{ std::cbrt(mantissa), 0.0, 0.0, 0.0 };
        if (exp2 != 0)
            y = ldexp(y, exp2 / 3);
    }

    y = (y + y + ax / (y * y)) / f256_s{ 3.0 };
    y = (y + y + ax / (y * y)) / f256_s{ 3.0 };

    if (bl::is_constant_evaluated())
        y = (y + y + ax / (y * y)) / f256_s{ 3.0 };

    if (neg)
        y = -y;

    return _f256_detail::canonicalize_math_result(y);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s hypot(const f256_s& x, const f256_s& y)
{
    if (isinf(x) || isinf(y))
        return std::numeric_limits<f256_s>::infinity();
    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    f256_s ax = abs(x);
    f256_s ay = abs(y);
    if (ax < ay)
        std::swap(ax, ay);

    if (iszero(ax))
        return f256_s{ 0.0 };
    if (iszero(ay))
        return _f256_detail::canonicalize_math_result(ax);

    int ex = 0;
    int ey = 0;
    if (bl::is_constant_evaluated())
    {
        ex = _f256_detail::frexp_exponent_constexpr(ax.x0);
        ey = _f256_detail::frexp_exponent_constexpr(ay.x0);
    }
    else
    {
        (void)std::frexp(ax.x0, &ex);
        (void)std::frexp(ay.x0, &ey);
    }

    if ((ex - ey) > 110)
        return _f256_detail::canonicalize_math_result(ax);

    const f256_s r = ay / ax;
    return _f256_detail::canonicalize_math_result(ax * sqrt(f256_s{ 1.0 } + r * r));
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s rint(const f256_s& x)
{
    return nearbyint(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(const f256_s& x)
{
    return _f256_detail::to_signed_integer_or_zero<long>(_f256_detail::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(const f256_s& x)
{
    return _f256_detail::to_signed_integer_or_zero<long long>(_f256_detail::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(const f256_s& x)
{
    return _f256_detail::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(const f256_s& x)
{
    return _f256_detail::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s remquo(const f256_s& x, const f256_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f256_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f256_s n = _f256_detail::nearest_integer_ties_even(x / y);
    f256_s r = x - n * y;

    if (quo)
    {
        const f256_s qbits = fmod(abs(n), f256_s{ 2147483648.0 });
        int bits = static_cast<int>(trunc(qbits).x0);
        if (signbit(n))
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        return f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };

    return _f256_detail::canonicalize_math_result(r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s remainder(const f256_s& x, const f256_s& y)
{
    int quotient_bits = 0;
    return remquo(x, y, &quotient_bits);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fma(const f256_s& x, const f256_s& y, const f256_s& z)
{
    return _f256_detail::canonicalize_math_result(x * y + z);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmin(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fmax(const f256_s& a, const f256_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s fdim(const f256_s& x, const f256_s& y)
{
    return (x > y) ? _f256_detail::canonicalize_math_result(x - y) : f256_s{ 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s copysign(const f256_s& x, const f256_s& y)
{
    return signbit(x) == signbit(y) ? x : -x;
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s frexp(const f256_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const double lead =
        (x.x0 != 0.0) ? x.x0 :
        (x.x1 != 0.0) ? x.x1 :
        (x.x2 != 0.0) ? x.x2 : x.x3;
    int e = 0;

    if (bl::is_constant_evaluated())
        e = fltx::common::fp::frexp_exponent_constexpr(lead);
    else
        (void)std::frexp(lead, &e);

    f256_s m = ldexp(x, -e);
    const f256_s am = abs(m);

    if (am < f256_s{ 0.5 })
    {
        m *= f256_s{ 2.0 };
        --e;
    }
    else if (am >= f256_s{ 1.0 })
    {
        m *= f256_s{ 0.5 };
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s modf(const f256_s& x, f256_s* iptr) noexcept
{
    const f256_s i = trunc(x);
    if (iptr)
        *iptr = i;

    f256_s frac = x - i;
    if (iszero(frac))
        frac = f256_s{ signbit(x) ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
    return frac;
}
[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(const f256_s& x) noexcept
{
    if (isnan(x))  return FP_ILOGBNAN;
    if (iszero(x)) return FP_ILOGB0;
    if (isinf(x))  return std::numeric_limits<int>::max();

    int e = 0;
    (void)frexp(abs(x), &e);
    return e - 1;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s logb(const f256_s& x) noexcept
{
    if (isnan(x))  return x;
    if (iszero(x)) return f256_s{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
    if (isinf(x))  return std::numeric_limits<f256_s>::infinity();

    return f256_s{ static_cast<double>(ilogb(x)), 0.0, 0.0, 0.0 };
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s scalbn(const f256_s& x, int e) noexcept
{
    return ldexp(x, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s scalbln(const f256_s& x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s nextafter(const f256_s& from, const f256_s& to) noexcept
{
    if (isnan(from) || isnan(to))
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

    return _f256_detail::renorm4(
        from.x0,
        from.x1,
        from.x2,
        _f256_detail::nextafter_double_constexpr(from.x3, toward)
    );
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, long double to) noexcept
{
    return nextafter(from, f256_s{ static_cast<double>(to), 0.0, 0.0, 0.0 });
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s erfc(const f256_s& x);
[[nodiscard]] BL_NO_INLINE constexpr f256_s erf(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f256_s{ -1.0 } : f256_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f256_s ax = neg ? -x : x;

    f256_s out{ 0.0 };

    if (ax < f256_s{ 3.0 })
    {
        const f256_s xx = ax * ax;
        f256_s power = ax;
        f256_s sum = ax;

        for (int n = 1; n < 1024; ++n)
        {
            power *= -xx / f256_s{ static_cast<double>(n) };
            const f256_s term = power / f256_s{ static_cast<double>(2 * n + 1) };
            sum += term;

            const f256_s abs_sum = abs(sum);
            const f256_s scale = (abs_sum > f256_s{ 1.0 }) ? abs_sum : f256_s{ 1.0 };
            if (abs(term) <= f256_s::eps() * scale)
                break;
        }

        out = f256_s{ 2.0 } * _f256_const::inv_sqrtpi * sum;
    }
    else
    {
        out = f256_s{ 1.0 } - erfc(ax);
    }

    if (neg)
        out = -out;

    return _f256_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s erfc(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (x == f256_s{ 0.0 })
        return f256_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f256_s{ 2.0 } : f256_s{ 0.0 };

    if (signbit(x))
        return _f256_detail::canonicalize_math_result(f256_s{ 1.0 } + erf(-x));

    if (x < f256_s{ 3.0 })
        return _f256_detail::canonicalize_math_result(f256_s{ 1.0 } - erf(x));

    if (x > f256_s{ 40.0 })
        return f256_s{ 0.0 };

    const f256_s z = x * x;
    constexpr f256_s a = f256_s{ 0.5 };
    constexpr f256_s tiny = f256_s{ 1.0e-300, 0.0, 0.0, 0.0 };

    f256_s b = z + f256_s{ 1.0 } - a;
    f256_s c = f256_s{ 1.0 } / tiny;
    f256_s d = f256_s{ 1.0 } / b;
    f256_s h = d;

    for (int i = 1; i <= 160; ++i)
    {
        const f256_s ii = f256_s{ static_cast<double>(i) };
        const f256_s an = -(ii * (ii - a));

        b += f256_s{ 2.0 };

        d = an * d + b;
        if (abs(d) < tiny)
            d = tiny;

        c = b + an / c;
        if (abs(c) < tiny)
            c = tiny;

        d = f256_s{ 1.0 } / d;
        const f256_s delta = d * c;
        h *= delta;

        if (abs(delta - f256_s{ 1.0 }) <= f256_s{ 64.0 } * f256_s::eps())
            break;
    }

    const f256_s out = exp(-z) * x * _f256_const::inv_sqrtpi * h;
    return _f256_detail::canonicalize_math_result(out);
}

[[nodiscard]] BL_NO_INLINE constexpr f256_s lgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return _f256_detail::canonicalize_math_result(_f256_detail::lgamma_positive_recurrence(x));

    const f256_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::infinity();

    const f256_s sinpix = sin(_f256_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::infinity();

    const f256_s out =
        log(_f256_const::pi)
        - log(abs(sinpix))
        - _f256_detail::lgamma_positive_recurrence(f256_s{ 1.0 } - x);

    return _f256_detail::canonicalize_math_result(out);
}
[[nodiscard]] BL_NO_INLINE constexpr f256_s tgamma(const f256_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
        ? std::numeric_limits<f256_s>::quiet_NaN()
        : std::numeric_limits<f256_s>::infinity();

    if (x > f256_s{ 0.0 })
        return _f256_detail::canonicalize_math_result(_f256_detail::gamma_positive_recurrence(x));

    const f256_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s sinpix = sin(_f256_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f256_s>::quiet_NaN();

    const f256_s out = _f256_const::pi / (sinpix * _f256_detail::gamma_positive_recurrence(f256_s{ 1.0 } - x));
    return _f256_detail::canonicalize_math_result(out);
}

}

#endif