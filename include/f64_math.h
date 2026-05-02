
/**
 * f64_math.h — f64 (double) constexpr <cmath> style API
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_INCLUDED
#define F64_INCLUDED

#include "fltx_common_math.h"
#include "fltx_math_utils.h"

#include <numbers>

namespace bl {

using f64 = double;

namespace detail::_f64
{
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::trunc_constexpr;
    using detail::fp::fmod_constexpr;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::frexp_exponent_constexpr;
    using detail::fp::ldexp_constexpr2;
    using detail::fp::log_constexpr;
    using detail::fp::log1p_constexpr;
    using detail::fp::sin_constexpr;
    using detail::fp::cos_constexpr;
    using detail::fp::tan_constexpr;
    using detail::fp::atan_constexpr;
    using detail::fp::atan2_constexpr;
    using detail::fp::sqrt_seed_constexpr;

    constexpr double pi          = std::numbers::pi_v<double>;
    constexpr double pi_2        = std::numbers::pi_v<double> * 0.5;
    constexpr double pi_4        = std::numbers::pi_v<double> * 0.25;
    constexpr double ln2         = std::numbers::ln2_v<double>;
    constexpr double inv_ln2     = std::numbers::log2e_v<double>;
    constexpr double inv_ln10    = std::numbers::log10e_v<double>;

    BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
    {
        return x == 0.0;
    }

    BL_FORCE_INLINE constexpr double abs(double x) noexcept
    {
        return fabs_constexpr(x);
    }

    BL_FORCE_INLINE constexpr int ilogb_finite_constexpr(double x) noexcept
    {
        return frexp_exponent_constexpr(x) - 1;
    }

    BL_FORCE_INLINE constexpr double nearbyint_constexpr(double x) noexcept
    {
        const double y = nearbyint_ties_even(x);
        if (iszero(y))
            return signbit_constexpr(x) ? -0.0 : 0.0;
        return y;
    }

    BL_FORCE_INLINE constexpr double round_half_away_zero(double x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        constexpr double integer_threshold = 4503599627370496.0;
        const double ax = signbit_constexpr(x) ? -x : x;
        if (ax >= integer_threshold)
            return x;

        if (signbit_constexpr(x))
        {
            const double y = -floor_constexpr((-x) + 0.5);
            return iszero(y) ? -0.0 : y;
        }

        return floor_constexpr(x + 0.5);
    }

    BL_FORCE_INLINE constexpr double nextafter_double_constexpr(double from, double to) noexcept
    {
        if (isnan(from) || isnan(to))
            return std::numeric_limits<double>::quiet_NaN();

        if (from == to)
            return to;

        if (from == 0.0)
            return signbit_constexpr(to)
                ? -std::numeric_limits<double>::denorm_min()
                :  std::numeric_limits<double>::denorm_min();

        std::uint64_t bits = std::bit_cast<std::uint64_t>(from);
        if ((from > 0.0) == (from < to))
            ++bits;
        else
            --bits;

        return std::bit_cast<double>(bits);
    }

    BL_FORCE_INLINE constexpr double copysign_constexpr(double magnitude, double sign_source) noexcept
    {
        const std::uint64_t magnitude_bits = std::bit_cast<std::uint64_t>(magnitude) & 0x7fffffffffffffffULL;
        const std::uint64_t sign_bits = std::bit_cast<std::uint64_t>(sign_source) & 0x8000000000000000ULL;
        return std::bit_cast<double>(magnitude_bits | sign_bits);
    }

    template<typename SignedInt>
    BL_FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(double x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);

        if (isnan(x) || isinf(x))
            return 0;

        constexpr double lo = static_cast<double>(std::numeric_limits<SignedInt>::lowest());
        constexpr double hi = static_cast<double>(std::numeric_limits<SignedInt>::max());
        if (x < lo || x > hi)
            return 0;

        return static_cast<SignedInt>(x);
    }

    BL_FORCE_INLINE constexpr double powi_nonneg(double base, std::uint64_t exp) noexcept
    {
        double result = 1.0;
        while (exp != 0)
        {
            if ((exp & 1u) != 0)
                result *= base;
            exp >>= 1u;
            if (exp != 0)
                base *= base;
        }
        return result;
    }
    BL_FORCE_INLINE constexpr double powi(double base, long long exp) noexcept
    {
        if (exp == 0)
            return 1.0;

        if (exp < 0)
            return 1.0 / powi_nonneg(base, static_cast<std::uint64_t>(-(exp + 1))) / base;

        return powi_nonneg(base, static_cast<std::uint64_t>(exp));
    }

    BL_FORCE_INLINE constexpr double exp_constexpr(double x) noexcept
    {
        constexpr double one = 1.0;
        constexpr double half_pos = 0.5;
        constexpr double half_neg = -0.5;
        constexpr double max_log = 7.09782712893383973096e+02;
        constexpr double min_log = -7.45133219101941108420e+02;
        constexpr double ln2_hi = 6.93147180369123816490e-01;
        constexpr double ln2_lo = 1.90821492927058770002e-10;
        constexpr double inv_ln2_local = 1.44269504088896338700e+00;
        constexpr double p1 = 1.66666666666666019037e-01;
        constexpr double p2 = -2.77777777770155933842e-03;
        constexpr double p3 = 6.61375632143793436117e-05;
        constexpr double p4 = -1.65339022054652515390e-06;
        constexpr double p5 = 4.13813679705723846039e-08;
        constexpr double half_ln2 = 3.46573590279972654709e-01;
        constexpr double one_and_half_ln2 = 1.03972077083991796413e+00;
        constexpr double tiny = 0x1.0p-28;

        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit_constexpr(x) ? 0.0 : std::numeric_limits<double>::infinity();
        if (x > max_log)
            return std::numeric_limits<double>::infinity();
        if (x < min_log)
            return 0.0;

        const bool neg = signbit_constexpr(x);
        const double ax = abs(x);

        double hi = 0.0;
        double lo = 0.0;
        int k = 0;

        if (ax > half_ln2)
        {
            if (ax < one_and_half_ln2)
            {
                hi = x - (neg ? -ln2_hi : ln2_hi);
                lo = neg ? -ln2_lo : ln2_lo;
                k = neg ? -1 : 1;
            }
            else
            {
                const double kd = trunc_constexpr(x * inv_ln2_local + (neg ? half_neg : half_pos));
                k = static_cast<int>(kd);
                hi = x - kd * ln2_hi;
                lo = kd * ln2_lo;
            }

            x = hi - lo;
        }
        else if (ax < tiny)
        {
            return one + x;
        }

        const double t = x * x;
        const double c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))));

        if (k == 0)
            return one - ((x * c) / (c - 2.0) - x);

        const double y = one - ((lo - (x * c) / (2.0 - c)) - hi);
        return ldexp_constexpr2(y, k);
    }
    BL_FORCE_INLINE constexpr double exp2_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit_constexpr(x) ? 0.0 : std::numeric_limits<double>::infinity();

        const double kd = nearbyint_ties_even(x);
        const int k = static_cast<int>(kd);
        const double frac = x - kd;
        return ldexp_constexpr2(exp_constexpr(frac * ln2), k);
    }

    BL_FORCE_INLINE constexpr double expm1_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit_constexpr(x) ? -1.0 : std::numeric_limits<double>::infinity();
        if (x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax < 0.125)
        {
            double term = x;
            double sum = x;
            for (int n = 2; n <= 32; ++n)
            {
                term *= x / static_cast<double>(n);
                sum += term;
            }
            return sum;
        }

        return exp_constexpr(x) - 1.0;
    }

    BL_FORCE_INLINE constexpr double sqrt_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0 || isinf(x))
            return x;

        double y = sqrt_seed_constexpr(x);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        return 0.5 * (y + x / y);
    }

    BL_FORCE_INLINE constexpr double fmod_constexpr_precise(double x, double y) noexcept
    {
        if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y) || x == 0.0)
            return x;

        const bool neg = signbit_constexpr(x);
        double ax = abs(x);
        const double ay = abs(y);

        if (ax < ay)
            return x;

        if (ax == ay)
            return neg ? -0.0 : 0.0;

        const int ey = ilogb_finite_constexpr(ay);

        while (ax >= ay)
        {
            int shift = ilogb_finite_constexpr(ax) - ey;
            double scaled = ldexp_constexpr2(ay, shift);

            if (scaled > ax)
                scaled = ldexp_constexpr2(ay, shift - 1);

            const double next = ax - scaled;
            if (next == ax)
                break;

            ax = next;
        }

        while (ax >= ay)
            ax -= ay;

        if (ax == 0.0)
            return neg ? -0.0 : 0.0;

        return neg ? -ax : ax;
    }
    BL_FORCE_INLINE constexpr double remainder_constexpr(double x, double y) noexcept
    {
        if (isnan(x) || isnan(y) || y == 0.0 || isinf(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y))
            return x;

        double r = fmod_constexpr_precise(x, y);
        if (isnan(r))
            return r;

        const double ay = abs(y);
        const double ar = abs(r);
        const double half = ay * 0.5;

        if (ar > half)
        {
            r -= signbit_constexpr(r) ? -ay : ay;
        }
        else if (ar == half)
        {
            const double n = nearbyint_constexpr(x / y);
            const double parity = fmod_constexpr(abs(n), 2.0);
            if (parity != 0.0)
                r -= signbit_constexpr(r) ? -ay : ay;
        }

        if (r == 0.0)
            r = signbit_constexpr(x) ? -0.0 : 0.0;
        return r;
    }

    BL_FORCE_INLINE constexpr double log2_constexpr(double x) noexcept
    {
        return log_constexpr(x) * inv_ln2;
    }
    BL_FORCE_INLINE constexpr double log10_constexpr(double x) noexcept
    {
        return log_constexpr(x) * inv_ln10;
    }
    BL_FORCE_INLINE constexpr double sinh_constexpr(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        const double ex = exp_constexpr(ax);
        double out = (ex - 1.0 / ex) * 0.5;
        return signbit_constexpr(x) ? -out : out;
    }
    BL_FORCE_INLINE constexpr double cosh_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return std::numeric_limits<double>::infinity();

        const double ax = abs(x);
        const double ex = exp_constexpr(ax);
        return (ex + 1.0 / ex) * 0.5;
    }
    BL_FORCE_INLINE constexpr double tanh_constexpr(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;
        if (isinf(x))
            return signbit_constexpr(x) ? -1.0 : 1.0;

        const double ax = abs(x);
        if (ax > 20.0)
            return signbit_constexpr(x) ? -1.0 : 1.0;

        const double em1 = expm1_constexpr(ax + ax);
        double out = em1 / (em1 + 2.0);
        if (signbit_constexpr(x))
            out = -out;
        return out;
    }
    BL_FORCE_INLINE constexpr double asinh_constexpr(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        double out = 0.0;
        if (ax > 0x1p500)
            out = log_constexpr(ax) + ln2;
        else
            out = log1p_constexpr(ax + (ax * ax) / (1.0 + sqrt_constexpr(1.0 + ax * ax)));

        return signbit_constexpr(x) ? -out : out;
    }
    BL_FORCE_INLINE constexpr double acosh_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x < 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 1.0)
            return 0.0;
        if (isinf(x))
            return x;

        if (x > 0x1p500)
            return log_constexpr(x) + ln2;

        return log_constexpr(x + sqrt_constexpr((x - 1.0) * (x + 1.0)));
    }
    BL_FORCE_INLINE constexpr double atanh_constexpr(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (ax == 1.0)
            return signbit_constexpr(x)
                ? -std::numeric_limits<double>::infinity()
                :  std::numeric_limits<double>::infinity();

        return 0.5 * (log1p_constexpr(x) - log1p_constexpr(-x));
    }

    BL_FORCE_INLINE constexpr double cbrt_constexpr(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const bool neg = signbit_constexpr(x);
        const double ax = neg ? -x : x;

        double y = exp_constexpr(log_constexpr(ax) / 3.0);
        for (int i = 0; i < 5; ++i)
            y = (y + y + ax / (y * y)) / 3.0;

        const double y_prev = nextafter_double_constexpr(y, 0.0);
        const double y_next = nextafter_double_constexpr(y, std::numeric_limits<double>::infinity());

        const double e = abs(y * y * y - ax);
        const double e_prev = abs(y_prev * y_prev * y_prev - ax);
        const double e_next = abs(y_next * y_next * y_next - ax);

        if (e_prev < e)
            y = y_prev;
        else if (e_next < e)
            y = y_next;

        return neg ? -y : y;
    }
    BL_FORCE_INLINE constexpr double hypot_constexpr(double x, double y) noexcept
    {
        if (isinf(x) || isinf(y))
            return std::numeric_limits<double>::infinity();
        if (isnan(x))
            return x;
        if (isnan(y))
            return y;

        double ax = abs(x);
        double ay = abs(y);
        if (ax < ay)
            std::swap(ax, ay);

        if (ax == 0.0)
            return 0.0;

        const double r = ay / ax;
        return ax * sqrt_constexpr(1.0 + r * r);
    }

    constexpr double erf_erx  = 8.45062911510467529297e-01;
    constexpr double erf_efx8 = 1.02703333676410069053e+00;

    constexpr double erf_pp0 = 1.28379167095512558561e-01;
    constexpr double erf_pp1 = -3.25042107247001499370e-01;
    constexpr double erf_pp2 = -2.84817495755985104766e-02;
    constexpr double erf_pp3 = -5.77027029648944159157e-03;
    constexpr double erf_pp4 = -2.37630166566501626084e-05;

    constexpr double erf_qq1 = 3.97917223959155352819e-01;
    constexpr double erf_qq2 = 6.50222499887672944485e-02;
    constexpr double erf_qq3 = 5.08130628187576562776e-03;
    constexpr double erf_qq4 = 1.32494738004321644526e-04;
    constexpr double erf_qq5 = -3.96022827877536812320e-06;

    constexpr double erf_pa0 = -2.36211856075265944077e-03;
    constexpr double erf_pa1 = 4.14856118683748331666e-01;
    constexpr double erf_pa2 = -3.72207876035701323847e-01;
    constexpr double erf_pa3 = 3.18346619901161753674e-01;
    constexpr double erf_pa4 = -1.10894694282396677476e-01;
    constexpr double erf_pa5 = 3.54783043256182359371e-02;
    constexpr double erf_pa6 = -2.16637559486879084300e-03;

    constexpr double erf_qa1 = 1.06420880400844228286e-01;
    constexpr double erf_qa2 = 5.40397917702171048937e-01;
    constexpr double erf_qa3 = 7.18286544141962662868e-02;
    constexpr double erf_qa4 = 1.26171219808761642112e-01;
    constexpr double erf_qa5 = 1.36370839120290507362e-02;
    constexpr double erf_qa6 = 1.19844998467991074170e-02;

    constexpr double erf_ra0 = -9.86494403484714822705e-03;
    constexpr double erf_ra1 = -6.93858572707181764372e-01;
    constexpr double erf_ra2 = -1.05586262253232909814e+01;
    constexpr double erf_ra3 = -6.23753324503260060396e+01;
    constexpr double erf_ra4 = -1.62396669462573470355e+02;
    constexpr double erf_ra5 = -1.84605092906711035994e+02;
    constexpr double erf_ra6 = -8.12874355063065934246e+01;
    constexpr double erf_ra7 = -9.81432934416914548592e+00;

    constexpr double erf_sa1 = 1.96512716674392571292e+01;
    constexpr double erf_sa2 = 1.37657754143519042600e+02;
    constexpr double erf_sa3 = 4.34565877475229228821e+02;
    constexpr double erf_sa4 = 6.45387271733267880336e+02;
    constexpr double erf_sa5 = 4.29008140027567833386e+02;
    constexpr double erf_sa6 = 1.08635005541779435134e+02;
    constexpr double erf_sa7 = 6.57024977031928170135e+00;
    constexpr double erf_sa8 = -6.04244152148580987438e-02;

    constexpr double erf_rb0 = -9.86494292470009928597e-03;
    constexpr double erf_rb1 = -7.99283237680523006574e-01;
    constexpr double erf_rb2 = -1.77579549177547519889e+01;
    constexpr double erf_rb3 = -1.60636384855821916062e+02;
    constexpr double erf_rb4 = -6.37566443368389627722e+02;
    constexpr double erf_rb5 = -1.02509513161107724954e+03;
    constexpr double erf_rb6 = -4.83519191608651397019e+02;

    constexpr double erf_sb1 = 3.03380607434824582924e+01;
    constexpr double erf_sb2 = 3.25792512996573918826e+02;
    constexpr double erf_sb3 = 1.53672958608443695994e+03;
    constexpr double erf_sb4 = 3.19985821950859553908e+03;
    constexpr double erf_sb5 = 2.55305040643316442583e+03;
    constexpr double erf_sb6 = 4.74528541206955367215e+02;
    constexpr double erf_sb7 = -2.24409524465858183362e+01;

    BL_FORCE_INLINE constexpr double truncate_low_word(double x) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x) & 0xffffffff00000000ull;
        return std::bit_cast<double>(bits);
    }

    BL_FORCE_INLINE constexpr double erfc1_constexpr(double ax) noexcept
    {
        const double s = ax - 1.0;
        const double p = erf_pa0 + s * (erf_pa1 + s * (erf_pa2 + s * (erf_pa3 + s * (erf_pa4 + s * (erf_pa5 + s * erf_pa6)))));
        const double q = 1.0 + s * (erf_qa1 + s * (erf_qa2 + s * (erf_qa3 + s * (erf_qa4 + s * (erf_qa5 + s * erf_qa6)))));
        return 1.0 - erf_erx - p / q;
    }

    BL_FORCE_INLINE constexpr double erfc2_constexpr(double ax) noexcept
    {
        const double s = 1.0 / (ax * ax);

        double r = 0.0;
        double q = 0.0;
        if (ax < 2.85714285714285714286)
        {
            r = erf_ra0 + s * (erf_ra1 + s * (erf_ra2 + s * (erf_ra3 + s * (erf_ra4 + s * (erf_ra5 + s * (erf_ra6 + s * erf_ra7))))));
            q = 1.0 + s * (erf_sa1 + s * (erf_sa2 + s * (erf_sa3 + s * (erf_sa4 + s * (erf_sa5 + s * (erf_sa6 + s * (erf_sa7 + s * erf_sa8)))))));
        }
        else
        {
            r = erf_rb0 + s * (erf_rb1 + s * (erf_rb2 + s * (erf_rb3 + s * (erf_rb4 + s * (erf_rb5 + s * erf_rb6)))));
            q = 1.0 + s * (erf_sb1 + s * (erf_sb2 + s * (erf_sb3 + s * (erf_sb4 + s * (erf_sb5 + s * (erf_sb6 + s * erf_sb7))))));
        }

        const double z = truncate_low_word(ax);
        return exp_constexpr(-z * z - 0.5625) * exp_constexpr((z - ax) * (z + ax) + r / q) / ax;
    }

    BL_FORCE_INLINE constexpr double erf_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit_constexpr(x) ? -1.0 : 1.0;
        if (x == 0.0)
            return x;

        const bool neg = signbit_constexpr(x);
        const double ax = neg ? -x : x;

        double out = 0.0;
        if (ax < 0.84375)
        {
            if (ax < 0x1p-28)
                return x * (1.0 + erf_efx8 * 0.125);

            const double z = ax * ax;
            const double r = erf_pp0 + z * (erf_pp1 + z * (erf_pp2 + z * (erf_pp3 + z * erf_pp4)));
            const double s = 1.0 + z * (erf_qq1 + z * (erf_qq2 + z * (erf_qq3 + z * (erf_qq4 + z * erf_qq5))));
            out = ax + ax * (r / s);
        }
        else if (ax < 1.25)
        {
            const double s = ax - 1.0;
            const double p = erf_pa0 + s * (erf_pa1 + s * (erf_pa2 + s * (erf_pa3 + s * (erf_pa4 + s * (erf_pa5 + s * erf_pa6)))));
            const double q = 1.0 + s * (erf_qa1 + s * (erf_qa2 + s * (erf_qa3 + s * (erf_qa4 + s * (erf_qa5 + s * erf_qa6)))));
            out = erf_erx + p / q;
        }
        else if (ax < 6.0)
        {
            out = 1.0 - erfc2_constexpr(ax);
        }
        else
        {
            out = 1.0 - 0x1p-1022;
        }

        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr double erfc_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit_constexpr(x) ? 2.0 : 0.0;

        const bool neg = signbit_constexpr(x);
        const double ax = neg ? -x : x;

        if (ax < 0.84375)
        {
            if (ax < 0x1p-56)
                return 1.0 - x;

            const double z = x * x;
            const double r = erf_pp0 + z * (erf_pp1 + z * (erf_pp2 + z * (erf_pp3 + z * erf_pp4)));
            const double s = 1.0 + z * (erf_qq1 + z * (erf_qq2 + z * (erf_qq3 + z * (erf_qq4 + z * erf_qq5))));
            const double y = r / s;

            if (neg || ax < 0.25)
                return 1.0 - (x + x * y);

            return 0.5 - (x - 0.5 + x * y);
        }

        if (ax < 28.0)
        {
            const double y = (ax < 1.25) ? erfc1_constexpr(ax) : erfc2_constexpr(ax);
            return neg ? 2.0 - y : y;
        }

        return neg ? 2.0 - 0x1p-1022 : 0x1p-1022 * 0x1p-1022;
    }

    BL_FORCE_INLINE constexpr double lgamma_positive_constexpr(double x) noexcept
    {
        constexpr double coeffs[] =
        {
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        };

        double y = coeffs[0];
        const double z = x - 1.0;
        for (int i = 1; i < static_cast<int>(sizeof(coeffs) / sizeof(coeffs[0])); ++i)
            y += coeffs[i] / (z + static_cast<double>(i));

        const double t = z + 7.5;
        return 0.91893853320467274178032973640562 + (z + 0.5) * log_constexpr(t) - t + log_constexpr(y);
    }

    BL_FORCE_INLINE constexpr double lgamma_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit_constexpr(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
            return lgamma_positive_constexpr(x);

        const double xi = trunc_constexpr(x);
        if (xi == x)
            return std::numeric_limits<double>::infinity();

        const double sinpix = sin_constexpr(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::infinity();

        return log_constexpr(pi) - log_constexpr(abs(sinpix)) - lgamma_positive_constexpr(1.0 - x);
    }

    BL_FORCE_INLINE constexpr double tgamma_constexpr(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit_constexpr(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
            return exp_constexpr(lgamma_positive_constexpr(x));

        const double xi = trunc_constexpr(x);
        if (xi == x)
            return std::numeric_limits<double>::quiet_NaN();

        const double sinpix = sin_constexpr(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::quiet_NaN();

        return pi / (sinpix * exp_constexpr(lgamma_positive_constexpr(1.0 - x)));
    }
}

[[nodiscard]] BL_FORCE_INLINE constexpr double abs(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::abs(x);
    return detail::_f64::abs(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double fabs(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fabs(x);
    return abs(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::signbit(x);
    return detail::_f64::signbit_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isnan(x);
    return detail::_f64::isnan(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isinf(x);
    return detail::_f64::isinf(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::isfinite(x);
    return detail::_f64::isfinite(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(double x) noexcept
{
    return x == 0.0;
}

[[nodiscard]] BL_FORCE_INLINE constexpr double floor(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::floor(x);
    return detail::_f64::floor_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double ceil(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::ceil(x);
    return detail::_f64::ceil_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double trunc(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::trunc(x);
    return detail::_f64::trunc_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double round(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::round(x);
    return detail::_f64::round_half_away_zero(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double nearbyint(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nearbyint(x);
    return detail::_f64::nearbyint_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double rint(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::rint(x);
    return detail::_f64::nearbyint_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lround(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lround(x);
    return detail::_f64::to_signed_integer_or_zero<long>(detail::_f64::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::llround(x);
    return detail::_f64::to_signed_integer_or_zero<long long>(detail::_f64::round_half_away_zero(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lrint(x);
    return detail::_f64::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::llrint(x);
    return detail::_f64::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmod(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fmod(x, y);
    return detail::_f64::fmod_constexpr_precise(x, y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double remainder(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::remainder(x, y);
    return detail::_f64::remainder_constexpr(x, y);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double remquo(double x, double y, int* quo) noexcept
{
    if (!bl::use_constexpr_math())
        return std::remquo(x, y, quo);

    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (isinf(y))
        return x;

    const bool quotient_negative = signbit(x) != signbit(y);
    const double ay = abs(y);
    double ax = abs(x);

    int quotient_low_bits = 0;
    bool quotient_nonzero = false;

    if (ax >= ay)
    {
        const int ey = detail::_f64::ilogb_finite_constexpr(ay);

        while (ax >= ay)
        {
            int shift = detail::_f64::ilogb_finite_constexpr(ax) - ey;
            double scaled = detail::_f64::ldexp_constexpr2(ay, shift);

            if (scaled > ax)
            {
                --shift;
                scaled = detail::_f64::ldexp_constexpr2(ay, shift);
            }

            const double next = ax - scaled;
            if (next == ax)
                break;

            ax = next;
            quotient_nonzero = true;

            if (shift < 3)
                quotient_low_bits = (quotient_low_bits + (1 << shift)) & 0x7;
        }

        while (ax >= ay)
        {
            ax -= ay;
            quotient_nonzero = true;
            quotient_low_bits = (quotient_low_bits + 1) & 0x7;
        }
    }

    double r = signbit(x) ? -ax : ax;
    const double half = ay * 0.5;

    if (ax > half || (ax == half && (quotient_low_bits & 1) != 0))
    {
        r -= signbit(r) ? -ay : ay;
        quotient_nonzero = true;
        quotient_low_bits = (quotient_low_bits + 1) & 0x7;
    }

    if (quo)
    {
        int bits = quotient_low_bits;
        if (bits == 0 && quotient_nonzero)
            bits = 8;
        if (quotient_negative)
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        r = signbit(x) ? -0.0 : 0.0;

    return r;
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fma(double x, double y, double z) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fma(x, y, z);
    return x * y + z;
}
[[nodiscard]] BL_FORCE_INLINE constexpr double fmin(double a, double b) noexcept
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}
[[nodiscard]] BL_FORCE_INLINE constexpr double fdim(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::fdim(x, y);
    return (x > y) ? (x - y) : 0.0;
}
[[nodiscard]] BL_FORCE_INLINE constexpr double copysign(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::copysign(x, y);
    return detail::_f64::copysign_constexpr(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ldexp(double x, int e) noexcept
{
    if (!bl::use_constexpr_math())
        return std::ldexp(x, e);
    return detail::_f64::ldexp_constexpr2(x, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double scalbn(double x, int e) noexcept
{
    return ldexp(x, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double scalbln(double x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double frexp(double x, int* exp) noexcept
{
    if (!bl::use_constexpr_math())
        return std::frexp(x, exp);

    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    int e = detail::_f64::frexp_exponent_constexpr(x);
    double m = ldexp(x, -e);
    const double am = abs(m);

    if (am < 0.5)
    {
        m *= 2.0;
        --e;
    }
    else if (am >= 1.0)
    {
        m *= 0.5;
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
{
    const double i = trunc(x);
    if (iptr)
        *iptr = i;

    double frac = x - i;
    if (iszero(frac))
        frac = signbit(x) ? -0.0 : 0.0;
    return frac;
}
[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(double x) noexcept
{
    if (isnan(x))
        return FP_ILOGBNAN;
    if (iszero(x))
        return FP_ILOGB0;
    if (isinf(x))
        return std::numeric_limits<int>::max();

    return detail::_f64::ilogb_finite_constexpr(abs(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr double logb(double x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return -std::numeric_limits<double>::infinity();
    if (isinf(x))
        return std::numeric_limits<double>::infinity();

    return static_cast<double>(ilogb(x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr double nextafter(double from, double to) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nextafter(from, to);
    return detail::_f64::nextafter_double_constexpr(from, to);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, long double to) noexcept
{
    if (!bl::use_constexpr_math())
        return std::nexttoward(from, to);
    return nextafter(from, static_cast<double>(to));
}
[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, double to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double exp(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::exp(x);
    return detail::_f64::exp_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double exp2(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::exp2(x);
    return detail::_f64::exp2_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double expm1(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::expm1(x);
    return detail::_f64::expm1_constexpr(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(double x) noexcept
{ 
    if (!bl::use_constexpr_math())
        return std::log(x);
    return detail::_f64::log_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double log(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log(x);
    return detail::_f64::log_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double log2(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log2(x);
    return detail::_f64::log2_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double log10(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log10(x);
    return detail::_f64::log10_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double log1p(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::log1p(x);
    return detail::_f64::log1p_constexpr(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double sqrt(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sqrt(x);
    return detail::_f64::sqrt_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double cbrt(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cbrt(x);
    return detail::_f64::cbrt_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double hypot(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::hypot(x, y);
    return detail::_f64::hypot_constexpr(x, y);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double sin(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sin(x);
    return detail::_f64::sin_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double cos(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cos(x);
    return detail::_f64::cos_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double tan(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tan(x);
    return detail::_f64::tan_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double atan(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atan(x);
    return detail::_f64::atan_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double atan2(double y, double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atan2(y, x);
    return detail::_f64::atan2_constexpr(y, x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double asin(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::asin(x);
    return atan2(x, sqrt(1.0 - x * x));
}
[[nodiscard]] BL_FORCE_INLINE constexpr double acos(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::acos(x);
    return atan2(sqrt(1.0 - x * x), x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double sinh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::sinh(x);
    return detail::_f64::sinh_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double cosh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::cosh(x);
    return detail::_f64::cosh_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double tanh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tanh(x);
    return detail::_f64::tanh_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double asinh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::asinh(x);
    return detail::_f64::asinh_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double acosh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::acosh(x);
    return detail::_f64::acosh_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double atanh(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::atanh(x);
    return detail::_f64::atanh_constexpr(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double pow(double x, double y) noexcept
{
    if (!bl::use_constexpr_math())
        return std::pow(x, y);

    if (iszero(y))
        return 1.0;

    if (isnan(x) || isnan(y))
        return std::numeric_limits<double>::quiet_NaN();

    const double yi = trunc(y);
    const bool y_is_int = (yi == y);

    if (y_is_int && yi >= static_cast<double>(std::numeric_limits<long long>::min()) &&
        yi <= static_cast<double>(std::numeric_limits<long long>::max()))
    {
        return detail::_f64::powi(x, static_cast<long long>(yi));
    }

    if (x < 0.0 || (x == 0.0 && signbit(x)))
    {
        if (!y_is_int)
            return std::numeric_limits<double>::quiet_NaN();

        const double magnitude = exp(y * log(-x));
        const double parity = fmod(abs(yi), 2.0);
        return (parity == 1.0) ? -magnitude : magnitude;
    }

    return exp(y * log(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double erf(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::erf(x);
    return detail::_f64::erf_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double erfc(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::erfc(x);
    return detail::_f64::erfc_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double lgamma(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::lgamma(x);
    return detail::_f64::lgamma_constexpr(x);
}
[[nodiscard]] BL_FORCE_INLINE constexpr double tgamma(double x) noexcept
{
    if (!bl::use_constexpr_math())
        return std::tgamma(x);
    return detail::_f64::tgamma_constexpr(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr int fpclassify(double x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<double>::min() ? FP_SUBNORMAL : FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(double x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(double a, double b) noexcept
{
    return isnan(a) || isnan(b);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a > b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a >= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(double a, double b) noexcept
{
    return !isunordered(a, b) && a < b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(double a, double b) noexcept
{
    return !isunordered(a, b) && a <= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(double a, double b) noexcept
{
    return !isunordered(a, b) && a != b;
}

} // namespace bl

#endif