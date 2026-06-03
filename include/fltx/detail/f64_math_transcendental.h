/**
 * fltx/detail/f64_math_transcendental.h - constexpr <cmath>-style transcendental math helpers for f64.
 *
 * f64 exp/log, roots, pow, trig, hyperbolic, erf, and gamma helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_MATH_TRANSCENDENTAL_INCLUDED
#define F64_MATH_TRANSCENDENTAL_INCLUDED

#include "fltx/detail/f64_math_basic.h"


namespace bl {

namespace detail::_f64_impl
{
    // exp / log
    BL_FORCE_INLINE constexpr double exp(double x) noexcept
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
            return signbit(x) ? 0.0 : std::numeric_limits<double>::infinity();
        if (x > max_log)
            return std::numeric_limits<double>::infinity();
        if (x < min_log)
            return 0.0;

        const bool neg = signbit(x);
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
                const double kd = trunc(x * inv_ln2_local + (neg ? half_neg : half_pos));
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
        return ldexp(y, k);
    }

    BL_FORCE_INLINE constexpr double exp2(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit(x) ? 0.0 : std::numeric_limits<double>::infinity();

        const double kd   = nearbyint_ties_even(x);
        const int k = static_cast<int>(kd);
        const double frac = x - kd;
        return ldexp(exp(frac * ln2), k);
    }

    BL_FORCE_INLINE constexpr double expm1(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x) ? -1.0 : std::numeric_limits<double>::infinity();
        if (x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax < 0.125)
        {
            double term = x;
            double sum  = x;
            for (int n = 2; n <= 32; ++n)
            {
                term *= x / static_cast<double>(n);
                sum += term;
            }
            return sum;
        }

        return exp(x) - 1.0;
    }

    BL_FORCE_INLINE constexpr double log2(double x) noexcept
    {
        return log(x) * inv_ln2;
    }

    BL_FORCE_INLINE constexpr double log10(double x) noexcept
    {
        return log(x) * inv_ln10;
    }

    // roots
    BL_FORCE_INLINE constexpr double sqrt(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0 || isinf(x))
            return x;

        double y = sqrt_seed(x);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        return 0.5 * (y + x / y);
    }

    BL_FORCE_INLINE constexpr double cbrt(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        double y = exp(log(ax) / 3.0);
        for (int i = 0; i < 5; ++i)
            y = (y + y + ax / (y * y)) / 3.0;

        const double y_prev = nextafter(y, 0.0);
        const double y_next = nextafter(y, std::numeric_limits<double>::infinity());

        const double e = abs(y * y * y - ax);
        const double e_prev = abs(y_prev * y_prev * y_prev - ax);
        const double e_next = abs(y_next * y_next * y_next - ax);

        if (e_prev < e)
            y = y_prev;
        else if (e_next < e)
            y = y_next;

        return neg ? -y : y;
    }

    BL_FORCE_INLINE constexpr double hypot(double x, double y) noexcept
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
        {
            const double tmp = ax;
            ax = ay;
            ay = tmp;
        }

        if (ax == 0.0)
            return 0.0;

        const double r = ay / ax;
        return ax * sqrt(1.0 + r * r);
    }

    // pow
    [[nodiscard]] BL_FORCE_INLINE constexpr double pow(double x, double y) noexcept
    {
        if (iszero(y))
            return 1.0;

        if (isnan(x) || isnan(y))
            return std::numeric_limits<double>::quiet_NaN();

        const double yi = trunc(y);
        const bool y_is_int = (yi == y);

        if (y_is_int && yi >= static_cast<double>(std::numeric_limits<long long>::min()) &&
            yi <= static_cast<double>(std::numeric_limits<long long>::max()))
        {
            return powi(x, static_cast<long long>(yi));
        }

        if (x < 0.0 || (x == 0.0 && signbit(x)))
        {
            if (!y_is_int)
                return std::numeric_limits<double>::quiet_NaN();

            const double magnitude = exp(y * log(-x));
            const double parity    = fmod_exact(abs(yi), 2.0);
            return (parity == 1.0) ? -magnitude : magnitude;
        }

        return exp(y * log(x));
    }

    // hyperbolic
    BL_FORCE_INLINE constexpr double sinh(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax < 0.25)
        {
            const double x2 = x * x;
            double p = 1.6059043836821613e-10;  // 1/13!
            p = p * x2 + 2.5052108385441720e-8; // 1/11!
            p = p * x2 + 2.7557319223985893e-6; // 1/9!
            p = p * x2 + 1.9841269841269841e-4; // 1/7!
            p = p * x2 + 8.3333333333333332e-3; // 1/5!
            p = p * x2 + 1.6666666666666666e-1; // 1/3!
            return x + x * x2 * p;
        }

        if (ax < 0.5)
        {
            const double em1 = expm1(ax);
            const double out = (em1 * (em1 + 2.0)) / ((em1 + 1.0) * 2.0);
            return signbit(x) ? -out : out;
        }

        const double ex = exp(ax);
        double out = (ex - 1.0 / ex) * 0.5;
        return signbit(x) ? -out : out;
    }

    BL_FORCE_INLINE constexpr double cosh(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return std::numeric_limits<double>::infinity();

        const double ax = abs(x);
        const double ex = exp(ax);
        return (ex + 1.0 / ex) * 0.5;
    }

    BL_FORCE_INLINE constexpr double tanh(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;
        if (isinf(x))
            return signbit(x) ? -1.0 : 1.0;

        const double ax = abs(x);
        if (ax > 20.0)
            return signbit(x) ? -1.0 : 1.0;

        const double em1 = expm1(ax + ax);
        double out = em1 / (em1 + 2.0);
        if (signbit(x))
            out = -out;
        return out;
    }

    BL_FORCE_INLINE constexpr double asinh(double x) noexcept
    {
        if (isnan(x) || isinf(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        double out = 0.0;
        if (ax > 0x1p500)
            out = log(ax) + ln2;
        else
            out = log1p(ax + (ax * ax) / (1.0 + sqrt(1.0 + ax * ax)));

        return signbit(x) ? -out : out;
    }

    BL_FORCE_INLINE constexpr double acosh(double x) noexcept
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
            return log(x) + ln2;

        return log(x + sqrt((x - 1.0) * (x + 1.0)));
    }

    BL_FORCE_INLINE constexpr double atanh(double x) noexcept
    {
        if (isnan(x) || x == 0.0)
            return x;

        const double ax = abs(x);
        if (ax > 1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (ax == 1.0)
            return signbit(x)
                ? -std::numeric_limits<double>::infinity()
                :  std::numeric_limits<double>::infinity();

        return 0.5 * (log1p(x) - log1p(-x));
    }

    // erf / erfc
    namespace erf_coeffs
    {
        constexpr double erx  = 8.45062911510467529297e-01;
        constexpr double efx8 = 1.02703333676410069053e+00;

        constexpr double pp0 = 1.28379167095512558561e-01;
        constexpr double pp1 = -3.25042107247001499370e-01;
        constexpr double pp2 = -2.84817495755985104766e-02;
        constexpr double pp3 = -5.77027029648944159157e-03;
        constexpr double pp4 = -2.37630166566501626084e-05;

        constexpr double qq1 = 3.97917223959155352819e-01;
        constexpr double qq2 = 6.50222499887672944485e-02;
        constexpr double qq3 = 5.08130628187576562776e-03;
        constexpr double qq4 = 1.32494738004321644526e-04;
        constexpr double qq5 = -3.96022827877536812320e-06;

        constexpr double pa0 = -2.36211856075265944077e-03;
        constexpr double pa1 = 4.14856118683748331666e-01;
        constexpr double pa2 = -3.72207876035701323847e-01;
        constexpr double pa3 = 3.18346619901161753674e-01;
        constexpr double pa4 = -1.10894694282396677476e-01;
        constexpr double pa5 = 3.54783043256182359371e-02;
        constexpr double pa6 = -2.16637559486879084300e-03;

        constexpr double qa1 = 1.06420880400844228286e-01;
        constexpr double qa2 = 5.40397917702171048937e-01;
        constexpr double qa3 = 7.18286544141962662868e-02;
        constexpr double qa4 = 1.26171219808761642112e-01;
        constexpr double qa5 = 1.36370839120290507362e-02;
        constexpr double qa6 = 1.19844998467991074170e-02;

        constexpr double ra0 = -9.86494403484714822705e-03;
        constexpr double ra1 = -6.93858572707181764372e-01;
        constexpr double ra2 = -1.05586262253232909814e+01;
        constexpr double ra3 = -6.23753324503260060396e+01;
        constexpr double ra4 = -1.62396669462573470355e+02;
        constexpr double ra5 = -1.84605092906711035994e+02;
        constexpr double ra6 = -8.12874355063065934246e+01;
        constexpr double ra7 = -9.81432934416914548592e+00;

        constexpr double sa1 = 1.96512716674392571292e+01;
        constexpr double sa2 = 1.37657754143519042600e+02;
        constexpr double sa3 = 4.34565877475229228821e+02;
        constexpr double sa4 = 6.45387271733267880336e+02;
        constexpr double sa5 = 4.29008140027567833386e+02;
        constexpr double sa6 = 1.08635005541779435134e+02;
        constexpr double sa7 = 6.57024977031928170135e+00;
        constexpr double sa8 = -6.04244152148580987438e-02;

        constexpr double rb0 = -9.86494292470009928597e-03;
        constexpr double rb1 = -7.99283237680523006574e-01;
        constexpr double rb2 = -1.77579549177547519889e+01;
        constexpr double rb3 = -1.60636384855821916062e+02;
        constexpr double rb4 = -6.37566443368389627722e+02;
        constexpr double rb5 = -1.02509513161107724954e+03;
        constexpr double rb6 = -4.83519191608651397019e+02;

        constexpr double sb1 = 3.03380607434824582924e+01;
        constexpr double sb2 = 3.25792512996573918826e+02;
        constexpr double sb3 = 1.53672958608443695994e+03;
        constexpr double sb4 = 3.19985821950859553908e+03;
        constexpr double sb5 = 2.55305040643316442583e+03;
        constexpr double sb6 = 4.74528541206955367215e+02;
        constexpr double sb7 = -2.24409524465858183362e+01;
    } // namespace erf_coeffs

    BL_FORCE_INLINE constexpr double truncate_low_word(double x) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x) & 0xffffffff00000000ull;
        return std::bit_cast<double>(bits);
    }

    BL_FORCE_INLINE constexpr double erfc1(double ax) noexcept
    {
        using namespace erf_coeffs;

        const double s = ax - 1.0;
        const double p = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        const double q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        return 1.0 - erx - p / q;
    }

    BL_FORCE_INLINE constexpr double erfc2(double ax) noexcept
    {
        using namespace erf_coeffs;

        const double s = 1.0 / (ax * ax);

        double r = 0.0;
        double q = 0.0;
        if (ax < 2.85714285714285714286)
        {
            r = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
            q = 1.0 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
        }
        else
        {
            r = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
            q = 1.0 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
        }

        const double z = truncate_low_word(ax);
        return exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + r / q) / ax;
    }

    BL_FORCE_INLINE constexpr double erf(double x) noexcept
    {
        using namespace erf_coeffs;

        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x) ? -1.0 : 1.0;
        if (x == 0.0)
            return x;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        double out = 0.0;
        if (ax < 0.84375)
        {
            if (ax < 0x1p-28)
                return x * (1.0 + efx8 * 0.125);

            const double z = ax * ax;
            const double r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            const double s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            out = ax + ax * (r / s);
        }
        else if (ax < 1.25)
        {
            const double s = ax - 1.0;
            const double p = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
            const double q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
            out = erx + p / q;
        }
        else if (ax < 6.0)
        {
            out = 1.0 - erfc2(ax);
        }
        else
        {
            out = 1.0 - 0x1p-1022;
        }

        return neg ? -out : out;
    }

    BL_FORCE_INLINE constexpr double erfc(double x) noexcept
    {
        using namespace erf_coeffs;

        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0)
            return 1.0;
        if (isinf(x))
            return signbit(x) ? 2.0 : 0.0;

        const bool neg = signbit(x);
        const double ax = neg ? -x : x;

        if (ax < 0.84375)
        {
            if (ax < 0x1p-56)
                return 1.0 - x;

            const double z = x * x;
            const double r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            const double s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            const double y = r / s;

            if (neg || ax < 0.25)
                return 1.0 - (x + x * y);

            return 0.5 - (x - 0.5 + x * y);
        }

        if (ax < 28.0)
        {
            const double y = (ax < 1.25) ? erfc1(ax) : erfc2(ax);
            return neg ? 2.0 - y : y;
        }

        return neg ? 2.0 - 0x1p-1022 : 0x1p-1022 * 0x1p-1022;
    }

    // gamma
    BL_FORCE_INLINE constexpr double lgamma_positive(double x) noexcept
    {
        if (x == 1.0 || x == 2.0)
            return 0.0;
        if (x == 0.5)
            return 0.57236494292470009;
        if (x == 1.5)
            return -0.12078223763524522;

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
        return 0.91893853320467274178032973640562 + (z + 0.5) * log(t) - t + log(y);
    }

    BL_FORCE_INLINE constexpr double lgamma(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
            return lgamma_positive(x);

        const double xi = trunc(x);
        if (xi == x)
            return std::numeric_limits<double>::infinity();

        const double sinpix = sin(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::infinity();

        return log(pi) - log(abs(sinpix)) - lgamma_positive(1.0 - x);
    }

    BL_FORCE_INLINE constexpr double tgamma(double x) noexcept
    {
        if (isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(x))
            return signbit(x)
                ? std::numeric_limits<double>::quiet_NaN()
                : std::numeric_limits<double>::infinity();

        if (x > 0.0)
        {
            const double xi = trunc(x);
            if (xi == x && x <= 171.0)
            {
                double result = 1.0;
                for (int i = 2; i < static_cast<int>(x); ++i)
                    result *= static_cast<double>(i);
                return result;
            }
        }

        if (x > 0.0)
            return exp(lgamma_positive(x));

        const double xi = trunc(x);
        if (xi == x)
            return std::numeric_limits<double>::quiet_NaN();

        const double sinpix = sin(pi * x);
        if (sinpix == 0.0)
            return std::numeric_limits<double>::quiet_NaN();

        return pi / (sinpix * exp(lgamma_positive(1.0 - x)));
    }

} // namespace detail::_f64_impl

// exp / log
[[nodiscard]] BL_FORCE_INLINE constexpr double exp(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::exp(x),
        std::exp(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double exp2(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::exp2(x),
        std::exp2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double expm1(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::expm1(x),
        std::expm1(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log_as_double(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::log(x),
        std::log(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::log(x),
        std::log(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log2(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::log2(x),
        std::log2(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log10(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::log10(x),
        std::log10(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double log1p(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::log1p(x),
        std::log1p(x)
    );
}


// roots
[[nodiscard]] BL_FORCE_INLINE constexpr double sqrt(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::sqrt(x),
        std::sqrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cbrt(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::cbrt(x),
        std::cbrt(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double hypot(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::hypot(x, y),
        std::hypot(x, y)
    );
}


// pow
[[nodiscard]] BL_FORCE_INLINE constexpr double pow(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::pow(x, y),
        std::pow(x, y)
    );
}


// trig
[[nodiscard]] BL_FORCE_INLINE constexpr double sin(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::sin(x),
        std::sin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cos(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::cos(x),
        std::cos(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(double x, double& s_out, double& c_out) noexcept
{
    s_out = bl::sin(x);
    c_out = bl::cos(x);
    return bl::isfinite(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::tan(x),
        std::tan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atan(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::atan(x),
        std::atan(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atan2(double y, double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::atan2(y, x),
        std::atan2(y, x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double asin(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        atan2(x, sqrt((1.0 - x) * (1.0 + x))),
        std::asin(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double acos(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        2.0 * atan2(sqrt(1.0 - x), sqrt(1.0 + x)),
        std::acos(x)
    );
}

template<class Vec> requires detail::fp::sincos_vector_assignable<Vec, double>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(double x, Vec& out)
{
    double s_out{};
    double c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    detail::fp::assign_sincos_vector(out, s_out, c_out);
    return ok;
}

template<class Value> requires std::same_as<std::remove_cvref_t<Value>, double>
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<double> sincos(double x)
{
    double s_out{};
    double c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    return detail::fp::make_sincos_result(s_out, c_out, ok);
}


// hyperbolic
[[nodiscard]] BL_FORCE_INLINE constexpr double sinh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::sinh(x),
        std::sinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double cosh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::cosh(x),
        std::cosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tanh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::tanh(x),
        std::tanh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double asinh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::asinh(x),
        std::asinh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double acosh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::acosh(x),
        std::acosh(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double atanh(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::atanh(x),
        std::atanh(x)
    );
}


// erf / erfc
[[nodiscard]] BL_FORCE_INLINE constexpr double erf(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::erf(x),
        std::erf(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double erfc(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::erfc(x),
        std::erfc(x)
    );
}


// gamma
[[nodiscard]] BL_FORCE_INLINE constexpr double lgamma(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::lgamma(x),
        std::lgamma(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double tgamma(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::tgamma(x),
        std::tgamma(x)
    );
}

} // namespace bl

#endif
