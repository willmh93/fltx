#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <boost/multiprecision/mpfr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fltx.h>

namespace
{
    template<int Digits10>
    using mpfr_number = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<Digits10>
    >;

    struct scan_result
    {
        int mpfr_digits10 = 0;
        long double matching_digits = 0.0L;
        bool found = false;
    };

    template<class T>
    std::string precise_string(const T& value)
    {
        constexpr int precision = std::numeric_limits<T>::max_digits10 > 0
            ? std::numeric_limits<T>::max_digits10
            : std::numeric_limits<T>::digits10 + 8;

        std::ostringstream out;
        out << std::setprecision(precision) << value;
        return out.str();
    }

    template<class Ref, class T>
    Ref parse_as_ref(const T& value)
    {
        return Ref{ precise_string(value) };
    }

    template<class Ref>
    long double relative_matching_digits(const Ref& candidate, const Ref& reference)
    {
        using boost::multiprecision::abs;

        Ref error = abs(candidate - reference);
        if (error == 0)
            return std::numeric_limits<long double>::infinity();

        Ref scale = abs(reference);
        if (scale != 0)
            error /= scale;

        const long double relative_error = error.template convert_to<long double>();
        if (relative_error == 0.0L)
            return std::numeric_limits<long double>::infinity();

        const long double digits = -std::log10(relative_error);
        return digits > 0.0L ? digits : 0.0L;
    }

    struct sqrt_op
    {
        static constexpr const char* name = "sqrt";

        template<class T>
        static T input_a()
        {
            return T{ 1 } + T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T apply()
        {
            using std::sqrt;
            return sqrt(input_a<T>());
        }
    };

    struct sin_op
    {
        static constexpr const char* name = "sin";

        template<class T>
        static T input_a()
        {
            return T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 } + T{ 3 } / T{ 101 };
        }

        template<class T>
        static T apply()
        {
            using std::sin;
            return sin(input_a<T>());
        }
    };

    struct exp_op
    {
        static constexpr const char* name = "exp";

        template<class T>
        static T input_a()
        {
            return T{ 1 } / T{ 7 } - T{ 2 } / T{ 37 };
        }

        template<class T>
        static T apply()
        {
            using std::exp;
            return exp(input_a<T>());
        }
    };

    struct log_op
    {
        static constexpr const char* name = "log";

        template<class T>
        static T input_a()
        {
            return T{ 1 } + T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T apply()
        {
            using std::log;
            return log(input_a<T>());
        }
    };

    struct pow_op
    {
        static constexpr const char* name = "pow";

        template<class T>
        static T input_a()
        {
            return T{ 1 } + T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T input_b()
        {
            return T{ 1 } + T{ 3 } / T{ 19 };
        }

        template<class T>
        static T apply()
        {
            using std::pow;
            return pow(input_a<T>(), input_b<T>());
        }
    };

    struct atan2_op
    {
        static constexpr const char* name = "atan2";

        template<class T>
        static T input_a()
        {
            return T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T input_b()
        {
            return T{ 1 } + T{ 3 } / T{ 19 };
        }

        template<class T>
        static T apply()
        {
            using std::atan2;
            return atan2(input_a<T>(), input_b<T>());
        }
    };

    struct erf_op
    {
        static constexpr const char* name = "erf";

        template<class T>
        static T input_a()
        {
            return T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T apply()
        {
            using std::erf;
            return erf(input_a<T>());
        }
    };

    struct lgamma_op
    {
        static constexpr const char* name = "lgamma";

        template<class T>
        static T input_a()
        {
            return T{ 2 } + T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T apply()
        {
            using std::lgamma;
            return lgamma(input_a<T>());
        }
    };

    struct fmod_op
    {
        static constexpr const char* name = "fmod";

        template<class T>
        static T input_a()
        {
            return T{ 17 } + T{ 1 } / T{ 7 } + T{ 2 } / T{ 37 };
        }

        template<class T>
        static T input_b()
        {
            return T{ 3 } + T{ 1 } / T{ 11 };
        }

        template<class T>
        static T apply()
        {
            using std::fmod;
            return fmod(input_a<T>(), input_b<T>());
        }
    };

    template<class Op, int MaxDigits10, int N, int RefDigits10>
    scan_result scan_mpfr_digits10(long double target_matching_digits)
    {
        using mpfr_t = mpfr_number<N>;
        using ref_t = mpfr_number<RefDigits10>;

        const ref_t reference_y = Op::template apply<ref_t>();
        const ref_t candidate_y = ref_t{ Op::template apply<mpfr_t>() };

        const long double mpfr_matching_digits =
            relative_matching_digits(candidate_y, reference_y);

        if (mpfr_matching_digits + 0.25L >= target_matching_digits)
            return { N, mpfr_matching_digits, true };

        if constexpr (N < MaxDigits10)
        {
            return scan_mpfr_digits10<Op, MaxDigits10, N + 1, RefDigits10>(
                target_matching_digits
            );
        }
        else
        {
            return { N, mpfr_matching_digits, false };
        }
    }

    template<class Float, class Op>
    void check_precision_case(const char* float_name)
    {
        constexpr int max_digits10 = std::numeric_limits<Float>::digits10;
        constexpr int ref_digits10 = max_digits10 + 80;

        using ref_t = mpfr_number<ref_digits10>;

        const ref_t reference_y = Op::template apply<ref_t>();
        const ref_t fltx_y = parse_as_ref<ref_t>(Op::template apply<Float>());

        const long double fltx_matching_digits =
            relative_matching_digits(fltx_y, reference_y);

        const scan_result result =
            scan_mpfr_digits10<Op, max_digits10, 2, ref_digits10>(
                fltx_matching_digits
            );

        const long double matching_gap =
            fltx_matching_digits - result.matching_digits;

        std::cout
            << float_name << " / " << Op::name << '\n'
            << "  std::numeric_limits<>::digits10:    " << max_digits10 << '\n'
            << "  fltx matching digits vs high MPFR:  "
            << std::fixed << std::setprecision(2) << fltx_matching_digits << '\n';

        if (result.found)
        {
            std::cout
                << "  first mpfr_float_backend<N> match: N = "
                << result.mpfr_digits10 << '\n'
                << "  MPFR matching digits at that N:    "
                << result.matching_digits << "\n\n";
        }
        else
        {
            std::cout
                << "  no MPFR N <= digits10 matched fltx precision\n"
                << "  best scanned N:                    "
                << result.mpfr_digits10 << '\n'
                << "  MPFR matching digits at best N:    "
                << result.matching_digits << "\n\n";
        }

        INFO("float type: " << float_name);
        INFO("operation: " << Op::name);
        INFO("fltx matching digits: " << fltx_matching_digits);
        INFO("best mpfr N: " << result.mpfr_digits10);
        INFO("best mpfr matching digits: " << result.matching_digits);
        INFO("matching gap: " << matching_gap);

        CHECK(result.mpfr_digits10 <= max_digits10);
        CHECK(matching_gap <= 1.5L);
    }

    template<class Float>
    void check_representative_precision_sweep(const char* float_name)
    {
        check_precision_case<Float, sqrt_op>(float_name);
        check_precision_case<Float, sin_op>(float_name);
        check_precision_case<Float, exp_op>(float_name);
        check_precision_case<Float, log_op>(float_name);
        check_precision_case<Float, pow_op>(float_name);
        check_precision_case<Float, atan2_op>(float_name);
        check_precision_case<Float, erf_op>(float_name);
        check_precision_case<Float, lgamma_op>(float_name);
        check_precision_case<Float, fmod_op>(float_name);
    }
}

TEST_CASE("MPFR benchmark precision is comparable to f128", "[fltx][mpfr][precision]")
{
    check_representative_precision_sweep<bl::f128>("bl::f128");
}

TEST_CASE("MPFR benchmark precision is comparable to f256", "[fltx][mpfr][precision]")
{
    check_representative_precision_sweep<bl::f256>("bl::f256");
}
