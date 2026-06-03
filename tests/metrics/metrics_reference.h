#ifndef FLTX_TESTS_METRICS_REFERENCE_INCLUDED
#define FLTX_TESTS_METRICS_REFERENCE_INCLUDED

#include <limits>
#include <string_view>

#include <boost/multiprecision/cpp_double_fp.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <qd/dd_real.h>
#include <qd/qd_real.h>

#include <fltx/f128_math.h>
#include <fltx/f256_math.h>

#include "metrics_types.h"

namespace bl::test::metrics
{
    template<class Float>
    struct reference_types;

    template<>
    struct reference_types<bl::f128>
    {
        static constexpr int oracle_digits10 = 200;
        static constexpr int qdpp_competitor_digits10 = 31;
        static constexpr int extra_competitor_digits10 = qdpp_competitor_digits10;

        using fltx_type = bl::f128;
        using perfect_ref = boost::multiprecision::number<
            boost::multiprecision::mpfr_float_backend<oracle_digits10>,
            boost::multiprecision::et_off>;

        using competitor_ref       = boost::multiprecision::cpp_double_double;
        using extra_competitor_ref = dd_real;

        static constexpr precision_type precision = precision_type::f128;
        static constexpr std::string_view precision_name = "f128";
        static constexpr std::string_view competitor_name = "boost::multiprecision::cpp_double_double";
        static constexpr std::string_view extra_competitor_name = "qdpp (dd_real)";
    };

    template<>
    struct reference_types<bl::f256>
    {
        static constexpr int oracle_digits10 = 1000;
        static constexpr int competitor_digits10 = 64;
        static constexpr int qdpp_competitor_digits10 = 62;
        static constexpr int extra_competitor_digits10 = qdpp_competitor_digits10;

        using fltx_type = bl::f256;
        using perfect_ref = boost::multiprecision::number<
            boost::multiprecision::mpfr_float_backend<oracle_digits10>,
            boost::multiprecision::et_off>;

        using competitor_ref       = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<competitor_digits10>, boost::multiprecision::et_off>;
        using extra_competitor_ref = qd_real;

        static constexpr precision_type precision = precision_type::f256;
        static constexpr std::string_view precision_name = "f256";
        static constexpr std::string_view competitor_name = "boost::multiprecision::mpfr_float_backend<64>";
        static constexpr std::string_view extra_competitor_name = "qdpp (qd_real)";
    };
}

#endif
