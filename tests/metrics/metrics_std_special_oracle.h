#ifndef FLTX_TESTS_METRICS_STD_SPECIAL_ORACLE_INCLUDED
#define FLTX_TESTS_METRICS_STD_SPECIAL_ORACLE_INCLUDED

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "metrics_samples.h"

namespace bl::test::metrics
{
    [[nodiscard]] inline double sample_as_double(const value_sample& sample) noexcept
    {
        return sample.hi;
    }

    [[nodiscard]] inline bool special_trace_enabled() noexcept
    {
        const char* text = std::getenv("FLTX_SPECIAL_TRACE");
        return text != nullptr && text[0] != '\0' && !(text[0] == '0' && text[1] == '\0');
    }

    [[nodiscard]] inline double stdlib_unary_special_result(std::string_view operation, double x)
    {
        if (operation == "sqrt")
            return std::sqrt(x);
        if (operation == "cbrt")
            return std::cbrt(x);
        if (operation == "sin")
            return std::sin(x);
        if (operation == "cos")
            return std::cos(x);
        if (operation == "tan")
            return std::tan(x);
        if (operation == "atan")
            return std::atan(x);
        if (operation == "asin")
            return std::asin(x);
        if (operation == "acos")
            return std::acos(x);
        if (operation == "exp")
            return std::exp(x);
        if (operation == "exp2")
            return std::exp2(x);
        if (operation == "expm1")
            return std::expm1(x);
        if (operation == "log")
            return std::log(x);
        if (operation == "log2")
            return std::log2(x);
        if (operation == "log10")
            return std::log10(x);
        if (operation == "log1p")
            return std::log1p(x);
        if (operation == "sinh")
            return std::sinh(x);
        if (operation == "cosh")
            return std::cosh(x);
        if (operation == "tanh")
            return std::tanh(x);
        if (operation == "asinh")
            return std::asinh(x);
        if (operation == "acosh")
            return std::acosh(x);
        if (operation == "atanh")
            return std::atanh(x);
        if (operation == "fabs")
            return std::fabs(x);
        if (operation == "floor")
            return std::floor(x);
        if (operation == "ceil")
            return std::ceil(x);
        if (operation == "trunc")
            return std::trunc(x);
        if (operation == "round")
            return std::round(x);
        if (operation == "nearbyint")
            return std::nearbyint(x);
        if (operation == "rint")
            return std::rint(x);
        if (operation == "logb")
            return std::logb(x);
        if (operation == "modf")
        {
            double integer_part = 0.0;
            return std::modf(x, &integer_part);
        }
        if (operation == "erf")
            return std::erf(x);
        if (operation == "erfc")
            return std::erfc(x);
        if (operation == "lgamma")
            return std::lgamma(x);
        if (operation == "tgamma")
            return std::tgamma(x);

        throw std::invalid_argument("unknown unary std special oracle operation");
    }

    [[nodiscard]] inline double stdlib_binary_special_result(std::string_view operation, double x, double y)
    {
        if (operation == "add")
            return x + y;
        if (operation == "subtract")
            return x - y;
        if (operation == "multiply")
            return x * y;
        if (operation == "divide")
            return x / y;
        if (operation == "hypot")
            return std::hypot(x, y);
        if (operation == "atan2")
            return std::atan2(x, y);
        if (operation == "pow")
            return std::pow(x, y);
        if (operation == "fmod")
            return std::fmod(x, y);
        if (operation == "remainder")
            return std::remainder(x, y);
        if (operation == "remquo")
        {
            int quotient = 0;
            return std::remquo(x, y, &quotient);
        }
        if (operation == "fmin")
            return std::fmin(x, y);
        if (operation == "fmax")
            return std::fmax(x, y);
        if (operation == "fdim")
            return std::fdim(x, y);
        if (operation == "copysign")
            return std::copysign(x, y);
        if (operation == "nextafter")
            return std::nextafter(x, y);
        if (operation == "nexttoward")
            return std::nextafter(x, y);

        throw std::invalid_argument("unknown binary std special oracle operation");
    }

    [[nodiscard]] inline double stdlib_ternary_special_result(std::string_view operation, double x, double y, double z)
    {
        if (operation == "fma")
            return std::fma(x, y, z);

        throw std::invalid_argument("unknown ternary std special oracle operation");
    }

    [[nodiscard]] inline double stdlib_unary_int_special_result(std::string_view operation, double x, int n)
    {
        if (operation == "ldexp")
            return std::ldexp(x, n);
        if (operation == "scalbn")
            return std::scalbn(x, n);
        if (operation == "scalbln")
            return std::scalbln(x, static_cast<long>(n));

        throw std::invalid_argument("unknown unary-int std special oracle operation");
    }

    [[nodiscard]] inline long long stdlib_unary_integer_special_result(std::string_view operation, double x)
    {
        if (operation == "lround")
            return static_cast<long long>(std::lround(x));
        if (operation == "llround")
            return std::llround(x);
        if (operation == "lrint")
            return static_cast<long long>(std::lrint(x));
        if (operation == "llrint")
            return std::llrint(x);
        if (operation == "ilogb")
            return static_cast<long long>(std::ilogb(x));

        throw std::invalid_argument("unknown unary-integer std special oracle operation");
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_unary_integer_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        (void)reference;
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            long long expected = 0;
            try
            {
                expected = stdlib_unary_integer_special_result(
                    operation,
                    sample_as_double(samples[index].x));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const long long actual = static_cast<long long>(eval(values[index]));
                if (actual != expected)
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' integer mismatch: "
                                  << actual << " != " << expected << '\n';
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }
}

#endif
