#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include <f128_math.h>
#include <f256_math.h>

namespace
{
    using f128_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<192>,
        boost::multiprecision::et_off>;

    using f256_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<320>,
        boost::multiprecision::et_off>;

    enum class unary_op
    {
        sqrt,
        exp,
        expm1,
        log,
        log1p,
        sin,
        cos,
        tan,
        atan,
        asin,
        acos,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        cbrt,
        erf,
        erfc,
        lgamma,
        tgamma
    };

    enum class binary_op
    {
        add,
        subtract,
        multiply,
        divide,
        fmod,
        remainder,
        pow,
        atan2,
        hypot
    };

    template<class Float>
    struct unary_case
    {
        const char* label = "";
        unary_op op = unary_op::sqrt;
        Float input{};
        Float result{};
    };

    template<class Float>
    struct binary_case
    {
        const char* label = "";
        binary_op op = binary_op::add;
        Float lhs{};
        Float rhs{};
        Float result{};
    };

    [[nodiscard]] constexpr bool real_constexpr_canary()
    {
        return bl::is_constant_evaluated();
    }

    static_assert(real_constexpr_canary());

    template<class Float>
    [[nodiscard]] constexpr Float fp(double value)
    {
        return Float{ value };
    }

    template<class Float>
    [[nodiscard]] constexpr Float q(double numerator, double denominator)
    {
        return fp<Float>(numerator) / fp<Float>(denominator);
    }

    template<class Float>
    [[nodiscard]] constexpr Float mixed(double whole, double numerator, double denominator)
    {
        return fp<Float>(whole) + q<Float>(numerator, denominator);
    }

    template<class Float>
    [[nodiscard]] constexpr Float apply_constexpr(unary_op op, Float input)
    {
        switch (op)
        {
        case unary_op::sqrt:   return bl::sqrt(input);
        case unary_op::exp:    return bl::exp(input);
        case unary_op::expm1:  return bl::expm1(input);
        case unary_op::log:    return bl::log(input);
        case unary_op::log1p:  return bl::log1p(input);
        case unary_op::sin:    return bl::sin(input);
        case unary_op::cos:    return bl::cos(input);
        case unary_op::tan:    return bl::tan(input);
        case unary_op::atan:   return bl::atan(input);
        case unary_op::asin:   return bl::asin(input);
        case unary_op::acos:   return bl::acos(input);
        case unary_op::sinh:   return bl::sinh(input);
        case unary_op::cosh:   return bl::cosh(input);
        case unary_op::tanh:   return bl::tanh(input);
        case unary_op::asinh:  return bl::asinh(input);
        case unary_op::acosh:  return bl::acosh(input);
        case unary_op::atanh:  return bl::atanh(input);
        case unary_op::cbrt:   return bl::cbrt(input);
        case unary_op::erf:    return bl::erf(input);
        case unary_op::erfc:   return bl::erfc(input);
        case unary_op::lgamma: return bl::lgamma(input);
        case unary_op::tgamma: return bl::tgamma(input);
        }

        return std::numeric_limits<Float>::quiet_NaN();
    }

    template<class Float>
    [[nodiscard]] constexpr Float apply_constexpr(binary_op op, Float lhs, Float rhs)
    {
        switch (op)
        {
        case binary_op::add:       return lhs + rhs;
        case binary_op::subtract:  return lhs - rhs;
        case binary_op::multiply:  return lhs * rhs;
        case binary_op::divide:    return lhs / rhs;
        case binary_op::fmod:      return bl::fmod(lhs, rhs);
        case binary_op::remainder: return bl::remainder(lhs, rhs);
        case binary_op::pow:       return bl::pow(lhs, rhs);
        case binary_op::atan2:     return bl::atan2(lhs, rhs);
        case binary_op::hypot:     return bl::hypot(lhs, rhs);
        }

        return std::numeric_limits<Float>::quiet_NaN();
    }

    template<class Float>
    [[nodiscard]] constexpr unary_case<Float> make_unary_case(const char* label, unary_op op, Float input)
    {
        return { label, op, input, apply_constexpr(op, input) };
    }

    template<class Float>
    [[nodiscard]] constexpr binary_case<Float> make_binary_case(const char* label, binary_op op, Float lhs, Float rhs)
    {
        return { label, op, lhs, rhs, apply_constexpr(op, lhs, rhs) };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_core_unary_cases()
    {
        return std::array{
            make_unary_case<Float>("sqrt tiny scaled", unary_op::sqrt, bl::ldexp(mixed<Float>(1.0, 1.0, 3.0), -80)),
            make_unary_case<Float>("sqrt two", unary_op::sqrt, fp<Float>(2.0)),
            make_unary_case<Float>("sqrt large scaled", unary_op::sqrt, bl::ldexp(mixed<Float>(1.0, 1.0, 5.0), 180)),
            make_unary_case<Float>("exp negative", unary_op::exp, q<Float>(-5.0, 4.0)),
            make_unary_case<Float>("exp positive", unary_op::exp, q<Float>(3.0, 2.0)),
            make_unary_case<Float>("expm1 moderate negative", unary_op::expm1, q<Float>(-1.0, 2.0)),
            make_unary_case<Float>("expm1 moderate positive", unary_op::expm1, q<Float>(1.0, 4.0)),
            make_unary_case<Float>("log subunit", unary_op::log, q<Float>(1.0, 8.0)),
            make_unary_case<Float>("log mixed", unary_op::log, mixed<Float>(1.0, 1.0, 7.0)),
            make_unary_case<Float>("log1p negative", unary_op::log1p, q<Float>(-3.0, 4.0)),
            make_unary_case<Float>("log1p positive", unary_op::log1p, q<Float>(5.0, 4.0)),
            make_unary_case<Float>("sin reduced negative", unary_op::sin, q<Float>(-3.0, 4.0)),
            make_unary_case<Float>("sin reduced positive", unary_op::sin, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("cos reduced negative", unary_op::cos, q<Float>(-3.0, 4.0)),
            make_unary_case<Float>("cos reduced positive", unary_op::cos, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("tan reduced", unary_op::tan, q<Float>(1.0, 4.0)),
            make_unary_case<Float>("atan negative", unary_op::atan, q<Float>(-5.0, 8.0)),
            make_unary_case<Float>("atan positive", unary_op::atan, q<Float>(3.0, 4.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_inverse_unary_cases()
    {
        return std::array{
            make_unary_case<Float>("asin negative", unary_op::asin, q<Float>(-1.0, 4.0)),
            make_unary_case<Float>("asin positive", unary_op::asin, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("acos negative", unary_op::acos, q<Float>(-1.0, 4.0)),
            make_unary_case<Float>("acos positive", unary_op::acos, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("asinh negative", unary_op::asinh, q<Float>(-5.0, 4.0)),
            make_unary_case<Float>("asinh positive", unary_op::asinh, q<Float>(3.0, 4.0)),
            make_unary_case<Float>("acosh near one", unary_op::acosh, mixed<Float>(1.0, 1.0, 16.0)),
            make_unary_case<Float>("acosh moderate", unary_op::acosh, q<Float>(5.0, 2.0)),
            make_unary_case<Float>("atanh negative", unary_op::atanh, q<Float>(-1.0, 4.0)),
            make_unary_case<Float>("atanh positive", unary_op::atanh, q<Float>(1.0, 2.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_hyperbolic_unary_cases()
    {
        return std::array{
            make_unary_case<Float>("sinh negative", unary_op::sinh, q<Float>(-3.0, 4.0)),
            make_unary_case<Float>("sinh positive", unary_op::sinh, q<Float>(5.0, 4.0)),
            make_unary_case<Float>("cosh small", unary_op::cosh, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("cosh moderate", unary_op::cosh, q<Float>(5.0, 4.0)),
            make_unary_case<Float>("tanh negative", unary_op::tanh, q<Float>(-3.0, 4.0)),
            make_unary_case<Float>("tanh positive", unary_op::tanh, q<Float>(5.0, 4.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_special_unary_cases()
    {
        return std::array{
            make_unary_case<Float>("cbrt negative", unary_op::cbrt, fp<Float>(-27.0)),
            make_unary_case<Float>("cbrt positive", unary_op::cbrt, q<Float>(5.0, 8.0)),
            make_unary_case<Float>("erf negative", unary_op::erf, q<Float>(-5.0, 4.0)),
            make_unary_case<Float>("erf positive", unary_op::erf, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("erfc negative", unary_op::erfc, q<Float>(-5.0, 4.0)),
            make_unary_case<Float>("erfc positive", unary_op::erfc, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("lgamma small", unary_op::lgamma, q<Float>(1.0, 8.0)),
            make_unary_case<Float>("lgamma moderate", unary_op::lgamma, q<Float>(15.0, 4.0)),
            make_unary_case<Float>("tgamma half", unary_op::tgamma, q<Float>(1.0, 2.0)),
            make_unary_case<Float>("tgamma moderate", unary_op::tgamma, q<Float>(9.0, 4.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_arithmetic_binary_cases()
    {
        return std::array{
            make_binary_case<Float>("add cancellation", binary_op::add, mixed<Float>(1.0, 1.0, 7.0), q<Float>(-8.0, 7.0)),
            make_binary_case<Float>("subtract close", binary_op::subtract, mixed<Float>(1.0, 1.0, 4096.0), fp<Float>(1.0)),
            make_binary_case<Float>("multiply mixed", binary_op::multiply, mixed<Float>(3.0, 1.0, 7.0), q<Float>(-11.0, 13.0)),
            make_binary_case<Float>("divide mixed", binary_op::divide, mixed<Float>(7.0, 2.0, 11.0), mixed<Float>(3.0, 1.0, 5.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_reduction_binary_cases()
    {
        return std::array{
            make_binary_case<Float>("fmod simple", binary_op::fmod, mixed<Float>(5.0, 1.0, 4.0), fp<Float>(2.0)),
            make_binary_case<Float>("fmod fractional divisor", binary_op::fmod, mixed<Float>(17.0, 1.0, 7.0), mixed<Float>(3.0, 1.0, 11.0)),
            make_binary_case<Float>("fmod scaled quotient", binary_op::fmod, bl::ldexp(mixed<Float>(1.0, 3.0, 17.0), 48), mixed<Float>(5.0, 1.0, 8.0)),
            make_binary_case<Float>("remainder simple", binary_op::remainder, mixed<Float>(17.0, 1.0, 7.0), mixed<Float>(3.0, 1.0, 11.0))
        };
    }

    template<class Float>
    [[nodiscard]] constexpr auto make_transcendental_binary_cases()
    {
        return std::array{
            make_binary_case<Float>("pow fractional", binary_op::pow, mixed<Float>(1.0, 1.0, 4.0), mixed<Float>(1.0, 1.0, 8.0)),
            make_binary_case<Float>("pow reciprocal", binary_op::pow, q<Float>(5.0, 2.0), q<Float>(-3.0, 2.0)),
            make_binary_case<Float>("atan2 quadrant one", binary_op::atan2, q<Float>(1.0, 2.0), mixed<Float>(1.0, 1.0, 4.0)),
            make_binary_case<Float>("atan2 quadrant three", binary_op::atan2, q<Float>(-3.0, 4.0), q<Float>(-5.0, 4.0)),
            make_binary_case<Float>("hypot 3 4", binary_op::hypot, fp<Float>(3.0), fp<Float>(4.0)),
            make_binary_case<Float>("hypot fractional", binary_op::hypot, q<Float>(5.0, 8.0), q<Float>(7.0, 9.0))
        };
    }

    constexpr auto f128_core_unary_constexpr_cases = make_core_unary_cases<bl::f128_s>();
    constexpr auto f128_inverse_unary_constexpr_cases = make_inverse_unary_cases<bl::f128_s>();
    constexpr auto f128_hyperbolic_unary_constexpr_cases = make_hyperbolic_unary_cases<bl::f128_s>();
    constexpr auto f128_special_unary_constexpr_cases = make_special_unary_cases<bl::f128_s>();
    constexpr auto f128_arithmetic_binary_constexpr_cases = make_arithmetic_binary_cases<bl::f128_s>();
    constexpr auto f128_reduction_binary_constexpr_cases = make_reduction_binary_cases<bl::f128_s>();
    constexpr auto f128_transcendental_binary_constexpr_cases = make_transcendental_binary_cases<bl::f128_s>();

    constexpr auto f256_core_unary_constexpr_cases = make_core_unary_cases<bl::f256_s>();
    constexpr auto f256_inverse_unary_constexpr_cases = make_inverse_unary_cases<bl::f256_s>();
    constexpr auto f256_hyperbolic_unary_constexpr_cases = make_hyperbolic_unary_cases<bl::f256_s>();
    constexpr auto f256_special_unary_constexpr_cases = make_special_unary_cases<bl::f256_s>();
    constexpr auto f256_arithmetic_binary_constexpr_cases = make_arithmetic_binary_cases<bl::f256_s>();
    constexpr auto f256_reduction_binary_constexpr_cases = make_reduction_binary_cases<bl::f256_s>();
    constexpr auto f256_transcendental_binary_constexpr_cases = make_transcendental_binary_cases<bl::f256_s>();

    static_assert(f128_core_unary_constexpr_cases.size() > 0);
    static_assert(f256_core_unary_constexpr_cases.size() == f128_core_unary_constexpr_cases.size());

    template<class Ref>
    [[nodiscard]] Ref abs_ref(const Ref& value)
    {
        return value < 0 ? -value : value;
    }

    template<class Ref>
    [[nodiscard]] Ref trunc_ref(const Ref& value)
    {
        return value < 0 ? boost::multiprecision::ceil(value) : boost::multiprecision::floor(value);
    }

    template<class Ref>
    [[nodiscard]] Ref round_to_even_ref(const Ref& value)
    {
        Ref rounded = boost::multiprecision::floor(value + Ref{ "0.5" });
        if ((rounded - value) == Ref{ "0.5" })
        {
            const Ref half = boost::multiprecision::floor(rounded / Ref{ 2 });
            if (half * Ref{ 2 } != rounded)
                rounded -= 1;
        }
        return rounded;
    }

    template<class Ref>
    [[nodiscard]] Ref fmod_ref(const Ref& lhs, const Ref& rhs)
    {
        return lhs - trunc_ref(lhs / rhs) * rhs;
    }

    template<class Ref>
    [[nodiscard]] Ref remainder_ref(const Ref& lhs, const Ref& rhs)
    {
        return lhs - round_to_even_ref(lhs / rhs) * rhs;
    }

    template<class Ref>
    [[nodiscard]] Ref cbrt_ref(const Ref& value)
    {
        if (value < 0)
            return -boost::multiprecision::cbrt(-value);
        return boost::multiprecision::cbrt(value);
    }

    template<class Ref>
    [[nodiscard]] Ref apply_ref(unary_op op, const Ref& input)
    {
        switch (op)
        {
        case unary_op::sqrt:   return boost::multiprecision::sqrt(input);
        case unary_op::exp:    return boost::multiprecision::exp(input);
        case unary_op::expm1:  return boost::multiprecision::expm1(input);
        case unary_op::log:    return boost::multiprecision::log(input);
        case unary_op::log1p:  return boost::multiprecision::log1p(input);
        case unary_op::sin:    return boost::multiprecision::sin(input);
        case unary_op::cos:    return boost::multiprecision::cos(input);
        case unary_op::tan:    return boost::multiprecision::tan(input);
        case unary_op::atan:   return boost::multiprecision::atan(input);
        case unary_op::asin:   return boost::multiprecision::asin(input);
        case unary_op::acos:   return boost::multiprecision::acos(input);
        case unary_op::sinh:   return boost::multiprecision::sinh(input);
        case unary_op::cosh:   return boost::multiprecision::cosh(input);
        case unary_op::tanh:   return boost::multiprecision::tanh(input);
        case unary_op::asinh:  return boost::multiprecision::asinh(input);
        case unary_op::acosh:  return boost::multiprecision::acosh(input);
        case unary_op::atanh:  return boost::multiprecision::atanh(input);
        case unary_op::cbrt:   return cbrt_ref(input);
        case unary_op::erf:    return boost::multiprecision::erf(input);
        case unary_op::erfc:   return boost::multiprecision::erfc(input);
        case unary_op::lgamma: return boost::multiprecision::lgamma(input);
        case unary_op::tgamma: return boost::multiprecision::tgamma(input);
        }

        return std::numeric_limits<Ref>::quiet_NaN();
    }

    template<class Ref>
    [[nodiscard]] Ref apply_ref(binary_op op, const Ref& lhs, const Ref& rhs)
    {
        switch (op)
        {
        case binary_op::add:       return lhs + rhs;
        case binary_op::subtract:  return lhs - rhs;
        case binary_op::multiply:  return lhs * rhs;
        case binary_op::divide:    return lhs / rhs;
        case binary_op::fmod:      return fmod_ref(lhs, rhs);
        case binary_op::remainder: return remainder_ref(lhs, rhs);
        case binary_op::pow:       return boost::multiprecision::pow(lhs, rhs);
        case binary_op::atan2:     return boost::multiprecision::atan2(lhs, rhs);
        case binary_op::hypot:     return boost::multiprecision::sqrt(lhs * lhs + rhs * rhs);
        }

        return std::numeric_limits<Ref>::quiet_NaN();
    }

    template<class Ref>
    [[nodiscard]] Ref to_ref_exact(const bl::f128_s& value)
    {
        Ref sum = 0;
        sum += Ref{ value.hi };
        sum += Ref{ value.lo };
        return sum;
    }

    template<class Ref>
    [[nodiscard]] Ref to_ref_exact(const bl::f256_s& value)
    {
        Ref sum = 0;
        sum += Ref{ value.x0 };
        sum += Ref{ value.x1 };
        sum += Ref{ value.x2 };
        sum += Ref{ value.x3 };
        return sum;
    }

    [[nodiscard]] std::string describe(const bl::f128_s& value)
    {
        std::ostringstream out;
        out << std::hexfloat << "{ " << value.hi << ", " << value.lo << " }";
        return out.str();
    }

    [[nodiscard]] std::string describe(const bl::f256_s& value)
    {
        std::ostringstream out;
        out << std::hexfloat
            << "{ " << value.x0 << ", " << value.x1 << ", " << value.x2 << ", " << value.x3 << " }";
        return out.str();
    }

    template<class Ref>
    [[nodiscard]] std::string describe_ref(const Ref& value)
    {
        std::ostringstream out;
        out << std::scientific << std::setprecision(std::numeric_limits<Ref>::digits10 + 8) << value;
        return out.str();
    }

    template<class Ref>
    [[nodiscard]] Ref decimal_epsilon(int digits)
    {
        Ref value = 1;
        for (int i = 0; i < digits; ++i)
            value /= 10;
        return value;
    }

    template<class Float>
    [[nodiscard]] constexpr int checked_digits()
    {
        if constexpr (std::is_same_v<Float, bl::f128_s>)
            return std::numeric_limits<Float>::digits10 - 2;
        else
            return std::numeric_limits<Float>::digits10 - 4;
    }

    template<class Ref>
    [[nodiscard]] Ref tolerance_for(const Ref& expected, int digits)
    {
        Ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;
        return decimal_epsilon<Ref>(digits) * scale;
    }

    template<class Float>
    struct ref_for;

    template<>
    struct ref_for<bl::f128_s>
    {
        using type = f128_ref;
        static constexpr const char* label = "f128";
    };

    template<>
    struct ref_for<bl::f256_s>
    {
        using type = f256_ref;
        static constexpr const char* label = "f256";
    };

    template<class Float, std::size_t N>
    void check_unary_constexpr_corpus(const std::array<unary_case<Float>, N>& cases)
    {
        using ref = typename ref_for<Float>::type;
        constexpr int digits = checked_digits<Float>();

        for (const auto& test : cases)
        {
            const ref input = to_ref_exact<ref>(test.input);
            const ref got = to_ref_exact<ref>(test.result);
            const ref expected = apply_ref(test.op, input);
            const ref diff = abs_ref(got - expected);
            const ref tolerance = tolerance_for(expected, digits);

            INFO("type=" << ref_for<Float>::label);
            INFO("case=" << test.label);
            INFO("input=" << describe(test.input));
            INFO("constexpr result=" << describe(test.result));
            INFO("expected=" << describe_ref(expected));
            INFO("got=" << describe_ref(got));
            INFO("diff=" << describe_ref(diff));
            INFO("tolerance=" << describe_ref(tolerance));
            CHECK(diff <= tolerance);
        }
    }

    template<class Float, std::size_t N>
    void check_binary_constexpr_corpus(const std::array<binary_case<Float>, N>& cases)
    {
        using ref = typename ref_for<Float>::type;
        constexpr int digits = checked_digits<Float>();

        for (const auto& test : cases)
        {
            const ref lhs = to_ref_exact<ref>(test.lhs);
            const ref rhs = to_ref_exact<ref>(test.rhs);
            const ref got = to_ref_exact<ref>(test.result);
            const ref expected = apply_ref(test.op, lhs, rhs);
            const ref diff = abs_ref(got - expected);
            const ref tolerance = tolerance_for(expected, digits);

            INFO("type=" << ref_for<Float>::label);
            INFO("case=" << test.label);
            INFO("lhs=" << describe(test.lhs));
            INFO("rhs=" << describe(test.rhs));
            INFO("constexpr result=" << describe(test.result));
            INFO("expected=" << describe_ref(expected));
            INFO("got=" << describe_ref(got));
            INFO("diff=" << describe_ref(diff));
            INFO("tolerance=" << describe_ref(tolerance));
            CHECK(diff <= tolerance);
        }
    }
}

TEST_CASE("real constexpr f128 libm corpus matches MPFR", "[fltx][constexpr][mpfr][f128]")
{
    check_unary_constexpr_corpus(f128_core_unary_constexpr_cases);
    check_unary_constexpr_corpus(f128_inverse_unary_constexpr_cases);
    check_unary_constexpr_corpus(f128_hyperbolic_unary_constexpr_cases);
    check_unary_constexpr_corpus(f128_special_unary_constexpr_cases);
    check_binary_constexpr_corpus(f128_arithmetic_binary_constexpr_cases);
    check_binary_constexpr_corpus(f128_reduction_binary_constexpr_cases);
    check_binary_constexpr_corpus(f128_transcendental_binary_constexpr_cases);
}

TEST_CASE("real constexpr f256 libm corpus matches MPFR", "[fltx][constexpr][mpfr][f256]")
{
    check_unary_constexpr_corpus(f256_core_unary_constexpr_cases);
    check_unary_constexpr_corpus(f256_inverse_unary_constexpr_cases);
    check_unary_constexpr_corpus(f256_hyperbolic_unary_constexpr_cases);
    check_unary_constexpr_corpus(f256_special_unary_constexpr_cases);
    check_binary_constexpr_corpus(f256_arithmetic_binary_constexpr_cases);
    check_binary_constexpr_corpus(f256_reduction_binary_constexpr_cases);
    check_binary_constexpr_corpus(f256_transcendental_binary_constexpr_cases);
}
