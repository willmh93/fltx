#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include <fltx/f128.h>

using big = boost::multiprecision::mpfr_float_50;

using namespace bl;

namespace
{
    using mpfr_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<192>,
        boost::multiprecision::et_off>;

    constexpr int checked_digits = std::numeric_limits<f128>::digits10 - 2;
    constexpr int printed_digits = std::numeric_limits<f128>::max_digits10;

    [[nodiscard]] mpfr_ref abs_ref(const mpfr_ref& value)
    {
        return value < 0 ? -value : value;
    }

    [[nodiscard]] mpfr_ref decimal_epsilon(int digits)
    {
        mpfr_ref epsilon = 1;
        for (int i = 0; i < digits; ++i)
            epsilon /= 10;
        return epsilon;
    }

    [[nodiscard]] std::string to_text(const f128& value)
    {
        return bl::to_string(value, printed_digits, false, true, false);
    }

    [[nodiscard]] std::string to_text(const mpfr_ref& value)
    {
        std::ostringstream out;
        out << std::setprecision(printed_digits + 20)
            << std::scientific
            << value;
        return out.str();
    }

    [[nodiscard]] mpfr_ref to_ref_exact(const f128& value)
    {
        mpfr_ref sum = 0;
        sum += mpfr_ref{ value.hi };
        sum += mpfr_ref{ value.lo };
        return sum;
    }

    [[nodiscard]] std::string to_text_double(double value)
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::scientific
            << value;
        return out.str();
    }

    [[nodiscard]] std::string to_text_double_hex(double value)
    {
        std::ostringstream out;
        out << std::hexfloat << value;
        return out.str();
    }

    [[nodiscard]] big random_finite_for_f128(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> exponent_dist(-80, 80);
        std::uniform_int_distribution<std::uint32_t> chunk_dist(0, 999999999);

        std::ostringstream mantissa_text;
        mantissa_text << (sign_dist(rng) != 0 ? "-0." : "0.");

        for (int i = 0; i < 4; ++i)
        {
            const std::uint32_t chunk = chunk_dist(rng);
            mantissa_text << std::setw(9) << std::setfill('0') << chunk;
        }

        big mantissa{ mantissa_text.str() };

        if (mantissa == 0)
            mantissa = big{ "0.5" };

        if (mantissa < 0)
            mantissa -= 0.5;
        else
            mantissa += 0.5;

        return ldexp(mantissa, exponent_dist(rng));
    }

    [[nodiscard]] std::string to_scientific_string(const big& value, int digits)
    {
        return value.str(digits, std::ios_base::scientific);
    }

    template<typename F128Op, typename RefOp>
    void check_binary_op(const char* op_name, const char* lhs_text, const char* rhs_text, F128Op&& f128_op, RefOp&& ref_op)
    {
        const f128 lhs = to_f128(lhs_text);
        const f128 rhs = to_f128(rhs_text);

        const f128 got = f128_op(lhs, rhs);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(to_ref_exact(lhs), to_ref_exact(rhs));

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = decimal_epsilon(checked_digits) * scale;
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(lhs_text);
        CAPTURE(rhs_text);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(got.hi));
        CAPTURE(to_text_double(got.lo));
        CAPTURE(to_text_double_hex(got.hi));
        CAPTURE(to_text_double_hex(got.lo));

        REQUIRE(diff <= tolerance);
    }

    template<typename F128Op, typename RefOp>
    void check_unary_op(const char* op_name, const char* input_text, F128Op&& f128_op, RefOp&& ref_op)
    {
        const f128 input = to_f128(input_text);

        const f128 got = f128_op(input);
        const mpfr_ref input_ref = to_ref_exact(input);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(input_ref);

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = decimal_epsilon(checked_digits) * scale;
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(input_text);
        CAPTURE(to_text(input));
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(input.hi));
        CAPTURE(to_text_double(input.lo));
        CAPTURE(to_text_double_hex(input.hi));
        CAPTURE(to_text_double_hex(input.lo));

        CAPTURE(to_text_double(got.hi));
        CAPTURE(to_text_double(got.lo));
        CAPTURE(to_text_double_hex(got.hi));
        CAPTURE(to_text_double_hex(got.lo));

        REQUIRE(diff <= tolerance);
    }

    template<typename F128Op, typename RefOp>
    void check_unary_op_with_tolerance(
        const char* op_name,
        const char* input_text,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance,
        F128Op&& f128_op,
        RefOp&& ref_op)
    {
        const f128 input = to_f128(input_text);

        const f128 got = f128_op(input);
        const mpfr_ref input_ref = to_ref_exact(input);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(input_ref);

        const mpfr_ref scale = abs_ref(expected);
        const mpfr_ref rel_based_tolerance = rel_tolerance * scale;
        mpfr_ref tolerance = abs_tolerance;
        if (rel_based_tolerance > tolerance)
            tolerance = rel_based_tolerance;

        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(input_text);
        CAPTURE(to_text(input));
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(to_text(abs_tolerance));
        CAPTURE(to_text(rel_tolerance));

        CAPTURE(to_text_double(input.hi));
        CAPTURE(to_text_double(input.lo));
        CAPTURE(to_text_double_hex(input.hi));
        CAPTURE(to_text_double_hex(input.lo));

        CAPTURE(to_text_double(got.hi));
        CAPTURE(to_text_double(got.lo));
        CAPTURE(to_text_double_hex(got.hi));
        CAPTURE(to_text_double_hex(got.lo));

        REQUIRE(diff <= tolerance);
    }

    [[nodiscard]] const mpfr_ref& pi_ref()
    {
        static const mpfr_ref value{
            "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
        };
        return value;
    }

    template<typename T>
    [[nodiscard]] std::string to_scientific_text(const T& value, int digits)
    {
        return value.str(digits, std::ios_base::scientific);
    }

    [[nodiscard]] mpfr_ref random_unit_interval_for_f128(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<std::uint32_t> chunk_dist(0, 999999999);

        std::ostringstream text;
        text << "0.";

        for (int i = 0; i < 4; ++i)
            text << std::setw(9) << std::setfill('0') << chunk_dist(rng);

        return mpfr_ref{ text.str() };
    }

    [[nodiscard]] mpfr_ref random_sine_kernel_argument_for_f128(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);

        mpfr_ref value = random_unit_interval_for_f128(rng) * (pi_ref() / 4);
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] mpfr_ref random_sine_reduction_argument_for_f128(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<long long> multiple_dist(-1000000LL, 1000000LL);

        const mpfr_ref half_pi = pi_ref() / 2;
        const mpfr_ref quarter_pi = pi_ref() / 4;

        mpfr_ref offset = random_unit_interval_for_f128(rng) * (quarter_pi * 2);
        offset -= quarter_pi;

        return mpfr_ref{ static_cast<std::int64_t>(multiple_dist(rng)) } * half_pi + offset;
    }

    void check_sin_case(
        const char* label,
        const mpfr_ref& input,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("label: " << label);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });
    }

    void check_cos_case(
        const char* label,
        const mpfr_ref& input,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("label: " << label);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });
    }

}

TEST_CASE("f128 matches MPFR for + - * /", "[fltx][f128][precision]")
{
    const std::array<std::pair<const char*, const char*>, 10> cases = {{
        { "1", "2" },
        { "1.25", "2.5" },
        { "-3.75", "2.125" },
        { "1.000000000000000000000000000001", "2.000000000000000000000000000002" },
        { "1234567890123456.125", "0.000000000000000000000000000125" },
        { "3.1415926535897932384626433832795", "2.7182818284590452353602874713527" },
        { "1e-30", "1e-12" },
        { "1e30", "1e-8" },
        { "-8.33333333333333333333333333333", "0.125" },
        { "0.333333333333333333333333333333", "7.000000000000000000000000000001" }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("add", lhs, rhs,
            [](const f128& a, const f128& b) { return a + b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a + b; });

        check_binary_op("subtract", lhs, rhs,
            [](const f128& a, const f128& b) { return a - b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a - b; });

        check_binary_op("multiply", lhs, rhs,
            [](const f128& a, const f128& b) { return a * b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a * b; });

        check_binary_op("divide", lhs, rhs,
            [](const f128& a, const f128& b) { return a / b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a / b; });
    }
}

TEST_CASE("f128 brute-force random arithmetic matches MPFR within tolerance", "[fltx][f128][precision]")
{
    //std::mt19937_64 rng{ 1ull };
    std::mt19937_64 rng{ std::random_device{}() };

    const int digits = printed_digits;
    const int count = 10000;

    std::cout << "f128 comparing: " << count << " random arithmetic cases...\n\n";

    for (int i = 0; i < count; ++i)
    {
        const big lhs_big = random_finite_for_f128(rng);
        const big rhs_big = random_finite_for_f128(rng);

        const std::string lhs_text = to_scientific_string(lhs_big, digits);
        const std::string rhs_text = to_scientific_string(rhs_big, digits);

        check_binary_op("add", lhs_text.c_str(), rhs_text.c_str(),
            [](const f128& a, const f128& b) { return a + b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a + b; });

        check_binary_op("subtract", lhs_text.c_str(), rhs_text.c_str(),
            [](const f128& a, const f128& b) { return a - b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a - b; });

        check_binary_op("multiply", lhs_text.c_str(), rhs_text.c_str(),
            [](const f128& a, const f128& b) { return a * b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a * b; });

        if (rhs_big != 0)
        {
            check_binary_op("divide", lhs_text.c_str(), rhs_text.c_str(),
                [](const f128& a, const f128& b) { return a / b; },
                [](const mpfr_ref& a, const mpfr_ref& b) { return a / b; });
        }
    }
}

TEST_CASE("f128 sin matches MPFR for fixed values", "[fltx][f128][precision][transcendental]")
{
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-20" };

    const mpfr_ref reduced_abs_tolerance{ "1e-31" };
    const mpfr_ref reduced_rel_tolerance{ "5e-30" };
    const mpfr_ref reduction_abs_tolerance{ "2e-26" };
    const mpfr_ref reduction_rel_tolerance{ "2e-25" };

    check_sin_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_over_2", half_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_minus_tiny", pi - tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_plus_tiny", pi + tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("two_pi_minus_tiny", two_pi - tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("two_pi_plus_tiny", two_pi + tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("large_decimal", mpfr_ref{ "1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("negative_large_decimal", mpfr_ref{ "-1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("million_pi_plus_offset", mpfr_ref{ "1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("negative_million_pi_plus_offset", mpfr_ref{ "-1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f128 sin matches MPFR on random reduced-range inputs", "[fltx][f128][precision][transcendental]")
{
    std::mt19937_64 rng{ 1ull };

    constexpr int count = 2000;
    const mpfr_ref abs_tolerance{ "1e-31" };
    const mpfr_ref rel_tolerance{ "5e-30" };
    std::cout << "f128 comparing: " << count << " random reduced-range sin cases...\n\n";

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_kernel_argument_for_f128(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });
    }
}

TEST_CASE("f128 sin matches MPFR on random range-reduced inputs", "[fltx][f128][precision][transcendental]")
{
    //std::mt19937_64 rng{ 1ull };
    std::mt19937_64 rng{ std::random_device{}() };

    constexpr int count = 2000;
    const mpfr_ref abs_tolerance{ "2e-26" };
    const mpfr_ref rel_tolerance{ "2e-25" };
    std::cout << "f128 comparing: " << count << " random range-reduced sin cases...\n\n";

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f128(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });
    }
}



TEST_CASE("f128 cos matches MPFR for fixed values", "[fltx][f128][precision][transcendental]")
{
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-20" };

    const mpfr_ref reduced_abs_tolerance{ "1e-31" };
    const mpfr_ref reduced_rel_tolerance{ "5e-30" };
    const mpfr_ref reduction_abs_tolerance{ "2e-26" };
    const mpfr_ref reduction_rel_tolerance{ "2e-25" };

    check_cos_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("pi_over_2", half_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("pi_minus_tiny", pi - tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("pi_plus_tiny", pi + tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("two_pi_minus_tiny", two_pi - tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("two_pi_plus_tiny", two_pi + tiny, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("large_decimal", mpfr_ref{ "1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("negative_large_decimal", mpfr_ref{ "-1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("million_pi_plus_offset", mpfr_ref{ "1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("negative_million_pi_plus_offset", mpfr_ref{ "-1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f128 cos matches MPFR on random reduced-range inputs", "[fltx][f128][precision][transcendental]")
{
    //std::mt19937_64 rng{ 1ull };
    std::mt19937_64 rng{ std::random_device{}() };

    constexpr int count = 2000;
    const mpfr_ref abs_tolerance{ "1e-31" };
    const mpfr_ref rel_tolerance{ "5e-30" };
    std::cout << "f128 comparing: " << count << " random reduced-range cos cases...\n\n";

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_kernel_argument_for_f128(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });
    }
}

TEST_CASE("f128 cos matches MPFR on random range-reduced inputs", "[fltx][f128][precision][transcendental]")
{
    //std::mt19937_64 rng{ 1ull };
    std::mt19937_64 rng{ std::random_device{}() };

    constexpr int count = 2000;
    const mpfr_ref abs_tolerance{ "2e-26" };
    const mpfr_ref rel_tolerance{ "2e-25" };
    std::cout << "f128 comparing: " << count << " random range-reduced cos cases...\n\n";

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f128(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 4);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f128& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });
    }
}
