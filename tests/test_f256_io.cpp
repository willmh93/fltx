#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include <fltx/f256.h>

using namespace bl;

namespace
{
    using mpfr_check = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<200>,
        boost::multiprecision::et_off>;

    using mpfr_random = boost::multiprecision::mpfr_float_100;

    constexpr int checked_digits = std::numeric_limits<f256>::digits10 - 4;
    constexpr int print_digits = std::numeric_limits<f256>::max_digits10;

    [[nodiscard]] mpfr_check abs_check(const mpfr_check& value)
    {
        return value < 0 ? -value : value;
    }

    [[nodiscard]] mpfr_check decimal_epsilon(int digits)
    {
        mpfr_check epsilon = 1;
        for (int i = 0; i < digits; ++i)
            epsilon /= 10;
        return epsilon;
    }

    [[nodiscard]] mpfr_check comparison_tolerance(const mpfr_check& expected)
    {
        mpfr_check scale = abs_check(expected);
        if (scale < 1)
            scale = 1;

        return decimal_epsilon(checked_digits) * scale;
    }

    [[nodiscard]] std::string to_text(const f256& value)
    {
        return bl::to_string(value, print_digits, false, true, false);
    }

    [[nodiscard]] std::string to_text(const mpfr_check& value)
    {
        std::ostringstream out;
        out << std::setprecision(print_digits + 20)
            << std::scientific
            << value;
        return out.str();
    }

    [[nodiscard]] mpfr_check to_ref(const char* text)
    {
        return mpfr_check{ std::string{text} };
    }

    [[nodiscard]] mpfr_check to_ref_exact(const f256& value)
    {
        mpfr_check sum = 0;
        sum += mpfr_check{ value.x0 };
        sum += mpfr_check{ value.x1 };
        sum += mpfr_check{ value.x2 };
        sum += mpfr_check{ value.x3 };
        return sum;
    }

    void require_close(const f256& actual, const mpfr_check& expected)
    {
        const mpfr_check got = to_ref_exact(actual);
        const mpfr_check tolerance = comparison_tolerance(expected);
        const mpfr_check diff = abs_check(got - expected);

        CAPTURE(to_text(actual));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(actual.x0);
        CAPTURE(actual.x1);
        CAPTURE(actual.x2);
        CAPTURE(actual.x3);

        REQUIRE(diff <= tolerance);
    }

    void check_parse_case(const char* text)
    {
        f256 parsed{};
        const char* end = nullptr;
        const bool ok = parse_flt256(text, parsed, &end);

        CAPTURE(text);
        REQUIRE(ok);
        REQUIRE(end != nullptr);
        REQUIRE(*end == '\0');

        require_close(parsed, to_ref(text));
    }

    void check_roundtrip_case(const char* label, const f256& value)
    {
        const std::string text = to_text(value);
        const f256 reparsed = to_f256(text);
        const mpfr_check expected = to_ref_exact(value);
        const mpfr_check got = to_ref_exact(reparsed);
        const mpfr_check tolerance = comparison_tolerance(expected);
        const mpfr_check diff = abs_check(got - expected);

        CAPTURE(label);
        CAPTURE(text);
        CAPTURE(to_text(value));
        CAPTURE(to_text(reparsed));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(value.x0);
        CAPTURE(value.x1);
        CAPTURE(value.x2);
        CAPTURE(value.x3);
        CAPTURE(reparsed.x0);
        CAPTURE(reparsed.x1);
        CAPTURE(reparsed.x2);
        CAPTURE(reparsed.x3);

        REQUIRE(diff <= tolerance);
    }

    [[nodiscard]] mpfr_random random_large_finite_for_f256(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> exponent_dist(-900, 900);
        std::uniform_real_distribution<double> mantissa_dist(0.5, 1.0);

        mpfr_random mantissa = mantissa_dist(rng);
        if (sign_dist(rng) != 0)
            mantissa = -mantissa;

        return ldexp(mantissa, exponent_dist(rng));
    }

    [[nodiscard]] std::string to_scientific_string(const mpfr_random& value, int digits)
    {
        return value.str(digits, std::ios_base::scientific);
    }
}

TEST_CASE("f256 parses decimal and scientific strings accurately", "[fltx][f256][io][parse]")
{
    constexpr std::array<const char*, 20> cases = {{
        "0",
        "-0",
        "1",
        "-1",
        "1.25",
        "2.125",
        "-3.75",
        "0.125",
        "0.5",
        "1e-50",
        "1e50",
        "1.0000000000000000000000000000000001",
        "2.0000000000000000000000000000000002",
        "123456789012345678901234567890.125",
        "0.000000000000000000000000000000125",
        "3.1415926535897932384626433832795028841971",
        "2.7182818284590452353602874713526624977572",
        "-8.333333333333333333333333333333333",
        "0.3333333333333333333333333333333333",
        "7.0000000000000000000000000000000001"
    }};

    for (const char* text : cases)
        check_parse_case(text);
}

TEST_CASE("f256 print and parse round-trip preserves explicit limb values", "[fltx][f256][io][roundtrip]")
{
    const std::array<std::pair<const char*, f256>, 8> cases = {{
        { "zero", f256{ 0.0, 0.0, 0.0, 0.0 } },
        { "one", f256{ 1.0, 0.0, 0.0, 0.0 } },
        { "neg_one_point_25", f256{ -1.25, 0.0, 0.0, 0.0 } },
        { "two_limb_small", f256{ 0.5, std::ldexp(1.0, -60), 0.0, 0.0 } },
        { "three_limb_mix", f256{ std::ldexp(1.0, 100), -std::ldexp(1.0, 40), std::ldexp(1.0, -20), 0.0 } },
        { "four_limb_mix", f256{ 1.0, std::ldexp(1.0, -55), -std::ldexp(1.0, -110), std::ldexp(1.0, -165) } },
        { "large_with_tail", f256{ std::ldexp(1.0, 200), std::ldexp(1.0, 140), -std::ldexp(1.0, 80), std::ldexp(1.0, 20) } },
        { "small_with_tail", f256{ std::ldexp(1.0, -200), -std::ldexp(1.0, -255), std::ldexp(1.0, -310), -std::ldexp(1.0, -365) } },
    }};

    for (const auto& [label, value] : cases)
        check_roundtrip_case(label, value);
}

TEST_CASE("f256 brute-force random io test", "[fltx][f256][io][rand]")
{
    std::mt19937_64 rng{ std::random_device{}() };
    constexpr int sample_count = 1000;

    std::cout << "f256 brute-force random roundtrip test: " << sample_count << " random values...\n\n";

    for (int i = 0; i < sample_count; ++i)
    {
        const mpfr_random value = random_large_finite_for_f256(rng);
        const std::string expected = to_scientific_string(value, print_digits);

        const f256 parsed = to_f256(expected);
        const std::string actual = to_text(parsed);

        INFO("iteration: " << i);
        INFO("random mpfr: " << expected);
        INFO("original n:  " << actual);

        REQUIRE(expected == actual);
    }
}
