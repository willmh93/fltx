#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <fltx/charconv.h>
#include <fltx/f128_io.h>
#include <fltx/random.h>

using namespace bl;

namespace
{
    using mpfr_check = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<100>,
        boost::multiprecision::et_off>;

    constexpr int checked_digits = std::numeric_limits<f128>::digits10 - 4;
    constexpr int print_digits   = std::numeric_limits<f128>::max_digits10;

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

    [[nodiscard]] std::string to_text(const f128& value)
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

    [[nodiscard]] mpfr_check to_ref_exact(const f128& value)
    {
        mpfr_check sum = 0;
        sum += mpfr_check{ value.hi };
        sum += mpfr_check{ value.lo };
        return sum;
    }

    void require_close(const f128& actual, const mpfr_check& expected)
    {
        const mpfr_check got       = to_ref_exact(actual);
        const mpfr_check tolerance = comparison_tolerance(expected);
        const mpfr_check diff      = abs_check(got - expected);

        CAPTURE(to_text(actual));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(actual.hi);
        CAPTURE(actual.lo);

        REQUIRE(diff <= tolerance);
    }

    void check_parse_case(const char* text)
    {
        f128 parsed{};
        const std::string_view input{ text };
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);

        CAPTURE(text);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr == input.data() + input.size());

        require_close(parsed, to_ref(text));
    }

    void check_roundtrip_case(const char* label, const f128& value)
    {
        const std::string text     = to_text(value);
        const f128 reparsed        = to_f128(text);
        const mpfr_check expected  = to_ref_exact(value);
        const mpfr_check got       = to_ref_exact(reparsed);
        const mpfr_check tolerance = comparison_tolerance(expected);
        const mpfr_check diff      = abs_check(got - expected);

        CAPTURE(label);
        CAPTURE(text);
        CAPTURE(to_text(value));
        CAPTURE(to_text(reparsed));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(value.hi);
        CAPTURE(value.lo);
        CAPTURE(reparsed.hi);
        CAPTURE(reparsed.lo);

        REQUIRE(diff <= tolerance);
    }

    [[nodiscard]] f128 random_large_finite_for_f128(bl::mt19937_64& rng)
    {
        bl::uniform_int_distribution<int> sign_dist(0, 1);
        bl::uniform_int_distribution<int> exponent_dist(-250, 250);
        bl::uniform_real_distribution<f128> mantissa_dist{ f128{ 0.5 }, f128{ 1.0 } };

        f128 mantissa = mantissa_dist(rng);
        if (sign_dist(rng) != 0)
            mantissa = -mantissa;

        return bl::ldexp(mantissa, exponent_dist(rng));
    }

} // namespace

TEST_CASE("f128 parses decimal and scientific strings accurately", "[fltx][f128][io][parse]")
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
        "1e-20",
        "1e20",
        "1.00000000000000000000000000000001",
        "2.00000000000000000000000000000002",
        "12345678901234567890.125",
        "0.0000000000000000000000000125",
        "3.1415926535897932384626433832795",
        "2.7182818284590452353602874713527",
        "-8.333333333333333333333333333333",
        "0.3333333333333333333333333333333",
        "7.0000000000000000000000000000001"
    }};

    for (const char* text : cases)
        check_parse_case(text);
}

TEST_CASE("f128 literals parse numeric and string source text", "[fltx][f128][io][literal]")
{
    using namespace bl::literals;

    constexpr f128 numeric    = 0.123456789012345678901234567890123_dd;
    constexpr f128 scientific = 1.25e-20_dd;
    constexpr f128 text       = "0.123456789012345678901234567890123"_dd;

    require_close(numeric, to_ref("0.123456789012345678901234567890123"));
    require_close(scientific, to_ref("1.25e-20"));
    require_close(text, to_ref("0.123456789012345678901234567890123"));
}

TEST_CASE("f128 parser handles special values, partial tokens, and invalid inputs", "[fltx][f128][io][parse][edge]")
{
    {
        const char* text = "infinity;";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr != nullptr);
        REQUIRE(*result.ptr == ';');
        REQUIRE(bl::isinf(parsed));
        REQUIRE(!bl::signbit(parsed));
    }
    {
        const char* text = "-inf tail";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr != nullptr);
        REQUIRE(*result.ptr == ' ');
        REQUIRE(bl::isinf(parsed));
        REQUIRE(bl::signbit(parsed));
    }
    {
        const char* text = "NaN(payload)";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr != nullptr);
        REQUIRE(*result.ptr == '(');
        REQUIRE(bl::isnan(parsed));
    }
    {
        const char* text = "1.25tail";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr != nullptr);
        REQUIRE(*result.ptr == 't');
        require_close(parsed, to_ref("1.25"));
    }
    {
        const char* text = "1e+oops";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr != nullptr);
        REQUIRE(*result.ptr == 'e');
        require_close(parsed, to_ref("1"));
    }
    {
        const char* text = "1e100000000";
        const std::string_view input{ text };
        f128 parsed{};
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr == input.data() + input.size());
        REQUIRE(bl::isinf(parsed));
        REQUIRE(!bl::signbit(parsed));
    }
    {
        const char* text = "-1e-100000000";
        const std::string_view input{ text };
        f128 parsed{ 1.0, 0.0 };
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc{});
        REQUIRE(result.ptr == input.data() + input.size());
        REQUIRE(bl::iszero(parsed));
        REQUIRE(bl::signbit(parsed));
    }

    constexpr std::array<const char*, 6> invalid_cases = {{
        "",
        " ",
        ".",
        "+.",
        "e10",
        "--1"
    }};

    for (const char* text : invalid_cases)
    {
        const std::string_view input{ text };
        f128 parsed{ 42.0, 0.0 };
        CAPTURE(text);
        const auto result = bl::from_chars(input.data(), input.data() + input.size(), parsed);
        REQUIRE(result.ec == std::errc::invalid_argument);
        REQUIRE(result.ptr == input.data());
        REQUIRE(parsed.hi == 42.0);
        REQUIRE(parsed.lo == 0.0);
    }

    REQUIRE(bl::iszero(to_f128("oops")));
}

TEST_CASE("f128 fixed zero formatting respects precision", "[fltx][f128][io][format]")
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(std::numeric_limits<f128>::digits10) << f128{ 0.0, 0.0 };

    std::string expected = "0.";
    expected.append(static_cast<std::size_t>(std::numeric_limits<f128>::digits10), '0');
    REQUIRE(out.str() == expected);
    REQUIRE(bl::to_string(f128{ 0.0, 0.0 }, 5, true, false, false) == "0.00000");
    REQUIRE(bl::to_string(f128{ 0.0, 0.0 }, 5, true, false, true) == "0");

    std::ostringstream neg_out;
    neg_out << std::fixed << std::setprecision(3) << f128{ -0.0, 0.0 };
    REQUIRE(neg_out.str() == "-0.000");
}

TEST_CASE("f128 formats special values and stream flags consistently", "[fltx][f128][io][format][edge]")
{
    const f128 inf     = std::numeric_limits<f128>::infinity();
    const f128 neg_inf = -inf;
    const f128 nan     = std::numeric_limits<f128>::quiet_NaN();

    REQUIRE(bl::to_string(inf) == "inf");
    REQUIRE(bl::to_string(neg_inf) == "-inf");
    REQUIRE(bl::to_string(nan) == "nan");
    REQUIRE(std::string(bl::to_static_string(to_f128("1.2500"), 4, true, false, true)) == "1.25");

    std::ostringstream pos_upper;
    pos_upper << std::showpos << std::uppercase << inf;
    REQUIRE(pos_upper.str() == "+INF");

    std::ostringstream neg_upper;
    neg_upper << std::uppercase << neg_inf;
    REQUIRE(neg_upper.str() == "-INF");

    std::ostringstream nan_upper;
    nan_upper << std::uppercase << nan;
    REQUIRE(nan_upper.str() == "NAN");
}

TEST_CASE("f128 stream extraction parses complete tokens", "[fltx][f128][io][stream]")
{
    std::istringstream in{ "1.25 -0 +inf nan tail" };
    f128 value{};
    f128_s neg_zero{};
    f128 inf{};
    f128_s nan{};
    std::string tail;

    REQUIRE(static_cast<bool>(in >> value >> neg_zero >> inf >> nan >> tail));
    require_close(value, to_ref("1.25"));
    REQUIRE(bl::iszero(neg_zero));
    REQUIRE(bl::signbit(neg_zero));
    REQUIRE(bl::isinf(inf));
    REQUIRE(!bl::signbit(inf));
    REQUIRE(bl::isnan(nan));
    REQUIRE(tail == "tail");
}

TEST_CASE("f128 stream extraction fails without overwriting on partial or invalid tokens", "[fltx][f128][io][stream][edge]")
{
    {
        std::istringstream in{ "1.25tail" };
        f128 value{ 42.0, 0.0 };

        REQUIRE_FALSE(static_cast<bool>(in >> value));
        REQUIRE(in.fail());
        REQUIRE(value.hi == 42.0);
        REQUIRE(value.lo == 0.0);
    }
    {
        std::istringstream in{ "oops" };
        f128_s value{ 7.0, 0.0 };

        REQUIRE_FALSE(static_cast<bool>(in >> value));
        REQUIRE(in.fail());
        REQUIRE(value.hi == 7.0);
        REQUIRE(value.lo == 0.0);
    }
}

TEST_CASE("f128 print and parse round-trip preserves explicit limb values", "[fltx][f128][io][roundtrip]")
{
    const std::array<std::pair<const char*, f128>, 8> cases = {{
        { "zero", f128{ 0.0, 0.0 } },
        { "one", f128{ 1.0, 0.0 } },
        { "neg_one_point_25", f128{ -1.25, 0.0 } },
        { "two_limb_small", f128{ 0.5, std::ldexp(1.0, -60) } },
        { "unit_with_tail", f128{ 1.0, std::ldexp(1.0, -55) } },
        { "large_with_tail", f128{ std::ldexp(1.0, 200), std::ldexp(1.0, 140) } },
        { "small_with_tail", f128{ std::ldexp(1.0, -200), -std::ldexp(1.0, -255) } },
        { "mixed_sign_tail", f128{ std::ldexp(1.0, 100), -std::ldexp(1.0, 40) } },
    }};

    for (const auto& [label, value] : cases)
        check_roundtrip_case(label, value);
}

TEST_CASE("f128 brute-force random roundtrip test", "[fltx][f128][io][rand]")
{
    bl::mt19937_64 rng{ 0x1020304050607080ull };
    constexpr int sample_count = 10000;

    std::cout << "f128 brute-force random roundtrip test: " << sample_count
              << " random values (seed 0x1020304050607080)...\n\n";

    for (int i = 0; i < sample_count; ++i)
    {
        const f128 value           = random_large_finite_for_f128(rng);
        const std::string expected = to_text(value);

        const f128 parsed        = to_f128(expected);
        const std::string actual = to_text(parsed);

        INFO("iteration: " << i);
        INFO("random f128: " << expected);
        INFO("original n:  " << actual);

        require_close(parsed, to_ref_exact(value));
    }
}
