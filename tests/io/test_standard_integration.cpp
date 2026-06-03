#include <catch2/catch_test_macros.hpp>

#include <charconv>
#include <chrono>
#include <compare>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>

#include <fltx.h>
#include <fltx/format.h>

namespace
{
    static_assert((bl::f128{ 1.0 } <=> bl::f128{ 2.0 }) == std::partial_ordering::less);
    static_assert((bl::f128{ 1.0 } <=> 2.0) == std::partial_ordering::less);
    static_assert((2.0 <=> bl::f128{ 1.0 }) == std::partial_ordering::greater);
    static_assert((bl::f256{ 2.0 } <=> bl::f256{ 1.0 }) == std::partial_ordering::greater);
    static_assert((bl::f256{ 1.0 } <=> 1.0) == std::partial_ordering::equivalent);

    static_assert(bl::fltx_f32<bl::f32>);
    static_assert(bl::fltx_f64<bl::f64>);
    static_assert(bl::fltx_f128<bl::f128>);
    static_assert(bl::fltx_f128<const bl::f128_s>);
    static_assert(bl::fltx_f256<bl::f256>);
    static_assert(bl::fltx_f256<const bl::f256_s>);
    static_assert(bl::fltx_extended_float<bl::f128>);
    static_assert(bl::fltx_extended_float<bl::f256_s>);
    static_assert(!bl::fltx_extended_float<bl::f64>);
    static_assert(!bl::fltx_extended_float<bl::f128&>);
    static_assert(bl::fltx_float<bl::f32>);
    static_assert(bl::fltx_float<bl::f256>);
    static_assert(bl::fltx_floating_point<bl::f128>);
    static_assert(bl::fltx_floating_point<bl::f64>);
    static_assert(bl::fltx_arithmetic<bl::f256>);
    static_assert(bl::fltx_arithmetic<int>);

    static_assert(bl::is_f32_v<bl::f32>);
    static_assert(bl::is_f64_v<bl::f64>);
    static_assert(bl::is_f128_v<bl::f128>);
    static_assert(bl::is_f256_v<bl::f256>);
    static_assert(bl::is_fltx_extended_float_v<bl::f128>);
    static_assert(bl::is_fltx_float_v<bl::f64>);
    static_assert(bl::is_floating_point_v<bl::f256>);
    static_assert(bl::is_arithmetic_v<bl::f128>);
    static_assert(bl::is_integral_v<int>);
    static_assert(!bl::is_integral_v<bl::f128>);

    static_assert(std::chrono::treat_as_floating_point_v<bl::f128>);
    static_assert(std::chrono::treat_as_floating_point_v<bl::f128_s>);
    static_assert(std::chrono::treat_as_floating_point_v<bl::f256>);
    static_assert(std::chrono::treat_as_floating_point_v<bl::f256_s>);

    static_assert(std::is_same_v<std::common_type_t<bl::f128, bl::f128, std::intmax_t>, bl::f128>);
    static_assert(std::is_same_v<std::common_type_t<bl::f256, bl::f256, std::intmax_t>, bl::f256>);
    static_assert(std::is_same_v<std::common_type_t<bl::f128_s, std::intmax_t>, bl::f128>);
    static_assert(std::is_same_v<std::common_type_t<bl::f128, bl::f256>, bl::f256>);

} // namespace

TEST_CASE("fltx charconv-shaped helpers format and parse", "[fltx][io][charconv]")
{
    char buffer[128]{};

    const bl::f128 f128_value{ 1.25 };
    const auto f128_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        f128_value,
        std::chars_format::fixed,
        3);

    REQUIRE(f128_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(f128_out.ptr - buffer)) == "1.250");

    bl::f128 f128_parsed{};
    const auto f128_in = bl::from_chars(buffer, f128_out.ptr, f128_parsed, std::chars_format::fixed);
    REQUIRE(f128_in.ec == std::errc{});
    REQUIRE(f128_in.ptr == f128_out.ptr);
    REQUIRE(f128_parsed == f128_value);

    const auto f256_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        bl::f256{ 2.5 },
        std::chars_format::fixed,
        4);
    REQUIRE(f256_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(f256_out.ptr - buffer)) == "2.5000");

    constexpr std::string_view text = "3.5tail";
    bl::f256 f256_parsed{};
    const auto f256_in = bl::from_chars(text.data(), text.data() + text.size(), f256_parsed);
    REQUIRE(f256_in.ec == std::errc{});
    REQUIRE(f256_in.ptr == text.data() + 3);
    REQUIRE(f256_parsed == bl::f256{ 3.5 });

    const auto hex_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        f128_value,
        std::chars_format::hex);
    REQUIRE(hex_out.ec == std::errc::invalid_argument);

    char tiny[3]{};
    const auto too_small_out = bl::to_chars(
        tiny,
        tiny + sizeof(tiny),
        f128_value,
        std::chars_format::fixed,
        3);
    REQUIRE(too_small_out.ec == std::errc::value_too_large);
    REQUIRE(too_small_out.ptr == tiny + sizeof(tiny));

    const auto inf_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        std::numeric_limits<bl::f256>::infinity());
    REQUIRE(inf_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(inf_out.ptr - buffer)) == "inf");

    const auto nan_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        std::numeric_limits<bl::f128>::quiet_NaN());
    REQUIRE(nan_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(nan_out.ptr - buffer)) == "nan");

    bl::f128 invalid{};
    constexpr std::string_view plus_text = "+1";
    const auto plus_in = bl::from_chars(plus_text.data(), plus_text.data() + plus_text.size(), invalid);
    REQUIRE(plus_in.ec == std::errc::invalid_argument);
    REQUIRE(plus_in.ptr == plus_text.data());

    constexpr std::string_view exponent_text = "1e2";
    bl::f128 fixed_parsed{};
    const auto fixed_in = bl::from_chars(
        exponent_text.data(),
        exponent_text.data() + exponent_text.size(),
        fixed_parsed,
        std::chars_format::fixed);
    REQUIRE(fixed_in.ec == std::errc{});
    REQUIRE(fixed_in.ptr == exponent_text.data() + 1);
    REQUIRE(fixed_parsed == bl::f128{ 1.0 });

    bl::f128 scientific_parsed{};
    constexpr std::string_view no_exponent_text = "1.25";
    const auto scientific_in = bl::from_chars(
        no_exponent_text.data(),
        no_exponent_text.data() + no_exponent_text.size(),
        scientific_parsed,
        std::chars_format::scientific);
    REQUIRE(scientific_in.ec == std::errc::invalid_argument);

    constexpr std::string_view neg_inf_text = "-inf!";
    bl::f128 neg_inf{};
    const auto neg_inf_in = bl::from_chars(neg_inf_text.data(), neg_inf_text.data() + neg_inf_text.size(), neg_inf);
    REQUIRE(neg_inf_in.ec == std::errc{});
    REQUIRE(neg_inf_in.ptr == neg_inf_text.data() + 4);
    REQUIRE(bl::isinf(neg_inf));
    REQUIRE(bl::signbit(neg_inf));

    constexpr std::string_view nan_text = "nan rest";
    bl::f256 parsed_nan{};
    const auto nan_in = bl::from_chars(nan_text.data(), nan_text.data() + nan_text.size(), parsed_nan);
    REQUIRE(nan_in.ec == std::errc{});
    REQUIRE(nan_in.ptr == nan_text.data() + 3);
    REQUIRE(bl::isnan(parsed_nan));
}

TEST_CASE("fltx partial ordering reports NaNs as unordered", "[fltx][compare]")
{
    const bl::f128 f128_nan = std::numeric_limits<bl::f128>::quiet_NaN();
    const bl::f256 f256_nan = std::numeric_limits<bl::f256>::quiet_NaN();

    REQUIRE((f128_nan <=> bl::f128{ 0.0 }) == std::partial_ordering::unordered);
    REQUIRE((bl::f128{ 0.0 } <=> f128_nan) == std::partial_ordering::unordered);
    REQUIRE((f256_nan <=> bl::f256{ 0.0 }) == std::partial_ordering::unordered);
    REQUIRE((bl::f256{ 0.0 } <=> f256_nan) == std::partial_ordering::unordered);

    const double scalar_nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE((bl::f128{ 0.0 } <=> scalar_nan) == std::partial_ordering::unordered);
    REQUIRE((scalar_nan <=> bl::f128{ 0.0 }) == std::partial_ordering::unordered);
}

TEST_CASE("fltx hash specializations support unordered containers", "[fltx][hash]")
{
    const std::hash<bl::f128> f128_hash{};
    const std::hash<bl::f256> f256_hash{};

    REQUIRE(f128_hash(bl::f128{ 0.0 }) == f128_hash(bl::f128{ -0.0 }));
    REQUIRE(f256_hash(bl::f256{ 0.0 }) == f256_hash(bl::f256{ -0.0 }));

    std::unordered_map<bl::f128, int> values;
    values.emplace(bl::f128{ 1.5 }, 42);

    const auto found = values.find(bl::f128{ 1.5 });
    REQUIRE(found != values.end());
    REQUIRE(found->second == 42);
}

TEST_CASE("fltx chrono floating duration reps are enabled", "[fltx][chrono]")
{
    using seconds = std::chrono::duration<bl::f128>;
    using milliseconds = std::chrono::duration<bl::f128, std::milli>;

    const seconds duration{ bl::f128{ 1.5 } };
    const milliseconds converted{ duration };

    REQUIRE(converted.count() == bl::f128{ 1500.0 });

    using f256_seconds = std::chrono::duration<bl::f256>;
    using f256_microseconds = std::chrono::duration<bl::f256, std::micro>;

    const f256_seconds f256_duration{ bl::f256{ 0.25 } };
    const f256_microseconds f256_converted{ f256_duration };

    REQUIRE(f256_converted.count() == bl::f256{ 250000.0 });
}

#if FLTX_HAS_STD_FORMAT
TEST_CASE("fltx std formatter supports common numeric specs", "[fltx][format]")
{
    REQUIRE(std::format("{}", bl::f128{ 1.25 }) == "1.25");
    REQUIRE(std::format("{:.4g}", bl::f128{ 1.25 }) == "1.25");
    REQUIRE(std::format("{:.3f}", bl::f128{ 1.25 }) == "1.250");
    REQUIRE(std::format("{:.2f}", bl::f128_s{ 2.5, 0.0 }) == "2.50");
    REQUIRE(std::format("{:.2f}", bl::f256_s{ 2.5, 0.0, 0.0, 0.0 }) == "2.50");
    REQUIRE(std::format("{:+.2e}", bl::f256{ 1.5 }) == "+1.50e+00");
    REQUIRE(std::format("{:+.2E}", bl::f256{ 1.5 }) == "+1.50E+00");
    REQUIRE(std::format("{: .2f}", bl::f128{ 1.25 }) == " 1.25");
    REQUIRE(std::format("{: .2f}", bl::f128{ -1.25 }) == "-1.25");
    REQUIRE(std::format("{:+08.2f}", bl::f128{ 1.25 }) == "+0001.25");
    REQUIRE(std::format("{:08.2f}", bl::f128{ -1.25 }) == "-0001.25");
    REQUIRE(std::format("{:>8.2f}", bl::f128{ 1.25 }) == "    1.25");
    REQUIRE(std::format("{:<8.2f}", bl::f128{ 1.25 }) == "1.25    ");
    REQUIRE(std::format("{:*^8.2f}", bl::f128{ 1.25 }) == "**1.25**");
    REQUIRE(std::format("{:#.0f}", bl::f128{ 1.0 }) == "1.");
    REQUIRE(std::format("{:#.3g}", bl::f128{ 1.0 }) == "1.00");
    REQUIRE(std::format("{:.2F}", std::numeric_limits<bl::f128>::infinity()) == "INF");
    REQUIRE(std::format("{:+.2F}", std::numeric_limits<bl::f128>::quiet_NaN()) == "+NAN");

    bl::f128 invalid_format_value{ 1.0 };
    REQUIRE_THROWS_AS(std::vformat("{:x}", std::make_format_args(invalid_format_value)), std::format_error);
}

TEST_CASE("fltx std formatter preserves long precision and padding", "[fltx][format]")
{
    const bl::f256 pi = std::numbers::pi_v<bl::f256>;

    const std::string fixed60 = std::format("{:.60f}", pi);
    REQUIRE(fixed60.size() == 62u);
    REQUIRE(fixed60.find('.') == 1u);
    REQUIRE(fixed60.starts_with("3.14159265358979323846264338327950288419716939937510"));

    const std::string scientific55 = std::format("{:.55e}", pi);
    REQUIRE(scientific55.size() == 61u);
    REQUIRE(scientific55.starts_with("3.14159265358979323846264338327950288419716939937510"));
    REQUIRE(scientific55.ends_with("e+00"));

    const std::string general64 = std::format("{:.64g}", pi);
    REQUIRE(general64.size() == 65u);
    REQUIRE(general64.find('.') == 1u);
    REQUIRE(general64.starts_with("3.14159265358979323846264338327950288419716939937510"));
    REQUIRE(general64.back() != '0');

    const std::string right80 = std::format("{:>80.60f}", pi);
    REQUIRE(right80.size() == 80u);
    REQUIRE(right80 == std::string(80u - fixed60.size(), ' ') + fixed60);

    const std::string left76 = std::format("{:_<76.60f}", pi);
    REQUIRE(left76.size() == 76u);
    REQUIRE(left76 == fixed60 + std::string(76u - fixed60.size(), '_'));

    const std::string centered67 = std::format("{:*^67.60f}", pi);
    REQUIRE(centered67.size() == 67u);
    REQUIRE(centered67 == std::string(2u, '*') + fixed60 + std::string(3u, '*'));

    REQUIRE(std::format("{:.10g}", bl::f256{ 1.25 }) == "1.25");
    REQUIRE(std::format("{:#.10g}", bl::f256{ 1.25 }) == "1.250000000");
    REQUIRE(std::format("{:.0f}", pi) == "3");
    REQUIRE(std::format("{:#.0f}", pi) == "3.");
}
#endif
