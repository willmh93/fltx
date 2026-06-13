#include <catch2/catch_test_macros.hpp>

#include <charconv>
#include <compare>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
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

    constexpr bl::f32 constexpr_f32_parse = bl::parse<bl::f32>("1.5");
    static_assert(constexpr_f32_parse == bl::f32{ 1.5f });

    constexpr bl::f64 constexpr_f64_parse = bl::parse<bl::f64>("-2.25");
    static_assert(constexpr_f64_parse == bl::f64{ -2.25 });

    constexpr bl::f128 constexpr_f128_parse = bl::parse<bl::f128>("3.5");
    static_assert(constexpr_f128_parse == bl::f128{ 3.5 });

    constexpr bl::f256 constexpr_f256_parse = bl::parse<bl::f256>("4.75");
    static_assert(constexpr_f256_parse == bl::f256{ 4.75 });

    static_assert(bl::parse<bl::f64>("oops", 9.0) == 9.0);

    template<class T>
    [[nodiscard]] bool close_to_decimal_power(T actual, T expected)
    {
        const T diff = bl::abs(actual - expected);
        T scale = bl::abs(expected);
        if (scale < T{ 1.0 })
            scale = T{ 1.0 };
        return diff <= scale * std::numeric_limits<T>::epsilon() * T{ 16.0 };
    }

    template<class T>
    void check_serialized_roundtrip(
        T value,
        int precision,
        std::ios_base::fmtflags flags,
        bl::round_format expected_round_format,
        int expected_round_precision)
    {
        const std::string text = bl::to_string(value, precision, flags);
        const auto static_text = bl::to_static_string(value, precision, flags);

        const T parsed = bl::parse<T>(std::string_view{ text });
        const T parsed_static = bl::parse<T>(static_text.view());
        const auto try_parsed = bl::try_parse<T>(static_text.view());

        const T expected = (flags & std::ios_base::floatfield) == (std::ios_base::fixed | std::ios_base::scientific)
            ? value
            : T{ bl::round_to(value, expected_round_precision, expected_round_format) };

        CAPTURE(text);
        REQUIRE(try_parsed);
        REQUIRE(try_parsed.consumed == static_text.size());
        REQUIRE(parsed == expected);
        REQUIRE(parsed_static == expected);
        REQUIRE(try_parsed.value == expected);
    }

    template<class T>
    [[nodiscard]] constexpr T round_to_stream_precision(
        T value,
        int precision,
        std::ios_base::fmtflags flags)
    {
        const auto floatfield = flags & std::ios_base::floatfield;
        if (floatfield == std::ios_base::fixed)
            return bl::round_to(value, precision, bl::decimals);
        if (floatfield == std::ios_base::scientific)
            return bl::round_to(value, precision + 1, bl::significant_figures);
        if (floatfield == (std::ios_base::fixed | std::ios_base::scientific))
            return value;
        return bl::round_to(value, precision, bl::significant_figures);
    }

    template<class T>
    [[nodiscard]] constexpr bool constexpr_static_string_roundtrip(
        T value,
        int precision,
        std::ios_base::fmtflags flags)
    {
        const auto text = bl::to_static_string(value, precision, flags);
        const auto parsed = bl::try_parse<T>(text.view());
        if (!parsed || parsed.consumed != text.size())
            return false;

        if (bl::isnan(value))
            return bl::isnan(parsed.value);
        if (bl::isinf(value))
            return parsed.value == value;

        return round_to_stream_precision(parsed.value, precision, flags) ==
               round_to_stream_precision(value, precision, flags);
    }

    template<class T>
    [[nodiscard]] constexpr bool constexpr_io_flag_matrix()
    {
        constexpr int precision = std::numeric_limits<T>::digits10;
        constexpr std::ios_base::fmtflags modes[] = {
            std::ios_base::fmtflags{},
            std::ios_base::fixed,
            std::ios_base::scientific,
            std::ios_base::fixed | std::ios_base::scientific
        };
        constexpr std::ios_base::fmtflags decorations[] = {
            std::ios_base::fmtflags{},
            std::ios_base::showpoint,
            std::ios_base::showpos,
            std::ios_base::uppercase,
            std::ios_base::showpoint | std::ios_base::showpos | std::ios_base::uppercase
        };
        constexpr T values[] = {
            T{ 0.0 },
            T{ 1.25 },
            T{ -123.75 },
            T{ 10.0 } + T{ 1.0 } / T{ 3.0 }
        };

        for (const T value : values)
        {
            for (const auto mode : modes)
            {
                for (const auto decoration : decorations)
                {
                    if (!constexpr_static_string_roundtrip(value, precision, mode | decoration))
                        return false;
                }
            }
        }

        constexpr T inf = std::numeric_limits<T>::infinity();
        constexpr T nan = std::numeric_limits<T>::quiet_NaN();
        constexpr auto special_flags = std::ios_base::showpos | std::ios_base::uppercase;
        return constexpr_static_string_roundtrip(inf, precision, special_flags) &&
               constexpr_static_string_roundtrip(-inf, precision, special_flags) &&
               constexpr_static_string_roundtrip(nan, precision, special_flags);
    }

    template<class T>
    [[nodiscard]] constexpr bool constexpr_extended_io_flag_matrix(T value)
    {
        constexpr int precision = std::numeric_limits<T>::digits10;
        constexpr auto all_decorations =
            std::ios_base::showpoint |
            std::ios_base::showpos |
            std::ios_base::uppercase;
        constexpr std::ios_base::fmtflags flags[] = {
            std::ios_base::fmtflags{},
            std::ios_base::showpoint,
            std::ios_base::showpos,
            std::ios_base::uppercase,
            all_decorations,
            std::ios_base::fixed | all_decorations,
            std::ios_base::scientific | all_decorations,
            std::ios_base::fixed | std::ios_base::scientific | all_decorations
        };
        for (const auto flag : flags)
        {
            if (!constexpr_static_string_roundtrip(value, precision, flag))
                return false;
        }

        constexpr T inf = std::numeric_limits<T>::infinity();
        constexpr T nan = std::numeric_limits<T>::quiet_NaN();
        return constexpr_static_string_roundtrip(inf, precision, all_decorations) &&
               constexpr_static_string_roundtrip(-inf, precision, all_decorations) &&
               constexpr_static_string_roundtrip(nan, precision, all_decorations);
    }

    // Keep the exhaustive matrix on f32; f128/f256 use targeted probes below to stay
    // inside MSVC's default constexpr step budget while still covering each flag family.
    static_assert(constexpr_io_flag_matrix<bl::f32>());
    static_assert(constexpr_extended_io_flag_matrix(bl::f64{ 1.25 }));
    static_assert(constexpr_extended_io_flag_matrix(bl::f64{ 10.0 } + bl::f64{ 1.0 } / bl::f64{ 3.0 }));
    static_assert(constexpr_extended_io_flag_matrix(bl::f128{ 1.25 }));
    static_assert(constexpr_extended_io_flag_matrix(bl::f256{ 1.25 }));

    constexpr auto constexpr_all_io_decorations =
        std::ios_base::showpoint |
        std::ios_base::showpos |
        std::ios_base::uppercase;
    constexpr bl::f128 constexpr_f128_fraction = bl::f128{ 10.0 } + bl::f128{ 1.0 } / bl::f128{ 3.0 };
    constexpr bl::f256 constexpr_f256_fraction = bl::f256{ bl::f256{ 10.0 } + bl::f256{ 1.0 } / bl::f256{ 3.0 } };

    static_assert(constexpr_static_string_roundtrip(
        constexpr_f128_fraction,
        std::numeric_limits<bl::f128>::digits10,
        std::ios_base::fmtflags{}));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f128_fraction,
        std::numeric_limits<bl::f128>::digits10,
        constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f128_fraction,
        std::numeric_limits<bl::f128>::digits10,
        std::ios_base::fixed | constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f128_fraction,
        std::numeric_limits<bl::f128>::digits10,
        std::ios_base::scientific | constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f128_fraction,
        std::numeric_limits<bl::f128>::digits10,
        std::ios_base::fixed | std::ios_base::scientific | constexpr_all_io_decorations));

    static_assert(constexpr_static_string_roundtrip(
        constexpr_f256_fraction,
        std::numeric_limits<bl::f256>::digits10,
        std::ios_base::fmtflags{}));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f256_fraction,
        std::numeric_limits<bl::f256>::digits10,
        constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f256_fraction,
        std::numeric_limits<bl::f256>::digits10,
        std::ios_base::fixed | constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f256_fraction,
        std::numeric_limits<bl::f256>::digits10,
        std::ios_base::scientific | constexpr_all_io_decorations));
    static_assert(constexpr_static_string_roundtrip(
        constexpr_f256_fraction,
        std::numeric_limits<bl::f256>::digits10,
        std::ios_base::fixed | std::ios_base::scientific | constexpr_all_io_decorations));

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

    constexpr std::string_view f32_text = "1.5tail";
    bl::f32 f32_parsed{};
    const auto f32_in = bl::from_chars(f32_text.data(), f32_text.data() + f32_text.size(), f32_parsed);
    REQUIRE(f32_in.ec == std::errc{});
    REQUIRE(f32_in.ptr == f32_text.data() + 3);
    REQUIRE(f32_parsed == bl::f32{ 1.5f });

    const auto f32_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        bl::f32{ 1.5f },
        std::chars_format::fixed,
        2);
    REQUIRE(f32_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(f32_out.ptr - buffer)) == "1.50");

    constexpr std::string_view f64_hex_text = "1.8p+1tail";
    bl::f64 f64_hex_parsed{};
    const auto f64_hex_in = bl::from_chars(
        f64_hex_text.data(),
        f64_hex_text.data() + f64_hex_text.size(),
        f64_hex_parsed,
        std::chars_format::hex);
    REQUIRE(f64_hex_in.ec == std::errc{});
    REQUIRE(f64_hex_in.ptr == f64_hex_text.data() + 6);
    REQUIRE(f64_hex_parsed == 3.0);

    char std_buffer[128]{};
    const auto f64_hex_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        bl::f64{ 3.0 },
        std::chars_format::hex);
    const auto std_f64_hex_out = std::to_chars(
        std_buffer,
        std_buffer + sizeof(std_buffer),
        bl::f64{ 3.0 },
        std::chars_format::hex);
    REQUIRE(f64_hex_out.ec == std::errc{});
    REQUIRE(std_f64_hex_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(f64_hex_out.ptr - buffer)) ==
            std::string_view(std_buffer, static_cast<std::size_t>(std_f64_hex_out.ptr - std_buffer)));

    constexpr std::string_view f128_hex_text = "1.00000000000001p+0tail";
    bl::f128 f128_hex_parsed{};
    const auto f128_hex_in = bl::from_chars(
        f128_hex_text.data(),
        f128_hex_text.data() + f128_hex_text.size(),
        f128_hex_parsed,
        std::chars_format::hex);
    REQUIRE(f128_hex_in.ec == std::errc{});
    REQUIRE(f128_hex_in.ptr == f128_hex_text.data() + 19);
    REQUIRE(f128_hex_parsed == bl::f128{ 1.0 } + bl::ldexp(bl::f128{ 1.0 }, -56));

    constexpr std::string_view f256_hex_text = "1.0000000000000000000000000001p+0tail";
    bl::f256 f256_hex_parsed{};
    const auto f256_hex_in = bl::from_chars(
        f256_hex_text.data(),
        f256_hex_text.data() + f256_hex_text.size(),
        f256_hex_parsed,
        std::chars_format::hex);
    REQUIRE(f256_hex_in.ec == std::errc{});
    REQUIRE(f256_hex_in.ptr == f256_hex_text.data() + 33);
    const bl::f256 f256_hex_expected = bl::f256{ 1.0 } + bl::ldexp(bl::f256{ 1.0 }, -112);
    REQUIRE(f256_hex_parsed == f256_hex_expected);

    constexpr std::string_view prefixed_hex_text = "0x1p+0";
    bl::f128 prefixed_hex_parsed{ 42.0 };
    const auto prefixed_hex_in = bl::from_chars(
        prefixed_hex_text.data(),
        prefixed_hex_text.data() + prefixed_hex_text.size(),
        prefixed_hex_parsed,
        std::chars_format::hex);
    REQUIRE(prefixed_hex_in.ec == std::errc{});
    REQUIRE(prefixed_hex_in.ptr == prefixed_hex_text.data() + 1);
    REQUIRE(prefixed_hex_parsed == bl::f128{ 0.0 });

    const auto hex_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        f128_value,
        std::chars_format::hex);
    REQUIRE(hex_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(hex_out.ptr - buffer)) == "1.4p+0");

    const auto hex_precision_out = bl::to_chars(
        buffer,
        buffer + sizeof(buffer),
        f128_value,
        std::chars_format::hex,
        2);
    REQUIRE(hex_precision_out.ec == std::errc{});
    REQUIRE(std::string_view(buffer, static_cast<std::size_t>(hex_precision_out.ptr - buffer)) == "1.40p+0");

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

TEST_CASE("fltx parse helpers require complete input", "[fltx][io][parse]")
{
    constexpr std::string_view f128_text = "1.25";
    const bl::f128 f128_value = bl::parse<bl::f128>(f128_text);
    REQUIRE(f128_value == bl::f128{ 1.25 });

    bl::f128 f128_out{};
    REQUIRE(bl::try_parse(f128_text, f128_out));
    REQUIRE(f128_out == bl::f128{ 1.25 });

    constexpr std::string_view partial_text = "1.25tail";
    const auto partial_result = bl::try_parse<bl::f128>(partial_text);
    REQUIRE(!partial_result);
    REQUIRE(partial_result.ec == std::errc::invalid_argument);
    REQUIRE(partial_result.consumed == 4u);
    REQUIRE_THROWS_AS(bl::parse<bl::f128>(partial_text), std::invalid_argument);

    constexpr std::string_view f256_hex_text = "1.0000000000000000000000000001p+0";
    const auto f256_hex_result = bl::try_parse<bl::f256>(f256_hex_text, std::chars_format::hex);
    REQUIRE(f256_hex_result);
    REQUIRE(f256_hex_result.consumed == f256_hex_text.size());
    const bl::f256 f256_hex_expected = bl::f256{ 1.0 } + bl::ldexp(bl::f256{ 1.0 }, -112);
    REQUIRE(f256_hex_result.value == f256_hex_expected);

    constexpr bl::f256 constexpr_prefixed_hex = bl::parse<bl::f256>("0x1.4p+0");
    static_assert(constexpr_prefixed_hex == bl::f256{ 1.25 });

    constexpr bl::f256 constexpr_showpos_decimal = bl::parse<bl::f256>("+1.25");
    static_assert(constexpr_showpos_decimal == bl::f256{ 1.25 });

    constexpr bl::f256 constexpr_showpos_prefixed_hex = bl::parse<bl::f256>("+0X1.4P+0");
    static_assert(constexpr_showpos_prefixed_hex == bl::f256{ 1.25 });

    constexpr std::string_view f256_prefixed_hex_text = "0x1.0000000000000000000000000001p+0";
    const auto f256_prefixed_hex_result = bl::try_parse<bl::f256>(f256_prefixed_hex_text, std::chars_format::hex);
    REQUIRE(f256_prefixed_hex_result);
    REQUIRE(f256_prefixed_hex_result.consumed == f256_prefixed_hex_text.size());
    REQUIRE(f256_prefixed_hex_result.value == f256_hex_expected);

    constexpr std::string_view f64_hex_text = "1.8p+1";
    REQUIRE(bl::parse<bl::f64>(f64_hex_text, std::chars_format::hex) == 3.0);
    REQUIRE(bl::parse<bl::f64>("0x1.8p+1") == 3.0);

    std::string f256_half_hex = "1.";
    f256_half_hex.append(52, '0');
    f256_half_hex += "1p+0";
    const bl::f256 f256_half_low_bit = bl::f256{ 1.0 } + bl::ldexp(bl::f256{ 1.0 }, -212);
    REQUIRE(bl::parse<bl::f256>(f256_half_hex, std::chars_format::hex) == f256_half_low_bit);

    std::string f256_above_half_hex = "1.";
    f256_above_half_hex.append(52, '0');
    f256_above_half_hex += '1';
    f256_above_half_hex.append(200, '0');
    f256_above_half_hex += "1p+0";
    REQUIRE(bl::parse<bl::f256>(f256_above_half_hex, std::chars_format::hex) == f256_half_low_bit);

    constexpr std::string_view fixed_text = "1e2";
    const auto fixed_result = bl::try_parse<bl::f128>(fixed_text, std::chars_format::fixed);
    REQUIRE(!fixed_result);
    REQUIRE(fixed_result.consumed == 1u);

    bl::f64 std_default_out{};
    REQUIRE(bl::try_parse("1.25", std_default_out));
    REQUIRE(std_default_out == bl::f64{ 1.25 });

    const auto range_result = bl::try_parse<bl::f32>("1e10000");
    REQUIRE(!range_result);
    REQUIRE(range_result.ec == std::errc::result_out_of_range);
    REQUIRE_THROWS_AS(bl::parse<bl::f32>("1e10000"), std::out_of_range);

    const bl::f32 fallback = bl::parse<bl::f32>("not a number", bl::f32{ 7.0f });
    REQUIRE(fallback == bl::f32{ 7.0f });
}

TEST_CASE("fltx parse and string helpers handle special values", "[fltx][io][parse][string][special]")
{
    constexpr auto f32_inf_text = bl::to_static_string(std::numeric_limits<bl::f32>::infinity());
    static_assert(f32_inf_text.view() == "inf");

    constexpr auto f64_nan_text = bl::to_static_string(std::numeric_limits<bl::f64>::quiet_NaN());
    static_assert(f64_nan_text.view() == "nan");

    constexpr auto f128_neg_inf_text = bl::to_static_string(-std::numeric_limits<bl::f128>::infinity());
    static_assert(f128_neg_inf_text.view() == "-inf");

    constexpr auto f256_pos_upper_inf = bl::to_static_string(
        std::numeric_limits<bl::f256>::infinity(),
        6,
        std::ios_base::uppercase | std::ios_base::showpos);
    static_assert(f256_pos_upper_inf.view() == "+INF");

    constexpr bl::f32 f32_inf = bl::parse<bl::f32>("inf");
    static_assert(f32_inf == std::numeric_limits<bl::f32>::infinity());

    constexpr bl::f64 f64_neg_inf = bl::parse<bl::f64>("-INFINITY", std::chars_format::scientific);
    static_assert(f64_neg_inf == -std::numeric_limits<bl::f64>::infinity());

    constexpr bl::f128 f128_inf = bl::parse<bl::f128>("Infinity", std::chars_format::fixed);
    static_assert(bl::isinf(f128_inf) && !bl::signbit(f128_inf));

    constexpr bl::f256 f256_neg_inf = bl::parse<bl::f256>("-INF", std::chars_format::hex);
    static_assert(bl::isinf(f256_neg_inf) && bl::signbit(f256_neg_inf));

    constexpr bl::f128 f128_nan = bl::parse<bl::f128>("NaN");
    static_assert(bl::isnan(f128_nan));

    REQUIRE(bl::to_string(std::numeric_limits<bl::f32>::infinity()) == "inf");
    REQUIRE(bl::to_string(-std::numeric_limits<bl::f64>::infinity()) == "-inf");
    REQUIRE(bl::to_string(std::numeric_limits<bl::f128>::quiet_NaN()) == "nan");
    REQUIRE(bl::to_string(
        std::numeric_limits<bl::f256>::infinity(),
        6,
        std::ios_base::uppercase | std::ios_base::showpos) == "+INF");

    REQUIRE(bl::isinf(bl::parse<bl::f128>("INF")));
    REQUIRE(bl::isnan(bl::parse<bl::f256>("nan")));

    const auto payload_result = bl::try_parse<bl::f256>("nan(payload)");
    REQUIRE(!payload_result);
    REQUIRE(payload_result.ec == std::errc::invalid_argument);
    REQUIRE(payload_result.consumed == 3u);

    const auto short_result = bl::try_parse<bl::f128>("in");
    REQUIRE(!short_result);
    REQUIRE(short_result.ec == std::errc::invalid_argument);
    REQUIRE(short_result.consumed == 0u);
}

TEST_CASE("fltx precision_info can collapse fixed fractional digits", "[fltx][io][string]")
{
    const bl::precision_info collapsed{ 15, 3, 4 };

    REQUIRE(bl::to_string(bl::to_f128("123.123456789012345"), collapsed, std::ios_base::fixed) == "123.123...2345");
    REQUIRE(bl::to_string(bl::to_f256("123.123456789012345"), collapsed, std::ios_base::fixed) == "123.123...2345");

    REQUIRE(bl::to_string(bl::f32{ 1.1234567f }, bl::precision_info{ 7, 1, 1 }, std::ios_base::fixed) == "1.1...7");
    REQUIRE(bl::to_string(bl::f64{ 1.123456789 }, bl::precision_info{ 9, 2, 3 }, std::ios_base::fixed) == "1.12...789");

    const bl::f128 value = bl::to_f128("123.123456789012345");
    REQUIRE(bl::to_string(value, collapsed) == bl::to_string(value, collapsed.digits));
    REQUIRE(bl::to_string(value, collapsed, std::ios_base::scientific) == bl::to_string(value, collapsed.digits, std::ios_base::scientific));
}

TEST_CASE("fltx string output accepts std fmtflags", "[fltx][io][string]")
{
    static_assert(bl::f32_io_string::static_capacity == 64);
    static_assert(bl::f64_io_string::static_capacity == 352);
    static_assert(bl::f128_io_string::static_capacity == 352);
    static_assert(bl::f256_io_string::static_capacity == 384);

    static_assert(std::is_same_v<decltype(bl::to_static_string(bl::f32{})), bl::f32_io_string>);
    static_assert(std::is_same_v<decltype(bl::to_static_string(bl::f64{})), bl::f64_io_string>);
    static_assert(std::is_same_v<decltype(bl::to_static_string(bl::f128{})), bl::f128_io_string>);
    static_assert(std::is_same_v<decltype(bl::to_static_string(bl::f256{})), bl::f256_io_string>);

    static_assert(bl::to_static_string(
        std::numeric_limits<bl::f32>::max(),
        std::numeric_limits<bl::f32>::max_digits10,
        std::ios_base::fixed).size() <= bl::f32_io_string::static_capacity);
    static_assert(bl::to_static_string(
        std::numeric_limits<bl::f64>::max(),
        std::numeric_limits<bl::f64>::max_digits10,
        std::ios_base::fixed).size() <= bl::f64_io_string::static_capacity);
    static_assert(bl::to_static_string(
        std::numeric_limits<bl::f128>::max(),
        std::numeric_limits<bl::f128>::max_digits10,
        std::ios_base::fixed).size() <= bl::f128_io_string::static_capacity);
    static_assert(bl::to_static_string(
        std::numeric_limits<bl::f256>::max(),
        std::numeric_limits<bl::f256>::max_digits10,
        std::ios_base::fixed).size() <= bl::f256_io_string::static_capacity);

    static_assert(bl::to_static_string(bl::f32{ 123.25f }, 2, std::ios_base::fixed).view() == "123.25");
    static_assert(bl::to_static_string(bl::f64{ 123.25 }, 2, std::ios_base::scientific).view() == "1.23e+02");

    const bl::f128 f128_value = bl::to_f128("123.25");
    const bl::f256 f256_value = bl::to_f256("123.25");

    REQUIRE(bl::to_string(bl::f32{ 123.25f }, 2, std::ios_base::fixed) == "123.25");
    REQUIRE(bl::to_string(bl::f64{ 123.25 }, 2, std::ios_base::scientific) == "1.23e+02");
    REQUIRE(std::string(bl::to_static_string(f128_value, 2, std::ios_base::fixed)) == "123.25");
    REQUIRE(std::string(bl::to_static_string(f256_value, 2, std::ios_base::scientific)) == "1.23e+02");

    REQUIRE(bl::to_string(f128_value, 2, std::ios_base::fixed) == "123.25");
    REQUIRE(bl::to_string(f256_value, 2, std::ios_base::scientific) == "1.23e+02");
    REQUIRE(bl::to_string(bl::to_f256("1.2500"), 6, std::ios_base::fmtflags{}) == "1.25");
    REQUIRE(bl::to_string(bl::f32{ 3.0f }, 2, (std::ios_base::fixed | std::ios_base::scientific)) == "0x1.800000p+1");
    REQUIRE(bl::to_string(bl::f64{ 3.0 }, 2, (std::ios_base::fixed | std::ios_base::scientific)) == "0x1.8000000000000p+1");
    REQUIRE(std::string(bl::to_static_string(bl::f128{ 1.0 }, 2, (std::ios_base::fixed | std::ios_base::scientific))) == "0x1.000000000000000000000000000p+0");
    REQUIRE(std::string(bl::to_static_string(bl::f256{ 1.0 }, 2, (std::ios_base::fixed | std::ios_base::scientific))) == "0x1.00000000000000000000000000000000000000000000000000000p+0");

    REQUIRE(bl::to_string(bl::f32{ 1.25f }, 5, std::ios_base::showpoint) == "1.2500");
    REQUIRE(bl::to_string(bl::f64{ 3.0 }, 2, std::ios_base::showpos | std::ios_base::showpoint) == "+3.0");
    const std::string upper_hex = bl::to_string(
        f256_value,
        2,
        std::ios_base::fixed | std::ios_base::scientific | std::ios_base::uppercase);
    REQUIRE(upper_hex.starts_with("0X1.ED"));
    REQUIRE(upper_hex.ends_with("P+6"));

    std::ostringstream hex_stream;
    hex_stream << std::hexfloat << bl::f128{ 1.0 };
    REQUIRE(hex_stream.str() == "0x1.000000000000000000000000000p+0");

    std::ostringstream upper_hex_stream;
    upper_hex_stream << std::uppercase << std::hexfloat << bl::f256{ 1.25 };
    REQUIRE(upper_hex_stream.str() == "0X1.40000000000000000000000000000000000000000000000000000P+0");

    const bl::precision_info collapsed{ 8, 1, 1 };
    REQUIRE(bl::to_string(bl::to_f128("1.12345678"), collapsed, std::ios_base::fixed) == "1.1...8");
}

TEST_CASE("fltx typed float formats serialize and deserialize predictably", "[fltx][io][string][parse]")
{
    check_serialized_roundtrip(bl::f32{ 123.25f }, 5, std::ios_base::fmtflags{}, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f32{ 123.25f }, 2, std::ios_base::fixed, bl::decimals, 2);
    check_serialized_roundtrip(bl::f32{ 123.25f }, 4, std::ios_base::scientific, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f32{ 123.25f }, 1, (std::ios_base::fixed | std::ios_base::scientific), bl::significant_figures, 0);

    check_serialized_roundtrip(bl::f64{ 123.25 }, 5, std::ios_base::fmtflags{}, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f64{ 123.25 }, 2, std::ios_base::fixed, bl::decimals, 2);
    check_serialized_roundtrip(bl::f64{ 123.25 }, 4, std::ios_base::scientific, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f64{ 123.25 }, 1, (std::ios_base::fixed | std::ios_base::scientific), bl::significant_figures, 0);

    check_serialized_roundtrip(bl::f128{ 123.25 }, 5, std::ios_base::fmtflags{}, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f128{ 123.25 }, 2, std::ios_base::fixed, bl::decimals, 2);
    check_serialized_roundtrip(bl::f128{ 123.25 }, 4, std::ios_base::scientific, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f128{ 123.25 }, 1, (std::ios_base::fixed | std::ios_base::scientific), bl::significant_figures, 0);
    check_serialized_roundtrip(bl::f128{ 123.25 }, 4, (std::ios_base::scientific | std::ios_base::showpos | std::ios_base::uppercase), bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f128{ 123.25 }, 1, (std::ios_base::fixed | std::ios_base::scientific | std::ios_base::showpos | std::ios_base::uppercase), bl::significant_figures, 0);

    check_serialized_roundtrip(bl::f256{ 123.25 }, 5, std::ios_base::fmtflags{}, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f256{ 123.25 }, 2, std::ios_base::fixed, bl::decimals, 2);
    check_serialized_roundtrip(bl::f256{ 123.25 }, 4, std::ios_base::scientific, bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f256{ 123.25 }, 1, (std::ios_base::fixed | std::ios_base::scientific), bl::significant_figures, 0);
    check_serialized_roundtrip(bl::f256{ 123.25 }, 4, (std::ios_base::scientific | std::ios_base::showpos | std::ios_base::uppercase), bl::significant_figures, 5);
    check_serialized_roundtrip(bl::f256{ 123.25 }, 1, (std::ios_base::fixed | std::ios_base::scientific | std::ios_base::showpos | std::ios_base::uppercase), bl::significant_figures, 0);
}

TEST_CASE("fltx streams f256 extremes without normalization stalls", "[fltx][io][stream]")
{
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<bl::f256>::digits10)
        << std::numeric_limits<bl::f256>::min();

    REQUIRE(out.str().starts_with("2.225073858507201383090232717332"));
    REQUIRE(out.str().ends_with("e-308"));
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

TEST_CASE("fltx generic pow10 covers the float family", "[fltx][math][pow10]")
{
    REQUIRE(bl::pow10<bl::f32>(3) == 1000.0f);
    REQUIRE(bl::pow10<bl::f64>(3) == 1000.0);
    REQUIRE(bl::pow10<bl::f128>(3) == bl::f128{ 1000.0 });
    REQUIRE(bl::pow10<bl::f256>(3) == bl::f256{ 1000.0 });

    for (int exponent = -45; exponent <= 38; ++exponent)
    {
        CAPTURE(exponent);
        const std::string token = "1e" + std::to_string(exponent);

        bl::f32 expected{};
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), expected);

        REQUIRE(ec == std::errc{});
        REQUIRE(ptr == token.data() + token.size());
        REQUIRE(bl::pow10<bl::f32>(exponent) == expected);
    }

    for (int exponent = -323; exponent <= 308; ++exponent)
    {
        CAPTURE(exponent);
        const std::string token = "1e" + std::to_string(exponent);

        bl::f64 expected{};
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), expected);

        REQUIRE(ec == std::errc{});
        REQUIRE(ptr == token.data() + token.size());
        REQUIRE(bl::pow10<bl::f64>(exponent) == expected);
    }

    const int extended_exponents[] = { -32, -8, -3, -1, 0, 1, 3, 8, 32 };
    for (int exponent : extended_exponents)
    {
        CAPTURE(exponent);
        const std::string token = "1e" + std::to_string(exponent);

        const bl::f128 expected_f128 = bl::to_f128(token);
        const bl::f256 expected_f256 = bl::to_f256(token);
        REQUIRE(close_to_decimal_power(bl::pow10<bl::f128>(exponent), expected_f128));
        REQUIRE(close_to_decimal_power(bl::pow10<bl::f256>(exponent), expected_f256));
    }
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
    REQUIRE(std::format("{:a}", bl::f128{ 1.25 }) == "0x1.4p+0");
    REQUIRE(std::format("{:.2a}", bl::f128{ 1.25 }) == "0x1.40p+0");
    REQUIRE(std::format("{:+#.0A}", bl::f256{ 1.0 }) == "+0X1.P+0");
    REQUIRE(std::format("{:>12.1a}", bl::f128{ 1.25 }) == "    0x1.4p+0");
    REQUIRE(std::format("{:.2F}", std::numeric_limits<bl::f128>::infinity()) == "INF");
    REQUIRE(std::format("{:+.2F}", std::numeric_limits<bl::f128>::quiet_NaN()) == "+NAN");
    REQUIRE(std::format("{:+A}", std::numeric_limits<bl::f256>::infinity()) == "+INF");

    #if !defined(__EMSCRIPTEN__)
    bl::f128 invalid_format_value{ 1.0 };
    REQUIRE_THROWS_AS(std::vformat("{:x}", std::make_format_args(invalid_format_value)), std::format_error);
    #endif
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
