#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include <fltx.h>
#include "generated_constexpr_cases.h"

using namespace bl;
using namespace bl::constexpr_case_data;

namespace
{
    template<typename T>
    struct sincos_result
    {
        bool ok = false;
        T s{};
        T c{};
    };

    template<typename T>
    struct binary_typed_case
    {
        T lhs{};
        T rhs{};
    };

    template<typename T>
    struct ldexp_typed_case
    {
        T value{};
        int exponent = 0;
    };

    constexpr std::size_t light_chunk_size = 16;
    constexpr std::size_t medium_chunk_size = 8;
    constexpr std::size_t heavy_chunk_size = 4;

    template<typename T>
    [[nodiscard]] constexpr T parse_from_text(const char* text) noexcept;

    template<>
    [[nodiscard]] constexpr f128_s parse_from_text<f128_s>(const char* text) noexcept
    {
        return bl::to_f128(text);
    }

    template<>
    [[nodiscard]] constexpr f256_s parse_from_text<f256_s>(const char* text) noexcept
    {
        return bl::to_f256(text);
    }

    template<typename T>
    [[nodiscard]] constexpr const char* type_name() noexcept;

    template<>
    [[nodiscard]] constexpr const char* type_name<f128_s>() noexcept
    {
        return "f128";
    }

    template<>
    [[nodiscard]] constexpr const char* type_name<f256_s>() noexcept
    {
        return "f256";
    }

    [[nodiscard]] constexpr std::uint64_t bits_of(double value) noexcept
    {
        return std::bit_cast<std::uint64_t>(value);
    }

    [[nodiscard]] std::string hex_bits(std::uint64_t bits)
    {
        std::ostringstream out;
        out << "0x" << std::hex << std::uppercase << bits;
        return out.str();
    }

    [[nodiscard]] std::string describe_double(double value)
    {
        std::ostringstream out;
        out << std::setprecision(36) << value << " [" << hex_bits(bits_of(value)) << "]";
        return out.str();
    }

    [[nodiscard]] std::string describe_value(const f128_s& value)
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<f128_s>::digits10) << value
            << "\n\n{ hi=" << describe_double(value.hi)
            << ", lo=" << describe_double(value.lo) << " }";
        return out.str();
    }

    [[nodiscard]] std::string describe_value(const f256_s& value)
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<f256_s>::digits10) << value
            << "{ x0=" << describe_double(value.x0)
            << ", x1=" << describe_double(value.x1)
            << ", x2=" << describe_double(value.x2)
            << ", x3=" << describe_double(value.x3) << " }";
        return out.str();
    }

    template<typename T>
    [[nodiscard]] std::string describe_value(const sincos_result<T>& value)
    {
        std::ostringstream out;
        out << "{ ok=" << (value.ok ? "true" : "false")
            << ", s=" << describe_value(value.s)
            << ", c=" << describe_value(value.c) << " }";
        return out.str();
    }

    [[nodiscard]] bool exact_match_or_both_nan(const f128_s& lhs, const f128_s& rhs) noexcept
    {
        if (bl::isnan(lhs) && bl::isnan(rhs))
            return true;

        return bits_of(lhs.hi) == bits_of(rhs.hi)
            && bits_of(lhs.lo) == bits_of(rhs.lo);
    }

    [[nodiscard]] bool exact_match_or_both_nan(const f256_s& lhs, const f256_s& rhs) noexcept
    {
        if (bl::isnan(lhs) && bl::isnan(rhs))
            return true;

        return bits_of(lhs.x0) == bits_of(rhs.x0)
            && bits_of(lhs.x1) == bits_of(rhs.x1)
            && bits_of(lhs.x2) == bits_of(rhs.x2)
            && bits_of(lhs.x3) == bits_of(rhs.x3);
    }

    template<typename T>
    [[nodiscard]] bool exact_match_or_both_nan(const sincos_result<T>& lhs, const sincos_result<T>& rhs) noexcept
    {
        return lhs.ok == rhs.ok
            && exact_match_or_both_nan(lhs.s, rhs.s)
            && exact_match_or_both_nan(lhs.c, rhs.c);
    }

    template<std::size_t Offset, std::size_t Total, std::size_t ChunkSize, typename Runner>
    void run_chunked(Runner&& runner)
    {
        if constexpr (Offset < Total)
        {
            constexpr std::size_t count = (Offset + ChunkSize <= Total) ? ChunkSize : (Total - Offset);
            runner.template operator()<Offset, count>();
            run_chunked<Offset + count, Total, ChunkSize>(std::forward<Runner>(runner));
        }
    }

    template<typename T, const auto& Inputs, std::size_t Offset, std::size_t Count>
    [[nodiscard]] consteval std::array<T, Count> make_parsed_unary_chunk()
    {
        std::array<T, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
            out[index] = parse_from_text<T>(Inputs[Offset + index]);
        return out;
    }

    template<typename T, const auto& Inputs, std::size_t Offset, std::size_t Count>
    [[nodiscard]] consteval std::array<binary_typed_case<T>, Count> make_parsed_binary_chunk()
    {
        std::array<binary_typed_case<T>, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
        {
            const auto& input = Inputs[Offset + index];
            out[index].lhs = parse_from_text<T>(input.lhs);
            out[index].rhs = parse_from_text<T>(input.rhs);
        }
        return out;
    }

    template<typename T, const auto& Inputs, std::size_t Offset, std::size_t Count>
    [[nodiscard]] consteval std::array<ldexp_typed_case<T>, Count> make_parsed_ldexp_chunk()
    {
        std::array<ldexp_typed_case<T>, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
        {
            const auto& input = Inputs[Offset + index];
            out[index].value = parse_from_text<T>(input.value);
            out[index].exponent = input.exponent;
        }
        return out;
    }

    template<typename T, std::size_t Count, auto Function>
    [[nodiscard]] consteval std::array<T, Count> make_unary_results_from_parsed(const std::array<T, Count>& inputs)
    {
        std::array<T, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
            out[index] = Function(inputs[index]);
        return out;
    }

    template<typename T, std::size_t Count, auto Function>
    [[nodiscard]] consteval std::array<T, Count> make_binary_results_from_parsed(const std::array<binary_typed_case<T>, Count>& inputs)
    {
        std::array<T, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
            out[index] = Function(inputs[index].lhs, inputs[index].rhs);
        return out;
    }

    template<typename T, std::size_t Count>
    [[nodiscard]] consteval std::array<sincos_result<T>, Count> make_sincos_results_from_parsed(const std::array<T, Count>& inputs)
    {
        std::array<sincos_result<T>, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
        {
            T s{};
            T c{};
            out[index].ok = bl::sincos(inputs[index], s, c);
            out[index].s = s;
            out[index].c = c;
        }
        return out;
    }

    template<typename T, std::size_t Count>
    [[nodiscard]] consteval std::array<T, Count> make_ldexp_results_from_parsed(const std::array<ldexp_typed_case<T>, Count>& inputs)
    {
        std::array<T, Count> out{};
        for (std::size_t index = 0; index < Count; ++index)
            out[index] = bl::ldexp(inputs[index].value, inputs[index].exponent);
        return out;
    }

    template<typename T>
    constexpr T op_add(const T& lhs, const T& rhs) { return lhs + rhs; }
    template<typename T>
    constexpr T op_sub(const T& lhs, const T& rhs) { return lhs - rhs; }
    template<typename T>
    constexpr T op_mul(const T& lhs, const T& rhs) { return lhs * rhs; }
    template<typename T>
    constexpr T op_div(const T& lhs, const T& rhs) { return lhs / rhs; }

    template<typename T>
    constexpr T op_floor(const T& value) { return bl::floor(value); }
    template<typename T>
    constexpr T op_ceil(const T& value) { return bl::ceil(value); }
    template<typename T>
    constexpr T op_trunc(const T& value) { return bl::trunc(value); }
    template<typename T>
    constexpr T op_round(const T& value) { return bl::round(value); }
    template<typename T>
    constexpr T op_nearbyint(const T& value) { return bl::nearbyint(value); }
    template<typename T>
    constexpr T op_sqrt(const T& value) { return bl::sqrt(value); }
    template<typename T>
    constexpr T op_log(const T& value) { return bl::log(value); }
    template<typename T>
    constexpr T op_log2(const T& value) { return bl::log2(value); }
    template<typename T>
    constexpr T op_log10(const T& value) { return bl::log10(value); }
    template<typename T>
    constexpr T op_exp(const T& value) { return bl::exp(value); }
    template<typename T>
    constexpr T op_exp2(const T& value) { return bl::exp2(value); }
    template<typename T>
    constexpr T op_sin(const T& value) { return bl::sin(value); }
    template<typename T>
    constexpr T op_cos(const T& value) { return bl::cos(value); }
    template<typename T>
    constexpr T op_tan(const T& value) { return bl::tan(value); }
    template<typename T>
    constexpr T op_atan(const T& value) { return bl::atan(value); }
    template<typename T>
    constexpr T op_asin(const T& value) { return bl::asin(value); }
    template<typename T>
    constexpr T op_acos(const T& value) { return bl::acos(value); }

    template<typename T>
    constexpr T op_pow(const T& lhs, const T& rhs) { return bl::pow(lhs, rhs); }
    template<typename T>
    constexpr T op_fmod(const T& lhs, const T& rhs) { return bl::fmod(lhs, rhs); }
    template<typename T>
    constexpr T op_atan2(const T& lhs, const T& rhs) { return bl::atan2(lhs, rhs); }
    template<typename T>
    constexpr T op_remainder(const T& lhs, const T& rhs) { return bl::remainder(lhs, rhs); }

    template<typename T, std::size_t Count, typename RuntimeFunction>
    void compare_unary_chunk(
        const char* operation_name,
        const std::array<T, Count>& inputs,
        const std::array<T, Count>& constexpr_results,
        RuntimeFunction runtime_function,
        std::size_t global_offset)
    {
        for (std::size_t index = 0; index < Count; ++index)
        {
            const T runtime_result = runtime_function(inputs[index]);

            INFO("type: " << type_name<T>());
            INFO("operation: " << operation_name);
            INFO("index: " << (global_offset + index));
            INFO("input_value: " << describe_value(inputs[index]));
            INFO("constexpr_result: " << describe_value(constexpr_results[index]));
            INFO("runtime_result:   " << describe_value(runtime_result));

            CHECK(exact_match_or_both_nan(constexpr_results[index], runtime_result));
        }
    }

    template<typename T, std::size_t Count, typename RuntimeFunction>
    void compare_binary_chunk(
        const char* operation_name,
        const std::array<binary_typed_case<T>, Count>& inputs,
        const std::array<T, Count>& constexpr_results,
        RuntimeFunction runtime_function,
        std::size_t global_offset)
    {
        for (std::size_t index = 0; index < Count; ++index)
        {
            const T runtime_result = runtime_function(inputs[index].lhs, inputs[index].rhs);

            INFO("type: " << type_name<T>());
            INFO("operation: " << operation_name);
            INFO("index: " << (global_offset + index));
            INFO("lhs_value: " << describe_value(inputs[index].lhs));
            INFO("rhs_value: " << describe_value(inputs[index].rhs));
            INFO("constexpr_result: " << describe_value(constexpr_results[index]));
            INFO("runtime_result:   " << describe_value(runtime_result));

            CHECK(exact_match_or_both_nan(constexpr_results[index], runtime_result));
        }
    }

    template<typename T, std::size_t Count>
    void compare_sincos_chunk(
        const char* operation_name,
        const std::array<T, Count>& inputs,
        const std::array<sincos_result<T>, Count>& constexpr_results,
        std::size_t global_offset)
    {
        for (std::size_t index = 0; index < Count; ++index)
        {
            sincos_result<T> runtime_result{};
            runtime_result.ok = bl::sincos(inputs[index], runtime_result.s, runtime_result.c);

            INFO("type: " << type_name<T>());
            INFO("operation: " << operation_name);
            INFO("index: " << (global_offset + index));
            INFO("input_value: " << describe_value(inputs[index]));
            INFO("constexpr_result: " << describe_value(constexpr_results[index]));
            INFO("runtime_result:   " << describe_value(runtime_result));

            CHECK(exact_match_or_both_nan(constexpr_results[index], runtime_result));
        }
    }

    template<typename T, std::size_t Count>
    void compare_ldexp_chunk(
        const char* operation_name,
        const std::array<ldexp_typed_case<T>, Count>& inputs,
        const std::array<T, Count>& constexpr_results,
        std::size_t global_offset)
    {
        for (std::size_t index = 0; index < Count; ++index)
        {
            const T runtime_result = bl::ldexp(inputs[index].value, inputs[index].exponent);

            INFO("type: " << type_name<T>());
            INFO("operation: " << operation_name);
            INFO("index: " << (global_offset + index));
            INFO("input_value: " << describe_value(inputs[index].value));
            INFO("exponent: " << inputs[index].exponent);
            INFO("constexpr_result: " << describe_value(constexpr_results[index]));
            INFO("runtime_result:   " << describe_value(runtime_result));

            CHECK(exact_match_or_both_nan(constexpr_results[index], runtime_result));
        }
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize, std::size_t Begin, std::size_t Requested>
    void check_parse_subset(const char* label)
    {
        static_assert(Begin + Requested <= std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>);

        run_chunked<0, Requested, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto constexpr_parsed = make_parsed_unary_chunk<T, Inputs, Begin + Offset, Count>();

            for (std::size_t index = 0; index < Count; ++index)
            {
                const char* text = Inputs[Begin + Offset + index];
                const T runtime_parsed = parse_from_text<T>(text);

                INFO("type: " << type_name<T>());
                INFO("operation: parse");
                INFO("label: " << label);
                INFO("index: " << (Offset + index));
                INFO("input_text: " << text);
                INFO("constexpr_result: " << describe_value(constexpr_parsed[index]));
                INFO("runtime_result:   " << describe_value(runtime_parsed));

                CHECK(exact_match_or_both_nan(constexpr_parsed[index], runtime_parsed));
            }
        });
    }

    template<typename T>
    void check_parse_stress_samples()
    {
        check_parse_subset<T, trig_cases, medium_chunk_size, 0, 40>("trig_cases[0..39]");
        check_parse_subset<T, exp_cases, medium_chunk_size, 0, 30>("exp_cases[0..29]");
        check_parse_subset<T, atan_cases, medium_chunk_size, 0, 30>("atan_cases[0..29]");
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_rounding_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto floor_results = make_unary_results_from_parsed<T, Count, op_floor<T>>(parsed);
            constexpr auto ceil_results = make_unary_results_from_parsed<T, Count, op_ceil<T>>(parsed);
            constexpr auto trunc_results = make_unary_results_from_parsed<T, Count, op_trunc<T>>(parsed);
            constexpr auto round_results = make_unary_results_from_parsed<T, Count, op_round<T>>(parsed);
            constexpr auto nearbyint_results = make_unary_results_from_parsed<T, Count, op_nearbyint<T>>(parsed);

            compare_unary_chunk("floor", parsed, floor_results, [](const T& value) { return bl::floor(value); }, Offset);
            compare_unary_chunk("ceil", parsed, ceil_results, [](const T& value) { return bl::ceil(value); }, Offset);
            compare_unary_chunk("trunc", parsed, trunc_results, [](const T& value) { return bl::trunc(value); }, Offset);
            compare_unary_chunk("round", parsed, round_results, [](const T& value) { return bl::round(value); }, Offset);
            compare_unary_chunk("nearbyint", parsed, nearbyint_results, [](const T& value) { return bl::nearbyint(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_positive_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto log_results = make_unary_results_from_parsed<T, Count, op_log<T>>(parsed);
            constexpr auto log2_results = make_unary_results_from_parsed<T, Count, op_log2<T>>(parsed);
            constexpr auto log10_results = make_unary_results_from_parsed<T, Count, op_log10<T>>(parsed);

            compare_unary_chunk("log", parsed, log_results, [](const T& value) { return bl::log(value); }, Offset);
            compare_unary_chunk("log2", parsed, log2_results, [](const T& value) { return bl::log2(value); }, Offset);
            compare_unary_chunk("log10", parsed, log10_results, [](const T& value) { return bl::log10(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_exp_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto exp_results = make_unary_results_from_parsed<T, Count, op_exp<T>>(parsed);
            constexpr auto exp2_results = make_unary_results_from_parsed<T, Count, op_exp2<T>>(parsed);

            compare_unary_chunk("exp", parsed, exp_results, [](const T& value) { return bl::exp(value); }, Offset);
            compare_unary_chunk("exp2", parsed, exp2_results, [](const T& value) { return bl::exp2(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_trig_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto sin_results = make_unary_results_from_parsed<T, Count, op_sin<T>>(parsed);
            constexpr auto cos_results = make_unary_results_from_parsed<T, Count, op_cos<T>>(parsed);
            constexpr auto tan_results = make_unary_results_from_parsed<T, Count, op_tan<T>>(parsed);
            constexpr auto sincos_results = make_sincos_results_from_parsed<T, Count>(parsed);

            compare_unary_chunk("sin", parsed, sin_results, [](const T& value) { return bl::sin(value); }, Offset);
            compare_unary_chunk("cos", parsed, cos_results, [](const T& value) { return bl::cos(value); }, Offset);
            compare_unary_chunk("tan", parsed, tan_results, [](const T& value) { return bl::tan(value); }, Offset);
            compare_sincos_chunk("sincos", parsed, sincos_results, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_inverse_trig_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto asin_results = make_unary_results_from_parsed<T, Count, op_asin<T>>(parsed);
            constexpr auto acos_results = make_unary_results_from_parsed<T, Count, op_acos<T>>(parsed);

            compare_unary_chunk("asin", parsed, asin_results, [](const T& value) { return bl::asin(value); }, Offset);
            compare_unary_chunk("acos", parsed, acos_results, [](const T& value) { return bl::acos(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_atan_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto atan_results = make_unary_results_from_parsed<T, Count, op_atan<T>>(parsed);
            compare_unary_chunk("atan", parsed, atan_results, [](const T& value) { return bl::atan(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_sqrt_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_unary_chunk<T, Inputs, Offset, Count>();
            constexpr auto sqrt_results = make_unary_results_from_parsed<T, Count, op_sqrt<T>>(parsed);
            compare_unary_chunk("sqrt", parsed, sqrt_results, [](const T& value) { return bl::sqrt(value); }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_arithmetic_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_binary_chunk<T, Inputs, Offset, Count>();
            constexpr auto add_results = make_binary_results_from_parsed<T, Count, op_add<T>>(parsed);
            constexpr auto sub_results = make_binary_results_from_parsed<T, Count, op_sub<T>>(parsed);
            constexpr auto mul_results = make_binary_results_from_parsed<T, Count, op_mul<T>>(parsed);
            constexpr auto div_results = make_binary_results_from_parsed<T, Count, op_div<T>>(parsed);

            compare_binary_chunk("operator+", parsed, add_results, [](const T& lhs, const T& rhs) { return lhs + rhs; }, Offset);
            compare_binary_chunk("operator-", parsed, sub_results, [](const T& lhs, const T& rhs) { return lhs - rhs; }, Offset);
            compare_binary_chunk("operator*", parsed, mul_results, [](const T& lhs, const T& rhs) { return lhs * rhs; }, Offset);
            compare_binary_chunk("operator/", parsed, div_results, [](const T& lhs, const T& rhs) { return lhs / rhs; }, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize, auto Function, typename RuntimeFunction>
    void check_binary_family(const char* operation_name, RuntimeFunction runtime_function)
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_binary_chunk<T, Inputs, Offset, Count>();
            constexpr auto results = make_binary_results_from_parsed<T, Count, Function>(parsed);
            compare_binary_chunk(operation_name, parsed, results, runtime_function, Offset);
        });
    }

    template<typename T, const auto& Inputs, std::size_t ChunkSize>
    void check_ldexp_family()
    {
        constexpr std::size_t total_count = std::tuple_size_v<std::remove_cvref_t<decltype(Inputs)>>;
        run_chunked<0, total_count, ChunkSize>([&]<std::size_t Offset, std::size_t Count>()
        {
            constexpr auto parsed = make_parsed_ldexp_chunk<T, Inputs, Offset, Count>();
            constexpr auto results = make_ldexp_results_from_parsed<T, Count>(parsed);
            compare_ldexp_chunk("ldexp", parsed, results, Offset);
        });
    }
}

TEST_CASE("f128 constexpr results exactly match runtime results (reused parsed values)", "[fltx][constexpr][f128][stress]")
{
    check_parse_stress_samples<f128_s>();

    check_arithmetic_family<f128_s, f128_arithmetic_cases, heavy_chunk_size>();
    check_rounding_family<f128_s, rounding_cases, medium_chunk_size>();
    check_sqrt_family<f128_s, sqrt_cases, medium_chunk_size>();
    check_positive_family<f128_s, positive_cases, heavy_chunk_size>();
    check_exp_family<f128_s, exp_cases, heavy_chunk_size>();
    check_trig_family<f128_s, trig_cases, heavy_chunk_size>();
    check_atan_family<f128_s, atan_cases, heavy_chunk_size>();
    check_inverse_trig_family<f128_s, inverse_trig_cases, heavy_chunk_size>();

    check_binary_family<f128_s, f128_pow_cases, heavy_chunk_size, op_pow<f128_s>>(
        "pow", [](const f128_s& lhs, const f128_s& rhs) { return bl::pow(lhs, rhs); });
    check_binary_family<f128_s, f128_fmod_cases, heavy_chunk_size, op_fmod<f128_s>>(
        "fmod", [](const f128_s& lhs, const f128_s& rhs) { return bl::fmod(lhs, rhs); });
    check_binary_family<f128_s, f128_atan2_cases, heavy_chunk_size, op_atan2<f128_s>>(
        "atan2", [](const f128_s& lhs, const f128_s& rhs) { return bl::atan2(lhs, rhs); });
    check_binary_family<f128_s, f128_remainder_cases, heavy_chunk_size, op_remainder<f128_s>>(
        "remainder", [](const f128_s& lhs, const f128_s& rhs) { return bl::remainder(lhs, rhs); });

    check_ldexp_family<f128_s, f128_ldexp_cases, medium_chunk_size>();
}

TEST_CASE("f256 constexpr results exactly match runtime results (reused parsed values)", "[fltx][constexpr][f256][stress]")
{
    check_parse_stress_samples<f256_s>();

    check_arithmetic_family<f256_s, f256_arithmetic_cases, heavy_chunk_size>();
    check_rounding_family<f256_s, rounding_cases, medium_chunk_size>();
    check_sqrt_family<f256_s, sqrt_cases, medium_chunk_size>();
    check_positive_family<f256_s, positive_cases, heavy_chunk_size>();
    check_exp_family<f256_s, exp_cases, heavy_chunk_size>();
    check_trig_family<f256_s, trig_cases, heavy_chunk_size>();
    check_atan_family<f256_s, atan_cases, heavy_chunk_size>();
    check_inverse_trig_family<f256_s, inverse_trig_cases, heavy_chunk_size>();

    check_binary_family<f256_s, f256_pow_cases, heavy_chunk_size, op_pow<f256_s>>(
        "pow", [](const f256_s& lhs, const f256_s& rhs) { return bl::pow(lhs, rhs); });
    check_binary_family<f256_s, f256_fmod_cases, heavy_chunk_size, op_fmod<f256_s>>(
        "fmod", [](const f256_s& lhs, const f256_s& rhs) { return bl::fmod(lhs, rhs); });
    check_binary_family<f256_s, f256_atan2_cases, heavy_chunk_size, op_atan2<f256_s>>(
        "atan2", [](const f256_s& lhs, const f256_s& rhs) { return bl::atan2(lhs, rhs); });

    check_ldexp_family<f256_s, f256_ldexp_cases, medium_chunk_size>();
}
