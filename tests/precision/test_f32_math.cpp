
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <f32_math.h>

using namespace bl;

namespace
{
    constexpr int checked_digits = std::numeric_limits<float>::digits10;
    constexpr int printed_digits = std::numeric_limits<float>::max_digits10;

    constexpr std::uint64_t random_seed = 1ull;
    constexpr int random_sample_count = 1000;
    constexpr const char* type_label = "f32";

    struct accuracy_stats_entry
    {
        int samples = 0;
        int passed = 0;
        std::vector<float> achieved_digits;
        std::uint32_t worst_ulp = 0;
    };

    class accuracy_report_scope;
    thread_local accuracy_report_scope* current_accuracy_report_scope = nullptr;

    [[nodiscard]] std::string to_text(float value)
    {
        if (std::isnan(value))
            return "nan";
        if (std::isinf(value))
            return std::signbit(value) ? "-inf" : "inf";

        std::ostringstream out;
        out << std::setprecision(printed_digits) << std::defaultfloat << value;
        return out.str();
    }

    [[nodiscard]] std::string to_text_hex(float value)
    {
        std::ostringstream out;
        out << std::hexfloat << value;
        return out.str();
    }

    [[nodiscard]] float abs_ref(float value)
    {
        return value < 0.0 ? -value : value;
    }

    [[nodiscard]] float achieved_digits_from_error(float diff, float scale)
    {
        if (diff == 0.0)
            return static_cast<float>(checked_digits);

        const float scaled_error = diff / scale;
        if (!(scaled_error > 0.0))
            return static_cast<float>(checked_digits);

        if (scaled_error >= 1.0)
            return 0.0;

        const float digits = -std::log10(scaled_error);
        return digits < 0.0 ? 0.0 : digits;
    }

    [[nodiscard]] float normalized_accuracy_percent(float digits)
    {
        if (checked_digits <= 0)
            return 100.0;

        float ratio = digits / static_cast<float>(checked_digits);
        if (ratio < 0.0)
            ratio = 0.0;
        if (ratio > 1.0)
            ratio = 1.0;
        return ratio * 100.0;
    }

    [[nodiscard]] float median_digits(std::vector<float> values)
    {
        if (values.empty())
            return 0.0;

        std::sort(values.begin(), values.end());
        const std::size_t mid = values.size() / 2;
        if ((values.size() & 1u) != 0u)
            return values[mid];

        return (values[mid - 1] + values[mid]) * 0.5;
    }

    [[nodiscard]] std::uint32_t ordered_bits(float value)
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        if ((bits & 0x80000000u) != 0u)
            return ~bits;

        return bits | 0x80000000u;
    }

    [[nodiscard]] std::uint32_t ulp_distance(float a, float b)
    {
        const std::uint32_t oa = ordered_bits(a);
        const std::uint32_t ob = ordered_bits(b);
        return (oa >= ob) ? (oa - ob) : (ob - oa);
    }

    class accuracy_report_scope
    {
    public:
        explicit accuracy_report_scope(const char* test_name)
            : test_name(test_name), previous(current_accuracy_report_scope)
        {
            current_accuracy_report_scope = this;
        }

        ~accuracy_report_scope()
        {
            current_accuracy_report_scope = previous;

            if (stats.empty())
                return;

            const std::ios_base::fmtflags old_flags = std::cout.flags();
            const std::streamsize old_precision = std::cout.precision();

            std::cout << "\naccuracy summary for " << test_name << ":\n";
            std::cout << std::fixed << std::setprecision(2);

            for (const auto& [op_name, entry] : stats)
            {
                const float median = median_digits(entry.achieved_digits);
                const float worst = *std::min_element(entry.achieved_digits.begin(), entry.achieved_digits.end());

                std::cout << "  " << op_name
                          << ": pass " << entry.passed << "/" << entry.samples
                          << ", median " << median << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(median) << "%)"
                          << ", worst " << worst << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(worst) << "%)"
                          << ", worst ulp " << entry.worst_ulp
                          << "\n";
            }

            std::cout.flags(old_flags);
            std::cout.precision(old_precision);
        }

        void record(const char* op_name, float diff, float scale, std::uint32_t ulp_diff, bool passed)
        {
            auto& entry = stats[op_name];
            ++entry.samples;
            if (passed)
                ++entry.passed;
            entry.achieved_digits.push_back(achieved_digits_from_error(diff, scale));
            if (ulp_diff > entry.worst_ulp)
                entry.worst_ulp = ulp_diff;
        }

    private:
        std::string test_name;
        accuracy_report_scope* previous = nullptr;
        std::map<std::string, accuracy_stats_entry> stats;
    };

    void record_accuracy_sample(const char* op_name, float diff, float scale, std::uint32_t ulp_diff, bool passed)
    {
        if (current_accuracy_report_scope != nullptr)
            current_accuracy_report_scope->record(op_name, diff, scale, ulp_diff, passed);
    }

    struct tolerance_spec
    {
        float abs_tolerance = 0.0;
        float rel_tolerance = 0.0;
        std::uint32_t max_ulps = 0;
    };

    [[nodiscard]] bool both_nan(float a, float b)
    {
        return std::isnan(a) && std::isnan(b);
    }

    [[nodiscard]] bool same_zero_with_same_sign(float a, float b)
    {
        return a == 0.0 && b == 0.0 && std::signbit(a) == std::signbit(b);
    }

    [[nodiscard]] bool same_infinity_with_same_sign(float a, float b)
    {
        return std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b);
    }

    [[nodiscard]] bool same_bits(float a, float b)
    {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }

    struct floating_compare_result
    {
        bool passed = false;
        float signed_diff = 0.0;
        float abs_diff = 0.0;
        float rel_diff = 0.0;
        float scale = 1.0;
        float allowed_diff = 0.0;
        float achieved_digits = 0.0;
        std::uint32_t ulp_diff = 0;
        const char* reason = "";
    };

    [[nodiscard]] floating_compare_result compare_floating_result(
        const char* op_name,
        float got,
        float expected,
        const tolerance_spec& tolerance)
    {
        floating_compare_result result{};

        if (both_nan(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "both NaN";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (same_infinity_with_same_sign(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "same signed infinity";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (same_zero_with_same_sign(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "same signed zero";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (got == 0.0 && expected == 0.0)
        {
            result.passed = false;
            result.abs_diff = 0.0;
            result.scale = 1.0;
            result.allowed_diff = tolerance.abs_tolerance;
            result.ulp_diff = 1;
            result.reason = "zero sign mismatch";
            record_accuracy_sample(op_name, 0.0, 1.0, 1, false);
            return result;
        }

        if (same_bits(got, expected))
        {
            result.passed = true;
            result.scale = std::max(1.0f, abs_ref(expected));
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "bitwise equal";
            record_accuracy_sample(op_name, 0.0, result.scale, 0, true);
            return result;
        }

        if (!std::isfinite(got) || !std::isfinite(expected))
        {
            result.passed = false;
            result.abs_diff = std::numeric_limits<float>::infinity();
            result.rel_diff = std::numeric_limits<float>::infinity();
            result.scale = 1.0;
            result.allowed_diff = tolerance.abs_tolerance;
            result.ulp_diff = std::numeric_limits<std::uint32_t>::max();
            result.reason = "non-finite mismatch";
            record_accuracy_sample(op_name, std::numeric_limits<float>::infinity(), 1.0, std::numeric_limits<std::uint32_t>::max(), false);
            return result;
        }

        result.signed_diff = got - expected;
        result.abs_diff = abs_ref(result.signed_diff);
        result.scale = abs_ref(expected);
        if (result.scale < 1.0)
            result.scale = 1.0;

        const float rel_based_tolerance = tolerance.rel_tolerance * result.scale;
        result.allowed_diff = tolerance.abs_tolerance;
        if (rel_based_tolerance > result.allowed_diff)
            result.allowed_diff = rel_based_tolerance;

        if (expected == 0.0)
            result.rel_diff = (result.abs_diff == 0.0) ? 0.0 : std::numeric_limits<float>::infinity();
        else
            result.rel_diff = result.abs_diff / abs_ref(expected);

        result.ulp_diff = ulp_distance(got, expected);
        result.passed = (result.abs_diff <= result.allowed_diff) || (result.ulp_diff <= tolerance.max_ulps);
        result.achieved_digits = achieved_digits_from_error(result.abs_diff, result.scale);
        result.reason = result.passed ? "within tolerance" : "outside tolerance";

        record_accuracy_sample(op_name, result.abs_diff, result.scale, result.ulp_diff, result.passed);
        return result;
    }

    [[nodiscard]] std::string build_comparison_message(
        const char* op_name,
        float got,
        float expected,
        const tolerance_spec& tolerance,
        const floating_compare_result& result)
    {
        std::ostringstream out;
        out << op_name << " mismatch"
            << "\n  got: " << to_text(got)
            << "\n  expected: " << to_text(expected)
            << "\n  got hex: " << to_text_hex(got)
            << "\n  expected hex: " << to_text_hex(expected)
            << "\n  signed diff (got - expected): " << to_text(result.signed_diff)
            << "\n  abs diff: " << to_text(result.abs_diff)
            << "\n  relative diff: " << to_text(result.rel_diff)
            << "\n  allowed abs diff: " << to_text(result.allowed_diff)
            << "\n  ulp diff: " << result.ulp_diff
            << " (limit " << tolerance.max_ulps << ")"
            << "\n  achieved digits: " << result.achieved_digits << "/" << checked_digits
            << "\n  reason: " << result.reason;
        return out.str();
    }

    template<typename F64Op, typename StdOp>
    void check_unary_op(
        const char* op_name,
        float input,
        const tolerance_spec& tolerance,
        F64Op&& f32_op,
        StdOp&& std_op)
    {
        const float got = f32_op(input);
        const float expected = std_op(input);

        INFO(op_name
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input));

        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, tolerance);
        INFO(build_comparison_message(op_name, got, expected, tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    template<typename F64Op, typename StdOp>
    void check_binary_op(
        const char* op_name,
        float lhs,
        float rhs,
        const tolerance_spec& tolerance,
        F64Op&& f32_op,
        StdOp&& std_op)
    {
        const float got = f32_op(lhs, rhs);
        const float expected = std_op(lhs, rhs);

        INFO(op_name
            << "\n  lhs: " << to_text(lhs)
            << "\n  rhs: " << to_text(rhs)
            << "\n  lhs hex: " << to_text_hex(lhs)
            << "\n  rhs hex: " << to_text_hex(rhs));

        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, tolerance);
        INFO(build_comparison_message(op_name, got, expected, tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    template<typename Int>
    void check_exact_integer_result(const char* op_name, Int got, Int expected, float input)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(input));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    template<typename Int>
    void check_exact_integer_result(const char* op_name, Int got, Int expected, float lhs, float rhs)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(lhs));
        CAPTURE(to_text(rhs));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    void check_exact_bool_result(const char* op_name, bool got, bool expected, float lhs, float rhs)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(lhs));
        CAPTURE(to_text(rhs));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    void check_frexp_result(float input)
    {
        int got_exp = 0;
        int expected_exp = 0;

        const float got = bl::frexp(input, &got_exp);
        const float expected = std::frexp(input, &expected_exp);

        INFO("frexp"
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input)
            << "\n  got exp: " << got_exp
            << "\n  expected exp: " << expected_exp);

        const floating_compare_result comparison = compare_floating_result("frexp", got, expected, tolerance_spec{});
        INFO(build_comparison_message("frexp", got, expected, tolerance_spec{}, comparison));
        REQUIRE(comparison.passed);
        REQUIRE(got_exp == expected_exp);
    }

    void check_modf_result(float input)
    {
        float got_int = 0.0;
        float expected_int = 0.0;

        const float got = bl::modf(input, &got_int);
        const float expected = std::modf(input, &expected_int);

        INFO("modf"
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input)
            << "\n  got int: " << to_text(got_int)
            << "\n  expected int: " << to_text(expected_int));

        const floating_compare_result frac_comparison = compare_floating_result("modf.frac", got, expected, tolerance_spec{});
        INFO(build_comparison_message("modf.frac", got, expected, tolerance_spec{}, frac_comparison));
        REQUIRE(frac_comparison.passed);

        const floating_compare_result int_comparison = compare_floating_result("modf.int", got_int, expected_int, tolerance_spec{});
        INFO(build_comparison_message("modf.int", got_int, expected_int, tolerance_spec{}, int_comparison));
        REQUIRE(int_comparison.passed);
    }

    void check_remquo_result(float lhs, float rhs)
    {
        int got_quo = 0;
        int expected_quo = 0;

        const float got = bl::remquo(lhs, rhs, &got_quo);
        const float expected = std::remquo(lhs, rhs, &expected_quo);

        INFO("remquo"
            << "\n  lhs: " << to_text(lhs)
            << "\n  rhs: " << to_text(rhs)
            << "\n  lhs hex: " << to_text_hex(lhs)
            << "\n  rhs hex: " << to_text_hex(rhs)
            << "\n  got quo: " << got_quo
            << "\n  expected quo: " << expected_quo);

        const floating_compare_result comparison = compare_floating_result("remquo", got, expected, tolerance_spec{});
        INFO(build_comparison_message("remquo", got, expected, tolerance_spec{}, comparison));
        REQUIRE(comparison.passed);

        if (!std::isnan(expected))
        {
            const int got_sign = (got_quo > 0) - (got_quo < 0);
            const int expected_sign = (expected_quo > 0) - (expected_quo < 0);

            CAPTURE(got_sign);
            CAPTURE(expected_sign);
            CAPTURE(got_quo & 0x7);
            CAPTURE(expected_quo & 0x7);

            REQUIRE(got_sign == expected_sign);
            REQUIRE((got_quo & 0x7) == (expected_quo & 0x7));
        }
    }

    void print_random_run(const char* description, int count)
    {
        std::cout << type_label << " comparing: " << count << " " << description
                  << " (seed " << random_seed << ")...\n\n";
    }

    [[nodiscard]] float random_finite_for_f32(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> exponent_dist(-149, 127);
        std::uniform_real_distribution<float> mantissa_dist(0.5f, 1.0f);

        float value = std::ldexp(mantissa_dist(rng), exponent_dist(rng));
        if (!std::isfinite(value) || value == 0.0f)
            value = std::numeric_limits<float>::denorm_min();
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] float random_signed_interval(std::mt19937_64& rng, float limit)
    {
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(rng);
    }

    [[nodiscard]] float random_unit_interval(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        return dist(rng);
    }

    [[nodiscard]] float random_positive(std::mt19937_64& rng)
    {
        float value = std::fabs(random_finite_for_f32(rng));
        if (value < std::numeric_limits<float>::denorm_min())
            value = std::numeric_limits<float>::denorm_min();
        return value;
    }

    [[nodiscard]] float random_nonzero(std::mt19937_64& rng)
    {
        float value = random_finite_for_f32(rng);
        if (value == 0.0)
            value = 1.0;
        return value;
    }

    [[nodiscard]] float random_log1p_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(-0.95f, 16.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_acosh_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(1.0f, 64.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_atanh_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(-0.95f, 0.95f);
        return dist(rng);
    }

    [[nodiscard]] float random_pow_base(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.125f, 8.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_gamma_positive(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.125f, 15.0f);
        return dist(rng);
    }

    [[nodiscard]] constexpr tolerance_spec exact_tol()
    {
        return tolerance_spec{};
    }

    [[nodiscard]] constexpr tolerance_spec close_tol(std::uint32_t ulps, float abs_tol = 0.0, float rel_tol = 0.0)
    {
        return tolerance_spec{ abs_tol, rel_tol, ulps };
    }
}

TEST_CASE("f32 matches std for + - * /", "[fltx][f32][precision][arithmetic]")
{
    accuracy_report_scope report("f32 matches std for + - * /");

    constexpr std::array<std::pair<float, float>, 10> cases{{
        { 0.0f, 0.0f },
        { 1.0f, 2.0f },
        { -1.0f, 2.0f },
        { 1.25f, -2.5f },
        { 1e-300f, 1e-200f },
        { -1e-100f, 1e-150f },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::pi, (f32)std::numbers::sqrt2 },
        { 1e30f, -1e-100f },
        { -1e30f, 3.0f }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("add", lhs, rhs, exact_tol(),
            [](float a, float b) { return a + b; },
            [](float a, float b) { return a + b; });

        check_binary_op("subtract", lhs, rhs, exact_tol(),
            [](float a, float b) { return a - b; },
            [](float a, float b) { return a - b; });

        check_binary_op("multiply", lhs, rhs, exact_tol(),
            [](float a, float b) { return a * b; },
            [](float a, float b) { return a * b; });

        if (rhs != 0.0)
        {
            check_binary_op("divide", lhs, rhs, exact_tol(),
                [](float a, float b) { return a / b; },
                [](float a, float b) { return a / b; });
        }
    }
}

TEST_CASE("f32 random arithmetic matches std", "[fltx][f32][precision][arithmetic]")
{
    accuracy_report_scope report("f32 random arithmetic matches std");
    print_random_run("random arithmetic pairs", random_sample_count);

    std::mt19937_64 rng(random_seed);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float lhs = random_finite_for_f32(rng);
        const float rhs = random_finite_for_f32(rng);

        check_binary_op("add", lhs, rhs, exact_tol(),
            [](float a, float b) { return a + b; },
            [](float a, float b) { return a + b; });

        check_binary_op("subtract", lhs, rhs, exact_tol(),
            [](float a, float b) { return a - b; },
            [](float a, float b) { return a - b; });

        check_binary_op("multiply", lhs, rhs, exact_tol(),
            [](float a, float b) { return a * b; },
            [](float a, float b) { return a * b; });

        if (rhs != 0.0)
        {
            check_binary_op("divide", lhs, rhs, exact_tol(),
                [](float a, float b) { return a / b; },
                [](float a, float b) { return a / b; });
        }
    }
}

TEST_CASE("f32 trig matches std for fixed values", "[fltx][f32][precision][transcendental][trig]")
{
    accuracy_report_scope report("f32 trig matches std for fixed values");

    constexpr std::array<float, 13> unary_cases{{
        -3.0,
        -1.5,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        1.5,
        3.0,
        std::numbers::pi / 6.0,
        std::numbers::pi / 4.0,
        std::numbers::pi / 3.0
    }};

    for (float input : unary_cases)
    {
        check_unary_op("sin", input, exact_tol(),
            [](float x) { return bl::sin(x); },
            [](float x) { return std::sin(x); });

        check_unary_op("cos", input, exact_tol(),
            [](float x) { return bl::cos(x); },
            [](float x) { return std::cos(x); });

        check_unary_op("tan", input, exact_tol(),
            [](float x) { return bl::tan(x); },
            [](float x) { return std::tan(x); });

        check_unary_op("atan", input, exact_tol(),
            [](float x) { return bl::atan(x); },
            [](float x) { return std::atan(x); });

        if (input >= -1.0 && input <= 1.0)
        {
            check_unary_op("asin", input, close_tol(4),
                [](float x) { return bl::asin(x); },
                [](float x) { return std::asin(x); });

            check_unary_op("acos", input, close_tol(4),
                [](float x) { return bl::acos(x); },
                [](float x) { return std::acos(x); });
        }
    }

    constexpr std::array<std::pair<float, float>, 8> atan2_cases{{
        { 0.0f, 1.0f },
        { 1.0f, 0.0f },
        { -1.0f, 0.0f },
        { 1.0f, 1.0f },
        { -1.0f, 1.0f },
        { 1.0f, -1.0f },
        { -1.0f, -1.0f },
        { (f32)std::numbers::pi, (f32)-std::numbers::e }
    }};

    for (const auto& [y, x] : atan2_cases)
    {
        check_binary_op("atan2", y, x, exact_tol(),
            [](float a, float b) { return bl::atan2(a, b); },
            [](float a, float b) { return std::atan2(a, b); });
    }
}

TEST_CASE("f32 trig matches std on random inputs", "[fltx][f32][precision][transcendental][trig]")
{
    accuracy_report_scope report("f32 trig matches std on random inputs");
    print_random_run("random trig inputs", random_sample_count);

    std::mt19937_64 rng(random_seed);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float trig_input = random_signed_interval(rng, 16.0);
        const float tan_input = random_signed_interval(rng, 1.2);
        const float unit_input = random_signed_interval(rng, 1.0);
        const float y = random_signed_interval(rng, 128.0);
        const float x = random_signed_interval(rng, 128.0);

        check_unary_op("sin", trig_input, exact_tol(),
            [](float v) { return bl::sin(v); },
            [](float v) { return std::sin(v); });

        check_unary_op("cos", trig_input, exact_tol(),
            [](float v) { return bl::cos(v); },
            [](float v) { return std::cos(v); });

        check_unary_op("tan", tan_input, exact_tol(),
            [](float v) { return bl::tan(v); },
            [](float v) { return std::tan(v); });

        check_unary_op("atan", random_signed_interval(rng, 128.0), exact_tol(),
            [](float v) { return bl::atan(v); },
            [](float v) { return std::atan(v); });

        check_unary_op("asin", unit_input, close_tol(4),
            [](float v) { return bl::asin(v); },
            [](float v) { return std::asin(v); });

        check_unary_op("acos", unit_input, close_tol(4),
            [](float v) { return bl::acos(v); },
            [](float v) { return std::acos(v); });

        if (x == 0.0 && y == 0.0)
            continue;

        check_binary_op("atan2", y, x, exact_tol(),
            [](float a, float b) { return bl::atan2(a, b); },
            [](float a, float b) { return std::atan2(a, b); });
    }
}

TEST_CASE("f32 rounding matches std", "[fltx][f32][precision][math][rounding]")
{
    accuracy_report_scope report("f32 rounding matches std");

    constexpr std::array<float, 16> fixed_inputs{{
        -3.75,
        -2.5,
        -1.5,
        -1.0,
        -0.5,
        -0.25,
        -0.0,
        0.0,
        0.25,
        0.5,
        1.0,
        1.5,
        2.5,
        3.75,
        123456.5,
        -123456.5
    }};

    for (float input : fixed_inputs)
    {
        check_unary_op("floor", input, exact_tol(),
            [](float x) { return bl::floor(x); },
            [](float x) { return std::floor(x); });

        check_unary_op("ceil", input, exact_tol(),
            [](float x) { return bl::ceil(x); },
            [](float x) { return std::ceil(x); });

        check_unary_op("trunc", input, exact_tol(),
            [](float x) { return bl::trunc(x); },
            [](float x) { return std::trunc(x); });

        check_unary_op("round", input, exact_tol(),
            [](float x) { return bl::round(x); },
            [](float x) { return std::round(x); });

        check_unary_op("nearbyint", input, exact_tol(),
            [](float x) { return bl::nearbyint(x); },
            [](float x) { return std::nearbyint(x); });

        check_unary_op("rint", input, exact_tol(),
            [](float x) { return bl::rint(x); },
            [](float x) { return std::rint(x); });

        check_exact_integer_result("lround", bl::lround(input), std::lround(input), input);
        check_exact_integer_result("llround", bl::llround(input), std::llround(input), input);
        check_exact_integer_result("lrint", bl::lrint(input), std::lrint(input), input);
        check_exact_integer_result("llrint", bl::llrint(input), std::llrint(input), input);
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random rounding inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float input = random_finite_for_f32(rng);

        check_unary_op("floor", input, exact_tol(),
            [](float x) { return bl::floor(x); },
            [](float x) { return std::floor(x); });

        check_unary_op("ceil", input, exact_tol(),
            [](float x) { return bl::ceil(x); },
            [](float x) { return std::ceil(x); });

        check_unary_op("trunc", input, exact_tol(),
            [](float x) { return bl::trunc(x); },
            [](float x) { return std::trunc(x); });

        check_unary_op("round", input, exact_tol(),
            [](float x) { return bl::round(x); },
            [](float x) { return std::round(x); });

        check_unary_op("nearbyint", input, exact_tol(),
            [](float x) { return bl::nearbyint(x); },
            [](float x) { return std::nearbyint(x); });

        check_unary_op("rint", input, exact_tol(),
            [](float x) { return bl::rint(x); },
            [](float x) { return std::rint(x); });

        if (std::fabs(input) < static_cast<float>(std::numeric_limits<long long>::max()) - 1.0)
        {
            check_exact_integer_result("lround", bl::lround(input), std::lround(input), input);
            check_exact_integer_result("llround", bl::llround(input), std::llround(input), input);
            check_exact_integer_result("lrint", bl::lrint(input), std::lrint(input), input);
            check_exact_integer_result("llrint", bl::llrint(input), std::llrint(input), input);
        }
    }
}

TEST_CASE("f32 exp and log families match std", "[fltx][f32][precision][transcendental]")
{
    accuracy_report_scope report("f32 exp and log families match std");

    constexpr std::array<float, 10> exp_inputs{{
        -20.0,
        -4.0,
        -1.0,
        -0.1,
        0.0,
        0.1,
        1.0,
        4.0,
        10.0,
        20.0
    }};

    for (float input : exp_inputs)
    {
        check_unary_op("exp", input, exact_tol(),
            [](float x) { return bl::exp(x); },
            [](float x) { return std::exp(x); });

        check_unary_op("exp2", input, exact_tol(),
            [](float x) { return bl::exp2(x); },
            [](float x) { return std::exp2(x); });

        check_unary_op("expm1", input, exact_tol(),
            [](float x) { return bl::expm1(x); },
            [](float x) { return std::expm1(x); });
    }

    constexpr std::array<float, 8> log_inputs{{
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.125,
        0.5,
        1.0,
        2.0,
        10.0,
        1e30f
    }};

    for (float input : log_inputs)
    {
        check_unary_op("log", input, exact_tol(),
            [](float x) { return bl::log(x); },
            [](float x) { return std::log(x); });

        check_unary_op("log2", input, exact_tol(),
            [](float x) { return bl::log2(x); },
            [](float x) { return std::log2(x); });

        check_unary_op("log10", input, exact_tol(),
            [](float x) { return bl::log10(x); },
            [](float x) { return std::log10(x); });
    }

    constexpr std::array<float, 8> log1p_inputs{{
        -0.95,
        -0.75,
        -0.5,
        -0.125,
        0.0,
        0.125,
        1.0,
        16.0
    }};

    for (float input : log1p_inputs)
    {
        check_unary_op("log1p", input, exact_tol(),
            [](float x) { return bl::log1p(x); },
            [](float x) { return std::log1p(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random exp/log inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float exp_input = random_signed_interval(rng, 20.0);
        const float log_input = random_positive(rng);
        const float log1p_input = random_log1p_argument(rng);

        check_unary_op("exp", exp_input, exact_tol(),
            [](float x) { return bl::exp(x); },
            [](float x) { return std::exp(x); });

        check_unary_op("exp2", exp_input, exact_tol(),
            [](float x) { return bl::exp2(x); },
            [](float x) { return std::exp2(x); });

        check_unary_op("expm1", exp_input, exact_tol(),
            [](float x) { return bl::expm1(x); },
            [](float x) { return std::expm1(x); });

        check_unary_op("log", log_input, exact_tol(),
            [](float x) { return bl::log(x); },
            [](float x) { return std::log(x); });

        check_unary_op("log2", log_input, exact_tol(),
            [](float x) { return bl::log2(x); },
            [](float x) { return std::log2(x); });

        check_unary_op("log10", log_input, exact_tol(),
            [](float x) { return bl::log10(x); },
            [](float x) { return std::log10(x); });

        check_unary_op("log1p", log1p_input, exact_tol(),
            [](float x) { return bl::log1p(x); },
            [](float x) { return std::log1p(x); });
    }
}

TEST_CASE("f32 root and power functions match std", "[fltx][f32][precision][math]")
{
    accuracy_report_scope report("f32 root and power functions match std");

    constexpr std::array<float, 8> root_inputs{{
        0.0,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.125,
        1.0,
        2.0,
        1000.0,
        1e30f
    }};

    for (float input : root_inputs)
    {
        check_unary_op("sqrt", input, exact_tol(),
            [](float x) { return bl::sqrt(x); },
            [](float x) { return std::sqrt(x); });

        check_unary_op("cbrt", input, exact_tol(),
            [](float x) { return bl::cbrt(x); },
            [](float x) { return std::cbrt(x); });

        check_unary_op("cbrt.neg", -input, exact_tol(),
            [](float x) { return bl::cbrt(x); },
            [](float x) { return std::cbrt(x); });
    }

    constexpr std::array<std::pair<float, float>, 8> hypot_cases{{
        { 0.0f, 0.0f },
        { 3.0f, 4.0f },
        { -3.0f, 4.0f },
        { 1e-200f, 1e-200f },
        { 1e30f, 1e30f },
        { 1e30f, 1.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::sqrt2, (f32)std::numbers::pi }
    }};

    for (const auto& [x, y] : hypot_cases)
    {
        check_binary_op("hypot", x, y, exact_tol(),
            [](float a, float b) { return bl::hypot(a, b); },
            [](float a, float b) { return std::hypot(a, b); });
    }

    constexpr std::array<std::pair<float, float>, 10> pow_cases{{
        { 2.0f, 3.0f },
        { 2.0f, -3.0f },
        { 0.5f, 0.5f },
        { 3.0f, 1.5f },
        { 10.0f, -2.0f },
        { -2.0f, 3.0f },
        { -2.0f, 4.0f },
        { -0.0f, 3.0f },
        { -0.0f, 4.0f },
        { 0.0f, 0.0f }
    }};

    for (const auto& [base, exponent] : pow_cases)
    {
        check_binary_op("pow", base, exponent, close_tol(8),
            [](float a, float b) { return bl::pow(a, b); },
            [](float a, float b) { return std::pow(a, b); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random root/pow inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float positive = random_positive(rng);
        const float signed_input = random_signed_interval(rng, 1e12);
        const float x = random_signed_interval(rng, 1e12);
        const float y = random_signed_interval(rng, 1e12);
        const float base = random_pow_base(rng);
        const float exponent = random_signed_interval(rng, 8.0);

        check_unary_op("sqrt", positive, exact_tol(),
            [](float v) { return bl::sqrt(v); },
            [](float v) { return std::sqrt(v); });

        check_unary_op("cbrt", signed_input, exact_tol(),
            [](float v) { return bl::cbrt(v); },
            [](float v) { return std::cbrt(v); });

        check_binary_op("hypot", x, y, exact_tol(),
            [](float a, float b) { return bl::hypot(a, b); },
            [](float a, float b) { return std::hypot(a, b); });

        check_binary_op("pow", base, exponent, close_tol(8),
            [](float a, float b) { return bl::pow(a, b); },
            [](float a, float b) { return std::pow(a, b); });
    }
}

TEST_CASE("f32 fmod and remainder match std", "[fltx][f32][precision][math][fmod][remainder]")
{
    accuracy_report_scope report("f32 fmod and remainder match std");

    constexpr std::array<std::pair<float, float>, 10> cases{{
        { 5.25f, 2.0f },
        { -5.25f, 2.0f },
        { 5.25f, -2.0f },
        { -5.25f, -2.0f },
        { 1e30f, 3.0f },
        { 1e-100f, 3.0f },
        { (f32)std::numbers::pi, 0.5f },
        { (f32)-std::numbers::pi, 0.5f },
        { 17.0f, 0.25f },
        { -17.0f, 0.25f }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("fmod", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmod(a, b); },
            [](float a, float b) { return std::fmod(a, b); });

        check_binary_op("remainder", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::remainder(a, b); },
            [](float a, float b) { return std::remainder(a, b); });

        check_remquo_result(lhs, rhs);
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random fmod/remainder inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float lhs = random_finite_for_f32(rng);
        const float rhs = random_nonzero(rng);

        check_binary_op("fmod", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmod(a, b); },
            [](float a, float b) { return std::fmod(a, b); });

        check_binary_op("remainder", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::remainder(a, b); },
            [](float a, float b) { return std::remainder(a, b); });

        check_remquo_result(lhs, rhs);
    }
}

TEST_CASE("f32 hyperbolic families match std", "[fltx][f32][precision][transcendental][hyperbolic]")
{
    accuracy_report_scope report("f32 hyperbolic families match std");

    constexpr std::array<float, 9> fixed_inputs{{
        -8.0,
        -2.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        2.0,
        4.0,
        8.0
    }};

    for (float input : fixed_inputs)
    {
        check_unary_op("sinh", input, exact_tol(),
            [](float x) { return bl::sinh(x); },
            [](float x) { return std::sinh(x); });

        check_unary_op("cosh", input, exact_tol(),
            [](float x) { return bl::cosh(x); },
            [](float x) { return std::cosh(x); });

        check_unary_op("tanh", input, exact_tol(),
            [](float x) { return bl::tanh(x); },
            [](float x) { return std::tanh(x); });

        check_unary_op("asinh", input, exact_tol(),
            [](float x) { return bl::asinh(x); },
            [](float x) { return std::asinh(x); });
    }

    constexpr std::array<float, 6> acosh_inputs{{ 1.0, 1.125, 1.5, 2.0, 10.0, 64.0 }};
    for (float input : acosh_inputs)
    {
        check_unary_op("acosh", input, exact_tol(),
            [](float x) { return bl::acosh(x); },
            [](float x) { return std::acosh(x); });
    }

    constexpr std::array<float, 8> atanh_inputs{{ -0.95, -0.5, -0.125, -0.0, 0.0, 0.125, 0.5, 0.95 }};
    for (float input : atanh_inputs)
    {
        check_unary_op("atanh", input, exact_tol(),
            [](float x) { return bl::atanh(x); },
            [](float x) { return std::atanh(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random hyperbolic inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float moderate = random_signed_interval(rng, 8.0);
        const float acosh_input = random_acosh_argument(rng);
        const float atanh_input = random_atanh_argument(rng);

        check_unary_op("sinh", moderate, exact_tol(),
            [](float x) { return bl::sinh(x); },
            [](float x) { return std::sinh(x); });

        check_unary_op("cosh", moderate, exact_tol(),
            [](float x) { return bl::cosh(x); },
            [](float x) { return std::cosh(x); });

        check_unary_op("tanh", moderate, exact_tol(),
            [](float x) { return bl::tanh(x); },
            [](float x) { return std::tanh(x); });

        check_unary_op("asinh", moderate, exact_tol(),
            [](float x) { return bl::asinh(x); },
            [](float x) { return std::asinh(x); });

        check_unary_op("acosh", acosh_input, exact_tol(),
            [](float x) { return bl::acosh(x); },
            [](float x) { return std::acosh(x); });

        check_unary_op("atanh", atanh_input, exact_tol(),
            [](float x) { return bl::atanh(x); },
            [](float x) { return std::atanh(x); });
    }
}

TEST_CASE("f32 special functions match std", "[fltx][f32][precision][transcendental][special]")
{
    accuracy_report_scope report("f32 special functions match std");

    constexpr std::array<float, 9> erf_inputs{{
        -4.0,
        -2.0,
        -1.0,
        -0.125,
        0.0,
        0.125,
        1.0,
        2.0,
        4.0
    }};

    for (float input : erf_inputs)
    {
        check_unary_op("erf", input, exact_tol(),
            [](float x) { return bl::erf(x); },
            [](float x) { return std::erf(x); });

        check_unary_op("erfc", input, exact_tol(),
            [](float x) { return bl::erfc(x); },
            [](float x) { return std::erfc(x); });
    }

    constexpr std::array<float, 7> gamma_inputs{{ 0.125, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0 }};
    for (float input : gamma_inputs)
    {
        check_unary_op("lgamma", input, exact_tol(),
            [](float x) { return bl::lgamma(x); },
            [](float x) { return std::lgamma(x); });

        check_unary_op("tgamma", input, exact_tol(),
            [](float x) { return bl::tgamma(x); },
            [](float x) { return std::tgamma(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random special-function inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float erf_input = random_signed_interval(rng, 4.0);
        const float gamma_input = random_gamma_positive(rng);

        check_unary_op("erf", erf_input, exact_tol(),
            [](float x) { return bl::erf(x); },
            [](float x) { return std::erf(x); });

        check_unary_op("erfc", erf_input, exact_tol(),
            [](float x) { return bl::erfc(x); },
            [](float x) { return std::erfc(x); });

        check_unary_op("lgamma", gamma_input, exact_tol(),
            [](float x) { return bl::lgamma(x); },
            [](float x) { return std::lgamma(x); });

        check_unary_op("tgamma", gamma_input, exact_tol(),
            [](float x) { return bl::tgamma(x); },
            [](float x) { return std::tgamma(x); });
    }
}

TEST_CASE("f32 decomposition and stepping functions match std", "[fltx][f32][precision][math][decomposition]")
{
    accuracy_report_scope report("f32 decomposition and stepping functions match std");

    constexpr std::array<float, 10> inputs{{
        -1e30f,
        -3.5,
        -0.5,
        -0.0,
        0.0,
        0.5,
        3.5,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        1e30f
    }};

    for (float input : inputs)
    {
        check_unary_op("fabs", input, exact_tol(),
            [](float x) { return bl::fabs(x); },
            [](float x) { return std::fabs(x); });

        check_unary_op("abs", input, exact_tol(),
            [](float x) { return bl::abs(x); },
            [](float x) { return std::fabs(x); });

        check_unary_op("logb", input, exact_tol(),
            [](float x) { return bl::logb(x); },
            [](float x) { return std::logb(x); });

        check_frexp_result(input);
        check_modf_result(input);

        check_exact_integer_result("ilogb", bl::ilogb(input), std::ilogb(input), input, 0.0);
        REQUIRE(bl::signbit(input) == std::signbit(input));
        REQUIRE(bl::isnan(input) == std::isnan(input));
        REQUIRE(bl::isinf(input) == std::isinf(input));
        REQUIRE(bl::isfinite(input) == std::isfinite(input));
        REQUIRE(bl::fpclassify(input) == std::fpclassify(input));
        REQUIRE(bl::isnormal(input) == std::isnormal(input));
    }

    constexpr std::array<std::pair<float, float>, 8> pairs{{
        { -0.0f, 0.0f },
        { 0.0f, -0.0f },
        { 1.0f, 2.0f },
        { 2.0f, 1.0f },
        { -1.0f, 1.0f },
        { std::numeric_limits<float>::denorm_min(), 0.0f },
        { 0.0f, std::numeric_limits<float>::denorm_min() },
        { (f32)std::numbers::pi, (f32)-std::numbers::e }
    }};

    for (const auto& [lhs, rhs] : pairs)
    {
        check_unary_op("nextafter", lhs, exact_tol(),
            [rhs](float x) { return bl::nextafter(x, rhs); },
            [rhs](float x) { return std::nextafter(x, rhs); });

        check_unary_op("nexttoward", lhs, exact_tol(),
            [rhs](float x) { return bl::nexttoward(x, static_cast<long double>(rhs)); },
            [rhs](float x) { return std::nexttoward(x, static_cast<long double>(rhs)); });

        check_exact_bool_result("isunordered", bl::isunordered(lhs, rhs), std::isunordered(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isgreater", bl::isgreater(lhs, rhs), std::isgreater(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isgreaterequal", bl::isgreaterequal(lhs, rhs), std::isgreaterequal(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isless", bl::isless(lhs, rhs), std::isless(lhs, rhs), lhs, rhs);
        check_exact_bool_result("islessequal", bl::islessequal(lhs, rhs), std::islessequal(lhs, rhs), lhs, rhs);
        check_exact_bool_result("islessgreater", bl::islessgreater(lhs, rhs), std::islessgreater(lhs, rhs), lhs, rhs);
    }

    constexpr std::array<std::pair<float, int>, 8> ldexp_cases{{
        { 0.0f, 0 },
        { -0.0f, 3 },
        { 0.5f, 1 },
        { -0.5f, 1 },
        { 1.0f, -10 },
        { (f32)std::numbers::pi, 7 },
        { 1e-200f, 64 },
        { 1e30f, -64 }
    }};

    for (const auto& [value, exponent] : ldexp_cases)
    {
        check_unary_op("ldexp", value, exact_tol(),
            [exponent](float x) { return bl::ldexp(x, exponent); },
            [exponent](float x) { return std::ldexp(x, exponent); });

        check_unary_op("scalbn", value, exact_tol(),
            [exponent](float x) { return bl::scalbn(x, exponent); },
            [exponent](float x) { return std::scalbn(x, exponent); });

        check_unary_op("scalbln", value, exact_tol(),
            [exponent](float x) { return bl::scalbln(x, exponent); },
            [exponent](float x) { return std::scalbln(x, exponent); });
    }
}

TEST_CASE("f32 utility helpers match std", "[fltx][f32][precision][math][utility]")
{
    accuracy_report_scope report("f32 utility helpers match std");

    constexpr std::array<std::pair<float, float>, 10> pairs{{
        { -0.0f, 0.0f },
        { 0.0f, -0.0f },
        { 1.0f, 2.0f },
        { 2.0f, 1.0f },
        { -2.0f, 3.0f },
        { 3.0f, -2.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::pi, (f32)std::numbers::e },
        { 1e30f, 1e-200f },
        { 1e-200f, 1e30f }
    }};

    for (const auto& [lhs, rhs] : pairs)
    {
        check_binary_op("fmin", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmin(a, b); },
            [](float a, float b) { return std::fmin(a, b); });

        check_binary_op("fmax", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmax(a, b); },
            [](float a, float b) { return std::fmax(a, b); });

        check_binary_op("fdim", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fdim(a, b); },
            [](float a, float b) { return std::fdim(a, b); });

        check_binary_op("copysign", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::copysign(a, b); },
            [](float a, float b) { return std::copysign(a, b); });
    }

    constexpr std::array<std::tuple<float, float, float>, 6> fma_cases{{
        { 2.0f, 3.0f, 4.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e, (f32)-std::numbers::sqrt2 },
        { -3.0f, 4.0f, 5.0f },
        { 0.5f, -0.25f, 0.125f },
        { 6.0f, -7.0f, 8.0f }
    }};

    for (const auto& [x, y, z] : fma_cases)
    {
        check_binary_op("fma.partial", x, y, close_tol(8),
            [z](float a, float b) { return bl::fma(a, b, z); },
            [z](float a, float b) { return std::fma(a, b, z); });
    }
}
