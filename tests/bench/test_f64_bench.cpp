#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <f64_math.h>
#include "benchmark_chart_writer.h"

namespace
{
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 500;
    constexpr std::size_t value_count = 64;

    bl::bench::benchmark_chart_writer chart_writer{
        "f64",
        "std",
        "f64 vs std typical benchmark ratios",
        "benchmark_charts/f64_typical_ratios.csv",
        "benchmark_charts/f64_typical_ratios.svg"
    };

    struct bench_result
    {
        double total_ms = 0.0;
        double ns_per_iter = 0.0;
        std::int64_t iteration_count = 0;
    };

    struct comparison_result
    {
        bench_result bl{};
        bench_result std{};
    };

    struct binary_value
    {
        double lhs = 0.0;
        double rhs = 0.0;
    };

    struct ternary_value
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct int_exponent_value
    {
        double value = 0.0;
        int exponent = 0;
    };

    struct long_exponent_value
    {
        double value = 0.0;
        long exponent = 0;
    };

    struct nexttoward_value
    {
        double from = 0.0;
        long double to = 0.0L;
    };

    struct frexp_result
    {
        double fraction = 0.0;
        int exponent = 0;
    };

    struct modf_result
    {
        double fractional = 0.0;
        double integral = 0.0;
    };

    struct remquo_result
    {
        double remainder = 0.0;
        int quotient = 0;
    };

    volatile double benchmark_sink_double = 0.0;
    volatile std::int64_t benchmark_sink_int64 = 0;

    void consume_result(double value)
    {
        benchmark_sink_double += value;
    }

    void consume_result(std::int64_t value)
    {
        benchmark_sink_int64 += value;
    }

    void consume_result(const frexp_result& value)
    {
        benchmark_sink_double += value.fraction;
        benchmark_sink_int64 += static_cast<std::int64_t>(value.exponent);
    }

    void consume_result(const modf_result& value)
    {
        benchmark_sink_double += value.fractional + value.integral;
    }

    void consume_result(const remquo_result& value)
    {
        benchmark_sink_double += value.remainder;
        benchmark_sink_int64 += static_cast<std::int64_t>(value.quotient);
    }

    [[nodiscard]] double blend_result(double value, double acc)
    {
        return value + acc * 0.25;
    }

    [[nodiscard]] std::int64_t blend_result(int value, std::int64_t acc)
    {
        return static_cast<std::int64_t>(value) + (acc * 3) / 4;
    }

    [[nodiscard]] std::int64_t blend_result(long value, std::int64_t acc)
    {
        return static_cast<std::int64_t>(value) + (acc * 3) / 4;
    }

    [[nodiscard]] std::int64_t blend_result(long long value, std::int64_t acc)
    {
        return static_cast<std::int64_t>(value) + (acc * 3) / 4;
    }

    [[nodiscard]] frexp_result blend_result(const frexp_result& value, const frexp_result& acc)
    {
        return {
            value.fraction + acc.fraction * 0.25,
            value.exponent + acc.exponent / 4
        };
    }

    [[nodiscard]] modf_result blend_result(const modf_result& value, const modf_result& acc)
    {
        return {
            value.fractional + acc.fractional * 0.25,
            value.integral + acc.integral * 0.25
        };
    }

    [[nodiscard]] remquo_result blend_result(const remquo_result& value, const remquo_result& acc)
    {
        return {
            value.remainder + acc.remainder * 0.25,
            value.quotient + acc.quotient / 4
        };
    }

    template<typename Result, typename Work>
    [[nodiscard]] bench_result run_benchmark(std::int64_t iteration_count, Work&& work)
    {
        const auto start = clock_type::now();
        const Result final_value = work();
        const auto end = clock_type::now();

        consume_result(final_value);

        const std::chrono::duration<double, std::milli> elapsed = end - start;

        bench_result result{};
        result.total_ms = elapsed.count();
        result.ns_per_iter = (elapsed.count() * 1'000'000.0) / static_cast<double>(iteration_count);
        result.iteration_count = iteration_count;
        return result;
    }

    [[nodiscard]] const char* benchmark_group_for_label(std::string_view label)
    {
        if (label == "floor" || label == "ceil" || label == "trunc" || label == "round" ||
            label == "nearbyint" || label == "rint" || label == "lround" || label == "llround" ||
            label == "lrint" || label == "llrint")
            return "Rounding";

        if (label == "fmod" || label == "remainder" || label == "remquo")
            return "Remainders";

        if (label == "abs" || label == "fabs" ||
            label == "fma" || label == "fmin" || label == "fmax" || label == "fdim" ||
            label == "copysign" || label == "ldexp" || label == "scalbn" || label == "scalbln" ||
            label == "frexp" || label == "modf" || label == "ilogb" || label == "logb" ||
            label == "nextafter" || label.starts_with("nexttoward"))
            return "Floating-point utilities";

        if (label == "sqrt" || label == "cbrt" || label == "hypot" || label == "pow")
            return "Roots & powers";

        if (label == "exp" || label == "exp2" || label == "expm1")
            return "Exponentials";

        if (label == "log" || label == "log2" || label == "log10" || label == "log1p")
            return "Logarithms";

        if (label == "sin" || label == "cos" || label == "tan" || label == "atan" ||
            label == "atan2" || label == "asin" || label == "acos")
            return "Trigonometric";

        if (label == "sinh" || label == "cosh" || label == "tanh")
            return "Hyperbolic";

        if (label == "asinh" || label == "acosh" || label == "atanh")
            return "Inverse hyperbolic";

        if (label == "erf" || label == "erfc" || label == "lgamma" || label == "tgamma")
            return "Special functions";

        return "Other";
    }

    void print_result(const char* label, const comparison_result& result)
    {
        const double ratio = result.std.ns_per_iter / result.bl.ns_per_iter;
        if (std::string_view(label) != "fabs")
            chart_writer.record_result(benchmark_group_for_label(label), label, result.bl.ns_per_iter, result.std.ns_per_iter);

        std::cout
            << std::fixed << std::setprecision(2)
            << label
            << "\n  bl::f64 : " << result.bl.total_ms << " ms total, " << result.bl.ns_per_iter << " ns/iter"
            << "  (total_iterations: " << result.bl.iteration_count << ")"
            << "\n  std     : " << result.std.total_ms << " ms total, " << result.std.ns_per_iter << " ns/iter"
            << "  (total_iterations: " << result.std.iteration_count << ")"
            << "\n  std/bl ratio: " << ratio << "x"
            << "\n";
    }

    [[nodiscard]] double random_unit(std::mt19937_64& rng)
    {
        return std::generate_canonical<double, 53>(rng);
    }

    [[nodiscard]] double random_real(std::mt19937_64& rng, double lo, double hi)
    {
        return lo + (hi - lo) * random_unit(rng);
    }

    [[nodiscard]] int random_int(std::mt19937_64& rng, int lo, int hi)
    {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    }

    [[nodiscard]] long random_long(std::mt19937_64& rng, long lo, long hi)
    {
        std::uniform_int_distribution<long> dist(lo, hi);
        return dist(rng);
    }

    [[nodiscard]] bool random_bool(std::mt19937_64& rng)
    {
        return (rng() & 1ull) != 0ull;
    }

    [[nodiscard]] double random_sign(std::mt19937_64& rng)
    {
        return random_bool(rng) ? -1.0 : 1.0;
    }

    [[nodiscard]] double make_finite_value(std::mt19937_64& rng, int exponent_lo, int exponent_hi)
    {
        const double mantissa = random_real(rng, 0.5, 1.9999999999999998);
        return random_sign(rng) * std::ldexp(mantissa, random_int(rng, exponent_lo, exponent_hi));
    }

    [[nodiscard]] double make_positive_value(std::mt19937_64& rng, int exponent_lo, int exponent_hi)
    {
        const double mantissa = random_real(rng, 0.5, 1.9999999999999998);
        return std::ldexp(mantissa, random_int(rng, exponent_lo, exponent_hi));
    }

    [[nodiscard]] std::array<double, value_count> make_generic_unary_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641001ull);
        for (double& value : values)
            value = make_finite_value(rng, -40, 40);
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_rounding_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641002ull);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double base = std::ldexp(random_real(rng, 0.5, 1.9999999999999998), random_int(rng, -6, 20));
            const double fractional = (static_cast<int>(i % 8) - 3.5) * 0.125;
            values[i] = random_sign(rng) * (std::floor(base) + fractional);
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_positive_log_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641003ull);
        for (double& value : values)
            value = make_positive_value(rng, -60, 60);
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_log1p_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641004ull);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            switch (i % 4)
            {
            case 0:
                values[i] = random_sign(rng) * std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 12, 50));
                break;
            case 1:
                values[i] = -1.0 + std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 4, 40));
                break;
            case 2:
                values[i] = random_real(rng, 0.0, 8.0);
                break;
            default:
                values[i] = std::ldexp(random_real(rng, 0.5, 1.5), random_int(rng, 3, 16));
                break;
            }
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_sqrt_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641005ull);
        for (double& value : values)
            value = make_positive_value(rng, -60, 60);
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_trig_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641006ull);
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double scale = std::ldexp(random_real(rng, 0.5, 1.5), random_int(rng, 0, 20));
            const double offset = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 6, 40));
            values[i] = random_sign(rng) * (scale * pi + offset * static_cast<double>((i % 5) + 1));
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_unit_interval_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641007ull);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if ((i % 3) == 0)
            {
                const double margin = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 8, 50));
                values[i] = random_sign(rng) * (1.0 - margin);
            }
            else
            {
                values[i] = random_real(rng, -0.999999999999, 0.999999999999);
            }
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_atanh_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641008ull);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if ((i % 2) == 0)
            {
                const double margin = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 8, 50));
                values[i] = random_sign(rng) * (1.0 - margin);
            }
            else
            {
                values[i] = random_real(rng, -0.9, 0.9);
            }
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_acosh_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x641009ull);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if ((i % 2) == 0)
                values[i] = 1.0 + std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 6, 40));
            else
                values[i] = make_positive_value(rng, 0, 40) + 1.0;
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_exponent_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x64100Aull);
        constexpr double ln2 = 0.693147180559945309417232121458176568;
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if ((i % 3) == 0)
            {
                const double k = static_cast<double>(random_int(rng, -64, 64));
                const double eps = random_sign(rng) * std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 10, 40));
                values[i] = k * ln2 + eps;
            }
            else
            {
                values[i] = random_real(rng, -40.0, 40.0);
            }
        }
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_hyperbolic_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x64100Bull);
        for (double& value : values)
            value = random_real(rng, -10.0, 10.0);
        return values;
    }

    [[nodiscard]] std::array<double, value_count> make_gamma_values()
    {
        std::array<double, value_count> values{};
        std::mt19937_64 rng(0x64100Cull);
        for (double& value : values)
        {
            double x = random_real(rng, -20.0, 20.0);
            const double nearest = std::nearbyint(x);
            if (std::fabs(x - nearest) < 0.15)
                x = nearest + (random_bool(rng) ? 0.25 : -0.25);
            if (std::fabs(x) < 0.15)
                x += random_bool(rng) ? 0.5 : -0.5;
            value = x;
        }
        return values;
    }

    [[nodiscard]] std::array<binary_value, value_count> make_generic_binary_values()
    {
        std::array<binary_value, value_count> values{};
        std::mt19937_64 rng(0x64100Dull);
        for (binary_value& value : values)
        {
            value.lhs = make_finite_value(rng, -40, 40);
            value.rhs = make_finite_value(rng, -40, 40);
        }
        return values;
    }

    [[nodiscard]] std::array<binary_value, value_count> make_remainder_values()
    {
        std::array<binary_value, value_count> values{};
        std::mt19937_64 rng(0x64100Eull);
        for (binary_value& value : values)
        {
            value.lhs = make_finite_value(rng, -60, 60);
            value.rhs = make_finite_value(rng, -20, 20);
            if (value.rhs == 0.0)
                value.rhs = 1.0;
        }
        return values;
    }

    [[nodiscard]] std::array<binary_value, value_count> make_hypot_values()
    {
        std::array<binary_value, value_count> values{};
        std::mt19937_64 rng(0x64100Full);
        for (binary_value& value : values)
        {
            value.lhs = make_finite_value(rng, -100, 100);
            value.rhs = make_finite_value(rng, -100, 100);
        }
        return values;
    }

    [[nodiscard]] std::array<binary_value, value_count> make_pow_values()
    {
        std::array<binary_value, value_count> values{};
        std::mt19937_64 rng(0x641010ull);
        for (binary_value& value : values)
        {
            if (random_bool(rng))
                value.lhs = 1.0 + random_sign(rng) * std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 8, 40));
            else
                value.lhs = make_positive_value(rng, -20, 20);
            value.rhs = random_real(rng, -8.0, 8.0);
        }
        return values;
    }

    [[nodiscard]] std::array<binary_value, value_count> make_nextafter_values()
    {
        std::array<binary_value, value_count> values{};
        std::mt19937_64 rng(0x641011ull);
        for (binary_value& value : values)
        {
            value.lhs = make_finite_value(rng, -100, 100);
            value.rhs = make_finite_value(rng, -100, 100);
            if (value.lhs == value.rhs)
                value.rhs = std::nextafter(value.rhs, std::numeric_limits<double>::infinity());
        }
        return values;
    }

    [[nodiscard]] std::array<ternary_value, value_count> make_fma_values()
    {
        std::array<ternary_value, value_count> values{};
        std::mt19937_64 rng(0x641012ull);
        for (ternary_value& value : values)
        {
            value.x = make_finite_value(rng, -30, 30);
            value.y = make_finite_value(rng, -30, 30);
            value.z = make_finite_value(rng, -30, 30);
        }
        return values;
    }

    [[nodiscard]] std::array<int_exponent_value, value_count> make_int_exponent_values()
    {
        std::array<int_exponent_value, value_count> values{};
        std::mt19937_64 rng(0x641013ull);
        for (int_exponent_value& value : values)
        {
            value.value = make_finite_value(rng, -20, 20);
            value.exponent = random_int(rng, -60, 60);
        }
        return values;
    }

    [[nodiscard]] std::array<long_exponent_value, value_count> make_long_exponent_values()
    {
        std::array<long_exponent_value, value_count> values{};
        std::mt19937_64 rng(0x641014ull);
        for (long_exponent_value& value : values)
        {
            value.value = make_finite_value(rng, -20, 20);
            value.exponent = random_long(rng, -60, 60);
        }
        return values;
    }

    [[nodiscard]] std::array<nexttoward_value, value_count> make_nexttoward_values()
    {
        std::array<nexttoward_value, value_count> values{};
        std::mt19937_64 rng(0x641015ull);
        for (nexttoward_value& value : values)
        {
            value.from = make_finite_value(rng, -100, 100);
            const long double delta = static_cast<long double>(random_sign(rng)) * std::ldexp(static_cast<long double>(random_real(rng, 0.25, 0.95)), -random_int(rng, 8, 48));
            value.to = static_cast<long double>(value.from) + delta;
            if (value.to == static_cast<long double>(value.from))
                value.to = static_cast<long double>(std::nextafter(value.from, std::numeric_limits<double>::infinity()));
        }
        return values;
    }

    [[nodiscard]] const auto& generic_unary_values()
    {
        static const auto values = make_generic_unary_values();
        return values;
    }

    [[nodiscard]] const auto& rounding_values()
    {
        static const auto values = make_rounding_values();
        return values;
    }

    [[nodiscard]] const auto& positive_log_values()
    {
        static const auto values = make_positive_log_values();
        return values;
    }

    [[nodiscard]] const auto& log1p_values()
    {
        static const auto values = make_log1p_values();
        return values;
    }

    [[nodiscard]] const auto& sqrt_values()
    {
        static const auto values = make_sqrt_values();
        return values;
    }

    [[nodiscard]] const auto& trig_values()
    {
        static const auto values = make_trig_values();
        return values;
    }

    [[nodiscard]] const auto& unit_interval_values()
    {
        static const auto values = make_unit_interval_values();
        return values;
    }

    [[nodiscard]] const auto& atanh_values()
    {
        static const auto values = make_atanh_values();
        return values;
    }

    [[nodiscard]] const auto& acosh_values()
    {
        static const auto values = make_acosh_values();
        return values;
    }

    [[nodiscard]] const auto& exponent_values()
    {
        static const auto values = make_exponent_values();
        return values;
    }

    [[nodiscard]] const auto& hyperbolic_values()
    {
        static const auto values = make_hyperbolic_values();
        return values;
    }

    [[nodiscard]] const auto& gamma_values()
    {
        static const auto values = make_gamma_values();
        return values;
    }

    [[nodiscard]] const auto& generic_binary_values()
    {
        static const auto values = make_generic_binary_values();
        return values;
    }

    [[nodiscard]] const auto& remainder_values()
    {
        static const auto values = make_remainder_values();
        return values;
    }

    [[nodiscard]] const auto& hypot_values()
    {
        static const auto values = make_hypot_values();
        return values;
    }

    [[nodiscard]] const auto& pow_values()
    {
        static const auto values = make_pow_values();
        return values;
    }

    [[nodiscard]] const auto& nextafter_values()
    {
        static const auto values = make_nextafter_values();
        return values;
    }

    [[nodiscard]] const auto& fma_values()
    {
        static const auto values = make_fma_values();
        return values;
    }

    [[nodiscard]] const auto& int_exponent_values()
    {
        static const auto values = make_int_exponent_values();
        return values;
    }

    [[nodiscard]] const auto& long_exponent_values()
    {
        static const auto values = make_long_exponent_values();
        return values;
    }

    [[nodiscard]] const auto& nexttoward_values()
    {
        static const auto values = make_nexttoward_values();
        return values;
    }

    template<typename Accumulator, typename Op>
    [[nodiscard]] bench_result benchmark_indexed(std::size_t count, std::int64_t total_iterations, Op&& op)
    {
        const std::int64_t iteration_count = std::max<std::int64_t>(static_cast<std::int64_t>(count), total_iterations);
        return run_benchmark<Accumulator>(iteration_count, [&]()
        {
            Accumulator acc{};
            for (std::int64_t i = 0; i < iteration_count; ++i)
                acc = blend_result(op(static_cast<std::size_t>(i % static_cast<std::int64_t>(count))), acc);
            return acc;
        });
    }

    template<typename Accumulator, typename BlWork, typename StdWork>
    [[nodiscard]] comparison_result benchmark_comparison(BlWork&& bl_work, StdWork&& std_work)
    {
        comparison_result result{};
        result.bl = std::forward<BlWork>(bl_work)();
        result.std = std::forward<StdWork>(std_work)();
        return result;
    }

    constexpr std::int64_t basic_iterations = 100000ll * benchmark_scale;
    constexpr std::int64_t core_iterations = 40000ll * benchmark_scale;
    constexpr std::int64_t transcendental_iterations = 20000ll * benchmark_scale;
    constexpr std::int64_t special_iterations = 4000ll * benchmark_scale;
}

#define BL_F64_UNARY_DOUBLE_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index]; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index]; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_UNARY_INT_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<std::int64_t>( \
        [&]() { return benchmark_indexed<std::int64_t>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index]; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<std::int64_t>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index]; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_BINARY_DOUBLE_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].lhs; const double y = values[index].rhs; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].lhs; const double y = values[index].rhs; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_TERNARY_DOUBLE_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].x; const double y = values[index].y; const double z = values[index].z; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].x; const double y = values[index].y; const double z = values[index].z; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_INT_EXP_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].value; const int e = values[index].exponent; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].value; const int e = values[index].exponent; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_LONG_EXP_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].value; const long e = values[index].exponent; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double x = values[index].value; const long e = values[index].exponent; return (std_expr); }); }); \
    print_result(label_text, results); \
}

#define BL_F64_FREXP_TEST(label_text, tags_text, values_name, iterations_value) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<frexp_result>( \
        [&]() { return benchmark_indexed<frexp_result>(values.size(), iterations_value, [&](std::size_t index) { int exponent = 0; const double fraction = bl::frexp(values[index], &exponent); return frexp_result{ fraction, exponent }; }); }, \
        [&]() { return benchmark_indexed<frexp_result>(values.size(), iterations_value, [&](std::size_t index) { int exponent = 0; const double fraction = std::frexp(values[index], &exponent); return frexp_result{ fraction, exponent }; }); }); \
    print_result(label_text, results); \
}

#define BL_F64_MODF_TEST(label_text, tags_text, values_name, iterations_value) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<modf_result>( \
        [&]() { return benchmark_indexed<modf_result>(values.size(), iterations_value, [&](std::size_t index) { double integral = 0.0; const double fractional = bl::modf(values[index], &integral); return modf_result{ fractional, integral }; }); }, \
        [&]() { return benchmark_indexed<modf_result>(values.size(), iterations_value, [&](std::size_t index) { double integral = 0.0; const double fractional = std::modf(values[index], &integral); return modf_result{ fractional, integral }; }); }); \
    print_result(label_text, results); \
}

#define BL_F64_REMQUO_TEST(label_text, tags_text, values_name, iterations_value) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<remquo_result>( \
        [&]() { return benchmark_indexed<remquo_result>(values.size(), iterations_value, [&](std::size_t index) { int quotient = 0; const double remainder = bl::remquo(values[index].lhs, values[index].rhs, &quotient); return remquo_result{ remainder, quotient }; }); }, \
        [&]() { return benchmark_indexed<remquo_result>(values.size(), iterations_value, [&](std::size_t index) { int quotient = 0; const double remainder = std::remquo(values[index].lhs, values[index].rhs, &quotient); return remquo_result{ remainder, quotient }; }); }); \
    print_result(label_text, results); \
}

#define BL_F64_NEXTTOWARD_TEST(label_text, tags_text, values_name, iterations_value, bl_expr, std_expr) \
TEST_CASE("f64 vs std " label_text " performance", tags_text) \
{ \
    const auto& values = values_name(); \
    const auto results = benchmark_comparison<double>( \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double from = values[index].from; const long double to = values[index].to; return (bl_expr); }); }, \
        [&]() { return benchmark_indexed<double>(values.size(), iterations_value, [&](std::size_t index) { const double from = values[index].from; const long double to = values[index].to; return (std_expr); }); }); \
    print_result(label_text, results); \
}

BL_F64_UNARY_DOUBLE_TEST("abs", "[bench][fltx][f64][abs]", generic_unary_values, basic_iterations, bl::abs(x), std::abs(x))
BL_F64_UNARY_DOUBLE_TEST("fabs", "[bench][fltx][f64][fabs]", generic_unary_values, basic_iterations, bl::fabs(x), std::fabs(x))
BL_F64_UNARY_DOUBLE_TEST("floor", "[bench][fltx][f64][rounding][floor]", rounding_values, basic_iterations, bl::floor(x), std::floor(x))
BL_F64_UNARY_DOUBLE_TEST("ceil", "[bench][fltx][f64][rounding][ceil]", rounding_values, basic_iterations, bl::ceil(x), std::ceil(x))
BL_F64_UNARY_DOUBLE_TEST("trunc", "[bench][fltx][f64][rounding][trunc]", rounding_values, basic_iterations, bl::trunc(x), std::trunc(x))
BL_F64_UNARY_DOUBLE_TEST("round", "[bench][fltx][f64][rounding][round]", rounding_values, basic_iterations, bl::round(x), std::round(x))
BL_F64_UNARY_DOUBLE_TEST("nearbyint", "[bench][fltx][f64][rounding][nearbyint]", rounding_values, basic_iterations, bl::nearbyint(x), std::nearbyint(x))
BL_F64_UNARY_DOUBLE_TEST("rint", "[bench][fltx][f64][rounding][rint]", rounding_values, basic_iterations, bl::rint(x), std::rint(x))
BL_F64_UNARY_INT_TEST("lround", "[bench][fltx][f64][rounding][lround]", rounding_values, basic_iterations, bl::lround(x), std::lround(x))
BL_F64_UNARY_INT_TEST("llround", "[bench][fltx][f64][rounding][llround]", rounding_values, basic_iterations, bl::llround(x), std::llround(x))
BL_F64_UNARY_INT_TEST("lrint", "[bench][fltx][f64][rounding][lrint]", rounding_values, basic_iterations, bl::lrint(x), std::lrint(x))
BL_F64_UNARY_INT_TEST("llrint", "[bench][fltx][f64][rounding][llrint]", rounding_values, basic_iterations, bl::llrint(x), std::llrint(x))

BL_F64_BINARY_DOUBLE_TEST("fmod", "[bench][fltx][f64][fmod]", remainder_values, core_iterations, bl::fmod(x, y), std::fmod(x, y))
BL_F64_BINARY_DOUBLE_TEST("remainder", "[bench][fltx][f64][remainder]", remainder_values, core_iterations, bl::remainder(x, y), std::remainder(x, y))
BL_F64_REMQUO_TEST("remquo", "[bench][fltx][f64][remquo]", remainder_values, core_iterations)
BL_F64_TERNARY_DOUBLE_TEST("fma", "[bench][fltx][f64][fma]", fma_values, core_iterations, bl::fma(x, y, z), std::fma(x, y, z))
BL_F64_BINARY_DOUBLE_TEST("fmin", "[bench][fltx][f64][fmin]", generic_binary_values, basic_iterations, bl::fmin(x, y), std::fmin(x, y))
BL_F64_BINARY_DOUBLE_TEST("fmax", "[bench][fltx][f64][fmax]", generic_binary_values, basic_iterations, bl::fmax(x, y), std::fmax(x, y))
BL_F64_BINARY_DOUBLE_TEST("fdim", "[bench][fltx][f64][fdim]", generic_binary_values, basic_iterations, bl::fdim(x, y), std::fdim(x, y))
BL_F64_BINARY_DOUBLE_TEST("copysign", "[bench][fltx][f64][copysign]", generic_binary_values, basic_iterations, bl::copysign(x, y), std::copysign(x, y))
BL_F64_INT_EXP_TEST("ldexp", "[bench][fltx][f64][ldexp]", int_exponent_values, core_iterations, bl::ldexp(x, e), std::ldexp(x, e))
BL_F64_INT_EXP_TEST("scalbn", "[bench][fltx][f64][scalbn]", int_exponent_values, core_iterations, bl::scalbn(x, e), std::scalbn(x, e))
BL_F64_LONG_EXP_TEST("scalbln", "[bench][fltx][f64][scalbln]", long_exponent_values, core_iterations, bl::scalbln(x, e), std::scalbln(x, e))
BL_F64_FREXP_TEST("frexp", "[bench][fltx][f64][frexp]", generic_unary_values, core_iterations)
BL_F64_MODF_TEST("modf", "[bench][fltx][f64][modf]", generic_unary_values, core_iterations)
BL_F64_UNARY_INT_TEST("ilogb", "[bench][fltx][f64][ilogb]", positive_log_values, basic_iterations, bl::ilogb(x), std::ilogb(x))
BL_F64_UNARY_DOUBLE_TEST("logb", "[bench][fltx][f64][logb]", positive_log_values, basic_iterations, bl::logb(x), std::logb(x))
BL_F64_BINARY_DOUBLE_TEST("nextafter", "[bench][fltx][f64][nextafter]", nextafter_values, basic_iterations, bl::nextafter(x, y), std::nextafter(x, y))
BL_F64_NEXTTOWARD_TEST("nexttoward(long double)", "[bench][fltx][f64][nexttoward]", nexttoward_values, basic_iterations, bl::nexttoward(from, to), std::nexttoward(from, to))
BL_F64_NEXTTOWARD_TEST("nexttoward(double)", "[bench][fltx][f64][nexttoward]", nexttoward_values, basic_iterations, bl::nexttoward(from, static_cast<double>(to)), std::nexttoward(from, static_cast<long double>(static_cast<double>(to))))

BL_F64_UNARY_DOUBLE_TEST("exp", "[bench][fltx][f64][exp]", exponent_values, transcendental_iterations, bl::exp(x), std::exp(x))
BL_F64_UNARY_DOUBLE_TEST("exp2", "[bench][fltx][f64][exp2]", exponent_values, transcendental_iterations, bl::exp2(x), std::exp2(x))
BL_F64_UNARY_DOUBLE_TEST("expm1", "[bench][fltx][f64][expm1]", exponent_values, transcendental_iterations, bl::expm1(x), std::expm1(x))
BL_F64_UNARY_DOUBLE_TEST("log", "[bench][fltx][f64][log]", positive_log_values, transcendental_iterations, bl::log(x), std::log(x))
BL_F64_UNARY_DOUBLE_TEST("log2", "[bench][fltx][f64][log2]", positive_log_values, transcendental_iterations, bl::log2(x), std::log2(x))
BL_F64_UNARY_DOUBLE_TEST("log10", "[bench][fltx][f64][log10]", positive_log_values, transcendental_iterations, bl::log10(x), std::log10(x))
BL_F64_UNARY_DOUBLE_TEST("log1p", "[bench][fltx][f64][log1p]", log1p_values, transcendental_iterations, bl::log1p(x), std::log1p(x))
BL_F64_UNARY_DOUBLE_TEST("sqrt", "[bench][fltx][f64][sqrt]", sqrt_values, transcendental_iterations, bl::sqrt(x), std::sqrt(x))
BL_F64_UNARY_DOUBLE_TEST("cbrt", "[bench][fltx][f64][cbrt]", generic_unary_values, transcendental_iterations, bl::cbrt(x), std::cbrt(x))
BL_F64_BINARY_DOUBLE_TEST("hypot", "[bench][fltx][f64][hypot]", hypot_values, transcendental_iterations, bl::hypot(x, y), std::hypot(x, y))

BL_F64_UNARY_DOUBLE_TEST("sin", "[bench][fltx][f64][trig][sin]", trig_values, transcendental_iterations, bl::sin(x), std::sin(x))
BL_F64_UNARY_DOUBLE_TEST("cos", "[bench][fltx][f64][trig][cos]", trig_values, transcendental_iterations, bl::cos(x), std::cos(x))
BL_F64_UNARY_DOUBLE_TEST("tan", "[bench][fltx][f64][trig][tan]", trig_values, transcendental_iterations, bl::tan(x), std::tan(x))
BL_F64_UNARY_DOUBLE_TEST("atan", "[bench][fltx][f64][trig][atan]", generic_unary_values, transcendental_iterations, bl::atan(x), std::atan(x))
BL_F64_BINARY_DOUBLE_TEST("atan2", "[bench][fltx][f64][trig][atan2]", generic_binary_values, transcendental_iterations, bl::atan2(x, y), std::atan2(x, y))
BL_F64_UNARY_DOUBLE_TEST("asin", "[bench][fltx][f64][trig][asin]", unit_interval_values, transcendental_iterations, bl::asin(x), std::asin(x))
BL_F64_UNARY_DOUBLE_TEST("acos", "[bench][fltx][f64][trig][acos]", unit_interval_values, transcendental_iterations, bl::acos(x), std::acos(x))

BL_F64_UNARY_DOUBLE_TEST("sinh", "[bench][fltx][f64][hyperbolic][sinh]", hyperbolic_values, transcendental_iterations, bl::sinh(x), std::sinh(x))
BL_F64_UNARY_DOUBLE_TEST("cosh", "[bench][fltx][f64][hyperbolic][cosh]", hyperbolic_values, transcendental_iterations, bl::cosh(x), std::cosh(x))
BL_F64_UNARY_DOUBLE_TEST("tanh", "[bench][fltx][f64][hyperbolic][tanh]", hyperbolic_values, transcendental_iterations, bl::tanh(x), std::tanh(x))
BL_F64_UNARY_DOUBLE_TEST("asinh", "[bench][fltx][f64][hyperbolic][asinh]", generic_unary_values, transcendental_iterations, bl::asinh(x), std::asinh(x))
BL_F64_UNARY_DOUBLE_TEST("acosh", "[bench][fltx][f64][hyperbolic][acosh]", acosh_values, transcendental_iterations, bl::acosh(x), std::acosh(x))
BL_F64_UNARY_DOUBLE_TEST("atanh", "[bench][fltx][f64][hyperbolic][atanh]", atanh_values, transcendental_iterations, bl::atanh(x), std::atanh(x))

BL_F64_BINARY_DOUBLE_TEST("pow", "[bench][fltx][f64][pow]", pow_values, special_iterations, bl::pow(x, y), std::pow(x, y))
BL_F64_UNARY_DOUBLE_TEST("erf", "[bench][fltx][f64][special][erf]", generic_unary_values, special_iterations, bl::erf(x), std::erf(x))
BL_F64_UNARY_DOUBLE_TEST("erfc", "[bench][fltx][f64][special][erfc]", generic_unary_values, special_iterations, bl::erfc(x), std::erfc(x))
BL_F64_UNARY_DOUBLE_TEST("lgamma", "[bench][fltx][f64][special][gamma][lgamma]", gamma_values, special_iterations, bl::lgamma(x), std::lgamma(x))
BL_F64_UNARY_DOUBLE_TEST("tgamma", "[bench][fltx][f64][special][gamma][tgamma]", gamma_values, special_iterations, bl::tgamma(x), std::tgamma(x))

#undef BL_F64_UNARY_DOUBLE_TEST
#undef BL_F64_UNARY_INT_TEST
#undef BL_F64_BINARY_DOUBLE_TEST
#undef BL_F64_TERNARY_DOUBLE_TEST
#undef BL_F64_INT_EXP_TEST
#undef BL_F64_LONG_EXP_TEST
#undef BL_F64_FREXP_TEST
#undef BL_F64_MODF_TEST
#undef BL_F64_REMQUO_TEST
#undef BL_F64_NEXTTOWARD_TEST
