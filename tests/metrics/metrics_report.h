#ifndef FLTX_TESTS_METRICS_REPORT_INCLUDED
#define FLTX_TESTS_METRICS_REPORT_INCLUDED

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include "metrics_records.h"

namespace bl::test::metrics
{
    [[nodiscard]] inline const char* to_string(precision_type value) noexcept
    {
        switch (value)
        {
        case precision_type::f128: return "f128";
        case precision_type::f256: return "f256";
        }
        return "unknown";
    }

    [[nodiscard]] inline const char* metrics_fltx_backend_name(precision_type value) noexcept
    {
        switch (value)
        {
        case precision_type::f128: return "bl::f128";
        case precision_type::f256: return "bl::f256";
        }
        return "fltx";
    }

    [[nodiscard]] inline const char* to_string(domain_role value) noexcept
    {
        switch (value)
        {
        case domain_role::primary: return "primary";
        case domain_role::stress:  return "stress";
        }
        return "unknown";
    }

    [[nodiscard]] inline std::string_view metrics_compiler_name() noexcept
    {
        #if defined(__EMSCRIPTEN__) && defined(FLTX_BENCH_RUNTIME_CHROME)
        return "Chrome";
        #elif defined(__EMSCRIPTEN__) && defined(FLTX_BENCH_RUNTIME_BROWSER)
        return "Chrome";
        #elif defined(__EMSCRIPTEN__) && defined(FLTX_BENCH_RUNTIME_NODE)
        return "Nodejs";
        #elif defined(__EMSCRIPTEN__)
        return "Wasm32";
        #elif defined(__MINGW32__) || defined(__MINGW64__)
        return "MinGW";
        #elif defined(__clang__) && defined(__apple_build_version__)
        return "AppleClang";
        #elif defined(__clang__)
        return "Clang";
        #elif defined(_MSC_VER)
        return "MSVC";
        #elif defined(__GNUC__)
        return "GCC";
        #else
        return "unknown";
        #endif
    }

    [[nodiscard]] inline std::string_view metrics_os_name() noexcept
    {
        #if defined(__EMSCRIPTEN__)
        return "wasm32";
        #elif defined(_WIN32)
        return "windows";
        #elif defined(__APPLE__) && defined(__MACH__)
        return "macOS";
        #elif defined(__linux__)
        return "linux";
        #elif defined(__unix__)
        return "unix";
        #else
        return "unknown";
        #endif
    }

    [[nodiscard]] inline std::string metrics_output_path(
        std::string_view precision_name,
        std::string_view suffix,
        std::string_view extension)
    {
        #ifdef FLTX_METRICS_OUTPUT_ROOT
        std::filesystem::path output_path = std::filesystem::path(FLTX_METRICS_OUTPUT_ROOT);
        #else
        std::filesystem::path output_path = std::filesystem::path("res") / "metrics";
        #endif
        output_path /= std::string(metrics_os_name());

        std::string filename = std::string(metrics_compiler_name()) + "_" + std::string(precision_name);
        if (!suffix.empty())
        {
            filename += "_";
            filename += suffix;
        }
        filename += ".";
        filename += extension;

        output_path /= filename;
        return output_path.string();
    }

    [[nodiscard]] inline std::string metrics_native_arithmetic_group(precision_type precision)
    {
        const char* name = to_string(precision);
        return std::string(name) + " <-> " + name;
    }

    [[nodiscard]] inline bool metrics_operation_is(
        std::string_view operation,
        std::initializer_list<std::string_view> names) noexcept
    {
        for (std::string_view name : names)
        {
            if (operation == name)
                return true;
        }
        return false;
    }

    [[nodiscard]] inline std::string metrics_csv_group(const metrics_record& record)
    {
        const std::string_view operation = record.suite.operation.name;
        if (record.suite.domain.role == domain_role::stress)
            return "Mixed Workloads";
        if (metrics_operation_is(operation, { "add", "subtract", "multiply", "divide" }))
            return metrics_native_arithmetic_group(record.suite.precision);
        if (metrics_operation_is(operation, {
                "floor", "ceil", "trunc", "round", "nearbyint", "rint",
                "lround", "llround", "lrint", "llrint" }))
            return "Rounding";
        if (metrics_operation_is(operation, { "fmod", "remainder", "remquo" }))
            return "Remainders";
        if (metrics_operation_is(operation, {
                "fabs", "fma", "fmin", "fmax", "fdim", "copysign",
                "ldexp", "scalbn", "scalbln", "frexp", "modf", "ilogb",
                "logb", "nextafter", "nexttoward" }))
            return "Floating-point utilities";
        if (metrics_operation_is(operation, { "sqrt", "cbrt", "hypot", "pow" }))
            return "Roots & powers";
        if (metrics_operation_is(operation, { "exp", "exp2", "expm1" }))
            return "Exponentials";
        if (metrics_operation_is(operation, { "log", "log2", "log10", "log1p" }))
            return "Logarithms";
        if (metrics_operation_is(operation, { "sin", "cos", "tan", "asin", "acos", "atan", "atan2" }))
            return "Trigonometric";
        if (metrics_operation_is(operation, { "sinh", "cosh", "tanh" }))
            return "Hyperbolic";
        if (metrics_operation_is(operation, { "asinh", "acosh", "atanh" }))
            return "Inverse hyperbolic";
        if (metrics_operation_is(operation, { "erfc", "erf", "tgamma", "lgamma" }))
            return "Special functions";
        if (metrics_operation_is(operation, {
                "operator==", "operator!=", "operator<", "operator>",
                "operator<=", "operator>=" }))
            return "Comparisons";
        return "Primary";
    }

    [[nodiscard]] inline std::string metrics_csv_label(const metrics_record& record)
    {
        const std::string_view operation = record.suite.operation.name;
        if (operation == "fabs")
            return "abs";
        if (operation == "nexttoward")
            return "nexttoward(type)";
        if (record.suite.domain.role == domain_role::stress)
        {
            if (operation == "affine transform")
                return "affine trig transform";
            if (operation == "mandelbrot")
                return "mandelbrot kernel";
        }
        return std::string(operation);
    }

    [[nodiscard]] inline int metrics_csv_group_order(const metrics_record& record)
    {
        const std::string group = metrics_csv_group(record);
        if (group == metrics_native_arithmetic_group(record.suite.precision)) return 0;
        if (group == "Mixed Workloads") return 1;
        if (group == "Rounding") return 2;
        if (group == "Remainders") return 3;
        if (group == "Floating-point utilities") return 4;
        if (group == "Roots & powers") return 5;
        if (group == "Exponentials") return 6;
        if (group == "Logarithms") return 7;
        if (group == "Trigonometric") return 8;
        if (group == "Hyperbolic") return 9;
        if (group == "Inverse hyperbolic") return 10;
        if (group == "Special functions") return 11;
        if (group == "Comparisons") return 12;
        return 100;
    }

    [[nodiscard]] inline int metrics_csv_row_order(const metrics_record& record)
    {
        const std::string_view operation = record.suite.operation.name;
        if (operation == "add") return 0;
        if (operation == "subtract") return 1;
        if (operation == "multiply") return 2;
        if (operation == "divide") return 3;
        if (operation == "affine transform") return 0;
        if (operation == "mandelbrot") return 1;
        if (operation == "mixed arithmetic") return 2;
        if (operation == "floor") return 0;
        if (operation == "ceil") return 1;
        if (operation == "trunc") return 2;
        if (operation == "round") return 3;
        if (operation == "nearbyint") return 4;
        if (operation == "rint") return 5;
        if (operation == "lround") return 6;
        if (operation == "llround") return 7;
        if (operation == "lrint") return 8;
        if (operation == "llrint") return 9;
        if (operation == "fmod") return 0;
        if (operation == "remainder") return 1;
        if (operation == "remquo") return 2;
        if (operation == "fabs") return 0;
        if (operation == "fma") return 1;
        if (operation == "fmin") return 2;
        if (operation == "fmax") return 3;
        if (operation == "fdim") return 4;
        if (operation == "copysign") return 5;
        if (operation == "ldexp") return 6;
        if (operation == "scalbn") return 7;
        if (operation == "scalbln") return 8;
        if (operation == "frexp") return 9;
        if (operation == "modf") return 10;
        if (operation == "ilogb") return 11;
        if (operation == "logb") return 12;
        if (operation == "nextafter") return 13;
        if (operation == "nexttoward") return 14;
        if (operation == "sqrt") return 0;
        if (operation == "cbrt") return 1;
        if (operation == "hypot") return 2;
        if (operation == "pow") return 3;
        if (operation == "exp") return 0;
        if (operation == "exp2") return 1;
        if (operation == "expm1") return 2;
        if (operation == "log") return 0;
        if (operation == "log2") return 1;
        if (operation == "log10") return 2;
        if (operation == "log1p") return 3;
        if (operation == "sin") return 0;
        if (operation == "cos") return 1;
        if (operation == "tan") return 2;
        if (operation == "asin") return 3;
        if (operation == "acos") return 4;
        if (operation == "atan") return 5;
        if (operation == "atan2") return 6;
        if (operation == "sinh") return 0;
        if (operation == "cosh") return 1;
        if (operation == "tanh") return 2;
        if (operation == "asinh") return 0;
        if (operation == "acosh") return 1;
        if (operation == "atanh") return 2;
        if (operation == "erfc") return 0;
        if (operation == "erf") return 1;
        if (operation == "tgamma") return 2;
        if (operation == "lgamma") return 3;
        if (operation == "operator==") return 0;
        if (operation == "operator!=") return 1;
        if (operation == "operator<") return 2;
        if (operation == "operator>") return 3;
        if (operation == "operator<=") return 4;
        if (operation == "operator>=") return 5;
        return 1000;
    }

    [[nodiscard]] inline bool metrics_csv_record_less(
        const metrics_record& lhs,
        const metrics_record& rhs)
    {
        if (metrics_csv_group_order(lhs) != metrics_csv_group_order(rhs))
            return metrics_csv_group_order(lhs) < metrics_csv_group_order(rhs);
        if (metrics_csv_row_order(lhs) != metrics_csv_row_order(rhs))
            return metrics_csv_row_order(lhs) < metrics_csv_row_order(rhs);
        return metrics_csv_label(lhs) < metrics_csv_label(rhs);
    }

    inline void write_csv_text(std::ostream& out, std::string_view text)
    {
        const bool needs_quotes = text.find_first_of(",\"\r\n") != std::string_view::npos;
        if (!needs_quotes)
        {
            out << text;
            return;
        }

        out << '"';
        for (char ch : text)
        {
            if (ch == '"')
                out << "\"\"";
            else
                out << ch;
        }
        out << '"';
    }

    [[nodiscard]] inline double precision_gap_bits(const metrics_record& record) noexcept
    {
        const bool fltx_exact = std::isinf(record.fltx_accuracy.worst_bits);
        const bool competitor_exact = std::isinf(record.competitor_accuracy.worst_bits);
        if (fltx_exact && competitor_exact)
            return 0.0;
        if (fltx_exact)
            return std::numeric_limits<double>::infinity();
        if (competitor_exact)
            return -std::numeric_limits<double>::infinity();
        return record.fltx_accuracy.worst_bits - record.competitor_accuracy.worst_bits;
    }

    [[nodiscard]] inline double precision_gap_bits(
        const accuracy_result& fltx_accuracy,
        const accuracy_result& competitor_accuracy) noexcept
    {
        const bool fltx_exact = std::isinf(fltx_accuracy.worst_bits);
        const bool competitor_exact = std::isinf(competitor_accuracy.worst_bits);
        if (fltx_exact && competitor_exact)
            return 0.0;
        if (fltx_exact)
            return std::numeric_limits<double>::infinity();
        if (competitor_exact)
            return -std::numeric_limits<double>::infinity();
        return fltx_accuracy.worst_bits - competitor_accuracy.worst_bits;
    }

    [[nodiscard]] inline double precision_advantage_bits(
        double fltx_bits,
        double competitor_bits) noexcept
    {
        const bool fltx_exact = std::isinf(fltx_bits);
        const bool competitor_exact = std::isinf(competitor_bits);
        if (fltx_exact && competitor_exact)
            return 0.0;
        if (fltx_exact)
            return std::numeric_limits<double>::infinity();
        if (competitor_exact)
            return -std::numeric_limits<double>::infinity();
        return fltx_bits - competitor_bits;
    }

    [[nodiscard]] inline bool accuracy_is_exact(const accuracy_result& accuracy) noexcept
    {
        return std::isinf(accuracy.worst_bits) && accuracy.worst_bits > 0.0;
    }

    [[nodiscard]] inline double accuracy_mean_comparison_bits(const accuracy_result& accuracy) noexcept
    {
        return accuracy_is_exact(accuracy)
            ? std::numeric_limits<double>::infinity()
            : accuracy.mean_bits;
    }

    [[nodiscard]] inline double finite_display_bits(double bits, double exact_bits) noexcept
    {
        if (!std::isinf(bits) || !std::isfinite(exact_bits))
            return bits;
        return bits > 0.0 ? exact_bits : -exact_bits;
    }

    [[nodiscard]] inline double precision_worst_advantage_bits(
        const accuracy_result& fltx_accuracy,
        const accuracy_result& competitor_accuracy) noexcept
    {
        return precision_advantage_bits(
            finite_display_bits(fltx_accuracy.worst_bits, fltx_accuracy.mean_bits),
            finite_display_bits(competitor_accuracy.worst_bits, competitor_accuracy.mean_bits));
    }

    [[nodiscard]] inline double precision_mean_advantage_bits(
        const accuracy_result& fltx_accuracy,
        const accuracy_result& competitor_accuracy) noexcept
    {
        return precision_advantage_bits(
            accuracy_mean_comparison_bits(fltx_accuracy),
            accuracy_mean_comparison_bits(competitor_accuracy));
    }

    [[nodiscard]] inline double domain_gap(
        const accuracy_result& fltx_accuracy,
        const accuracy_result& competitor_accuracy) noexcept
    {
        return fltx_accuracy.domain_score - competitor_accuracy.domain_score;
    }

    [[nodiscard]] inline bool has_accuracy_data(const accuracy_result& accuracy) noexcept
    {
        return accuracy.sample_count > 0;
    }

    [[nodiscard]] inline bool has_benchmark_data(const benchmark_result& benchmark) noexcept
    {
        return benchmark.iteration_count > 0 && benchmark.ns_per_iter > 0.0;
    }

    [[nodiscard]] inline double speed_ratio(const metrics_record& record) noexcept
    {
        return record.competitor_supported &&
                has_benchmark_data(record.fltx_benchmark) &&
                has_benchmark_data(record.competitor_benchmark)
            ? record.competitor_benchmark.ns_per_iter / record.fltx_benchmark.ns_per_iter
            : 0.0;
    }

    [[nodiscard]] inline double speed_ratio(
        const benchmark_result& fltx_benchmark,
        const benchmark_result& competitor_benchmark) noexcept
    {
        return has_benchmark_data(fltx_benchmark) && has_benchmark_data(competitor_benchmark)
            ? competitor_benchmark.ns_per_iter / fltx_benchmark.ns_per_iter
            : 0.0;
    }

    [[nodiscard]] inline std::vector<std::string_view> collect_extra_competitor_names(
        const std::vector<metrics_record>& records)
    {
        std::vector<std::string_view> names;
        for (const metrics_record& record : records)
        {
            for (const competitor_result& competitor : record.extra_competitors)
            {
                if (std::find(names.begin(), names.end(), competitor.name) == names.end())
                    names.push_back(competitor.name);
            }
        }
        return names;
    }

    [[nodiscard]] inline const competitor_result* find_extra_competitor(
        const metrics_record& record,
        std::string_view name) noexcept
    {
        for (const competitor_result& competitor : record.extra_competitors)
        {
            if (competitor.name == name)
                return &competitor;
        }
        return nullptr;
    }

    struct preferred_reference_result
    {
        std::string_view name;
        const accuracy_result* accuracy = nullptr;
        const benchmark_result* benchmark = nullptr;
        bool supported = false;
    };

    struct preferred_reference_candidate
    {
        std::string_view name;
        const accuracy_result* accuracy = nullptr;
        const benchmark_result* benchmark = nullptr;
        special_correctness special_values = special_correctness::unavailable;
        bool supported = false;
    };

    [[nodiscard]] inline double precision_target_bits(precision_type precision) noexcept
    {
        switch (precision)
        {
        case precision_type::f128: return 106.0;
        case precision_type::f256: return 212.0;
        }
        return 0.0;
    }

    [[nodiscard]] inline int special_correctness_rank(special_correctness value) noexcept
    {
        switch (value)
        {
        case special_correctness::pass:        return 2;
        case special_correctness::unavailable: return 1;
        case special_correctness::fail:        return 0;
        }
        return 0;
    }

    [[nodiscard]] inline bool benchmark_is_faster(
        const benchmark_result& candidate,
        const benchmark_result& current) noexcept
    {
        if (candidate.ns_per_iter <= 0.0)
            return false;
        if (current.ns_per_iter <= 0.0)
            return true;
        return candidate.ns_per_iter < current.ns_per_iter;
    }

    [[nodiscard]] inline bool benchmark_is_materially_faster(
        const benchmark_result& candidate,
        const benchmark_result& current) noexcept
    {
        if (candidate.ns_per_iter <= 0.0)
            return false;
        if (current.ns_per_iter <= 0.0)
            return true;
        return candidate.ns_per_iter < current.ns_per_iter * 0.97;
    }

    [[nodiscard]] inline bool operation_uses_discrete_accuracy_gate(
        std::string_view operation) noexcept
    {
        return operation == "floor" ||
               operation == "ceil" ||
               operation == "trunc" ||
               operation == "round" ||
               operation == "nearbyint" ||
               operation == "rint" ||
               operation == "lround" ||
               operation == "llround" ||
               operation == "lrint" ||
               operation == "llrint";
    }

    [[nodiscard]] inline bool reference_has_stronger_result(
        const preferred_reference_candidate& candidate,
        const preferred_reference_candidate& current) noexcept
    {
        const double candidate_mean = accuracy_mean_comparison_bits(*candidate.accuracy);
        const double current_mean = accuracy_mean_comparison_bits(*current.accuracy);
        if (candidate_mean != current_mean)
            return candidate_mean > current_mean;
        if (candidate.accuracy->worst_bits != current.accuracy->worst_bits)
            return candidate.accuracy->worst_bits > current.accuracy->worst_bits;
        if (candidate.accuracy->domain_score != current.accuracy->domain_score)
            return candidate.accuracy->domain_score > current.accuracy->domain_score;

        const int candidate_special_rank = special_correctness_rank(candidate.special_values);
        const int current_special_rank = special_correctness_rank(current.special_values);
        if (candidate_special_rank != current_special_rank)
            return candidate_special_rank > current_special_rank;

        return benchmark_is_faster(*candidate.benchmark, *current.benchmark);
    }

    [[nodiscard]] inline bool reference_is_quality_comparable(
        std::string_view operation,
        precision_type precision,
        domain_role role,
        const preferred_reference_candidate& candidate,
        const preferred_reference_candidate& strongest) noexcept
    {
        const double target_bits = precision_target_bits(precision);

        if (operation_uses_discrete_accuracy_gate(operation))
            return candidate.accuracy->worst_bits >= target_bits - 4.0;

        if (role == domain_role::stress)
        {
            constexpr double domain_score_tolerance = 3.0;
            constexpr double worst_bits_tolerance = 4.0;
            return candidate.accuracy->domain_score + domain_score_tolerance >= strongest.accuracy->domain_score &&
                   candidate.accuracy->worst_bits + worst_bits_tolerance >= strongest.accuracy->worst_bits;
        }

        constexpr double mean_bits_tolerance      = 4.0;
        constexpr double primary_domain_floor     = 70.0;
        const double candidate_mean = accuracy_mean_comparison_bits(*candidate.accuracy);
        const double strongest_mean = accuracy_mean_comparison_bits(*strongest.accuracy);
        const bool close_to_strongest_mean =
            candidate_mean + mean_bits_tolerance >= strongest_mean;
        const bool precision_adequate_for_primary =
            candidate_mean >= target_bits &&
            candidate.accuracy->domain_score >= primary_domain_floor;

        return close_to_strongest_mean || precision_adequate_for_primary;
    }

    [[nodiscard]] inline preferred_reference_result preferred_available_reference(
        const metrics_record& record) noexcept
    {
        preferred_reference_candidate strongest{};
        bool has_strongest = false;

        const auto consider_strongest = [&](const preferred_reference_candidate& candidate)
        {
            if (!candidate.supported ||
                candidate.accuracy == nullptr ||
                candidate.benchmark == nullptr ||
                !has_benchmark_data(*candidate.benchmark))
            {
                return;
            }

            if (!has_strongest || reference_has_stronger_result(candidate, strongest))
            {
                strongest = candidate;
                has_strongest = true;
            }
        };

        const auto consider_all = [&](const auto& consider)
        {
            consider({
                record.competitor_name,
                &record.competitor_accuracy,
                &record.competitor_benchmark,
                record.competitor_special_values,
                record.competitor_supported
            });

            for (const competitor_result& competitor : record.extra_competitors)
            {
                consider({
                    competitor.name,
                    &competitor.accuracy,
                    &competitor.benchmark,
                    competitor.special_values,
                    competitor.supported
                });
            }
        };

        consider_all(consider_strongest);

        if (!has_strongest)
            return {};

        preferred_reference_candidate best = strongest;
        const auto consider_preferred = [&](const preferred_reference_candidate& candidate)
        {
            if (!candidate.supported ||
                candidate.accuracy == nullptr ||
                candidate.benchmark == nullptr ||
                !has_benchmark_data(*candidate.benchmark))
            {
                return;
            }
            if (!reference_is_quality_comparable(
                    record.suite.operation.name,
                    record.suite.precision,
                    record.suite.domain.role,
                    candidate,
                    strongest))
            {
                return;
            }

            if (benchmark_is_materially_faster(*candidate.benchmark, *best.benchmark) ||
                (!benchmark_is_materially_faster(*best.benchmark, *candidate.benchmark) &&
                    reference_has_stronger_result(candidate, best)))
            {
                best = candidate;
            }
        };

        consider_all(consider_preferred);

        return { best.name, best.accuracy, best.benchmark, true };
    }

    [[nodiscard]] inline std::string format_metrics_number(double value, int precision)
    {
        if (std::isinf(value))
            return value > 0.0 ? "exact" : "-exact";
        if (std::isnan(value))
            return "nan";

        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << value;
        return out.str();
    }

    [[nodiscard]] inline std::string format_metrics_benchmark(double value)
    {
        if (value <= 0.0)
            return "-";
        return format_metrics_number(value, 2);
    }

    [[nodiscard]] inline std::string format_metrics_ratio(double value)
    {
        if (value <= 0.0)
            return "-";
        return format_metrics_number(value, 2) + "x";
    }

    inline void write_csv_metric_number(std::ostream& out, double value)
    {
        if (std::isinf(value))
            out << (value > 0.0 ? "exact" : "-exact");
        else if (std::isnan(value))
            out << "nan";
        else
            out << value;
    }

    [[nodiscard]] inline std::string_view format_special_correctness(special_correctness value) noexcept
    {
        switch (value)
        {
        case special_correctness::pass: return "Yes";
        case special_correctness::fail: return "No";
        case special_correctness::unavailable: return "-";
        }
        return "-";
    }

    [[nodiscard]] inline std::string metrics_comparison_name(std::string_view name)
    {
        if (name.find("cpp_double_double") != std::string_view::npos)
            return "cppdd";
        if (name.find("mpfr_float_backend<64>") != std::string_view::npos)
            return "mpfr<64>";
        if (name.find("dd_real") != std::string_view::npos)
            return "dd_real";
        if (name.find("qd_real") != std::string_view::npos)
            return "qd_real";
        if (name.size() > 14)
            return std::string(name.substr(0, 14));
        return std::string(name);
    }

    struct metrics_console_report_summary
    {
        bool has_records = false;
        bool has_benchmarks = false;
        double average_precision_samples = 0.0;
        double average_benchmark_iterations = 0.0;
    };

    struct metrics_console_column_visibility
    {
        bool accuracy = true;
        bool domain = true;
        bool benchmark = true;
        bool special = true;

        [[nodiscard]] bool has_raw_metric() const noexcept
        {
            return accuracy || domain || benchmark || special;
        }

        [[nodiscard]] bool has_comparison_metric() const noexcept
        {
            return accuracy || domain || benchmark;
        }

        [[nodiscard]] bool has_preferred_reference() const noexcept
        {
            return benchmark;
        }
    };

    [[nodiscard]] inline std::string format_metrics_count(double value)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << value;
        return out.str();
    }

    [[nodiscard]] inline metrics_console_report_summary summarize_console_report(
        const std::vector<metrics_record>& records)
    {
        metrics_console_report_summary summary{};
        summary.has_records = !records.empty();
        if (records.empty())
            return summary;

        double precision_sample_total = 0.0;
        double benchmark_iteration_total = 0.0;
        std::size_t benchmark_record_count = 0;

        for (const metrics_record& record : records)
        {
            precision_sample_total += static_cast<double>(record.fltx_accuracy.sample_count);
            if (record.fltx_benchmark.iteration_count > 0)
            {
                benchmark_iteration_total += static_cast<double>(record.fltx_benchmark.iteration_count);
                ++benchmark_record_count;
            }
        }

        summary.average_precision_samples =
            precision_sample_total / static_cast<double>(records.size());
        if (benchmark_record_count > 0)
        {
            summary.has_benchmarks = true;
            summary.average_benchmark_iterations =
                benchmark_iteration_total / static_cast<double>(benchmark_record_count);
        }
        return summary;
    }

    class metrics_console_report_writer
    {
    public:
        metrics_console_report_writer(
            std::ostream& out,
            std::string_view title,
            std::string_view primary_competitor_name = "comp",
            std::vector<std::string_view> extra_competitor_names = {},
            std::string_view fltx_backend_name = "fltx",
            int operation_column_width_value = 10,
            double accuracy_equal_tolerance_bits = 0.005,
            double speed_equal_tolerance_ratio = 0.01,
            metrics_console_report_summary summary = {},
            metrics_console_column_visibility visible_columns = {})
            : stream(out),
              primary_competitor(primary_competitor_name),
              extra_competitors(std::move(extra_competitor_names)),
              fltx_backend(fltx_backend_name),
              operation_column_width(std::max(operation_column_width_value, default_operation_width)),
              accuracy_tolerance(accuracy_equal_tolerance_bits),
              speed_tolerance(speed_equal_tolerance_ratio),
              columns(visible_columns.has_raw_metric() ? visible_columns : metrics_console_column_visibility{})
        {
            state.copyfmt(stream);
            write_header(title, summary);
        }

        metrics_console_report_writer(const metrics_console_report_writer&) = delete;
        metrics_console_report_writer& operator=(const metrics_console_report_writer&) = delete;

        ~metrics_console_report_writer()
        {
            stream.copyfmt(state);
        }

        void write_record(const metrics_record& record)
        {
            begin_report_line();
            write_left_cell(record.suite.operation.name, operation_column_width, metadata_background, {}, true);
            write_raw_backend(
                record.competitor_accuracy,
                record.competitor_benchmark,
                record.competitor_special_values,
                record.competitor_supported,
                primary_competitor_background,
                true);
            for (std::string_view competitor_name : extra_competitors)
            {
                const competitor_result* competitor = find_extra_competitor(record, competitor_name);
                if (competitor == nullptr)
                {
                    write_empty_raw_backend(extra_competitor_background, true);
                    continue;
                }

                write_raw_backend(
                    competitor->accuracy,
                    competitor->benchmark,
                    competitor->special_values,
                    competitor->supported,
                    extra_competitor_background,
                    true);
            }
            write_raw_backend(
                record.fltx_accuracy,
                record.fltx_benchmark,
                record.fltx_special_values,
                true,
                fltx_background,
                true);
            write_comparison(
                record.fltx_accuracy,
                record.fltx_benchmark,
                record.competitor_accuracy,
                record.competitor_benchmark,
                record.competitor_supported,
                fltx_background,
                true);
            for (std::string_view competitor_name : extra_competitors)
            {
                const competitor_result* competitor = find_extra_competitor(record, competitor_name);
                if (competitor == nullptr)
                {
                    write_empty_comparison(fltx_background, true);
                    continue;
                }

                write_comparison(
                    record.fltx_accuracy,
                    record.fltx_benchmark,
                    competitor->accuracy,
                    competitor->benchmark,
                    competitor->supported,
                    fltx_background,
                    true);
            }

            if (columns.has_preferred_reference())
                write_preferred_reference_comparison(record, fltx_background, true);
            write_section_boundary();
            end_report_line();
            stream << std::flush;
        }

    private:
        static constexpr int default_operation_width = 10;
        static constexpr int bits_width = 6;
        static constexpr int domain_width = 6;
        static constexpr int benchmark_ns_width = 8;
        static constexpr int benchmark_speed_width = 6;
        static constexpr int special_width = 4;
        static constexpr int pair_group_width = bits_width * 2 + 1;
        static constexpr int domain_single_group_width = domain_width;
        static constexpr int benchmark_single_group_width = benchmark_ns_width;
        static constexpr int special_single_group_width = special_width;
        static constexpr int raw_backend_group_width =
            pair_group_width + domain_single_group_width + benchmark_single_group_width +
            special_single_group_width + 9;
        static constexpr int comparison_bits_group_width = pair_group_width;
        static constexpr int comparison_group_width =
            comparison_bits_group_width + domain_single_group_width + benchmark_speed_width + 4;
        static constexpr int preferred_reference_width = 9;
        static constexpr int preferred_reference_group_width =
            preferred_reference_width + comparison_group_width + 3;
        static constexpr std::string_view red = "\033[31m";
        static constexpr std::string_view green = "\033[32m";
        static constexpr std::string_view reset_foreground = "\033[39m";
        static constexpr std::string_view reset_background = "\033[49m";
        static constexpr std::string_view reset_all = "\033[0m";
        static constexpr std::string_view clear_to_end_of_line = "\033[K";
        static constexpr std::string_view metadata_background = "\033[48;2;24;25;28m";
        static constexpr std::string_view fltx_background = "\033[48;2;17;31;48m";
        static constexpr std::string_view primary_competitor_background = "\033[48;2;37;30;50m";
        static constexpr std::string_view extra_competitor_background = "\033[48;2;31;41;52m";

        std::ostream& stream;
        std::ios state{ nullptr };
        std::string_view primary_competitor;
        std::vector<std::string_view> extra_competitors;
        std::string_view fltx_backend;
        int operation_column_width;
        double accuracy_tolerance;
        double speed_tolerance;
        metrics_console_column_visibility columns;

        void write_repeated(char value, int count)
        {
            stream << std::string(static_cast<std::size_t>(std::max(count, 0)), value);
        }

        void write_background_repeated(std::string_view background, char value, int count)
        {
            if (!background.empty())
                stream << background;
            write_repeated(value, count);
            if (!background.empty())
                stream << reset_all;
        }

        void write_separator(std::string_view background = {})
        {
            if (!background.empty())
                stream << background;
            stream << '|';
            if (!background.empty())
                stream << reset_all;
        }

        void write_section_boundary()
        {
            stream << ' ';
        }

        void write_border_join(std::string_view background = {})
        {
            if (!background.empty())
                stream << background;
            stream << '+';
            if (!background.empty())
                stream << reset_all;
        }

        void end_report_line()
        {
            stream << reset_all << reset_background << clear_to_end_of_line << '\n';
        }

        void begin_report_line()
        {
            stream << reset_all;
        }

        [[nodiscard]] static std::string centered(std::string_view text, int width)
        {
            if (static_cast<int>(text.size()) >= width)
                return std::string(text.substr(0, static_cast<std::size_t>(width)));

            const int total_padding = width - static_cast<int>(text.size());
            const int left_padding = total_padding / 2;
            const int right_padding = total_padding - left_padding;
            return std::string(static_cast<std::size_t>(left_padding), ' ') +
                std::string(text) +
                std::string(static_cast<std::size_t>(right_padding), ' ');
        }

        [[nodiscard]] static int regular_group_width(std::initializer_list<int> widths) noexcept
        {
            int total = 0;
            int count = 0;
            for (const int width : widths)
            {
                if (width <= 0)
                    continue;
                total += width;
                ++count;
            }
            return count == 0 ? 0 : total + (count - 1) * 3;
        }

        [[nodiscard]] int raw_backend_visible_widths() const noexcept
        {
            return regular_group_width({
                columns.accuracy ? pair_group_width : 0,
                columns.domain ? domain_single_group_width : 0,
                columns.benchmark ? benchmark_single_group_width : 0,
                columns.special ? special_single_group_width : 0
            });
        }

        [[nodiscard]] int comparison_visible_width() const noexcept
        {
            if (columns.accuracy)
            {
                int width = pair_group_width - 2;
                if (columns.domain)
                    width += domain_single_group_width + 3;
                if (columns.benchmark)
                    width += comparison_speed_visible_width() + 3;
                return width;
            }

            return regular_group_width({
                columns.domain ? domain_single_group_width : 0,
                columns.benchmark ? comparison_speed_visible_width() : 0
            });
        }

        [[nodiscard]] int preferred_reference_visible_width() const noexcept
        {
            return columns.has_preferred_reference()
                ? preferred_reference_width + comparison_visible_width() + 3
                : 0;
        }

        [[nodiscard]] int comparison_speed_visible_width() const noexcept
        {
            return columns.accuracy || columns.domain ? benchmark_speed_width : 10;
        }

        void write_regular_border_group(
            std::string_view background,
            std::initializer_list<int> widths)
        {
            bool first = true;
            write_section_boundary();
            for (const int width : widths)
            {
                if (width <= 0)
                    continue;
                if (!first)
                    write_border_join(background);
                write_background_repeated(background, '-', width + 2);
                first = false;
            }
        }

        void write_raw_backend_border(std::string_view background)
        {
            write_regular_border_group(
                background,
                {
                    columns.accuracy ? pair_group_width : 0,
                    columns.domain ? domain_single_group_width : 0,
                    columns.benchmark ? benchmark_single_group_width : 0,
                    columns.special ? special_single_group_width : 0
                });
        }

        void write_comparison_border(std::string_view background)
        {
            if (columns.accuracy)
            {
                write_section_boundary();
                write_background_repeated(background, '-', comparison_bits_group_width);
                if (columns.domain)
                {
                    write_border_join(background);
                    write_background_repeated(background, '-', domain_single_group_width + 2);
                }
                if (columns.benchmark)
                {
                    write_border_join(background);
                    write_background_repeated(background, '-', comparison_speed_visible_width() + 2);
                }
                return;
            }

            write_regular_border_group(
                background,
                {
                    columns.domain ? domain_single_group_width : 0,
                    columns.benchmark ? comparison_speed_visible_width() : 0
                });
        }

        void write_preferred_reference_border(std::string_view background)
        {
            if (!columns.has_preferred_reference())
                return;

            write_regular_border_group(background, { preferred_reference_width });
            write_border_join(background);
            if (columns.accuracy)
            {
                write_background_repeated(background, '-', comparison_bits_group_width);
                if (columns.domain)
                {
                    write_border_join(background);
                    write_background_repeated(background, '-', domain_single_group_width + 2);
                }
                write_border_join(background);
                write_background_repeated(background, '-', comparison_speed_visible_width() + 2);
                return;
            }

            if (columns.domain)
            {
                write_background_repeated(background, '-', domain_single_group_width + 2);
                write_border_join(background);
            }
            write_background_repeated(background, '-', comparison_speed_visible_width() + 2);
        }

        void write_detail_border()
        {
            begin_report_line();
            write_section_boundary();
            write_background_repeated(metadata_background, '-', operation_column_width + 2);
            write_raw_backend_border(primary_competitor_background);
            for (std::size_t index = 0; index < extra_competitors.size(); ++index)
                write_raw_backend_border(extra_competitor_background);
            write_raw_backend_border(fltx_background);
            if (columns.has_comparison_metric())
            {
                write_comparison_border(fltx_background);
                for (std::size_t index = 0; index < extra_competitors.size(); ++index)
                    write_comparison_border(fltx_background);
            }
            write_preferred_reference_border(fltx_background);
            write_section_boundary();
            end_report_line();
        }

        void begin_cell(
            std::string_view background,
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            if (leading_section_boundary)
                write_section_boundary();
            else
                write_separator(leading_border_background);
            if (!background.empty())
                stream << background;
            stream << ' ';
        }

        void end_cell(std::string_view background)
        {
            stream << ' ';
            if (!background.empty())
                stream << reset_all;
        }

        void begin_tight_cell(
            std::string_view background,
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            if (leading_section_boundary)
                write_section_boundary();
            else
                write_separator(leading_border_background);
            if (!background.empty())
                stream << background;
        }

        void end_tight_cell(std::string_view background)
        {
            if (!background.empty())
                stream << reset_all;
        }

        void write_left_cell(
            std::string_view text,
            int width,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << std::left << std::setw(width) << text << std::right;
            end_cell(background);
        }

        void write_right_cell(
            std::string_view text,
            int width,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << std::right << std::setw(width) << text;
            end_cell(background);
        }

        void write_center_cell(
            std::string_view text,
            int width,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << centered(text, width);
            end_cell(background);
        }

        void write_tight_center_cell(
            std::string_view text,
            int width,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_tight_cell(background, leading_border_background, leading_section_boundary);
            stream << centered(text, width);
            end_tight_cell(background);
        }

        void write_final_superheader_cell(
            std::string_view text,
            int width,
            std::string_view background,
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            const int text_width = std::min(static_cast<int>(text.size()), width);
            const int total_padding = width - text_width;
            const int left_padding = total_padding / 2;
            const int right_padding = total_padding - left_padding;
            write_repeated(' ', left_padding);
            stream << text.substr(0, static_cast<std::size_t>(text_width));
            write_repeated(' ', right_padding);
            stream << ' ';
            stream << reset_all << reset_background;
        }

        [[nodiscard]] static double round_metrics_display_value(double value, int precision) noexcept
        {
            if (!std::isfinite(value))
                return value;
            const double scale = std::pow(10.0, static_cast<double>(precision));
            return std::round(value * scale) / scale;
        }

        [[nodiscard]] static std::string_view color_for_signed_value(double value, int precision) noexcept
        {
            const double rounded = round_metrics_display_value(value, precision);
            if (rounded < 0.0)
                return red;
            if (rounded > 0.0)
                return green;
            return {};
        }

        [[nodiscard]] static std::string_view color_for_speed_ratio(double value) noexcept
        {
            if (value <= 0.0)
                return {};

            const double rounded = round_metrics_display_value(value, 2);
            if (rounded < 1.0)
                return red;
            if (rounded > 1.0)
                return green;
            return {};
        }

        void write_metric_value(double value, int width, int precision)
        {
            const std::string text = format_metrics_number(value, precision);
            const int padding = width - static_cast<int>(text.size());
            if (padding > 0)
                write_repeated(' ', padding);

            stream << text;
        }

        void write_colored_metric_value(double value, std::string_view color, int width, int precision)
        {
            const std::string text = format_metrics_number(value, precision);
            const int padding = width - static_cast<int>(text.size());
            if (padding > 0)
                write_repeated(' ', padding);

            if (!color.empty())
                stream << color << text << reset_foreground;
            else
                stream << text;
        }

        void write_metric_text(std::string_view text, int width)
        {
            const int padding = width - static_cast<int>(text.size());
            if (padding > 0)
                write_repeated(' ', padding);
            stream << text;
        }

        void write_benchmark_metric(std::string_view text)
        {
            const int padding = benchmark_ns_width - static_cast<int>(text.size());
            if (padding > 0)
                write_repeated(' ', padding);
            stream << text;
        }

        void write_colored_benchmark_metric(
            std::string_view text,
            std::string_view color,
            int width = benchmark_speed_width)
        {
            const int padding = width - static_cast<int>(text.size());
            if (padding > 0)
                write_repeated(' ', padding);

            if (!color.empty())
                stream << color << text << reset_foreground;
            else
                stream << text;
        }

        void write_bits_metric(double value)
        {
            write_metric_value(value, bits_width, 1);
        }

        void write_colored_bits_metric(double value)
        {
            write_colored_metric_value(value, color_for_signed_value(value, 1), bits_width, 1);
        }

        void write_domain_metric(double value)
        {
            write_metric_value(value, domain_width, 1);
        }

        void write_colored_domain_metric(double value)
        {
            write_colored_metric_value(value, color_for_signed_value(value, 1), domain_width, 1);
        }

        void write_accuracy_pair(
            const accuracy_result& accuracy,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            if (std::isinf(accuracy.worst_bits) && accuracy.worst_bits > 0.0)
            {
                write_metric_text("exact", bits_width);
                stream << ' ';
                write_metric_text("exact", bits_width);
            }
            else
            {
                write_bits_metric(finite_display_bits(accuracy.mean_bits, accuracy.mean_bits));
                stream << ' ';
                write_bits_metric(finite_display_bits(accuracy.worst_bits, accuracy.mean_bits));
            }
            end_cell(background);
        }

        void write_domain_single(
            double score,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            write_domain_metric(score);
            end_cell(background);
        }

        void write_benchmark_single(
            double ns_per_iter,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            write_benchmark_metric(format_metrics_benchmark(ns_per_iter));
            end_cell(background);
        }

        void write_special_single(
            special_correctness special_values,
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            write_metric_text(format_special_correctness(special_values), special_width);
            end_cell(background);
        }

        void write_raw_backend(
            const accuracy_result& accuracy,
            const benchmark_result& benchmark,
            special_correctness special_values,
            bool supported,
            std::string_view background,
            bool leading_section_boundary = false)
        {
            if (!supported)
            {
                write_empty_raw_backend(background, leading_section_boundary);
                return;
            }

            struct cell_leading
            {
                std::string_view border_background;
                bool section_boundary = false;
            };
            bool first = true;
            const auto next_cell =
                [&first, leading_section_boundary, background]()
                {
                    const cell_leading leading{
                        first ? std::string_view{} : background,
                        first && leading_section_boundary
                    };
                    first = false;
                    return leading;
                };

            if (columns.accuracy)
            {
                const cell_leading leading = next_cell();
                if (has_accuracy_data(accuracy))
                    write_accuracy_pair(accuracy, background, leading.border_background, leading.section_boundary);
                else
                {
                    begin_cell(background, leading.border_background, leading.section_boundary);
                    write_metric_text("-", bits_width);
                    stream << ' ';
                    write_metric_text("-", bits_width);
                    end_cell(background);
                }
            }

            if (columns.domain)
            {
                const cell_leading leading = next_cell();
                if (has_accuracy_data(accuracy))
                    write_domain_single(accuracy.domain_score, background, leading.border_background, leading.section_boundary);
                else
                {
                    begin_cell(background, leading.border_background, leading.section_boundary);
                    write_metric_text("-", domain_width);
                    end_cell(background);
                }
            }

            if (columns.benchmark)
            {
                const cell_leading leading = next_cell();
                write_benchmark_single(benchmark.ns_per_iter, background, leading.border_background, leading.section_boundary);
            }
            if (columns.special)
            {
                const cell_leading leading = next_cell();
                write_special_single(special_values, background, leading.border_background, leading.section_boundary);
            }
        }

        void write_empty_raw_backend(
            std::string_view background,
            bool leading_section_boundary = false)
        {
            struct cell_leading
            {
                std::string_view border_background;
                bool section_boundary = false;
            };
            bool first = true;
            const auto next_cell =
                [&first, leading_section_boundary, background]()
                {
                    const cell_leading leading{
                        first ? std::string_view{} : background,
                        first && leading_section_boundary
                    };
                    first = false;
                    return leading;
                };

            if (columns.accuracy)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_metric_text("-", bits_width);
                stream << ' ';
                write_metric_text("-", bits_width);
                end_cell(background);
            }
            if (columns.domain)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_metric_text("-", domain_width);
                end_cell(background);
            }
            if (columns.benchmark)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_benchmark_metric("-");
                end_cell(background);
            }
            if (columns.special)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_metric_text("-", special_width);
                end_cell(background);
            }
        }

        void write_comparison_bits_pair(
            double mean_bits,
            double worst_bits,
            std::string_view background,
            bool leading_section_boundary = false)
        {
            begin_tight_cell(background, {}, leading_section_boundary);
            write_colored_metric_value(mean_bits, color_for_signed_value(mean_bits, 1), bits_width - 1, 1);
            stream << ' ';
            write_colored_bits_metric(worst_bits);
            stream << ' ';
            end_tight_cell(background);
        }

        void write_comparison_domain(
            double score,
            std::string_view background,
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            write_colored_domain_metric(score);
            end_cell(background);
        }

        void write_comparison_speed(
            double ratio,
            std::string_view background,
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            write_colored_benchmark_metric(
                format_metrics_ratio(ratio),
                color_for_speed_ratio(ratio),
                comparison_speed_visible_width());
            end_cell(background);
        }

        void write_comparison(
            const accuracy_result& fltx_accuracy,
            const benchmark_result& fltx_benchmark,
            const accuracy_result& competitor_accuracy,
            const benchmark_result& competitor_benchmark,
            bool supported,
            std::string_view background,
            bool leading_section_boundary = false)
        {
            if (!supported)
            {
                write_empty_comparison(background, leading_section_boundary);
                return;
            }

            struct cell_leading
            {
                std::string_view border_background;
                bool section_boundary = false;
            };
            bool first = true;
            const auto next_cell =
                [&first, leading_section_boundary, background]()
                {
                    const cell_leading leading{
                        first ? std::string_view{} : background,
                        first && leading_section_boundary
                    };
                    first = false;
                    return leading;
                };
            const bool has_accuracy =
                has_accuracy_data(fltx_accuracy) && has_accuracy_data(competitor_accuracy);

            if (columns.accuracy)
            {
                const cell_leading leading = next_cell();
                if (has_accuracy)
                {
                    write_comparison_bits_pair(
                        precision_mean_advantage_bits(fltx_accuracy, competitor_accuracy),
                        precision_worst_advantage_bits(fltx_accuracy, competitor_accuracy),
                        background,
                        leading.section_boundary);
                }
                else
                {
                    begin_tight_cell(background, leading.border_background, leading.section_boundary);
                    write_metric_text("-", bits_width - 1);
                    stream << ' ';
                    write_metric_text("-", bits_width);
                    stream << ' ';
                    end_tight_cell(background);
                }
            }

            if (columns.domain)
            {
                const cell_leading leading = next_cell();
                if (has_accuracy)
                    write_comparison_domain(domain_gap(fltx_accuracy, competitor_accuracy), background, leading.border_background, leading.section_boundary);
                else
                {
                    begin_cell(background, leading.border_background, leading.section_boundary);
                    write_metric_text("-", domain_width);
                    end_cell(background);
                }
            }

            if (columns.benchmark)
            {
                const cell_leading leading = next_cell();
                write_comparison_speed(
                    speed_ratio(fltx_benchmark, competitor_benchmark),
                    background,
                    leading.border_background,
                    leading.section_boundary);
            }
        }

        void write_empty_comparison(
            std::string_view background,
            bool leading_section_boundary = false)
        {
            struct cell_leading
            {
                std::string_view border_background;
                bool section_boundary = false;
            };
            bool first = true;
            const auto next_cell =
                [&first, leading_section_boundary, background]()
                {
                    const cell_leading leading{
                        first ? std::string_view{} : background,
                        first && leading_section_boundary
                    };
                    first = false;
                    return leading;
                };

            if (columns.accuracy)
            {
                const cell_leading leading = next_cell();
                begin_tight_cell(background, leading.border_background, leading.section_boundary);
                write_metric_text("-", bits_width - 1);
                stream << ' ';
                write_metric_text("-", bits_width);
                stream << ' ';
                end_tight_cell(background);
            }
            if (columns.domain)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_metric_text("-", domain_width);
                end_cell(background);
            }
            if (columns.benchmark)
            {
                const cell_leading leading = next_cell();
                begin_cell(background, leading.border_background, leading.section_boundary);
                write_colored_benchmark_metric("-", {}, comparison_speed_visible_width());
                end_cell(background);
            }
        }

        void write_preferred_reference_comparison(
            const metrics_record& record,
            std::string_view background,
            bool leading_section_boundary = false)
        {
            const preferred_reference_result reference = preferred_available_reference(record);
            if (!reference.supported || reference.accuracy == nullptr || reference.benchmark == nullptr)
            {
                write_left_cell("-", preferred_reference_width, background, {}, leading_section_boundary);
                write_empty_comparison(background);
                return;
            }

            const std::string reference_name = metrics_comparison_name(reference.name);
            write_left_cell(reference_name, preferred_reference_width, background, {}, leading_section_boundary);
            write_comparison(
                record.fltx_accuracy,
                record.fltx_benchmark,
                *reference.accuracy,
                *reference.benchmark,
                true,
                background);
        }

        void write_pair_subheader(
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << std::right << std::setw(bits_width) << "mean" << ' '
                   << std::right << std::setw(bits_width) << "worst";
            end_cell(background);
        }

        void write_tight_pair_subheader(
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_tight_cell(background, leading_border_background, leading_section_boundary);
            stream << std::right << std::setw(bits_width) << "mean" << ' '
                   << std::right << std::setw(bits_width) << "worst";
            end_tight_cell(background);
        }

        void write_comparison_bits_header(
            std::string_view background = {},
            bool leading_section_boundary = false)
        {
            begin_tight_cell(background, {}, leading_section_boundary);
            stream << " bits diff";
            write_repeated(' ', comparison_bits_group_width - 10);
            end_tight_cell(background);
        }

        void write_comparison_bits_subheader(
            std::string_view background = {},
            bool leading_section_boundary = false)
        {
            begin_tight_cell(background, {}, leading_section_boundary);
            stream << " mean  worst ";
            end_tight_cell(background);
        }

        void write_benchmark_single_subheader(
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << std::right << std::setw(benchmark_ns_width) << "ns";
            end_cell(background);
        }

        void write_comparison_group_header(
            std::string_view background = {},
            bool leading_section_boundary = false)
        {
            bool first = true;
            if (columns.accuracy)
            {
                write_comparison_bits_header(background, leading_section_boundary);
                first = false;
            }
            if (columns.domain)
            {
                write_center_cell(
                    "domain",
                    domain_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.benchmark)
            {
                write_center_cell(
                    "speed",
                    comparison_speed_visible_width(),
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
            }
        }

        void write_comparison_group_subheader(
            std::string_view background = {},
            bool leading_section_boundary = false)
        {
            bool first = true;
            if (columns.accuracy)
            {
                write_comparison_bits_subheader(background, leading_section_boundary);
                first = false;
            }
            if (columns.domain)
            {
                write_center_cell(
                    "score",
                    domain_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.benchmark)
            {
                write_center_cell(
                    "ratio",
                    comparison_speed_visible_width(),
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
            }
        }

        void write_domain_single_subheader(
            std::string_view background = {},
            std::string_view leading_border_background = {},
            bool leading_section_boundary = false)
        {
            begin_cell(background, leading_border_background, leading_section_boundary);
            stream << std::right << std::setw(domain_width) << "score";
            end_cell(background);
        }

        void write_raw_backend_metric_header(
            std::string_view background,
            bool leading_section_boundary = false)
        {
            bool first = true;
            if (columns.accuracy)
            {
                write_center_cell("bits accurate", pair_group_width, background, {}, leading_section_boundary);
                first = false;
            }
            if (columns.domain)
            {
                write_center_cell(
                    "domain",
                    domain_single_group_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.benchmark)
            {
                write_center_cell(
                    "bench",
                    benchmark_single_group_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.special)
            {
                write_center_cell(
                    "Inf/",
                    special_single_group_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
            }
        }

        void write_raw_backend_metric_subheader(
            std::string_view background,
            bool leading_section_boundary = false)
        {
            bool first = true;
            if (columns.accuracy)
            {
                write_pair_subheader(background, {}, leading_section_boundary);
                first = false;
            }
            if (columns.domain)
            {
                write_domain_single_subheader(
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.benchmark)
            {
                write_benchmark_single_subheader(
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
                first = false;
            }
            if (columns.special)
            {
                write_center_cell(
                    "NaN",
                    special_single_group_width,
                    background,
                    first ? std::string_view{} : background,
                    first && leading_section_boundary);
            }
        }

        void write_header(std::string_view title, const metrics_console_report_summary& summary)
        {
            stream << '\n';
            begin_report_line();
            stream << "[ " << title << " ]";
            end_report_line();
            if (summary.has_records)
            {
                if (columns.accuracy || columns.domain)
                {
                    begin_report_line();
                    stream << "average samples: "
                           << format_metrics_count(summary.average_precision_samples);
                    end_report_line();
                }
                if (columns.benchmark)
                {
                    begin_report_line();
                    stream << "benchmark iterations: ";
                    if (summary.has_benchmarks)
                        stream << format_metrics_count(summary.average_benchmark_iterations);
                    else
                        stream << '-';
                    end_report_line();
                }
            }
            begin_report_line();
            end_report_line();
            begin_report_line();
            write_center_cell("operation", operation_column_width, metadata_background, {}, true);
            const std::string primary_backend_name = metrics_comparison_name(primary_competitor);
            write_center_cell(primary_backend_name, raw_backend_visible_widths(), primary_competitor_background, {}, true);
            for (std::string_view competitor_name : extra_competitors)
            {
                const std::string backend_name = metrics_comparison_name(competitor_name);
                write_center_cell(
                    backend_name,
                    raw_backend_visible_widths(),
                    extra_competitor_background,
                    {},
                    true);
            }
            write_center_cell(fltx_backend, raw_backend_visible_widths(), fltx_background, {}, true);
            if (columns.has_comparison_metric())
            {
                const std::string primary_comparison_name = "VS " + metrics_comparison_name(primary_competitor);
                write_center_cell(primary_comparison_name, comparison_visible_width(), fltx_background, {}, true);
                for (std::string_view competitor_name : extra_competitors)
                {
                    const std::string comparison_name = "VS " + metrics_comparison_name(competitor_name);
                    write_center_cell(comparison_name, comparison_visible_width(), fltx_background, {}, true);
                }
            }
            if (columns.has_preferred_reference())
            {
                const std::string_view preferred_reference_title =
                    columns.accuracy || columns.domain
                        ? "VS preferred available reference"
                        : "VS preferred reference";
                write_final_superheader_cell(
                    preferred_reference_title,
                    preferred_reference_visible_width(),
                    fltx_background,
                    {},
                    true);
            }
            write_section_boundary();
            end_report_line();
            write_detail_border();
            begin_report_line();
            write_center_cell("", operation_column_width, metadata_background, {}, true);
            write_raw_backend_metric_header(primary_competitor_background, true);
            for (std::size_t remaining = extra_competitors.size(); remaining > 0; --remaining)
                write_raw_backend_metric_header(extra_competitor_background, true);
            write_raw_backend_metric_header(fltx_background, true);
            if (columns.has_comparison_metric())
            {
                write_comparison_group_header(fltx_background, true);
                for (std::size_t index = 0; index < extra_competitors.size(); ++index)
                    write_comparison_group_header(fltx_background, true);
            }
            if (columns.has_preferred_reference())
            {
                write_center_cell("reference", preferred_reference_width, fltx_background, {}, true);
                write_comparison_group_header(fltx_background);
            }
            write_section_boundary();
            end_report_line();
            begin_report_line();
            write_center_cell("", operation_column_width, metadata_background, {}, true);
            write_raw_backend_metric_subheader(primary_competitor_background, true);
            for (std::size_t index = 0; index < extra_competitors.size(); ++index)
                write_raw_backend_metric_subheader(extra_competitor_background, true);
            write_raw_backend_metric_subheader(fltx_background, true);
            if (columns.has_comparison_metric())
            {
                write_comparison_group_subheader(fltx_background, true);
                for (std::size_t index = 0; index < extra_competitors.size(); ++index)
                    write_comparison_group_subheader(fltx_background, true);
            }
            if (columns.has_preferred_reference())
            {
                write_center_cell("", preferred_reference_width, fltx_background, {}, true);
                write_comparison_group_subheader(fltx_background);
            }
            write_section_boundary();
            end_report_line();
            write_detail_border();
        }
    };

    inline void write_csv_header(
        std::ostream& out,
        const std::vector<std::string_view>& extra_competitor_names = {})
    {
        out << "group,label,precision,operation,domain,domain_role,samples,"
            << "primary_competitor,primary_competitor_supported,"
            << "fltx_worst_bits,fltx_mean_bits,fltx_domain_score,fltx_inf_nan_correct,"
            << "competitor_worst_bits,competitor_mean_bits,competitor_domain_score,competitor_inf_nan_correct,"
            << "precision_gap_bits,domain_gap_pp,fltx_ns_iter,competitor_ns_iter,speed_ratio,"
            << "preferred_reference,preferred_mean_gap_bits,preferred_worst_gap_bits,"
            << "preferred_domain_gap_pp,preferred_speed_ratio";
        for (std::string_view name : extra_competitor_names)
        {
            out << ','
                << name << "_supported,"
                << name << "_worst_bits,"
                << name << "_mean_bits,"
                << name << "_domain_score,"
                << name << "_inf_nan_correct,"
                << name << "_gap_bits,"
                << name << "_domain_gap_pp,"
                << name << "_ns_iter,"
                << name << "_speed_ratio";
        }
        out << '\n';
    }

    inline void write_csv_accuracy_fields(
        std::ostream& out,
        const accuracy_result& accuracy,
        special_correctness special_values)
    {
        if (has_accuracy_data(accuracy))
        {
            write_csv_metric_number(out, accuracy.worst_bits);
            out << ',';
            write_csv_metric_number(out, accuracy.mean_bits);
            out << ',';
            write_csv_metric_number(out, accuracy.domain_score);
            out << ',';
        }
        else
        {
            out << "-,-,-,";
        }
        write_csv_text(out, format_special_correctness(special_values));
    }

    inline void write_csv_gap_fields(
        std::ostream& out,
        const accuracy_result& fltx_accuracy,
        const accuracy_result& reference_accuracy)
    {
        if (has_accuracy_data(fltx_accuracy) && has_accuracy_data(reference_accuracy))
        {
            write_csv_metric_number(out, precision_gap_bits(fltx_accuracy, reference_accuracy));
            out << ',';
            write_csv_metric_number(out, domain_gap(fltx_accuracy, reference_accuracy));
        }
        else
        {
            out << "-,-";
        }
    }

    inline void write_csv_benchmark_fields(
        std::ostream& out,
        const benchmark_result& fltx_benchmark,
        const benchmark_result* reference_benchmark)
    {
        if (has_benchmark_data(fltx_benchmark))
            out << fltx_benchmark.ns_per_iter;
        else
            out << '-';
        out << ',';

        if (reference_benchmark != nullptr && has_benchmark_data(*reference_benchmark))
            out << reference_benchmark->ns_per_iter;
        else
            out << '-';
        out << ',';

        if (reference_benchmark != nullptr)
        {
            const double ratio = speed_ratio(fltx_benchmark, *reference_benchmark);
            if (ratio > 0.0)
                out << ratio;
            else
                out << '-';
        }
        else
        {
            out << '-';
        }
    }

    inline void write_csv_record(
        std::ostream& out,
        const metrics_record& record,
        const std::vector<std::string_view>& extra_competitor_names = {})
    {
        write_csv_text(out, metrics_csv_group(record));
        out << ',';
        write_csv_text(out, metrics_csv_label(record));
        out << ',';
        write_csv_text(out, to_string(record.suite.precision));
        out << ',';
        write_csv_text(out, record.suite.operation.name);
        out << ',';
        write_csv_text(out, record.suite.domain.name);
        out << ',';
        write_csv_text(out, to_string(record.suite.domain.role));
        out << ',';
        if (has_accuracy_data(record.fltx_accuracy))
            out << record.fltx_accuracy.sample_count;
        else
            out << '-';
        out << ',';
        write_csv_text(out, record.competitor_name);
        out << ','
            << (record.competitor_supported ? "yes" : "no") << ',';

        write_csv_accuracy_fields(out, record.fltx_accuracy, record.fltx_special_values);
        out << ',';

        if (record.competitor_supported)
        {
            write_csv_accuracy_fields(out, record.competitor_accuracy, record.competitor_special_values);
            out << ',';
            write_csv_gap_fields(out, record.fltx_accuracy, record.competitor_accuracy);
            out << ',';
            write_csv_benchmark_fields(out, record.fltx_benchmark, &record.competitor_benchmark);
        }
        else
        {
            out << "-,-,-,-,-,-,";
            write_csv_benchmark_fields(out, record.fltx_benchmark, nullptr);
        }

        const preferred_reference_result preferred = preferred_available_reference(record);
        if (preferred.supported && preferred.benchmark != nullptr)
        {
            out << ',';
            write_csv_text(out, preferred.name);
            out << ',';
            if (preferred.accuracy != nullptr &&
                has_accuracy_data(record.fltx_accuracy) &&
                has_accuracy_data(*preferred.accuracy))
            {
                write_csv_metric_number(out, precision_mean_advantage_bits(record.fltx_accuracy, *preferred.accuracy));
                out << ',';
                write_csv_metric_number(out, precision_worst_advantage_bits(record.fltx_accuracy, *preferred.accuracy));
                out << ',';
                write_csv_metric_number(out, domain_gap(record.fltx_accuracy, *preferred.accuracy));
            }
            else
            {
                out << "-,-,-";
            }
            out << ',';
            const double ratio = speed_ratio(record.fltx_benchmark, *preferred.benchmark);
            if (ratio > 0.0)
                out << ratio;
            else
                out << '-';
        }
        else
        {
            out << ",-,-,-,-,-";
        }

        for (std::string_view name : extra_competitor_names)
        {
            const competitor_result* competitor = find_extra_competitor(record, name);
            if (competitor == nullptr)
            {
                out << ",,,,,,,,,";
                continue;
            }

            out << ','
                << (competitor->supported ? "yes" : "no");
            if (competitor->supported)
            {
                out << ',';
                write_csv_accuracy_fields(out, competitor->accuracy, competitor->special_values);
                out << ',';
                write_csv_gap_fields(out, record.fltx_accuracy, competitor->accuracy);
                out << ',';
                if (has_benchmark_data(competitor->benchmark))
                    out << competitor->benchmark.ns_per_iter;
                else
                    out << '-';
                out << ',';
                const double ratio = speed_ratio(record.fltx_benchmark, competitor->benchmark);
                if (ratio > 0.0)
                    out << ratio;
                else
                    out << '-';
            }
            else
            {
                out << ",-,-,-,-,-,-,-,-";
            }
        }
        out << '\n';
    }

    #if defined(__EMSCRIPTEN__)
    inline bool write_node_host_text_file(
        const std::string& output_path,
        const std::string& content)
    {
        const int result = EM_ASM_INT({
            if (typeof ENVIRONMENT_IS_NODE === "undefined" || !ENVIRONMENT_IS_NODE)
                return 0;

            try {
                const fs = require("fs");
                const path = require("path");
                const outputPath = UTF8ToString($0);
                const text = UTF8ToString($1, $2);
                fs.mkdirSync(path.dirname(outputPath), { recursive: true });
                fs.writeFileSync(outputPath, text, "utf8");
                return 1;
            } catch (error) {
                if (typeof err === "function") {
                    const message = error && error.message ? error.message : String(error);
                    err("[metrics report] failed to write host file: " + message);
                }
                return -1;
            }
        }, output_path.c_str(), content.c_str(), content.size());

        if (result < 0)
            throw std::runtime_error("unable to write metrics report host output path: " + output_path);
        return result > 0;
    }
    #endif

    inline void write_text_file(
        const std::filesystem::path& output_path,
        const std::string& content)
    {
        #if defined(__EMSCRIPTEN__)
        if (write_node_host_text_file(output_path.string(), content))
            return;
        #endif

        if (const std::filesystem::path parent = output_path.parent_path(); !parent.empty())
            std::filesystem::create_directories(parent);

        std::ofstream out(output_path, std::ios::trunc);
        if (!out)
            throw std::runtime_error("unable to open metrics report output path: " + output_path.string());

        out << content;
    }

    inline void write_csv_report(
        const std::filesystem::path& output_path,
        const std::vector<metrics_record>& records)
    {
        std::ostringstream out;
        const std::vector<std::string_view> extra_competitor_names = collect_extra_competitor_names(records);
        write_csv_header(out, extra_competitor_names);
        std::vector<metrics_record> sorted_records = records;
        std::stable_sort(sorted_records.begin(), sorted_records.end(), metrics_csv_record_less);
        for (const metrics_record& record : sorted_records)
            write_csv_record(out, record, extra_competitor_names);

        write_text_file(output_path, out.str());
    }

    inline void write_console_report(
        std::ostream& out,
        std::string_view title,
        const std::vector<metrics_record>& records,
        double accuracy_equal_tolerance_bits = 0.005,
        double speed_equal_tolerance_ratio = 0.01,
        metrics_console_column_visibility visible_columns = {})
    {
        int operation_column_width = 10;
        for (const metrics_record& record : records)
        {
            operation_column_width = std::max(
                operation_column_width,
                static_cast<int>(record.suite.operation.name.size()));
        }

        const std::string_view primary_competitor_name =
            records.empty() || records.front().competitor_name.empty()
                ? std::string_view{ "comp" }
                : records.front().competitor_name;
        metrics_console_report_writer writer{
            out,
            title,
            primary_competitor_name,
            collect_extra_competitor_names(records),
            records.empty() ? std::string_view{ "fltx" } : metrics_fltx_backend_name(records.front().suite.precision),
            operation_column_width,
            accuracy_equal_tolerance_bits,
            speed_equal_tolerance_ratio,
            summarize_console_report(records),
            visible_columns
        };
        for (const metrics_record& record : records)
            writer.write_record(record);
    }
}

#endif
