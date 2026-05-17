#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <f128.h>
#include <f256_math.h>
#include <f256_io.h>

using big = boost::multiprecision::mpfr_float_100;

using namespace bl;

namespace
{
    using mpfr_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<320>,
        boost::multiprecision::et_off>;

    constexpr int checked_digits = std::numeric_limits<f256>::digits10 - 4;
    constexpr int printed_digits = std::numeric_limits<f256>::max_digits10;

    constexpr std::uint64_t random_seed = 1ull;
    constexpr int random_sample_count_scale = 500;

    constexpr const char* type_label = "f256";

    void print_random_run(const char* description, int count)
    {
        std::cout << type_label << " comparing: " << count << " " << description
                  << " (seed " << random_seed << ")...\n\n";
    }

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

    struct scaled_tolerance_entry
    {
        const char* op_name;
        const char* tolerance;
    };

    [[nodiscard]] mpfr_ref function_scaled_tolerance(const char* op_name)
    {
        constexpr std::array<scaled_tolerance_entry, 39> tolerances{{
            { "acos", "2e-60" },
            { "acosh", "5e-61" },
            { "add", "2e-65" },
            { "asin", "2e-60" },
            { "asinh", "2e-60" },
            { "atan", "8e-63" },
            { "atan2", "9e-63" },
            { "atanh", "1e-62" },
            { "cbrt", "1e-62" },
            { "cos", "3e-60" },
            { "cosh", "1e-62" },
            { "divide", "2e-64" },
            { "erf", "5e-60" },
            { "erfc", "5e-60" },
            { "exp", "1e-62" },
            { "exp2", "1e-62" },
            { "expm1", "9e-63" },
            { "fmod", "1e-62" },
            { "frexp", "2e-64" },
            { "hypot", "3e-60" },
            { "ldexp", "1e-62" },
            { "lgamma", "7e-60" },
            { "log", "1e-62" },
            { "log10", "1e-62" },
            { "log1p", "9e-63" },
            { "log2", "1e-62" },
            { "multiply", "2e-65" },
            { "pow", "1e-62" },
            { "pow10_256", "2e-65" },
            { "recip", "4e-66" },
            { "sin", "6e-60" },
            { "sincos.cos", "4e-63" },
            { "sincos.sin", "3e-63" },
            { "sinh", "9e-63" },
            { "sqrt", "3e-60" },
            { "subtract", "2e-65" },
            { "tan", "8e-63" },
            { "tanh", "5e-63" },
            { "tgamma", "6e-60" }
        }};

        for (const auto& entry : tolerances)
        {
            if (std::strcmp(op_name, entry.op_name) == 0)
                return mpfr_ref{ entry.tolerance };
        }

        return mpfr_ref{ 0 };
    }

    [[nodiscard]] mpfr_ref function_tolerance(const char* op_name, const mpfr_ref& scale)
    {
        return function_scaled_tolerance(op_name) * scale;
    }

    [[nodiscard]] mpfr_ref combined_tolerance(
        const char* op_name,
        const mpfr_ref& accuracy_scale,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance,
        const mpfr_ref& rel_scale)
    {
        const mpfr_ref rel_based_tolerance = rel_tolerance * rel_scale;
        mpfr_ref explicit_tolerance = abs_tolerance;
        if (rel_based_tolerance > explicit_tolerance)
            explicit_tolerance = rel_based_tolerance;

        const mpfr_ref scaled_tolerance = function_tolerance(op_name, accuracy_scale);
        if (scaled_tolerance == 0 || explicit_tolerance < scaled_tolerance)
            return explicit_tolerance;

        return scaled_tolerance;
    }

    [[nodiscard]] std::string to_text(const f256& value)
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

    [[nodiscard]] mpfr_ref to_ref_exact(const f256& value)
    {
        mpfr_ref sum = 0;
        sum += mpfr_ref{ value.x0 };
        sum += mpfr_ref{ value.x1 };
        sum += mpfr_ref{ value.x2 };
        sum += mpfr_ref{ value.x3 };
        return sum;
    }

    [[nodiscard]] mpfr_ref to_ref_exact(const f128_s& value)
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

    template<typename Scalar>
    [[nodiscard]] mpfr_ref scalar_to_ref(Scalar scalar)
    {
        using scalar_t = std::remove_cv_t<std::remove_reference_t<Scalar>>;
        if constexpr (std::is_same_v<scalar_t, f128_s>)
            return to_ref_exact(scalar);
        else
            return mpfr_ref{ scalar };
    }

    template<typename Scalar>
    [[nodiscard]] std::string scalar_to_text(Scalar scalar)
    {
        using scalar_t = std::remove_cv_t<std::remove_reference_t<Scalar>>;
        if constexpr (std::is_same_v<scalar_t, f128_s>)
            return to_text(f256{ scalar });
        else
            return to_text_double(static_cast<double>(scalar));
    }

    using ulp_count = boost::multiprecision::cpp_int;

    struct ulp_distance_result
    {
        ulp_count value = 0;
        bool exact = true;
    };

    [[nodiscard]] bool same_double_bits(double lhs, double rhs) noexcept
    {
        return std::bit_cast<std::uint64_t>(lhs) == std::bit_cast<std::uint64_t>(rhs);
    }

    [[nodiscard]] bool same_value_bits(const f256& lhs, const f256& rhs) noexcept
    {
        return same_double_bits(lhs.x0, rhs.x0) &&
               same_double_bits(lhs.x1, rhs.x1) &&
               same_double_bits(lhs.x2, rhs.x2) &&
               same_double_bits(lhs.x3, rhs.x3);
    }

    [[nodiscard]] mpfr_ref scale_by_power_of_two(mpfr_ref value, int exponent)
    {
        if (exponent > 0)
        {
            for (int i = 0; i < exponent; ++i)
                value *= 2;
        }
        else if (exponent < 0)
        {
            for (int i = 0; i < -exponent; ++i)
                value /= 2;
        }

        return value;
    }

    [[nodiscard]] f256 round_ref_to_f256(const mpfr_ref& value)
    {
        const std::string text = to_text(value);
        return to_f256(text.c_str());
    }

    [[nodiscard]] mpfr_ref nominal_ulp_size(const f256& reference, const f256& fallback)
    {
        double lead = std::fabs(reference.x0);
        if (lead == 0.0)
            lead = std::fabs(fallback.x0);
        if (lead == 0.0)
            return mpfr_ref{ std::numeric_limits<double>::denorm_min() };

        int exponent = 0;
        (void)std::frexp(lead, &exponent);
        return scale_by_power_of_two(mpfr_ref{ 1 }, exponent - std::numeric_limits<f256>::digits);
    }

    [[nodiscard]] ulp_count ceil_to_ulp_count(const mpfr_ref& value)
    {
        if (value <= 0)
            return 0;

        return boost::multiprecision::ceil(value).template convert_to<ulp_count>();
    }

    [[nodiscard]] ulp_distance_result ulp_distance(const f256& lhs, const f256& rhs)
    {
        if (bl::isnan(lhs) || bl::isnan(rhs))
            return { 0, false };

        if (!bl::isfinite(lhs) || !bl::isfinite(rhs))
            return { same_value_bits(lhs, rhs) ? ulp_count{ 0 } : ulp_count{ 1 }, same_value_bits(lhs, rhs) };

        const mpfr_ref lhs_ref = to_ref_exact(lhs);
        const mpfr_ref rhs_ref = to_ref_exact(rhs);
        if (lhs_ref == rhs_ref)
            return { 0, true };

        const mpfr_ref diff = abs_ref(lhs_ref - rhs_ref);
        const mpfr_ref ulp = nominal_ulp_size(rhs, lhs);
        return { ceil_to_ulp_count(diff / ulp), true };
    }

    [[nodiscard]] ulp_distance_result true_ulp_distance_from_reference(const f256& got, const mpfr_ref& expected)
    {
        return ulp_distance(got, round_ref_to_f256(expected));
    }

    [[nodiscard]] big random_finite_for_f256(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> exponent_dist(-100, 100);
        std::uniform_int_distribution<std::uint32_t> chunk_dist(0, 999999999);

        std::ostringstream mantissa_text;
        mantissa_text << (sign_dist(rng) != 0 ? "-0." : "0.");

        for (int i = 0; i < 8; ++i)
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


    struct accuracy_stats_entry
    {
        int samples = 0;
        int passed = 0;
        std::vector<double> achieved_digits;
        ulp_count worst_ulp = 0;
        bool worst_ulp_exact = true;
        int inexact_ulp_samples = 0;
        mpfr_ref worst_scaled_error = 0;
    };

    class accuracy_report_scope;
    thread_local accuracy_report_scope* current_accuracy_report_scope = nullptr;

    [[nodiscard]] double achieved_digits_from_error(const mpfr_ref& diff, const mpfr_ref& scale)
    {
        if (diff == 0)
            return static_cast<double>(checked_digits);

        const mpfr_ref scaled_error = diff / scale;
        if (scaled_error >= 1)
            return 0.0;

        const double scaled_error_double = scaled_error.convert_to<double>();
        if (!(scaled_error_double > 0.0))
            return static_cast<double>(checked_digits);

        const double digits = -std::log10(scaled_error_double);
        return digits < 0.0 ? 0.0 : digits;
    }

    [[nodiscard]] double normalized_accuracy_percent(double digits)
    {
        if (checked_digits <= 0)
            return 100.0;

        double ratio = digits / static_cast<double>(checked_digits);
        if (ratio < 0.0)
            ratio = 0.0;
        if (ratio > 1.0)
            ratio = 1.0;
        return ratio * 100.0;
    }

    [[nodiscard]] double median_digits(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;

        std::sort(values.begin(), values.end());
        const std::size_t mid = values.size() / 2;
        if ((values.size() & 1u) != 0u)
            return values[mid];

        return (values[mid - 1] + values[mid]) * 0.5;
    }

    class accuracy_report_scope
    {
    public:
        explicit accuracy_report_scope(const char* test_name, bool report_ulp = true)
            : test_name(test_name), report_ulp(report_ulp), previous(current_accuracy_report_scope)
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
                const double median = median_digits(entry.achieved_digits);
                const double worst = *std::min_element(entry.achieved_digits.begin(), entry.achieved_digits.end());

                std::cout << "  " << op_name
                          << ": pass " << entry.passed << "/" << entry.samples
                          << ", median " << median << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(median) << "%)"
                          << ", worst " << worst << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(worst) << "%)";

                if (report_ulp)
                {
                    std::cout << ", worst ulp ";
                    if (!entry.worst_ulp_exact)
                        std::cout << ">=";
                    std::cout << entry.worst_ulp;
                    if (entry.inexact_ulp_samples != 0)
                        std::cout << " (" << entry.inexact_ulp_samples << " capped)";
                }
                else
                {
                    std::cout << ", local ulp omitted";
                }
                std::cout << ", worst scaled error " << std::scientific << entry.worst_scaled_error << std::fixed;
                std::cout << "\n";
            }

            std::cout.flags(old_flags);
            std::cout.precision(old_precision);
        }

        void record(
            const char* op_name,
            const mpfr_ref& diff,
            const mpfr_ref& scale,
            const f256& got,
            const mpfr_ref& expected,
            bool passed)
        {
            auto& entry = stats[op_name];
            ++entry.samples;
            if (passed)
                ++entry.passed;
            entry.achieved_digits.push_back(achieved_digits_from_error(diff, scale));
            if (scale != 0)
            {
                const mpfr_ref scaled_error = diff / scale;
                if (scaled_error > entry.worst_scaled_error)
                    entry.worst_scaled_error = scaled_error;
            }

            const ulp_distance_result ulps = true_ulp_distance_from_reference(got, expected);
            if (!ulps.exact)
                ++entry.inexact_ulp_samples;

            if (ulps.value > entry.worst_ulp)
            {
                entry.worst_ulp = ulps.value;
                entry.worst_ulp_exact = ulps.exact;
            }
            else if (ulps.value == entry.worst_ulp && !ulps.exact)
            {
                entry.worst_ulp_exact = false;
            }
        }

    private:
        std::string test_name;
        bool report_ulp = true;
        accuracy_report_scope* previous = nullptr;
        std::map<std::string, accuracy_stats_entry> stats;
    };

    void record_accuracy_sample(
        const char* op_name,
        const f256& got,
        const mpfr_ref& expected,
        const mpfr_ref& diff,
        const mpfr_ref& scale,
        bool passed)
    {
        if (current_accuracy_report_scope != nullptr)
            current_accuracy_report_scope->record(op_name, diff, scale, got, expected, passed);
    }

    template<typename F256Op, typename RefOp>
    void check_binary_op(const char* op_name, const char* lhs_text, const char* rhs_text, F256Op&& f256_op, RefOp&& ref_op)
    {
        const f256 lhs = to_f256(lhs_text);
        const f256 rhs = to_f256(rhs_text);

        const f256 got = f256_op(lhs, rhs);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(to_ref_exact(lhs), to_ref_exact(rhs));

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = function_tolerance(op_name, scale);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(lhs_text);
        CAPTURE(rhs_text);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(op_name, got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    template<typename Scalar, typename F256Op, typename RefOp>
    void check_scalar_binary_op(
        const char* op_name,
        const char* case_label,
        const char* scalar_kind,
        const char* value_text,
        Scalar scalar,
        F256Op&& f256_op,
        RefOp&& ref_op)
    {
        const f256 value = to_f256(value_text);
        const mpfr_ref value_ref = to_ref_exact(value);
        const mpfr_ref scalar_ref = scalar_to_ref(scalar);

        const f256 got = f256_op(value, scalar);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(value_ref, scalar_ref);

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = function_tolerance(op_name, scale);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(case_label);
        CAPTURE(scalar_kind);
        CAPTURE(value_text);
        CAPTURE(scalar_to_text(scalar));
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(op_name, got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    template<typename F256Op, typename RefOp>
    void check_unary_op(const char* op_name, const char* input_text, F256Op&& f256_op, RefOp&& ref_op)
    {
        const f256 input = to_f256(input_text);

        const f256 got = f256_op(input);
        const mpfr_ref input_ref = to_ref_exact(input);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(input_ref);

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = function_tolerance(op_name, scale);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(input_text);
        CAPTURE(to_text(input));
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(input.x0));
        CAPTURE(to_text_double(input.x1));
        CAPTURE(to_text_double(input.x2));
        CAPTURE(to_text_double(input.x3));

        CAPTURE(to_text_double_hex(input.x0));
        CAPTURE(to_text_double_hex(input.x1));
        CAPTURE(to_text_double_hex(input.x2));
        CAPTURE(to_text_double_hex(input.x3));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(op_name, got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    template<typename F256Op, typename RefOp>
    void check_unary_op_with_tolerance(
        const char* op_name,
        const char* input_text,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance,
        F256Op&& f256_op,
        RefOp&& ref_op)
    {
        const f256 input = to_f256(input_text);

        const f256 got = f256_op(input);
        const mpfr_ref input_ref = to_ref_exact(input);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(input_ref);

        mpfr_ref accuracy_scale = abs_ref(expected);
        if (accuracy_scale < 1)
            accuracy_scale = 1;

        const mpfr_ref scale = abs_ref(expected);
        const mpfr_ref tolerance = combined_tolerance(op_name, accuracy_scale, abs_tolerance, rel_tolerance, scale);

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

        CAPTURE(to_text_double(input.x0));
        CAPTURE(to_text_double(input.x1));
        CAPTURE(to_text_double(input.x2));
        CAPTURE(to_text_double(input.x3));

        CAPTURE(to_text_double_hex(input.x0));
        CAPTURE(to_text_double_hex(input.x1));
        CAPTURE(to_text_double_hex(input.x2));
        CAPTURE(to_text_double_hex(input.x3));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(op_name, got, expected, diff, accuracy_scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    template<typename F256Op, typename RefOp>
    void check_binary_op_with_tolerance(
        const char* op_name,
        const char* lhs_text,
        const char* rhs_text,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance,
        F256Op&& f256_op,
        RefOp&& ref_op)
    {
        const f256 lhs = to_f256(lhs_text);
        const f256 rhs = to_f256(rhs_text);

        const f256 got = f256_op(lhs, rhs);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(to_ref_exact(lhs), to_ref_exact(rhs));

        mpfr_ref accuracy_scale = abs_ref(expected);
        if (accuracy_scale < 1)
            accuracy_scale = 1;

        const mpfr_ref scale = abs_ref(expected);
        const mpfr_ref tolerance = combined_tolerance(op_name, accuracy_scale, abs_tolerance, rel_tolerance, scale);

        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(op_name);
        CAPTURE(lhs_text);
        CAPTURE(rhs_text);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(to_text(abs_tolerance));
        CAPTURE(to_text(rel_tolerance));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(op_name, got, expected, diff, accuracy_scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    [[nodiscard]] mpfr_ref ref_floor(const mpfr_ref& value)
    {
        return boost::multiprecision::floor(value);
    }

    [[nodiscard]] mpfr_ref ref_ceil(const mpfr_ref& value)
    {
        return boost::multiprecision::ceil(value);
    }

    [[nodiscard]] mpfr_ref ref_trunc(const mpfr_ref& value)
    {
        return value < 0 ? ref_ceil(value) : ref_floor(value);
    }

    [[nodiscard]] mpfr_ref ref_fmod(const mpfr_ref& x, const mpfr_ref& y)
    {
        return x - ref_trunc(x / y) * y;
    }

    [[nodiscard]] mpfr_ref ref_round_to_even(const mpfr_ref& value)
    {
        mpfr_ref rounded = ref_floor(value + mpfr_ref{ "0.5" });
        if ((rounded - value) == mpfr_ref{ "0.5" } && ref_fmod(rounded, mpfr_ref{ 2 }) != mpfr_ref{ 0 })
            rounded -= 1;
        return rounded;
    }
    [[nodiscard]] mpfr_ref ref_round_half_away_zero(const mpfr_ref& value)
    {
        return value < 0
            ? ref_ceil(value - mpfr_ref{ "0.5" })
            : ref_floor(value + mpfr_ref{ "0.5" });
    }

    [[nodiscard]] const mpfr_ref& ln2_ref()
    {
        static const mpfr_ref value = boost::multiprecision::log(mpfr_ref{ 2 });
        return value;
    }

    [[nodiscard]] const mpfr_ref& ln10_ref()
    {
        static const mpfr_ref value = boost::multiprecision::log(mpfr_ref{ 10 });
        return value;
    }

    [[nodiscard]] mpfr_ref ref_exp2(const mpfr_ref& value)
    {
        return boost::multiprecision::exp(value * ln2_ref());
    }

    [[nodiscard]] mpfr_ref ref_log2(const mpfr_ref& value)
    {
        return boost::multiprecision::log(value) / ln2_ref();
    }

    [[nodiscard]] mpfr_ref ref_log10(const mpfr_ref& value)
    {
        return boost::multiprecision::log(value) / ln10_ref();
    }

    [[nodiscard]] bool ref_is_integer(const mpfr_ref& value)
    {
        return boost::multiprecision::floor(value) == value;
    }

    [[nodiscard]] mpfr_ref ref_powi(mpfr_ref base, long long exponent)
    {
        if (exponent == 0)
            return mpfr_ref{ 1 };

        bool invert = exponent < 0;
        unsigned long long e = invert
            ? static_cast<unsigned long long>(-(exponent + 1)) + 1ull
            : static_cast<unsigned long long>(exponent);

        mpfr_ref result{ 1 };
        while (e != 0)
        {
            if ((e & 1ull) != 0)
                result *= base;
            e >>= 1ull;
            if (e != 0)
                base *= base;
        }

        return invert ? (mpfr_ref{ 1 } / result) : result;
    }

    [[nodiscard]] mpfr_ref ref_pow(const mpfr_ref& base, const mpfr_ref& exponent)
    {
        if (base < 0 && ref_is_integer(exponent))
        {
            const long long n = exponent.convert_to<long long>();
            const mpfr_ref mag = ref_powi(-base, n);
            return (n & 1LL) ? -mag : mag;
        }

        return boost::multiprecision::exp(exponent * boost::multiprecision::log(base));
    }

    [[nodiscard]] mpfr_ref ref_ldexp(mpfr_ref value, int exponent)
    {
        if (exponent > 0)
        {
            for (int i = 0; i < exponent; ++i)
                value *= 2;
        }
        else if (exponent < 0)
        {
            for (int i = 0; i < -exponent; ++i)
                value /= 2;
        }

        return value;
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

    [[nodiscard]] mpfr_ref random_unit_interval_for_f256(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<std::uint32_t> chunk_dist(0, 999999999);

        std::ostringstream text;
        text << "0.";

        for (int i = 0; i < 8; ++i)
            text << std::setw(9) << std::setfill('0') << chunk_dist(rng);

        return mpfr_ref{ text.str() };
    }

    [[nodiscard]] mpfr_ref random_sine_kernel_argument_for_f256(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);

        mpfr_ref value = random_unit_interval_for_f256(rng) * (pi_ref() / 4);
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] mpfr_ref random_sine_reduction_argument_for_f256(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<long long> multiple_dist(-1000000LL, 1000000LL);

        const mpfr_ref half_pi = pi_ref() / 2;
        const mpfr_ref quarter_pi = pi_ref() / 4;

        mpfr_ref offset = random_unit_interval_for_f256(rng) * (quarter_pi * 2);
        offset -= quarter_pi;

        return mpfr_ref{ static_cast<std::int64_t>(multiple_dist(rng)) } * half_pi + offset;
    }

    void check_sin_case(
        const char* label,
        const mpfr_ref& input,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("label: " << label);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });
    }

    void check_cos_case(
        const char* label,
        const mpfr_ref& input,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("label: " << label);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });
    }

    [[nodiscard]] mpfr_ref random_signed_interval_for_f256(std::mt19937_64& rng, const mpfr_ref& limit)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);

        mpfr_ref value = random_unit_interval_for_f256(rng) * limit;
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] mpfr_ref random_signed_interval_away_from_zero_for_f256(
        std::mt19937_64& rng,
        const mpfr_ref& minimum,
        const mpfr_ref& width)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);

        mpfr_ref value = minimum + random_unit_interval_for_f256(rng) * width;
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] mpfr_ref random_positive_for_f256(std::mt19937_64& rng)
    {
        mpfr_ref value = abs_ref(mpfr_ref{ random_finite_for_f256(rng) });
        if (value == 0)
            value = mpfr_ref{ "0.5" };
        return value;
    }

    [[nodiscard]] mpfr_ref random_nonzero_for_f256(std::mt19937_64& rng)
    {
        mpfr_ref value = mpfr_ref{ random_finite_for_f256(rng) };
        if (value == 0)
            value = mpfr_ref{ "0.5" };
        return value;
    }

    [[nodiscard]] mpfr_ref random_pow_base_for_f256(std::mt19937_64& rng)
    {
        return mpfr_ref{ "0.125" } + random_unit_interval_for_f256(rng) * mpfr_ref{ "7.875" };
    }

    void check_ldexp_case(const char* label, const mpfr_ref& input, int exponent)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("label: " << label);
        INFO("input_text: " << input_text);
        INFO("exponent: " << exponent);

        const f256 input_value = to_f256(input_text.c_str());
        const f256 got = bl::ldexp(input_value, exponent);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_ldexp(to_ref_exact(input_value), exponent);

        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = function_tolerance("ldexp", scale);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(label);
        CAPTURE(input_text);
        CAPTURE(exponent);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample("ldexp", got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    void check_pow_case(
        const char* label,
        const mpfr_ref& base,
        const mpfr_ref& exponent,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string base_text = to_scientific_text(base, printed_digits + 6);
        const std::string exponent_text = to_scientific_text(exponent, printed_digits + 6);

        INFO("label: " << label);
        INFO("base_text: " << base_text);
        INFO("exponent_text: " << exponent_text);

        check_binary_op_with_tolerance(
            "pow",
            base_text.c_str(),
            exponent_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& x, const f256& y) { return bl::pow(x, y); },
            [](const mpfr_ref& x, const mpfr_ref& y) { return ref_pow(x, y); });
    }


[[nodiscard]] mpfr_ref ref_remainder_ties_even(const mpfr_ref& x, const mpfr_ref& y)
    {
        return x - ref_round_to_even(x / y) * y;
    }

    [[nodiscard]] mpfr_ref ref_cbrt(const mpfr_ref& value)
    {
        if (value < 0)
            return -boost::multiprecision::cbrt(-value);
        return boost::multiprecision::cbrt(value);
    }

    [[nodiscard]] mpfr_ref ref_hypot(const mpfr_ref& x, const mpfr_ref& y)
    {
        return boost::multiprecision::sqrt(x * x + y * y);
    }

    [[nodiscard]] mpfr_ref ref_pow10(int exponent)
    {
        return ref_powi(mpfr_ref{ 10 }, exponent);
    }

    [[nodiscard]] mpfr_ref random_unit_symmetric_for_f256(std::mt19937_64& rng)
    {
        return random_signed_interval_for_f256(rng, mpfr_ref{ 1.0 });
    }

    [[nodiscard]] mpfr_ref random_atanh_argument_for_f256(std::mt19937_64& rng)
    {
        return random_signed_interval_for_f256(rng, mpfr_ref{ "0.95" });
    }

    [[nodiscard]] mpfr_ref random_acosh_argument_for_f256(std::mt19937_64& rng)
    {
        return mpfr_ref{ 1.0 } + random_unit_interval_for_f256(rng) * mpfr_ref{ 64.0 };
    }

    [[nodiscard]] mpfr_ref random_log1p_argument_for_f256(std::mt19937_64& rng)
    {
        return mpfr_ref{ "-0.95" } + random_unit_interval_for_f256(rng) * mpfr_ref{ "16.95" };
    }

    [[nodiscard]] mpfr_ref random_gamma_positive_for_f256(std::mt19937_64& rng)
    {
        return mpfr_ref{ "0.125" } + random_unit_interval_for_f256(rng) * mpfr_ref{ "15.875" };
    }

    [[nodiscard]] mpfr_ref random_erf_argument_for_f256(std::mt19937_64& rng)
    {
        return random_signed_interval_for_f256(rng, mpfr_ref{ 4.0 });
    }

    void require_exact_value(const char* label, const f256& got, const f256& expected)
    {
        CAPTURE(label);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));
        CAPTURE(to_text_double(expected.x0));
        CAPTURE(to_text_double(expected.x1));
        CAPTURE(to_text_double(expected.x2));
        CAPTURE(to_text_double(expected.x3));
        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));
        CAPTURE(to_text_double_hex(expected.x0));
        CAPTURE(to_text_double_hex(expected.x1));
        CAPTURE(to_text_double_hex(expected.x2));
        CAPTURE(to_text_double_hex(expected.x3));

        REQUIRE(got.x0 == expected.x0);
        REQUIRE(got.x1 == expected.x1);
        REQUIRE(got.x2 == expected.x2);
        REQUIRE(got.x3 == expected.x3);
    }

    [[nodiscard]] bool same_bits(double lhs, double rhs) noexcept
    {
        return std::bit_cast<std::uint64_t>(lhs) == std::bit_cast<std::uint64_t>(rhs);
    }

    void require_canonical_value(const char* label, const f256& got)
    {
        CAPTURE(label);
        CAPTURE(to_text(got));
        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        if (!bl::isfinite(got) || bl::iszero(got))
            return;

        const f256 expected = detail::_f256::renorm4(got.x0, got.x1, got.x2, got.x3);
        CAPTURE(to_text(expected));
        CAPTURE(to_text_double_hex(expected.x0));
        CAPTURE(to_text_double_hex(expected.x1));
        CAPTURE(to_text_double_hex(expected.x2));
        CAPTURE(to_text_double_hex(expected.x3));

        REQUIRE(same_bits(got.x0, expected.x0));
        REQUIRE(same_bits(got.x1, expected.x1));
        REQUIRE(same_bits(got.x2, expected.x2));
        REQUIRE(same_bits(got.x3, expected.x3));
    }

    void check_sincos_case(
        const char* label,
        const mpfr_ref& input,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string input_text = to_scientific_text(input, printed_digits + 6);
        const f256 input_value = to_f256(input_text.c_str());

        f256 got_s{};
        f256 got_c{};
        const bool ok = bl::sincos(input_value, got_s, got_c);

        INFO("label: " << label);
        INFO("input_text: " << input_text);
        REQUIRE(ok);

        const mpfr_ref expected_s = boost::multiprecision::sin(to_ref_exact(input_value));
        const mpfr_ref expected_c = boost::multiprecision::cos(to_ref_exact(input_value));

        {
            const mpfr_ref got_ref = to_ref_exact(got_s);
            mpfr_ref accuracy_scale = abs_ref(expected_s);
            if (accuracy_scale < 1)
                accuracy_scale = 1;

            const mpfr_ref scale = abs_ref(expected_s);
            const mpfr_ref tolerance = combined_tolerance("sincos.sin", accuracy_scale, abs_tolerance, rel_tolerance, scale);

            const mpfr_ref diff = abs_ref(got_ref - expected_s);
            CAPTURE(to_text(got_s));
            CAPTURE(to_text(expected_s));
            CAPTURE(to_text(diff));
            CAPTURE(to_text(tolerance));
            record_accuracy_sample("sincos.sin", got_s, expected_s, diff, accuracy_scale, diff <= tolerance);
            REQUIRE(diff <= tolerance);
        }

        {
            const mpfr_ref got_ref = to_ref_exact(got_c);
            mpfr_ref accuracy_scale = abs_ref(expected_c);
            if (accuracy_scale < 1)
                accuracy_scale = 1;

            const mpfr_ref scale = abs_ref(expected_c);
            const mpfr_ref tolerance = combined_tolerance("sincos.cos", accuracy_scale, abs_tolerance, rel_tolerance, scale);

            const mpfr_ref diff = abs_ref(got_ref - expected_c);
            CAPTURE(to_text(got_c));
            CAPTURE(to_text(expected_c));
            CAPTURE(to_text(diff));
            CAPTURE(to_text(tolerance));
            record_accuracy_sample("sincos.cos", got_c, expected_c, diff, accuracy_scale, diff <= tolerance);
            REQUIRE(diff <= tolerance);
        }
    }

    void check_remquo_case(
        const char* label,
        const mpfr_ref& x,
        const mpfr_ref& y,
        const mpfr_ref& abs_tolerance,
        const mpfr_ref& rel_tolerance)
    {
        const std::string lhs_text = to_scientific_text(x, printed_digits + 6);
        const std::string rhs_text = to_scientific_text(y, printed_digits + 6);

        const f256 lhs = to_f256(lhs_text.c_str());
        const f256 rhs = to_f256(rhs_text.c_str());

        int got_quo = 0;
        const f256 got = bl::remquo(lhs, rhs, &got_quo);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_remainder_ties_even(to_ref_exact(lhs), to_ref_exact(rhs));

        const mpfr_ref n = ref_round_to_even(to_ref_exact(lhs) / to_ref_exact(rhs));
        int expected_quo = static_cast<int>(ref_trunc(ref_fmod(abs_ref(n), mpfr_ref{ 2147483648.0 })).convert_to<long long>());
        if (n < 0)
            expected_quo = -expected_quo;

        mpfr_ref accuracy_scale = abs_ref(expected);
        if (accuracy_scale < 1)
            accuracy_scale = 1;

        const mpfr_ref scale = abs_ref(expected);
        const mpfr_ref tolerance = combined_tolerance("remquo", accuracy_scale, abs_tolerance, rel_tolerance, scale);

        const mpfr_ref diff = abs_ref(got_ref - expected);

        INFO("label: " << label);
        CAPTURE(lhs_text);
        CAPTURE(rhs_text);
        CAPTURE(got_quo);
        CAPTURE(expected_quo);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));

        record_accuracy_sample("remquo", got, expected, diff, accuracy_scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
        REQUIRE(got_quo == expected_quo);
    }

    struct scalar_mixed_recurrence_case
    {
        const char* x = "0";
        const char* y = "0";
        const char* a = "0";
        const char* b = "0";
        const char* c = "1";
        const char* d = "1";
    };

    template<typename T>
    struct scalar_mixed_recurrence_state
    {
        T x{};
        T y{};
        T a{};
        T b{};
        T c{};
        T d{};
    };

    template<typename T>
    [[nodiscard]] T scalar_mixed_recurrence_value(const char* text);

    template<>
    [[nodiscard]] f256 scalar_mixed_recurrence_value<f256>(const char* text)
    {
        return to_f256(text);
    }

    template<>
    [[nodiscard]] mpfr_ref scalar_mixed_recurrence_value<mpfr_ref>(const char* text)
    {
        return to_ref_exact(to_f256(text));
    }

    template<typename T>
    [[nodiscard]] T blend_scalar_mixed_recurrence_result(const T& value, const T& acc)
    {
        const T scaled_acc = acc * 0.25;
        return value + scaled_acc;
    }

    template<typename T, std::size_t ValueCount>
    [[nodiscard]] std::pair<T, T> run_scalar_mixed_recurrence_precision(
        const std::array<scalar_mixed_recurrence_case, ValueCount>& specs,
        int rounds)
    {
        std::array<scalar_mixed_recurrence_state<T>, ValueCount> state{};
        for (std::size_t i = 0; i < ValueCount; ++i)
        {
            state[i].x = scalar_mixed_recurrence_value<T>(specs[i].x);
            state[i].y = scalar_mixed_recurrence_value<T>(specs[i].y);
            state[i].a = scalar_mixed_recurrence_value<T>(specs[i].a);
            state[i].b = scalar_mixed_recurrence_value<T>(specs[i].b);
            state[i].c = scalar_mixed_recurrence_value<T>(specs[i].c);
            state[i].d = scalar_mixed_recurrence_value<T>(specs[i].d);
        }

        constexpr std::array<double, 8> add_rhs{ 0.125, -0.1875, 0.3125, -0.4375, 0.5625, -0.6875, 0.8125, -0.9375 };
        constexpr std::array<double, 8> add_lhs{ -0.03125, 0.09375, -0.15625, 0.21875, -0.28125, 0.34375, -0.40625, 0.46875 };
        constexpr std::array<double, 8> mul_rhs{ 0.875, -1.125, 1.375, -0.625, 0.5625, -0.8125, 1.0625, -1.3125 };
        constexpr std::array<double, 8> mul_lhs{ -1.0625, 0.6875, -0.9375, 1.1875, -0.75, 1.5, -1.25, 0.8125 };
        constexpr std::array<double, 8> div_rhs{ 1.125, -1.375, 1.625, -1.875, 2.125, -2.375, 2.625, -2.875 };

        T acc_x = state.front().x;
        T acc_y = state.front().y;

        for (int round = 0; round < rounds; ++round)
        {
            for (std::size_t i = 0; i < ValueCount; ++i)
            {
                auto& item = state[i];
                const std::size_t scalar_index = (i + static_cast<std::size_t>(round)) % add_rhs.size();

                const T scalar_rhs_add = item.x + add_rhs[scalar_index];
                const T scalar_lhs_add = add_lhs[scalar_index] + item.y;
                const T scalar_rhs_mul = scalar_rhs_add * mul_rhs[scalar_index];
                const T scalar_lhs_mul = mul_lhs[scalar_index] * scalar_lhs_add;
                const T scalar_div = scalar_rhs_mul / div_rhs[scalar_index];

                const T qd_add = scalar_div + item.a;
                const T qd_sub = scalar_lhs_mul - item.b;
                const T qd_mul = qd_add * qd_sub;
                const T c2 = item.c * item.c;
                const T d2 = item.d * item.d;
                const T denominator_base = c2 + d2;
                const T denominator = denominator_base + 1.0;
                const T qd_div = qd_mul / denominator;

                const T x_denominator = denominator + 2.0;
                const T c_half = item.c * 0.5;
                const T y_denominator_base = denominator + c_half;
                const T y_denominator = y_denominator_base + 2.5;

                const T x_step = qd_div / x_denominator;
                const T x_damping = 0.125 * item.a;
                const T y_delta = qd_add - qd_sub;
                const T y_step = y_delta / y_denominator;
                const T y_damping = item.b * 0.0625;

                item.x = x_step + x_damping;
                item.y = y_step - y_damping;

                acc_x = blend_scalar_mixed_recurrence_result(item.x, acc_x);
                acc_y = blend_scalar_mixed_recurrence_result(item.y, acc_y);
            }
        }

        return { acc_x, acc_y };
    }

    [[nodiscard]] mpfr_ref scalar_mixed_recurrence_tolerance(const mpfr_ref& scale, int rounds, std::size_t state_count)
    {
        constexpr int operations_per_state_update = 24;
        const mpfr_ref operation_count =
            mpfr_ref{ static_cast<int>(state_count) } *
            mpfr_ref{ rounds } *
            mpfr_ref{ operations_per_state_update };
        return function_scaled_tolerance("divide") * operation_count * scale;
    }

    void check_scalar_mixed_recurrence_component(
        const char* component,
        const f256& got,
        const mpfr_ref& expected,
        int rounds,
        std::size_t state_count)
    {
        const mpfr_ref got_ref = to_ref_exact(got);
        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = scalar_mixed_recurrence_tolerance(scale, rounds, state_count);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(component);
        CAPTURE(rounds);
        CAPTURE(state_count);
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));
        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample("scalar_mixed_recurrence", got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    struct fused_expression_case
    {
        std::string a;
        std::string b;
        std::string c;
        std::string d;
        std::string e;
        double scalar = 1.0;
        double denom_plus = 2.0;
        double denom_minus = 3.0;
    };

    struct fused_expression_budget
    {
        int add = 0;
        int subtract = 0;
        int multiply = 0;
        int divide = 0;
    };

    [[nodiscard]] mpfr_ref fused_expression_tolerance(const mpfr_ref& scale, fused_expression_budget budget)
    {
        mpfr_ref tolerance = 0;
        tolerance += function_scaled_tolerance("add") * budget.add;
        tolerance += function_scaled_tolerance("subtract") * budget.subtract;
        tolerance += function_scaled_tolerance("multiply") * budget.multiply;
        tolerance += function_scaled_tolerance("divide") * budget.divide;
        return tolerance * scale;
    }

    void update_expression_scale(mpfr_ref& scale, const mpfr_ref& value)
    {
        const mpfr_ref magnitude = abs_ref(value);
        if (magnitude > scale)
            scale = magnitude;
    }

    [[nodiscard]] mpfr_ref fused_expression_scale(
        const mpfr_ref& expected,
        const mpfr_ref& a,
        const mpfr_ref& b,
        const mpfr_ref& c,
        const mpfr_ref& d,
        const mpfr_ref& e,
        const mpfr_ref& scalar,
        const mpfr_ref& denom_plus,
        const mpfr_ref& denom_minus)
    {
        mpfr_ref scale = 1;
        update_expression_scale(scale, expected);
        update_expression_scale(scale, a);
        update_expression_scale(scale, b);
        update_expression_scale(scale, c);
        update_expression_scale(scale, d);
        update_expression_scale(scale, e);
        update_expression_scale(scale, a * b);
        update_expression_scale(scale, c * d);
        update_expression_scale(scale, a * scalar);
        update_expression_scale(scale, b * scalar);
        update_expression_scale(scale, e + denom_plus);
        update_expression_scale(scale, denom_minus - e);
        return scale;
    }

    template<typename F256Op, typename RefOp>
    void check_fused_expression_case(
        const char* label,
        const fused_expression_case& spec,
        std::size_t case_index,
        fused_expression_budget budget,
        F256Op&& f256_op,
        RefOp&& ref_op)
    {
        const f256 a = to_f256(spec.a.c_str());
        const f256 b = to_f256(spec.b.c_str());
        const f256 c = to_f256(spec.c.c_str());
        const f256 d = to_f256(spec.d.c_str());
        const f256 e = to_f256(spec.e.c_str());

        const mpfr_ref a_ref = to_ref_exact(a);
        const mpfr_ref b_ref = to_ref_exact(b);
        const mpfr_ref c_ref = to_ref_exact(c);
        const mpfr_ref d_ref = to_ref_exact(d);
        const mpfr_ref e_ref = to_ref_exact(e);
        const mpfr_ref scalar_ref{ spec.scalar };
        const mpfr_ref denom_plus_ref{ spec.denom_plus };
        const mpfr_ref denom_minus_ref{ spec.denom_minus };

        const f256 got = f256_op(a, b, c, d, e, spec.scalar, spec.denom_plus, spec.denom_minus);
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref expected = ref_op(
            a_ref,
            b_ref,
            c_ref,
            d_ref,
            e_ref,
            scalar_ref,
            denom_plus_ref,
            denom_minus_ref);

        const mpfr_ref scale = fused_expression_scale(
            expected,
            a_ref,
            b_ref,
            c_ref,
            d_ref,
            e_ref,
            scalar_ref,
            denom_plus_ref,
            denom_minus_ref);
        const mpfr_ref tolerance = fused_expression_tolerance(scale, budget);
        const mpfr_ref diff = abs_ref(got_ref - expected);

        CAPTURE(label);
        CAPTURE(case_index);
        CAPTURE(spec.a);
        CAPTURE(spec.b);
        CAPTURE(spec.c);
        CAPTURE(spec.d);
        CAPTURE(spec.e);
        CAPTURE(to_text_double(spec.scalar));
        CAPTURE(to_text_double(spec.denom_plus));
        CAPTURE(to_text_double(spec.denom_minus));
        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        CAPTURE(to_text(scale));

        CAPTURE(to_text_double(got.x0));
        CAPTURE(to_text_double(got.x1));
        CAPTURE(to_text_double(got.x2));
        CAPTURE(to_text_double(got.x3));

        CAPTURE(to_text_double_hex(got.x0));
        CAPTURE(to_text_double_hex(got.x1));
        CAPTURE(to_text_double_hex(got.x2));
        CAPTURE(to_text_double_hex(got.x3));

        record_accuracy_sample(label, got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }

    template<typename Cases>
    void run_fused_expression_precision_suite(const Cases& cases)
    {
        std::size_t case_index = 0;
        for (const auto& spec : cases)
        {
            check_fused_expression_case(
                "fused add/add leaf chain",
                spec,
                case_index,
                { 2, 0, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a + b) + c;
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a + b) + c;
                });

            check_fused_expression_case(
                "fused scaled add leaf chain",
                spec,
                case_index,
                { 2, 0, 0, 0 },
                [](const auto& a, const auto&, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a + a) + c;
                },
                [](const auto& a, const auto&, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a + a) + c;
                });

            check_fused_expression_case(
                "fused sub/add leaf chain",
                spec,
                case_index,
                { 1, 1, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a - b) + c;
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a - b) + c;
                });

            check_fused_expression_case(
                "fused value plus leaf difference",
                spec,
                case_index,
                { 1, 1, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return a + (b - c);
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return a + (b - c);
                });

            check_fused_expression_case(
                "fused add/sub leaf chain",
                spec,
                case_index,
                { 1, 1, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a + b) - c;
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a + b) - c;
                });

            check_fused_expression_case(
                "fused value minus leaf sum",
                spec,
                case_index,
                { 1, 1, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return a - (b + c);
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return a - (b + c);
                });

            check_fused_expression_case(
                "fused value minus leaf difference",
                spec,
                case_index,
                { 1, 1, 0, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return a - (b - c);
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return a - (b - c);
                });

            check_fused_expression_case(
                "fused product plus value",
                spec,
                case_index,
                { 1, 0, 1, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a * b) + c;
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a * b) + c;
                });

            check_fused_expression_case(
                "fused value plus product",
                spec,
                case_index,
                { 1, 0, 1, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return a + (b * c);
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return a + (b * c);
                });

            check_fused_expression_case(
                "fused product minus value",
                spec,
                case_index,
                { 0, 1, 1, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return (a * b) - c;
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return (a * b) - c;
                });

            check_fused_expression_case(
                "fused value minus product",
                spec,
                case_index,
                { 0, 1, 1, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, auto, auto, auto) {
                    return a - (b * c);
                },
                [](const auto& a, const auto& b, const auto& c, const auto&, const auto&, const auto&, const auto&, const auto&) {
                    return a - (b * c);
                });

            check_fused_expression_case(
                "fused product sum",
                spec,
                case_index,
                { 1, 0, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto&, auto, auto, auto) {
                    return (a * b) + (c * d);
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto&, const auto&, const auto&, const auto&) {
                    return (a * b) + (c * d);
                });

            check_fused_expression_case(
                "fused product difference",
                spec,
                case_index,
                { 0, 1, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto&, auto, auto, auto) {
                    return (a * b) - (c * d);
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto&, const auto&, const auto&, const auto&) {
                    return (a * b) - (c * d);
                });

            check_fused_expression_case(
                "fused product sum plus value",
                spec,
                case_index,
                { 2, 0, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto) {
                    return ((a * b) + (c * d)) + e;
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto&) {
                    return ((a * b) + (c * d)) + e;
                });

            check_fused_expression_case(
                "fused product sum minus value",
                spec,
                case_index,
                { 1, 1, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto) {
                    return ((a * b) + (c * d)) - e;
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto&) {
                    return ((a * b) + (c * d)) - e;
                });

            check_fused_expression_case(
                "fused product difference plus value",
                spec,
                case_index,
                { 1, 1, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto) {
                    return ((a * b) - (c * d)) + e;
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto&) {
                    return ((a * b) - (c * d)) + e;
                });

            check_fused_expression_case(
                "fused product difference minus value",
                spec,
                case_index,
                { 0, 2, 2, 0 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto) {
                    return ((a * b) - (c * d)) - e;
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto&) {
                    return ((a * b) - (c * d)) - e;
                });

            check_fused_expression_case(
                "fused scalar product plus value",
                spec,
                case_index,
                { 1, 0, 1, 0 },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, auto scalar, auto, auto) {
                    return (a * scalar) + b;
                },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, const auto& scalar, const auto&, const auto&) {
                    return (a * scalar) + b;
                });

            check_fused_expression_case(
                "fused value plus scalar product",
                spec,
                case_index,
                { 1, 0, 1, 0 },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, auto scalar, auto, auto) {
                    return a + (b * scalar);
                },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, const auto& scalar, const auto&, const auto&) {
                    return a + (b * scalar);
                });

            check_fused_expression_case(
                "fused scalar product minus value",
                spec,
                case_index,
                { 0, 1, 1, 0 },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, auto scalar, auto, auto) {
                    return (a * scalar) - b;
                },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, const auto& scalar, const auto&, const auto&) {
                    return (a * scalar) - b;
                });

            check_fused_expression_case(
                "fused value minus scalar product",
                spec,
                case_index,
                { 0, 1, 1, 0 },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, auto scalar, auto, auto) {
                    return a - (b * scalar);
                },
                [](const auto& a, const auto& b, const auto&, const auto&, const auto&, const auto& scalar, const auto&, const auto&) {
                    return a - (b * scalar);
                });

            check_fused_expression_case(
                "fused product sum over add-double denominator",
                spec,
                case_index,
                { 2, 0, 2, 1 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto denom_plus, auto) {
                    return ((a * b) + (c * d)) / (e + denom_plus);
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto& denom_plus, const auto&) {
                    return ((a * b) + (c * d)) / (e + denom_plus);
                });

            check_fused_expression_case(
                "fused product difference over double-sub denominator",
                spec,
                case_index,
                { 0, 2, 2, 1 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto denom_minus) {
                    return ((a * b) - (c * d)) / (denom_minus - e);
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto& denom_minus) {
                    return ((a * b) - (c * d)) / (denom_minus - e);
                });

            check_fused_expression_case(
                "fused product sum over product-sum denominator",
                spec,
                case_index,
                { 2, 0, 4, 1 },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, auto, auto, auto) {
                    return (((a * b) + (c * d)) + e) / ((c * c) + (d * d));
                },
                [](const auto& a, const auto& b, const auto& c, const auto& d, const auto& e, const auto&, const auto&, const auto&) {
                    return (((a * b) + (c * d)) + e) / ((c * c) + (d * d));
                });

            ++case_index;
        }
    }

    [[nodiscard]] std::string fused_expression_random_text(const mpfr_ref& value)
    {
        return to_scientific_text(value, printed_digits + 6);
    }

    [[nodiscard]] fused_expression_case random_fused_expression_case(std::mt19937_64& rng)
    {
        constexpr std::array<double, 12> scalars{{
            0.5, -0.75, 1.125, -1.375, 1.625, -1.875,
            0.3125, -0.4375, 2.25, -2.5, 0.0625, -0.09375
        }};
        constexpr std::array<double, 8> add_denominators{{
            1.5, 1.875, 2.25, 2.625, 3.0, 3.375, 3.75, 4.125
        }};
        constexpr std::array<double, 8> sub_denominators{{
            2.0, 2.375, 2.75, 3.125, 3.5, 3.875, 4.25, 4.625
        }};

        std::uniform_int_distribution<std::size_t> scalar_dist(0, scalars.size() - 1);
        std::uniform_int_distribution<std::size_t> denominator_dist(0, add_denominators.size() - 1);

        return {
            fused_expression_random_text(random_signed_interval_for_f256(rng, mpfr_ref{ 2.0 })),
            fused_expression_random_text(random_signed_interval_for_f256(rng, mpfr_ref{ 2.0 })),
            fused_expression_random_text(random_signed_interval_away_from_zero_for_f256(rng, mpfr_ref{ "0.25" }, mpfr_ref{ "1.75" })),
            fused_expression_random_text(random_signed_interval_away_from_zero_for_f256(rng, mpfr_ref{ "0.25" }, mpfr_ref{ "1.75" })),
            fused_expression_random_text(random_signed_interval_for_f256(rng, mpfr_ref{ "0.75" })),
            scalars[scalar_dist(rng)],
            add_denominators[denominator_dist(rng)],
            sub_denominators[denominator_dist(rng)]
        };
    }

}

TEST_CASE("f256 matches MPFR for + - * /", "[fltx][f256][precision][arithmetic]")
{
    accuracy_report_scope report_scope{ "f256 matches MPFR for + - * /" };
    const std::array<std::pair<const char*, const char*>, 10> cases = {{
        { "1", "2" },
        { "1.25", "2.5" },
        { "-3.75", "2.125" },
        { "1.0000000000000000000000000000000001", "2.0000000000000000000000000000000002" },
        { "123456789012345678901234567890.125", "0.000000000000000000000000000000125" },
        { "3.1415926535897932384626433832795028841971", "2.7182818284590452353602874713526624977572" },
        { "1e-50", "1e-20" },
        { "1e50", "1e-10" },
        { "-8.333333333333333333333333333333333", "0.125" },
        { "0.3333333333333333333333333333333333", "7.0000000000000000000000000000000001" }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("add", lhs, rhs,
            [](const f256& a, const f256& b) { return a + b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a + b; });

        check_binary_op("subtract", lhs, rhs,
            [](const f256& a, const f256& b) { return a - b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a - b; });

        check_binary_op("multiply", lhs, rhs,
            [](const f256& a, const f256& b) { return a * b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a * b; });

        check_binary_op("divide", lhs, rhs,
            [](const f256& a, const f256& b) { return a / b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a / b; });
    }
}

TEST_CASE("f256 brute-force random arithmetic matches MPFR within tolerance", "[fltx][f256][precision][arithmetic]")
{
    accuracy_report_scope report_scope{ "f256 brute-force random arithmetic matches MPFR within tolerance" };
    std::mt19937_64 rng{ random_seed };

    const int digits = printed_digits;
    const int count = 1000;
    print_random_run("random arithmetic cases", count);

    for (int i = 0; i < count; ++i)
    {
        const big lhs_big = random_finite_for_f256(rng);
        const big rhs_big = random_finite_for_f256(rng);

        const std::string lhs_text = to_scientific_string(lhs_big, digits);
        const std::string rhs_text = to_scientific_string(rhs_big, digits);

        INFO("iteration: " << i);
        INFO("lhs_text: " << lhs_text);
        INFO("rhs_text: " << rhs_text);

        check_binary_op("add", lhs_text.c_str(), rhs_text.c_str(),
            [](const f256& a, const f256& b) { return a + b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a + b; });

        check_binary_op("subtract", lhs_text.c_str(), rhs_text.c_str(),
            [](const f256& a, const f256& b) { return a - b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a - b; });

        check_binary_op("multiply", lhs_text.c_str(), rhs_text.c_str(),
            [](const f256& a, const f256& b) { return a * b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a * b; });

        if (rhs_big != 0)
        {
            check_binary_op("divide", lhs_text.c_str(), rhs_text.c_str(),
                [](const f256& a, const f256& b) { return a / b; },
                [](const mpfr_ref& a, const mpfr_ref& b) { return a / b; });
        }
    }
}

TEST_CASE("f256 mixed scalar arithmetic matches MPFR within tolerance", "[fltx][f256][precision][arithmetic][scalar]")
{
    accuracy_report_scope report_scope{ "f256 mixed scalar arithmetic matches MPFR within tolerance" };

    constexpr std::array<const char*, 6> values{{
        "3.1415926535897932384626433832795028841971",
        "-2.7182818284590452353602874713526624977572",
        "1.0000000000000000000000000000000001",
        "-123456789012345678901234567890.125",
        "1e-50",
        "1e50"
    }};

    constexpr std::array<double, 6> double_scalars{{
        0.5,
        -1.5,
        3.141592653589793,
        -0.125,
        1.0e-20,
        -1.0e20
    }};

    constexpr std::array<float, 6> float_scalars{{
        0.5f,
        -1.25f,
        3.75f,
        -0.125f,
        1.0e-10f,
        -1.0e10f
    }};

    constexpr std::array<f128_s, 6> f128_scalars{{
        { 0.5, 0.0 },
        { -1.5, 0.0 },
        { 3.141592653589793, 1.2246467991473532e-16 },
        { -0.125, 0.0 },
        { 1.0e-20, -5.4846728545790429e-37 },
        { -1.0e20, 0.0 }
    }};

    auto check_all_scalar_orders = [](const char* scalar_kind, auto scalar, const char* value_text)
    {
        check_scalar_binary_op("add", "f256 + scalar", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return a + b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a + b; });

        check_scalar_binary_op("add", "scalar + f256", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return b + a; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return b + a; });

        check_scalar_binary_op("subtract", "f256 - scalar", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return a - b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a - b; });

        check_scalar_binary_op("subtract", "scalar - f256", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return b - a; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return b - a; });

        check_scalar_binary_op("multiply", "f256 * scalar", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return a * b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a * b; });

        check_scalar_binary_op("multiply", "scalar * f256", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return b * a; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return b * a; });

        check_scalar_binary_op("divide", "f256 / scalar", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return a / b; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return a / b; });

        check_scalar_binary_op("divide", "scalar / f256", scalar_kind, value_text, scalar,
            [](const f256& a, auto b) { return b / a; },
            [](const mpfr_ref& a, const mpfr_ref& b) { return b / a; });
    };

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        check_all_scalar_orders("double", double_scalars[i], values[i]);
        check_all_scalar_orders("float", float_scalars[i], values[i]);
        check_all_scalar_orders("f128", f128_scalars[i], values[i]);
    }
}

TEST_CASE("f256 integer overloads preserve exact integer values", "[fltx][f256][precision][arithmetic][scalar][integer]")
{
    auto check_signed = [](auto rhs, const char* label)
    {
        const f256 base = to_f256("1.2345678901234567890123456789012345678901234567890123456789");
        const f256 rhs_value = to_f256(static_cast<std::int64_t>(rhs));
        const bool rhs_fits_double = detail::_f256::integer_fits_exact_double(rhs);
        const double rhs_double = static_cast<double>(rhs);

        auto add_right = [&]() -> f256 { return rhs_fits_double ? f256{ base + rhs_double } : f256{ base + rhs_value }; };
        auto add_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double + base } : f256{ rhs_value + base }; };
        auto sub_right = [&]() -> f256 { return rhs_fits_double ? f256{ base - rhs_double } : f256{ base - rhs_value }; };
        auto sub_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double - base } : f256{ rhs_value - base }; };
        auto mul_right = [&]() -> f256 { return rhs_fits_double ? f256{ base * rhs_double } : f256{ base * rhs_value }; };
        auto mul_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double * base } : f256{ rhs_value * base }; };
        auto div_right = [&]() -> f256 { return rhs_fits_double ? f256{ base / rhs_double } : f256{ base / rhs_value }; };
        auto div_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double / base } : f256{ rhs_value / base }; };

        f256 got = base;
        got += rhs;
        require_exact_value(label, got, add_right());

        got = base;
        got -= rhs;
        require_exact_value(label, got, sub_right());

        got = base;
        got *= rhs;
        require_exact_value(label, got, mul_right());

        got = base;
        got /= rhs;
        require_exact_value(label, got, div_right());

        require_exact_value(label, base + rhs, add_right());
        require_exact_value(label, rhs + base, add_left());
        require_exact_value(label, base - rhs, sub_right());
        require_exact_value(label, rhs - base, sub_left());
        require_exact_value(label, base * rhs, mul_right());
        require_exact_value(label, rhs * base, mul_left());
        require_exact_value(label, base / rhs, div_right());
        require_exact_value(label, rhs / base, div_left());
    };

    auto check_unsigned = [](auto rhs, const char* label)
    {
        const f256 base = to_f256("1.2345678901234567890123456789012345678901234567890123456789");
        const f256 rhs_value = to_f256(static_cast<std::uint64_t>(rhs));
        const bool rhs_fits_double = detail::_f256::integer_fits_exact_double(rhs);
        const double rhs_double = static_cast<double>(rhs);

        auto add_right = [&]() -> f256 { return rhs_fits_double ? f256{ base + rhs_double } : f256{ base + rhs_value }; };
        auto add_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double + base } : f256{ rhs_value + base }; };
        auto sub_right = [&]() -> f256 { return rhs_fits_double ? f256{ base - rhs_double } : f256{ base - rhs_value }; };
        auto sub_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double - base } : f256{ rhs_value - base }; };
        auto mul_right = [&]() -> f256 { return rhs_fits_double ? f256{ base * rhs_double } : f256{ base * rhs_value }; };
        auto mul_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double * base } : f256{ rhs_value * base }; };
        auto div_right = [&]() -> f256 { return rhs_fits_double ? f256{ base / rhs_double } : f256{ base / rhs_value }; };
        auto div_left  = [&]() -> f256 { return rhs_fits_double ? f256{ rhs_double / base } : f256{ rhs_value / base }; };

        f256 got = base;
        got += rhs;
        require_exact_value(label, got, add_right());

        got = base;
        got -= rhs;
        require_exact_value(label, got, sub_right());

        got = base;
        got *= rhs;
        require_exact_value(label, got, mul_right());

        got = base;
        got /= rhs;
        require_exact_value(label, got, div_right());

        require_exact_value(label, base + rhs, add_right());
        require_exact_value(label, rhs + base, add_left());
        require_exact_value(label, base - rhs, sub_right());
        require_exact_value(label, rhs - base, sub_left());
        require_exact_value(label, base * rhs, mul_right());
        require_exact_value(label, rhs * base, mul_left());
        require_exact_value(label, base / rhs, div_right());
        require_exact_value(label, rhs / base, div_left());
    };

    check_signed(std::int8_t{ -7 }, "int8");
    check_unsigned(std::uint8_t{ 7 }, "uint8");
    check_signed(std::int16_t{ -257 }, "int16");
    check_unsigned(std::uint16_t{ 257 }, "uint16");
    check_signed(std::int32_t{ -65537 }, "int32");
    check_unsigned(std::uint32_t{ 65537 }, "uint32");
    check_signed(static_cast<std::int64_t>(-9007199254740993ll), "int64");
    check_unsigned(std::uint64_t{ 9007199254740993ull }, "uint64");
}

TEST_CASE("f256 scalar mixed recurrence stays within MPFR tolerance", "[fltx][f256][precision][arithmetic][scalar][mixed]")
{
    accuracy_report_scope report_scope{ "f256 scalar mixed recurrence stays within MPFR tolerance" };

    constexpr std::array<scalar_mixed_recurrence_case, 8> cases{{
        { "-0.75", "0.125", "0.875", "-0.375", "1.25", "0.625" },
        { "0.5", "-0.25", "-0.625", "0.75", "0.875", "1.125" },
        { "-0.3437500000000000000000000000000001", "0.8437500000000000000000000000000001", "0.4687500000000000000000000000000001", "0.6562500000000000000000000000000001", "1.5625000000000000000000000000000001", "0.9062500000000000000000000000000001" },
        { "0.1562500000000000000000000000000001", "-0.7187500000000000000000000000000001", "-0.8437500000000000000000000000000001", "0.3437500000000000000000000000000001", "0.7187500000000000000000000000000001", "1.4062500000000000000000000000000001" },
        { "-1.3333333333333333333333333333333333", "0.4142135623730950488016887242096981", "0.7071067811865475244008443621048490", "-0.5773502691896257645091487805019575", "1.6180339887498948482045868343656381", "0.6180339887498948482045868343656381" },
        { "0.5773502691896257645091487805019575", "-1.2247448713915890490986420373529457", "-0.4142135623730950488016887242096981", "0.8660254037844386467637231707529362", "0.7071067811865475244008443621048490", "1.4142135623730950488016887242096981" },
        { "0.8660254037844386467637231707529362", "-1.1180339887498948482045868343656381", "0.7639320225002103035908263312687238", "0.2679491924311227064725536584941276", "1.5", "0.5773502691896257645091487805019575" },
        { "-0.2679491924311227064725536584941276", "1.2247448713915890490986420373529457", "-0.6180339887498948482045868343656381", "-0.7639320225002103035908263312687238", "0.6666666666666666666666666666666667", "1.5" }
    }};

    constexpr int rounds = 32;
    const auto got = run_scalar_mixed_recurrence_precision<f256>(cases, rounds);
    const auto expected = run_scalar_mixed_recurrence_precision<mpfr_ref>(cases, rounds);

    check_scalar_mixed_recurrence_component("x accumulator", got.first, expected.first, rounds, cases.size());
    check_scalar_mixed_recurrence_component("y accumulator", got.second, expected.second, rounds, cases.size());
}

TEST_CASE("f256 fused expression evaluator matches MPFR for routed shapes", "[fltx][f256][precision][arithmetic][fused]")
{
    accuracy_report_scope report_scope{ "f256 fused expression evaluator matches MPFR for routed shapes" };

    const std::array<fused_expression_case, 10> cases{{
        {
            "1.0000000000000000000000000000000001",
            "-0.9999999999999999999999999999999997",
            "0.3333333333333333333333333333333333",
            "-0.2500000000000000000000000000000001",
            "0.1250000000000000000000000000000003",
            0.5,
            2.0,
            3.0
        },
        {
            "-1.4142135623730950488016887242096981",
            "0.7071067811865475244008443621048490",
            "-0.5773502691896257645091487805019575",
            "1.7320508075688772935274463415058724",
            "-0.2187500000000000000000000000000001",
            -0.75,
            2.25,
            3.25
        },
        {
            "3.1415926535897932384626433832795029",
            "-2.7182818284590452353602874713526625",
            "1.6180339887498948482045868343656381",
            "-0.6180339887498948482045868343656381",
            "0.4142135623730950488016887242096981",
            1.125,
            3.0,
            4.0
        },
        {
            "-0.00000000000000000000000000000125",
            "0.00000000000000000000000000000250",
            "-1.0000000000000000000000000000000001",
            "0.9999999999999999999999999999999999",
            "-0.3333333333333333333333333333333333",
            -1.375,
            1.5,
            2.5
        },
        {
            "1099511627776.0000000000000000000000000001",
            "-0.0000000000009094947017729282379150390625",
            "0.8750000000000000000000000000000001",
            "-1.1250000000000000000000000000000001",
            "0.0625000000000000000000000000000001",
            0.3125,
            2.625,
            3.625
        },
        {
            "-4503599627370496.000000000000000000000000001",
            "0.0000000000000002220446049250313080847263336181640625",
            "1.2500000000000000000000000000000001",
            "0.7500000000000000000000000000000001",
            "-0.1562500000000000000000000000000001",
            -0.4375,
            3.375,
            4.375
        },
        {
            "0.875",
            "-1.125",
            "1.375",
            "-1.625",
            "0.00000000000000000000000000000000000000000000000001",
            2.25,
            4.125,
            5.125
        },
        {
            "-0.8750000000000000000000000000000001",
            "1.1250000000000000000000000000000001",
            "-1.3750000000000000000000000000000001",
            "1.6250000000000000000000000000000001",
            "-0.00000000000000000000000000000000000000000000000001",
            -2.5,
            1.875,
            2.875
        },
        {
            "0.1562500000000000000000000000000001",
            "-0.8437500000000000000000000000000001",
            "1.5625000000000000000000000000000001",
            "0.9062500000000000000000000000000001",
            "-0.7187500000000000000000000000000001",
            0.0625,
            2.75,
            3.75
        },
        {
            "-1.3333333333333333333333333333333333",
            "0.4142135623730950488016887242096981",
            "0.7071067811865475244008443621048490",
            "-0.5773502691896257645091487805019575",
            "0.8660254037844386467637231707529362",
            -0.09375,
            3.5,
            4.5
        }
    }};

    run_fused_expression_precision_suite(cases);
}

TEST_CASE("f256 fused expression evaluator matches MPFR on random routed shapes", "[fltx][f256][precision][arithmetic][fused][random]")
{
    accuracy_report_scope report_scope{ "f256 fused expression evaluator matches MPFR on random routed shapes" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 48;
    print_random_run("random fused expression cases", count);

    std::vector<fused_expression_case> cases;
    cases.reserve(count);
    for (int i = 0; i < count; ++i)
        cases.push_back(random_fused_expression_case(rng));

    run_fused_expression_precision_suite(cases);
}

TEST_CASE("f256 sin matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][sin]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR for fixed values" };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduced_abs_tolerance{ "1e-67" };
    const mpfr_ref reduced_rel_tolerance{ "5e-61" };
    const mpfr_ref reduction_abs_tolerance{ "1e-59" };
    const mpfr_ref reduction_rel_tolerance{ "1e-59" };

    check_sin_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_over_2", half_pi, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f256 sin matches MPFR for fixed argument-reduction stress values", "[fltx][f256][precision][transcendental][trig][sin][reduction]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR for fixed argument-reduction stress values", false };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduction_abs_tolerance{ "1e-59" };
    const mpfr_ref reduction_rel_tolerance{ "1e-59" };
    const mpfr_ref tiny_offset_abs_tolerance{ "2e-65" };

    check_sin_case("pi_minus_tiny", pi - tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_plus_tiny", pi + tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("two_pi_minus_tiny", two_pi - tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("two_pi_plus_tiny", two_pi + tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("large_decimal", mpfr_ref{ "1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("negative_large_decimal", mpfr_ref{ "-1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("million_pi_plus_offset", mpfr_ref{ "1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("negative_million_pi_plus_offset", mpfr_ref{ "-1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f256 sin matches MPFR on random reduced-range inputs", "[fltx][f256][precision][transcendental][trig][sin]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR on random reduced-range inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-67" };
    const mpfr_ref rel_tolerance{ "5e-61" };
    print_random_run("random reduced-range sin cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_kernel_argument_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });
    }
}

TEST_CASE("f256 sin matches MPFR on random range-reduced inputs away from zero-crossings", "[fltx][f256][precision][transcendental][trig][sin][reduction]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR on random range-reduced inputs away from zero-crossings" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    const mpfr_ref zero_crossing_threshold{ "0.125" };
    print_random_run("random range-reduced sin cases away from zero-crossings", count);

    for (int i = 0, attempts = 0; i < count; ++attempts)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
        const mpfr_ref expected = boost::multiprecision::sin(input);
        if (abs_ref(expected) < zero_crossing_threshold)
            continue;

        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("attempt: " << attempts);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });

        ++i;
    }
}

TEST_CASE("f256 sin matches MPFR on random range-reduced zero-crossing stress inputs", "[fltx][f256][precision][transcendental][trig][sin][reduction][zero-crossing]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR on random range-reduced zero-crossing stress inputs", false };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 64 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    const mpfr_ref zero_crossing_threshold{ "0.125" };
    print_random_run("random range-reduced sin zero-crossing stress cases", count);

    for (int i = 0, attempts = 0; i < count; ++attempts)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
        const mpfr_ref expected = boost::multiprecision::sin(input);
        if (abs_ref(expected) >= zero_crossing_threshold)
            continue;

        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("attempt: " << attempts);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sin(value); });

        ++i;
    }
}

TEST_CASE("f256 cos matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][cos]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR for fixed values" };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduced_abs_tolerance{ "1e-67" };
    const mpfr_ref reduced_rel_tolerance{ "5e-61" };
    const mpfr_ref reduction_abs_tolerance{ "1e-59" };
    const mpfr_ref reduction_rel_tolerance{ "1e-59" };

    check_cos_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f256 cos matches MPFR for fixed argument-reduction stress values", "[fltx][f256][precision][transcendental][trig][cos][reduction]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR for fixed argument-reduction stress values", false };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduction_abs_tolerance{ "1e-59" };
    const mpfr_ref reduction_rel_tolerance{ "1e-59" };
    check_cos_case("pi_over_2", half_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    const mpfr_ref tiny_offset_abs_tolerance{ "2e-65" };

    check_cos_case("pi_minus_tiny", pi - tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("pi_plus_tiny", pi + tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("two_pi_minus_tiny", two_pi - tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("two_pi_plus_tiny", two_pi + tiny, tiny_offset_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("large_decimal", mpfr_ref{ "1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("negative_large_decimal", mpfr_ref{ "-1000000.25" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("million_pi_plus_offset", mpfr_ref{ "1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
    check_cos_case("negative_million_pi_plus_offset", mpfr_ref{ "-1000000" } * pi + mpfr_ref{ "0.125" }, reduction_abs_tolerance, reduction_rel_tolerance);
}

TEST_CASE("f256 cos matches MPFR on random reduced-range inputs", "[fltx][f256][precision][transcendental][trig][cos]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR on random reduced-range inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-67" };
    const mpfr_ref rel_tolerance{ "5e-61" };
    print_random_run("random reduced-range cos cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_kernel_argument_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });
    }
}

TEST_CASE("f256 cos matches MPFR on random range-reduced inputs away from zero-crossings", "[fltx][f256][precision][transcendental][trig][cos][reduction]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR on random range-reduced inputs away from zero-crossings" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    const mpfr_ref zero_crossing_threshold{ "0.125" };
    print_random_run("random range-reduced cos cases away from zero-crossings", count);

    for (int i = 0, attempts = 0; i < count; ++attempts)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
        const mpfr_ref expected = boost::multiprecision::cos(input);
        if (abs_ref(expected) < zero_crossing_threshold)
            continue;

        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("attempt: " << attempts);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });

        ++i;
    }
}

TEST_CASE("f256 cos matches MPFR on random range-reduced zero-crossing stress inputs", "[fltx][f256][precision][transcendental][trig][cos][reduction][zero-crossing]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR on random range-reduced zero-crossing stress inputs", false };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 64 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    const mpfr_ref zero_crossing_threshold{ "0.125" };
    print_random_run("random range-reduced cos zero-crossing stress cases", count);

    for (int i = 0, attempts = 0; i < count; ++attempts)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
        const mpfr_ref expected = boost::multiprecision::cos(input);
        if (abs_ref(expected) >= zero_crossing_threshold)
            continue;

        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("attempt: " << attempts);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "cos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cos(value); });

        ++i;
    }
}

TEST_CASE("f256 floor ceil trunc and round match MPFR for fixed values", "[fltx][f256][precision][math][rounding]")
{
    accuracy_report_scope report_scope{ "f256 floor ceil trunc and round match MPFR for fixed values" };
    const std::array<const char*, 16> cases = {{
        "0",
        "-0",
        "0.25",
        "-0.25",
        "0.5",
        "-0.5",
        "1.5",
        "-1.5",
        "2.5",
        "-2.5",
        "0.999999999999999999999999999999999999999999999999999999999999",
        "1.000000000000000000000000000000000000000000000000000000000001",
        "-0.999999999999999999999999999999999999999999999999999999999999",
        "-1.000000000000000000000000000000000000000000000000000000000001",
        "123456789012345678901234567890.000000000000000000000000000000000000000000000000000000000001",
        "-123456789012345678901234567890.000000000000000000000000000000000000000000000000000000000001"
    }};

    for (const char* input : cases)
    {
        check_unary_op("floor", input,
            [](const f256& value) { return bl::floor(value); },
            [](const mpfr_ref& value) { return ref_floor(value); });

        check_unary_op("ceil", input,
            [](const f256& value) { return bl::ceil(value); },
            [](const mpfr_ref& value) { return ref_ceil(value); });

        check_unary_op("trunc", input,
            [](const f256& value) { return bl::trunc(value); },
            [](const mpfr_ref& value) { return ref_trunc(value); });

        check_unary_op("round", input,
            [](const f256& value) { return bl::round(value); },
            [](const mpfr_ref& value) { return ref_round_half_away_zero(value); });
    }
}

TEST_CASE("f256 floor ceil trunc and round match MPFR for large-limb regression cases", "[fltx][f256][precision][math][rounding]")
{
    accuracy_report_scope report_scope{ "f256 floor ceil trunc and round match MPFR for large-limb regression cases" };
    check_unary_op(
        "floor",
        "4.6958550912494028428400315673292717414e+19",
        [](const f256& value) { return bl::floor(value); },
        [](const mpfr_ref& value) { return boost::multiprecision::floor(value); });

    check_unary_op(
        "trunc",
        "-1.5848854675958108400285213569604012722e+23",
        [](const f256& value) { return bl::trunc(value); },
        [](const mpfr_ref& value) { return boost::multiprecision::trunc(value); });
}

TEST_CASE("f256 floor ceil trunc and round match MPFR on random finite inputs", "[fltx][f256][precision][math][rounding]")
{
    accuracy_report_scope report_scope{ "f256 floor ceil trunc and round match MPFR on random finite inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    print_random_run("random floor/ceil/trunc/round cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = mpfr_ref{ random_finite_for_f256(rng) };
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op("floor", input_text.c_str(),
            [](const f256& value) { return bl::floor(value); },
            [](const mpfr_ref& value) { return ref_floor(value); });

        check_unary_op("ceil", input_text.c_str(),
            [](const f256& value) { return bl::ceil(value); },
            [](const mpfr_ref& value) { return ref_ceil(value); });

        check_unary_op("trunc", input_text.c_str(),
            [](const f256& value) { return bl::trunc(value); },
            [](const mpfr_ref& value) { return ref_trunc(value); });

        check_unary_op("round", input_text.c_str(),
            [](const f256& value) { return bl::round(value); },
            [](const mpfr_ref& value) { return ref_round_half_away_zero(value); });
    }
}

TEST_CASE("f256 fmod matches MPFR for fixed values", "[fltx][f256][precision][math][fmod]")
{
    accuracy_report_scope report_scope{ "f256 fmod matches MPFR for fixed values" };
    const std::array<std::pair<const char*, const char*>, 10> cases = {{
        { "5.25", "2" },
        { "-5.25", "2" },
        { "5.25", "-2" },
        { "-5.25", "-2" },
        { "1.000000000000000000000000000000000000000000000000000000000001", "0.1" },
        { "-1.000000000000000000000000000000000000000000000000000000000001", "0.1" },
        { "123456789.125", "0.5" },
        { "-123456789.125", "0.5" },
        { "1e-40", "3e-41" },
        { "-1e40", "3.125" }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("fmod", lhs, rhs,
            [](const f256& x, const f256& y) { return bl::fmod(x, y); },
            [](const mpfr_ref& x, const mpfr_ref& y) { return ref_fmod(x, y); });
    }
}

TEST_CASE("f256 fmod matches MPFR for huge-quotient regression cases", "[fltx][f256][precision][math][fmod]")
{
    accuracy_report_scope report_scope{ "f256 fmod matches MPFR for huge-quotient regression cases" };
    check_binary_op(
        "fmod",
        "4.6958550912494028428400315673292717414e+19",
        "2.9410562077176174010123838366180003482e+02",
        [](const f256& lhs, const f256& rhs) { return bl::fmod(lhs, rhs); },
        [](const mpfr_ref& lhs, const mpfr_ref& rhs) { return boost::multiprecision::fmod(lhs, rhs); });
}

TEST_CASE("f256 fmod matches MPFR on random finite inputs", "[fltx][f256][precision][math][fmod]")
{
    accuracy_report_scope report_scope{ "f256 fmod matches MPFR on random finite inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 10000;
    print_random_run("random fmod cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref lhs = mpfr_ref{ random_finite_for_f256(rng) };
        const mpfr_ref rhs = random_nonzero_for_f256(rng);

        const std::string lhs_text = to_scientific_text(lhs, printed_digits + 6);
        const std::string rhs_text = to_scientific_text(rhs, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("lhs_text: " << lhs_text);
        INFO("rhs_text: " << rhs_text);

        check_binary_op_with_tolerance(
            "fmod",
            lhs_text.c_str(),
            rhs_text.c_str(),
            mpfr_ref{ "1e-54" },
            mpfr_ref{ "1e-54" },
            [](const f256& x, const f256& y) { return bl::fmod(x, y); },
            [](const mpfr_ref& x, const mpfr_ref& y) { return ref_fmod(x, y); });
    }
}

TEST_CASE("f256 sqrt matches MPFR for fixed values", "[fltx][f256][precision][math][sqrt]")
{
    accuracy_report_scope report_scope{ "f256 sqrt matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0",
        "1",
        "2",
        "4",
        "1e-60",
        "1e60",
        "0.125",
        "123456789.125"
    }};

    for (const char* input : cases)
    {
        check_unary_op("sqrt", input,
            [](const f256& value) { return bl::sqrt(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sqrt(value); });
    }
}

TEST_CASE("f256 sqrt matches MPFR on random positive inputs", "[fltx][f256][precision][math][sqrt]")
{
    accuracy_report_scope report_scope{ "f256 sqrt matches MPFR on random positive inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    print_random_run("random sqrt cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_positive_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op("sqrt", input_text.c_str(),
            [](const f256& value) { return bl::sqrt(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sqrt(value); });
    }
}

TEST_CASE("f256 ldexp matches MPFR for fixed values", "[fltx][f256][precision][math][ldexp]")
{
    accuracy_report_scope report_scope{ "f256 ldexp matches MPFR for fixed values" };
    const std::array<std::pair<const char*, int>, 10> cases = {{
        { "0", 0 },
        { "1", 0 },
        { "1", 1 },
        { "1", -1 },
        { "1.5", 10 },
        { "-1.5", 10 },
        { "3.1415926535897932384626433832795028841971", -20 },
        { "1e-40", 120 },
        { "1e40", -120 },
        { "123456789.125", 37 }
    }};

    for (const auto& [input, exponent] : cases)
        check_ldexp_case("fixed", mpfr_ref{ input }, exponent);
}

TEST_CASE("f256 ldexp matches MPFR on random finite inputs", "[fltx][f256][precision][math][ldexp]")
{
    accuracy_report_scope report_scope{ "f256 ldexp matches MPFR on random finite inputs" };
    std::mt19937_64 rng{ random_seed };
    std::uniform_int_distribution<int> exponent_dist(-180, 180);

    constexpr int count = 128 * random_sample_count_scale;
    print_random_run("random ldexp cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = mpfr_ref{ random_finite_for_f256(rng) };
        const int exponent = exponent_dist(rng);

        INFO("iteration: " << i);
        check_ldexp_case("random", input, exponent);
    }
}

TEST_CASE("f256 exp matches MPFR for fixed values", "[fltx][f256][precision][transcendental][exp]")
{
    accuracy_report_scope report_scope{ "f256 exp matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "-10",
        "-1",
        "-0.125",
        "0",
        "0.125",
        "1",
        "10",
        "20"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "exp",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::exp(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::exp(value); });
    }
}

TEST_CASE("f256 exp matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][exp]")
{
    accuracy_report_scope report_scope{ "f256 exp matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random exp cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 20 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "exp",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::exp(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::exp(value); });
    }
}

TEST_CASE("f256 exp2 matches MPFR for fixed values", "[fltx][f256][precision][transcendental][exp2]")
{
    accuracy_report_scope report_scope{ "f256 exp2 matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "-10",
        "-1",
        "-0.125",
        "0",
        "0.125",
        "1",
        "10",
        "20"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "exp2",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::exp2(value); },
            [](const mpfr_ref& value) { return ref_exp2(value); });
    }
}

TEST_CASE("f256 exp2 matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][exp2]")
{
    accuracy_report_scope report_scope{ "f256 exp2 matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random exp2 cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 20 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "exp2",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::exp2(value); },
            [](const mpfr_ref& value) { return ref_exp2(value); });
    }
}

TEST_CASE("f256 log matches MPFR for fixed values", "[fltx][f256][precision][transcendental][log]")
{
    accuracy_report_scope report_scope{ "f256 log matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0.125",
        "0.5",
        "0.999999999999999999999999999999999999999999999999999999999999",
        "1",
        "1.000000000000000000000000000000000000000000000000000000000001",
        "2",
        "10",
        "123456789.125"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "log",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::log(value); });
    }
}

TEST_CASE("f256 log matches MPFR on random positive inputs", "[fltx][f256][precision][transcendental][log]")
{
    accuracy_report_scope report_scope{ "f256 log matches MPFR on random positive inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random log cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_positive_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "log",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::log(value); });
    }
}

TEST_CASE("f256 log2 matches MPFR for fixed values", "[fltx][f256][precision][transcendental][log2]")
{
    accuracy_report_scope report_scope{ "f256 log2 matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0.125",
        "0.5",
        "0.999999999999999999999999999999999999999999999999999999999999",
        "1",
        "1.000000000000000000000000000000000000000000000000000000000001",
        "2",
        "10",
        "123456789.125"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "log2",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log2(value); },
            [](const mpfr_ref& value) { return ref_log2(value); });
    }
}

TEST_CASE("f256 log2 matches MPFR on random positive inputs", "[fltx][f256][precision][transcendental][log2]")
{
    accuracy_report_scope report_scope{ "f256 log2 matches MPFR on random positive inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random log2 cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_positive_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "log2",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log2(value); },
            [](const mpfr_ref& value) { return ref_log2(value); });
    }
}

TEST_CASE("f256 log10 matches MPFR for fixed values", "[fltx][f256][precision][transcendental][log10]")
{
    accuracy_report_scope report_scope{ "f256 log10 matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0.125",
        "0.5",
        "0.999999999999999999999999999999999999999999999999999999999999",
        "1",
        "1.000000000000000000000000000000000000000000000000000000000001",
        "2",
        "10",
        "123456789.125"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "log10",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log10(value); },
            [](const mpfr_ref& value) { return ref_log10(value); });
    }
}

TEST_CASE("f256 log10 matches MPFR on random positive inputs", "[fltx][f256][precision][transcendental][log10]")
{
    accuracy_report_scope report_scope{ "f256 log10 matches MPFR on random positive inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random log10 cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_positive_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "log10",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log10(value); },
            [](const mpfr_ref& value) { return ref_log10(value); });
    }
}

TEST_CASE("f256 pow matches MPFR for fixed values", "[fltx][f256][precision][transcendental][pow]")
{
    accuracy_report_scope report_scope{ "f256 pow matches MPFR for fixed values" };
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    check_pow_case("two_to_ten", mpfr_ref{ 2 }, mpfr_ref{ 10 }, abs_tolerance, rel_tolerance);
    check_pow_case("two_to_minus_ten", mpfr_ref{ 2 }, mpfr_ref{ -10 }, abs_tolerance, rel_tolerance);
    check_pow_case("minus_two_to_three", mpfr_ref{ -2 }, mpfr_ref{ 3 }, abs_tolerance, rel_tolerance);
    check_pow_case("minus_two_to_four", mpfr_ref{ -2 }, mpfr_ref{ 4 }, abs_tolerance, rel_tolerance);
    check_pow_case("minus_two_to_minus_three", mpfr_ref{ -2 }, mpfr_ref{ -3 }, abs_tolerance, rel_tolerance);
    check_pow_case("ten_to_half", mpfr_ref{ 10 }, mpfr_ref{ "0.5" }, abs_tolerance, rel_tolerance);
    check_pow_case("half_to_ten", mpfr_ref{ "0.5" }, mpfr_ref{ 10 }, abs_tolerance, rel_tolerance);
    check_pow_case("oneish_to_large", mpfr_ref{ "1.000000000000000000000000000000000000000000000000000000000001" }, mpfr_ref{ "123.5" }, abs_tolerance, rel_tolerance);
    check_pow_case("fractional_exp", mpfr_ref{ "123.456" }, mpfr_ref{ "0.125" }, abs_tolerance, rel_tolerance);
    check_pow_case("tiny_base", mpfr_ref{ "1e-20" }, mpfr_ref{ 2 }, abs_tolerance, rel_tolerance);
    check_pow_case("large_base_negative_exp", mpfr_ref{ "1e20" }, mpfr_ref{ -2 }, abs_tolerance, rel_tolerance);
}



TEST_CASE("f256 pow matches MPFR on random positive-base inputs", "[fltx][f256][precision][transcendental][pow]")
{
    accuracy_report_scope report_scope{ "f256 pow matches MPFR on random positive-base inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random pow cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref base = random_pow_base_for_f256(rng);
        const mpfr_ref exponent = random_signed_interval_for_f256(rng, mpfr_ref{ 8 });

        INFO("iteration: " << i);
        check_pow_case("random", base, exponent, abs_tolerance, rel_tolerance);
    }
}

TEST_CASE("f256 utility math helpers behave correctly for fixed values", "[fltx][f256][math][utility]")
{
    accuracy_report_scope report_scope{ "f256 utility math helpers behave correctly for fixed values" };
    
    check_unary_op("recip", "2.5",
        [](const f256& value) { return bl::recip(value); },
        [](const mpfr_ref& value) { return mpfr_ref{ 1 } / value; });
    
    check_unary_op("recip", "-0.125",
        [](const f256& value) { return bl::recip(value); },
        [](const mpfr_ref& value) { return mpfr_ref{ 1 } / value; });
    
    check_binary_op("fdim", "5.25", "2.0",
        [](const f256& x, const f256& y) { return bl::fdim(x, y); },
        [](const mpfr_ref& x, const mpfr_ref& y) { return x > y ? (x - y) : mpfr_ref{ 0 }; });
    
    check_binary_op("fdim", "-5.25", "2.0",
        [](const f256& x, const f256& y) { return bl::fdim(x, y); },
        [](const mpfr_ref& x, const mpfr_ref& y) { return x > y ? (x - y) : mpfr_ref{ 0 }; });
    
    {
        const f256 got = bl::abs(to_f256("-123.5"));
        require_exact_value("abs", got, to_f256("123.5"));
    }
    {
        const f256 got = bl::fabs(to_f256("-0.25"));
        require_exact_value("fabs", got, to_f256("0.25"));
    }
    {
        const f256 got = bl::clamp(to_f256("-5"), to_f256("-2"), to_f256("3"));
        require_exact_value("clamp.low", got, to_f256("-2"));
    }
    {
        const f256 got = bl::clamp(to_f256("1.5"), to_f256("-2"), to_f256("3"));
        require_exact_value("clamp.mid", got, to_f256("1.5"));
    }
    {
        const f256 got = bl::clamp(to_f256("5"), to_f256("-2"), to_f256("3"));
        require_exact_value("clamp.high", got, to_f256("3"));
    }
    {
        const f256 got = bl::fma(to_f256("1.25"), to_f256("2.5"), to_f256("-0.5"));
        const mpfr_ref expected = to_ref_exact(to_f256("1.25")) * to_ref_exact(to_f256("2.5")) + to_ref_exact(to_f256("-0.5"));
        const mpfr_ref diff = abs_ref(to_ref_exact(got) - expected);
        mpfr_ref scale = abs_ref(expected);
        if (scale < 1)
            scale = 1;
        const mpfr_ref tolerance = function_tolerance("fma", scale);
        record_accuracy_sample("fma", got, expected, diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }
    {
        const f256 nan = std::numeric_limits<f256>::quiet_NaN();
        const f256 pos = to_f256("2.5");
        const f256 neg = to_f256("-3.5");
        const f256 pos_zero{ 0.0 };
        const f256 neg_zero{ -0.0 };
    
        require_exact_value("fmin", bl::fmin(pos, neg), neg);
        require_exact_value("fmax", bl::fmax(pos, neg), pos);
        require_exact_value("fmin.nan", bl::fmin(nan, pos), pos);
        require_exact_value("fmax.nan", bl::fmax(nan, neg), neg);
        require_exact_value("fmin.zero", bl::fmin(pos_zero, neg_zero), neg_zero);
        require_exact_value("fmax.zero", bl::fmax(pos_zero, neg_zero), pos_zero);
    
        require_exact_value("copysign.pos_to_neg", bl::copysign(to_f256("1.25"), neg), to_f256("-1.25"));
        require_exact_value("copysign.neg_to_pos", bl::copysign(to_f256("-1.25"), pos), to_f256("1.25"));
    }
    {
        REQUIRE(bl::isnan(std::numeric_limits<f256>::quiet_NaN()));
        REQUIRE(bl::isinf(std::numeric_limits<f256>::infinity()));
        REQUIRE(bl::isfinite(to_f256("1.25")));
        REQUIRE(bl::iszero(f256{ 0.0, 0.0, 0.0, 0.0 }));
        REQUIRE(bl::ispositive(to_f256("0.25")));
        REQUIRE(!bl::ispositive(to_f256("-0.25")));
        REQUIRE(bl::signbit(f256{ -0.0, 0.0, 0.0, 0.0 }));
        REQUIRE(!bl::signbit(f256{ 0.0, 0.0, 0.0, 0.0 }));
        REQUIRE(bl::fpclassify(std::numeric_limits<f256>::quiet_NaN()) == FP_NAN);
        REQUIRE(bl::fpclassify(std::numeric_limits<f256>::infinity()) == FP_INFINITE);
        REQUIRE(bl::fpclassify(f256{ 0.0, 0.0, 0.0, 0.0 }) == FP_ZERO);
        REQUIRE(bl::isnormal(to_f256("1.0")));
        REQUIRE(bl::isunordered(std::numeric_limits<f256>::quiet_NaN(), to_f256("1.0")));
        REQUIRE(bl::isgreater(to_f256("2.0"), to_f256("1.0")));
        REQUIRE(bl::isgreaterequal(to_f256("2.0"), to_f256("2.0")));
        REQUIRE(bl::isless(to_f256("1.0"), to_f256("2.0")));
        REQUIRE(bl::islessequal(to_f256("2.0"), to_f256("2.0")));
        REQUIRE(bl::islessgreater(to_f256("1.0"), to_f256("2.0")));
    }
    {
        const std::array<int, 9> exponents = {{ -8, -3, -1, 0, 1, 3, 8, 16, 32 }};
        for (int exponent : exponents)
        {
            const f256 got = bl::pow10_256(exponent);
            const mpfr_ref expected = ref_pow10(exponent);
            const mpfr_ref got_ref = to_ref_exact(got);
            mpfr_ref scale = abs_ref(expected);
            if (scale < 1)
                scale = 1;
            const mpfr_ref tolerance = function_tolerance("pow10_256", scale);
            const mpfr_ref diff = abs_ref(got_ref - expected);
            CAPTURE(exponent);
            CAPTURE(to_text(got));
            CAPTURE(to_text(expected));
            record_accuracy_sample("pow10_256", got, expected, diff, scale, diff <= tolerance);
            REQUIRE(diff <= tolerance);
        }
    }
    {
        require_exact_value("round_to_decimals.2", bl::round_to_decimals(to_f256("1.2345"), 2), to_f256("1.23"));
        require_exact_value("round_to_decimals.3", bl::round_to_decimals(to_f256("1.2345"), 3), to_f256("1.234"));
        require_exact_value("round_to_decimals.tie_even", bl::round_to_decimals(to_f256("1.2355"), 3), to_f256("1.236"));
    
        REQUIRE(bl::lround(to_f256("2.5")) == 3L);    
        REQUIRE(bl::lround(to_f256("-2.5")) == -3L);  
        REQUIRE(bl::llround(to_f256("2.5")) == 3LL);  
        REQUIRE(bl::llround(to_f256("-2.5")) == -3LL);
        REQUIRE(bl::lrint(to_f256("2.5")) == 2L);     
        REQUIRE(bl::lrint(to_f256("3.5")) == 4L);     
        REQUIRE(bl::llrint(to_f256("-2.5")) == -2LL); 
        REQUIRE(bl::llrint(to_f256("-3.5")) == -4LL); 
    }
}

TEST_CASE("f256 public math results remain canonical on edge-shaped inputs", "[fltx][f256][math][canonical]")
{
    if (bl::is_constant_evaluated() && !bl::use_constexpr_parity())
        SKIP("canonicalization is not required when simulating consteval mode without FLTX_CONSTEXPR_PARITY");

    accuracy_report_scope report_scope{ "f256 public math results remain canonical on edge-shaped inputs" };

    const f256 a = detail::_f256::renorm4(1.0, std::ldexp(1.0, -60), -std::ldexp(1.0, -121), std::ldexp(1.0, -180));
    const f256 b = detail::_f256::renorm4(-0.375, -std::ldexp(1.0, -64), std::ldexp(1.0, -126), -std::ldexp(1.0, -190));
    const f256 domain = detail::_f256::renorm4(0.625, std::ldexp(1.0, -62), -std::ldexp(1.0, -123), std::ldexp(1.0, -184));
    const f256 positive = detail::_f256::renorm4(1.125, std::ldexp(1.0, -60), std::ldexp(1.0, -122), -std::ldexp(1.0, -186));
    const f256 gamma_arg = detail::_f256::renorm4(1.75, std::ldexp(1.0, -62), -std::ldexp(1.0, -124), std::ldexp(1.0, -188));
    const f256 target = to_f256("2.0");

    require_canonical_value("operator+", a + b);
    require_canonical_value("operator-", a - b);
    require_canonical_value("operator*", a * domain);
    require_canonical_value("operator/", a / positive);
    require_canonical_value("unary-", -a);

    require_canonical_value("abs", bl::abs(b));
    require_canonical_value("fabs", bl::fabs(b));
    require_canonical_value("clamp", bl::clamp(a, domain, positive));
    require_canonical_value("floor", bl::floor(b));
    require_canonical_value("ceil", bl::ceil(b));
    require_canonical_value("trunc", bl::trunc(b));
    require_canonical_value("round", bl::round(b));
    require_canonical_value("nearbyint", bl::nearbyint(b));
    require_canonical_value("rint", bl::rint(b));

    require_canonical_value("fma", bl::fma(a, positive, b));
    require_canonical_value("fmin", bl::fmin(a, b));
    require_canonical_value("fmax", bl::fmax(a, b));
    require_canonical_value("fdim", bl::fdim(a, domain));
    require_canonical_value("copysign", bl::copysign(a, b));

    require_canonical_value("fmod", bl::fmod(a, domain));
    require_canonical_value("remainder", bl::remainder(a, domain));
    int quo = 0;
    require_canonical_value("remquo", bl::remquo(a, domain, &quo));

    require_canonical_value("sqrt", bl::sqrt(positive));
    require_canonical_value("cbrt", bl::cbrt(b));
    require_canonical_value("hypot", bl::hypot(a, b));
    require_canonical_value("pow", bl::pow(positive, domain));
    require_canonical_value("pow.double", bl::pow(positive, 2.25));
    require_canonical_value("pow10_256", bl::pow10_256(-3));

    require_canonical_value("exp", bl::exp(domain));
    require_canonical_value("exp2", bl::exp2(domain));
    require_canonical_value("expm1", bl::expm1(domain));
    require_canonical_value("log", bl::log(positive));
    require_canonical_value("log2", bl::log2(positive));
    require_canonical_value("log10", bl::log10(positive));
    require_canonical_value("log1p", bl::log1p(domain));

    f256 sin_value{};
    f256 cos_value{};
    REQUIRE(bl::sincos(domain, sin_value, cos_value));
    require_canonical_value("sincos.sin", sin_value);
    require_canonical_value("sincos.cos", cos_value);
    require_canonical_value("sin", bl::sin(domain));
    require_canonical_value("cos", bl::cos(domain));
    require_canonical_value("tan", bl::tan(domain));
    require_canonical_value("atan", bl::atan(domain));
    require_canonical_value("atan2", bl::atan2(b, a));
    require_canonical_value("asin", bl::asin(domain));
    require_canonical_value("acos", bl::acos(domain));

    require_canonical_value("sinh", bl::sinh(domain));
    require_canonical_value("cosh", bl::cosh(domain));
    require_canonical_value("tanh", bl::tanh(domain));
    require_canonical_value("asinh", bl::asinh(b));
    require_canonical_value("acosh", bl::acosh(positive));
    require_canonical_value("atanh", bl::atanh(domain));

    int exponent = 0;
    require_canonical_value("frexp", bl::frexp(a, &exponent));
    f256 integer_part{};
    require_canonical_value("modf.frac", bl::modf(a, &integer_part));
    require_canonical_value("modf.int", integer_part);
    require_canonical_value("ldexp", bl::ldexp(a, 7));
    require_canonical_value("scalbn", bl::scalbn(a, -7));
    require_canonical_value("scalbln", bl::scalbln(a, 7));
    require_canonical_value("logb", bl::logb(a));
    require_canonical_value("nextafter", bl::nextafter(a, target));
    require_canonical_value("nexttoward.f256", bl::nexttoward(a, target));
    require_canonical_value("nexttoward.longdouble", bl::nexttoward(a, static_cast<long double>(2.0)));
    require_canonical_value("round_to_decimals", bl::round_to_decimals(to_f256("1.23456789"), 5));

    require_canonical_value("erf", bl::erf(domain));
    require_canonical_value("erfc", bl::erfc(domain));
    require_canonical_value("lgamma", bl::lgamma(gamma_arg));
    require_canonical_value("tgamma", bl::tgamma(gamma_arg));
}

TEST_CASE("f256 sincos matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][sincos]")
{
    accuracy_report_scope report_scope{ "f256 sincos matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0",
        "0.125",
        "-0.125",
        "0.785398163397448309615660845819875721",
        "-0.785398163397448309615660845819875721",
        "1.5",
        "-1.5",
        "1234.56789"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
        check_sincos_case("fixed", mpfr_ref{ input }, abs_tolerance, rel_tolerance);
}

TEST_CASE("f256 tan matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][tan]")
{
    accuracy_report_scope report_scope{ "f256 tan matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "-1.0",
        "-0.5",
        "-0.125",
        "0",
        "0.125",
        "0.5",
        "1.0",
        "3.0"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "tan",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tan(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tan(value); });
    }
}

TEST_CASE("f256 tan matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][trig][tan]")
{
    accuracy_report_scope report_scope{ "f256 tan matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random tan cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 1.2 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "tan",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tan(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tan(value); });
    }
}

TEST_CASE("f256 atan matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][atan]")
{
    accuracy_report_scope report_scope{ "f256 atan matches MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "-100",
        "-10",
        "-1",
        "-0.125",
        "0.125",
        "1",
        "10",
        "100"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "atan",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::atan(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::atan(value); });
    }
}

TEST_CASE("f256 atan matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][trig][atan]")
{
    accuracy_report_scope report_scope{ "f256 atan matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random atan cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 128.0 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "atan",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::atan(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::atan(value); });
    }
}

TEST_CASE("f256 atan2 matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][atan2]")
{
    accuracy_report_scope report_scope{ "f256 atan2 matches MPFR for fixed values" };
    const std::array<std::pair<const char*, const char*>, 10> cases = {{
        { "1", "1" },
        { "1", "-1" },
        { "-1", "1" },
        { "-1", "-1" },
        { "0.125", "10" },
        { "10", "0.125" },
        { "-0.125", "10" },
        { "10", "-0.125" },
        { "123.456", "789.25" },
        { "-123.456", "789.25" }
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op_with_tolerance(
            "atan2",
            lhs,
            rhs,
            abs_tolerance,
            rel_tolerance,
            [](const f256& y, const f256& x) { return bl::atan2(y, x); },
            [](const mpfr_ref& y, const mpfr_ref& x) { return boost::multiprecision::atan2(y, x); });
    }
}

TEST_CASE("f256 atan2 matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][trig][atan2]")
{
    accuracy_report_scope report_scope{ "f256 atan2 matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random atan2 cases", count);

    for (int i = 0; i < count; ++i)
    {
        mpfr_ref y = random_signed_interval_for_f256(rng, mpfr_ref{ 128.0 });
        mpfr_ref x = random_signed_interval_for_f256(rng, mpfr_ref{ 128.0 });
        if (x == 0 && y == 0)
            x = mpfr_ref{ 1.0 };

        const std::string lhs_text = to_scientific_text(y, printed_digits + 6);
        const std::string rhs_text = to_scientific_text(x, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("lhs_text: " << lhs_text);
        INFO("rhs_text: " << rhs_text);

        check_binary_op_with_tolerance(
            "atan2",
            lhs_text.c_str(),
            rhs_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& lhs, const f256& rhs) { return bl::atan2(lhs, rhs); },
            [](const mpfr_ref& lhs, const mpfr_ref& rhs) { return boost::multiprecision::atan2(lhs, rhs); });
    }
}

TEST_CASE("f256 asin and acos match MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][asin][acos]")
{
    accuracy_report_scope report_scope{ "f256 asin and acos match MPFR for fixed values" };
    const std::array<const char*, 9> cases = {{
        "-1",
        "-0.875",
        "-0.5",
        "-0.125",
        "0",
        "0.125",
        "0.5",
        "0.875",
        "1"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "asin",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::asin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::asin(value); });

        check_unary_op_with_tolerance(
            "acos",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::acos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::acos(value); });
    }
}

TEST_CASE("f256 asin and acos match MPFR on random unit inputs", "[fltx][f256][precision][transcendental][trig][asin][acos]")
{
    accuracy_report_scope report_scope{ "f256 asin and acos match MPFR on random unit inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random asin/acos cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_unit_symmetric_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "asin",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::asin(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::asin(value); });

        check_unary_op_with_tolerance(
            "acos",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::acos(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::acos(value); });
    }
}

TEST_CASE("f256 nearbyint and rint match ties-to-even references", "[fltx][f256][precision][math][nearbyint][rint]")
{
    accuracy_report_scope report_scope{ "f256 nearbyint and rint match ties-to-even references" };
    const std::array<const char*, 16> cases = {{
        "0",
        "-0",
        "0.25",
        "-0.25",
        "0.5",
        "-0.5",
        "1.5",
        "-1.5",
        "2.5",
        "-2.5",
        "3.5",
        "-3.5",
        "0.999999999999999999999999999999",
        "1.000000000000000000000000000001",
        "-0.999999999999999999999999999999",
        "-1.000000000000000000000000000001"
    }};

    for (const char* input : cases)
    {
        check_unary_op("nearbyint", input,
            [](const f256& value) { return bl::nearbyint(value); },
            [](const mpfr_ref& value) { return ref_round_to_even(value); });

        check_unary_op("rint", input,
            [](const f256& value) { return bl::rint(value); },
            [](const mpfr_ref& value) { return ref_round_to_even(value); });
    }

    {
        const f256 got = bl::nearbyint(to_f256("-0.5"));
        REQUIRE(bl::iszero(got));
        REQUIRE(bl::signbit(got));
    }
    {
        const f256 got = bl::rint(to_f256("-0.5"));
        REQUIRE(bl::iszero(got));
        REQUIRE(bl::signbit(got));
    }
}

TEST_CASE("f256 nearbyint and rint match ties-to-even references on random inputs", "[fltx][f256][precision][math][nearbyint][rint]")
{
    accuracy_report_scope report_scope{ "f256 nearbyint and rint match ties-to-even references on random inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    print_random_run("random nearbyint/rint cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = mpfr_ref{ random_finite_for_f256(rng) };
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op("nearbyint", input_text.c_str(),
            [](const f256& value) { return bl::nearbyint(value); },
            [](const mpfr_ref& value) { return ref_round_to_even(value); });

        check_unary_op("rint", input_text.c_str(),
            [](const f256& value) { return bl::rint(value); },
            [](const mpfr_ref& value) { return ref_round_to_even(value); });
    }
}

TEST_CASE("f256 expm1 matches MPFR for fixed values", "[fltx][f256][precision][transcendental][expm1]")
{
    accuracy_report_scope report_scope{ "f256 expm1 matches MPFR for fixed values" };
    const std::array<const char*, 9> cases = {{
        "-20",
        "-1",
        "-0.125",
        "-1e-10",
        "0",
        "1e-10",
        "0.125",
        "1",
        "20"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "expm1",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::expm1(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::expm1(value); });
    }
}

TEST_CASE("f256 expm1 matches MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][expm1]")
{
    accuracy_report_scope report_scope{ "f256 expm1 matches MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random expm1 cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 20.0 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "expm1",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::expm1(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::expm1(value); });
    }
}

TEST_CASE("f256 log1p matches MPFR for fixed values", "[fltx][f256][precision][transcendental][log1p]")
{
    accuracy_report_scope report_scope{ "f256 log1p matches MPFR for fixed values" };
    const std::array<const char*, 9> cases = {{
        "-0.875",
        "-0.5",
        "-0.125",
        "-1e-10",
        "0",
        "1e-10",
        "0.125",
        "1",
        "10"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "log1p",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log1p(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::log1p(value); });
    }
}

TEST_CASE("f256 log1p matches MPFR on random valid inputs", "[fltx][f256][precision][transcendental][log1p]")
{
    accuracy_report_scope report_scope{ "f256 log1p matches MPFR on random valid inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random log1p cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_log1p_argument_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "log1p",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::log1p(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::log1p(value); });
    }
}

TEST_CASE("f256 hyperbolic functions match MPFR for fixed values", "[fltx][f256][precision][transcendental][hyperbolic]")
{
    accuracy_report_scope report_scope{ "f256 hyperbolic functions match MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "-4",
        "-1",
        "-0.125",
        "0",
        "0.125",
        "1",
        "4",
        "8"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "sinh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sinh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sinh(value); });

        check_unary_op_with_tolerance(
            "cosh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cosh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cosh(value); });

        check_unary_op_with_tolerance(
            "tanh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tanh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tanh(value); });
    }
}

TEST_CASE("f256 hyperbolic functions match MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][hyperbolic]")
{
    accuracy_report_scope report_scope{ "f256 hyperbolic functions match MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random hyperbolic cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_signed_interval_for_f256(rng, mpfr_ref{ 8.0 });
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "sinh",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::sinh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::sinh(value); });

        check_unary_op_with_tolerance(
            "cosh",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cosh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::cosh(value); });

        check_unary_op_with_tolerance(
            "tanh",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tanh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tanh(value); });
    }
}

TEST_CASE("f256 inverse hyperbolic functions match MPFR for fixed values", "[fltx][f256][precision][transcendental][inverse_hyperbolic]")
{
    accuracy_report_scope report_scope{ "f256 inverse hyperbolic functions match MPFR for fixed values" };
    const std::array<const char*, 7> asinh_cases = {{ "-8", "-1", "-0.125", "0", "0.125", "1", "8" }};
    const std::array<const char*, 7> acosh_cases = {{ "1", "1.125", "1.5", "2", "4", "8", "16" }};
    const std::array<const char*, 8> atanh_cases = {{ "-0.95", "-0.5", "-0.125", "-1e-10", "0", "1e-10", "0.125", "0.95" }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : asinh_cases)
    {
        check_unary_op_with_tolerance(
            "asinh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::asinh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::asinh(value); });
    }

    for (const char* input : acosh_cases)
    {
        check_unary_op_with_tolerance(
            "acosh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::acosh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::acosh(value); });
    }

    for (const char* input : atanh_cases)
    {
        check_unary_op_with_tolerance(
            "atanh",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::atanh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::atanh(value); });
    }
}

TEST_CASE("f256 inverse hyperbolic functions match MPFR on random valid inputs", "[fltx][f256][precision][transcendental][inverse_hyperbolic]")
{
    accuracy_report_scope report_scope{ "f256 inverse hyperbolic functions match MPFR on random valid inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random inverse hyperbolic cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref asinh_input = random_signed_interval_for_f256(rng, mpfr_ref{ 8.0 });
        const mpfr_ref acosh_input = random_acosh_argument_for_f256(rng);
        const mpfr_ref atanh_input = random_atanh_argument_for_f256(rng);

        const std::string asinh_text = to_scientific_text(asinh_input, printed_digits + 6);
        const std::string acosh_text = to_scientific_text(acosh_input, printed_digits + 6);
        const std::string atanh_text = to_scientific_text(atanh_input, printed_digits + 6);

        INFO("iteration: " << i);

        check_unary_op_with_tolerance(
            "asinh",
            asinh_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::asinh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::asinh(value); });

        check_unary_op_with_tolerance(
            "acosh",
            acosh_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::acosh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::acosh(value); });

        check_unary_op_with_tolerance(
            "atanh",
            atanh_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::atanh(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::atanh(value); });
    }
}

TEST_CASE("f256 cbrt and hypot match MPFR for fixed values", "[fltx][f256][precision][math][cbrt][hypot]")
{
    accuracy_report_scope report_scope{ "f256 cbrt and hypot match MPFR for fixed values" };
    const std::array<const char*, 8> cbrt_cases = {{ "-1e9", "-8", "-0.125", "0", "0.125", "8", "27", "1e9" }};
    const std::array<std::pair<const char*, const char*>, 8> hypot_cases = {{
        { "0", "0" },
        { "3", "4" },
        { "-3", "4" },
        { "1e-40", "3e-20" },
        { "1e40", "3e20" },
        { "123.456", "789.25" },
        { "-123.456", "789.25" },
        { "0.125", "0.5" }
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cbrt_cases)
    {
        check_unary_op_with_tolerance(
            "cbrt",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cbrt(value); },
            [](const mpfr_ref& value) { return ref_cbrt(value); });
    }

    for (const auto& [lhs, rhs] : hypot_cases)
    {
        check_binary_op_with_tolerance(
            "hypot",
            lhs,
            rhs,
            abs_tolerance,
            rel_tolerance,
            [](const f256& x, const f256& y) { return bl::hypot(x, y); },
            [](const mpfr_ref& x, const mpfr_ref& y) { return ref_hypot(x, y); });
    }
}

TEST_CASE("f256 cbrt and hypot match MPFR on random moderate inputs", "[fltx][f256][precision][math][cbrt][hypot]")
{
    accuracy_report_scope report_scope{ "f256 cbrt and hypot match MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 256;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random cbrt/hypot cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref cbrt_input = random_signed_interval_for_f256(rng, mpfr_ref{ 1e12 });
        const mpfr_ref hypot_x = random_signed_interval_for_f256(rng, mpfr_ref{ 1e12 });
        const mpfr_ref hypot_y = random_signed_interval_for_f256(rng, mpfr_ref{ 1e12 });

        const std::string cbrt_text = to_scientific_text(cbrt_input, printed_digits + 6);
        const std::string x_text = to_scientific_text(hypot_x, printed_digits + 6);
        const std::string y_text = to_scientific_text(hypot_y, printed_digits + 6);

        INFO("iteration: " << i);

        check_unary_op_with_tolerance(
            "cbrt",
            cbrt_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::cbrt(value); },
            [](const mpfr_ref& value) { return ref_cbrt(value); });

        check_binary_op_with_tolerance(
            "hypot",
            x_text.c_str(),
            y_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& x, const f256& y) { return bl::hypot(x, y); },
            [](const mpfr_ref& x, const mpfr_ref& y) { return ref_hypot(x, y); });
    }
}

TEST_CASE("f256 decomposition and stepping functions behave correctly", "[fltx][f256][math][decomposition]")
{
    accuracy_report_scope report_scope{ "f256 decomposition and stepping functions behave correctly" };

    {
        const f256 input = to_f256("123.456");
        int exponent = 0;
        const f256 mantissa = bl::frexp(input, &exponent);
        const f256 rebuilt = bl::ldexp(mantissa, exponent);
        const mpfr_ref diff = abs_ref(to_ref_exact(rebuilt) - to_ref_exact(input));
        const mpfr_ref scale = abs_ref(to_ref_exact(input));
        const mpfr_ref tolerance = function_tolerance("frexp", scale);
        CAPTURE(exponent);
        CAPTURE(to_text(mantissa));
        CAPTURE(to_text(rebuilt));
        REQUIRE(abs_ref(to_ref_exact(mantissa)) >= mpfr_ref{ 0.5 });
        REQUIRE(abs_ref(to_ref_exact(mantissa)) < mpfr_ref{ 1.0 });
        record_accuracy_sample("frexp", rebuilt, to_ref_exact(input), diff, scale, diff <= tolerance);
        REQUIRE(diff <= tolerance);
    }
    {
        const f256 input = to_f256("-123.456");
        f256 ip{};
        const f256 frac = bl::modf(input, &ip);
        const mpfr_ref sum_diff = abs_ref((to_ref_exact(frac) + to_ref_exact(ip)) - to_ref_exact(input));
        CAPTURE(to_text(frac));
        CAPTURE(to_text(ip));
        const mpfr_ref sum_scale = abs_ref(to_ref_exact(input));
        REQUIRE(sum_diff <= function_tolerance("modf", sum_scale));
        require_exact_value("modf.integer", ip, bl::trunc(input));
    }
    {
        const f256 input = to_f256("8.0");
        REQUIRE(bl::ilogb(input) == 3);
        require_exact_value("logb", bl::logb(input), to_f256("3.0"));
    }
    {
        const f256 input = to_f256("1.5");
        require_exact_value("scalbn", bl::scalbn(input, 5), bl::ldexp(input, 5));
        require_exact_value("scalbln", bl::scalbln(input, -5), bl::ldexp(input, -5));
    }
    {
        const f256 from = to_f256("1.25");
        const f256 to = to_f256("2.0");
        const f256 expected = detail::_f256::renorm4(from.x0, from.x1, from.x2, std::nextafter(from.x3, std::numeric_limits<double>::infinity()));
        REQUIRE(expected.x1 != 0.0);
        REQUIRE(expected.x2 == 0.0);
        REQUIRE(expected.x3 == 0.0);
        require_exact_value("nextafter.up", bl::nextafter(from, to), expected);
        require_exact_value("nexttoward.f256", bl::nexttoward(from, to), expected);
        require_exact_value("nexttoward.longdouble", bl::nexttoward(from, static_cast<long double>(2.0)), expected);
    }
    {
        const f256 from = to_f256("1.25");
        const f256 to = to_f256("-2.0");
        const f256 expected = detail::_f256::renorm4(from.x0, from.x1, from.x2, std::nextafter(from.x3, -std::numeric_limits<double>::infinity()));
        REQUIRE(expected.x1 != 0.0);
        REQUIRE(expected.x2 == 0.0);
        REQUIRE(expected.x3 == 0.0);
        require_exact_value("nextafter.down", bl::nextafter(from, to), expected);
    }
    {
        const f256 from = detail::_f256::renorm4(1.0, 0x1p-60, 0x1p-120, 0.0);
        const f256 to = to_f256("2.0");
        const f256 expected = detail::_f256::renorm4(from.x0, from.x1, from.x2, std::nextafter(from.x3, std::numeric_limits<double>::infinity()));
        REQUIRE(expected.x1 != 0.0);
        REQUIRE(expected.x2 != 0.0);
        REQUIRE(expected.x3 != 0.0);
        require_exact_value("nextafter.normalized_tail_up", bl::nextafter(from, to), expected);
    }
    {
        const f256 got = bl::nextafter(f256{ 0.0, 0.0, 0.0, 0.0 }, to_f256("-1.0"));
        require_exact_value("nextafter.zero", got, f256{ -std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 });
    }
    {
        const mpfr_ref abs_tolerance{ "1e-59" };
        const mpfr_ref rel_tolerance{ "1e-59" };
        check_remquo_case("fixed.positive", mpfr_ref{ "5.25" }, mpfr_ref{ "2" }, abs_tolerance, rel_tolerance);
        check_remquo_case("fixed.negative", mpfr_ref{ "-5.25" }, mpfr_ref{ "2" }, abs_tolerance, rel_tolerance);
        check_remquo_case("fixed.fractional", mpfr_ref{ "123.456" }, mpfr_ref{ "0.5" }, abs_tolerance, rel_tolerance);
    }
}

TEST_CASE("f256 erf and erfc match MPFR for fixed values", "[fltx][f256][precision][transcendental][erf][erfc]")
{
    accuracy_report_scope report_scope{ "f256 erf and erfc match MPFR for fixed values" };
    const std::array<const char*, 9> cases = {{
        "-4",
        "-1",
        "-0.125",
        "-1e-10",
        "0",
        "1e-10",
        "0.125",
        "1",
        "4"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "erf",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::erf(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::erf(value); });

        check_unary_op_with_tolerance(
            "erfc",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::erfc(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::erfc(value); });
    }
}

TEST_CASE("f256 erf and erfc match MPFR on random moderate inputs", "[fltx][f256][precision][transcendental][erf][erfc]")
{
    accuracy_report_scope report_scope{ "f256 erf and erfc match MPFR on random moderate inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random erf/erfc cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_erf_argument_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "erf",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::erf(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::erf(value); });

        check_unary_op_with_tolerance(
            "erfc",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::erfc(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::erfc(value); });
    }
}

TEST_CASE("f256 lgamma and tgamma match MPFR for fixed values", "[fltx][f256][precision][transcendental][gamma]")
{
    accuracy_report_scope report_scope{ "f256 lgamma and tgamma match MPFR for fixed values" };
    const std::array<const char*, 8> cases = {{
        "0.125",
        "0.5",
        "0.75",
        "1",
        "1.5",
        "2.5",
        "5.5",
        "10"
    }};

    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };

    for (const char* input : cases)
    {
        check_unary_op_with_tolerance(
            "lgamma",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::lgamma(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::lgamma(value); });

        check_unary_op_with_tolerance(
            "tgamma",
            input,
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tgamma(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tgamma(value); });
    }
}

TEST_CASE("f256 lgamma and tgamma match MPFR on random positive inputs", "[fltx][f256][precision][transcendental][gamma]")
{
    accuracy_report_scope report_scope{ "f256 lgamma and tgamma match MPFR on random positive inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 128 * random_sample_count_scale;
    const mpfr_ref abs_tolerance{ "1e-59" };
    const mpfr_ref rel_tolerance{ "1e-59" };
    print_random_run("random gamma cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_gamma_positive_for_f256(rng);
        const std::string input_text = to_scientific_text(input, printed_digits + 6);

        INFO("iteration: " << i);
        INFO("input_text: " << input_text);

        check_unary_op_with_tolerance(
            "lgamma",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::lgamma(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::lgamma(value); });

        check_unary_op_with_tolerance(
            "tgamma",
            input_text.c_str(),
            abs_tolerance,
            rel_tolerance,
            [](const f256& value) { return bl::tgamma(value); },
            [](const mpfr_ref& value) { return boost::multiprecision::tgamma(value); });
    }
}
