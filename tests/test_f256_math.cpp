#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include <fltx/f256.h>

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
                const double median = median_digits(entry.achieved_digits);
                const double worst = *std::min_element(entry.achieved_digits.begin(), entry.achieved_digits.end());

                std::cout << "  " << op_name
                          << ": pass " << entry.passed << "/" << entry.samples
                          << ", median " << median << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(median) << "%)"
                          << ", worst " << worst << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(worst) << "%)\n";
            }

            std::cout.flags(old_flags);
            std::cout.precision(old_precision);
        }

        void record(const char* op_name, const mpfr_ref& diff, const mpfr_ref& scale, bool passed)
        {
            auto& entry = stats[op_name];
            ++entry.samples;
            if (passed)
                ++entry.passed;
            entry.achieved_digits.push_back(achieved_digits_from_error(diff, scale));
        }

    private:
        std::string test_name;
        accuracy_report_scope* previous = nullptr;
        std::map<std::string, accuracy_stats_entry> stats;
    };

    void record_accuracy_sample(const char* op_name, const mpfr_ref& diff, const mpfr_ref& scale, bool passed)
    {
        if (current_accuracy_report_scope != nullptr)
            current_accuracy_report_scope->record(op_name, diff, scale, passed);
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

        const mpfr_ref tolerance = decimal_epsilon(checked_digits) * scale;
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

        record_accuracy_sample(op_name, diff, scale, diff <= tolerance);
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

        const mpfr_ref tolerance = decimal_epsilon(checked_digits) * scale;
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

        record_accuracy_sample(op_name, diff, scale, diff <= tolerance);
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

        record_accuracy_sample(op_name, diff, accuracy_scale, diff <= tolerance);
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
        const mpfr_ref rel_based_tolerance = rel_tolerance * scale;
        mpfr_ref tolerance = abs_tolerance;
        if (rel_based_tolerance > tolerance)
            tolerance = rel_based_tolerance;

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

        record_accuracy_sample(op_name, diff, accuracy_scale, diff <= tolerance);
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

        const mpfr_ref tolerance = decimal_epsilon(checked_digits) * scale;
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

        record_accuracy_sample("ldexp", diff, scale, diff <= tolerance);
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

TEST_CASE("f256 sin matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][sin]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR for fixed values" };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduced_abs_tolerance{ "1e-67" };
    const mpfr_ref reduced_rel_tolerance{ "5e-61" };
    const mpfr_ref reduction_abs_tolerance{ "3e-60" };
    const mpfr_ref reduction_rel_tolerance{ "6e-59" };

    check_sin_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_sin_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
    check_sin_case("pi_over_2", half_pi, reduction_abs_tolerance, reduction_rel_tolerance);
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

    constexpr int count = 1000;
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

TEST_CASE("f256 sin matches MPFR on random range-reduced inputs", "[fltx][f256][precision][transcendental][trig][sin]")
{
    accuracy_report_scope report_scope{ "f256 sin matches MPFR on random range-reduced inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "3e-60" };
    const mpfr_ref rel_tolerance{ "6e-59" };
    print_random_run("random range-reduced sin cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
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

TEST_CASE("f256 cos matches MPFR for fixed values", "[fltx][f256][precision][transcendental][trig][cos]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR for fixed values" };
    const mpfr_ref pi = pi_ref();
    const mpfr_ref half_pi = pi / 2;
    const mpfr_ref quarter_pi = pi / 4;
    const mpfr_ref third_pi = pi / 3;
    const mpfr_ref sixth_pi = pi / 6;
    const mpfr_ref two_pi = pi * 2;
    const mpfr_ref tiny{ "1e-40" };

    const mpfr_ref reduced_abs_tolerance{ "1e-67" };
    const mpfr_ref reduced_rel_tolerance{ "5e-61" };
    const mpfr_ref reduction_abs_tolerance{ "3e-60" };
    const mpfr_ref reduction_rel_tolerance{ "6e-59" };

    check_cos_case("zero", mpfr_ref{ 0 }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("minus_zero", mpfr_ref{ "-0" }, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_positive", tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("tiny_negative", -tiny, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_6", sixth_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_4", quarter_pi, reduced_abs_tolerance, reduced_rel_tolerance);
    check_cos_case("pi_over_3", third_pi, reduction_abs_tolerance, reduction_rel_tolerance);
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

    constexpr int count = 1000;
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

TEST_CASE("f256 cos matches MPFR on random range-reduced inputs", "[fltx][f256][precision][transcendental][trig][cos]")
{
    accuracy_report_scope report_scope{ "f256 cos matches MPFR on random range-reduced inputs" };
    std::mt19937_64 rng{ random_seed };

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "3e-60" };
    const mpfr_ref rel_tolerance{ "6e-59" };
    print_random_run("random range-reduced cos cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref input = random_sine_reduction_argument_for_f256(rng);
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
            [](const mpfr_ref& value) { return ref_round_to_even(value); });
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

    constexpr int count = 1000;
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
            [](const mpfr_ref& value) { return ref_round_to_even(value); });
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

    constexpr int count = 1000;
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

    constexpr int count = 1000;
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

    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };
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

    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };
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

    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };
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

    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };
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

    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "2e-58" };
    const mpfr_ref rel_tolerance{ "2e-57" };
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
    const mpfr_ref abs_tolerance{ "5e-57" };
    const mpfr_ref rel_tolerance{ "5e-56" };

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

    constexpr int count = 1000;
    const mpfr_ref abs_tolerance{ "5e-57" };
    const mpfr_ref rel_tolerance{ "5e-56" };
    print_random_run("random pow cases", count);

    for (int i = 0; i < count; ++i)
    {
        const mpfr_ref base = random_pow_base_for_f256(rng);
        const mpfr_ref exponent = random_signed_interval_for_f256(rng, mpfr_ref{ 8 });

        INFO("iteration: " << i);
        check_pow_case("random", base, exponent, abs_tolerance, rel_tolerance);
    }
}
