#include <fltx/f128_math.h>
#include <fltx/f128_rounding.h>

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using fltx_type = bl::f128;
using boost_type = boost::multiprecision::cpp_double_double;
using mpfr_type = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<200>, boost::multiprecision::et_off>;

constexpr std::size_t samples_scale = 1;
constexpr std::size_t brute_force_samples = 10'000 * samples_scale;
#ifdef NDEBUG
constexpr std::size_t benchmark_min_iterations = 200'000;
#else
constexpr std::size_t benchmark_min_iterations = 20'000;
#endif

struct value_sample
{
    const char* name;
    double hi;
    double lo;
};

struct unary_sample
{
    const char* name;
    value_sample x;
};

struct binary_sample
{
    const char* name;
    value_sample x;
    value_sample y;
};

struct ternary_sample
{
    const char* name;
    value_sample x;
    value_sample y;
    value_sample z;
};

struct unary_int_sample
{
    const char* name;
    value_sample x;
    int n;
};

template <typename T>
struct binary_value
{
    T x;
    T y;
};

template <typename T>
struct ternary_value
{
    T x;
    T y;
    T z;
};

template <typename T>
struct unary_int_value
{
    T x;
    int n;
};

template <typename T>
struct frexp_value
{
    T fraction;
    int exponent;
};

struct quality_result
{
    double min_bits = std::numeric_limits<double>::infinity();
    double mean_bits = 0.0;
    std::size_t sample_count = 0;
};

struct benchmark_result
{
    double ns_per_iter = 0.0;
    std::size_t iteration_count = 0;
};

[[nodiscard]] std::string bits_text(double bits);
[[nodiscard]] std::string bit_gap_text(double fltx_bits, double boost_bits);
[[nodiscard]] std::string ns_text(double ns_per_iter);
[[nodiscard]] std::string performance_ratio_text(double fltx_ns_per_iter, double boost_ns_per_iter);

struct quality_record
{
    std::string operation;
    quality_result fltx;
    quality_result boost;
    benchmark_result fltx_bench;
    benchmark_result boost_bench;
};

struct accuracy_report
{
    bool printed_header = false;

    void print_header()
    {
        std::cout << "\n[f128 accuracy summary]\n";
        std::cout << std::left << std::setw(12) << "operation"
                  << std::right << std::setw(9) << "samples" << "  "
                  << std::left
                  << std::setw(16) << "f128 min"
                  << std::setw(16) << "f128 mean"
                  << std::setw(20) << "cpp_dd min"
                  << std::setw(20) << "cpp_dd mean"
                  << std::setw(14) << "min gap"
                  << std::setw(16) << "f128 ns/iter"
                  << std::setw(18) << "cpp_dd ns/iter"
                  << "performance ratio\n";
        printed_header = true;
    }

    void print_record(const quality_record& record)
    {
        if (!printed_header)
            print_header();

        std::cout << std::left << std::setw(12) << record.operation
                  << std::right << std::setw(9) << record.fltx.sample_count << "  "
                  << std::left
                  << std::setw(16) << bits_text(record.fltx.min_bits)
                  << std::setw(16) << bits_text(record.fltx.mean_bits)
                  << std::setw(20) << bits_text(record.boost.min_bits)
                  << std::setw(20) << bits_text(record.boost.mean_bits)
                  << std::setw(14) << bit_gap_text(record.fltx.min_bits, record.boost.min_bits)
                  << std::setw(16) << ns_text(record.fltx_bench.ns_per_iter)
                  << std::setw(18) << ns_text(record.boost_bench.ns_per_iter)
                  << performance_ratio_text(record.fltx_bench.ns_per_iter, record.boost_bench.ns_per_iter) << '\n'
                  << std::flush;
    }
};

[[nodiscard]] accuracy_report& report()
{
    static accuracy_report instance;
    return instance;
}

[[nodiscard]] fltx_type make_fltx(const value_sample& x) noexcept
{
    return fltx_type{x.hi, x.lo};
}

[[nodiscard]] boost_type make_boost(const value_sample& x)
{
    return boost_type(x.hi) + boost_type(x.lo);
}

[[nodiscard]] mpfr_type make_mpfr(const value_sample& x)
{
    return mpfr_type(x.hi) + mpfr_type(x.lo);
}

[[nodiscard]] mpfr_type to_mpfr(const fltx_type& x)
{
    return mpfr_type(x.hi) + mpfr_type(x.lo);
}

[[nodiscard]] mpfr_type to_mpfr(const boost_type& x)
{
    const auto parts = x.backend().crep();
    return mpfr_type(parts.first) + mpfr_type(parts.second);
}

void require_exact_value(const char* label, const fltx_type& got, const fltx_type& expected)
{
    INFO(label);
    REQUIRE(got.hi == expected.hi);
    REQUIRE(got.lo == expected.lo);
}

[[nodiscard]] mpfr_type abs_mpfr(const mpfr_type& x)
{
    return x < 0 ? -x : x;
}

[[nodiscard]] mpfr_type reference_scale(const mpfr_type& expected)
{
    const auto scale = abs_mpfr(expected);
    return scale < 1 ? mpfr_type(1) : scale;
}

[[nodiscard]] double accuracy_bits(const mpfr_type& actual, const mpfr_type& expected)
{
    const auto error = abs_mpfr(actual - expected);
    if (error == 0)
        return std::numeric_limits<double>::infinity();

    const auto scaled_error = error / reference_scale(expected);
    using std::log2;
    return static_cast<double>(-log2(scaled_error));
}

[[nodiscard]] double finite_for_mean(double bits)
{
    constexpr double exact_result_bits = 160.0;
    return std::isinf(bits) ? exact_result_bits : bits;
}

[[nodiscard]] std::string bits_text(double bits)
{
    if (std::isinf(bits))
        return "exact";

    std::ostringstream out;
    out.precision(2);
    out << std::fixed << bits << " bits";
    return out.str();
}

[[nodiscard]] std::string bit_gap_text(double fltx_bits, double boost_bits)
{
    if (std::isinf(fltx_bits) && std::isinf(boost_bits))
        return "0.00 bits";
    if (std::isinf(fltx_bits))
        return "+inf";
    if (std::isinf(boost_bits))
        return "-inf";

    std::ostringstream out;
    out.precision(2);
    out << std::showpos << std::fixed << (fltx_bits - boost_bits) << " bits";
    return out.str();
}

[[nodiscard]] std::string ns_text(double ns_per_iter)
{
    if (ns_per_iter <= 0.0)
        return "-";

    std::ostringstream out;
    out.precision(2);
    out << std::fixed << ns_per_iter << " ns";
    return out.str();
}

[[nodiscard]] std::string performance_ratio_text(double fltx_ns_per_iter, double boost_ns_per_iter)
{
    if (fltx_ns_per_iter <= 0.0 || boost_ns_per_iter <= 0.0)
        return "-";

    std::ostringstream out;
    out.precision(2);
    out << std::fixed << (boost_ns_per_iter / fltx_ns_per_iter) << 'x';
    return out.str();
}

struct splitmix64
{
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept
    {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    [[nodiscard]] double unit() noexcept
    {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

    [[nodiscard]] double sign() noexcept
    {
        return (next() & 1ull) ? 1.0 : -1.0;
    }
};

[[nodiscard]] value_sample make_runtime_value(const char* name, double hi, double lo = 0.0) noexcept
{
    const auto value = bl::detail::_f128::renorm(hi, lo);
    return {name, value.hi, value.lo};
}

[[nodiscard]] double residual_for(double hi, splitmix64& rng) noexcept
{
    if (hi == 0.0 || !std::isfinite(hi))
        return 0.0;

    const int exponent = std::ilogb(std::fabs(hi));
    return rng.sign() * std::ldexp(0.5 + rng.unit(), exponent - 60);
}

[[nodiscard]] double signed_log_value(splitmix64& rng, int min_exp, int max_exp) noexcept
{
    const int exponent = min_exp + static_cast<int>(rng.unit() * static_cast<double>(max_exp - min_exp + 1));
    return rng.sign() * std::ldexp(1.0 + rng.unit(), exponent);
}

[[nodiscard]] double positive_log_value(splitmix64& rng, int min_exp, int max_exp) noexcept
{
    const int exponent = min_exp + static_cast<int>(rng.unit() * static_cast<double>(max_exp - min_exp + 1));
    return std::ldexp(1.0 + rng.unit(), exponent);
}

[[nodiscard]] double uniform_value(splitmix64& rng, double min, double max) noexcept
{
    return min + (max - min) * rng.unit();
}

[[nodiscard]] double positive_residual_for(double hi, splitmix64& rng) noexcept
{
    return std::fabs(residual_for(hi, rng));
}

volatile double benchmark_sink = 0.0;

void consume_benchmark_value(const fltx_type& value) noexcept
{
    benchmark_sink = benchmark_sink + value.hi;
}

void consume_benchmark_value(const boost_type& value)
{
    benchmark_sink = benchmark_sink + value.backend().crep().first;
}

template <typename T>
void consume_benchmark_value(const frexp_value<T>& value)
{
    consume_benchmark_value(value.fraction);
    benchmark_sink = benchmark_sink + static_cast<double>(value.exponent);
}

[[nodiscard]] std::size_t benchmark_repetitions(std::size_t sample_count) noexcept
{
    if (sample_count == 0)
        return 0;
    return std::max<std::size_t>(1, (benchmark_min_iterations + sample_count - 1) / sample_count);
}

template <typename Samples, typename T, typename MakeFn>
[[nodiscard]] std::vector<T> make_unary_inputs(const Samples& samples, MakeFn make_value)
{
    std::vector<T> values;
    values.reserve(samples.size());
    for (const auto& sample : samples)
        values.push_back(make_value(sample.x));
    return values;
}

template <typename Samples, typename T, typename MakeFn>
[[nodiscard]] std::vector<binary_value<T>> make_binary_inputs(const Samples& samples, MakeFn make_value)
{
    std::vector<binary_value<T>> values;
    values.reserve(samples.size());
    for (const auto& sample : samples)
        values.push_back({make_value(sample.x), make_value(sample.y)});
    return values;
}

template <typename Samples, typename T, typename MakeFn>
[[nodiscard]] std::vector<ternary_value<T>> make_ternary_inputs(const Samples& samples, MakeFn make_value)
{
    std::vector<ternary_value<T>> values;
    values.reserve(samples.size());
    for (const auto& sample : samples)
        values.push_back({make_value(sample.x), make_value(sample.y), make_value(sample.z)});
    return values;
}

template <typename Samples, typename T, typename MakeFn>
[[nodiscard]] std::vector<unary_int_value<T>> make_unary_int_inputs(const Samples& samples, MakeFn make_value)
{
    std::vector<unary_int_value<T>> values;
    values.reserve(samples.size());
    for (const auto& sample : samples)
        values.push_back({make_value(sample.x), sample.n});
    return values;
}

template <typename Values, typename EvalFn>
[[nodiscard]] benchmark_result benchmark_unary_values(const Values& values, EvalFn eval)
{
    const std::size_t repetitions = benchmark_repetitions(values.size());
    if (repetitions == 0)
        return {};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        for (const auto& value : values)
            consume_benchmark_value(eval(value));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const std::size_t iterations = repetitions * values.size();
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
    return {ns, iterations};
}

template <typename Values, typename EvalFn>
[[nodiscard]] benchmark_result benchmark_binary_values(const Values& values, EvalFn eval)
{
    const std::size_t repetitions = benchmark_repetitions(values.size());
    if (repetitions == 0)
        return {};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        for (const auto& value : values)
            consume_benchmark_value(eval(value.x, value.y));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const std::size_t iterations = repetitions * values.size();
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
    return {ns, iterations};
}

template <typename Values, typename EvalFn>
[[nodiscard]] benchmark_result benchmark_ternary_values(const Values& values, EvalFn eval)
{
    const std::size_t repetitions = benchmark_repetitions(values.size());
    if (repetitions == 0)
        return {};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        for (const auto& value : values)
            consume_benchmark_value(eval(value.x, value.y, value.z));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const std::size_t iterations = repetitions * values.size();
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
    return {ns, iterations};
}

template <typename Values, typename EvalFn>
[[nodiscard]] benchmark_result benchmark_unary_int_values(const Values& values, EvalFn eval)
{
    const std::size_t repetitions = benchmark_repetitions(values.size());
    if (repetitions == 0)
        return {};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        for (const auto& value : values)
            consume_benchmark_value(eval(value.x, value.n));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const std::size_t iterations = repetitions * values.size();
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
    return {ns, iterations};
}

template <typename Samples, typename Values, typename ToMpfrFn, typename EvalFn, typename RefFn>
[[nodiscard]] quality_result check_unary_backend(
    const char* operation,
    const char* backend_name,
    double required_min_bits,
    bool enforce_min_bits,
    const Samples& samples,
    const Values& values,
    ToMpfrFn to_reference_value,
    EvalFn eval,
    RefFn reference)
{
    double total_bits = 0.0;
    double min_bits = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        const auto actual = to_reference_value(eval(values[i]));
        const auto expected = reference(make_mpfr(sample.x));
        const auto bits = accuracy_bits(actual, expected);

        INFO(operation << ' ' << backend_name << " sample '" << sample.name << "' measured " << bits_text(bits));
        CHECK(!std::isnan(bits));
        if (enforce_min_bits)
            CHECK(bits >= required_min_bits);

        min_bits = std::min(min_bits, bits);
        total_bits += finite_for_mean(bits);
    }

    return {min_bits, total_bits / static_cast<double>(samples.size()), samples.size()};
}

template <typename Samples, typename Values, typename ToMpfrFn, typename EvalFn, typename RefFn>
[[nodiscard]] quality_result check_binary_backend(
    const char* operation,
    const char* backend_name,
    double required_min_bits,
    bool enforce_min_bits,
    const Samples& samples,
    const Values& values,
    ToMpfrFn to_reference_value,
    EvalFn eval,
    RefFn reference)
{
    double total_bits = 0.0;
    double min_bits = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        const auto actual = to_reference_value(eval(values[i].x, values[i].y));
        const auto expected = reference(make_mpfr(sample.x), make_mpfr(sample.y));
        const auto bits = accuracy_bits(actual, expected);

        INFO(operation << ' ' << backend_name << " sample '" << sample.name << "' measured " << bits_text(bits));
        CHECK(!std::isnan(bits));
        if (enforce_min_bits)
            CHECK(bits >= required_min_bits);

        min_bits = std::min(min_bits, bits);
        total_bits += finite_for_mean(bits);
    }

    return {min_bits, total_bits / static_cast<double>(samples.size()), samples.size()};
}

template <typename Samples, typename Values, typename ToMpfrFn, typename EvalFn, typename RefFn>
[[nodiscard]] quality_result check_ternary_backend(
    const char* operation,
    const char* backend_name,
    double required_min_bits,
    bool enforce_min_bits,
    const Samples& samples,
    const Values& values,
    ToMpfrFn to_reference_value,
    EvalFn eval,
    RefFn reference)
{
    double total_bits = 0.0;
    double min_bits = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        const auto actual = to_reference_value(eval(values[i].x, values[i].y, values[i].z));
        const auto expected = reference(make_mpfr(sample.x), make_mpfr(sample.y), make_mpfr(sample.z));
        const auto bits = accuracy_bits(actual, expected);

        INFO(operation << ' ' << backend_name << " sample '" << sample.name << "' measured " << bits_text(bits));
        CHECK(!std::isnan(bits));
        if (enforce_min_bits)
            CHECK(bits >= required_min_bits);

        min_bits = std::min(min_bits, bits);
        total_bits += finite_for_mean(bits);
    }

    return {min_bits, total_bits / static_cast<double>(samples.size()), samples.size()};
}

template <typename Samples, typename Values, typename ToMpfrFn, typename EvalFn, typename RefFn>
[[nodiscard]] quality_result check_unary_int_backend(
    const char* operation,
    const char* backend_name,
    double required_min_bits,
    bool enforce_min_bits,
    const Samples& samples,
    const Values& values,
    ToMpfrFn to_reference_value,
    EvalFn eval,
    RefFn reference)
{
    double total_bits = 0.0;
    double min_bits = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        const auto actual = to_reference_value(eval(values[i].x, values[i].n));
        const auto expected = reference(make_mpfr(sample.x), sample.n);
        const auto bits = accuracy_bits(actual, expected);

        INFO(operation << ' ' << backend_name << " sample '" << sample.name << "' measured " << bits_text(bits));
        CHECK(!std::isnan(bits));
        if (enforce_min_bits)
            CHECK(bits >= required_min_bits);

        min_bits = std::min(min_bits, bits);
        total_bits += finite_for_mean(bits);
    }

    return {min_bits, total_bits / static_cast<double>(samples.size()), samples.size()};
}

template <typename Samples, typename Values, typename EvalFn>
[[nodiscard]] quality_result check_frexp_backend(
    const char* operation,
    const char* backend_name,
    double required_min_bits,
    bool enforce_min_bits,
    const Samples& samples,
    const Values& values,
    EvalFn eval)
{
    double total_bits = 0.0;
    double min_bits = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        const auto actual = eval(values[i]);
        const auto expected = make_mpfr(sample.x);
        const auto bits = accuracy_bits(actual, expected);

        INFO(operation << ' ' << backend_name << " sample '" << sample.name << "' measured " << bits_text(bits));
        CHECK(!std::isnan(bits));
        if (enforce_min_bits)
            CHECK(bits >= required_min_bits);

        min_bits = std::min(min_bits, bits);
        total_bits += finite_for_mean(bits);
    }

    return {min_bits, total_bits / static_cast<double>(samples.size()), samples.size()};
}

void compare_quality(
    const char* operation,
    const quality_result& fltx,
    const quality_result& boost,
    const benchmark_result& fltx_bench,
    const benchmark_result& boost_bench)
{
    constexpr double comparison_slack_bits = 16.0;

    report().print_record({operation, fltx, boost, fltx_bench, boost_bench});

    INFO(operation << " f128 minimum " << bits_text(fltx.min_bits) << ", cpp_double_double minimum "
                   << bits_text(boost.min_bits));
    INFO(operation << " f128 mean " << bits_text(fltx.mean_bits) << ", cpp_double_double mean "
                   << bits_text(boost.mean_bits));

    CHECK(fltx.min_bits + comparison_slack_bits >= boost.min_bits);
}

template <typename Samples, typename EvalFn, typename RefFn>
void check_unary_operation(
    const char* operation,
    double required_min_bits,
    const Samples& samples,
    EvalFn eval,
    RefFn reference)
{
    const auto fltx_inputs = make_unary_inputs<Samples, fltx_type>(samples, [](const value_sample& x) { return make_fltx(x); });
    const auto boost_inputs = make_unary_inputs<Samples, boost_type>(samples, [](const value_sample& x) { return make_boost(x); });

    const auto fltx = check_unary_backend(
        operation,
        "f128",
        required_min_bits,
        true,
        samples,
        fltx_inputs,
        [](const fltx_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto boost = check_unary_backend(
        operation,
        "cpp_double_double",
        required_min_bits,
        false,
        samples,
        boost_inputs,
        [](const boost_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto fltx_bench = benchmark_unary_values(fltx_inputs, eval);
    const auto boost_bench = benchmark_unary_values(boost_inputs, eval);

    compare_quality(operation, fltx, boost, fltx_bench, boost_bench);
}

template <typename Samples, typename EvalFn, typename RefFn>
void check_binary_operation(
    const char* operation,
    double required_min_bits,
    const Samples& samples,
    EvalFn eval,
    RefFn reference)
{
    const auto fltx_inputs = make_binary_inputs<Samples, fltx_type>(samples, [](const value_sample& x) { return make_fltx(x); });
    const auto boost_inputs = make_binary_inputs<Samples, boost_type>(samples, [](const value_sample& x) { return make_boost(x); });

    const auto fltx = check_binary_backend(
        operation,
        "f128",
        required_min_bits,
        true,
        samples,
        fltx_inputs,
        [](const fltx_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto boost = check_binary_backend(
        operation,
        "cpp_double_double",
        required_min_bits,
        false,
        samples,
        boost_inputs,
        [](const boost_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto fltx_bench = benchmark_binary_values(fltx_inputs, eval);
    const auto boost_bench = benchmark_binary_values(boost_inputs, eval);

    compare_quality(operation, fltx, boost, fltx_bench, boost_bench);
}

template <typename Samples, typename EvalFn, typename RefFn>
void check_ternary_operation(
    const char* operation,
    double required_min_bits,
    const Samples& samples,
    EvalFn eval,
    RefFn reference)
{
    const auto fltx_inputs = make_ternary_inputs<Samples, fltx_type>(samples, [](const value_sample& x) { return make_fltx(x); });
    const auto boost_inputs = make_ternary_inputs<Samples, boost_type>(samples, [](const value_sample& x) { return make_boost(x); });

    const auto fltx = check_ternary_backend(
        operation,
        "f128",
        required_min_bits,
        true,
        samples,
        fltx_inputs,
        [](const fltx_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto boost = check_ternary_backend(
        operation,
        "cpp_double_double",
        required_min_bits,
        false,
        samples,
        boost_inputs,
        [](const boost_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto fltx_bench = benchmark_ternary_values(fltx_inputs, eval);
    const auto boost_bench = benchmark_ternary_values(boost_inputs, eval);

    compare_quality(operation, fltx, boost, fltx_bench, boost_bench);
}

template <typename Samples, typename EvalFn, typename RefFn>
void check_unary_int_operation(
    const char* operation,
    double required_min_bits,
    const Samples& samples,
    EvalFn eval,
    RefFn reference)
{
    const auto fltx_inputs = make_unary_int_inputs<Samples, fltx_type>(samples, [](const value_sample& x) { return make_fltx(x); });
    const auto boost_inputs = make_unary_int_inputs<Samples, boost_type>(samples, [](const value_sample& x) { return make_boost(x); });

    const auto fltx = check_unary_int_backend(
        operation,
        "f128",
        required_min_bits,
        true,
        samples,
        fltx_inputs,
        [](const fltx_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto boost = check_unary_int_backend(
        operation,
        "cpp_double_double",
        required_min_bits,
        false,
        samples,
        boost_inputs,
        [](const boost_type& x) { return to_mpfr(x); },
        eval,
        reference);

    const auto fltx_bench = benchmark_unary_int_values(fltx_inputs, eval);
    const auto boost_bench = benchmark_unary_int_values(boost_inputs, eval);

    compare_quality(operation, fltx, boost, fltx_bench, boost_bench);
}

template <typename T>
[[nodiscard]] mpfr_type rebuild_frexp_value(const frexp_value<T>& value)
{
    using boost::multiprecision::ldexp;
    using std::ldexp;
    return ldexp(to_mpfr(value.fraction), value.exponent);
}

template <typename Samples, typename EvalFn>
void check_frexp_operation(
    const char* operation,
    double required_min_bits,
    const Samples& samples,
    EvalFn eval)
{
    const auto fltx_inputs = make_unary_inputs<Samples, fltx_type>(samples, [](const value_sample& x) { return make_fltx(x); });
    const auto boost_inputs = make_unary_inputs<Samples, boost_type>(samples, [](const value_sample& x) { return make_boost(x); });

    const auto fltx = check_frexp_backend(
        operation,
        "f128",
        required_min_bits,
        true,
        samples,
        fltx_inputs,
        [&](const fltx_type& x) { return rebuild_frexp_value(eval(x)); });

    const auto boost = check_frexp_backend(
        operation,
        "cpp_double_double",
        required_min_bits,
        false,
        samples,
        boost_inputs,
        [&](const boost_type& x) { return rebuild_frexp_value(eval(x)); });

    const auto fltx_bench = benchmark_unary_values(fltx_inputs, eval);
    const auto boost_bench = benchmark_unary_values(boost_inputs, eval);

    compare_quality(operation, fltx, boost, fltx_bench, boost_bench);
}

template <typename T>
[[nodiscard]] T call_sqrt(const T& x)
{
    using std::sqrt;
    return sqrt(x);
}

template <typename T>
[[nodiscard]] T call_cbrt(const T& x)
{
    using std::cbrt;
    return cbrt(x);
}

template <typename T>
[[nodiscard]] T call_hypot(const T& x, const T& y)
{
    using std::hypot;
    return hypot(x, y);
}

template <typename T>
[[nodiscard]] T call_sin(const T& x)
{
    using std::sin;
    return sin(x);
}

template <typename T>
[[nodiscard]] T call_cos(const T& x)
{
    using std::cos;
    return cos(x);
}

template <typename T>
[[nodiscard]] T call_tan(const T& x)
{
    using std::tan;
    return tan(x);
}

template <typename T>
[[nodiscard]] T call_atan(const T& x)
{
    using std::atan;
    return atan(x);
}

template <typename T>
[[nodiscard]] T call_atan2(const T& y, const T& x)
{
    using std::atan2;
    return atan2(y, x);
}

template <typename T>
[[nodiscard]] T call_asin(const T& x)
{
    using std::asin;
    return asin(x);
}

template <typename T>
[[nodiscard]] T call_acos(const T& x)
{
    using std::acos;
    return acos(x);
}

template <typename T>
[[nodiscard]] T call_exp(const T& x)
{
    using std::exp;
    return exp(x);
}

template <typename T>
[[nodiscard]] T call_exp2(const T& x)
{
    using std::exp2;
    return exp2(x);
}

template <typename T>
[[nodiscard]] T call_expm1(const T& x)
{
    using std::expm1;
    return expm1(x);
}

template <typename T>
[[nodiscard]] T call_log(const T& x)
{
    using std::log;
    return log(x);
}

template <typename T>
[[nodiscard]] T call_log2(const T& x)
{
    using std::log2;
    return log2(x);
}

template <typename T>
[[nodiscard]] T call_log10(const T& x)
{
    using std::log10;
    return log10(x);
}

template <typename T>
[[nodiscard]] T call_log1p(const T& x)
{
    using std::log1p;
    return log1p(x);
}

template <typename T>
[[nodiscard]] T call_pow(const T& x, const T& y)
{
    using std::pow;
    return pow(x, y);
}

template <typename T>
[[nodiscard]] T call_pow_double(const T& x, double y)
{
    using std::pow;
    return pow(x, y);
}

template <typename T>
[[nodiscard]] T call_sinh(const T& x)
{
    using std::sinh;
    return sinh(x);
}

template <typename T>
[[nodiscard]] T call_cosh(const T& x)
{
    using std::cosh;
    return cosh(x);
}

template <typename T>
[[nodiscard]] T call_tanh(const T& x)
{
    using std::tanh;
    return tanh(x);
}

template <typename T>
[[nodiscard]] T call_asinh(const T& x)
{
    using std::asinh;
    return asinh(x);
}

template <typename T>
[[nodiscard]] T call_acosh(const T& x)
{
    using std::acosh;
    return acosh(x);
}

template <typename T>
[[nodiscard]] T call_atanh(const T& x)
{
    using std::atanh;
    return atanh(x);
}

template <typename T>
[[nodiscard]] T call_fma(const T& x, const T& y, const T& z)
{
    using std::fma;
    return fma(x, y, z);
}

template <typename T>
[[nodiscard]] T call_fabs(const T& x)
{
    using boost::multiprecision::fabs;
    using std::fabs;
    return fabs(x);
}

template <typename T>
[[nodiscard]] T call_floor(const T& x)
{
    using boost::multiprecision::floor;
    using std::floor;
    return floor(x);
}

template <typename T>
[[nodiscard]] T call_ceil(const T& x)
{
    using boost::multiprecision::ceil;
    using std::ceil;
    return ceil(x);
}

template <typename T>
[[nodiscard]] T call_trunc(const T& x)
{
    using boost::multiprecision::trunc;
    using std::trunc;
    return trunc(x);
}

template <typename T>
[[nodiscard]] T call_round(const T& x)
{
    using boost::multiprecision::round;
    using std::round;
    return round(x);
}

template <typename T>
[[nodiscard]] T call_nearbyint(const T& x)
{
    using boost::multiprecision::nearbyint;
    using std::nearbyint;
    return nearbyint(x);
}

template <typename T>
[[nodiscard]] T call_rint(const T& x)
{
    using boost::multiprecision::rint;
    using std::rint;
    return rint(x);
}

template <typename T>
[[nodiscard]] T call_fmod(const T& x, const T& y)
{
    using boost::multiprecision::fmod;
    using std::fmod;
    return fmod(x, y);
}

template <typename T>
[[nodiscard]] T call_remainder(const T& x, const T& y)
{
    using boost::multiprecision::remainder;
    using std::remainder;
    return remainder(x, y);
}

template <typename T>
[[nodiscard]] T call_remquo_value(const T& x, const T& y)
{
    int quo = 0;
    using boost::multiprecision::remquo;
    using std::remquo;
    return remquo(x, y, &quo);
}

template <typename T>
[[nodiscard]] T call_fmin(const T& x, const T& y)
{
    using boost::multiprecision::fmin;
    using std::fmin;
    return fmin(x, y);
}

template <typename T>
[[nodiscard]] T call_fmax(const T& x, const T& y)
{
    using boost::multiprecision::fmax;
    using std::fmax;
    return fmax(x, y);
}

template <typename T>
[[nodiscard]] T call_fdim(const T& x, const T& y)
{
    using boost::multiprecision::fdim;
    using std::fdim;
    return fdim(x, y);
}

template <typename T>
[[nodiscard]] T call_copysign(const T& x, const T& y)
{
    using boost::multiprecision::copysign;
    using std::copysign;
    return copysign(x, y);
}

template <typename T>
[[nodiscard]] T call_ldexp(const T& x, int n)
{
    using boost::multiprecision::ldexp;
    using std::ldexp;
    return ldexp(x, n);
}

template <typename T>
[[nodiscard]] T call_scalbn(const T& x, int n)
{
    using boost::multiprecision::scalbn;
    using std::scalbn;
    return scalbn(x, n);
}

template <typename T>
[[nodiscard]] T call_scalbln(const T& x, int n)
{
    using boost::multiprecision::scalbln;
    using std::scalbln;
    return scalbln(x, static_cast<long>(n));
}

template <typename T>
[[nodiscard]] T call_logb(const T& x)
{
    using boost::multiprecision::logb;
    using std::logb;
    return logb(x);
}

template <typename T>
[[nodiscard]] frexp_value<T> call_frexp(const T& x)
{
    int exponent = 0;
    using boost::multiprecision::frexp;
    using std::frexp;
    return {frexp(x, &exponent), exponent};
}

template <typename T>
[[nodiscard]] T call_modf_fraction(const T& x)
{
    T integer_part{};
    using boost::multiprecision::modf;
    using std::modf;
    return modf(x, &integer_part);
}

template <typename T>
[[nodiscard]] T call_erf(const T& x)
{
    using boost::multiprecision::erf;
    using std::erf;
    return erf(x);
}

template <typename T>
[[nodiscard]] T call_erfc(const T& x)
{
    using boost::multiprecision::erfc;
    using std::erfc;
    return erfc(x);
}

template <typename T>
[[nodiscard]] T call_lgamma(const T& x)
{
    using boost::multiprecision::lgamma;
    using std::lgamma;
    return lgamma(x);
}

template <typename T>
[[nodiscard]] T call_tgamma(const T& x)
{
    using boost::multiprecision::tgamma;
    using std::tgamma;
    return tgamma(x);
}

constexpr double bits_90 = 90.0;
constexpr double bits_80 = 80.0;
constexpr double bits_75 = 75.0;
constexpr double bits_40 = 40.0;

[[nodiscard]] std::vector<binary_sample> make_arithmetic_samples()
{
    std::vector<binary_sample> samples;
    samples.reserve(5 + brute_force_samples);

    samples.push_back({"unit mixed signs", {"x", 1.25, 0x1p-54}, {"y", -0.75, 0x1p-55}});
    samples.push_back({"large plus small", {"x", 0x1.3c0ca428c59f8p+32, 0x1p-22}, {"y", -0x1.f972474538ef3p-20, 0x1p-74}});
    samples.push_back({"near cancellation", {"x", 1.0, 0x1p-54}, {"y", -1.0, 0x1p-55}});
    samples.push_back({"fractional", {"x", -0x1.1f9add3739636p-4, 0x1p-60}, {"y", 0x1.3be76c8b43958p+3, -0x1p-52}});
    samples.push_back({"wide finite", {"x", 0x1.2d6444d013d18p+80, -0x1p+25}, {"y", 0x1.8p-40, 0x1p-96}});

    splitmix64 rng{0x128acca217ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -200, 200);
        double y = 0.0;
        do
        {
            y = signed_log_value(rng, -100, 100);
        } while (y == 0.0);

        samples.push_back({
            "brute force",
            make_runtime_value("x", x, residual_for(x, rng)),
            make_runtime_value("y", y, residual_for(y, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_positive_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(5 + brute_force_samples);

    samples.push_back({"quarter", {"x", 0.25, 0x1p-56}});
    samples.push_back({"near one", {"x", 1.0, 0x1p-54}});
    samples.push_back({"two", {"x", 2.0, 0x1p-55}});
    samples.push_back({"large", {"x", 0x1.3c0ca428c59f8p+24, 0x1p-30}});
    samples.push_back({"tiny normal", {"x", 0x1p-32, 0x1p-86}});

    splitmix64 rng{0x128acc90517ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = positive_log_value(rng, -200, 200);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_trig_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(6 + brute_force_samples);

    samples.push_back({"tiny", {"x", 0x1p-20, 0x1p-74}});
    samples.push_back({"pi / 6", {"x", 0x1.0c152382d7366p-1, 0x1.1a62633145c07p-55}});
    samples.push_back({"pi / 4", {"x", 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55}});
    samples.push_back({"negative moderate", {"x", -2.5, 0x1p-54}});
    samples.push_back({"ten pi plus offset", {"x", 31.75, -0x1p-52}});
    samples.push_back({"wide reduction", {"x", 12345.678901234567, 0x1p-43}});

    splitmix64 rng{0x128acc7e16ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -0x1p20, 0x1p20);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_tan_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(6 + brute_force_samples);

    samples.push_back({"tiny", {"x", 0x1p-20, 0x1p-74}});
    samples.push_back({"pi / 6", {"x", 0x1.0c152382d7366p-1, 0x1.1a62633145c07p-55}});
    samples.push_back({"pi / 4", {"x", 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55}});
    samples.push_back({"negative moderate", {"x", -2.5, 0x1p-54}});
    samples.push_back({"ten pi plus offset", {"x", 31.75, -0x1p-52}});
    samples.push_back({"wide reduction", {"x", 12345.678901234567, 0x1p-43}});

    const auto fixed_sample_count = samples.size();

    splitmix64 rng{0x128acc7a9ull};
    while (samples.size() < fixed_sample_count + brute_force_samples)
    {
        const double x = uniform_value(rng, -0x1p20, 0x1p20);
        if (std::fabs(std::cos(x)) < 0x1p-20)
            continue;

        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_exp_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(5 + brute_force_samples);

    samples.push_back({"negative ten", {"x", -10.0, 0x1p-50}});
    samples.push_back({"near zero", {"x", 0.0, 0x1p-60}});
    samples.push_back({"quarter", {"x", 0.25, 0x1p-56}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"five", {"x", 5.0, 0x1p-50}});

    splitmix64 rng{0x128acce901ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -700.0, 700.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<binary_sample> make_pow_samples()
{
    std::vector<binary_sample> samples;
    samples.reserve(5 + brute_force_samples);

    samples.push_back({"square root", {"x", 2.0, 0x1p-55}, {"y", 0.5, 0x1p-56}});
    samples.push_back({"fractional base", {"x", 0.75, 0x1p-56}, {"y", 3.25, -0x1p-54}});
    samples.push_back({"near one", {"x", 1.0, 0x1p-53}, {"y", 512.0, 0x1p-45}});
    samples.push_back({"moderate", {"x", 8.5, -0x1p-53}, {"y", -1.25, 0x1p-55}});
    samples.push_back({"large base", {"x", 1024.0, 0x1p-42}, {"y", 1.75, -0x1p-55}});

    splitmix64 rng{0x128acc90dull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = positive_log_value(rng, -20, 20);
        const double y = uniform_value(rng, -8.0, 8.0);
        samples.push_back({
            "brute force",
            make_runtime_value("x", x, residual_for(x, rng)),
            make_runtime_value("y", y, residual_for(y, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<ternary_sample> make_fma_samples()
{
    std::vector<ternary_sample> samples;
    samples.reserve(5 + brute_force_samples);

    samples.push_back({"unit positive", {"x", 1.0, 0x1p-54}, {"y", 1.0, -0x1p-55}, {"z", 0.5, 0x1p-56}});
    samples.push_back({"mixed signs", {"x", -1.25, 0x1p-54}, {"y", 0.75, 0x1p-55}, {"z", -0.5, -0x1p-56}});
    samples.push_back({"large product", {"x", 0x1.3c0ca428c59f8p+24, 0x1p-30}, {"y", -0x1.8p+12, 0x1p-42}, {"z", 0x1.2p+20, -0x1p-34}});
    samples.push_back({"tiny product", {"x", 0x1.2p-80, 0x1p-136}, {"y", -0x1.6p-40, 0x1p-96}, {"z", 0x1p-110, -0x1p-166}});
    samples.push_back({"product offset", {"x", 0x1.0000002p+20, 0x1p-34}, {"y", 0x1.fffffep-20, -0x1p-74}, {"z", 0.25, 0x1p-56}});

    splitmix64 rng{0x128accf0a1ull};
    while (samples.size() < 5 + brute_force_samples)
    {
        const double x = signed_log_value(rng, -40, 40);
        const double y = signed_log_value(rng, -40, 40);
        const double z = signed_log_value(rng, -80, 80);
        const double product = x * y;
        const double result = product + z;
        const double scale = std::max(std::fabs(product), std::fabs(z));
        if (std::fabs(result) < std::ldexp(scale, -20))
            continue;

        samples.push_back({
            "brute force",
            make_runtime_value("x", x, residual_for(x, rng)),
            make_runtime_value("y", y, residual_for(y, rng)),
            make_runtime_value("z", z, residual_for(z, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_cbrt_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"large negative", {"x", -1e9, 0x1p-24}});
    samples.push_back({"negative eight", {"x", -8.0, 0x1p-52}});
    samples.push_back({"negative fractional", {"x", -0.125, 0x1p-58}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive fractional", {"x", 0.125, 0x1p-58}});
    samples.push_back({"eight", {"x", 8.0, -0x1p-52}});
    samples.push_back({"twenty seven", {"x", 27.0, 0x1p-48}});
    samples.push_back({"large positive", {"x", 1e9, -0x1p-24}});

    splitmix64 rng{0x128accc0b7ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -200, 200);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<binary_sample> make_hypot_samples()
{
    std::vector<binary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"zero", {"x", 0.0, 0.0}, {"y", 0.0, 0.0}});
    samples.push_back({"three four", {"x", 3.0, 0x1p-54}, {"y", 4.0, -0x1p-54}});
    samples.push_back({"signed three four", {"x", -3.0, 0x1p-54}, {"y", 4.0, -0x1p-54}});
    samples.push_back({"tiny pair", {"x", 1e-20, 0x1p-120}, {"y", 3e-20, -0x1p-120}});
    samples.push_back({"large pair", {"x", 1e20, 0x1p+12}, {"y", 3e20, -0x1p+12}});
    samples.push_back({"mixed scale", {"x", 1e20, 0x1p+12}, {"y", 1e-20, 0x1p-120}});
    samples.push_back({"moderate", {"x", 123.456, 0x1p-48}, {"y", 789.25, -0x1p-44}});
    samples.push_back({"fractional", {"x", 0.125, 0x1p-58}, {"y", 0.5, -0x1p-58}});

    splitmix64 rng{0x128acc1707ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -100, 100);
        const double y = signed_log_value(rng, -100, 100);
        samples.push_back({
            "brute force",
            make_runtime_value("x", x, residual_for(x, rng)),
            make_runtime_value("y", y, residual_for(y, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_exp2_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(7 + brute_force_samples);

    samples.push_back({"negative thousand", {"x", -1000.0, 0x1p-42}});
    samples.push_back({"negative ten", {"x", -10.0, 0x1p-50}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"ten", {"x", 10.0, -0x1p-50}});
    samples.push_back({"thousand", {"x", 1000.0, -0x1p-42}});

    splitmix64 rng{0x128acce2e2ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -1000.0, 1000.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_expm1_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(9 + brute_force_samples);

    samples.push_back({"negative twenty", {"x", -20.0, 0x1p-48}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"negative tiny", {"x", -1e-10, 0x1p-88}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive tiny", {"x", 1e-10, -0x1p-88}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"twenty", {"x", 20.0, -0x1p-48}});

    splitmix64 rng{0x128acce111ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -20.0, 20.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_log1p_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(9 + brute_force_samples);

    samples.push_back({"near negative one", {"x", -0.875, 0x1p-58}});
    samples.push_back({"negative half", {"x", -0.5, 0x1p-58}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"negative tiny", {"x", -1e-10, 0x1p-88}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive tiny", {"x", 1e-10, -0x1p-88}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"ten", {"x", 10.0, 0x1p-50}});

    splitmix64 rng{0x128acc1011ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -0.95, 20.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_atan_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"large negative", {"x", -100.0, 0x1p-46}});
    samples.push_back({"negative ten", {"x", -10.0, 0x1p-50}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"ten", {"x", 10.0, -0x1p-50}});
    samples.push_back({"large positive", {"x", 100.0, -0x1p-46}});

    splitmix64 rng{0x128acca7a1ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -20, 20);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<binary_sample> make_atan2_samples()
{
    std::vector<binary_sample> samples;
    samples.reserve(10 + brute_force_samples);

    samples.push_back({"one one", {"y", 1.0, 0x1p-54}, {"x", 1.0, -0x1p-54}});
    samples.push_back({"one negative one", {"y", 1.0, 0x1p-54}, {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative one one", {"y", -1.0, -0x1p-54}, {"x", 1.0, -0x1p-54}});
    samples.push_back({"negative one negative one", {"y", -1.0, -0x1p-54}, {"x", -1.0, 0x1p-54}});
    samples.push_back({"shallow", {"y", 0.125, 0x1p-58}, {"x", 10.0, -0x1p-50}});
    samples.push_back({"steep", {"y", 10.0, 0x1p-50}, {"x", 0.125, -0x1p-58}});
    samples.push_back({"negative shallow", {"y", -0.125, -0x1p-58}, {"x", 10.0, -0x1p-50}});
    samples.push_back({"quadrant two steep", {"y", 10.0, 0x1p-50}, {"x", -0.125, 0x1p-58}});
    samples.push_back({"moderate", {"y", 123.456, 0x1p-48}, {"x", 789.25, -0x1p-44}});
    samples.push_back({"negative moderate", {"y", -123.456, -0x1p-48}, {"x", 789.25, -0x1p-44}});

    splitmix64 rng{0x128acca722ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        double y = signed_log_value(rng, -20, 20);
        double x = signed_log_value(rng, -20, 20);
        if (x == 0.0 && y == 0.0)
            x = 1.0;

        samples.push_back({
            "brute force",
            make_runtime_value("y", y, residual_for(y, rng)),
            make_runtime_value("x", x, residual_for(x, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_unit_interval_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(9 + brute_force_samples);

    samples.push_back({"negative one", {"x", -1.0, 0.0}});
    samples.push_back({"negative seven eighths", {"x", -0.875, 0x1p-58}});
    samples.push_back({"negative half", {"x", -0.5, 0x1p-58}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"half", {"x", 0.5, -0x1p-58}});
    samples.push_back({"seven eighths", {"x", 0.875, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, 0.0}});

    splitmix64 rng{0x128acc1eafull};
    while (samples.size() < 9 + brute_force_samples)
    {
        const double x = uniform_value(rng, -0.999999, 0.999999);
        const auto sample = make_runtime_value("x", x, residual_for(x, rng));
        if (std::fabs(sample.hi + sample.lo) > 1.0)
            continue;

        samples.push_back({"brute force", sample});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_hyperbolic_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"negative four", {"x", -4.0, 0x1p-52}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"four", {"x", 4.0, -0x1p-52}});
    samples.push_back({"eight", {"x", 8.0, 0x1p-50}});

    splitmix64 rng{0x128accb1b0ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -8.0, 8.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_asinh_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(7 + brute_force_samples);

    samples.push_back({"negative eight", {"x", -8.0, 0x1p-50}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"eight", {"x", 8.0, -0x1p-50}});

    splitmix64 rng{0x128acca511ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -8.0, 8.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_acosh_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(7 + brute_force_samples);

    samples.push_back({"one", {"x", 1.0, 0.0}});
    samples.push_back({"near one", {"x", 1.0, 0x1p-52}});
    samples.push_back({"nine eighths", {"x", 1.125, 0x1p-56}});
    samples.push_back({"one half", {"x", 1.5, -0x1p-54}});
    samples.push_back({"two", {"x", 2.0, 0x1p-55}});
    samples.push_back({"eight", {"x", 8.0, -0x1p-50}});
    samples.push_back({"sixteen", {"x", 16.0, 0x1p-48}});

    splitmix64 rng{0x128acca051ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, 1.0, 16.0);
        samples.push_back({"brute force", make_runtime_value("x", x, positive_residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_atanh_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"negative near one", {"x", -0.95, 0x1p-58}});
    samples.push_back({"negative half", {"x", -0.5, 0x1p-58}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"negative tiny", {"x", -1e-10, 0x1p-88}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive tiny", {"x", 1e-10, -0x1p-88}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"positive near one", {"x", 0.95, -0x1p-58}});

    splitmix64 rng{0x128acca7a9ull};
    while (samples.size() < 8 + brute_force_samples)
    {
        const double x = uniform_value(rng, -0.95, 0.95);
        const auto sample = make_runtime_value("x", x, residual_for(x, rng));
        if (std::fabs(sample.hi + sample.lo) >= 1.0)
            continue;

        samples.push_back({"brute force", sample});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_signed_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(9 + brute_force_samples);

    samples.push_back({"negative large", {"x", -1e20, 0x1p+12}});
    samples.push_back({"negative ten", {"x", -10.0, 0x1p-50}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative tiny", {"x", -1e-20, 0x1p-120}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive tiny", {"x", 1e-20, -0x1p-120}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"ten", {"x", 10.0, -0x1p-50}});
    samples.push_back({"positive large", {"x", 1e20, -0x1p+12}});

    splitmix64 rng{0x128acc51a5ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -120, 120);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_rounding_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(12 + brute_force_samples);

    samples.push_back({"negative large fraction", {"x", -123456.75, 0x1p-48}});
    samples.push_back({"negative ten", {"x", -10.25, 0x1p-54}});
    samples.push_back({"negative one", {"x", -1.75, -0x1p-54}});
    samples.push_back({"negative below half", {"x", -0.49, 0x1p-58}});
    samples.push_back({"negative above half", {"x", -0.51, -0x1p-58}});
    samples.push_back({"negative tiny", {"x", -0x1p-20, 0x1p-80}});
    samples.push_back({"positive tiny", {"x", 0x1p-20, -0x1p-80}});
    samples.push_back({"below half", {"x", 0.49, 0x1p-58}});
    samples.push_back({"above half", {"x", 0.51, -0x1p-58}});
    samples.push_back({"one", {"x", 1.75, 0x1p-54}});
    samples.push_back({"ten", {"x", 10.25, -0x1p-54}});
    samples.push_back({"large fraction", {"x", 123456.75, -0x1p-48}});

    splitmix64 rng{0x128acc9080ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -1e6, 1e6);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<binary_sample> make_remainder_samples()
{
    std::vector<binary_sample> samples;
    samples.reserve(12 + brute_force_samples);

    samples.push_back({"positive", {"x", 5.25, 0x1p-54}, {"y", 2.0, 0.0}});
    samples.push_back({"negative dividend", {"x", -5.25, -0x1p-54}, {"y", 2.0, 0.0}});
    samples.push_back({"negative divisor", {"x", 5.25, 0x1p-54}, {"y", -2.0, 0.0}});
    samples.push_back({"both negative", {"x", -5.25, -0x1p-54}, {"y", -2.0, 0.0}});
    samples.push_back({"decimal divisor", {"x", 1.0, 0x1p-54}, {"y", 0.1, 0x1p-58}});
    samples.push_back({"large half", {"x", 123456789.125, 0x1p-28}, {"y", 0.5, 0.0}});
    samples.push_back({"small", {"x", 1e-20, 0x1p-120}, {"y", 3e-21, -0x1p-124}});
    samples.push_back({"large", {"x", -1e20, 0x1p+12}, {"y", 3.125, -0x1p-50}});
    samples.push_back({"wide quotient", {"x", 0x1.4591bb3c5afdep+65, 0x1p+8}, {"y", 293.0, 0x1p-44}});
    samples.push_back({"fractional quotient", {"x", -123.456, -0x1p-48}, {"y", 0.5, 0.0}});
    samples.push_back({"x less than y", {"x", 0.125, 0x1p-58}, {"y", 10.0, -0x1p-50}});
    samples.push_back({"near divisor", {"x", 9.999, 0x1p-52}, {"y", 10.0, -0x1p-50}});

    splitmix64 rng{0x128accf00dull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -60, 60);
        double y = 0.0;
        do
        {
            y = signed_log_value(rng, -30, 30);
        } while (y == 0.0);

        samples.push_back({
            "brute force",
            make_runtime_value("x", x, residual_for(x, rng)),
            make_runtime_value("y", y, residual_for(y, rng))
        });
    }

    return samples;
}

[[nodiscard]] std::vector<unary_int_sample> make_scaling_samples()
{
    std::vector<unary_int_sample> samples;
    samples.reserve(10 + brute_force_samples);

    samples.push_back({"zero", {"x", 0.0, 0.0}, 32});
    samples.push_back({"one up", {"x", 1.0, 0x1p-54}, 1});
    samples.push_back({"one down", {"x", 1.0, -0x1p-54}, -1});
    samples.push_back({"negative up", {"x", -1.5, 0x1p-54}, 10});
    samples.push_back({"negative down", {"x", -1.5, -0x1p-54}, -10});
    samples.push_back({"pi down", {"x", 3.141592653589793, 0x1p-52}, -20});
    samples.push_back({"tiny up", {"x", 1e-20, 0x1p-120}, 80});
    samples.push_back({"large down", {"x", 1e20, -0x1p+12}, -80});
    samples.push_back({"moderate up", {"x", 123456789.125, 0x1p-28}, 37});
    samples.push_back({"moderate down", {"x", -123456789.125, -0x1p-28}, -37});

    splitmix64 rng{0x128acc5ca1eull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = signed_log_value(rng, -100, 100);
        const int n = -120 + static_cast<int>(rng.unit() * 241.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng)), n});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_erf_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(9 + brute_force_samples);

    samples.push_back({"negative four", {"x", -4.0, 0x1p-52}});
    samples.push_back({"negative one", {"x", -1.0, 0x1p-54}});
    samples.push_back({"negative eighth", {"x", -0.125, 0x1p-58}});
    samples.push_back({"negative tiny", {"x", -1e-10, 0x1p-88}});
    samples.push_back({"zero", {"x", 0.0, 0.0}});
    samples.push_back({"positive tiny", {"x", 1e-10, -0x1p-88}});
    samples.push_back({"eighth", {"x", 0.125, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, -0x1p-54}});
    samples.push_back({"four", {"x", 4.0, -0x1p-52}});

    splitmix64 rng{0x128accecfull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, -6.0, 6.0);
        samples.push_back({"brute force", make_runtime_value("x", x, residual_for(x, rng))});
    }

    return samples;
}

[[nodiscard]] std::vector<unary_sample> make_gamma_samples()
{
    std::vector<unary_sample> samples;
    samples.reserve(8 + brute_force_samples);

    samples.push_back({"eighth", {"x", 0.125, 0x1p-58}});
    samples.push_back({"half", {"x", 0.5, 0x1p-58}});
    samples.push_back({"three quarters", {"x", 0.75, -0x1p-58}});
    samples.push_back({"one", {"x", 1.0, 0.0}});
    samples.push_back({"one half", {"x", 1.5, -0x1p-54}});
    samples.push_back({"two half", {"x", 2.5, 0x1p-54}});
    samples.push_back({"five half", {"x", 5.5, -0x1p-52}});
    samples.push_back({"ten", {"x", 10.0, 0x1p-50}});

    splitmix64 rng{0x128acc6a99ull};
    for (std::size_t i = 0; i < brute_force_samples; ++i)
    {
        const double x = uniform_value(rng, 0.125, 20.0);
        samples.push_back({"brute force", make_runtime_value("x", x, positive_residual_for(x, rng))});
    }

    return samples;
}

} // namespace

TEST_CASE("f128 arithmetic special values", "[accuracy][f128][arithmetic][special]")
{
    const fltx_type nan = std::numeric_limits<fltx_type>::quiet_NaN();
    const fltx_type inf = std::numeric_limits<fltx_type>::infinity();
    const fltx_type neg_inf = -inf;
    const fltx_type one{1.0};
    const fltx_type two{2.0};
    const fltx_type zero{0.0, 0.0};
    const fltx_type neg_zero{-0.0, 0.0};
    const fltx_type max = std::numeric_limits<fltx_type>::max();
    const fltx_type lowest = std::numeric_limits<fltx_type>::lowest();

    require_exact_value("arith.inf_add", inf + one, inf);
    require_exact_value("arith.neg_inf_add", neg_inf + one, neg_inf);
    require_exact_value("arith.inf_add_inf", inf + inf, inf);
    require_exact_value("arith.neg_inf_add_neg_inf", neg_inf + neg_inf, neg_inf);
    REQUIRE(bl::isnan(nan + one));
    REQUIRE(bl::isnan(one + nan));
    REQUIRE(bl::isnan(nan + nan));
    REQUIRE(bl::isnan(inf + neg_inf));

    REQUIRE(bl::isnan(nan - one));
    REQUIRE(bl::isnan(one - nan));
    REQUIRE(bl::isnan(nan - nan));
    REQUIRE(bl::isnan(inf - inf));
    require_exact_value("arith.inf_sub", inf - one, inf);
    require_exact_value("arith.sub_inf", one - inf, neg_inf);
    require_exact_value("arith.inf_sub_neg_inf", inf - neg_inf, inf);
    require_exact_value("arith.neg_inf_sub_inf", neg_inf - inf, neg_inf);

    require_exact_value("arith.inf_mul", inf * two, inf);
    require_exact_value("arith.inf_mul_neg", inf * -two, neg_inf);
    require_exact_value("arith.neg_inf_mul", neg_inf * two, neg_inf);
    require_exact_value("arith.neg_inf_mul_neg", neg_inf * -two, inf);
    REQUIRE(bl::isnan(inf * zero));
    REQUIRE(bl::isnan(zero * inf));
    REQUIRE(bl::isnan(nan * one));
    REQUIRE(bl::isnan(one * nan));
    REQUIRE(bl::isnan(nan * nan));

    require_exact_value("arith.div_by_zero", one / zero, inf);
    require_exact_value("arith.neg_div_by_zero", -one / zero, neg_inf);
    require_exact_value("arith.neg_div_by_neg_zero", -one / neg_zero, inf);
    REQUIRE(bl::isnan(nan / one));
    REQUIRE(bl::isnan(one / nan));
    REQUIRE(bl::isnan(nan / nan));
    REQUIRE(bl::isnan(zero / zero));
    REQUIRE(bl::isnan(inf / inf));
    require_exact_value("arith.inf_div", inf / two, inf);

    const fltx_type pos_zero = two / inf;
    REQUIRE(bl::iszero(pos_zero));
    REQUIRE(!bl::signbit(pos_zero));

    const fltx_type signed_zero = -two / inf;
    REQUIRE(bl::iszero(signed_zero));
    REQUIRE(bl::signbit(signed_zero));

    const fltx_type neg_zero_div_finite = neg_zero / two;
    REQUIRE(bl::iszero(neg_zero_div_finite));
    REQUIRE(bl::signbit(neg_zero_div_finite));

    const fltx_type zero_div_neg_finite = zero / -two;
    REQUIRE(bl::iszero(zero_div_neg_finite));
    REQUIRE(bl::signbit(zero_div_neg_finite));

    const fltx_type neg_zero_div_neg_finite = neg_zero / -two;
    REQUIRE(bl::iszero(neg_zero_div_neg_finite));
    REQUIRE(!bl::signbit(neg_zero_div_neg_finite));

    require_exact_value("arith.scalar_add_inf", one + std::numeric_limits<double>::infinity(), inf);
    require_exact_value("arith.scalar_sub_inf", one - std::numeric_limits<double>::infinity(), neg_inf);
    require_exact_value("arith.scalar_mul_inf", two * std::numeric_limits<double>::infinity(), inf);
    require_exact_value("arith.scalar_div_zero", one / 0.0, inf);
    REQUIRE(bl::isnan(one + std::numeric_limits<double>::quiet_NaN()));
    REQUIRE(bl::isnan(one - std::numeric_limits<double>::quiet_NaN()));
    REQUIRE(bl::isnan(one * std::numeric_limits<double>::quiet_NaN()));
    REQUIRE(bl::isnan(one / std::numeric_limits<double>::quiet_NaN()));

    require_exact_value("arith.add_overflow", max + max, inf);
    require_exact_value("arith.sub_overflow", lowest - max, neg_inf);
    require_exact_value("arith.mul_overflow", max * two, inf);
    require_exact_value("arith.neg_mul_overflow", lowest * two, neg_inf);
    require_exact_value("arith.div_overflow", max / fltx_type{0.5}, inf);
    require_exact_value("arith.neg_div_overflow", lowest / fltx_type{0.5}, neg_inf);
}

TEST_CASE("f128 tier 1 arithmetic accuracy", "[accuracy][f128][arithmetic]")
{
    const auto samples = make_arithmetic_samples();

    check_binary_operation("add", bits_90, samples, [](const auto& x, const auto& y) { return x + y; },
        [](const auto& x, const auto& y) { return x + y; });

    check_binary_operation("subtract", bits_90, samples, [](const auto& x, const auto& y) { return x - y; },
        [](const auto& x, const auto& y) { return x - y; });

    check_binary_operation("multiply", bits_90, samples, [](const auto& x, const auto& y) { return x * y; },
        [](const auto& x, const auto& y) { return x * y; });

    check_binary_operation("divide", bits_90, samples, [](const auto& x, const auto& y) { return x / y; },
        [](const auto& x, const auto& y) { return x / y; });
}

TEST_CASE("f128 tier 1 elementary accuracy", "[accuracy][f128][math]")
{
    const auto positive = make_positive_samples();
    const auto trig = make_trig_samples();
    const auto tan = make_tan_samples();
    const auto exp = make_exp_samples();
    const auto pow = make_pow_samples();

    check_unary_operation("sqrt", bits_90, positive, [](const auto& x) { return call_sqrt(x); },
        [](const auto& x) { return call_sqrt(x); });

    check_unary_operation("sin", bits_80, trig, [](const auto& x) { return call_sin(x); },
        [](const auto& x) { return call_sin(x); });

    check_unary_operation("cos", bits_80, trig, [](const auto& x) { return call_cos(x); },
        [](const auto& x) { return call_cos(x); });

    check_unary_operation("tan", bits_75, tan, [](const auto& x) { return call_tan(x); },
        [](const auto& x) { return call_tan(x); });

    check_unary_operation("exp", bits_80, exp, [](const auto& x) { return call_exp(x); },
        [](const auto& x) { return call_exp(x); });

    check_unary_operation("log", bits_80, positive, [](const auto& x) { return call_log(x); },
        [](const auto& x) { return call_log(x); });

    check_binary_operation("pow", bits_80, pow, [](const auto& x, const auto& y) { return call_pow(x, y); },
        [](const auto& x, const auto& y) { return call_pow(x, y); });

    check_binary_operation("pow.double", bits_80, pow, [](const auto& x, const auto& y) { return call_pow_double(x, static_cast<double>(y)); },
        [](const auto& x, const auto& y) { return call_pow_double(x, static_cast<double>(y)); });
}

TEST_CASE("f128 tier 2 elementary accuracy", "[accuracy][f128][math]")
{
    const auto positive = make_positive_samples();
    const auto fma = make_fma_samples();
    const auto cbrt = make_cbrt_samples();
    const auto hypot = make_hypot_samples();
    const auto exp2 = make_exp2_samples();
    const auto expm1 = make_expm1_samples();
    const auto log1p = make_log1p_samples();
    const auto atan = make_atan_samples();
    const auto atan2 = make_atan2_samples();
    const auto unit = make_unit_interval_samples();
    const auto hyperbolic = make_hyperbolic_samples();
    const auto asinh = make_asinh_samples();
    const auto acosh = make_acosh_samples();
    const auto atanh = make_atanh_samples();

    check_ternary_operation("fma", bits_90, fma, [](const auto& x, const auto& y, const auto& z) { return call_fma(x, y, z); },
        [](const auto& x, const auto& y, const auto& z) { return x * y + z; });

    check_unary_operation("cbrt", bits_90, cbrt, [](const auto& x) { return call_cbrt(x); },
        [](const auto& x) { return call_cbrt(x); });

    check_binary_operation("hypot", bits_90, hypot, [](const auto& x, const auto& y) { return call_hypot(x, y); },
        [](const auto& x, const auto& y) { return call_hypot(x, y); });

    check_unary_operation("exp2", bits_80, exp2, [](const auto& x) { return call_exp2(x); },
        [](const auto& x) { return call_exp2(x); });

    check_unary_operation("expm1", bits_80, expm1, [](const auto& x) { return call_expm1(x); },
        [](const auto& x) { return call_expm1(x); });

    check_unary_operation("log2", bits_80, positive, [](const auto& x) { return call_log2(x); },
        [](const auto& x) { return call_log2(x); });

    check_unary_operation("log10", bits_80, positive, [](const auto& x) { return call_log10(x); },
        [](const auto& x) { return call_log10(x); });

    check_unary_operation("log1p", bits_80, log1p, [](const auto& x) { return call_log1p(x); },
        [](const auto& x) { return call_log1p(x); });

    check_unary_operation("atan", bits_80, atan, [](const auto& x) { return call_atan(x); },
        [](const auto& x) { return call_atan(x); });

    check_binary_operation("atan2", bits_80, atan2, [](const auto& y, const auto& x) { return call_atan2(y, x); },
        [](const auto& y, const auto& x) { return call_atan2(y, x); });

    check_unary_operation("asin", bits_80, unit, [](const auto& x) { return call_asin(x); },
        [](const auto& x) { return call_asin(x); });

    check_unary_operation("acos", bits_80, unit, [](const auto& x) { return call_acos(x); },
        [](const auto& x) { return call_acos(x); });

    check_unary_operation("sinh", bits_80, hyperbolic, [](const auto& x) { return call_sinh(x); },
        [](const auto& x) { return call_sinh(x); });

    check_unary_operation("cosh", bits_80, hyperbolic, [](const auto& x) { return call_cosh(x); },
        [](const auto& x) { return call_cosh(x); });

    check_unary_operation("tanh", bits_80, hyperbolic, [](const auto& x) { return call_tanh(x); },
        [](const auto& x) { return call_tanh(x); });

    check_unary_operation("asinh", bits_80, asinh, [](const auto& x) { return call_asinh(x); },
        [](const auto& x) { return call_asinh(x); });

    check_unary_operation("acosh", bits_80, acosh, [](const auto& x) { return call_acosh(x); },
        [](const auto& x) { return call_acosh(x); });

    check_unary_operation("atanh", bits_80, atanh, [](const auto& x) { return call_atanh(x); },
        [](const auto& x) { return call_atanh(x); });
}

TEST_CASE("f128 tier 3 supporting math accuracy", "[accuracy][f128][math]")
{
    const auto signed_samples = make_signed_samples();
    const auto rounding = make_rounding_samples();
    const auto remainder = make_remainder_samples();
    const auto scaling = make_scaling_samples();
    const auto positive = make_positive_samples();
    const auto arithmetic = make_arithmetic_samples();
    const auto erf = make_erf_samples();
    const auto gamma = make_gamma_samples();

    check_unary_operation("fabs", bits_90, signed_samples, [](const auto& x) { return call_fabs(x); },
        [](const auto& x) { return call_fabs(x); });

    check_unary_operation("floor", bits_90, rounding, [](const auto& x) { return call_floor(x); },
        [](const auto& x) { return call_floor(x); });

    check_unary_operation("ceil", bits_90, rounding, [](const auto& x) { return call_ceil(x); },
        [](const auto& x) { return call_ceil(x); });

    check_unary_operation("trunc", bits_90, rounding, [](const auto& x) { return call_trunc(x); },
        [](const auto& x) { return call_trunc(x); });

    check_unary_operation("round", bits_90, rounding, [](const auto& x) { return call_round(x); },
        [](const auto& x) { return call_round(x); });

    check_unary_operation("nearbyint", bits_90, rounding, [](const auto& x) { return call_nearbyint(x); },
        [](const auto& x) { return call_nearbyint(x); });

    check_unary_operation("rint", bits_90, rounding, [](const auto& x) { return call_rint(x); },
        [](const auto& x) { return call_rint(x); });

    check_binary_operation("fmin", bits_90, arithmetic, [](const auto& x, const auto& y) { return call_fmin(x, y); },
        [](const auto& x, const auto& y) { return call_fmin(x, y); });

    check_binary_operation("fmax", bits_90, arithmetic, [](const auto& x, const auto& y) { return call_fmax(x, y); },
        [](const auto& x, const auto& y) { return call_fmax(x, y); });

    check_binary_operation("fdim", bits_90, arithmetic, [](const auto& x, const auto& y) { return call_fdim(x, y); },
        [](const auto& x, const auto& y) { return call_fdim(x, y); });

    check_binary_operation("copysign", bits_90, arithmetic, [](const auto& x, const auto& y) { return call_copysign(x, y); },
        [](const auto& x, const auto& y) { return call_copysign(x, y); });

    check_binary_operation("fmod", bits_80, remainder, [](const auto& x, const auto& y) { return call_fmod(x, y); },
        [](const auto& x, const auto& y) { return call_fmod(x, y); });

    check_binary_operation("remainder", bits_80, remainder, [](const auto& x, const auto& y) { return call_remainder(x, y); },
        [](const auto& x, const auto& y) { return call_remainder(x, y); });

    check_binary_operation("remquo", bits_40, remainder, [](const auto& x, const auto& y) { return call_remquo_value(x, y); },
        [](const auto& x, const auto& y) { return call_remainder(x, y); });

    check_unary_int_operation("ldexp", bits_90, scaling, [](const auto& x, int n) { return call_ldexp(x, n); },
        [](const auto& x, int n) { return call_ldexp(x, n); });

    check_unary_int_operation("scalbn", bits_90, scaling, [](const auto& x, int n) { return call_scalbn(x, n); },
        [](const auto& x, int n) { return call_scalbn(x, n); });

    check_unary_int_operation("scalbln", bits_90, scaling, [](const auto& x, int n) { return call_scalbln(x, n); },
        [](const auto& x, int n) { return call_scalbln(x, n); });

    check_unary_operation("logb", bits_90, positive, [](const auto& x) { return call_logb(x); },
        [](const auto& x) { return call_logb(x); });

    check_frexp_operation("frexp", bits_90, signed_samples, [](const auto& x) { return call_frexp(x); });

    check_unary_operation("modf", bits_90, rounding, [](const auto& x) { return call_modf_fraction(x); },
        [](const auto& x) { return call_modf_fraction(x); });

    check_unary_operation("erf", bits_80, erf, [](const auto& x) { return call_erf(x); },
        [](const auto& x) { return call_erf(x); });

    check_unary_operation("erfc", bits_80, erf, [](const auto& x) { return call_erfc(x); },
        [](const auto& x) { return call_erfc(x); });

    check_unary_operation("lgamma", bits_80, gamma, [](const auto& x) { return call_lgamma(x); },
        [](const auto& x) { return call_lgamma(x); });

    check_unary_operation("tgamma", bits_80, gamma, [](const auto& x) { return call_tgamma(x); },
        [](const auto& x) { return call_tgamma(x); });
}
