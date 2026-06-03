#ifndef FLTX_TESTS_METRICS_F128_PRIMARY_INCLUDED
#define FLTX_TESTS_METRICS_F128_PRIMARY_INCLUDED

#include <catch2/catch_test_macros.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/next.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fltx/f128_math.h>
#include <fltx/random.h>

#include "metrics_config.h"
#include "metrics_records.h"
#include "metrics_domain_samples.h"
#include "metrics_qdpp.h"
#include "metrics_reference.h"
#include "metrics_samples.h"
#include "metrics_std_special_oracle.h"

namespace bl::test::metrics::f128_primary
{
    using references = reference_types<bl::f128>;
    using fltx_type = references::fltx_type;
    using perfect_ref = references::perfect_ref;
    using competitor_ref = references::competitor_ref;
    using extra_competitor_ref = references::extra_competitor_ref;

    constexpr domain_id primary_domain{ "primary", domain_role::primary };
    constexpr double competitor_accuracy_slack_bits = 16.0;

    constexpr std::size_t random_sample_count = config::f128_primary_random_sample_count;
    constexpr std::size_t benchmark_min_iterations = config::f128_primary_benchmark_min_iterations;

    constexpr double bits_90 = 90.0;
    constexpr double bits_80 = 80.0;
    constexpr double bits_75 = 75.0;
    constexpr double bits_40 = 40.0;
    constexpr double domain_ideal_bits = 106.0;

    constexpr std::size_t default_domain_random_sample_count = config::primary_domain_random_sample_count;

    [[nodiscard]] inline std::size_t domain_random_sample_count() noexcept
    {
        return configured_domain_random_sample_count(default_domain_random_sample_count);
    }

    class sample_rng
    {
    public:
        BL_FORCE_INLINE constexpr explicit sample_rng(std::uint64_t seed) noexcept
            : engine(seed)
        {
        }

        [[nodiscard]] BL_FORCE_INLINE fltx_type unit() noexcept
        {
            bl::uniform_real_distribution<fltx_type> distribution{ fltx_type{ 0.0 }, fltx_type{ 1.0 } };
            return distribution(engine);
        }

        [[nodiscard]] BL_FORCE_INLINE double sign() noexcept
        {
            bl::uniform_int_distribution<int> distribution{ 0, 1 };
            return distribution(engine) != 0 ? 1.0 : -1.0;
        }

        [[nodiscard]] BL_FORCE_INLINE int integer(int min, int max) noexcept
        {
            bl::uniform_int_distribution<int> distribution{ min, max };
            return distribution(engine);
        }

        [[nodiscard]] BL_FORCE_INLINE fltx_type uniform(double min, double max) noexcept
        {
            bl::uniform_real_distribution<fltx_type> distribution{ fltx_type{ min }, fltx_type{ max } };
            return distribution(engine);
        }

    private:
        bl::mt19937_64 engine;
    };

    [[nodiscard]] inline value_sample make_runtime_value(std::string_view label, double hi, double lo = 0.0) noexcept
    {
        const auto value = bl::detail::_f128::renorm(hi, lo);
        return { label, value.hi, value.lo };
    }

    [[nodiscard]] inline value_sample make_runtime_value(std::string_view label, const fltx_type& value) noexcept
    {
        return { label, value.hi, value.lo };
    }

    [[nodiscard]] inline value_sample make_runtime_value(
        std::string_view label,
        const fltx_type& value,
        const fltx_type& residual) noexcept
    {
        return make_runtime_value(label, value + residual);
    }

    [[nodiscard]] inline fltx_type uniform_value(sample_rng& rng, double min, double max) noexcept
    {
        return rng.uniform(min, max);
    }

    [[nodiscard]] inline fltx_type signed_log_value(sample_rng& rng, int min_exp, int max_exp) noexcept
    {
        const int exponent = rng.integer(min_exp, max_exp);
        const fltx_type value = bl::ldexp(fltx_type{ 1.0 } + rng.unit(), exponent);
        return rng.sign() > 0.0 ? value : fltx_type{ -value };
    }

    [[nodiscard]] inline fltx_type positive_log_value(sample_rng& rng, int min_exp, int max_exp) noexcept
    {
        const int exponent = rng.integer(min_exp, max_exp);
        return bl::ldexp(fltx_type{ 1.0 } + rng.unit(), exponent);
    }

    [[nodiscard]] inline fltx_type residual_for(const fltx_type& value, sample_rng& rng) noexcept
    {
        const double hi = value.hi;
        if (hi == 0.0 || !std::isfinite(hi))
            return fltx_type{ 0.0 };

        const int exponent = std::ilogb(std::fabs(hi));
        const fltx_type residual = bl::ldexp(fltx_type{ 0.5 } + rng.unit(), exponent - 60);
        return rng.sign() > 0.0 ? residual : fltx_type{ -residual };
    }

    [[nodiscard]] inline fltx_type positive_residual_for(const fltx_type& value, sample_rng& rng) noexcept
    {
        const fltx_type residual = residual_for(value, rng);
        return residual.hi < 0.0 ? fltx_type{ -residual } : residual;
    }

    [[nodiscard]] inline fltx_type make_fltx(const value_sample& value) noexcept
    {
        return fltx_type{ value.hi, value.lo };
    }

    [[nodiscard]] inline bool value_sample_is_zero(const value_sample& value) noexcept
    {
        return value.hi == 0.0 && value.lo == 0.0;
    }

    [[nodiscard]] inline bool value_sample_signbit(const value_sample& value) noexcept
    {
        return detail::fp::signbit(value.hi);
    }

    [[nodiscard]] inline perfect_ref signed_zero_perfect(bool negative)
    {
        return perfect_ref{ negative ? "-0" : "0" };
    }

    [[nodiscard]] inline competitor_ref make_competitor(const value_sample& value)
    {
        if (value_sample_is_zero(value))
            return competitor_ref{ value_sample_signbit(value) ? "-0" : "0" };
        return competitor_ref{ value.hi } + competitor_ref{ value.lo };
    }

    [[nodiscard]] inline extra_competitor_ref make_extra_competitor(const value_sample& value)
    {
        return qdpp::make(value.hi, value.lo);
    }

    [[nodiscard]] inline perfect_ref make_perfect(const value_sample& value)
    {
        if (value_sample_is_zero(value))
            return signed_zero_perfect(value_sample_signbit(value));
        return perfect_ref{ value.hi } + perfect_ref{ value.lo };
    }

    [[nodiscard]] inline perfect_ref to_perfect(const fltx_type& value)
    {
        if (bl::iszero(value))
            return signed_zero_perfect(bl::signbit(value));
        return perfect_ref{ value.hi } + perfect_ref{ value.lo };
    }

    [[nodiscard]] inline perfect_ref to_perfect(const competitor_ref& value)
    {
        using boost::multiprecision::signbit;
        if (value == 0 && signbit(value))
            return signed_zero_perfect(true);
        const auto parts = value.backend().crep();
        return perfect_ref{ parts.first } + perfect_ref{ parts.second };
    }

    [[nodiscard]] inline perfect_ref to_perfect(const extra_competitor_ref& value)
    {
        if (value.x[0] == 0.0 && value.x[1] == 0.0)
        {
            return signed_zero_perfect(detail::fp::signbit(value.x[0]));
        }
        return perfect_ref{ value.x[0] } + perfect_ref{ value.x[1] };
    }

    enum class special_value_kind
    {
        nan,
        infinity,
        zero,
        finite
    };

    struct special_value_state
    {
        special_value_kind kind = special_value_kind::finite;
        bool negative = false;
        perfect_ref finite_value = 0;
    };

    [[nodiscard]] inline special_value_state classify_special_ref(const perfect_ref& value)
    {
        const double approximate = static_cast<double>(value);
        if (std::isnan(approximate))
            return { special_value_kind::nan, false, 0 };
        if (std::isinf(approximate))
            return { special_value_kind::infinity, std::signbit(approximate), 0 };

        using boost::multiprecision::signbit;
        if (value == 0)
            return { special_value_kind::zero, signbit(value), 0 };
        return { special_value_kind::finite, signbit(value), value };
    }

    [[nodiscard]] inline special_value_state classify_special_double(double value)
    {
        if (std::isnan(value))
            return { special_value_kind::nan, false, 0 };
        if (std::isinf(value))
            return { special_value_kind::infinity, std::signbit(value), 0 };
        if (value == 0.0)
            return { special_value_kind::zero, std::signbit(value), 0 };
        return { special_value_kind::finite, std::signbit(value), perfect_ref{ value } };
    }

    [[nodiscard]] inline special_value_state classify_special_value(const fltx_type& value)
    {
        if (bl::isnan(value))
            return { special_value_kind::nan, false, 0 };
        if (bl::isinf(value))
            return { special_value_kind::infinity, bl::signbit(value), 0 };
        if (bl::iszero(value))
            return { special_value_kind::zero, bl::signbit(value), 0 };
        return { special_value_kind::finite, bl::signbit(value), to_perfect(value) };
    }

    [[nodiscard]] inline special_value_state classify_special_value(const competitor_ref& value)
    {
        return classify_special_ref(to_perfect(value));
    }

    [[nodiscard]] inline special_value_state classify_special_value(const extra_competitor_ref& value)
    {
        if (std::isnan(value.x[0]) || (std::isfinite(value.x[0]) && std::isnan(value.x[1])))
            return { special_value_kind::nan, false, 0 };
        if (std::isinf(value.x[0]))
            return { special_value_kind::infinity, std::signbit(value.x[0]), 0 };
        if (value.x[0] == 0.0 && value.x[1] == 0.0)
            return { special_value_kind::zero, std::signbit(value.x[0]), 0 };
        return { special_value_kind::finite, std::signbit(value.x[0]), to_perfect(value) };
    }

    [[nodiscard]] inline perfect_ref abs_ref(const perfect_ref& value)
    {
        return value < 0 ? -value : value;
    }

    [[nodiscard]] inline perfect_ref reference_scale(const perfect_ref& expected)
    {
        const perfect_ref scale = abs_ref(expected);
        return scale < 1 ? perfect_ref{ 1 } : scale;
    }

    [[nodiscard]] inline double matching_bits(const perfect_ref& actual, const perfect_ref& expected)
    {
        const perfect_ref error = abs_ref(actual - expected);
        if (error == 0)
            return std::numeric_limits<double>::infinity();

        const perfect_ref scaled_error = error / reference_scale(expected);
        using std::log2;
        return static_cast<double>(-log2(scaled_error));
    }

    [[nodiscard]] inline double domain_matching_bits(const perfect_ref& actual, const perfect_ref& expected)
    {
        const perfect_ref error = abs_ref(actual - expected);
        if (error == 0)
            return std::numeric_limits<double>::infinity();

        const perfect_ref scaled_error = error / reference_scale(expected);
        using std::log2;
        return static_cast<double>(-log2(scaled_error));
    }

    [[nodiscard]] inline double domain_score_bits(const perfect_ref& actual, const perfect_ref& expected)
    {
        const double bits = domain_matching_bits(actual, expected);
        return std::isnan(bits) ? 0.0 : bits;
    }

    [[nodiscard]] inline bool special_values_match(
        const special_value_state& actual,
        const special_value_state& expected)
    {
        constexpr double finite_match_bits = 20.0;
        if (actual.kind != expected.kind)
            return false;

        if (expected.kind == special_value_kind::nan)
            return true;
        if (expected.kind == special_value_kind::infinity || expected.kind == special_value_kind::zero)
            return actual.negative == expected.negative;

        const double bits = matching_bits(actual.finite_value, expected.finite_value);
        return std::isinf(bits) || bits >= finite_match_bits;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_unary_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        (void)reference;
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            special_value_state expected;
            try
            {
                expected = classify_special_double(
                    stdlib_unary_special_result(operation, sample_as_double(samples[index].x)));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const special_value_state actual = classify_special_value(eval(values[index]));
                if (!special_values_match(actual, expected))
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' mismatch\n";
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_binary_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        (void)reference;
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            special_value_state expected;
            try
            {
                expected = classify_special_double(stdlib_binary_special_result(
                    operation,
                    sample_as_double(samples[index].x),
                    sample_as_double(samples[index].y)));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const special_value_state actual = classify_special_value(
                    eval(values[index].x, values[index].y));
                if (!special_values_match(actual, expected))
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' mismatch\n";
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_ternary_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        (void)reference;
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            special_value_state expected;
            try
            {
                expected = classify_special_double(stdlib_ternary_special_result(
                    operation,
                    sample_as_double(samples[index].x),
                    sample_as_double(samples[index].y),
                    sample_as_double(samples[index].z)));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const special_value_state actual = classify_special_value(
                    eval(values[index].x, values[index].y, values[index].z));
                if (!special_values_match(actual, expected))
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' mismatch\n";
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_unary_int_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        (void)reference;
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            special_value_state expected;
            try
            {
                expected = classify_special_double(stdlib_unary_int_special_result(
                    operation,
                    sample_as_double(samples[index].x),
                    samples[index].n));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const special_value_state actual = classify_special_value(
                    eval(values[index].x, values[index].n));
                if (!special_values_match(actual, expected))
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' mismatch\n";
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    [[nodiscard]] inline double finite_for_mean(double bits) noexcept
    {
        constexpr double exact_result_bits = 160.0;
        return std::isinf(bits) ? exact_result_bits : bits;
    }

    template<class Samples, class T, class MakeFn>
    [[nodiscard]] std::vector<T> make_unary_inputs(const Samples& samples, MakeFn make_value)
    {
        std::vector<T> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back(make_value(sample.x));
        return values;
    }

    template<class Samples, class T, class MakeFn>
    [[nodiscard]] std::vector<binary_value<T>> make_binary_inputs(const Samples& samples, MakeFn make_value)
    {
        std::vector<binary_value<T>> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back({ make_value(sample.x), make_value(sample.y) });
        return values;
    }

    template<class Samples, class T, class MakeFn>
    [[nodiscard]] std::vector<ternary_value<T>> make_ternary_inputs(const Samples& samples, MakeFn make_value)
    {
        std::vector<ternary_value<T>> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back({ make_value(sample.x), make_value(sample.y), make_value(sample.z) });
        return values;
    }

    template<class Samples, class T, class MakeFn>
    [[nodiscard]] std::vector<unary_int_value<T>> make_unary_int_inputs(const Samples& samples, MakeFn make_value)
    {
        std::vector<unary_int_value<T>> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back({ make_value(sample.x), sample.n });
        return values;
    }

    template<class T>
    [[nodiscard]] T call_add(const T& x, const T& y) { return x + y; }

    template<class T>
    [[nodiscard]] T call_subtract(const T& x, const T& y) { return x - y; }

    template<class T>
    [[nodiscard]] T call_multiply(const T& x, const T& y) { return x * y; }

    template<class T>
    [[nodiscard]] T call_divide(const T& x, const T& y) { return x / y; }

    template<class T>
    [[nodiscard]] bool call_equal(const T& x, const T& y) { return x == y; }

    template<class T>
    [[nodiscard]] bool call_not_equal(const T& x, const T& y) { return x != y; }

    template<class T>
    [[nodiscard]] bool call_less(const T& x, const T& y) { return x < y; }

    template<class T>
    [[nodiscard]] bool call_greater(const T& x, const T& y) { return x > y; }

    template<class T>
    [[nodiscard]] bool call_less_equal(const T& x, const T& y) { return x <= y; }

    template<class T>
    [[nodiscard]] bool call_greater_equal(const T& x, const T& y) { return x >= y; }

    template<class T>
    [[nodiscard]] T call_sqrt(const T& x) { using std::sqrt; return sqrt(x); }

    [[nodiscard]] inline extra_competitor_ref call_cbrt(const extra_competitor_ref& x)
    {
        return qdpp::cbrt(x);
    }

    template<class T>
    [[nodiscard]] T call_cbrt(const T& x) { using std::cbrt; return cbrt(x); }

    template<class T>
    [[nodiscard]] T call_hypot(const T& x, const T& y) { using std::hypot; return hypot(x, y); }

    template<class T>
    [[nodiscard]] T call_sin(const T& x) { using std::sin; return sin(x); }

    template<class T>
    [[nodiscard]] T call_cos(const T& x) { using std::cos; return cos(x); }

    template<class T>
    [[nodiscard]] T call_tan(const T& x) { using std::tan; return tan(x); }

    template<class T>
    [[nodiscard]] T call_atan(const T& x) { using std::atan; return atan(x); }

    template<class T>
    [[nodiscard]] T call_atan2(const T& y, const T& x) { using std::atan2; return atan2(y, x); }

    template<class T>
    [[nodiscard]] T call_asin(const T& x) { using std::asin; return asin(x); }

    template<class T>
    [[nodiscard]] T call_acos(const T& x) { using std::acos; return acos(x); }

    template<class T>
    [[nodiscard]] T call_exp(const T& x) { using std::exp; return exp(x); }

    template<class T>
    [[nodiscard]] T call_exp2(const T& x) { using std::exp2; return exp2(x); }

    template<class T>
    [[nodiscard]] T call_expm1(const T& x) { using std::expm1; return expm1(x); }

    template<class T>
    [[nodiscard]] T call_log(const T& x) { using std::log; return log(x); }

    template<class T>
    [[nodiscard]] T call_log2(const T& x) { using std::log2; return log2(x); }

    template<class T>
    [[nodiscard]] T call_log10(const T& x) { using std::log10; return log10(x); }

    template<class T>
    [[nodiscard]] T call_log1p(const T& x) { using std::log1p; return log1p(x); }

    template<class T>
    [[nodiscard]] T call_pow(const T& x, const T& y) { using std::pow; return pow(x, y); }

    template<class T>
    [[nodiscard]] T call_sinh(const T& x) { using std::sinh; return sinh(x); }

    template<class T>
    [[nodiscard]] T call_cosh(const T& x) { using std::cosh; return cosh(x); }

    template<class T>
    [[nodiscard]] T call_tanh(const T& x) { using std::tanh; return tanh(x); }

    template<class T>
    [[nodiscard]] T call_asinh(const T& x) { using std::asinh; return asinh(x); }

    template<class T>
    [[nodiscard]] T call_acosh(const T& x) { using std::acosh; return acosh(x); }

    template<class T>
    [[nodiscard]] T call_atanh(const T& x) { using std::atanh; return atanh(x); }

    template<class T>
    [[nodiscard]] T call_fma(const T& x, const T& y, const T& z) { using std::fma; return fma(x, y, z); }

    template<class T>
    [[nodiscard]] T call_fma_reference(const T& x, const T& y, const T& z) { return x * y + z; }

    template<class T>
    [[nodiscard]] T call_fabs(const T& x) { using boost::multiprecision::fabs; using std::fabs; return fabs(x); }

    template<class T>
    [[nodiscard]] T call_floor(const T& x) { using boost::multiprecision::floor; using std::floor; return floor(x); }

    template<class T>
    [[nodiscard]] T call_ceil(const T& x) { using boost::multiprecision::ceil; using std::ceil; return ceil(x); }

    [[nodiscard]] inline extra_competitor_ref call_trunc(const extra_competitor_ref& x)
    {
        return qdpp::trunc(x);
    }

    template<class T>
    [[nodiscard]] T call_trunc(const T& x) { using boost::multiprecision::trunc; using std::trunc; return trunc(x); }

    template<class T>
    [[nodiscard]] T call_round(const T& x) { using boost::multiprecision::round; using std::round; return round(x); }

    [[nodiscard]] inline extra_competitor_ref call_nearbyint(const extra_competitor_ref& x)
    {
        return qdpp::nearbyint(x);
    }

    template<class T>
    [[nodiscard]] T call_nearbyint(const T& x)
    {
        using boost::multiprecision::nearbyint;
        using std::nearbyint;
        return nearbyint(x);
    }

    [[nodiscard]] inline extra_competitor_ref call_rint(const extra_competitor_ref& x)
    {
        return qdpp::rint(x);
    }

    template<class T>
    [[nodiscard]] T call_rint(const T& x) { using boost::multiprecision::rint; using std::rint; return rint(x); }

    template<class T>
    [[nodiscard]] long call_lround(const T& x)
    {
        using boost::multiprecision::lround;
        using std::lround;
        return lround(x);
    }

    template<class T>
    [[nodiscard]] long long call_llround(const T& x)
    {
        using boost::multiprecision::llround;
        using std::llround;
        return llround(x);
    }

    template<class T>
    [[nodiscard]] long call_lrint(const T& x)
    {
        using boost::multiprecision::lrint;
        using std::lrint;
        return lrint(x);
    }

    template<class T>
    [[nodiscard]] long long call_llrint(const T& x)
    {
        using boost::multiprecision::llrint;
        using std::llrint;
        return llrint(x);
    }

    template<class T>
    [[nodiscard]] T round_nearest_even_integer_reference(const T& x)
    {
        using boost::multiprecision::floor;
        using std::floor;

        const T lower = floor(x);
        const T fraction = x - lower;
        if (fraction < T{ 0.5 })
            return lower;
        if (fraction > T{ 0.5 })
            return lower + T{ 1 };

        const auto lower_integer = static_cast<long long>(lower);
        return (lower_integer % 2ll) == 0 ? lower : lower + T{ 1 };
    }

    template<class T>
    [[nodiscard]] long call_lrint_reference(const T& x)
    {
        return static_cast<long>(round_nearest_even_integer_reference(x));
    }

    template<class T>
    [[nodiscard]] long long call_llrint_reference(const T& x)
    {
        return static_cast<long long>(round_nearest_even_integer_reference(x));
    }

    template<class T>
    [[nodiscard]] T call_fmod(const T& x, const T& y)
    {
        using boost::multiprecision::fmod;
        using std::fmod;
        return fmod(x, y);
    }

    [[nodiscard]] inline extra_competitor_ref call_remainder(const extra_competitor_ref& x, const extra_competitor_ref& y)
    {
        return qdpp::remainder(x, y);
    }

    template<class T>
    [[nodiscard]] T call_remainder(const T& x, const T& y)
    {
        using boost::multiprecision::remainder;
        using std::remainder;
        return remainder(x, y);
    }

    template<class T>
    [[nodiscard]] T call_remquo_value(const T& x, const T& y)
    {
        int quotient = 0;
        using boost::multiprecision::remquo;
        using std::remquo;
        return remquo(x, y, &quotient);
    }

    template<class T>
    [[nodiscard]] T call_fmin(const T& x, const T& y)
    {
        using boost::multiprecision::fmin;
        using std::fmin;
        return fmin(x, y);
    }

    template<class T>
    [[nodiscard]] T call_fmax(const T& x, const T& y)
    {
        using boost::multiprecision::fmax;
        using std::fmax;
        return fmax(x, y);
    }

    template<class T>
    [[nodiscard]] T call_fdim(const T& x, const T& y)
    {
        using boost::multiprecision::fdim;
        using std::fdim;
        return fdim(x, y);
    }

    template<class T>
    [[nodiscard]] T call_copysign(const T& x, const T& y)
    {
        using boost::multiprecision::copysign;
        using std::copysign;
        return copysign(x, y);
    }

    template<class T>
    [[nodiscard]] T call_ldexp(const T& x, int n)
    {
        using boost::multiprecision::ldexp;
        using std::ldexp;
        return ldexp(x, n);
    }

    [[nodiscard]] inline extra_competitor_ref call_scalbn(const extra_competitor_ref& x, int n)
    {
        return qdpp::scalbn(x, n);
    }

    template<class T>
    [[nodiscard]] T call_scalbn(const T& x, int n)
    {
        using boost::multiprecision::scalbn;
        using std::scalbn;
        return scalbn(x, n);
    }

    [[nodiscard]] inline extra_competitor_ref call_scalbln(const extra_competitor_ref& x, int n)
    {
        return qdpp::scalbln(x, static_cast<long>(n));
    }

    template<class T>
    [[nodiscard]] T call_scalbln(const T& x, int n)
    {
        using boost::multiprecision::scalbln;
        using std::scalbln;
        return scalbln(x, static_cast<long>(n));
    }

    [[nodiscard]] inline fltx_type call_nextafter(const fltx_type& from, const fltx_type& to) noexcept
    {
        return bl::nextafter(from, to);
    }

    template<class T>
    [[nodiscard]] T call_nextafter(const T& from, const T& to)
    {
        return boost::math::nextafter(from, to);
    }

    [[nodiscard]] inline fltx_type call_nexttoward(const fltx_type& from, const fltx_type& to) noexcept
    {
        return bl::nexttoward(from, to);
    }

    template<class T>
    [[nodiscard]] T call_nexttoward(const T& from, const T& to)
    {
        return call_nextafter(from, to);
    }

    template<class T>
    [[nodiscard]] T call_logb(const T& x)
    {
        using boost::multiprecision::logb;
        using std::logb;
        return logb(x);
    }

    template<class T>
    [[nodiscard]] int call_ilogb(const T& x)
    {
        using boost::multiprecision::ilogb;
        using std::ilogb;
        return static_cast<int>(ilogb(x));
    }

    template<class T>
    [[nodiscard]] frexp_value<T> call_frexp(const T& x)
    {
        int exponent = 0;
        using boost::multiprecision::frexp;
        using std::frexp;
        return { frexp(x, &exponent), exponent };
    }

    template<class T>
    [[nodiscard]] T call_modf_fraction(const T& x)
    {
        T integer_part{};
        using boost::multiprecision::modf;
        using std::modf;
        return modf(x, &integer_part);
    }

    template<class T>
    [[nodiscard]] T call_erf(const T& x)
    {
        using boost::multiprecision::erf;
        using std::erf;
        return erf(x);
    }

    template<class T>
    [[nodiscard]] T call_erfc(const T& x)
    {
        using boost::multiprecision::erfc;
        using std::erfc;
        return erfc(x);
    }

    template<class T>
    [[nodiscard]] T call_lgamma(const T& x)
    {
        using boost::multiprecision::lgamma;
        using std::lgamma;
        return lgamma(x);
    }

    [[nodiscard]] inline perfect_ref gamma_reference_sinpi(const perfect_ref& x)
    {
        using boost::multiprecision::floor;
        using boost::multiprecision::sin;

        const perfect_ref n = floor(x);
        const perfect_ref r = x - n;
        perfect_ref out = sin(boost::math::constants::pi<perfect_ref>() * r);

        const long long n_int = n.convert_to<long long>();
        if ((n_int & 1ll) != 0)
            out = -out;
        return out;
    }

    [[nodiscard]] inline bool gamma_reference_is_integer(const perfect_ref& x)
    {
        using boost::multiprecision::trunc;
        return trunc(x) == x;
    }

    [[nodiscard]] inline perfect_ref call_lgamma(const perfect_ref& x)
    {
        const double approximate = static_cast<double>(x);
        if (std::isnan(approximate))
            return std::numeric_limits<perfect_ref>::quiet_NaN();
        if (std::isinf(approximate))
            return std::numeric_limits<perfect_ref>::infinity();

        if (x > 0)
        {
            using boost::multiprecision::lgamma;
            return lgamma(x);
        }

        if (gamma_reference_is_integer(x))
            return std::numeric_limits<perfect_ref>::infinity();

        using boost::multiprecision::lgamma;
        using boost::multiprecision::log;

        const perfect_ref sinpix = gamma_reference_sinpi(x);
        const perfect_ref abs_sinpix = sinpix < 0 ? -sinpix : sinpix;
        return log(boost::math::constants::pi<perfect_ref>()) - log(abs_sinpix) - lgamma(perfect_ref{ 1 } - x);
    }

    template<class T>
    [[nodiscard]] T call_tgamma(const T& x)
    {
        using boost::multiprecision::tgamma;
        using std::tgamma;
        return tgamma(x);
    }

    [[nodiscard]] inline perfect_ref call_tgamma(const perfect_ref& x)
    {
        const double approximate = static_cast<double>(x);
        if (std::isnan(approximate))
            return std::numeric_limits<perfect_ref>::quiet_NaN();
        if (std::isinf(approximate))
        {
            return std::signbit(approximate)
                ? std::numeric_limits<perfect_ref>::quiet_NaN()
                : std::numeric_limits<perfect_ref>::infinity();
        }

        using boost::multiprecision::signbit;
        if (x == 0)
        {
            return signbit(x)
                ? -std::numeric_limits<perfect_ref>::infinity()
                :  std::numeric_limits<perfect_ref>::infinity();
        }

        if (x > 0)
        {
            using boost::multiprecision::tgamma;
            return tgamma(x);
        }

        if (gamma_reference_is_integer(x))
            return std::numeric_limits<perfect_ref>::quiet_NaN();

        using boost::multiprecision::tgamma;

        const perfect_ref sinpix = gamma_reference_sinpi(x);
        if (sinpix == 0)
            return std::numeric_limits<perfect_ref>::quiet_NaN();

        return boost::math::constants::pi<perfect_ref>() / (sinpix * tgamma(perfect_ref{ 1 } - x));
    }

    [[nodiscard]] inline std::vector<unary_sample> make_positive_samples()
    {
        std::vector<unary_sample> samples{
            { "quarter", { "x", 0.25, 0x1p-56 } },
            { "near one", { "x", 1.0, 0x1p-54 } },
            { "two", { "x", 2.0, 0x1p-55 } },
            { "large", { "x", 0x1.3c0ca428c59f8p+24, 0x1p-30 } },
            { "tiny normal", { "x", 0x1p-32, 0x1p-86 } }
        };

        sample_rng rng{ 0x128acc90517ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = positive_log_value(rng, -80, 80);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_arithmetic_samples()
    {
        std::vector<binary_sample> samples{
            { "unit mixed signs", { "x", 1.25, 0x1p-54 }, { "y", -0.75, 0x1p-55 } },
            { "large plus small", { "x", 0x1.3c0ca428c59f8p+32, 0x1p-22 }, { "y", -0x1.f972474538ef3p-20, 0x1p-74 } },
            { "near cancellation", { "x", 1.0, 0x1p-54 }, { "y", -1.0, 0x1p-55 } },
            { "fractional", { "x", -0x1.1f9add3739636p-4, 0x1p-60 }, { "y", 0x1.3be76c8b43958p+3, -0x1p-52 } },
            { "wide finite", { "x", 0x1.2d6444d013d18p+48, -0x1p-8 }, { "y", 0x1.8p-24, 0x1p-80 } }
        };

        sample_rng rng{ 0x128acca217ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -80, 80);
            const fltx_type x_value = x + residual_for(x, rng);
            fltx_type y_value{ 0.0 };
            switch (index % 4)
            {
            case 0:
                y_value = signed_log_value(rng, -80, 80) + residual_for(x, rng);
                break;
            case 1:
                y_value = -x_value + residual_for(x, rng);
                break;
            case 2:
                y_value = x_value + residual_for(x, rng);
                break;
            default:
                y_value = signed_log_value(rng, -12, 12) + residual_for(x, rng);
                break;
            }

            samples.push_back({
                "random",
                make_runtime_value("x", x_value),
                make_runtime_value("y", y_value)
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_comparison_samples()
    {
        std::vector<binary_sample> samples{
            { "equal positive", { "x", 1.25, 0x1p-54 }, { "y", 1.25, 0x1p-54 } },
            { "equal negative", { "x", -2.5, -0x1p-53 }, { "y", -2.5, -0x1p-53 } },
            { "signed zeros", { "x", 0.0, 0.0 }, { "y", std::copysign(0.0, -1.0), 0.0 } },
            { "same high x lower tail", { "x", 1.0, -0x1p-54 }, { "y", 1.0, 0x1p-54 } },
            { "same high x higher tail", { "x", 1.0, 0x1p-54 }, { "y", 1.0, -0x1p-54 } },
            { "negative same high", { "x", -1.0, -0x1p-54 }, { "y", -1.0, 0x1p-54 } },
            { "wide x greater", { "x", 0x1.2d6444d013d18p+48, -0x1p-8 }, { "y", 0x1.2d6444d013d18p+47, 0x1p-8 } },
            { "tiny x less", { "x", -0x1p-80, 0x1p-134 }, { "y", 0x1p-80, -0x1p-134 } }
        };

        sample_rng rng{ 0x128c0a9a11ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -80, 80);
            fltx_type y{};
            switch (index % 5)
            {
            case 0:
                y = x;
                break;
            case 1:
                y = x + residual_for(x, rng);
                break;
            case 2:
                y = x - residual_for(x, rng);
                break;
            case 3:
                y = -x;
                break;
            default:
                y = signed_log_value(rng, -80, 80);
                break;
            }

            samples.push_back({
                "random",
                make_runtime_value("x", x),
                make_runtime_value("y", y)
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_nextafter_samples()
    {
        std::vector<binary_sample> samples{
            { "equal positive", { "from", 1.25, 0x1p-54 }, { "to", 1.25, 0x1p-54 } },
            { "up tail", { "from", 1.25, 0x1p-54 }, { "to", 2.0, 0.0 } },
            { "down tail", { "from", 1.25, 0x1p-54 }, { "to", -2.0, 0.0 } },
            { "zero to positive", { "from", 0.0, 0.0 }, { "to", 1.0, 0.0 } },
            { "zero to negative", { "from", 0.0, 0.0 }, { "to", -1.0, 0.0 } },
            { "negative zero to positive", { "from", std::copysign(0.0, -1.0), 0.0 }, { "to", 1.0, 0.0 } },
            { "large up", { "from", 0x1.2d6444d013d18p+48, -0x1p-8 }, { "to", 0x1.2d6444d013d18p+49, 0.0 } },
            { "tiny down", { "from", -0x1p-80, 0x1p-134 }, { "to", -1.0, 0.0 } }
        };

        sample_rng rng{ 0x128ad3a9e5ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type from = signed_log_value(rng, -80, 80);
            const fltx_type delta = positive_residual_for(from, rng);
            fltx_type to{};
            switch (index % 5)
            {
            case 0:
                to = from + delta;
                break;
            case 1:
                to = from - delta;
                break;
            case 2:
                to = -from;
                break;
            case 3:
                to = signed_log_value(rng, -80, 80);
                break;
            default:
                to = from;
                break;
            }

            if (to == from && (index % 5) != 4)
                to = from + fltx_type{ rng.sign() };

            samples.push_back({
                "random",
                make_runtime_value("from", from, residual_for(from, rng)),
                make_runtime_value("to", to, residual_for(to, rng))
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_signed_samples()
    {
        std::vector<unary_sample> samples{
            { "negative large", { "x", -1e20, 0x1p+12 } },
            { "negative ten", { "x", -10.0, 0x1p-50 } },
            { "negative one", { "x", -1.0, 0x1p-54 } },
            { "negative tiny", { "x", -1e-20, 0x1p-120 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "positive tiny", { "x", 1e-20, -0x1p-120 } },
            { "one", { "x", 1.0, -0x1p-54 } },
            { "ten", { "x", 10.0, -0x1p-50 } },
            { "positive large", { "x", 1e20, -0x1p+12 } }
        };

        sample_rng rng{ 0x128acc51a5ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -80, 80);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline fltx_type trig_half_pi() noexcept
    {
        return fltx_type{ 0x1.921fb54442d18p+0 } + fltx_type{ 0x1.1a62633145c07p-54 };
    }

    inline void append_trig_landmark_samples(std::vector<unary_sample>& samples)
    {
        samples.push_back({ "tiny", { "x", 0x1p-20, 0x1p-74 } });
        samples.push_back({ "pi / 6", { "x", 0x1.0c152382d7366p-1, 0x1.1a62633145c07p-55 } });
        samples.push_back({ "pi / 4", { "x", 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55 } });
        samples.push_back({ "negative moderate", { "x", -2.5, 0x1p-54 } });
        samples.push_back({ "ten pi plus offset", { "x", 31.75, -0x1p-52 } });
        samples.push_back({ "wide reduction", { "x", 12345.678901234567, 0x1p-43 } });
    }

    [[nodiscard]] inline std::vector<unary_sample> make_unshifted_trig_samples()
    {
        std::vector<unary_sample> samples;
        append_trig_landmark_samples(samples);

        sample_rng rng{ 0x128acc7e16ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -0x1p16, 0x1p16);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_trig_samples()
    {
        std::vector<unary_sample> samples;
        append_trig_landmark_samples(samples);

        const fltx_type half_pi = trig_half_pi();
        const std::size_t landmark_count = samples.size();
        for (std::size_t index = 0; index < landmark_count; ++index)
        {
            const fltx_type shifted = make_fltx(samples[index].x) + half_pi;
            samples.push_back({ "phase shifted", make_runtime_value("x", shifted) });
        }

        sample_rng rng{ 0x128acc7e16ull };
        for (std::size_t index = 0; index < random_sample_count / 2; ++index)
        {
            const fltx_type x = uniform_value(rng, -0x1p16, 0x1p16);
            const fltx_type value = x + residual_for(x, rng);
            samples.push_back({ "random", make_runtime_value("x", value) });
            samples.push_back({ "random phase shifted", make_runtime_value("x", value + half_pi) });
        }
        if ((random_sample_count & 1u) != 0u)
        {
            const fltx_type x = uniform_value(rng, -0x1p16, 0x1p16);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_tan_samples()
    {
        std::vector<unary_sample> samples = make_unshifted_trig_samples();
        sample_rng rng{ 0x128acc7a9ull };
        while (samples.size() < random_sample_count + 6)
        {
            const fltx_type x = uniform_value(rng, -0x1p16, 0x1p16);
            if (std::fabs(std::cos(static_cast<double>(x))) < 0x1p-20)
                continue;
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_exp_samples()
    {
        std::vector<unary_sample> samples{
            { "negative ten", { "x", -10.0, 0x1p-50 } },
            { "near zero", { "x", 0.0, 0x1p-60 } },
            { "quarter", { "x", 0.25, 0x1p-56 } },
            { "one", { "x", 1.0, -0x1p-54 } },
            { "five", { "x", 5.0, 0x1p-50 } }
        };

        sample_rng rng{ 0x128acce901ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -40.0, 40.0);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_exp2_samples()
    {
        std::vector<unary_sample> samples{
            { "negative ten", { "x", -10.0, 0x1p-50 } },
            { "negative one", { "x", -1.0, 0x1p-54 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "one", { "x", 1.0, -0x1p-54 } },
            { "ten", { "x", 10.0, -0x1p-50 } }
        };

        sample_rng rng{ 0x128acce2e2ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -128.0, 128.0);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_expm1_samples()
    {
        std::vector<unary_sample> samples{
            { "negative twenty", { "x", -20.0, 0x1p-48 } },
            { "negative tiny", { "x", -1e-10, 0x1p-88 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "positive tiny", { "x", 1e-10, -0x1p-88 } },
            { "twenty", { "x", 20.0, -0x1p-48 } }
        };

        sample_rng rng{ 0x128acce111ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -20.0, 20.0);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_log1p_samples()
    {
        std::vector<unary_sample> samples{
            { "near negative one", { "x", -0.875, 0x1p-58 } },
            { "negative tiny", { "x", -1e-10, 0x1p-88 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "positive tiny", { "x", 1e-10, -0x1p-88 } },
            { "ten", { "x", 10.0, 0x1p-50 } }
        };

        sample_rng rng{ 0x128acc1011ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -0.95, 20.0);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_pow_samples()
    {
        std::vector<binary_sample> samples{
            { "square root", { "x", 2.0, 0x1p-55 }, { "y", 0.5, 0x1p-56 } },
            { "fractional base", { "x", 0.75, 0x1p-56 }, { "y", 3.25, -0x1p-54 } },
            { "near one", { "x", 1.0, 0x1p-53 }, { "y", 512.0, 0x1p-45 } },
            { "moderate", { "x", 8.5, -0x1p-53 }, { "y", -1.25, 0x1p-55 } },
            { "large base", { "x", 1024.0, 0x1p-42 }, { "y", 1.75, -0x1p-55 } }
        };

        sample_rng rng{ 0x128acc90dull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            fltx_type x{};
            fltx_type y{};
            switch (index % 4)
            {
            case 0:
                x = positive_log_value(rng, -12, 12);
                y = uniform_value(rng, -6.0, 6.0);
                break;
            case 1:
                x = fltx_type{ 1.0 } + uniform_value(rng, -0.125, 0.125);
                y = uniform_value(rng, -256.0, 256.0);
                break;
            case 2:
                x = positive_log_value(rng, -32, 32);
                y = uniform_value(rng, -2.0, 2.0);
                break;
            default:
                x = uniform_value(rng, 0.125, 64.0);
                y = fltx_type{ rng.integer(-16, 16) };
                break;
            }
            samples.push_back({
                "random",
                make_runtime_value("x", x, residual_for(x, rng)),
                make_runtime_value("y", y, residual_for(y, rng))
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<ternary_sample> make_fma_samples()
    {
        std::vector<ternary_sample> samples{
            { "unit positive", { "x", 1.0, 0x1p-54 }, { "y", 1.0, -0x1p-55 }, { "z", 0.5, 0x1p-56 } },
            { "mixed signs", { "x", -1.25, 0x1p-54 }, { "y", 0.75, 0x1p-55 }, { "z", -0.5, -0x1p-56 } },
            { "large product", { "x", 0x1.3c0ca428c59f8p+24, 0x1p-30 }, { "y", -0x1.8p+12, 0x1p-42 }, { "z", 0x1.2p+20, -0x1p-34 } }
        };

        sample_rng rng{ 0x128accf0a1ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -24, 24);
            const fltx_type y = signed_log_value(rng, -24, 24);
            const fltx_type product = x * y;
            fltx_type z{};
            switch (index % 4)
            {
            case 0:
                z = signed_log_value(rng, -48, 48);
                break;
            case 1:
                z = -product + (rng.sign() > 0.0 ? bl::ldexp(product, -12) : -bl::ldexp(product, -12));
                break;
            case 2:
                z = product * fltx_type{ -0.5 } + residual_for(product, rng);
                break;
            default:
                z = signed_log_value(rng, -12, 12);
                break;
            }
            samples.push_back({
                "random",
                make_runtime_value("x", x, residual_for(x, rng)),
                make_runtime_value("y", y, residual_for(y, rng)),
                make_runtime_value("z", z, residual_for(z, rng))
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_hypot_samples()
    {
        std::vector<binary_sample> samples{
            { "zero", { "x", 0.0, 0.0 }, { "y", 0.0, 0.0 } },
            { "three four", { "x", 3.0, 0x1p-54 }, { "y", 4.0, -0x1p-54 } },
            { "mixed scale", { "x", 1e20, 0x1p+12 }, { "y", 1e-20, 0x1p-120 } },
            { "moderate", { "x", 123.456, 0x1p-48 }, { "y", 789.25, -0x1p-44 } }
        };

        sample_rng rng{ 0x128acc1707ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -60, 60);
            const fltx_type y = signed_log_value(rng, -60, 60);
            samples.push_back({
                "random",
                make_runtime_value("x", x, residual_for(x, rng)),
                make_runtime_value("y", y, residual_for(y, rng))
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_unit_interval_samples()
    {
        std::vector<unary_sample> samples{
            { "negative one", { "x", -1.0, 0.0 } },
            { "negative half", { "x", -0.5, 0x1p-58 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "half", { "x", 0.5, -0x1p-58 } },
            { "one", { "x", 1.0, 0.0 } }
        };

        sample_rng rng{ 0x128acc1eafull };
        while (samples.size() < random_sample_count + 5)
        {
            const fltx_type x = uniform_value(rng, -0.999999, 0.999999);
            const value_sample sample = make_runtime_value("x", x, residual_for(x, rng));
            if (std::fabs(sample.hi + sample.lo) <= 1.0)
                samples.push_back({ "random", sample });
        }
        return samples;
    }

    inline void append_hyperbolic_landmark_samples(std::vector<unary_sample>& samples)
    {
        samples.push_back({ "negative four", { "x", -4.0, 0x1p-52 } });
        samples.push_back({ "negative one", { "x", -1.0, 0x1p-54 } });
        samples.push_back({ "zero", { "x", 0.0, 0.0 } });
        samples.push_back({ "one", { "x", 1.0, -0x1p-54 } });
        samples.push_back({ "four", { "x", 4.0, -0x1p-52 } });
    }

    [[nodiscard]] inline std::vector<unary_sample> make_hyperbolic_samples()
    {
        std::vector<unary_sample> samples;
        append_hyperbolic_landmark_samples(samples);

        sample_rng rng{ 0x128accb1b0ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            fltx_type x{};
            switch (index % 4)
            {
            case 0:
                x = uniform_value(rng, -4.0, 4.0);
                break;
            case 1:
                x = uniform_value(rng, -16.0, 16.0);
                break;
            case 2:
                x = uniform_value(rng, -40.0, 40.0);
                break;
            default:
                x = signed_log_value(rng, -8, 5);
                break;
            }
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_asinh_samples()
    {
        std::vector<unary_sample> samples;
        append_hyperbolic_landmark_samples(samples);
        sample_rng rng{ 0x128acca511ull };
        while (samples.size() < random_sample_count + 5)
        {
            const fltx_type x = (samples.size() & 1u) == 0u
                ? uniform_value(rng, -32.0, 32.0)
                : signed_log_value(rng, -60, 60);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_acosh_samples()
    {
        std::vector<unary_sample> samples{
            { "one", { "x", 1.0, 0.0 } },
            { "near one", { "x", 1.0, 0x1p-52 } },
            { "one half", { "x", 1.5, -0x1p-54 } },
            { "two", { "x", 2.0, 0x1p-55 } },
            { "sixteen", { "x", 16.0, 0x1p-48 } }
        };

        sample_rng rng{ 0x128acca051ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            fltx_type x{};
            switch (index % 4)
            {
            case 0:
                x = uniform_value(rng, 1.0, 16.0);
                break;
            case 1:
                x = fltx_type{ 1.0 } + positive_log_value(rng, -60, -1);
                break;
            case 2:
                x = positive_log_value(rng, 0, 60);
                break;
            default:
                x = uniform_value(rng, 16.0, 1024.0);
                break;
            }
            samples.push_back({ "random", make_runtime_value("x", x, positive_residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_atanh_samples()
    {
        std::vector<unary_sample> samples{
            { "negative near one", { "x", -0.95, 0x1p-58 } },
            { "negative tiny", { "x", -1e-10, 0x1p-88 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "positive tiny", { "x", 1e-10, -0x1p-88 } },
            { "positive near one", { "x", 0.95, -0x1p-58 } }
        };

        sample_rng rng{ 0x128acca7a9ull };
        while (samples.size() < random_sample_count + 5)
        {
            fltx_type x{};
            switch (samples.size() % 4)
            {
            case 0:
                x = uniform_value(rng, -0.95, 0.95);
                break;
            case 1:
                if (rng.sign() > 0.0)
                    x = fltx_type{ 1.0 } - positive_log_value(rng, -60, -1);
                else
                    x = fltx_type{ -1.0 } + positive_log_value(rng, -60, -1);
                break;
            case 2:
                x = uniform_value(rng, -0.999999, 0.999999);
                break;
            default:
                x = signed_log_value(rng, -80, -1);
                break;
            }
            const value_sample sample = make_runtime_value("x", x, residual_for(x, rng));
            if (std::fabs(sample.hi + sample.lo) < 1.0)
                samples.push_back({ "random", sample });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_rounding_samples()
    {
        std::vector<unary_sample> samples{
            { "negative large fraction", { "x", -123456.75, 0x1p-48 } },
            { "negative above half", { "x", -0.51, -0x1p-58 } },
            { "negative tiny", { "x", -0x1p-20, 0x1p-80 } },
            { "positive tiny", { "x", 0x1p-20, -0x1p-80 } },
            { "above half", { "x", 0.51, -0x1p-58 } },
            { "large fraction", { "x", 123456.75, -0x1p-48 } }
        };

        sample_rng rng{ 0x128acc9080ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            fltx_type x{};
            switch (index % 4)
            {
            case 0:
                x = uniform_value(rng, -1e6, 1e6);
                break;
            case 1:
                x = fltx_type{ rng.integer(-1000000, 1000000) } + fltx_type{ 0.5 } + residual_for(fltx_type{ 1.0 }, rng);
                break;
            case 2:
                x = fltx_type{ rng.integer(-1000000, 1000000) } + residual_for(fltx_type{ 1.0 }, rng);
                break;
            default:
                x = signed_log_value(rng, -20, 29);
                break;
            }
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_remainder_samples()
    {
        std::vector<binary_sample> samples{
            { "positive", { "x", 5.25, 0x1p-54 }, { "y", 2.0, 0.0 } },
            { "negative dividend", { "x", -5.25, -0x1p-54 }, { "y", 2.0, 0.0 } },
            { "decimal divisor", { "x", 1.0, 0x1p-54 }, { "y", 0.1, 0x1p-58 } },
            { "large half", { "x", 123456789.125, 0x1p-28 }, { "y", 0.5, 0.0 } },
            { "x less than y", { "x", 0.125, 0x1p-58 }, { "y", 10.0, -0x1p-50 } }
        };

        sample_rng rng{ 0x128accf00dull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -40, 40);
            fltx_type y{ 0.0 };
            do
            {
                y = signed_log_value(rng, -16, 16);
            }
            while (y == 0.0);

            samples.push_back({
                "random",
                make_runtime_value("x", x, residual_for(x, rng)),
                make_runtime_value("y", y, residual_for(y, rng))
            });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_int_sample> make_scaling_samples()
    {
        std::vector<unary_int_sample> samples{
            { "zero", { "x", 0.0, 0.0 }, 32 },
            { "one up", { "x", 1.0, 0x1p-54 }, 1 },
            { "one down", { "x", 1.0, -0x1p-54 }, -1 },
            { "negative up", { "x", -1.5, 0x1p-54 }, 10 },
            { "large down", { "x", 1e20, -0x1p+12 }, -32 }
        };

        sample_rng rng{ 0x128acc5ca1eull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = signed_log_value(rng, -60, 60);
            const int n = rng.integer(-512, 512);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)), n });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_erf_samples()
    {
        std::vector<unary_sample> samples{
            { "negative four", { "x", -4.0, 0x1p-52 } },
            { "negative tiny", { "x", -1e-10, 0x1p-88 } },
            { "zero", { "x", 0.0, 0.0 } },
            { "positive tiny", { "x", 1e-10, -0x1p-88 } },
            { "four", { "x", 4.0, -0x1p-52 } }
        };

        sample_rng rng{ 0x128accecfull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            const fltx_type x = uniform_value(rng, -6.0, 6.0);
            samples.push_back({ "random", make_runtime_value("x", x, residual_for(x, rng)) });
        }
        return samples;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_gamma_samples()
    {
        std::vector<unary_sample> samples{
            { "eighth", { "x", 0.125, 0x1p-58 } },
            { "half", { "x", 0.5, 0x1p-58 } },
            { "one", { "x", 1.0, 0.0 } },
            { "one half", { "x", 1.5, -0x1p-54 } },
            { "ten", { "x", 10.0, 0x1p-50 } }
        };

        sample_rng rng{ 0x128acc6a99ull };
        for (std::size_t index = 0; index < random_sample_count; ++index)
        {
            fltx_type x{};
            switch (index % 4)
            {
            case 0:
                x = uniform_value(rng, 0.125, 12.0);
                break;
            case 1:
                x = positive_log_value(rng, -20, 5);
                break;
            case 2:
                x = uniform_value(rng, 0.125, 35.0);
                break;
            default:
                x = uniform_value(rng, 12.0, 35.0);
                break;
            }
            samples.push_back({ "random", make_runtime_value("x", x, positive_residual_for(x, rng)) });
        }
        return samples;
    }

    template<class Samples, class Values, class ToRefFn, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_unary_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        ToRefFn to_reference_value,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const perfect_ref actual = to_reference_value(eval(values[index]));
            const perfect_ref expected = reference(make_perfect(sample.x));
            double bits = matching_bits(actual, expected);
            trace_domain_result(operation, backend_name, index, bits);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (std::isnan(bits))
            {
                if (enforce_required_bits)
                    CHECK(!std::isnan(bits));
                bits = 0.0;
            }
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    template<class Samples, class Values, class ToRefFn, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_binary_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        ToRefFn to_reference_value,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const perfect_ref actual = to_reference_value(eval(values[index].x, values[index].y));
            const perfect_ref expected = reference(make_perfect(sample.x), make_perfect(sample.y));
            double bits = matching_bits(actual, expected);
            trace_domain_result(operation, backend_name, index, bits);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (std::isnan(bits))
            {
                if (enforce_required_bits)
                    CHECK(!std::isnan(bits));
                bits = 0.0;
            }
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    template<class Actual, class Expected>
    [[nodiscard]] inline double integer_match_bits(Actual actual, Expected expected) noexcept
    {
        return actual == expected ? std::numeric_limits<double>::infinity() : 0.0;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_unary_integer_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const auto actual = eval(values[index]);
            const auto expected = reference(make_perfect(sample.x));
            const double bits = integer_match_bits(actual, expected);
            trace_domain_result(operation, backend_name, index, bits);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    [[nodiscard]] inline double bool_match_bits(bool actual, bool expected) noexcept
    {
        constexpr double exact_bool_bits = 160.0;
        return actual == expected ? exact_bool_bits : 0.0;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] special_correctness measure_binary_bool_special_values(
        std::string_view operation,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            bool expected = false;
            try
            {
                expected = reference(sample_as_double(samples[index].x), sample_as_double(samples[index].y));
            }
            catch (...)
            {
                return special_correctness::unavailable;
            }

            try
            {
                const bool actual = eval(values[index].x, values[index].y);
                if (actual != expected)
                {
                    if (special_trace_enabled())
                    {
                        std::cerr << "[special trace] " << operation << " sample " << index
                                  << " '" << samples[index].label << "' mismatch\n";
                    }
                    return special_correctness::fail;
                }
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    template<class Samples, class Values, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_binary_bool_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const bool actual = eval(values[index].x, values[index].y);
            const bool expected = reference(make_perfect(sample.x), make_perfect(sample.y));
            const double bits = bool_match_bits(actual, expected);
            trace_domain_result(operation, backend_name, index, bits);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += bits;
            domain_scores.push_back(domain_sample_score(bits, metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    template<class Samples, class Values, class ToRefFn, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_ternary_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        ToRefFn to_reference_value,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const perfect_ref actual = to_reference_value(eval(values[index].x, values[index].y, values[index].z));
            const perfect_ref expected =
                reference(make_perfect(sample.x), make_perfect(sample.y), make_perfect(sample.z));
            double bits = matching_bits(actual, expected);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (std::isnan(bits))
            {
                if (enforce_required_bits)
                    CHECK(!std::isnan(bits));
                bits = 0.0;
            }
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    template<class Samples, class Values, class ToRefFn, class EvalFn, class RefFn>
    [[nodiscard]] accuracy_result measure_unary_int_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        ToRefFn to_reference_value,
        EvalFn eval,
        RefFn reference)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const perfect_ref actual = to_reference_value(eval(values[index].x, values[index].n));
            const perfect_ref expected = reference(make_perfect(sample.x), sample.n);
            double bits = matching_bits(actual, expected);
            trace_domain_result(operation, backend_name, index, bits);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (std::isnan(bits))
            {
                if (enforce_required_bits)
                    CHECK(!std::isnan(bits));
                bits = 0.0;
            }
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    template<class T>
    [[nodiscard]] perfect_ref rebuild_frexp_result(const frexp_value<T>& value)
    {
        using boost::multiprecision::ldexp;
        return ldexp(to_perfect(value.fraction), value.exponent);
    }

    template<class Samples, class Values, class EvalFn>
    [[nodiscard]] special_correctness measure_frexp_special_values(
        const Samples& samples,
        const Values& values,
        EvalFn eval)
    {
        if (samples.empty())
            return special_correctness::unavailable;

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const special_value_state expected = classify_special_ref(make_perfect(samples[index].x));
            try
            {
                const special_value_state actual = classify_special_ref(rebuild_frexp_result(eval(values[index])));
                if (!special_values_match(actual, expected))
                    return special_correctness::fail;
            }
            catch (...)
            {
                return special_correctness::fail;
            }
        }
        return special_correctness::pass;
    }

    template<class Samples, class Values, class EvalFn>
    [[nodiscard]] accuracy_result measure_frexp_accuracy(
        std::string_view operation,
        std::string_view backend_name,
        double required_bits,
        double metrics_ideal_bits,
        bool enforce_required_bits,
        const Samples& samples,
        const Values& values,
        EvalFn eval)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& sample = samples[index];
            trace_domain_sample(operation, backend_name, index, sample);
            const perfect_ref actual = rebuild_frexp_result(eval(values[index]));
            const perfect_ref expected = make_perfect(sample.x);
            double bits = matching_bits(actual, expected);

            INFO(operation << ' ' << backend_name << " sample '" << sample.label << "' matched " << bits << " bits");
            if (std::isnan(bits))
            {
                if (enforce_required_bits)
                    CHECK(!std::isnan(bits));
                bits = 0.0;
            }
            if (enforce_required_bits)
                CHECK(bits >= required_bits);

            worst_bits = std::min(worst_bits, bits);
            total_bits += finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(domain_score_bits(actual, expected), metrics_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    inline volatile double benchmark_sink = 0.0;

    inline void consume_benchmark_value(const fltx_type& value) noexcept
    {
        benchmark_sink = benchmark_sink + value.hi;
    }

    inline void consume_benchmark_value(const competitor_ref& value)
    {
        benchmark_sink = benchmark_sink + value.backend().crep().first;
    }

    inline void consume_benchmark_value(const extra_competitor_ref& value)
    {
        benchmark_sink = benchmark_sink + value.x[0];
    }

    inline void consume_benchmark_value(bool value) noexcept
    {
        benchmark_sink = benchmark_sink + (value ? 1.0 : 0.0);
    }

    template<class Integer, std::enable_if_t<std::is_integral_v<Integer> && !std::is_same_v<Integer, bool>, int> = 0>
    inline void consume_benchmark_value(Integer value) noexcept
    {
        benchmark_sink = benchmark_sink + static_cast<double>(value);
    }

    template<class T>
    inline void consume_benchmark_value(const frexp_value<T>& value)
    {
        consume_benchmark_value(value.fraction);
        benchmark_sink = benchmark_sink + static_cast<double>(value.exponent);
    }

    [[nodiscard]] inline double benchmark_iteration_scale(std::string_view operation) noexcept
    {
        if (operation == "erf" || operation == "erfc")
            return 0.125;

        if (operation == "lgamma" || operation == "tgamma")
            return 0.2;

        if (operation == "asin" || operation == "acos" || operation == "atan" || operation == "atan2"
            || operation == "atanh" || operation == "asinh" || operation == "acosh" || operation == "pow")
            return 0.5;

        if (operation == "cbrt" || operation == "sin" || operation == "cos" || operation == "tan"
            || operation == "exp" || operation == "exp2" || operation == "expm1" || operation == "log"
            || operation == "log2" || operation == "log10" || operation == "log1p"
            || operation == "sinh" || operation == "cosh" || operation == "tanh")
            return 0.75;

        return 1.0;
    }

    [[nodiscard]] inline std::size_t scaled_benchmark_min_iterations(std::string_view operation) noexcept
    {
        const auto scaled = static_cast<std::size_t>(
            static_cast<double>(benchmark_min_iterations) * benchmark_iteration_scale(operation));
        return std::max<std::size_t>(1, scaled);
    }

    [[nodiscard]] inline std::size_t benchmark_repetitions(
        std::size_t sample_count,
        std::string_view operation = {}) noexcept
    {
        if (sample_count == 0)
            return 0;
        constexpr std::size_t min_repetitions = 3;
        const std::size_t min_iterations = scaled_benchmark_min_iterations(operation);
        return std::max<std::size_t>(min_repetitions, (min_iterations + sample_count - 1) / sample_count);
    }

    template<class Values, class EvalFn>
    [[nodiscard]] benchmark_result benchmark_unary_values(
        const Values& values,
        EvalFn eval,
        std::string_view operation = {})
    {
        const std::size_t repetitions = benchmark_repetitions(values.size(), operation);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (const auto& value : values)
                consume_benchmark_value(eval(value));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    template<class Values, class EvalFn>
    [[nodiscard]] benchmark_result benchmark_binary_values(
        const Values& values,
        EvalFn eval,
        std::string_view operation = {})
    {
        const std::size_t repetitions = benchmark_repetitions(values.size(), operation);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (const auto& value : values)
                consume_benchmark_value(eval(value.x, value.y));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    template<class Values, class EvalFn>
    [[nodiscard]] benchmark_result benchmark_ternary_values(
        const Values& values,
        EvalFn eval,
        std::string_view operation = {})
    {
        const std::size_t repetitions = benchmark_repetitions(values.size(), operation);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (const auto& value : values)
                consume_benchmark_value(eval(value.x, value.y, value.z));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    template<class Values, class EvalFn>
    [[nodiscard]] benchmark_result benchmark_unary_int_values(
        const Values& values,
        EvalFn eval,
        std::string_view operation = {})
    {
        const std::size_t repetitions = benchmark_repetitions(values.size(), operation);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (const auto& value : values)
                consume_benchmark_value(eval(value.x, value.n));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    [[nodiscard]] inline suite_id make_suite(std::string_view operation) noexcept
    {
        return {
            precision_type::f128,
            operation_id{ operation, operation },
            primary_domain
        };
    }

    [[nodiscard]] inline domain_value_kind domain_unary_kind(std::string_view operation) noexcept
    {
        if (operation == "sqrt" || operation == "log" || operation == "log2" || operation == "log10")
            return domain_value_kind::positive;
        if (operation == "asin" || operation == "acos")
            return domain_value_kind::unit_closed;
        if (operation == "sin" || operation == "cos" || operation == "tan")
            return domain_value_kind::reduction_real;
        if (operation == "cbrt")
            return domain_value_kind::balanced_real;
        if (operation == "atanh")
            return domain_value_kind::unit_open;
        if (operation == "exp" || operation == "expm1")
            return domain_value_kind::exp_argument;
        if (operation == "sinh" || operation == "cosh" || operation == "tanh")
            return domain_value_kind::hyperbolic_argument;
        if (operation == "exp2")
            return domain_value_kind::exp2_argument;
        if (operation == "log1p")
            return domain_value_kind::log1p_argument;
        if (operation == "acosh")
            return domain_value_kind::acosh_argument;
        if (operation == "lgamma")
            return domain_value_kind::gamma_argument;
        if (operation == "tgamma")
            return domain_value_kind::tgamma_argument;
        if (operation == "lround" || operation == "llround" ||
            operation == "lrint" || operation == "llrint")
            return domain_value_kind::integer_rounding_argument;
        if (operation == "ilogb")
            return domain_value_kind::nonzero_wide_real;
        return domain_value_kind::wide_real;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_domain_unary_samples(std::string_view operation)
    {
        return make_domain_unary_values(
            domain_unary_kind(operation),
            domain_random_sample_count(),
            domain_seed(operation, 0x128d0a11ull));
    }

    [[nodiscard]] inline std::vector<binary_sample> make_domain_binary_samples(std::string_view operation)
    {
        domain_value_kind x_kind = domain_value_kind::wide_real;
        domain_value_kind y_kind = domain_value_kind::wide_real;

        if (operation == "multiply" || operation == "hypot")
        {
            x_kind = domain_value_kind::balanced_real;
            y_kind = domain_value_kind::balanced_real;
        }
        else if (operation == "divide")
        {
            x_kind = domain_value_kind::balanced_real;
            y_kind = domain_value_kind::nonzero_balanced_real;
        }
        else if (operation == "atan2")
        {
            x_kind = domain_value_kind::nonzero_wide_real;
            y_kind = domain_value_kind::wide_real;
        }
        else if (operation == "pow")
        {
            return make_pow_domain_binary_values(
                domain_random_sample_count(),
                domain_seed(operation, 0x128b1a11ull));
        }
        else if (operation == "fmod" || operation == "remainder" || operation == "remquo")
        {
            x_kind = domain_value_kind::wide_real;
            y_kind = domain_value_kind::nonzero_balanced_real;
        }

        return make_domain_binary_values(
            x_kind,
            y_kind,
            domain_random_sample_count(),
            domain_seed(operation, 0x128b1a11ull));
    }

    [[nodiscard]] inline std::vector<ternary_sample> make_domain_ternary_samples(std::string_view operation)
    {
        return make_domain_ternary_values(
            domain_value_kind::balanced_real,
            domain_value_kind::balanced_real,
            domain_value_kind::balanced_real,
            domain_random_sample_count(),
            domain_seed(operation, 0x1287e311ull));
    }

    [[nodiscard]] inline std::vector<unary_int_sample> make_domain_unary_int_samples(std::string_view operation)
    {
        return make_domain_unary_int_values(
            domain_value_kind::balanced_real,
            domain_random_sample_count(),
            domain_seed(operation, 0x1281e711ull));
    }

    inline void check_competitor_slack(const metrics_record& record)
    {
        INFO("operation: " << record.suite.operation.name);
        INFO("fltx worst bits: " << record.fltx_accuracy.worst_bits);
        INFO("competitor worst bits: " << record.competitor_accuracy.worst_bits);
        CHECK(record.fltx_accuracy.worst_bits + competitor_accuracy_slack_bits >=
              record.competitor_accuracy.worst_bits);
    }

    inline void check_benchmark_claim_is_meaningful(const metrics_record& record)
    {
        if (record.fltx_benchmark.ns_per_iter >= record.competitor_benchmark.ns_per_iter)
            return;

        INFO("operation: " << record.suite.operation.name);
        INFO("fltx ns/iter: " << record.fltx_benchmark.ns_per_iter);
        INFO("competitor ns/iter: " << record.competitor_benchmark.ns_per_iter);
        INFO("fltx worst bits: " << record.fltx_accuracy.worst_bits);
        INFO("competitor worst bits: " << record.competitor_accuracy.worst_bits);
        CHECK(record.fltx_accuracy.worst_bits + competitor_accuracy_slack_bits >=
              record.competitor_accuracy.worst_bits);
    }

    [[nodiscard]] inline bool extra_competitor_special_probe_is_unsafe(std::string_view operation) noexcept
    {
        return operation == "tan" || operation == "atan" || operation == "asin" || operation == "acos";
    }

    [[nodiscard]] inline bool extra_competitor_binary_special_probe_is_unsafe(std::string_view operation) noexcept
    {
        return operation == "atan2";
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_unary_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_unary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_unary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_unary_samples();
        const auto special_fltx_inputs = make_unary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_unary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_unary_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_unary_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_unary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_unary_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_unary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values = extra_competitor_special_probe_is_unsafe(operation)
                    ? special_correctness::fail
                    : measure_unary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_unary_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        auto to_reference_value = [](const auto& value) { return to_perfect(value); };
        record.fltx_accuracy =
            measure_unary_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, to_reference_value, eval, reference);
        record.competitor_accuracy =
            measure_unary_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, to_reference_value, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_unary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values = extra_competitor_special_probe_is_unsafe(operation)
                ? special_correctness::fail
                : measure_unary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_unary_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_unary_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, to_reference_value, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_unary_integer_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_unary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_unary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;

        const auto special_samples = make_special_unary_samples();
        const auto special_fltx_inputs =
            make_unary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_unary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_unary_integer_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_unary_integer_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto special_extra_competitor_inputs =
                make_unary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values =
                measure_unary_integer_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
        }
        else
        {
            extra_competitor.special_values = special_correctness::unavailable;
        }

        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_unary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_unary_values(competitor_inputs, eval, operation);
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                extra_competitor.benchmark = benchmark_unary_values(extra_competitor_inputs, eval, operation);
            }
        }
        if (!measure_accuracy)
            return record;

        record.fltx_accuracy =
            measure_unary_integer_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, eval, reference);
        record.competitor_accuracy =
            measure_unary_integer_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            extra_competitor.accuracy =
                measure_unary_integer_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_binary_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_binary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_binary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_binary_samples();
        const auto special_fltx_inputs = make_binary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_binary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_binary_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_binary_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_binary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_binary_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_binary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_binary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values = extra_competitor_binary_special_probe_is_unsafe(operation)
                    ? special_correctness::fail
                    : measure_binary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_binary_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        auto to_reference_value = [](const auto& value) { return to_perfect(value); };
        record.fltx_accuracy =
            measure_binary_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, to_reference_value, eval, reference);
        record.competitor_accuracy =
            measure_binary_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, to_reference_value, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_binary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_binary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values = extra_competitor_binary_special_probe_is_unsafe(operation)
                ? special_correctness::fail
                : measure_binary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_binary_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_binary_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, to_reference_value, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_binary_bool_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_binary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_binary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_binary_samples();
        const auto special_fltx_inputs =
            make_binary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_binary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_binary_bool_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_binary_bool_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_binary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_binary_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_binary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_binary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values =
                    measure_binary_bool_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_binary_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        record.fltx_accuracy =
            measure_binary_bool_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, eval, reference);
        record.competitor_accuracy =
            measure_binary_bool_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_binary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_binary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values =
                measure_binary_bool_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_binary_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_binary_bool_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_ternary_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_ternary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_ternary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_ternary_samples();
        const auto special_fltx_inputs = make_ternary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_ternary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_ternary_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_ternary_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_ternary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_ternary_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_ternary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_ternary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values =
                    measure_ternary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_ternary_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        auto to_reference_value = [](const auto& value) { return to_perfect(value); };
        record.fltx_accuracy =
            measure_ternary_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, to_reference_value, eval, reference);
        record.competitor_accuracy =
            measure_ternary_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, to_reference_value, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_ternary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_ternary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values =
                measure_ternary_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_ternary_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_ternary_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, to_reference_value, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn, class RefFn>
    [[nodiscard]] metrics_record run_unary_int_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        RefFn reference,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_unary_int_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_unary_int_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_unary_int_samples();
        const auto special_fltx_inputs =
            make_unary_int_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_unary_int_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_unary_int_special_values(operation, special_samples, special_fltx_inputs, eval, reference);
        record.competitor_special_values =
            measure_unary_int_special_values(operation, special_samples, special_competitor_inputs, eval, reference);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_unary_int_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_unary_int_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_unary_int_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_unary_int_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values =
                    measure_unary_int_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_unary_int_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        auto to_reference_value = [](const auto& value) { return to_perfect(value); };
        record.fltx_accuracy =
            measure_unary_int_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, to_reference_value, eval, reference);
        record.competitor_accuracy =
            measure_unary_int_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, to_reference_value, eval, reference);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_unary_int_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_unary_int_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values =
                measure_unary_int_special_values(operation, special_samples, special_extra_competitor_inputs, eval, reference);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_unary_int_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_unary_int_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, to_reference_value, eval, reference);
        }
        return record;
    }

    template<bool ExtraSupported, class Samples, class EvalFn>
    [[nodiscard]] metrics_record run_frexp_case(
        std::string_view operation,
        double required_bits,
        const Samples& samples,
        EvalFn eval,
        bool include_benchmarks,
        bool enforce_fltx_accuracy = true,
        double metrics_ideal_bits = domain_ideal_bits,
        bool measure_accuracy = true)
    {
        const auto fltx_inputs = make_unary_inputs<Samples, fltx_type>(samples, make_fltx);
        const auto competitor_inputs = make_unary_inputs<Samples, competitor_ref>(samples, make_competitor);

        metrics_record record{};
        record.suite = make_suite(operation);
        record.competitor_name = references::competitor_name;
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = references::extra_competitor_name;
        extra_competitor.supported = ExtraSupported;
        const auto special_samples = make_special_unary_samples();
        const auto special_fltx_inputs = make_unary_inputs<decltype(special_samples), fltx_type>(special_samples, make_fltx);
        const auto special_competitor_inputs =
            make_unary_inputs<decltype(special_samples), competitor_ref>(special_samples, make_competitor);
        record.fltx_special_values =
            measure_frexp_special_values(special_samples, special_fltx_inputs, eval);
        record.competitor_special_values =
            measure_frexp_special_values(special_samples, special_competitor_inputs, eval);
        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_unary_values(fltx_inputs, eval, operation);
            record.competitor_benchmark = benchmark_unary_values(competitor_inputs, eval, operation);
        }
        if (!measure_accuracy)
        {
            if constexpr (ExtraSupported)
            {
                const auto extra_competitor_inputs =
                    make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
                const auto special_extra_competitor_inputs =
                    make_unary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
                extra_competitor.special_values =
                    measure_frexp_special_values(special_samples, special_extra_competitor_inputs, eval);
                if (include_benchmarks)
                    extra_competitor.benchmark = benchmark_unary_values(extra_competitor_inputs, eval, operation);
            }
            return record;
        }

        record.fltx_accuracy =
            measure_frexp_accuracy(operation, "fltx", required_bits, metrics_ideal_bits, enforce_fltx_accuracy, samples, fltx_inputs, eval);
        record.competitor_accuracy =
            measure_frexp_accuracy(operation, references::competitor_name, required_bits, metrics_ideal_bits, false, samples, competitor_inputs, eval);
        if constexpr (ExtraSupported)
        {
            const auto extra_competitor_inputs =
                make_unary_inputs<Samples, extra_competitor_ref>(samples, make_extra_competitor);
            const auto special_extra_competitor_inputs =
                make_unary_inputs<decltype(special_samples), extra_competitor_ref>(special_samples, make_extra_competitor);
            extra_competitor.special_values =
                measure_frexp_special_values(special_samples, special_extra_competitor_inputs, eval);
            if (include_benchmarks)
                extra_competitor.benchmark = benchmark_unary_values(extra_competitor_inputs, eval, operation);
            extra_competitor.accuracy =
                measure_frexp_accuracy(operation, references::extra_competitor_name, required_bits, metrics_ideal_bits, false, samples, extra_competitor_inputs, eval);
        }
        return record;
    }
}

#endif
