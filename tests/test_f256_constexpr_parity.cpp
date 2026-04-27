
#include <catch2/catch_test_macros.hpp>

#define FLTX_CONSTEXPR_PARITY_TEST_MODE
#include <f256_math.h>
#include <f256_io.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace
{
using sample_type = bl::f256;
using value_type = bl::f256_s;

#ifdef FLTX_PARITY_SAMPLES_PER_BUCKET
constexpr int kSamplesPerBucket = FLTX_PARITY_SAMPLES_PER_BUCKET;
#else
constexpr int kSamplesPerBucket = 10000;
#endif

constexpr int kBucketCount = 4;
constexpr std::array<const char*, kBucketCount> kBucketNames =
{
    "lead_only",
    "one_tail",
    "two_tails",
    "three_tails"
};

struct forced_path_scope
{
    explicit forced_path_scope(bool force_constexpr) noexcept
    {
        if (force_constexpr)
            bl::_fltx_debug::set_forced_constexpr_path();
        else
            bl::_fltx_debug::set_forced_runtime_path();
    }

    ~forced_path_scope() noexcept
    {
        bl::_fltx_debug::set_forced_runtime_path();
    }
};

template<typename Function>
auto eval_constexpr_path(Function&& function)
{
    forced_path_scope scope(true);
    return function();
}

template<typename Function>
auto eval_runtime_path(Function&& function)
{
    forced_path_scope scope(false);
    return function();
}

[[nodiscard]] std::uint64_t hash_name(const char* text) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    while (*text != '\0')
    {
        hash ^= static_cast<unsigned char>(*text);
        hash *= 1099511628211ull;
        ++text;
    }
    return hash;
}

[[nodiscard]] std::mt19937_64 make_rng(const char* test_name, int bucket) noexcept
{
    return std::mt19937_64{
        0xd6e8feb86659fd93ull ^
        hash_name(test_name) ^
        (0xbf58476d1ce4e5b9ull * static_cast<std::uint64_t>(bucket + 1))
    };
}

[[nodiscard]] double unit_01(std::mt19937_64& rng) noexcept
{
    const std::uint64_t mantissa = (rng() >> 11) | 1ull;
    return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
}

[[nodiscard]] double signed_unit(std::mt19937_64& rng) noexcept
{
    const double value = unit_01(rng);
    return (rng() & 1ull) == 0 ? value : -value;
}

[[nodiscard]] int random_int(std::mt19937_64& rng, int lo, int hi) noexcept
{
    return std::uniform_int_distribution<int>(lo, hi)(rng);
}

[[nodiscard]] double scaled_double(std::mt19937_64& rng, int exp_lo, int exp_hi, bool force_positive = false) noexcept
{
    const int exponent = random_int(rng, exp_lo, exp_hi);
    const double magnitude = std::ldexp(unit_01(rng), exponent);
    if (force_positive)
        return magnitude;
    return (rng() & 1ull) == 0 ? magnitude : -magnitude;
}

[[nodiscard]] value_type special_value(int index)
{
    switch (index & 7)
    {
    case 0: return value_type{ 0.0 };
    case 1: return value_type{ -0.0 };
    case 2: return value_type{ 1.0 };
    case 3: return value_type{ -1.0 };
    case 4: return std::numeric_limits<value_type>::denorm_min();
    case 5: return std::numeric_limits<value_type>::min();
    case 6: return std::numeric_limits<value_type>::max();
    default: return std::numeric_limits<value_type>::lowest();
    }
}

[[nodiscard]] value_type special_classification_value(int index)
{
    switch (index & 7)
    {
    case 0: return value_type{ 0.0 };
    case 1: return value_type{ -0.0 };
    case 2: return std::numeric_limits<value_type>::quiet_NaN();
    case 3: return std::numeric_limits<value_type>::infinity();
    case 4: return -std::numeric_limits<value_type>::infinity();
    case 5: return std::numeric_limits<value_type>::denorm_min();
    case 6: return std::numeric_limits<value_type>::min();
    default: return value_type{ -1.0 };
    }
}

[[nodiscard]] sample_type compose_terms(std::mt19937_64& rng, int term_count, int exp_lo, int exp_hi, int gap_lo, int gap_hi, bool positive_lead = false)
{
    int exponent = random_int(rng, exp_lo, exp_hi);
    sample_type value{ scaled_double(rng, exponent, exponent, positive_lead) };
    for (int index = 1; index < term_count; ++index)
    {
        exponent -= random_int(rng, gap_lo, gap_hi);
        value += sample_type{ scaled_double(rng, exponent, exponent) };
    }
    return value;
}

[[nodiscard]] sample_type gen_any_value(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 31ull) == 0)
        return sample_type{ special_value(bucket + static_cast<int>(rng())) };

    switch (bucket)
    {
    case 0:
        return sample_type{ scaled_double(rng, -20, 20) };

    case 1:
        return compose_terms(rng, 2, -250, 250, 25, 55);

    case 2:
        return compose_terms(rng, 3, -500, 500, 25, 55);

    default:
        return compose_terms(rng, 4, -900, 900, 25, 60);
    }
}

[[nodiscard]] sample_type gen_classification_value(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 3ull) == 0)
        return sample_type{ special_classification_value(bucket + static_cast<int>(rng())) };
    return gen_any_value(rng, bucket);
}

[[nodiscard]] sample_type gen_nonzero_value(std::mt19937_64& rng, int bucket)
{
    sample_type value = gen_any_value(rng, bucket);
    if (bl::iszero(value))
        value = sample_type{ std::ldexp(1.0, -40 - bucket * 100) };
    return value;
}

[[nodiscard]] sample_type gen_positive_nonzero_value(std::mt19937_64& rng, int bucket)
{
    sample_type value = sample_type{ bl::abs(gen_nonzero_value(rng, bucket)) };
    value += sample_type{ std::numeric_limits<double>::denorm_min() };
    return value;
}

[[nodiscard]] sample_type gen_unit_value(std::mt19937_64& rng, int bucket)
{
    sample_type value{ signed_unit(rng) * 0.95 };
    if (bucket >= 1)
        value += sample_type{ std::ldexp(signed_unit(rng), -40) };
    if (bucket >= 2)
        value += sample_type{ std::ldexp(signed_unit(rng), -80) };
    if (bucket >= 3)
        value += sample_type{ std::ldexp(signed_unit(rng), -120) };

    if (value <= sample_type{ -1.0 })
        value = sample_type{ -0.9999999999999999 };
    if (value >= sample_type{ 1.0 })
        value = sample_type{ 0.9999999999999999 };
    return value;
}

[[nodiscard]] sample_type gen_gt_minus_one_value(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 3ull) == 0)
        return gen_unit_value(rng, bucket);

    sample_type value = gen_positive_nonzero_value(rng, bucket);
    if ((rng() & 1ull) != 0)
        value *= sample_type{ 0.125 };
    return value;
}

[[nodiscard]] sample_type gen_ge_one_value(std::mt19937_64& rng, int bucket)
{
    return sample_type{ 1.0 } + gen_positive_nonzero_value(rng, bucket);
}

[[nodiscard]] sample_type gen_gamma_value(std::mt19937_64& rng, int bucket)
{
    sample_type value{ static_cast<double>(random_int(rng, -20, 20)) + signed_unit(rng) };
    if (bl::abs(value - sample_type{ std::round(static_cast<double>(value)) }) < sample_type{ 0.125 } && value <= sample_type{ 0.0 })
        value += sample_type{ 0.375 };

    if (bucket >= 1)
        value += sample_type{ std::ldexp(signed_unit(rng), -40) };
    if (bucket >= 2)
        value += sample_type{ std::ldexp(signed_unit(rng), -80) };
    if (bucket >= 3)
        value += sample_type{ std::ldexp(signed_unit(rng), -120) };

    if (bl::iszero(value))
        value = sample_type{ 0.375 };

    return value;
}

[[nodiscard]] int gen_exponent_value(std::mt19937_64& rng, int bucket) noexcept
{
    switch (bucket)
    {
    case 0: return random_int(rng, -32, 32);
    case 1: return random_int(rng, -256, 256);
    case 2: return random_int(rng, -1024, 1024);
    default: return random_int(rng, -4096, 4096);
    }
}

[[nodiscard]] int gen_round_digits_value(std::mt19937_64& rng, int bucket) noexcept
{
    switch (bucket)
    {
    case 0: return random_int(rng, -8, 8);
    case 1: return random_int(rng, -24, 24);
    case 2: return random_int(rng, -48, 48);
    default: return random_int(rng, -80, 80);
    }
}

[[nodiscard]] int gen_pow10_exponent(std::mt19937_64& rng, int bucket) noexcept
{
    switch (bucket)
    {
    case 0: return random_int(rng, -16, 16);
    case 1: return random_int(rng, -64, 64);
    case 2: return random_int(rng, -128, 128);
    default: return random_int(rng, -300, 300);
    }
}

[[nodiscard]] std::string describe_bits(double value)
{
    std::ostringstream stream;
    stream << std::hex << std::showbase << std::bit_cast<std::uint64_t>(value);
    return stream.str();
}

[[nodiscard]] std::string describe(const value_type& value)
{
    std::ostringstream stream;
    stream << value
           << " [decimal=" << bl::to_string(value, std::numeric_limits<value_type>::max_digits10, false, true)
           << ", x0=" << describe_bits(value.x0)
           << ", x1=" << describe_bits(value.x1)
           << ", x2=" << describe_bits(value.x2)
           << ", x3=" << describe_bits(value.x3) << "]";
    return stream.str();
}

[[nodiscard]] std::string describe(long double value)
{
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
    return stream.str();
}

[[nodiscard]] std::string describe(double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value
           << " [decimal=" << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific << value
           << std::defaultfloat << ", bits=" << describe_bits(value) << "]";
    return stream.str();
}

[[nodiscard]] std::string describe(bool value)
{
    return value ? "true" : "false";
}

template<typename T>
[[nodiscard]] std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, std::string>
describe(T value)
{
    return std::to_string(value);
}

[[nodiscard]] bool string_equals(const char* lhs, const char* rhs) noexcept
{
    while (*lhs != '\0' && *rhs != '\0')
    {
        if (*lhs != *rhs)
            return false;

        ++lhs;
        ++rhs;
    }

    return *lhs == *rhs;
}

[[nodiscard]] double constexpr_dekker_max_safe_magnitude() noexcept
{
    constexpr double split = 134217729.0;
    return std::numeric_limits<double>::max() / split;
}

[[nodiscard]] bool double_has_constexpr_dekker_overflow_risk(double value) noexcept
{
    return std::isfinite(value) && std::fabs(value) > constexpr_dekker_max_safe_magnitude();
}

[[nodiscard]] bool double_recip_estimate_has_constexpr_dekker_overflow_risk(double value) noexcept
{
    if (!std::isfinite(value) || value == 0.0)
        return false;

    return std::fabs(value) < (1.0 / constexpr_dekker_max_safe_magnitude());
}

[[nodiscard]] bool value_has_constexpr_dekker_overflow_risk(const value_type& value) noexcept
{
    if (double_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x1))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x2))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x3))
        return true;

    return false;
}

[[nodiscard]] bool value_has_constexpr_recip_dekker_overflow_risk(const value_type& value) noexcept
{
    if (double_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    if (double_recip_estimate_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    return false;
}

[[nodiscard]] bool value_has_constexpr_dekker_overflow_risk(const sample_type& value) noexcept
{
    if (double_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x1))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x2))
        return true;

    if (double_has_constexpr_dekker_overflow_risk(value.x3))
        return true;

    return false;
}

[[nodiscard]] bool value_has_constexpr_recip_dekker_overflow_risk(const sample_type& value) noexcept
{
    if (double_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    if (double_recip_estimate_has_constexpr_dekker_overflow_risk(value.x0))
        return true;

    return false;
}

template<typename T>
[[nodiscard]] bool value_has_constexpr_dekker_overflow_risk(const T&) noexcept
{
    return false;
}

template<typename T>
[[nodiscard]] bool value_has_constexpr_recip_dekker_overflow_risk(const T&) noexcept
{
    return false;
}

template<typename... Values>
[[nodiscard]] bool tuple_has_constexpr_dekker_overflow_risk(const std::tuple<Values...>& values) noexcept
{
    return std::apply([](const auto&... item)
    {
        return (... || value_has_constexpr_dekker_overflow_risk(item));
    }, values);
}

template<typename... Values>
[[nodiscard]] bool tuple_has_constexpr_recip_dekker_overflow_risk(const std::tuple<Values...>& values) noexcept
{
    return std::apply([](const auto&... item)
    {
        return (... || value_has_constexpr_recip_dekker_overflow_risk(item));
    }, values);
}

[[nodiscard]] bool test_can_use_constexpr_dekker_multiply(const char* test_name) noexcept
{
    return
        string_equals(test_name, "fma");
}

[[nodiscard]] bool test_can_use_constexpr_recip_dekker_multiply(const char* test_name) noexcept
{
    return
        string_equals(test_name, "recip") ||
        string_equals(test_name, "inv");
}

template<typename... Values>
[[nodiscard]] bool should_skip_constexpr_dekker_extreme_case(
    const char* test_name,
    const std::tuple<Values...>& values) noexcept
{
    if (test_can_use_constexpr_dekker_multiply(test_name) &&
        tuple_has_constexpr_dekker_overflow_risk(values))
    {
        return true;
    }

    return
        test_can_use_constexpr_recip_dekker_multiply(test_name) &&
        tuple_has_constexpr_recip_dekker_overflow_risk(values);
}

[[nodiscard]] bool equal(const value_type& lhs, const value_type& rhs) noexcept
{
    return std::bit_cast<std::uint64_t>(lhs.x0) == std::bit_cast<std::uint64_t>(rhs.x0) &&
           std::bit_cast<std::uint64_t>(lhs.x1) == std::bit_cast<std::uint64_t>(rhs.x1) &&
           std::bit_cast<std::uint64_t>(lhs.x2) == std::bit_cast<std::uint64_t>(rhs.x2) &&
           std::bit_cast<std::uint64_t>(lhs.x3) == std::bit_cast<std::uint64_t>(rhs.x3);
}

[[nodiscard]] bool equal(double lhs, double rhs) noexcept
{
    return std::bit_cast<std::uint64_t>(lhs) == std::bit_cast<std::uint64_t>(rhs);
}

[[nodiscard]] bool equal(long double lhs, long double rhs) noexcept
{
    return lhs == rhs;
}

[[nodiscard]] bool equal(bool lhs, bool rhs) noexcept
{
    return lhs == rhs;
}

template<typename T>
[[nodiscard]] std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, bool>
equal(T lhs, T rhs) noexcept
{
    return lhs == rhs;
}

template<typename... Values, std::size_t... Indices>
[[nodiscard]] std::string describe_tuple_impl(const std::tuple<Values...>& values, std::index_sequence<Indices...>)
{
    std::ostringstream stream;
    stream << '(';
    std::size_t index = 0;
    ((stream << (index++ == 0 ? "" : ", ") << describe(std::get<Indices>(values))), ...);
    stream << ')';
    return stream.str();
}

template<typename... Values>
[[nodiscard]] std::string describe_tuple(const std::tuple<Values...>& values)
{
    return describe_tuple_impl(values, std::index_sequence_for<Values...>{});
}

template<typename Value>
struct unary_out_result
{
    Value value;
    int extra;
};

template<typename Value>
[[nodiscard]] bool equal(const unary_out_result<Value>& lhs, const unary_out_result<Value>& rhs) noexcept
{
    return equal(lhs.value, rhs.value) && lhs.extra == rhs.extra;
}

template<typename Value>
[[nodiscard]] std::string describe(const unary_out_result<Value>& result)
{
    std::ostringstream stream;
    stream << "{value=" << describe(result.value) << ", extra=" << result.extra << '}';
    return stream.str();
}

template<typename Value>
struct binary_out_result
{
    Value value;
    int extra;
};

template<typename Value>
[[nodiscard]] bool equal(const binary_out_result<Value>& lhs, const binary_out_result<Value>& rhs) noexcept
{
    return equal(lhs.value, rhs.value) && lhs.extra == rhs.extra;
}

template<typename Value>
[[nodiscard]] std::string describe(const binary_out_result<Value>& result)
{
    std::ostringstream stream;
    stream << "{value=" << describe(result.value) << ", extra=" << result.extra << '}';
    return stream.str();
}

template<typename Value>
struct split_result
{
    Value fractional;
    Value integral;
};

template<typename Value>
[[nodiscard]] bool equal(const split_result<Value>& lhs, const split_result<Value>& rhs) noexcept
{
    return equal(lhs.fractional, rhs.fractional) && equal(lhs.integral, rhs.integral);
}

template<typename Value>
[[nodiscard]] std::string describe(const split_result<Value>& result)
{
    std::ostringstream stream;
    stream << "{fractional=" << describe(result.fractional) << ", integral=" << describe(result.integral) << '}';
    return stream.str();
}

template<typename Value>
struct sincos_result
{
    bool ok;
    Value sine;
    Value cosine;
};

template<typename Value>
[[nodiscard]] bool equal(const sincos_result<Value>& lhs, const sincos_result<Value>& rhs) noexcept
{
    if (lhs.ok != rhs.ok)
        return false;
    if (!lhs.ok)
        return true;
    return equal(lhs.sine, rhs.sine) && equal(lhs.cosine, rhs.cosine);
}

template<typename Value>
[[nodiscard]] std::string describe(const sincos_result<Value>& result)
{
    std::ostringstream stream;
    stream << "{ok=" << describe(result.ok);
    if (result.ok)
        stream << ", sine=" << describe(result.sine) << ", cosine=" << describe(result.cosine);
    stream << '}';
    return stream.str();
}

template<typename Generator, typename Function>
void run_tuple_test(const char* test_name, Generator&& generator, Function&& function)
{
    bl::_fltx_debug::set_forced_runtime_path();

    for (int bucket = 0; bucket < kBucketCount; ++bucket)
    {
        auto rng = make_rng(test_name, bucket);
        for (int iteration = 0; iteration < kSamplesPerBucket; ++iteration)
        {
            const auto args = generator(rng, bucket);
            if (should_skip_constexpr_dekker_extreme_case(test_name, args))
                continue;

            const auto constexpr_result = eval_constexpr_path([&]() { return std::apply(function, args); });
            const auto runtime_result = eval_runtime_path([&]() { return std::apply(function, args); });

            INFO("function=" << test_name << ", bucket=" << kBucketNames[static_cast<std::size_t>(bucket)] << ", iteration=" << iteration);
            INFO("args=" << describe_tuple(args));
            INFO("constexpr=" << describe(constexpr_result));
            INFO("runtime=" << describe(runtime_result));
            REQUIRE(equal(constexpr_result, runtime_result));
        }
    }
}

[[nodiscard]] auto gen_unary_any(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_classification(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_classification_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_nonzero(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_nonzero_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_positive(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_positive_nonzero_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_unit(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_unit_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_gt_minus_one(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_gt_minus_one_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_ge_one(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_ge_one_value(rng, bucket) };
}

[[nodiscard]] auto gen_unary_gamma(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_gamma_value(rng, bucket) };
}

[[nodiscard]] auto gen_binary_any(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_binary_classification(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_classification_value(rng, bucket), gen_classification_value(rng, bucket) };
}

[[nodiscard]] auto gen_binary_rhs_nonzero(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_nonzero_value(rng, bucket) };
}

[[nodiscard]] auto gen_fma_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_any_value(rng, bucket), gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_ldexp_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_exponent_value(rng, bucket) };
}

[[nodiscard]] auto gen_scalbln_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), static_cast<long>(gen_exponent_value(rng, bucket)) };
}

[[nodiscard]] auto gen_round_digits_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_round_digits_value(rng, bucket) };
}

[[nodiscard]] auto gen_pow10_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_pow10_exponent(rng, bucket) };
}

[[nodiscard]] auto gen_nexttoward_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), static_cast<long double>(scaled_double(rng, -300, 300)) };
}

[[nodiscard]] auto gen_clamp_args(std::mt19937_64& rng, int bucket)
{
    sample_type a = gen_any_value(rng, bucket);
    sample_type b = gen_any_value(rng, bucket);
    if (b < a)
        std::swap(a, b);
    return std::tuple{ gen_any_value(rng, bucket), a, b };
}

[[nodiscard]] auto gen_pow_args(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 3ull) == 0)
    {
        sample_type base{ static_cast<double>(random_int(rng, -16, 16)) };
        if (bl::iszero(base))
            base = sample_type{ -2.0 };
        if (base > sample_type{ 0.0 })
            base = -base;
        sample_type exponent{ static_cast<double>(random_int(rng, -12, 12)) };
        return std::tuple{ base, exponent };
    }

    return std::tuple{ gen_positive_nonzero_value(rng, bucket), gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_pow_double_args(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 3ull) == 0)
    {
        sample_type base{ static_cast<double>(random_int(rng, -16, 16)) };
        if (bl::iszero(base))
            base = sample_type{ -2.0 };
        if (base > sample_type{ 0.0 })
            base = -base;
        return std::tuple{ base, static_cast<double>(random_int(rng, -12, 12)) };
    }

    return std::tuple{ gen_positive_nonzero_value(rng, bucket), static_cast<double>(scaled_double(rng, -20, 20)) };
}

#define FLTX_TEST_UNARY(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x) { return bl::NAME(x); }); \
}

#define FLTX_TEST_BINARY(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x, const value_type& y) { return bl::NAME(x, y); }); \
}

#define FLTX_TEST_BINARY_LONG(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x, long double y) { return bl::NAME(x, y); }); \
}

#define FLTX_TEST_TERNARY(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x, const value_type& y, const value_type& z) { return bl::NAME(x, y, z); }); \
}

#define FLTX_TEST_VALUE_INT(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x, int exponent) { return bl::NAME(x, exponent); }); \
}

#define FLTX_TEST_VALUE_LONG(NAME, GENERATOR) \
TEST_CASE("f256 constexpr parity: " #NAME, "[fltx][constexpr][parity][f256][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](const value_type& x, long exponent) { return bl::NAME(x, exponent); }); \
}

FLTX_TEST_UNARY(isnan, gen_unary_classification)
FLTX_TEST_UNARY(isinf, gen_unary_classification)
FLTX_TEST_UNARY(isfinite, gen_unary_classification)
FLTX_TEST_UNARY(iszero, gen_unary_any)
FLTX_TEST_UNARY(ispositive, gen_unary_any)
FLTX_TEST_UNARY(signbit, gen_unary_classification)
FLTX_TEST_UNARY(fpclassify, gen_unary_classification)
FLTX_TEST_UNARY(isnormal, gen_unary_classification)
FLTX_TEST_BINARY(isunordered, gen_binary_classification)
FLTX_TEST_BINARY(isgreater, gen_binary_any)
FLTX_TEST_BINARY(isgreaterequal, gen_binary_any)
FLTX_TEST_BINARY(isless, gen_binary_any)
FLTX_TEST_BINARY(islessequal, gen_binary_any)
FLTX_TEST_BINARY(islessgreater, gen_binary_any)
FLTX_TEST_UNARY(recip, gen_unary_nonzero)
FLTX_TEST_UNARY(inv, gen_unary_nonzero)

TEST_CASE("f256 constexpr parity: clamp", "[fltx][constexpr][parity][f256][clamp]")
{
    run_tuple_test("clamp", gen_clamp_args, [](const value_type& x, const value_type& lo, const value_type& hi)
    {
        return bl::clamp(x, lo, hi);
    });
}

FLTX_TEST_UNARY(abs, gen_unary_any)
FLTX_TEST_UNARY(fabs, gen_unary_any)
FLTX_TEST_UNARY(floor, gen_unary_any)
FLTX_TEST_UNARY(ceil, gen_unary_any)
FLTX_TEST_UNARY(trunc, gen_unary_any)
FLTX_TEST_BINARY(fmod, gen_binary_rhs_nonzero)
FLTX_TEST_UNARY(round, gen_unary_any)
TEST_CASE("f256 constexpr parity: round_to_decimals", "[fltx][constexpr][parity][f256][round_to_decimals]")
{
    run_tuple_test("round_to_decimals", gen_round_digits_args, [](const value_type& x, int digits)
    {
        return bl::round_to_decimals(x, digits);
    });
}
FLTX_TEST_UNARY(sqrt, gen_unary_positive)
FLTX_TEST_UNARY(nearbyint, gen_unary_any)
FLTX_TEST_UNARY(log_as_double, gen_unary_positive)
FLTX_TEST_VALUE_INT(ldexp, gen_ldexp_args)
FLTX_TEST_UNARY(exp, gen_unary_any)
FLTX_TEST_UNARY(exp2, gen_unary_any)
FLTX_TEST_UNARY(log, gen_unary_positive)
FLTX_TEST_UNARY(log2, gen_unary_positive)
FLTX_TEST_UNARY(log10, gen_unary_positive)
TEST_CASE("f256 constexpr parity: pow", "[fltx][constexpr][parity][f256][pow]")
{
    run_tuple_test("pow", gen_pow_args, [](const value_type& x, const value_type& y) { return bl::pow(x, y); });
}
TEST_CASE("f256 constexpr parity: pow(double)", "[fltx][constexpr][parity][f256][pow_double]")
{
    run_tuple_test("pow_double", gen_pow_double_args, [](const value_type& x, double y) { return bl::pow(x, y); });
}
TEST_CASE("f256 constexpr parity: pow10_256", "[fltx][constexpr][parity][f256][pow10_256]")
{
    run_tuple_test("pow10_256", gen_pow10_args, [](int exponent) { return bl::pow10_256(exponent); });
}

TEST_CASE("f256 constexpr parity: sincos", "[fltx][constexpr][parity][f256][sincos]")
{
    run_tuple_test("sincos", gen_unary_any, [](const value_type& x)
    {
        value_type sine = std::numeric_limits<value_type>::quiet_NaN();
        value_type cosine = std::numeric_limits<value_type>::quiet_NaN();
        const bool ok = bl::sincos(x, sine, cosine);
        return sincos_result<value_type>{ ok, sine, cosine };
    });
}

FLTX_TEST_UNARY(sin, gen_unary_any)
FLTX_TEST_UNARY(cos, gen_unary_any)
FLTX_TEST_UNARY(tan, gen_unary_any)
FLTX_TEST_UNARY(atan, gen_unary_any)
FLTX_TEST_BINARY(atan2, gen_binary_any)
FLTX_TEST_UNARY(asin, gen_unary_unit)
FLTX_TEST_UNARY(acos, gen_unary_unit)
FLTX_TEST_UNARY(expm1, gen_unary_any)
FLTX_TEST_UNARY(log1p, gen_unary_gt_minus_one)
FLTX_TEST_UNARY(sinh, gen_unary_any)
FLTX_TEST_UNARY(cosh, gen_unary_any)
FLTX_TEST_UNARY(tanh, gen_unary_any)
FLTX_TEST_UNARY(asinh, gen_unary_any)
FLTX_TEST_UNARY(acosh, gen_unary_ge_one)
FLTX_TEST_UNARY(atanh, gen_unary_unit)
FLTX_TEST_UNARY(cbrt, gen_unary_any)
FLTX_TEST_BINARY(hypot, gen_binary_any)
FLTX_TEST_UNARY(rint, gen_unary_any)
FLTX_TEST_UNARY(lround, gen_unary_any)
FLTX_TEST_UNARY(llround, gen_unary_any)
FLTX_TEST_UNARY(lrint, gen_unary_any)
FLTX_TEST_UNARY(llrint, gen_unary_any)

TEST_CASE("f256 constexpr parity: remquo", "[fltx][constexpr][parity][f256][remquo]")
{
    run_tuple_test("remquo", gen_binary_rhs_nonzero, [](const value_type& x, const value_type& y)
    {
        int quotient = 0;
        return binary_out_result<value_type>{ bl::remquo(x, y, &quotient), quotient };
    });
}

FLTX_TEST_BINARY(remainder, gen_binary_rhs_nonzero)
FLTX_TEST_TERNARY(fma, gen_fma_args)
FLTX_TEST_BINARY(fmin, gen_binary_any)
FLTX_TEST_BINARY(fmax, gen_binary_any)
FLTX_TEST_BINARY(fdim, gen_binary_any)
FLTX_TEST_BINARY(copysign, gen_binary_any)

TEST_CASE("f256 constexpr parity: frexp", "[fltx][constexpr][parity][f256][frexp]")
{
    run_tuple_test("frexp", gen_unary_any, [](const value_type& x)
    {
        int exponent = 0;
        return unary_out_result<value_type>{ bl::frexp(x, &exponent), exponent };
    });
}

TEST_CASE("f256 constexpr parity: modf", "[fltx][constexpr][parity][f256][modf]")
{
    run_tuple_test("modf", gen_unary_any, [](const value_type& x)
    {
        value_type integral = std::numeric_limits<value_type>::quiet_NaN();
        return split_result<value_type>{ bl::modf(x, &integral), integral };
    });
}

FLTX_TEST_UNARY(ilogb, gen_unary_nonzero)
FLTX_TEST_UNARY(logb, gen_unary_nonzero)
FLTX_TEST_VALUE_INT(scalbn, gen_ldexp_args)
FLTX_TEST_VALUE_LONG(scalbln, gen_scalbln_args)
FLTX_TEST_BINARY(nextafter, gen_binary_any)
FLTX_TEST_BINARY_LONG(nexttoward, gen_nexttoward_args)
FLTX_TEST_UNARY(erfc, gen_unary_any)
FLTX_TEST_UNARY(erf, gen_unary_any)
FLTX_TEST_UNARY(lgamma, gen_unary_gamma)
FLTX_TEST_UNARY(tgamma, gen_unary_gamma)

#undef FLTX_TEST_UNARY
#undef FLTX_TEST_BINARY
#undef FLTX_TEST_BINARY_LONG
#undef FLTX_TEST_TERNARY
#undef FLTX_TEST_VALUE_INT
#undef FLTX_TEST_VALUE_LONG
} // namespace
