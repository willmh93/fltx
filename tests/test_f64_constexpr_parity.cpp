#include <catch2/catch_test_macros.hpp>

#define FLTX_CONSTEXPR_PARITY
#define FLTX_CONSTEXPR_PARITY_TEST_MODE
#include <f64_math.h>

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
#ifdef FLTX_PARITY_SAMPLES_PER_BUCKET
constexpr int kSamplesPerBucket = FLTX_PARITY_SAMPLES_PER_BUCKET;
#else
constexpr int kSamplesPerBucket = 10000;
#endif

constexpr int kBucketCount = 4;
constexpr std::array<const char*, kBucketCount> kBucketNames =
{
    "simple",
    "broad",
    "integer_adjacent",
    "edge_range"
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
        0x9e3779b97f4a7c15ull ^
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

[[nodiscard]] long long random_long_long(std::mt19937_64& rng, long long lo, long long hi) noexcept
{
    return std::uniform_int_distribution<long long>(lo, hi)(rng);
}

[[nodiscard]] double scaled_double(std::mt19937_64& rng, int exp_lo, int exp_hi, bool force_positive = false) noexcept
{
    const int exponent = random_int(rng, exp_lo, exp_hi);
    const double magnitude = std::ldexp(unit_01(rng), exponent);
    if (force_positive)
        return magnitude;
    return (rng() & 1ull) == 0 ? magnitude : -magnitude;
}

[[nodiscard]] double special_f64(std::mt19937_64& rng, int bucket) noexcept
{
    static constexpr std::array<double, 8> special_values =
    {
        0.0,
        -0.0,
        1.0,
        -1.0,
        std::numeric_limits<double>::denorm_min(),
        -std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(),
        -std::numeric_limits<double>::min()
    };

    if ((rng() & 31ull) != 0)
        return std::numeric_limits<double>::quiet_NaN();

    const std::size_t index = static_cast<std::size_t>((bucket * 3 + static_cast<int>(rng() % special_values.size())) % static_cast<int>(special_values.size()));
    return special_values[index];
}

[[nodiscard]] double special_classification_f64(std::mt19937_64& rng, int bucket) noexcept
{
    static constexpr std::array<double, 8> special_values =
    {
        0.0,
        -0.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::denorm_min(),
        -std::numeric_limits<double>::denorm_min(),
        -1.0
    };

    const std::size_t index = static_cast<std::size_t>((bucket * 5 + static_cast<int>(rng() % special_values.size())) % static_cast<int>(special_values.size()));
    return special_values[index];
}

[[nodiscard]] double gen_any_value(std::mt19937_64& rng, int bucket) noexcept
{
    if ((rng() & 31ull) == 0)
    {
        const double value = special_f64(rng, bucket);
        if (!std::isnan(value))
            return value;
    }

    switch (bucket)
    {
    case 0:
        return scaled_double(rng, -20, 20);

    case 1:
        return scaled_double(rng, -250, 250);

    case 2:
    {
        const double base = static_cast<double>(random_long_long(rng, -1000000, 1000000));
        const double delta = std::ldexp(signed_unit(rng), -random_int(rng, 10, 50));
        return base + delta;
    }

    default:
    {
        if ((rng() & 1ull) == 0)
            return scaled_double(rng, -1074, -1020);
        return scaled_double(rng, 900, 1023);
    }
    }
}

[[nodiscard]] double gen_nonzero_value(std::mt19937_64& rng, int bucket) noexcept
{
    double value = gen_any_value(rng, bucket);
    if (value == 0.0)
        value = bucket == 3 ? std::numeric_limits<double>::denorm_min() : std::ldexp(1.0, -40 - bucket * 100);
    return value;
}

[[nodiscard]] double gen_positive_nonzero_value(std::mt19937_64& rng, int bucket) noexcept
{
    double value = std::fabs(gen_nonzero_value(rng, bucket));
    if (value == 0.0)
        value = std::numeric_limits<double>::denorm_min();
    return value;
}

[[nodiscard]] double gen_unit_value(std::mt19937_64& rng, int bucket) noexcept
{
    switch (bucket)
    {
    case 0:
        return signed_unit(rng) * 0.75;
    case 1:
        return signed_unit(rng) * 0.95;
    case 2:
    {
        const double base = static_cast<double>(random_int(rng, -1, 1));
        const double delta = std::ldexp(signed_unit(rng), -random_int(rng, 4, 40));
        double value = base + delta;
        if (value <= -1.0)
            value = -0.999999999999;
        if (value >= 1.0)
            value = 0.999999999999;
        return value;
    }
    default:
        return signed_unit(rng) * (1.0 - std::ldexp(1.0, -40));
    }
}

[[nodiscard]] double gen_gt_minus_one_value(std::mt19937_64& rng, int bucket) noexcept
{
    if ((rng() & 3ull) == 0)
        return signed_unit(rng) * 0.99;

    double value = gen_positive_nonzero_value(rng, bucket);
    if ((rng() & 1ull) != 0)
        value *= 0.125;
    return value;
}

[[nodiscard]] double gen_ge_one_value(std::mt19937_64& rng, int bucket) noexcept
{
    return 1.0 + gen_positive_nonzero_value(rng, bucket);
}

[[nodiscard]] double gen_gamma_value(std::mt19937_64& rng, int bucket) noexcept
{
    double value = static_cast<double>(random_int(rng, -20, 20)) + signed_unit(rng);
    if (std::fabs(value - std::round(value)) < 0.125 && value <= 0.0)
        value += 0.375;

    if (bucket >= 2)
        value += std::ldexp(signed_unit(rng), -random_int(rng, 8, 40));

    if (value == 0.0)
        value = 0.375;

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

[[nodiscard]] std::string describe(double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value
           << " [decimal=" << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific << value
           << std::defaultfloat << ", bits=" << describe_bits(value) << "]";
    return stream.str();
}

[[nodiscard]] std::string describe(long double value)
{
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
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
    if ((rng() & 3ull) == 0)
        return std::tuple{ special_classification_f64(rng, bucket) };
    return std::tuple{ gen_any_value(rng, bucket) };
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
    const double lhs = (rng() & 3ull) == 0 ? special_classification_f64(rng, bucket) : gen_any_value(rng, bucket);
    const double rhs = (rng() & 3ull) == 0 ? special_classification_f64(rng, bucket + 1) : gen_any_value(rng, bucket);
    return std::tuple{ lhs, rhs };
}

[[nodiscard]] auto gen_binary_rhs_nonzero(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_nonzero_value(rng, bucket) };
}

[[nodiscard]] auto gen_binary_positive_any(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_positive_nonzero_value(rng, bucket), gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_binary_unit(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_unit_value(rng, bucket), gen_unit_value(rng, bucket) };
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

[[nodiscard]] auto gen_nexttoward_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), static_cast<long double>(gen_any_value(rng, bucket)) };
}

[[nodiscard]] auto gen_pow_args(std::mt19937_64& rng, int bucket)
{
    if ((rng() & 3ull) == 0)
    {
        double base = static_cast<double>(random_int(rng, -16, 16));
        if (base == 0.0)
            base = -2.0;
        if (base > 0.0)
            base = -base;
        const double exponent = static_cast<double>(random_int(rng, -12, 12));
        return std::tuple{ base, exponent };
    }

    return std::tuple{ gen_positive_nonzero_value(rng, bucket), gen_any_value(rng, bucket) };
}

[[nodiscard]] auto gen_remquo_args(std::mt19937_64& rng, int bucket)
{
    return std::tuple{ gen_any_value(rng, bucket), gen_nonzero_value(rng, bucket) };
}

#define FLTX_TEST_UNARY(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x) { return bl::NAME(x); }); \
}

#define FLTX_TEST_BINARY(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x, double y) { return bl::NAME(x, y); }); \
}

#define FLTX_TEST_BINARY_LONG(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x, long double y) { return bl::NAME(x, y); }); \
}

#define FLTX_TEST_TERNARY(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x, double y, double z) { return bl::NAME(x, y, z); }); \
}

#define FLTX_TEST_UNARY_INT(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x, int exponent) { return bl::NAME(x, exponent); }); \
}

#define FLTX_TEST_UNARY_LONG(NAME, GENERATOR) \
TEST_CASE("f64 constexpr parity: " #NAME, "[fltx][constexpr][parity][f64][" #NAME "]") \
{ \
    run_tuple_test(#NAME, GENERATOR, [](double x, long exponent) { return bl::NAME(x, exponent); }); \
}

FLTX_TEST_UNARY(abs, gen_unary_any)
FLTX_TEST_UNARY(fabs, gen_unary_any)
FLTX_TEST_UNARY(signbit, gen_unary_classification)
FLTX_TEST_UNARY(isnan, gen_unary_classification)
FLTX_TEST_UNARY(isinf, gen_unary_classification)
FLTX_TEST_UNARY(isfinite, gen_unary_classification)
FLTX_TEST_UNARY(iszero, gen_unary_any)
FLTX_TEST_UNARY(floor, gen_unary_any)
FLTX_TEST_UNARY(ceil, gen_unary_any)
FLTX_TEST_UNARY(trunc, gen_unary_any)
FLTX_TEST_UNARY(round, gen_unary_any)
FLTX_TEST_UNARY(nearbyint, gen_unary_any)
FLTX_TEST_UNARY(rint, gen_unary_any)
FLTX_TEST_UNARY(lround, gen_unary_any)
FLTX_TEST_UNARY(llround, gen_unary_any)
FLTX_TEST_UNARY(lrint, gen_unary_any)
FLTX_TEST_UNARY(llrint, gen_unary_any)
FLTX_TEST_BINARY(fmod, gen_binary_rhs_nonzero)
FLTX_TEST_BINARY(remainder, gen_binary_rhs_nonzero)

TEST_CASE("f64 constexpr parity: remquo", "[fltx][constexpr][parity][f64][remquo]")
{
    run_tuple_test("remquo", gen_remquo_args, [](double x, double y)
    {
        int quotient = 0;
        return binary_out_result<double>{ bl::remquo(x, y, &quotient), quotient };
    });
}

FLTX_TEST_TERNARY(fma, gen_fma_args)
FLTX_TEST_BINARY(fmin, gen_binary_any)
FLTX_TEST_BINARY(fmax, gen_binary_any)
FLTX_TEST_BINARY(fdim, gen_binary_any)
FLTX_TEST_BINARY(copysign, gen_binary_any)
FLTX_TEST_UNARY_INT(ldexp, gen_ldexp_args)
FLTX_TEST_UNARY_INT(scalbn, gen_ldexp_args)
FLTX_TEST_UNARY_LONG(scalbln, gen_scalbln_args)

TEST_CASE("f64 constexpr parity: frexp", "[fltx][constexpr][parity][f64][frexp]")
{
    run_tuple_test("frexp", gen_unary_any, [](double x)
    {
        int exponent = 0;
        return unary_out_result<double>{ bl::frexp(x, &exponent), exponent };
    });
}

TEST_CASE("f64 constexpr parity: modf", "[fltx][constexpr][parity][f64][modf]")
{
    run_tuple_test("modf", gen_unary_any, [](double x)
    {
        double integral = 0.0;
        return split_result<double>{ bl::modf(x, &integral), integral };
    });
}

FLTX_TEST_UNARY(ilogb, gen_unary_nonzero)
FLTX_TEST_UNARY(logb, gen_unary_nonzero)
FLTX_TEST_BINARY(nextafter, gen_binary_any)
FLTX_TEST_BINARY_LONG(nexttoward, gen_nexttoward_args)
FLTX_TEST_UNARY(exp, gen_unary_any)
FLTX_TEST_UNARY(exp2, gen_unary_any)
FLTX_TEST_UNARY(expm1, gen_unary_any)
FLTX_TEST_UNARY(log, gen_unary_positive)
FLTX_TEST_UNARY(log2, gen_unary_positive)
FLTX_TEST_UNARY(log10, gen_unary_positive)
FLTX_TEST_UNARY(log1p, gen_unary_gt_minus_one)
FLTX_TEST_UNARY(sqrt, gen_unary_positive)
FLTX_TEST_UNARY(cbrt, gen_unary_any)
FLTX_TEST_BINARY(hypot, gen_binary_any)
FLTX_TEST_UNARY(sin, gen_unary_any)
FLTX_TEST_UNARY(cos, gen_unary_any)
FLTX_TEST_UNARY(tan, gen_unary_any)
FLTX_TEST_UNARY(atan, gen_unary_any)
FLTX_TEST_BINARY(atan2, gen_binary_any)
FLTX_TEST_UNARY(asin, gen_unary_unit)
FLTX_TEST_UNARY(acos, gen_unary_unit)
FLTX_TEST_UNARY(sinh, gen_unary_any)
FLTX_TEST_UNARY(cosh, gen_unary_any)
FLTX_TEST_UNARY(tanh, gen_unary_any)
FLTX_TEST_UNARY(asinh, gen_unary_any)
FLTX_TEST_UNARY(acosh, gen_unary_ge_one)
FLTX_TEST_UNARY(atanh, gen_unary_unit)

TEST_CASE("f64 constexpr parity: pow", "[fltx][constexpr][parity][f64][pow]")
{
    run_tuple_test("pow", gen_pow_args, [](double x, double y) { return bl::pow(x, y); });
}

FLTX_TEST_UNARY(erf, gen_unary_any)
FLTX_TEST_UNARY(erfc, gen_unary_any)
FLTX_TEST_UNARY(lgamma, gen_unary_gamma)
FLTX_TEST_UNARY(tgamma, gen_unary_gamma)
FLTX_TEST_UNARY(fpclassify, gen_unary_classification)
FLTX_TEST_UNARY(isnormal, gen_unary_classification)
FLTX_TEST_BINARY(isunordered, gen_binary_classification)
FLTX_TEST_BINARY(isgreater, gen_binary_any)
FLTX_TEST_BINARY(isgreaterequal, gen_binary_any)
FLTX_TEST_BINARY(isless, gen_binary_any)
FLTX_TEST_BINARY(islessequal, gen_binary_any)
FLTX_TEST_BINARY(islessgreater, gen_binary_any)

#undef FLTX_TEST_UNARY
#undef FLTX_TEST_BINARY
#undef FLTX_TEST_BINARY_LONG
#undef FLTX_TEST_TERNARY
#undef FLTX_TEST_UNARY_INT
#undef FLTX_TEST_UNARY_LONG

} // namespace
