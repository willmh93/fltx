#define FLTX_CONSTEXPR_PARITY_TEST_MODE

#include <fltx_dispatch.h>
#include <fltx_io.h>
#include <fltx_math.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

using namespace bl;
using namespace bl::literals;

namespace
{
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

template<typename T>
inline constexpr bool is_f128_type =
    std::is_same_v<std::remove_cv_t<T>, bl::f128> ||
    std::is_same_v<std::remove_cv_t<T>, bl::f128_s>;

template<typename T>
inline constexpr bool is_f256_type =
    std::is_same_v<std::remove_cv_t<T>, bl::f256> ||
    std::is_same_v<std::remove_cv_t<T>, bl::f256_s>;

template<typename... Values>
void consume(const Values&... values) noexcept
{
    (static_cast<void>(values), ...);
}

template<typename T>
constexpr T value(double x)
{
    return static_cast<T>(x);
}

template<typename A, typename B>
void expect_equal(const A& actual, const B& expected)
{
    if (!(actual == expected))
        throw std::runtime_error("coverage check failed");
}

constexpr bool exercise_integral_math_utils_constexpr()
{
    return
        bl::sq(3) == 9 &&
        bl::sq(-3) == 9 &&
        bl::clamp(5, 0, 3) == 3 &&
        bl::clamp(-1, 0, 3) == 0 &&
        bl::clamp(2, 0, 3) == 2 &&
        bl::pow(2, 10) == 1024 &&
        bl::pow(-2, 3) == -8 &&
        bl::pow(-2, 4u) == 16 &&
        bl::pow(2u, 5u) == 32u &&
        bl::pow(2, -3) == 0 &&
        bl::pow(1, -99) == 1 &&
        bl::pow(-1, -99) == -1 &&
        bl::pow(-1, -100) == 1;
}

template<typename T>
constexpr bool exercise_typed_math_utils_constexpr()
{
    return
        bl::sq(value<T>(2.0)) == value<T>(4.0) &&
        bl::sq(value<T>(-2.0)) == value<T>(4.0) &&
        bl::sq(value<T>(0.0)) == value<T>(0.0) &&
        bl::clamp(value<T>(0.5), value<T>(-1.0), value<T>(1.0)) == value<T>(0.5) &&
        bl::clamp(value<T>(-2.0), value<T>(-1.0), value<T>(1.0)) == value<T>(-1.0) &&
        bl::clamp(value<T>(2.0), value<T>(-1.0), value<T>(1.0)) == value<T>(1.0) &&
        bl::pow(value<T>(2.0), 0) == value<T>(1.0) &&
        bl::pow(value<T>(2.0), 1) == value<T>(2.0) &&
        bl::pow(value<T>(2.0), 3) == value<T>(8.0) &&
        bl::pow(value<T>(-2.0), 3) == value<T>(-8.0) &&
        bl::pow(value<T>(-2.0), 4u) == value<T>(16.0) &&
        bl::pow(value<T>(2.0), -2) == value<T>(0.25) &&
        bl::pow(value<T>(-1.0), -3) == value<T>(-1.0) &&
        bl::pow(value<T>(-1.0), -4) == value<T>(1.0);
}

template<typename T>
void exercise_classification(T x, T y)
{
    consume(
        bl::signbit(x),
        bl::isnan(x),
        bl::isinf(x),
        bl::isfinite(x),
        bl::iszero(x),
        bl::fpclassify(x),
        bl::isnormal(x),
        bl::isunordered(x, y),
        bl::isgreater(x, y),
        bl::isgreaterequal(x, y),
        bl::isless(x, y),
        bl::islessequal(x, y),
        bl::islessgreater(x, y)
    );

    if constexpr (bl::is_fltx_v<T>)
        consume(bl::ispositive(x));
}

template<typename T>
void exercise_rounding(T x, T nonzero)
{
    consume(
        bl::abs(x),
        bl::fabs(x),
        bl::floor(x),
        bl::ceil(x),
        bl::trunc(x),
        bl::round(x),
        bl::nearbyint(x),
        bl::rint(x),
        bl::lround(x),
        bl::llround(x),
        bl::lrint(x),
        bl::llrint(x),
        bl::fmod(x, nonzero),
        bl::remainder(x, nonzero)
    );

    int quotient = 0;
    consume(bl::remquo(x, nonzero, &quotient), quotient);

    if constexpr (bl::is_fltx_v<T>)
        consume(bl::round_to_decimals(x, 3));
}

template<typename T>
void exercise_arithmetic(T x, T y, T z)
{
    T mutated = x;
    mutated += y;
    mutated -= z;
    mutated *= y;
    mutated /= y;

    consume(
        x + y,
        x - y,
        x * y,
        x / y,
        mutated,
        -x,
        +x,
        x == y,
        x != y,
        x < y,
        x <= y,
        x > y,
        x >= y,
        bl::fma(x, y, z),
        bl::fmin(x, y),
        bl::fmax(x, y),
        bl::fdim(x, y),
        bl::copysign(x, y)
    );

    if constexpr (bl::is_fltx_v<T>)
    {
        consume(
            bl::recip(y),
            bl::inv(y),
            bl::clamp(x, value<T>(-2.0), value<T>(2.0))
        );
    }
}

template<typename T>
void exercise_math_utils()
{
    static_assert(exercise_integral_math_utils_constexpr());
    static_assert(exercise_typed_math_utils_constexpr<T>());

    const T negative_two = value<T>(-2.0);
    const T negative_one = value<T>(-1.0);
    const T zero = value<T>(0.0);
    const T half = value<T>(0.5);
    const T one = value<T>(1.0);
    const T two = value<T>(2.0);
    const T four = value<T>(4.0);
    const T eight = value<T>(8.0);
    const T sixteen = value<T>(16.0);
    const T quarter = value<T>(0.25);

    expect_equal(bl::sq(two), four);
    expect_equal(bl::sq(negative_two), four);
    expect_equal(bl::sq(zero), zero);

    expect_equal(bl::clamp(half, negative_one, one), half);
    expect_equal(bl::clamp(value<T>(-2.0), negative_one, one), negative_one);
    expect_equal(bl::clamp(two, negative_one, one), one);
    expect_equal(bl::clamp(negative_one, negative_one, one), negative_one);
    expect_equal(bl::clamp(one, negative_one, one), one);

    expect_equal(bl::pow(two, 0), one);
    expect_equal(bl::pow(two, 1), two);
    expect_equal(bl::pow(two, 3), eight);
    expect_equal(bl::pow(negative_two, 3), value<T>(-8.0));
    expect_equal(bl::pow(negative_two, 4u), sixteen);
    expect_equal(bl::pow(two, -2), quarter);
    expect_equal(bl::pow(negative_one, -3), negative_one);
    expect_equal(bl::pow(negative_one, -4), one);

    consume(
        bl::pow(two, std::int8_t{ 3 }),
        bl::pow(two, std::uint8_t{ 3 }),
        bl::pow(two, std::int16_t{ -2 }),
        bl::pow(two, std::uint16_t{ 4 }),
        bl::pow(two, std::int32_t{ 3 }),
        bl::pow(two, std::uint32_t{ 4 }),
        bl::pow(two, std::int64_t{ -2 }),
        bl::pow(two, std::uint64_t{ 4 })
    );

    expect_equal(bl::sq(3), 9);
    expect_equal(bl::sq(-3), 9);
    expect_equal(bl::clamp(5, 0, 3), 3);
    expect_equal(bl::clamp(-1, 0, 3), 0);
    expect_equal(bl::clamp(2, 0, 3), 2);
    expect_equal(bl::clamp(0, 0, 3), 0);
    expect_equal(bl::clamp(3, 0, 3), 3);

    expect_equal(bl::pow(2, 10), 1024);
    expect_equal(bl::pow(-2, 3), -8);
    expect_equal(bl::pow(-2, 4u), 16);
    expect_equal(bl::pow(2u, 5u), 32u);
    expect_equal(bl::pow(std::int64_t{ 3 }, std::uint32_t{ 4 }), std::int64_t{ 81 });
    expect_equal(bl::pow(2, -3), 0);
    expect_equal(bl::pow(1, -99), 1);
    expect_equal(bl::pow(-1, -99), -1);
    expect_equal(bl::pow(-1, -100), 1);

    consume(
        bl::pow(std::int8_t{ -2 }, std::uint8_t{ 3 }),
        bl::pow(std::int16_t{ -2 }, std::int16_t{ 4 }),
        bl::pow(std::int32_t{ 2 }, std::int32_t{ -3 }),
        bl::pow(std::uint32_t{ 2 }, std::uint32_t{ 5 }),
        bl::pow(std::int64_t{ -1 }, std::int64_t{ -127 })
    );
}

template<typename T>
void exercise_decomposition(T x)
{
    int exponent = 0;
    T integral = value<T>(0.0);

    consume(
        bl::ldexp(x, 4),
        bl::scalbn(x, 4),
        bl::scalbln(x, 4L),
        bl::frexp(x, &exponent),
        bl::modf(x, &integral),
        bl::ilogb(x),
        bl::logb(x),
        bl::nextafter(x, value<T>(2.0)),
        bl::nexttoward(x, static_cast<long double>(2.0)),
        exponent,
        integral
    );
}

template<typename T>
void exercise_exp_log_pow(T x, T positive, T gt_minus_one)
{
    consume(
        bl::exp(x),
        bl::exp2(x),
        bl::expm1(x),
        bl::log(positive),
        bl::log2(positive),
        bl::log10(positive),
        bl::log1p(gt_minus_one),
        bl::sqrt(positive),
        bl::cbrt(x),
        bl::hypot(x, positive),
        bl::pow(positive, value<T>(1.25))
    );

    if constexpr (bl::is_fltx_v<T>)
    {
        consume(
            bl::log_as_double(positive),
            bl::pow(positive, 1.25)
        );
    }

    if constexpr (is_f128_type<T>)
        consume(bl::pow10_128(12));

    if constexpr (is_f256_type<T>)
        consume(bl::pow10_256(12));
}

template<typename T>
void exercise_trig(T x, T unit, T ge_one)
{
    consume(
        bl::sin(x),
        bl::cos(x),
        bl::tan(x),
        bl::atan(x),
        bl::atan2(x, ge_one),
        bl::asin(unit),
        bl::acos(unit),
        bl::sinh(x),
        bl::cosh(x),
        bl::tanh(x),
        bl::asinh(x),
        bl::acosh(ge_one),
        bl::atanh(unit)
    );

    if constexpr (bl::is_fltx_v<T>)
    {
        T sine = value<T>(0.0);
        T cosine = value<T>(0.0);
        consume(bl::sincos(x, sine, cosine), sine, cosine);
    }
}

template<typename T>
void exercise_special(T x, T gamma_safe)
{
    consume(
        bl::erf(x),
        bl::erfc(x),
        bl::lgamma(gamma_safe),
        bl::tgamma(gamma_safe)
    );
}

template<typename T>
void exercise_io(T x)
{
    const auto text = bl::to_string(x, 16, false, false, true);
    const auto fixed = bl::to_string(x, 8, true, false, true);
    const auto scientific = bl::to_string(x, 8, false, true, true);
    const auto collapsed = bl::to_string_collapsed(x, 24);
    std::ostringstream stream;
    stream << x;
    consume(text, fixed, scientific, collapsed, stream.str());

    if constexpr (is_f128_type<T>)
    {
        constexpr bl::f128 parsed = bl::to_f128("1.234567890123456789");
        constexpr auto literal = 1.25_dd;
        constexpr auto static_text = bl::to_static_string(parsed, 24, false, false, true);
        const auto std_text = bl::to_std_string(parsed, 24, false, false, true);
        consume(parsed, literal, static_text, std_text);
    }

    if constexpr (is_f256_type<T>)
    {
        constexpr bl::f256 parsed = bl::to_f256("1.23456789012345678901234567890123456789");
        constexpr auto literal = 1.25_qd;
        constexpr auto static_text = bl::to_static_string(parsed, 48, false, false, true);
        const auto std_text = bl::to_std_string(parsed, 48, false, false, true);
        consume(parsed, literal, static_text, std_text);
    }
}

template<typename T, bool ForceConstexpr>
void exercise_all_functions(const char* type_name)
{
    forced_path_scope path(ForceConstexpr);

    const T x = value<T>(0.625);
    const T y = value<T>(1.25);
    const T z = value<T>(-0.375);
    const T nonzero = value<T>(0.75);
    const T positive = value<T>(1.5);
    const T unit = value<T>(0.25);
    const T ge_one = value<T>(2.0);
    const T gt_minus_one = value<T>(0.25);
    const T gamma_safe = value<T>(3.5);

    exercise_classification(x, y);
    exercise_rounding(z, nonzero);
    exercise_arithmetic(x, y, z);
    exercise_math_utils<T>();
    exercise_decomposition(positive);
    exercise_exp_log_pow(x, positive, gt_minus_one);
    exercise_trig(x, unit, ge_one);
    exercise_special(x, gamma_safe);
    exercise_io(x);

    std::cout << type_name << " " << (ForceConstexpr ? "constexpr" : "runtime") << " path compiled and ran\n";
}
}

int main()
{
    for (int index = 0; index < static_cast<int>(bl::FloatType::COUNT); ++index)
    {
        const auto float_type = static_cast<bl::FloatType>(index);
        const char* type_name = bl::FloatTypeNames[index];

        bl::table_invoke(
            bl::dispatch_table(exercise_all_functions, type_name),
            bl::enum_type(float_type),
            false
        );

        bl::table_invoke(
            bl::dispatch_table(exercise_all_functions, type_name),
            bl::enum_type(float_type),
            true
        );
    }

    bl::dispatch_table_info::print_from_args(
        "exercise_all_functions",
        bl::enum_type(bl::FloatType::F32),
        true
    );
}
