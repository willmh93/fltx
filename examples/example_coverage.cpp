#define FLTX_CONSTEXPR_PARITY_TEST_MODE

#include <fltx_dispatch.h>
#include <fltx_io.h>
#include <fltx_math.h>

#include <iostream>
#include <limits>
#include <sstream>
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
T value(double x)
{
    return static_cast<T>(x);
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
