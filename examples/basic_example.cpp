#include <iostream>
#include <iomanip>

#include <fltx.h>
using namespace bl;
using namespace bl::literals;

void print_precision_example()
{
    constexpr f256 a = 1_qd / 3_qd;
    constexpr f256 b = 2_qd / 3_qd;
    constexpr f256 c = a + b;

    std::cout << std::setprecision(std::numeric_limits<f256_s>::digits10)
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "a + b = " << c << "\n\n";

    // output: a = 0.333333333333333333333333333333333333333333333333333333333333333
    //         b = 0.666666666666666666666666666666666666666666666666666666666666667
    //         a + b = 1
}

template<typename T>
void print_math(const char* name)
{
    constexpr T pi_v   = std::numbers::pi_v<T>;
    constexpr T min_v  = std::numeric_limits<T>::min();
    constexpr T max_v  = std::numeric_limits<T>::max();
    constexpr T sin_v  = std::numeric_limits<T>::max();

    std::cout
        << std::setprecision(std::numeric_limits<T>::digits10)
        << name << " min = " << min_v << "\n"
        << name << " max = " << max_v << "\n"
        << name << " PI = " << pi_v << "\n"
        << "sin(pi_v / 6.0) = " << bl::sin(pi_v / 6.0) << "\n"
        << "sin(pi_v / 5.0) = " << bl::sin(pi_v / 5.0) << "\n\n";
}

void print_math_example()
{
    // direct invocation
    //print_math<f32>("F32");

    // dynamic dispatch table invocation
    for (int i = 0; i < (int)FloatType::COUNT; i++)
    {
        FloatType float_type = (FloatType)i;
        bl::table_invoke(bl::dispatch_table(print_math, FloatTypeNames[i]), float_type);
    }

    //bl::table_invoke(bl::dispatch_table(print_math, "F32"),  FloatType::F32);
    //bl::table_invoke(bl::dispatch_table(print_math, "F64"),  FloatType::F64);
    //bl::table_invoke(bl::dispatch_table(print_math, "F128"), FloatType::F128);
    //bl::table_invoke(bl::dispatch_table(print_math, "F256"), FloatType::F256);

    //constexpr f128 pi_128   = std::numbers::pi_v<f128>;
    //constexpr f256 pi_256   = std::numbers::pi_v<f256>;
    //constexpr f128 f128_max = std::numeric_limits<f128>::max();
    //constexpr f256 f256_max = std::numeric_limits<f256>::max();
    //
    //std::cout
    //    << std::setprecision(std::numeric_limits<f128>::digits10)
    //    << "f128_max = " << f128_max << "\n"
    //    << "pi_128 = " << pi_128 << "\n"
    //    << "sin(pi_128 / 6.0) = " << bl::sin(pi_128 / 6.0) << "\n"
    //    << "sin(pi_128 / 5.0) = " << bl::sin(pi_128 / 5.0) << "\n\n"
    //
    //    << std::setprecision(std::numeric_limits<f256>::digits10)
    //    << "f256_max = " << f256_max << "\n"
    //    << "pi_256 = " << pi_256 << "\n"
    //    << "sin(pi_256 / 6.0) = " << bl::sin(pi_256 / 6.0) << "\n"
    //    << "sin(pi_256 / 5.0) = " << bl::sin(pi_256 / 5.0) << "\n";
    //
}

int main()
{
    print_precision_example();
    print_math_example();
}

