#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <fltx_math.h>
#include <fltx_io.h>

using namespace bl;
using namespace bl::literals;

TEST_CASE("constexpr construction and conversion", "[fltx][consistency]")
{
    constexpr f32  a =    3.14159265358979323846264338327954634575678678942457586525785215783452e+10f;
    constexpr f64  b =    3.14159265358979323846264338327954634575678678942457586525785215783452e+10;
    constexpr f128 c = DD(3.14159265358979323846264338327954634575678678942457586525785215783452e+10);
    constexpr f256 d = QD(3.14159265358979323846264338327954634575678678942457586525785215783452e+10);

    constexpr f64  A = bl::pow(123.456f,   2.0f);
    constexpr f64  B = bl::pow(123.456,    2.0);
    constexpr f128 C = bl::pow(123.456_dd, 2.0_dd);
    constexpr f256 D = bl::pow("123.456"_qd, "2.0"_qd);

    std::cout << std::setprecision(std::numeric_limits<f256>::digits10);

    std::cout << bl::to_string(a) << "\n";
    std::cout << bl::to_string(b) << "\n";
    std::cout << bl::to_string(c) << "\n";
    std::cout << bl::to_string(d) << "\n\n";

    std::cout << bl::to_string(A) << "\n";
    std::cout << bl::to_string(B) << "\n";
    std::cout << bl::to_string(C) << "\n";
    std::cout << bl::to_string(D) << "\n";
}