#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <fltx.h>

using namespace bl;

TEST_CASE("constexpr construction and conversion", "[fltx][constexpr]")
{
    constexpr double a =  3.14159265358979323846264338327954634575678678942457586525785215783452e+100;
    constexpr f128_s   b = DD(3.14159265358979323846264338327954634575678678942457586525785215783452e+100);
    constexpr f256_s   c = QD(3.14159265358979323846264338327954634575678678942457586525785215783452e+100);

    std::cout << "PI f64: "  << bl::to_string(a).c_str() << "\n";
    std::cout << "PI f128: " << bl::to_string(b).c_str() << "\n";
    std::cout << "PI f256: " << bl::to_string(c).c_str() << "\n";
}