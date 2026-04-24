#include <iostream>
#include <iomanip>

#include <fltx_io.h>
using namespace bl;
using namespace bl::literals;

int main()
{
    constexpr f256 a = 1_qd / 3_qd;
    constexpr f256 b = 2_qd / 3_qd;
    constexpr f256 c = a + b;

    std::cout
        << std::setprecision(std::numeric_limits<f256>::digits10)
        << "a     = " << a << "\n"
        << "b     = " << b << "\n"
        << "a + b = " << c << "\n\n";
}

// output: a = 0.333333333333333333333333333333333333333333333333333333333333333
//         b = 0.666666666666666666666666666666666666666666666666666666666666667
//         a + b = 1
