#include <iostream>
#include <iomanip>

#include <fltx/fltx.h>
using namespace bl::literals;

int main()
{
    constexpr bl::f256_t a = (1_qd / 3_qd);
    constexpr bl::f256_t b = (2_qd / 3_qd);
    bl::f256_t c = a + b;

    std::cout << std::setprecision(std::numeric_limits<bl::f256>::digits10)
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "a + b = " << c << "\n";
    
    // output:
    //   a = 0.333333333333333333333333333333333333333333333333333333333333333
    //   b = 0.666666666666666666666666666666666666666666666666666666666666667
    //   a + b = 1

}