#include <iostream>
#include <iomanip>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

int main()
{
    constexpr f256 a = 1_qd / 3_qd;
    constexpr f256 b = 2_qd / 3_qd;

    constexpr f256 c = a + b;
    constexpr f256 d = bl::atan2(a, b);

    std::cout 
        << std::fixed 
        << std::setprecision(std::numeric_limits<f256>::digits10)
        << "a           = " << a << "\n"
        << "b           = " << b << "\n"
        << "a + b       = " << c << "\n"
        << "atan2(a, b) = " << d << "\n";
}

// output:  a           = 0.333333333333333333333333333333333333333333333333333333333333333
//          b           = 0.666666666666666666666666666666666666666666666666666666666666667
//          a + b       = 1.000000000000000000000000000000000000000000000000000000000000000
//          atan2(a, b) = 0.463647609000806116214256231461214402028537054286120263810933090
