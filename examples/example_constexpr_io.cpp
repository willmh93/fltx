#include <iostream>
#include <iomanip>

#include <fltx_io.h>
using namespace bl;
using namespace bl::literals;

int main()
{
    // calculate pi * 0.5:
    // = 1.57079632679489661923132169163975144209858469968755291048747230
    constexpr f256 pio2_f256 = std::numbers::pi_v<f256> * 0.5_qd;

    // generate static_string<512> at compile-time
    // = "1.57079632679489661923132169163975144209858469968755291048747230"
    constexpr auto str_f256 = bl::to_string(pio2_f256);
    std::cout << "str_f256: " << str_f256 << "\n";

    // parse back to f128 value
    constexpr f128 parsed_f128 = bl::to_f128(str_f256);
    std::cout << std::setprecision(bl::numeric_limits<f128>::digits10);
    std::cout << "parsed_f128: " << parsed_f128;
}

