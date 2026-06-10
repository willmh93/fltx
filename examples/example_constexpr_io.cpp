#include <iostream>
#include <iomanip>
#include <limits>

#include <fltx/io.h>
using namespace bl;

int main()
{
    std::cout << std::setprecision(std::numeric_limits<f256>::digits10);

    // calculate pi/2
    constexpr f256 value = std::numbers::pi_v<f256> / 2;
    std::cout << value << "\n";

    // to compile-time string
    constexpr auto txt = bl::to_static_string(value);

    // parse back to f256 value
    constexpr f256 parsed = bl::parse<f256>(txt);
    std::cout << parsed;

    //
    std::cout << "\n\n";

    using namespace bl::literals;

    // literals:
    //   _dd for f128
    //   _qd for f256
    constexpr f256 pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164_qd;

    // serializing
    constexpr auto s1 = bl::to_static_string(pi);    // generate static_string<512>
    std::string    s2 = bl::to_string(pi);           // generate string using default precision
    std::string    s3 = bl::to_string(pi, 16, true); // generate string using fixed precision

    // deserializing
    constexpr f256 value1 = bl::parse<f256>("3.1415_err");     // invalid input, throws exception
    constexpr f256 value2 = bl::parse<f256>("3.1415_err", pi); // invalid input, use fallback value
    constexpr auto result = bl::try_parse<f256>("3.1415_err"); // invalid input, error in result.ec
    if (result)
        std::cout << "success: " << result.value;
    else
        std::cout << "invalid input: ";
}

