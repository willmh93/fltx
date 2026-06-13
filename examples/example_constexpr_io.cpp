#include <iostream>
#include <iomanip>
#include <limits>

#include <fltx/io.h>
#include <fltx/f256_math.h>
#include <fltx/numbers.h>

using namespace bl;

constexpr f256 round_to_stream_precision(f256 value, int precision, std::ios_base::fmtflags flags)
{
    const auto floatfield = flags & std::ios_base::floatfield;

    if (floatfield == std::ios_base::fixed)
        return bl::round_to(value, precision, bl::decimals);

    if (floatfield == std::ios_base::scientific)
        return bl::round_to(value, precision + 1, bl::significant_figures);

    if (floatfield == (std::ios_base::fixed | std::ios_base::scientific))
        return value; // hexfloat

    return bl::round_to(value, precision, bl::significant_figures);
}

int main()
{
    constexpr int digits = std::numeric_limits<f256>::digits10;

    constexpr auto mode = std::ios_base::scientific;
    // constexpr auto mode = std::ios_base::fixed;
    // constexpr auto mode = std::ios_base::fmtflags{}; // defaultfloat
    // constexpr auto mode = std::ios_base::fixed | std::ios_base::scientific; // hexfloat

    constexpr auto flags =
        mode |
        std::ios_base::showpoint |
        std::ios_base::showpos |
        std::ios_base::uppercase;

    std::cout.setf(flags & std::ios_base::floatfield, std::ios_base::floatfield);
    std::cout.setf(
        flags & (std::ios_base::showpoint | std::ios_base::showpos | std::ios_base::uppercase),
        std::ios_base::showpoint | std::ios_base::showpos | std::ios_base::uppercase);
    std::cout << std::setprecision(digits);

    // calculate pi/2
    constexpr f256 value = 10 + std::numbers::pi_v<f256> / 2;
    std::cout << "value:  " << value << "\n";

    // to compile-time string
    constexpr auto txt = bl::to_static_string(value, digits, flags);
    std::cout << "txt:    " << txt << "\n";

    // parse back to f256 value
    constexpr f256 parsed = bl::parse<f256>(txt);
    std::cout << "parsed: " << parsed << "\n\n";

    constexpr f256 a = round_to_stream_precision(value, digits, flags);
    constexpr f256 b = round_to_stream_precision(parsed, digits, flags);

    std::cout << std::setprecision(std::numeric_limits<f256>::max_digits10);
    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n\n";

    return (a == b) ? 0 : 1;
}

