#include <iostream>
#include <iomanip>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

template<std::size_t N>
consteval f256 sum(std::array<f256, N> values)
{
    f256 total = 0;
    for (const auto& v : values)
        total += v;
    return total;
}

int main()
{
    constexpr auto constexpr_values = bl::random_array<4>(
        bl::mt19937_64{ 0x1020304050607080ull },
        bl::uniform_real_distribution<bl::f256>{ -0_qd, 10_qd }
    );

    constexpr f256 total = sum(constexpr_values);

    std::cout
        << std::fixed
        << std::setprecision(std::numeric_limits<f256>::digits10)
        << "sum: " << total;
}
