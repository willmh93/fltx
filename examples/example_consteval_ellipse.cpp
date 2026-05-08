#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

template<typename T>
struct Vec2
{
    T x;
    T y;
};

template<typename T, std::size_t C>
consteval std::array<Vec2<T>, C> ellipse(T rx, T ry)
{
    static_assert(C > 0);

    std::array<Vec2<T>, C> points{};

    constexpr T tau = std::numbers::pi_v<T> *T{ 2 };

    for (std::size_t i = 0; i < C; ++i)
    {
        T ratio = T(static_cast<std::uint64_t>(i)) / T(static_cast<std::uint64_t>(C));
        T angle = ratio * tau;

        points[i] = {
            bl::cos(angle) * rx,
            bl::sin(angle) * ry
        };
    }

    return points;
}

int main()
{
    using T = f256;

    std::cout << std::setprecision(std::numeric_limits<T>::digits10);

    constexpr auto points = ellipse<T, 100>(T{ 100 }, T{ 50 });

    for (auto [x, y] : points)
    {
        std::cout << x << ", " << y << "\n";
    }
}
