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
    std::array<Vec2<T>, C> points{};
    T tau = std::numbers::pi_v<T> * 2;

    for (std::size_t i = 0; i < C; ++i)
    {
        T ratio = T{ static_cast<std::uint64_t>(i) } / T{ static_cast<std::uint64_t>(C) };
        T angle = ratio * tau;
        Vec2<T> sc = bl::sincos<T>(angle);

        points[i] = {
            sc.x * rx,
            sc.y * ry
        };
    }

    return points;
}

int main()
{
    using T = f256;

    std::cout << std::setprecision(std::numeric_limits<T>::digits10);

    // generate 100 high-precision points
    constexpr auto points = ellipse<T, 100>(100, 50);

    for (auto [x, y] : points)
    {
        std::cout << x << ", " << y << "\n";
    }
}
