#include <iostream>
#include <iomanip>
#include <array>

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
    constexpr T tau = std::numbers::pi_v<T> * T{ 2.0 };

    for (int i = 0; i < C; i++)
    {
        T ratio = T(i) / T(C);
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

    std::cout << std::fixed << std::setprecision(std::numeric_limits<T>::digits10);

    constexpr auto points = ellipse<T, 12>(1.0, 1.0);

    for (auto [x, y] : points)
    {
        std::cout << x << ", " << y << "\n";
    }
}

