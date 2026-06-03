#include <iostream>
#include <iomanip>
#include <fltx.h>
using namespace bl;
using namespace bl::literals;

template<class Engine, class Dist>
struct cx_random_generator
{
    Engine engine;
    Dist dist;

    using result_type = typename Dist::result_type;

    constexpr result_type operator()()
    {
        return dist(engine);
    }
};

consteval auto test()
{
    cx_random_generator gen{
        bl::mt19937_64{ 0x1020304050607080ull },
        bl::uniform_real_distribution<bl::f256>{ -2_qd, 3_qd }
    };

    bl::f256 a = gen();
    bl::f256 b = gen();

    return std::array{ a, b };
}

int main()
{
    //constexpr f256 a = 1_qd / 3_qd;
    //constexpr f256 b = 2_qd / 3_qd;

    //constexpr auto values = test();
    //constexpr bl::f256 a = values[0];
    //constexpr bl::f256 b = values[1];
    //constexpr f256 c = a + b;
    //constexpr f256 d = bl::atan2(a, b);

    bl::mt19937_64 rng{ 0x1020304050607080ull };
    bl::uniform_real_distribution<bl::f256> dist{ -2.0, 3.0 };
    bl::f256 a = dist(rng);
    bl::f256 b = dist(rng);
    f256 c = a + b;
    f256 d = bl::atan2(a, b);


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
