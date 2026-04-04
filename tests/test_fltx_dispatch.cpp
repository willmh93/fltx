#include <catch2/catch_test_macros.hpp>
#include <fltx/fltx.h>

template<typename T1, typename T2, typename T3>
void testFunc(bool B)
{
    T1 x = 5;
    T2 y = 10;
    T3 z = 20;

    auto w = x * z; // works, w = f256

    //T3 w = x * z; // fails. testFunc is instantiated with all combinations of T1,T2,T3 = f32,f128,f256 because of runtime selection,
                    // so the compiler will attempt downcasting to f32/f64/f128, which isn't supported due to potential for data loss

    T3 q = (T3)(x * z); // fine

    int a = 5;
}

TEST_CASE("constexpr construction and conversion", "[fltx][constexpr_dispatch]")
{
    using namespace bl;

    {
        f256_t a = 3.0;
        f32 b = 2.0;
        f256_t c = a * b;
    }
    {
        f32 a = 3.0;
        f128_t b = 2.0;
        f256_t c = b * a;
    }

    bl::table_invoke(
        bl::dispatch_table(testFunc, true), 
        bl::FloatType::F32, bl::FloatType::F128, bl::FloatType::F256
    );
}