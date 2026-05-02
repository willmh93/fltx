#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_fma()
{
    bl::f128 x = bl::isolated::runtime_f128(1.23456789);
    bl::f128 y = bl::isolated::runtime_f128(0.123456789);
    bl::f128 z = bl::isolated::runtime_f128(-0.333333333);
    bl::f128 value = bl::fma(x, y, z);

    bl::isolated::keep_value(value);
}
