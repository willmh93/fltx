#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_fma()
{
    bl::f256 x = bl::isolated::runtime_f256(1.23456789);
    bl::f256 y = bl::isolated::runtime_f256(0.123456789);
    bl::f256 z = bl::isolated::runtime_f256(-0.333333333);
    bl::f256 value = bl::fma(x, y, z);

    bl::isolated::keep_value(value);
}
