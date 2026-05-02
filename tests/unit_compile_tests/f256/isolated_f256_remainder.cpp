#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_remainder()
{
    bl::f256 x = bl::isolated::runtime_f256(1.23456789);
    bl::f256 y = bl::isolated::runtime_f256(0.75);
    bl::f256 value = bl::remainder(x, y);

    bl::isolated::keep_value(value);
}
