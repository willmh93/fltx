#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_pow()
{
    bl::f256 x = bl::isolated::runtime_f256(1.23456789);
    bl::f256 y = bl::isolated::runtime_f256(2.5);
    bl::f256 value = bl::pow(x, y);

    bl::isolated::keep_value(value);
}
