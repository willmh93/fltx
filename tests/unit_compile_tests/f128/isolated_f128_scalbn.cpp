#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_scalbn()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    int exponent = bl::isolated::runtime_i32(7);
    bl::f128 value = bl::scalbn(input, exponent);

    bl::isolated::keep_value(value);
}
