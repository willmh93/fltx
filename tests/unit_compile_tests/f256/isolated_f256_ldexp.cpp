#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_ldexp()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    int exponent = bl::isolated::runtime_i32(7);
    bl::f256 value = bl::ldexp(input, exponent);

    bl::isolated::keep_value(value);
}
