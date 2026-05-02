#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_floor()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    bl::f128 value = bl::floor(input);

    bl::isolated::keep_value(value);
}
