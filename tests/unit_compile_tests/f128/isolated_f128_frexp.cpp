#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_frexp()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    int exponent = 0;
    bl::f128 value = bl::frexp(input, &exponent);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(exponent);
}
