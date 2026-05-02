#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_frexp()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    int exponent = 0;
    bl::f256 value = bl::frexp(input, &exponent);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(exponent);
}
