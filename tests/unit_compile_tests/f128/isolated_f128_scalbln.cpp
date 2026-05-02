#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_scalbln()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    long exponent = bl::isolated::runtime_long(7);
    bl::f128 value = bl::scalbln(input, exponent);

    bl::isolated::keep_value(value);
}
