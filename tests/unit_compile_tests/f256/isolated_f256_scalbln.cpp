#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_scalbln()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    long exponent = bl::isolated::runtime_long(7);
    bl::f256 value = bl::scalbln(input, exponent);

    bl::isolated::keep_value(value);
}
