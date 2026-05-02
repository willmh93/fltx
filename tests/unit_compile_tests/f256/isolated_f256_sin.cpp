// isolated_f256_sin.cpp

#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_sin()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bl::f256 value = bl::sin(input);

    bl::isolated::keep_value(value);
}
