#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_lrint()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    long value = bl::lrint(input);

    bl::isolated::keep_value(value);
}
