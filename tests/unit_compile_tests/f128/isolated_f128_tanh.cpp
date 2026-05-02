#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_tanh()
{
    bl::f128 input = bl::isolated::runtime_f128(0.123456789);
    bl::f128 value = bl::tanh(input);

    bl::isolated::keep_value(value);
}
