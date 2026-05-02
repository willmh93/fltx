#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_remquo()
{
    bl::f128 x = bl::isolated::runtime_f128(1.23456789);
    bl::f128 y = bl::isolated::runtime_f128(0.75);
    int quotient = 0;
    bl::f128 value = bl::remquo(x, y, &quotient);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(quotient);
}
