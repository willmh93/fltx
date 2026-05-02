#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_hypot()
{
    bl::f128 x = bl::isolated::runtime_f128(1.23456789);
    bl::f128 y = bl::isolated::runtime_f128(0.123456789);
    bl::f128 value = bl::hypot(x, y);

    bl::isolated::keep_value(value);
}
