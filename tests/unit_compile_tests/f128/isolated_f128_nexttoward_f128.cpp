#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_nexttoward_f128()
{
    bl::f128 from = bl::isolated::runtime_f128(0.123456789);
    bl::f128 to = bl::isolated::runtime_f128(1.23456789);
    bl::f128 value = bl::nexttoward(from, to);

    bl::isolated::keep_value(value);
}
