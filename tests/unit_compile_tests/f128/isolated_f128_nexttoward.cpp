#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_nexttoward()
{
    bl::f128 from = bl::isolated::runtime_f128(0.123456789);
    long double to = bl::isolated::runtime_long_double(1.23456789L);
    bl::f128 value = bl::nexttoward(from, to);

    bl::isolated::keep_value(value);
}
