#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_nexttoward()
{
    bl::f256 from = bl::isolated::runtime_f256(0.123456789);
    long double to = bl::isolated::runtime_long_double(1.23456789L);
    bl::f256 value = bl::nexttoward(from, to);

    bl::isolated::keep_value(value);
}
