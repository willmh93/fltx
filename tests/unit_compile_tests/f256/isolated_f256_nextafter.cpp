#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_nextafter()
{
    bl::f256 from = bl::isolated::runtime_f256(0.123456789);
    bl::f256 to = bl::isolated::runtime_f256(1.23456789);
    bl::f256 value = bl::nextafter(from, to);

    bl::isolated::keep_value(value);
}
