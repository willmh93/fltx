#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_nearbyint()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    bl::f256 value = bl::nearbyint(input);

    bl::isolated::keep_value(value);
}
