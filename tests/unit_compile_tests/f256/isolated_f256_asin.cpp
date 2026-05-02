#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_asin()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bl::f256 value = bl::asin(input);

    bl::isolated::keep_value(value);
}
