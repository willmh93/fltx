#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_ilogb()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    int value = bl::ilogb(input);

    bl::isolated::keep_value(value);
}
