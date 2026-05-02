#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_log2()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    bl::f256 value = bl::log2(input);

    bl::isolated::keep_value(value);
}
