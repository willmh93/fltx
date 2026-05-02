#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_llround()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    long long value = bl::llround(input);

    bl::isolated::keep_value(value);
}
