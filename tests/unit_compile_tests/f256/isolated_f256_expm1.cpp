#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_expm1()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bl::f256 value = bl::expm1(input);

    bl::isolated::keep_value(value);
}
