#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_isunordered()
{
    bl::f256 a = bl::isolated::runtime_f256(0.123456789);
    bl::f256 b = bl::isolated::runtime_f256(1.23456789);
    bool value = bl::isunordered(a, b);

    bl::isolated::keep_value(value);
}
