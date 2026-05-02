#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_islessgreater()
{
    bl::f128 a = bl::isolated::runtime_f128(0.123456789);
    bl::f128 b = bl::isolated::runtime_f128(1.23456789);
    bool value = bl::islessgreater(a, b);

    bl::isolated::keep_value(value);
}
