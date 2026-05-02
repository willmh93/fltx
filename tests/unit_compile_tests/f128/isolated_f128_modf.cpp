#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_modf()
{
    bl::f128 input = bl::isolated::runtime_f128(1.23456789);
    bl::f128 integer_part;
    bl::f128 value = bl::modf(input, &integer_part);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(integer_part);
}
