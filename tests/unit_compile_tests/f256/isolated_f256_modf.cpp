#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_modf()
{
    bl::f256 input = bl::isolated::runtime_f256(1.23456789);
    bl::f256 integer_part;
    bl::f256 value = bl::modf(input, &integer_part);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(integer_part);
}
