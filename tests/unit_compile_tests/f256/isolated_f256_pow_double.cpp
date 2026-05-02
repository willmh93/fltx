#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_pow_double()
{
    bl::f256 x = bl::isolated::runtime_f256(1.23456789);
    double y = bl::isolated::runtime_f64(2.5);
    bl::f256 value = bl::pow(x, y);

    bl::isolated::keep_value(value);
}
