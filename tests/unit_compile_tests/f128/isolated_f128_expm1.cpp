#include <fltx/f128/math/exp_log.h>
#include "isolated_runtime.h"

void isolated_f128_expm1()
{
    bl::f128 input = bl::isolated::runtime_f128(0.123456789);
    bl::f128 value = bl::expm1(input);

    bl::isolated::keep_value(value);
}
