#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_isfinite()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bool value = bl::isfinite(input);

    bl::isolated::keep_value(value);
}
