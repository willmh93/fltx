#include <fltx/f256/classification.h>
#include "isolated_runtime.h"

void isolated_f256_isinf()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bool value     = bl::isinf(input);

    bl::isolated::keep_value(value);
}
