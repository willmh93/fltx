#include <f128_math.h>
#include "isolated_runtime.h"

void isolated_f128_sincos()
{
    bl::f128 input = bl::isolated::runtime_f128(0.123456789);
    bl::f128 sine;
    bl::f128 cosine;
    bool value = bl::sincos(input, sine, cosine);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(sine);
    bl::isolated::keep_value(cosine);
}
