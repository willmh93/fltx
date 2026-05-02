#include <f256_math.h>
#include "isolated_runtime.h"

void isolated_f256_sincos()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bl::f256 sine;
    bl::f256 cosine;
    bool value = bl::sincos(input, sine, cosine);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(sine);
    bl::isolated::keep_value(cosine);
}
