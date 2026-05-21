#include <fltx/f256/math/trig.h>
#include "isolated_runtime.h"

namespace
{
    struct f256_sincos_pair
    {
        bl::f256 c;
        bl::f256 s;
    };

} // namespace

void isolated_f256_sincos()
{
    bl::f256 input = bl::isolated::runtime_f256(0.123456789);
    bl::f256 sine;
    bl::f256 cosine;
    bool value = bl::sincos(input, sine, cosine);
    f256_sincos_pair out{};
    bool vector_value = bl::sincos(input, out);
    f256_sincos_pair returned = bl::sincos<bl::f256>(input);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(vector_value);
    bl::isolated::keep_value(sine);
    bl::isolated::keep_value(cosine);
    bl::isolated::keep_value(out.c);
    bl::isolated::keep_value(out.s);
    bl::isolated::keep_value(returned.c);
    bl::isolated::keep_value(returned.s);
}
