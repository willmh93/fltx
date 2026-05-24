#include <fltx/f128_math.h>
#include "isolated_runtime.h"

namespace
{
    struct f128_sincos_pair
    {
        bl::f128 c;
        bl::f128 s;
    };

} // namespace

void isolated_f128_sincos()
{
    bl::f128 input = bl::isolated::runtime_f128(0.123456789);
    bl::f128 sine;
    bl::f128 cosine;
    bool value = bl::sincos(input, sine, cosine);
    f128_sincos_pair out{};
    bool vector_value = bl::sincos(input, out);
    f128_sincos_pair returned = bl::sincos<bl::f128>(input);

    bl::isolated::keep_value(value);
    bl::isolated::keep_value(vector_value);
    bl::isolated::keep_value(sine);
    bl::isolated::keep_value(cosine);
    bl::isolated::keep_value(out.c);
    bl::isolated::keep_value(out.s);
    bl::isolated::keep_value(returned.c);
    bl::isolated::keep_value(returned.s);
}
