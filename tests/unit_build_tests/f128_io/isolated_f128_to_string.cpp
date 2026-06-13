#include <fltx/string.h>
#include "isolated_runtime.h"

#include <ios>
#include <limits>
#include <string>

void isolated_f128_to_string()
{
    bl::f128 input = bl::isolated::runtime_f128(1.2345678901234567, 1.0e-30);
    std::string text = bl::to_string(
        input,
        std::numeric_limits<bl::f128>::digits10,
        std::ios_base::scientific);

    bl::isolated::keep_value(text);
}
