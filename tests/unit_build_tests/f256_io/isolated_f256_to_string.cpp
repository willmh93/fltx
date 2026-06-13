#include <fltx/string.h>
#include "isolated_runtime.h"

#include <ios>
#include <limits>
#include <string>

void isolated_f256_to_string()
{
    bl::f256 input = bl::isolated::runtime_f256(1.2345678901234567, 1.0e-30, -1.0e-46, 1.0e-62);
    std::string text = bl::to_string(
        input,
        std::numeric_limits<bl::f256>::digits10,
        std::ios_base::scientific);

    bl::isolated::keep_value(text);
}
