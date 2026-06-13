#include <fltx/charconv.h>
#include "isolated_runtime.h"

#include <string_view>

void isolated_f128_parse()
{
    const char* text = bl::isolated::runtime_i32(0) == 0
        ? "1.234567890123456789012345678901"
        : "2.0";

    bl::f128 value = bl::parse<bl::f128>(std::string_view{ text });

    bl::isolated::keep_value(value);
}
