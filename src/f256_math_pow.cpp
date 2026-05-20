#include "f256_math.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y) { return detail::_f256_constexpr::pow(x, y); }
    BL_NO_INLINE f256_s pow(const f256_s& x, double y)        { return detail::_f256_constexpr::pow(x, y); }
}
