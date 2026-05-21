#include "fltx/detail/f256/math/erf.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s erf(const f256_s& x) { return detail::_f256_constexpr::erf(x); }
    BL_NO_INLINE f256_s erfc(const f256_s& x) { return detail::_f256_constexpr::erfc(x); }

} // namespace bl::detail::_f256_runtime
