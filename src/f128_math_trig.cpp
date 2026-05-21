#include "fltx/detail/f128/math/trig.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out) { return detail::_f128_constexpr::sincos(x, s_out, c_out); }
    BL_NO_INLINE f128_s sin(const f128_s& x) { return detail::_f128_constexpr::sin(x); }
    BL_NO_INLINE f128_s cos(const f128_s& x) { return detail::_f128_constexpr::cos(x); }
    BL_NO_INLINE f128_s tan(const f128_s& x) { return detail::_f128_constexpr::tan(x); }
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x) { return detail::_f128_constexpr::atan2(y, x); }

} // namespace bl::detail::_f128_runtime
