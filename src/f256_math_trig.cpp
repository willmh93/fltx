#include "fltx/detail/f256/math/trig.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out) { return detail::_f256_constexpr::sincos(x, s_out, c_out); }
    BL_NO_INLINE f256_s sin(const f256_s& x) { return detail::_f256_constexpr::sin(x); }
    BL_NO_INLINE f256_s cos(const f256_s& x) { return detail::_f256_constexpr::cos(x); }
    BL_NO_INLINE f256_s tan(const f256_s& x) { return detail::_f256_constexpr::tan(x); }
    BL_NO_INLINE f256_s atan(const f256_s& x) { return detail::_f256_constexpr::atan(x); }
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x) { return detail::_f256_constexpr::atan2(y, x); }
    BL_NO_INLINE f256_s asin(const f256_s& x) { return detail::_f256_constexpr::asin(x); }
    BL_NO_INLINE f256_s acos(const f256_s& x) { return detail::_f256_constexpr::acos(x); }

} // namespace bl::detail::_f256_runtime
