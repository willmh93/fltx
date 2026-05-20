#include "f256_math.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s exp(const f256_s& x) { return detail::_f256_constexpr::exp(x); }
    BL_NO_INLINE f256_s exp2(const f256_s& x) { return detail::_f256_constexpr::exp2(x); }
    BL_NO_INLINE f256_s expm1(const f256_s& x) { return detail::_f256_constexpr::expm1(x); }

    BL_NO_INLINE f256_s log(const f256_s& a) { return detail::_f256_constexpr::log(a); }
    BL_NO_INLINE f256_s log2(const f256_s& a) { return detail::_f256_constexpr::log2(a); }
    BL_NO_INLINE f256_s log10(const f256_s& a) { return detail::_f256_constexpr::log10(a); }
    BL_NO_INLINE f256_s log1p(const f256_s& x) { return detail::_f256_constexpr::log1p(x); }

    BL_NO_INLINE bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out) { return detail::_f256_constexpr::sincos(x, s_out, c_out); }
    BL_NO_INLINE f256_s sin(const f256_s& x) { return detail::_f256_constexpr::sin(x); }
    BL_NO_INLINE f256_s cos(const f256_s& x) { return detail::_f256_constexpr::cos(x); }
    BL_NO_INLINE f256_s tan(const f256_s& x) { return detail::_f256_constexpr::tan(x); }
    BL_NO_INLINE f256_s atan(const f256_s& x) { return detail::_f256_constexpr::atan(x); }
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x) { return detail::_f256_constexpr::atan2(y, x); }
    BL_NO_INLINE f256_s asin(const f256_s& x) { return detail::_f256_constexpr::asin(x); }
    BL_NO_INLINE f256_s acos(const f256_s& x) { return detail::_f256_constexpr::acos(x); }

    BL_NO_INLINE f256_s sinh(const f256_s& x) { return detail::_f256_constexpr::sinh(x); }
    BL_NO_INLINE f256_s cosh(const f256_s& x) { return detail::_f256_constexpr::cosh(x); }
    BL_NO_INLINE f256_s tanh(const f256_s& x) { return detail::_f256_constexpr::tanh(x); }
    BL_NO_INLINE f256_s asinh(const f256_s& x) { return detail::_f256_constexpr::asinh(x); }
    BL_NO_INLINE f256_s acosh(const f256_s& x) { return detail::_f256_constexpr::acosh(x); }
    BL_NO_INLINE f256_s atanh(const f256_s& x) { return detail::_f256_constexpr::atanh(x); }
}
