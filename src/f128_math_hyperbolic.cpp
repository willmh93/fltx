#include "fltx/detail/f128/math/hyperbolic.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s sinh(const f128_s& x) { return detail::_f128_constexpr::sinh(x); }
    BL_NO_INLINE f128_s cosh(const f128_s& x) { return detail::_f128_constexpr::cosh(x); }
    BL_NO_INLINE f128_s tanh(const f128_s& x) { return detail::_f128_constexpr::tanh(x); }
    BL_NO_INLINE f128_s asinh(const f128_s& x) { return detail::_f128_constexpr::asinh(x); }
    BL_NO_INLINE f128_s acosh(const f128_s& x) { return detail::_f128_constexpr::acosh(x); }
    BL_NO_INLINE f128_s atanh(const f128_s& x) { return detail::_f128_constexpr::atanh(x); }

} // namespace bl::detail::_f128_runtime
