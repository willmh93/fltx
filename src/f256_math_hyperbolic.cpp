#include "fltx/detail/f256/math/hyperbolic.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s sinh(const f256_s& x) { return detail::_f256_constexpr::sinh(x); }
    BL_NO_INLINE f256_s cosh(const f256_s& x) { return detail::_f256_constexpr::cosh(x); }
    BL_NO_INLINE f256_s tanh(const f256_s& x) { return detail::_f256_constexpr::tanh(x); }
    BL_NO_INLINE f256_s asinh(const f256_s& x) { return detail::_f256_constexpr::asinh(x); }
    BL_NO_INLINE f256_s acosh(const f256_s& x) { return detail::_f256_constexpr::acosh(x); }
    BL_NO_INLINE f256_s atanh(const f256_s& x) { return detail::_f256_constexpr::atanh(x); }

} // namespace bl::detail::_f256_runtime
