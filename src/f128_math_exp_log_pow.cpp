#include "fltx/detail/f128/math/pow.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s exp(const f128_s& x) { return detail::_f128_constexpr::exp(x); }
    BL_NO_INLINE f128_s exp2(const f128_s& x) { return detail::_f128_constexpr::exp2(x); }
    BL_NO_INLINE f128_s expm1(const f128_s& x) { return detail::_f128_constexpr::expm1(x); }

    BL_NO_INLINE f128_s log(const f128_s& a) { return detail::_f128_constexpr::log(a); }
    BL_NO_INLINE f128_s log2(const f128_s& a) { return detail::_f128_constexpr::log2(a); }
    BL_NO_INLINE f128_s log10(const f128_s& x) { return detail::_f128_constexpr::log10(x); }
    BL_NO_INLINE f128_s log1p(const f128_s& x) { return detail::_f128_constexpr::log1p(x); }

    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::pow(x, y); }
    BL_NO_INLINE f128_s pow(const f128_s& x, double y) { return detail::_f128_constexpr::pow(x, y); }

} // namespace bl::detail::_f128_runtime
