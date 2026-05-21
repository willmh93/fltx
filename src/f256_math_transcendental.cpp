#include "fltx/detail/f256/math/exp_log.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s exp(const f256_s& x) { return detail::_f256_constexpr::exp(x); }
    BL_NO_INLINE f256_s exp2(const f256_s& x) { return detail::_f256_constexpr::exp2(x); }
    BL_NO_INLINE f256_s expm1(const f256_s& x) { return detail::_f256_constexpr::expm1(x); }

    BL_NO_INLINE f256_s log(const f256_s& a) { return detail::_f256_constexpr::log(a); }
    BL_NO_INLINE f256_s log2(const f256_s& a) { return detail::_f256_constexpr::log2(a); }
    BL_NO_INLINE f256_s log10(const f256_s& a) { return detail::_f256_constexpr::log10(a); }
    BL_NO_INLINE f256_s log1p(const f256_s& x) { return detail::_f256_constexpr::log1p(x); }

} // namespace bl::detail::_f256_runtime
