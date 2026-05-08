#include "f128_math.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::fmod(x, y); }
    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec) { return detail::_f128_constexpr::round_to_decimals(v, prec); }
    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::remainder(x, y); }
    BL_NO_INLINE f128_s sqrt(f128_s a) { return detail::_f128_constexpr::sqrt(a); }
    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e) { return detail::_f128_constexpr::ldexp(x, e); }

    BL_NO_INLINE f128_s exp(const f128_s& x) { return detail::_f128_constexpr::exp(x); }
    BL_NO_INLINE f128_s exp2(const f128_s& x) { return detail::_f128_constexpr::exp2(x); }
    BL_NO_INLINE f128_s log(const f128_s& a) { return detail::_f128_constexpr::log(a); }
    BL_NO_INLINE f128_s log2(const f128_s& a) { return detail::_f128_constexpr::log2(a); }
    BL_NO_INLINE f128_s log10(const f128_s& x) { return detail::_f128_constexpr::log10(x); }
    BL_NO_INLINE f128_s pow(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::pow(x, y); }
    BL_NO_INLINE f128_s pow(const f128_s& x, double y) { return detail::_f128_constexpr::pow(x, y); }

    BL_NO_INLINE bool   sincos(const f128_s& x, f128_s& s_out, f128_s& c_out) { return detail::_f128_constexpr::sincos(x, s_out, c_out); }
    BL_NO_INLINE f128_s sin(const f128_s& x) { return detail::_f128_constexpr::sin(x); }
    BL_NO_INLINE f128_s cos(const f128_s& x) { return detail::_f128_constexpr::cos(x); }
    BL_NO_INLINE f128_s tan(const f128_s& x) { return detail::_f128_constexpr::tan(x); }
    BL_NO_INLINE f128_s atan2(const f128_s& y, const f128_s& x) { return detail::_f128_constexpr::atan2(y, x); }

    BL_NO_INLINE f128_s expm1(const f128_s& x) { return detail::_f128_constexpr::expm1(x); }
    BL_NO_INLINE f128_s log1p(const f128_s& x) { return detail::_f128_constexpr::log1p(x); }
    BL_NO_INLINE f128_s sinh(const f128_s& x) { return detail::_f128_constexpr::sinh(x); }
    BL_NO_INLINE f128_s cosh(const f128_s& x) { return detail::_f128_constexpr::cosh(x); }
    BL_NO_INLINE f128_s tanh(const f128_s& x) { return detail::_f128_constexpr::tanh(x); }
    BL_NO_INLINE f128_s asinh(const f128_s& x) { return detail::_f128_constexpr::asinh(x); }
    BL_NO_INLINE f128_s acosh(const f128_s& x) { return detail::_f128_constexpr::acosh(x); }
    BL_NO_INLINE f128_s atanh(const f128_s& x) { return detail::_f128_constexpr::atanh(x); }

    BL_NO_INLINE f128_s cbrt(const f128_s& x) { return detail::_f128_constexpr::cbrt(x); }
    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::hypot(x, y); }
    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo) { return detail::_f128_constexpr::remquo(x, y, quo); }
    BL_NO_INLINE f128_s frexp(const f128_s& x, int* exp) noexcept { return detail::_f128_constexpr::frexp(x, exp); }
    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept { return detail::_f128_constexpr::modf(x, iptr); }
    BL_NO_INLINE f128_s nextafter(const f128_s& from, const f128_s& to) noexcept { return detail::_f128_constexpr::nextafter(from, to); }

    BL_NO_INLINE f128_s erf(const f128_s& x) { return detail::_f128_constexpr::erf(x); }
    BL_NO_INLINE f128_s erfc(const f128_s& x) { return detail::_f128_constexpr::erfc(x); }
    BL_NO_INLINE f128_s lgamma(const f128_s& x) { return detail::_f128_constexpr::lgamma(x); }
    BL_NO_INLINE f128_s tgamma(const f128_s& x) { return detail::_f128_constexpr::tgamma(x); }
}
