#include "f256_math.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s f256_mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::f256_mul_add_horner_step_constexpr(a, b, c);
    }

    BL_NO_INLINE f256_s fmod(const f256_s& x, const f256_s& y) { return detail::_f256_constexpr::fmod(x, y); }
    BL_NO_INLINE f256_s round(const f256_s& a) { return detail::_f256_constexpr::round(a); }
    BL_NO_INLINE f256_s round_to_decimals(f256_s v, int prec) { return detail::_f256_constexpr::round_to_decimals(v, prec); }
    BL_NO_INLINE f256_s sqrt(const f256_s& a) { return detail::_f256_constexpr::sqrt(a); }
    BL_NO_INLINE f256_s nearbyint(const f256_s& a) { return detail::_f256_constexpr::nearbyint(a); }
    BL_NO_INLINE f256_s rint(const f256_s& x) { return detail::_f256_constexpr::nearbyint(x); }
    BL_NO_INLINE long lround(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256::round_half_away_zero(x));
    }
    BL_NO_INLINE long long llround(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256::round_half_away_zero(x));
    }
    BL_NO_INLINE long lrint(const f256_s& x)
    {
        long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long>(detail::_f256_constexpr::nearbyint(x));
    }
    BL_NO_INLINE long long llrint(const f256_s& x)
    {
        long long out = 0;
        if (detail::_f256::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f256::to_signed_integer_or_zero<long long>(detail::_f256_constexpr::nearbyint(x));
    }
    BL_NO_INLINE f256_s ldexp(const f256_s& a, int e) { return detail::_f256_constexpr::ldexp(a, e); }

    BL_NO_INLINE f256_s exp(const f256_s& x) { return detail::_f256_constexpr::exp(x); }
    BL_NO_INLINE f256_s exp2(const f256_s& x) { return detail::_f256_constexpr::exp2(x); }
    BL_NO_INLINE f256_s log(const f256_s& a) { return detail::_f256_constexpr::log(a); }
    BL_NO_INLINE f256_s log2(const f256_s& a) { return detail::_f256_constexpr::log2(a); }
    BL_NO_INLINE f256_s log10(const f256_s& a) { return detail::_f256_constexpr::log10(a); }
    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y) { return detail::_f256_constexpr::pow(x, y); }
    BL_NO_INLINE f256_s pow(const f256_s& x, double y) { return detail::_f256_constexpr::pow(x, y); }

    BL_NO_INLINE bool   sincos(const f256_s& x, f256_s& s_out, f256_s& c_out) { return detail::_f256_constexpr::sincos(x, s_out, c_out); }
    BL_NO_INLINE f256_s sin(const f256_s& x) { return detail::_f256_constexpr::sin(x); }
    BL_NO_INLINE f256_s cos(const f256_s& x) { return detail::_f256_constexpr::cos(x); }
    BL_NO_INLINE f256_s tan(const f256_s& x) { return detail::_f256_constexpr::tan(x); }
    BL_NO_INLINE f256_s atan(const f256_s& x) { return detail::_f256_constexpr::atan(x); }
    BL_NO_INLINE f256_s atan2(const f256_s& y, const f256_s& x) { return detail::_f256_constexpr::atan2(y, x); }
    BL_NO_INLINE f256_s asin(const f256_s& x) { return detail::_f256_constexpr::asin(x); }
    BL_NO_INLINE f256_s acos(const f256_s& x) { return detail::_f256_constexpr::acos(x); }

    BL_NO_INLINE f256_s expm1(const f256_s& x) { return detail::_f256_constexpr::expm1(x); }
    BL_NO_INLINE f256_s log1p(const f256_s& x) { return detail::_f256_constexpr::log1p(x); }
    BL_NO_INLINE f256_s sinh(const f256_s& x) { return detail::_f256_constexpr::sinh(x); }
    BL_NO_INLINE f256_s cosh(const f256_s& x) { return detail::_f256_constexpr::cosh(x); }
    BL_NO_INLINE f256_s tanh(const f256_s& x) { return detail::_f256_constexpr::tanh(x); }
    BL_NO_INLINE f256_s asinh(const f256_s& x) { return detail::_f256_constexpr::asinh(x); }
    BL_NO_INLINE f256_s acosh(const f256_s& x) { return detail::_f256_constexpr::acosh(x); }
    BL_NO_INLINE f256_s atanh(const f256_s& x) { return detail::_f256_constexpr::atanh(x); }

    BL_NO_INLINE f256_s cbrt(const f256_s& x) { return detail::_f256_constexpr::cbrt(x); }
    BL_NO_INLINE f256_s hypot(const f256_s& x, const f256_s& y) { return detail::_f256::hypot_impl(x, y); }
    BL_NO_INLINE f256_s remquo(const f256_s& x, const f256_s& y, int* quo) { return detail::_f256_constexpr::remquo(x, y, quo); }
    BL_NO_INLINE f256_s remainder(const f256_s& x, const f256_s& y) { return detail::_f256_constexpr::remainder(x, y); }
    BL_NO_INLINE f256_s frexp(const f256_s& x, int* exp) noexcept { return detail::_f256_constexpr::frexp(x, exp); }
    BL_NO_INLINE f256_s modf(const f256_s& x, f256_s* iptr) noexcept { return detail::_f256_constexpr::modf(x, iptr); }
    BL_NO_INLINE int    ilogb(const f256_s& x) noexcept { return detail::_f256_constexpr::ilogb(x); }
    BL_NO_INLINE f256_s logb(const f256_s& x) noexcept { return detail::_f256_constexpr::logb(x); }
    BL_NO_INLINE f256_s scalbn(const f256_s& x, int e) noexcept { return detail::_f256_constexpr::scalbn(x, e); }
    BL_NO_INLINE f256_s scalbln(const f256_s& x, long e) noexcept { return detail::_f256_constexpr::scalbln(x, e); }
    BL_NO_INLINE f256_s nextafter(const f256_s& from, const f256_s& to) noexcept { return detail::_f256_constexpr::nextafter(from, to); }
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, long double to) noexcept { return detail::_f256_constexpr::nexttoward(from, to); }
    BL_NO_INLINE f256_s nexttoward(const f256_s& from, const f256_s& to) noexcept { return detail::_f256_constexpr::nexttoward(from, to); }

    BL_NO_INLINE f256_s erf(const f256_s& x) { return detail::_f256_constexpr::erf(x); }
    BL_NO_INLINE f256_s erfc(const f256_s& x) { return detail::_f256_constexpr::erfc(x); }
    BL_NO_INLINE f256_s lgamma(const f256_s& x) { return detail::_f256_constexpr::lgamma(x); }
    BL_NO_INLINE f256_s tgamma(const f256_s& x) { return detail::_f256_constexpr::tgamma(x); }
}
