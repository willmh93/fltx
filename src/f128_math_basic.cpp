#include "fltx/detail/f128/math/basic.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s fmod(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::fmod(x, y); }
    BL_NO_INLINE f128_s round(const f128_s& a) { return detail::_f128_constexpr::round(a); }
    BL_NO_INLINE f128_s nearbyint(const f128_s& a) { return detail::_f128_constexpr::nearbyint(a); }
    BL_NO_INLINE f128_s rint(const f128_s& x) { return detail::_f128_constexpr::rint(x); }
    BL_NO_INLINE f128_s round_to_decimals(f128_s v, int prec) { return detail::_f128_constexpr::round_to_decimals(v, prec); }
    BL_NO_INLINE f128_s remainder(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::remainder(x, y); }
    BL_NO_INLINE f128_s sqrt(f128_s a) { return detail::_f128_constexpr::sqrt(a); }
    BL_NO_INLINE f128_s ldexp(const f128_s& x, int e) { return detail::_f128_constexpr::ldexp(x, e); }

    BL_NO_INLINE long lround(const f128_s& x)
    {
        long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f128::lround_impl(x);
    }

    BL_NO_INLINE long long llround(const f128_s& x)
    {
        long long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, false, out))
            return out;
        return detail::_f128::llround_impl(x);
    }

    BL_NO_INLINE long lrint(const f128_s& x)
    {
        long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f128::lrint_impl(x);
    }

    BL_NO_INLINE long long llrint(const f128_s& x)
    {
        long long out = 0;
        if (detail::_f128::try_round_to_signed_integer(x, true, out))
            return out;
        return detail::_f128::llrint_impl(x);
    }

    BL_NO_INLINE f128_s cbrt(const f128_s& x) { return detail::_f128_constexpr::cbrt(x); }
    BL_NO_INLINE f128_s hypot(const f128_s& x, const f128_s& y) { return detail::_f128_constexpr::hypot(x, y); }
    BL_NO_INLINE f128_s remquo(const f128_s& x, const f128_s& y, int* quo) { return detail::_f128_constexpr::remquo(x, y, quo); }
    BL_NO_INLINE f128_s frexp(const f128_s& x, int* exp) noexcept { return detail::_f128_constexpr::frexp(x, exp); }
    BL_NO_INLINE f128_s modf(const f128_s& x, f128_s* iptr) noexcept { return detail::_f128_constexpr::modf(x, iptr); }
    BL_NO_INLINE f128_s nextafter(const f128_s& from, const f128_s& to) noexcept { return detail::_f128_constexpr::nextafter(from, to); }

} // namespace bl::detail::_f128_runtime
