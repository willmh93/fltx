#include "fltx/detail/f128/math/basic.h"
#include "fltx/detail/f128/math/hyperbolic.h"
#include "fltx/detail/f128/math_shared.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s horner_forward(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f128_s horner_reverse(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f128_s* left_coeffs,
        const f128_s* right_coeffs,
        std::size_t count,
        const f128_s& x,
        f128_s& left_out,
        f128_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f128_s{};
            right_out = f128_s{};
            return;
        }

        f128_s left  = left_coeffs[0];
        f128_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            detail::_f128::mul_add_pair_same_rhs_inline(left, right, x, left_coeffs[i], right_coeffs[i], left, right);

        left_out = left;
        right_out = right;
    }

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

    BL_NO_INLINE f128_s sinh(const f128_s& x) { return detail::_f128_constexpr::sinh(x); }
    BL_NO_INLINE f128_s cosh(const f128_s& x) { return detail::_f128_constexpr::cosh(x); }
    BL_NO_INLINE f128_s tanh(const f128_s& x) { return detail::_f128_constexpr::tanh(x); }
    BL_NO_INLINE f128_s asinh(const f128_s& x) { return detail::_f128_constexpr::asinh(x); }
    BL_NO_INLINE f128_s acosh(const f128_s& x) { return detail::_f128_constexpr::acosh(x); }
    BL_NO_INLINE f128_s atanh(const f128_s& x) { return detail::_f128_constexpr::atanh(x); }

} // namespace bl::detail::_f128_runtime
