#include "f128.h"

namespace bl::detail::_f128_runtime
{
    f128_s floor(const f128_s& a)
    {
        return detail::_f128_constexpr::floor(a);
    }

    f128_s ceil(const f128_s& a)
    {
        return detail::_f128_constexpr::ceil(a);
    }

    f128_s trunc(const f128_s& a)
    {
        return detail::_f128_constexpr::trunc(a);
    }

    f128_s pow10_128(int k)
    {
        return detail::_f128_constexpr::pow10_128(k);
    }

    f128_s sub_double_f128(double a, const f128_s& b) noexcept
    {
        return detail::_f128_constexpr::sub_double_f128(a, b);
    }

    f128_s div_double_f128(double a, const f128_s& b) noexcept
    {
        return detail::_f128_constexpr::div_double_f128(a, b);
    }
}
