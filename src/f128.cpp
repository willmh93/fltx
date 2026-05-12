#include "f128.h"

namespace bl::detail::_f128_runtime
{
    f128_s to_f128(uint64_t u) noexcept
    {
        return detail::_f128_constexpr::to_f128(u);
    }

    f128_s to_f128(int64_t v) noexcept
    {
        return detail::_f128_constexpr::to_f128(v);
    }

    f128_s& assign(f128_s& out, uint64_t u) noexcept
    {
        return detail::_f128_constexpr::assign(out, u);
    }

    f128_s& assign(f128_s& out, int64_t v) noexcept
    {
        return detail::_f128_constexpr::assign(out, v);
    }

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
}
