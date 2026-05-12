#include "f256.h"

namespace bl::detail::_f256_runtime
{
    f256_s floor(const f256_s& a)
    {
        return detail::_f256_constexpr::floor(a);
    }

    f256_s ceil(const f256_s& a)
    {
        return detail::_f256_constexpr::ceil(a);
    }

    f256_s trunc(const f256_s& a)
    {
        return detail::_f256_constexpr::trunc(a);
    }

    f256_s pow10_256(int k)
    {
        return detail::_f256_constexpr::pow10_256(k);
    }

    f256_s to_f256(uint64_t u) noexcept
    {
        return detail::_f256_constexpr::to_f256(u);
    }

    f256_s to_f256(int64_t v) noexcept
    {
        return detail::_f256_constexpr::to_f256(v);
    }

    f256_s& assign(f256_s& out, uint64_t u) noexcept
    {
        return detail::_f256_constexpr::assign(out, u);
    }

    f256_s& assign(f256_s& out, int64_t v) noexcept
    {
        return detail::_f256_constexpr::assign(out, v);
    }

}
