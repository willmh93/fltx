/**
 * fltx/f128.cpp - Runtime f128 conversion, rounding, and decimal power helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#include "fltx/f128_rounding.h"

namespace bl::detail::_f128_runtime
{
    f128_s to_f128(uint64_t u) noexcept
    {
        return detail::_f128_impl::to_f128(u);
    }

    f128_s to_f128(int64_t v) noexcept
    {
        return detail::_f128_impl::to_f128(v);
    }

    f128_s& assign(f128_s& out, uint64_t u) noexcept
    {
        return detail::_f128_impl::assign(out, u);
    }

    f128_s& assign(f128_s& out, int64_t v) noexcept
    {
        return detail::_f128_impl::assign(out, v);
    }

    f128_s trunc(const f128_s& a)
    {
        return detail::_f128_impl::trunc(a);
    }

    f128_s pow10_128(int k)
    {
        return detail::_f128_impl::pow10_128(k);
    }

} // namespace bl::detail::_f128_runtime
