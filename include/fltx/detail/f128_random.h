/**
 * fltx/detail/f128_random.h - f128 hooks for constexpr random facilities.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_F128_RANDOM_INCLUDED
#define FLTX_DETAIL_F128_RANDOM_INCLUDED
#include <cstdint>
#include <limits>

#include "fltx/f128.h"
#include "fltx/f128_limits.h"

namespace bl::detail::random
{
    template<>
    struct real_traits<f128_s>
    {
        static constexpr bool enabled = true;
        static constexpr int digits = std::numeric_limits<f128_s>::digits;

        [[nodiscard]] BL_FORCE_INLINE static constexpr f128_s zero() noexcept { return f128_s{ 0.0 }; }
        [[nodiscard]] BL_FORCE_INLINE static constexpr f128_s one() noexcept { return f128_s{ 1.0 }; }

        template<class UInt>
        [[nodiscard]] BL_FORCE_INLINE static constexpr f128_s from_uint(UInt value) noexcept
        {
            f128_s out{};
            out = static_cast<std::uint64_t>(value);
            return out;
        }
    };

    template<>
    struct real_traits<f128>
    {
        static constexpr bool enabled = true;
        static constexpr int digits = std::numeric_limits<f128>::digits;

        [[nodiscard]] BL_FORCE_INLINE static constexpr f128 zero() noexcept { return f128{ 0.0 }; }
        [[nodiscard]] BL_FORCE_INLINE static constexpr f128 one() noexcept { return f128{ 1.0 }; }

        template<class UInt>
        [[nodiscard]] BL_FORCE_INLINE static constexpr f128 from_uint(UInt value) noexcept
        {
            return f128{ static_cast<std::uint64_t>(value) };
        }
    };
}

#endif
