/**
 * fltx/detail/f256_random.h - f256 hooks for constexpr random facilities.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_F256_RANDOM_INCLUDED
#define FLTX_DETAIL_F256_RANDOM_INCLUDED
#include <cstdint>
#include <limits>

#include "fltx/f256.h"
#include "fltx/f256_limits.h"

namespace bl::detail::random
{
    template<>
    struct real_traits<f256_s>
    {
        static constexpr bool enabled = true;
        static constexpr int digits = std::numeric_limits<f256_s>::digits;

        [[nodiscard]] BL_FORCE_INLINE static constexpr f256_s zero() noexcept { return f256_s{ 0.0 }; }
        [[nodiscard]] BL_FORCE_INLINE static constexpr f256_s one() noexcept { return f256_s{ 1.0 }; }

        template<class UInt>
        [[nodiscard]] BL_FORCE_INLINE static constexpr f256_s from_uint(UInt value) noexcept
        {
            f256_s out{};
            out = static_cast<std::uint64_t>(value);
            return out;
        }
    };

    template<>
    struct real_traits<f256>
    {
        static constexpr bool enabled = true;
        static constexpr int digits = std::numeric_limits<f256>::digits;

        [[nodiscard]] BL_FORCE_INLINE static constexpr f256 zero() noexcept { return f256{ 0.0 }; }
        [[nodiscard]] BL_FORCE_INLINE static constexpr f256 one() noexcept { return f256{ 1.0 }; }

        template<class UInt>
        [[nodiscard]] BL_FORCE_INLINE static constexpr f256 from_uint(UInt value) noexcept
        {
            return f256{ static_cast<std::uint64_t>(value) };
        }
    };
}

#endif
