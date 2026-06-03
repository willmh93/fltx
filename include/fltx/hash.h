/**
 * fltx/hash.h - std::hash specializations for fltx types.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_HASH_INCLUDED
#define FLTX_HASH_INCLUDED

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "fltx/f128_type.h"
#include "fltx/f256_type.h"

namespace bl::detail
{
    [[nodiscard]] BL_FORCE_INLINE constexpr std::uint64_t hash_double_bits(double value) noexcept
    {
        return value == 0.0 ? 0u : std::bit_cast<std::uint64_t>(value);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr std::size_t hash_mix(std::size_t seed, std::uint64_t value) noexcept
    {
        if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t))
        {
            std::uint64_t x = value + 0x9e3779b97f4a7c15ull + (static_cast<std::uint64_t>(seed) << 6) + (static_cast<std::uint64_t>(seed) >> 2);
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ull;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebull;
            x ^= x >> 31;
            return static_cast<std::size_t>(x);
        }
        else
        {
            std::uint32_t x = static_cast<std::uint32_t>(value) ^
                              static_cast<std::uint32_t>(value >> 32) ^
                              static_cast<std::uint32_t>(seed);
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return static_cast<std::size_t>(x);
        }
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr std::size_t hash_f128(const f128_s& value) noexcept
    {
        std::size_t seed = 0x4ddc2d0f5b0d3911ull;
        seed = hash_mix(seed, hash_double_bits(value.hi));
        seed = hash_mix(seed, hash_double_bits(value.lo));
        return seed;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr std::size_t hash_f256(const f256_s& value) noexcept
    {
        std::size_t seed = 0x94d049bb133111ebull;
        seed = hash_mix(seed, hash_double_bits(value.x0));
        seed = hash_mix(seed, hash_double_bits(value.x1));
        seed = hash_mix(seed, hash_double_bits(value.x2));
        seed = hash_mix(seed, hash_double_bits(value.x3));
        return seed;
    }

} // namespace bl::detail

template<>
struct std::hash<bl::f128_s>
{
    [[nodiscard]] std::size_t operator()(const bl::f128_s& value) const noexcept
    {
        return bl::detail::hash_f128(value);
    }
};

template<>
struct std::hash<bl::f128>
{
    [[nodiscard]] std::size_t operator()(const bl::f128& value) const noexcept
    {
        return bl::detail::hash_f128(static_cast<const bl::f128_s&>(value));
    }
};

template<>
struct std::hash<bl::f256_s>
{
    [[nodiscard]] std::size_t operator()(const bl::f256_s& value) const noexcept
    {
        return bl::detail::hash_f256(value);
    }
};

template<>
struct std::hash<bl::f256>
{
    [[nodiscard]] std::size_t operator()(const bl::f256& value) const noexcept
    {
        return bl::detail::hash_f256(static_cast<const bl::f256_s&>(value));
    }
};

#endif
