/**
 * fltx/aliases.h - Cheap scalar aliases and fltx type forward declarations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_ALIASES_INCLUDED
#define FLTX_ALIASES_INCLUDED
#include <cstdint>

namespace bl
{
    struct f128;
    struct f128_s;
    struct f256;
    struct f256_s;

    using i8  = std::int8_t;
    using i16 = std::int16_t;
    using i32 = std::int32_t;
    using i64 = std::int64_t;

    using u8  = std::uint8_t;
    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;

    using f32 = float;
    using f64 = double;

    using ddreal = f128;
    using qdreal = f256;

} // namespace bl

#endif
