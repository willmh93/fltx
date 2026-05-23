/**
 * fltx/detail/f256_conversions.h - f256 conversion primitives and implementation.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_F256_CONVERSIONS_INCLUDED
#define FLTX_DETAIL_F256_CONVERSIONS_INCLUDED
#include "fltx/f256_type.h"

namespace bl {

namespace detail::_f256_impl
{
    using namespace detail::_f256;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(uint64_t u) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(int64_t v) noexcept;

    BL_FORCE_INLINE constexpr f256_s& assign(f256_s& out, uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f256_s& assign(f256_s& out, int64_t v) noexcept;

} // namespace detail::_f256_impl

namespace detail::_f256_runtime
{
    BL_NO_INLINE f256_s to_f256(uint64_t u) noexcept;
    BL_NO_INLINE f256_s to_f256(int64_t v) noexcept;
    BL_NO_INLINE f256_s& assign(f256_s& out, uint64_t u) noexcept;
    BL_NO_INLINE f256_s& assign(f256_s& out, int64_t v) noexcept;

} // namespace detail::_f256_runtime

namespace detail::_f256 // primitives and kernels
{
    [[nodiscard]] BL_FORCE_INLINE constexpr dd_scalar uint64_to_double_double(uint64_t value) noexcept
    {
        double hi{}, lo{};
        uint64_to_exact_double_pair(value, hi, lo);
        return { hi, lo };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr dd_scalar int64_to_double_double(int64_t value) noexcept
    {
        double hi{}, lo{};
        int64_to_exact_double_pair(value, hi, lo);
        return { hi, lo };
    }

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr dd_scalar integer_to_double_double(T value) noexcept
    {
        if constexpr (std::is_signed_v<std::remove_cv_t<T>>)
            return int64_to_double_double(static_cast<int64_t>(value));
        else
            return uint64_to_double_double(static_cast<uint64_t>(value));
    }

} // namespace detail::_f256

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::to_f256(uint64_t u) noexcept
{
    const auto value = detail::_f256::uint64_to_double_double(u);
    return f256_s{ value.hi, value.lo };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_impl::to_f256(int64_t v) noexcept
{
    const auto value = detail::_f256::int64_to_double_double(v);
    return f256_s{ value.hi, value.lo };
}

BL_FORCE_INLINE constexpr f256_s& detail::_f256_impl::assign(f256_s& out, uint64_t u) noexcept
{
    out = to_f256(u);
    return out;
}

BL_FORCE_INLINE constexpr f256_s& detail::_f256_impl::assign(f256_s& out, int64_t v) noexcept
{
    out = to_f256(v);
    return out;
}

namespace detail::_f256 // primitives and kernels
{
    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s integer_to_f256(T value) noexcept
    {
        const auto expanded = integer_to_double_double(value);
        return f256_s{ expanded.hi, expanded.lo };
    }

} // namespace detail::_f256

} // namespace bl

#endif
