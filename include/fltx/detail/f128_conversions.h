/**
 * fltx/detail/f128_conversions.h - f128 conversion primitives and implementation.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_DETAIL_CONVERSIONS_INCLUDED
#define F128_DETAIL_CONVERSIONS_INCLUDED
#include "fltx/f128_type.h"

namespace bl {

namespace detail::_f128 // primitives and kernels
{
    BL_FORCE_INLINE constexpr f128_s renorm(double hi, double lo)
    {
        double s{}, e{};
        two_sum_precise(hi, lo, s, e);
        return { s, e };
    }

    BL_FORCE_INLINE constexpr f128_s canonicalize_math_result(f128_s value) noexcept
    {
        value.lo = detail::fp::zero_low_fraction_bits_finite<8>(value.lo);
        return value;
    }

    #if defined(FLTX_CONSTEXPR_PARITY)
        #define F128_CANONICALIZE_MATH_RESULT(value) ::bl::detail::_f128::canonicalize_math_result(value)
    #else
        #define F128_CANONICALIZE_MATH_RESULT(value) (value)
    #endif

    BL_FORCE_INLINE constexpr f128_s uint64_to_f128(uint64_t value) noexcept
    {
        double sum{}, err{};
        uint64_to_exact_double_pair(value, sum, err);
        return renorm(sum, err);
    }

    BL_FORCE_INLINE constexpr f128_s int64_to_f128(int64_t value) noexcept
    {
        double sum{}, err{};
        int64_to_exact_double_pair(value, sum, err);
        return renorm(sum, err);
    }

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s integer_to_f128(T value) noexcept
    {
        if constexpr (std::is_signed_v<std::remove_cv_t<T>>)
            return int64_to_f128(static_cast<int64_t>(value));
        else
            return uint64_to_f128(static_cast<uint64_t>(value));
    }

} // namespace detail::_f128

namespace detail::_f128_impl
{
    using namespace detail::_f128;

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(uint64_t u) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(int64_t v) noexcept;

    BL_FORCE_INLINE constexpr f128_s& assign(f128_s& out, uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f128_s& assign(f128_s& out, int64_t v) noexcept;

} // namespace detail::_f128_impl

namespace detail::_f128_runtime
{
    BL_NO_INLINE f128_s to_f128(uint64_t u) noexcept;
    BL_NO_INLINE f128_s to_f128(int64_t v) noexcept;
    BL_NO_INLINE f128_s& assign(f128_s& out, uint64_t u) noexcept;
    BL_NO_INLINE f128_s& assign(f128_s& out, int64_t v) noexcept;

} // namespace detail::_f128_runtime

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::to_f128(uint64_t u) noexcept
{
    return detail::_f128::uint64_to_f128(u);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_impl::to_f128(int64_t v) noexcept
{
    return detail::_f128::int64_to_f128(v);
}

BL_FORCE_INLINE constexpr f128_s& detail::_f128_impl::assign(f128_s& out, uint64_t u) noexcept
{
    out = to_f128(u);
    return out;
}

BL_FORCE_INLINE constexpr f128_s& detail::_f128_impl::assign(f128_s& out, int64_t v) noexcept
{
    out = to_f128(v);
    return out;
}

} // namespace bl

#endif
