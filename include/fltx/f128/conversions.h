/**
 * fltx/f128/conversions.h - f128 representation and scalar conversion helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_CONVERSIONS_INCLUDED
#define FLTX_F128_CONVERSIONS_INCLUDED
#include "fltx/f128/type.h"

namespace bl {

namespace detail::_f128
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

namespace detail::_f128_constexpr
{
    using namespace detail::_f128;

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(uint64_t u) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(int64_t v) noexcept;

    BL_FORCE_INLINE constexpr f128_s& assign(f128_s& out, uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f128_s& assign(f128_s& out, int64_t v) noexcept;

    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s floor(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s ceil(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s trunc(const f128_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s pow10_128(int k);

} // namespace detail::_f128_constexpr

namespace detail::_f128_runtime
{
    BL_NO_INLINE f128_s to_f128(uint64_t u) noexcept;
    BL_NO_INLINE f128_s to_f128(int64_t v) noexcept;
    BL_NO_INLINE f128_s& assign(f128_s& out, uint64_t u) noexcept;
    BL_NO_INLINE f128_s& assign(f128_s& out, int64_t v) noexcept;

} // namespace detail::_f128_runtime

BL_MSVC_NOINLINE constexpr f128_s& f128_s::operator=(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::assign(*this, u),
        detail::_f128_runtime::assign(*this, u)
    );
}

BL_MSVC_NOINLINE constexpr f128_s& f128_s::operator=(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::assign(*this, v),
        detail::_f128_runtime::assign(*this, v)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(double x) noexcept
{
    return f128_s{ x, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(float x) noexcept
{
    return f128_s{ (double)x, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(int32_t v) noexcept
{
    return f128_s{ (double)v, 0.0 };
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s to_f128(uint32_t v) noexcept
{
    return f128_s{ (double)v, 0.0 };
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s to_f128(uint64_t u) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::to_f128(u),
        detail::_f128_runtime::to_f128(u)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128_s to_f128(int64_t v) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::to_f128(v),
        detail::_f128_runtime::to_f128(v)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 floor(const f128_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 ceil(const f128_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 trunc(const f128_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 pow10_128(int k);

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::to_f128(uint64_t u) noexcept
{
    return detail::_f128::uint64_to_f128(u);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s detail::_f128_constexpr::to_f128(int64_t v) noexcept
{
    return detail::_f128::int64_to_f128(v);
}

BL_FORCE_INLINE constexpr f128_s& detail::_f128_constexpr::assign(f128_s& out, uint64_t u) noexcept
{
    out = to_f128(u);
    return out;
}

BL_FORCE_INLINE constexpr f128_s& detail::_f128_constexpr::assign(f128_s& out, int64_t v) noexcept
{
    out = to_f128(v);
    return out;
}

} // namespace bl

#endif
