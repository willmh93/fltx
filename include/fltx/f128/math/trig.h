/**
 * fltx/f128/math/trig.h - f128 trigonometric functions.
 *
 * Range-reduced sine/cosine kernels, inverse trig wrappers, tangent, atan2, and sincos convenience overloads.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_TRIG_INCLUDED
#define FLTX_F128_TRIG_INCLUDED
#include "fltx/detail/f128/math/trig.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr f128 atan(const f128_s& x)
{
    return detail::_f128::atan_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 asin(const f128_s& x)
{
    return detail::_f128::asin_impl(x);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128 acos(const f128_s& x)
{
    return detail::_f128::acos_impl(x);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sincos(x, s_out, c_out),
        detail::_f128_runtime::sincos(x, s_out, c_out)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 sin(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::sin(x),
        detail::_f128_runtime::sin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 cos(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::cos(x),
        detail::_f128_runtime::cos(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 tan(const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::tan(x),
        detail::_f128_runtime::tan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f128 atan2(const f128_s& y, const f128_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f128_constexpr::atan2(y, x),
        detail::_f128_runtime::atan2(y, x)
    );
}

template<class Vec>
    requires detail::fp::sincos_vector_assignable<Vec, f128_s>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(const f128_s& x, Vec& out)
{
    f128_s s_out{};
    f128_s c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    if (!ok)
    {
        s_out = bl::sin(x);
        c_out = bl::cos(x);
    }
    detail::fp::assign_sincos_vector(out, s_out, c_out);
    return ok;
}

template<class Value>
    requires (std::same_as<std::remove_cvref_t<Value>, f128> || std::same_as<std::remove_cvref_t<Value>, f128_s>)
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<std::remove_cvref_t<Value>> sincos(const f128_s& x)
{
    using Result = std::remove_cvref_t<Value>;

    Result s_out{};
    Result c_out{};
    const bool ok = bl::sincos(x, s_out, c_out);
    if (!ok)
    {
        s_out = bl::sin(x);
        c_out = bl::cos(x);
    }
    return detail::fp::make_sincos_result(s_out, c_out, ok);
}

} // namespace bl

#endif
