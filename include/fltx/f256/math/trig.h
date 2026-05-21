/**
 * fltx/f256/math/trig.h - f256 trigonometric functions.
 *
 * Range-reduced sine/cosine kernels, inverse trig wrappers, and sincos convenience overloads.
 * Runtime calls dispatch to compiled library bodies; constant evaluation uses
 * the matching detail core header.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_TRIG_INCLUDED
#define FLTX_F256_TRIG_INCLUDED
#include "fltx/detail/f256/math/trig.h"

namespace bl {

[[nodiscard]] BL_MSVC_NOINLINE constexpr bool sincos(const f256_s& x, f256_s& s_out, f256_s& c_out)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sincos(x, s_out, c_out),
        detail::_f256_runtime::sincos(x, s_out, c_out)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 sin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::sin(x),
        detail::_f256_runtime::sin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 cos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::cos(x),
        detail::_f256_runtime::cos(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 tan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::tan(x),
        detail::_f256_runtime::tan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 atan(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::atan(x),
        detail::_f256_runtime::atan(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 atan2(const f256_s& y, const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::atan2(y, x),
        detail::_f256_runtime::atan2(y, x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 asin(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::asin(x),
        detail::_f256_runtime::asin(x)
    );
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256 acos(const f256_s& x)
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f256_constexpr::acos(x),
        detail::_f256_runtime::acos(x)
    );
}

template<class Vec>
    requires detail::fp::sincos_vector_assignable<Vec, f256_s>
[[nodiscard]] BL_FORCE_INLINE constexpr bool sincos(const f256_s& x, Vec& out)
{
    f256_s s_out{};
    f256_s c_out{};
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
    requires (std::same_as<std::remove_cvref_t<Value>, f256> || std::same_as<std::remove_cvref_t<Value>, f256_s>)
[[nodiscard]] BL_FORCE_INLINE constexpr detail::fp::sincos_vector_result<std::remove_cvref_t<Value>> sincos(const f256_s& x)
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
