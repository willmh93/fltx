/**
 * fltx/detail/interop.h - Cross-type conversions and arithmetic for FLTX types.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

// This header is intentionally multi-pass. fltx/f128.h and fltx/f256.h both include it
// at the end; the interop bodies are emitted only after both types are complete.

#if defined(F128_INCLUDED) && defined(F256_INCLUDED) && !defined(FLTX_INTEROP_INCLUDED)
#define FLTX_INTEROP_INCLUDED

namespace bl
{
    BL_FORCE_INLINE constexpr f256_s::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f256_s::operator f128() const noexcept { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f128_s::operator f256_s() const noexcept { return f256_s{ hi, lo }; }

    BL_FORCE_INLINE constexpr f128::operator f256_s() const noexcept { return f256_s{ hi, lo }; }
    BL_FORCE_INLINE constexpr f128::operator f256() const noexcept { return f256_s{ hi, lo }; }

    BL_FORCE_INLINE constexpr f256::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f256::operator f128() const noexcept { return f128_s{ x0, x1 }; }

    BL_FORCE_INLINE constexpr f256::f256(f128_s x) noexcept
    {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
    }

    BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(f256_s x) noexcept
    {
        hi = x.x0; lo = x.x1;
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& f256_s::operator=(f128_s x) noexcept
    {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
        return *this;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, const f128_s& b) noexcept
    {
        return detail::_f256::add_dd(a, detail::_f256::dd_scalar{ b.hi, b.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, const f128_s& b) noexcept
    {
        return detail::_f256::sub_dd(a, detail::_f256::dd_scalar{ b.hi, b.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, const f128_s& b) noexcept
    {
        return detail::_f256::mul_dd(a, detail::_f256::dd_scalar{ b.hi, b.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, const f128_s& b) noexcept
    {
        return detail::_f256::div_dd(a, detail::_f256::dd_scalar{ b.hi, b.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f128_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::add_dd(b, detail::_f256::dd_scalar{ a.hi, a.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f128_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::sub_dd(detail::_f256::dd_scalar{ a.hi, a.lo }, b);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f128_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::mul_dd(b, detail::_f256::dd_scalar{ a.hi, a.lo });
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f128_s& a, const f256_s& b) noexcept
    {
        return detail::_f256::div_dd(detail::_f256::dd_scalar{ a.hi, a.lo }, b);
    }

    BL_FORCE_INLINE constexpr f256_s& f256_s::operator+=(const f128_s& rhs) noexcept
    {
        *this = *this + rhs;
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& f256_s::operator-=(const f128_s& rhs) noexcept
    {
        *this = *this - rhs;
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& f256_s::operator*=(const f128_s& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& f256_s::operator/=(const f128_s& rhs) noexcept
    {
        *this = *this / rhs;
        return *this;
    }

} // namespace bl

#endif
