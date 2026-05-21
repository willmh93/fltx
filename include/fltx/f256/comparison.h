/**
 * fltx/f256/comparison.h - comparison operators for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F256_COMPARISON_INCLUDED
#define FLTX_F256_COMPARISON_INCLUDED
#include "fltx/f256/type.h"

namespace bl {

namespace detail::_f256
{
    BL_FORCE_INLINE constexpr bool compare_terms_less(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        if (ax0 < bx0) return true;
        if (ax0 > bx0) return false;
        if (ax1 < bx1) return true;
        if (ax1 > bx1) return false;
        if (ax2 < bx2) return true;
        if (ax2 > bx2) return false;
        return ax3 < bx3;
    }

    BL_FORCE_INLINE constexpr bool compare_terms_less_equal(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        if (ax0 < bx0) return true;
        if (ax0 > bx0) return false;
        if (ax1 < bx1) return true;
        if (ax1 > bx1) return false;
        if (ax2 < bx2) return true;
        if (ax2 > bx2) return false;
        return ax3 <= bx3;
    }

    BL_FORCE_INLINE constexpr bool compare_terms_equal(
        double ax0, double ax1, double ax2, double ax3,
        double bx0, double bx1, double bx2, double bx3) noexcept
    {
        if (isnan(ax0) || isnan(bx0))
            return false;

        return ax0 == bx0 && ax1 == bx1 && ax2 == bx2 && ax3 == bx3;
    }

    BL_FORCE_INLINE constexpr void uint64_compare_terms(uint64_t value, double& x0, double& x1, double& x2, double& x3) noexcept
    {
        uint64_to_exact_double_pair(value, x0, x1);
        x2 = 0.0;
        x3 = 0.0;
    }

    BL_FORCE_INLINE constexpr void int64_compare_terms(int64_t value, double& x0, double& x1, double& x2, double& x3) noexcept
    {
        int64_to_exact_double_pair(value, x0, x1);
        x2 = 0.0;
        x3 = 0.0;
    }

} // namespace detail::_f256


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return false;

    if (a.x0 < b.x0) return true;
    if (a.x0 > b.x0) return false;

    if (a.x1 < b.x1) return true;
    if (a.x1 > b.x1) return false;

    if (a.x2 < b.x2) return true;
    if (a.x2 > b.x2) return false;

    return a.x3 < b.x3;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return false;
    return b < a;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return false;
    return !(b < a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return false;
    return !(a < b);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return false;
    return a.x0 == b.x0 && a.x1 == b.x1 && a.x2 == b.x2 && a.x3 == b.x3;
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, const f256_s& b)
{
    if (detail::_f256::isnan(a.x0) || detail::_f256::isnan(b.x0))
        return true;
    return !(a == b);
}


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, double b) { return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, b, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(double a, const f256_s& b) { return detail::_f256::compare_terms_less(a, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, double b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(double a, const f256_s& b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, double b) { return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, b, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(double a, const f256_s& b) { return detail::_f256::compare_terms_less_equal(a, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, double b) { return detail::_f256::compare_terms_less_equal(b, 0.0, 0.0, 0.0, a.x0, a.x1, a.x2, a.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(double a, const f256_s& b) { return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, a, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, double b) { return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, b, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(double a, const f256_s& b) { return detail::_f256::compare_terms_equal(a, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, double b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(double a, const f256_s& b) { return !(a == b); }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, float b)  { const double bd = (double)b; return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(float a, const f256_s& b)  { const double ad = (double)a; return detail::_f256::compare_terms_less(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, float b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(float a, const f256_s& b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, float b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(float a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, float b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(bd, 0.0, 0.0, 0.0, a.x0, a.x1, a.x2, a.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(float a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, ad, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, float b) { const double bd = (double)b; return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(float a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, float b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(float a, const f256_s& b) { return !(a == b); }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, int32_t b)  { const double bd = (double)b; return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(int32_t a, const f256_s& b)  { const double ad = (double)a; return detail::_f256::compare_terms_less(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, int32_t b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(int32_t a, const f256_s& b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, int32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(int32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, int32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(bd, 0.0, 0.0, 0.0, a.x0, a.x1, a.x2, a.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(int32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, ad, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, int32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(int32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, int32_t b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(int32_t a, const f256_s& b) { return !(a == b); }

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, uint32_t b)  { const double bd = (double)b; return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(uint32_t a, const f256_s& b)  { const double ad = (double)a; return detail::_f256::compare_terms_less(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, uint32_t b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(uint32_t a, const f256_s& b)  { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, uint32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(uint32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, uint32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_less_equal(bd, 0.0, 0.0, 0.0, a.x0, a.x1, a.x2, a.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(uint32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, ad, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, uint32_t b) { const double bd = (double)b; return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, bd, 0.0, 0.0, 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(uint32_t a, const f256_s& b) { const double ad = (double)a; return detail::_f256::compare_terms_equal(ad, 0.0, 0.0, 0.0, b.x0, b.x1, b.x2, b.x3); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, uint32_t b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(uint32_t a, const f256_s& b) { return !(a == b); }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, int64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(int64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, int64_t b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(int64_t a, const f256_s& b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, int64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(int64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, int64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(x0, x1, x2, x3, a.x0, a.x1, a.x2, a.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(int64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, int64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(int64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::int64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_equal(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, int64_t b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(int64_t a, const f256_s& b) { return !(a == b); }

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f256_s& a, uint64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(uint64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f256_s& a, uint64_t b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(uint64_t a, const f256_s& b) { return b < a; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f256_s& a, uint64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(uint64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f256_s& a, uint64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(x0, x1, x2, x3, a.x0, a.x1, a.x2, a.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(uint64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_less_equal(b.x0, b.x1, b.x2, b.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f256_s& a, uint64_t b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(b, x0, x1, x2, x3);
    return detail::_f256::compare_terms_equal(a.x0, a.x1, a.x2, a.x3, x0, x1, x2, x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(uint64_t a, const f256_s& b)
{
    double x0{}, x1{}, x2{}, x3{};
    detail::_f256::uint64_compare_terms(a, x0, x1, x2, x3);
    return detail::_f256::compare_terms_equal(x0, x1, x2, x3, b.x0, b.x1, b.x2, b.x3);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f256_s& a, uint64_t b) { return !(a == b); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(uint64_t a, const f256_s& b) { return !(a == b); }

} // namespace bl

#endif
