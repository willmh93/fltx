/**
 * fltx/f128/comparison.h - comparison operators for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_COMPARISON_INCLUDED
#define FLTX_F128_COMPARISON_INCLUDED
#include "fltx/f128/type.h"

namespace bl {

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, const f128_s& b)  { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, const f128_s& b)  { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, const f128_s& b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, const f128_s& b) { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, const f128_s& b) { return a.hi == b.hi && a.lo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, const f128_s& b) { return a.hi != b.hi || a.lo != b.lo; }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, double b)  { return (a.hi < b) || (a.hi == b && a.lo <  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(double a, const f128_s& b)  { return (a < b.hi) || (a == b.hi && 0.0 <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, double b)  { return (a.hi > b) || (a.hi == b && a.lo >  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(double a, const f128_s& b)  { return (a > b.hi) || (a == b.hi && 0.0 >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, double b) { return (a.hi < b) || (a.hi == b && a.lo <= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(double a, const f128_s& b) { return (a < b.hi) || (a == b.hi && 0.0 <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, double b) { return (a.hi > b) || (a.hi == b && a.lo >= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(double a, const f128_s& b) { return (a > b.hi) || (a == b.hi && 0.0 >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, double b) { return a.hi == b && a.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(double a, const f128_s& b) { return a == b.hi && 0.0 == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, double b) { return a.hi != b || a.lo != 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(double a, const f128_s& b) { return a != b.hi || 0.0 != b.lo; }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, float b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(float a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, float b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(float a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, float b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(float a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, float b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(float a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, float b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(float a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, float b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(float a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, int32_t b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(int32_t a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, int32_t b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(int32_t a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, int32_t b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(int32_t a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, int32_t b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(int32_t a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, int32_t b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(int32_t a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, int32_t b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(int32_t a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, uint32_t b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(uint32_t a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, uint32_t b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(uint32_t a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint32_t b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(uint32_t a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint32_t b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(uint32_t a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, uint32_t b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(uint32_t a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint32_t b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(uint32_t a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }


[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, int64_t b)  { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(int64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, int64_t b)  { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(int64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::fp::int64_to_exact_double_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::int64_to_exact_double_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::fp::uint64_to_exact_double_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::fp::uint64_to_exact_double_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

} // namespace bl

#endif
