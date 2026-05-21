/**
 * fltx/f128/type.h - f128 storage and value type declarations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_TYPE_INCLUDED
#define FLTX_F128_TYPE_INCLUDED
#include "fltx/detail/fp.h"
#include "fltx/detail/simd.h"

#if !defined(BL_F128_ENABLE_SIMD)
#  if BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD
#    define BL_F128_ENABLE_SIMD 1
#  else
#    define BL_F128_ENABLE_SIMD 0
#  endif
#endif

#if BL_F128_ENABLE_SIMD && !(BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
#  error "BL_F128_ENABLE_SIMD requires AArch64 NEON or wasm128 SIMD support."
#endif

namespace bl {

struct f128_s;
struct f256_s;
struct f256;

namespace detail::_f128
{
    using detail::fp::absd;
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::uint64_to_exact_double_pair;
    using detail::fp::int64_to_exact_double_pair;
    using detail::fp::two_prod_precise;
    using detail::fp::two_sum_precise;

    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::integer_fits_exact_double;

    BL_FORCE_INLINE constexpr bool f128_runtime_product_pair_simd_enabled() noexcept
    {
        #if BL_F128_ENABLE_SIMD && !defined(FMA_AVAILABLE) && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        return !bl::use_constexpr_math();
        #else
        return false;
        #endif
    }

    #if BL_F128_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
    namespace simd = ::bl::detail::simd;
    #endif

    template<class T>
    inline constexpr bool is_integer_scalar_v = detail::fp::is_integer_scalar_v<T>;

    template<class T>
    inline constexpr bool integer_type_fits_exact_double_v = detail::fp::integer_type_fits_exact_double_v<T>;

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s integer_to_f128(T value) noexcept;

} // namespace detail::_f128

BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, const f128_s& b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, const f128_s& b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept;

BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept;

BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, float b) noexcept;

struct f128_s
{
    double hi, lo;

    BL_FORCE_INLINE constexpr f128_s& operator=(f256_s x) noexcept;
    BL_FORCE_INLINE constexpr f128_s& operator=(double x) noexcept { hi = x; lo = 0.0; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator=(float x) noexcept { hi = static_cast<double>(x); lo = 0.0; return *this; }

    BL_MSVC_NOINLINE constexpr f128_s& operator=(uint64_t u) noexcept;
    BL_MSVC_NOINLINE constexpr f128_s& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator=(T v) noexcept
    {
        return (*this = static_cast<int64_t>(v));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator=(T v) noexcept
    {
        return (*this = static_cast<uint64_t>(v));
    }

    BL_FORCE_INLINE constexpr f128_s& operator+=(f128_s rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(f128_s rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(f128_s rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(f128_s rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f128_s& operator+=(double rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(double rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(double rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(double rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f128_s& operator+=(float rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(float rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(float rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(float rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f128_s& operator+=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this + static_cast<double>(rhs);
        else {
            const f128_s value = detail::_f128::integer_to_f128(rhs);
            *this = *this + value;
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator-=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this - static_cast<double>(rhs);
        else {
            const f128_s value = detail::_f128::integer_to_f128(rhs);
            *this = *this - value;
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator*=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this * static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this * value; }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator/=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this / static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this / value; }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator+=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this + static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this + value; }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator-=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this - static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this - value; }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator*=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this * static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this * value; }
        return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator/=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this / static_cast<double>(rhs);
        else { const f128_s value = detail::_f128::integer_to_f128(rhs); *this = *this / value; }
        return *this;
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator+=(T rhs) noexcept
    {
        return (*this += static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator-=(T rhs) noexcept
    {
        return (*this -= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator*=(T rhs) noexcept
    {
        return (*this *= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator/=(T rhs) noexcept
    {
        return (*this /= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator+=(T rhs) noexcept
    {
        return (*this += static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator-=(T rhs) noexcept
    {
        return (*this -= static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator*=(T rhs) noexcept
    {
        return (*this *= static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator/=(T rhs) noexcept
    {
        return (*this /= static_cast<uint64_t>(rhs));
    }

    [[nodiscard]] constexpr operator f256_s() const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return hi + lo; }
    [[nodiscard]] explicit constexpr operator float() const noexcept { return static_cast<float>(hi + lo); }
    [[nodiscard]] explicit constexpr operator int() const noexcept { return static_cast<int>(hi + lo); }

    [[nodiscard]] constexpr f128_s operator+() const { return *this; }
    [[nodiscard]] constexpr f128_s operator-() const noexcept { return f128_s{ -hi, -lo }; }

    [[nodiscard]] static constexpr f128_s eps() { return { 1.232595164407831e-32, 0.0 }; }
};

struct f128 : public f128_s
{
    f128() = default;
    constexpr f128(double _hi, double _lo) noexcept : f128_s{ _hi, _lo } {}
    constexpr f128(float  x) noexcept : f128_s{ ((double)x), 0.0 } {}
    constexpr f128(double x) noexcept : f128_s{ ((double)x), 0.0 } {}
    constexpr f128(int64_t v) noexcept : f128_s{} { static_cast<f128_s&>(*this)  = static_cast<int64_t>(v); }
    constexpr f128(uint64_t u) noexcept : f128_s{} { static_cast<f128_s&>(*this) = static_cast<uint64_t>(u); }
    constexpr f128(int32_t  v) noexcept : f128((int64_t)v) {}
    constexpr f128(uint32_t u) noexcept : f128((int64_t)u) {}
    constexpr f128(const char*);

    constexpr f128(const f128_s& f) noexcept : f128_s{ f.hi, f.lo } {}

    using f128_s::operator=;

    [[nodiscard]] constexpr operator f256_s() const noexcept;
    [[nodiscard]] constexpr operator f256() const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return hi + lo; }
    [[nodiscard]] explicit constexpr operator float() const noexcept { return (float)(hi + lo); }
};

} // namespace bl

#endif
