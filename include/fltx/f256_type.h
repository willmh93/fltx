/**
 * fltx/f256_type.h - f256 storage and value type declarations.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_TYPE_INCLUDED
#define F256_TYPE_INCLUDED
#include <type_traits>

#include "fltx/detail/common_fp.h"
#include "fltx/detail/simd.h"

#if !defined(BL_F256_ENABLE_SIMD)
#  if defined(BL_F256_ENABLE_TRIG_SIMD)
#    define BL_F256_ENABLE_SIMD BL_F256_ENABLE_TRIG_SIMD
#  elif BL_FLTX_HAS_SSE2 || BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD
#    define BL_F256_ENABLE_SIMD 1
#  else
#    define BL_F256_ENABLE_SIMD 0
#  endif
#endif

#if !defined(BL_F256_ENABLE_TRIG_SIMD)
#  define BL_F256_ENABLE_TRIG_SIMD BL_F256_ENABLE_SIMD
#endif

#if BL_F256_ENABLE_SIMD && !(BL_FLTX_HAS_SSE2 || BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
#  error "BL_F256_ENABLE_SIMD requires SSE2, AArch64 NEON, or wasm128 SIMD support."
#endif

namespace bl {

struct f128_s;
struct f128;
struct f256_s;

namespace detail::_f256 // primitives and kernels
{
    using detail::fp::absd;
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::quick_two_sum_precise;
    using detail::fp::uint64_to_exact_double_pair;
    using detail::fp::int64_to_exact_double_pair;
    using detail::fp::two_prod_precise;
    using detail::fp::two_sum_precise;
    using detail::fp::signbit;
    using detail::fp::fabs;
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::integer_fits_exact_double;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_fast(const f256_s& r, const f256_s& b, double q) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_exact(const f256_s& r, const f256_s& b, double q) noexcept;

    BL_FORCE_INLINE constexpr bool f256_runtime_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_SIMD && (BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD)
        return !bl::detail::use_constexpr_math();
        #else
        return false;
        #endif
    }

    BL_FORCE_INLINE constexpr bool f256_runtime_addsub_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_SIMD && BL_FLTX_HAS_NEON
        return !bl::detail::use_constexpr_math();
        #else
        return false;
        #endif
    }

    BL_FORCE_INLINE constexpr bool f256_runtime_trig_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_TRIG_SIMD
        return !bl::detail::use_constexpr_math();
        #else
        return false;
        #endif
    }

    BL_FORCE_INLINE constexpr bool f256_runtime_product_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_SIMD && BL_FLTX_HAS_NEON
        return !bl::detail::use_constexpr_math();
        #elif BL_F256_ENABLE_SIMD && BL_FLTX_HAS_WASM_SIMD
        return !bl::detail::use_constexpr_math();
        #elif BL_F256_ENABLE_SIMD && BL_FLTX_HAS_SSE2 && (!BL_FLTX_SIMD_USE_FMA_TWO_PROD || BL_FLTX_HAS_X86_FMA)
        return !bl::detail::use_constexpr_math();
        #else
        return false;
        #endif
    }

    template<class T>
    inline constexpr bool is_integer_scalar_v = detail::fp::is_integer_scalar_v<T>;

    template<class T>
    inline constexpr bool integer_type_fits_exact_double_v = detail::fp::integer_type_fits_exact_double_v<T>;

    // lightweight double-double type to avoid dragging in fltx/f128.h
    using dd_scalar = detail::fp::double_double;

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr dd_scalar integer_to_double_double(T value) noexcept;

    template<class T>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s integer_to_f256(T value) noexcept;

    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_dd(const f256_s& a, dd_scalar b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_dd(const f256_s& a, dd_scalar b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_dd(dd_scalar a, const f256_s& b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_dd(const f256_s& a, dd_scalar b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_dd(const f256_s& a, dd_scalar b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_dd(dd_scalar a, const f256_s& b) noexcept;

#if BL_F256_ENABLE_SIMD
    namespace simd = bl::detail::simd;
#endif

} // namespace detail::_f256

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, const f256_s& b) noexcept;

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, double b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, double b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, double b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, double b) noexcept;

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, float b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, float b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, float b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, float b) noexcept;

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, const f128_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, const f128_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, const f128_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, const f128_s& b) noexcept;

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f128_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f128_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f128_s& a, const f256_s& b) noexcept;
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f128_s& a, const f256_s& b) noexcept;

struct f256_s
{
    double x0, x1, x2, x3; // largest -> smallest

    BL_FORCE_INLINE constexpr f256_s& operator=(f128_s x) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator=(double x) noexcept { x0 = x; x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator=(float x) noexcept { x0 = static_cast<double>(x); x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this; }

    BL_FORCE_INLINE constexpr f256_s& operator=(uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator=(T v) noexcept
    {
        return (*this = static_cast<int64_t>(v));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator=(T v) noexcept
    {
        return (*this = static_cast<uint64_t>(v));
    }

    BL_FORCE_INLINE constexpr f256_s& operator+=(f256_s rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator-=(f256_s rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator*=(f256_s rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator/=(f256_s rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f256_s& operator+=(double rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator-=(double rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator*=(double rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator/=(double rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f256_s& operator+=(float rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator-=(float rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator*=(float rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator/=(float rhs) noexcept { *this = *this / rhs; return *this; }

    BL_FORCE_INLINE constexpr f256_s& operator+=(const f128_s& rhs) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator-=(const f128_s& rhs) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator*=(const f128_s& rhs) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator/=(const f128_s& rhs) noexcept;

    BL_FORCE_INLINE constexpr f256_s& operator+=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this + static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::add_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator-=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this - static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::sub_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator*=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this * static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::mul_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator/=(int64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this / static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::div_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator+=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this + static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::add_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator-=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this - static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::sub_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator*=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this * static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::mul_dd(*this, value);
        }
        return *this;
    }

    BL_FORCE_INLINE constexpr f256_s& operator/=(uint64_t rhs) noexcept
    {
        if (detail::fp::integer_fits_exact_double(rhs))
            *this = *this / static_cast<double>(rhs);
        else
        {
            const auto value = detail::_f256::integer_to_double_double(rhs);
            *this = detail::_f256::div_dd(*this, value);
        }
        return *this;
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator+=(T rhs) noexcept
    {
        return (*this += static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator-=(T rhs) noexcept
    {
        return (*this -= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator*=(T rhs) noexcept
    {
        return (*this *= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator/=(T rhs) noexcept
    {
        return (*this /= static_cast<int64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator+=(T rhs) noexcept
    {
        return (*this += static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator-=(T rhs) noexcept
    {
        return (*this -= static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator*=(T rhs) noexcept
    {
        return (*this *= static_cast<uint64_t>(rhs));
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator/=(T rhs) noexcept
    {
        return (*this /= static_cast<uint64_t>(rhs));
    }

    [[nodiscard]] explicit constexpr operator f128() const noexcept;
    [[nodiscard]] explicit constexpr operator f128_s() const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return ((x0 + x1) + (x2 + x3)); }
    [[nodiscard]] explicit constexpr operator float() const noexcept { return static_cast<float>(((x0 + x1) + (x2 + x3))); }
    [[nodiscard]] explicit constexpr operator int() const noexcept { return static_cast<int>(((x0 + x1) + (x2 + x3))); }

    [[nodiscard]] constexpr f256_s operator+() const { return *this; }
    [[nodiscard]] constexpr f256_s operator-() const noexcept { return f256_s{ -x0, -x1, -x2, -x3 }; }

    [[nodiscard]] static constexpr f256_s eps() { return { 3.038581678643134e-64, 0.0, 0.0, 0.0 }; } // ~2^-211
};

namespace detail::_f256_expr
{
    // defined in fltx/detail/f256_expressions.h
	// f256 needs this hook before its full definition.
    template<class Expr>
    struct is_expr : std::false_type {};

    template<class Expr>
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s eval_to_f256_s(const Expr& expr) noexcept;

} // namespace detail::_f256_expr

struct f256 : public f256_s
{
    f256() = default;
    constexpr f256(double _x0, double _x1, double _x2, double _x3) noexcept : f256_s{ _x0, _x1, _x2, _x3 } {}
    constexpr f256(float  x) noexcept : f256_s{ ((double)x), 0.0 } {}
    constexpr f256(double x) noexcept : f256_s{ ((double)x), 0.0 } {}
    constexpr f256(int64_t v) noexcept : f256_s{} { static_cast<f256_s&>(*this)  = static_cast<int64_t>(v); }
    constexpr f256(uint64_t u) noexcept : f256_s{} { static_cast<f256_s&>(*this) = static_cast<uint64_t>(u); }
    constexpr f256(int32_t  v) noexcept : f256((int64_t)v) {}
    constexpr f256(uint32_t u) noexcept : f256((int64_t)u) {}
    constexpr f256(const char*);

    constexpr f256(f128_s f) noexcept;
    constexpr f256(const f256_s& f) noexcept : f256_s{ f.x0, f.x1, f.x2, f.x3 } {}

    template<class Expr, std::enable_if_t<detail::_f256_expr::is_expr<std::remove_cv_t<std::remove_reference_t<Expr>>>::value && !std::is_lvalue_reference_v<Expr> && !std::is_const_v<std::remove_reference_t<Expr>>, int> = 0>
    BL_FORCE_INLINE constexpr f256(Expr&& expr) noexcept : f256_s{ detail::_f256_expr::eval_to_f256_s(expr) } {}

    using f256_s::operator=;

    template<class Expr, std::enable_if_t<detail::_f256_expr::is_expr<std::remove_cv_t<std::remove_reference_t<Expr>>>::value && !std::is_lvalue_reference_v<Expr> && !std::is_const_v<std::remove_reference_t<Expr>>, int> = 0>
    BL_FORCE_INLINE constexpr f256& operator=(Expr&& expr) noexcept
    {
        static_cast<f256_s&>(*this) = detail::_f256_expr::eval_to_f256_s(expr);
        return *this;
    }

    [[nodiscard]] explicit constexpr operator f128_s() const noexcept;
    [[nodiscard]] explicit constexpr operator f128()   const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return ((x0 + x1) + (x2 + x3)); }
    [[nodiscard]] explicit constexpr operator float()  const noexcept { return static_cast<float>(((x0 + x1) + (x2 + x3))); }
};

} // namespace bl

#endif
