/**
 * f128.h - f128 double-double type and core arithmetic.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_INCLUDED
#define F128_INCLUDED

#include "fltx_common_math.h"
#include <numbers>

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
    using detail::fp::magnitude_u64;
    using detail::fp::split_uint64_to_doubles;
    using detail::fp::two_prod_precise;
    using detail::fp::two_sum_precise;

    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::double_integer_is_odd;
    using detail::fp::fmod_constexpr;
    using detail::fp::sqrt_seed_constexpr;
    using detail::fp::nearbyint_ties_even;
}

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
    BL_FORCE_INLINE constexpr f128_s& operator=(double x) noexcept {
        hi = x; lo = 0.0; return *this;
    }
    BL_FORCE_INLINE constexpr f128_s& operator=(float x) noexcept {
        hi = static_cast<double>(x); lo = 0.0; return *this;
    }

    BL_FORCE_INLINE constexpr f128_s& operator=(uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f128_s& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator=(T v) noexcept {
        return (*this = static_cast<int64_t>(v));
    }
    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f128_s& operator=(T v) noexcept {
        return (*this = static_cast<uint64_t>(v));
    }

    // f128 ops
    BL_FORCE_INLINE constexpr f128_s& operator+=(f128_s rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(f128_s rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(f128_s rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(f128_s rhs) noexcept { *this = *this / rhs; return *this; }

    // f64 ops
    BL_FORCE_INLINE constexpr f128_s& operator+=(double rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(double rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(double rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(double rhs) noexcept { *this = *this / rhs; return *this; }

    // f32 ops
    BL_FORCE_INLINE constexpr f128_s& operator+=(float rhs) noexcept { *this = *this + rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator-=(float rhs) noexcept { *this = *this - rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator*=(float rhs) noexcept { *this = *this * rhs; return *this; }
    BL_FORCE_INLINE constexpr f128_s& operator/=(float rhs) noexcept { *this = *this / rhs; return *this; }

    /// ======== conversions ========
    [[nodiscard]] constexpr operator f256_s() const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return hi + lo; }
    [[nodiscard]] explicit constexpr operator float() const noexcept { return static_cast<float>(hi + lo); }
    [[nodiscard]] explicit constexpr operator int() const noexcept { return static_cast<int>(hi + lo); }

    [[nodiscard]] constexpr f128_s operator+() const { return *this; }
    [[nodiscard]] constexpr f128_s operator-() const noexcept { return f128_s{ -hi, -lo }; }

    /// ======== utility ========
    [[nodiscard]] static constexpr f128_s eps() { return { 1.232595164407831e-32, 0.0 }; }
};

struct f128 : public f128_s
{
    f128() = default;
    constexpr f128(double _hi, double _lo) noexcept : f128_s{ _hi, _lo } {}
    constexpr f128(float  x) noexcept : f128_s{ ((double)x), 0.0 } {}
    constexpr f128(double x) noexcept : f128_s{ ((double)x), 0.0 } {}
    constexpr f128(int64_t  v) noexcept : f128_s{} { static_cast<f128_s&>(*this) = static_cast<int64_t>(v); }
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

} // end bl


template<>
struct std::numeric_limits<bl::f128_s>
{
    static constexpr bool is_specialized = true;

    // limits
    static constexpr bl::f128_s min()            noexcept { return { numeric_limits<double>::min(), 0.0 }; }
    static constexpr bl::f128_s max()            noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon() }; }
    static constexpr bl::f128_s lowest()         noexcept { return { -numeric_limits<double>::max(), numeric_limits<double>::epsilon() }; }

    // special values                        
    static constexpr bl::f128_s epsilon()        noexcept { return { 1.232595164407831e-32, 0.0 }; } // ~2^-106, a single ulp of double-double
    static constexpr bl::f128_s round_error()    noexcept { return { 0.5, 0.0 }; }
    static constexpr bl::f128_s infinity()       noexcept { return { numeric_limits<double>::infinity(), 0.0 }; }
    static constexpr bl::f128_s quiet_NaN()      noexcept { return { numeric_limits<double>::quiet_NaN(), 0.0 }; }
    static constexpr bl::f128_s signaling_NaN()  noexcept { return { numeric_limits<double>::signaling_NaN(), 0.0 }; }
    static constexpr bl::f128_s denorm_min()     noexcept { return { numeric_limits<double>::denorm_min(), 0.0 }; }

    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;

    // properties                            
    static constexpr int  digits = 106;  // ~53 bits * 2
    static constexpr int  digits10 = 31;   // log10(2^106) ≈ 31.9
    static constexpr int  max_digits10 = 33;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int  radix = 2;

    // exponent range                        
    static constexpr int  min_exponent = numeric_limits<double>::min_exponent;
    static constexpr int  max_exponent = numeric_limits<double>::max_exponent;
    static constexpr int  min_exponent10 = numeric_limits<double>::min_exponent10;
    static constexpr int  max_exponent10 = numeric_limits<double>::max_exponent10;

    // properties                            
    static constexpr bool is_iec559 = false; // not IEEE-754 compliant
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;

    // rounding                              
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr float_round_style round_style = round_to_nearest;
};

BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS(bl::f128, bl::f128_s)

namespace std::numbers
{
    template<> inline constexpr bl::f128_s e_v<bl::f128_s>          = { 0x1.5bf0a8b145769p+1,  0x1.4d57ee2b1013ap-53 };
    template<> inline constexpr bl::f128_s log2e_v<bl::f128_s>      = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56 };
    template<> inline constexpr bl::f128_s log10e_v<bl::f128_s>     = { 0x1.bcb7b1526e50ep-2,  0x1.95355baaafad3p-57 };
    template<> inline constexpr bl::f128_s pi_v<bl::f128_s>         = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53 };
    template<> inline constexpr bl::f128_s inv_pi_v<bl::f128_s>     = { 0x1.45f306dc9c883p-2, -0x1.6b01ec5417056p-56 };
    template<> inline constexpr bl::f128_s inv_sqrtpi_v<bl::f128_s> = { 0x1.20dd750429b6dp-1,  0x1.1ae3a914fed80p-57 };
    template<> inline constexpr bl::f128_s ln2_v<bl::f128_s>        = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56 };
    template<> inline constexpr bl::f128_s ln10_v<bl::f128_s>       = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53 };
    template<> inline constexpr bl::f128_s sqrt2_v<bl::f128_s>      = { 0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54 };
    template<> inline constexpr bl::f128_s sqrt3_v<bl::f128_s>      = { 0x1.bb67ae8584caap+0,  0x1.cec95d0b5c1e3p-54 };
    template<> inline constexpr bl::f128_s inv_sqrt3_v<bl::f128_s>  = { 0x1.279a74590331cp-1,  0x1.34863e0792bedp-55 };
    template<> inline constexpr bl::f128_s egamma_v<bl::f128_s>     = { 0x1.2788cfc6fb619p-1, -0x1.6cb90701fbfabp-58 };
    template<> inline constexpr bl::f128_s phi_v<bl::f128_s>        = { 0x1.9e3779b97f4a8p+0, -0x1.f506319fcfd19p-55 };

    template<> inline constexpr bl::f128 e_v<bl::f128>          = bl::f128{ e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 log2e_v<bl::f128>      = bl::f128{ log2e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 log10e_v<bl::f128>     = bl::f128{ log10e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 pi_v<bl::f128>         = bl::f128{ pi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_pi_v<bl::f128>     = bl::f128{ inv_pi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_sqrtpi_v<bl::f128> = bl::f128{ inv_sqrtpi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 ln2_v<bl::f128>        = bl::f128{ ln2_v<bl::f128_s> };
    template<> inline constexpr bl::f128 ln10_v<bl::f128>       = bl::f128{ ln10_v<bl::f128_s> };
    template<> inline constexpr bl::f128 sqrt2_v<bl::f128>      = bl::f128{ sqrt2_v<bl::f128_s> };
    template<> inline constexpr bl::f128 sqrt3_v<bl::f128>      = bl::f128{ sqrt3_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_sqrt3_v<bl::f128>  = bl::f128{ inv_sqrt3_v<bl::f128_s> };
    template<> inline constexpr bl::f128 egamma_v<bl::f128>     = bl::f128{ egamma_v<bl::f128_s> };
    template<> inline constexpr bl::f128 phi_v<bl::f128>        = bl::f128{ phi_v<bl::f128_s> };
}

namespace bl {

using std::numeric_limits;
namespace numbers = std::numbers;

/// ======== Representation helpers and scalar conversions ========

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr f128_s renorm(double hi, double lo)
    {
        double s{}, e{};
        detail::_f128::two_sum_precise(hi, lo, s, e);
        return { s, e };
    }
    BL_FORCE_INLINE constexpr f128_s canonicalize_math_result(f128_s value) noexcept
    {
        value.lo = detail::fp::zero_low_fraction_bits_finite<8>(value.lo);
        return value;
    }
    BL_FORCE_INLINE constexpr f128_s uint64_to_f128(uint64_t value) noexcept
    {
        double hi{}, lo{};
        split_uint64_to_doubles(value, hi, lo);

        double sum{}, err{};
        two_sum_precise(hi, lo, sum, err);
        return renorm(sum, err);
    }
    BL_FORCE_INLINE constexpr f128_s int64_to_f128(int64_t value) noexcept
    {
        f128_s result = uint64_to_f128(magnitude_u64(value));
        if (value < 0)
            result = -result;
        return result;
    }
}

BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(uint64_t u) noexcept {
    const f128_s result = detail::_f128::uint64_to_f128(u);
    hi = result.hi; lo = result.lo; return *this;
}
BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(int64_t v) noexcept {
    const f128_s result = detail::_f128::int64_to_f128(v);
    hi = result.hi; lo = result.lo; return *this;
}

[[nodiscard]] constexpr f128_s to_f128(double x) noexcept { return f128_s{ x, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(float x) noexcept { return f128_s{ (double)x, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(int32_t v) noexcept { return f128_s{ (double)v, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(uint32_t v) noexcept { return f128_s{ (double)v, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(int64_t v) noexcept { return detail::_f128::int64_to_f128(v); }
[[nodiscard]] constexpr f128_s to_f128(uint64_t u)  noexcept { return detail::_f128::uint64_to_f128(u); }

/// ======== Comparisons ========

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr void uint64_compare_pair(uint64_t value, double& hi, double& lo) noexcept
    {
        double a{}, b{};
        detail::_f128::split_uint64_to_doubles(value, a, b);
        detail::_f128::two_sum_precise(a, b, hi, lo);
    }

    BL_FORCE_INLINE constexpr void int64_compare_pair(int64_t value, double& hi, double& lo) noexcept
    {
        detail::_f128::uint64_compare_pair(detail::_f128::magnitude_u64(value), hi, lo);
        if (value < 0) { hi = -hi; lo = -lo; }
    }
}
// ------------------ f128 <=> f128 ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, const f128_s& b)  { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, const f128_s& b)  { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, const f128_s& b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, const f128_s& b) { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, const f128_s& b) { return a.hi == b.hi && a.lo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, const f128_s& b) { return a.hi != b.hi || a.lo != b.lo; }

// ------------------ double <=> f128 ------------------

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

// ------------------ float <=> f128 ------------------

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

// --------------- ints <=> f128 ---------------

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

// ------------------ int64_t/uint64_t <=> f128 ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, int64_t b)  { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(int64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, int64_t b)  { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(int64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, int64_t b) { double bhi{}, blo{}; detail::_f128::int64_compare_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(int64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::int64_compare_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator<=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator>=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator==(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; detail::_f128::uint64_compare_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool operator!=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; detail::_f128::uint64_compare_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

// ------------------ classification ------------------


BL_FORCE_INLINE constexpr f128_s abs(const f128_s& a);
[[nodiscard]] BL_NO_INLINE constexpr f128_s pow10_128(int k)
{
    if (k == 0) [[unlikely]]
        return f128_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    // fast small-exponent path
    if (n <= 16) {
        f128_s r = f128_s{ 1.0 };
        const f128_s ten = f128_s{ 10.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f128_s{ 1.0 } / r);
    }

    f128_s r = f128_s{ 1.0 };
    f128_s base = f128_s{ 10.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f128_s{ 1.0 } / r);
}

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(const f128_s& x) noexcept      { return detail::fp::isnan(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(const f128_s& x) noexcept      { return detail::fp::isinf(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(const f128_s& x) noexcept   { return detail::fp::isfinite(x.hi); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(const f128_s& x) noexcept     { return x.hi == 0.0 && x.lo == 0.0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool ispositive(const f128_s& x) noexcept { return x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0); }

[[nodiscard]] BL_FORCE_INLINE constexpr bool signbit(const f128_s& x) noexcept
{
    return detail::_f128::signbit_constexpr(x.hi) || (x.hi == 0.0 && detail::_f128::signbit_constexpr(x.lo));
}
[[nodiscard]] BL_FORCE_INLINE constexpr int  fpclassify(const f128_s& x) noexcept
{
    if (isnan(x))  [[unlikely]] return FP_NAN;
    if (isinf(x))  [[unlikely]] return FP_INFINITE;
    if (iszero(x)) [[unlikely]] return FP_ZERO;

    return abs(x) < std::numeric_limits<f128_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isnormal(const f128_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isunordered(const f128_s& a, const f128_s& b) noexcept
{
    return isnan(a) || isnan(b);
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isgreaterequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool isless(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}
[[nodiscard]] BL_FORCE_INLINE constexpr bool islessgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

/// ------------------ arithmetic operators ------------------

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s clamp(const f128_s& v, const f128_s& lo, const f128_s& hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s abs(const f128_s& a)
{
    return (a.hi < 0.0) ? -a : a;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s floor(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = 4503599627370496.0;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128::floor_constexpr(a.lo));
    }

    f128_s r{ detail::_f128::floor_constexpr(a.hi), 0.0 };
    if (r > a)
        r -= 1.0;
    return r;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s ceil(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    constexpr double integer_hi_threshold = 4503599627370496.0;

    if (detail::_f128::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return detail::_f128::renorm(a.hi, detail::_f128::ceil_constexpr(a.lo));
    }

    f128_s r{ detail::_f128::ceil_constexpr(a.hi), 0.0 };
    if (r < a)
        r += 1.0;
    return r;
}
[[nodiscard]] BL_NO_INLINE constexpr f128_s trunc(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a)) [[unlikely]]
        return a;

    return (a.hi < 0.0) ? ceil(a) : floor(a);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s recip(f128_s b)
{
    constexpr f128_s one = f128_s{ 1.0 };
    f128_s y = f128_s{ 1.0 / b.hi };
    f128_s e = one - b * y;

    y += y * e;
    e = one - b * y;
    y += y * e;

    return y;
}

/// ------------------ core helpers ------------------

namespace detail::_f128
{
    BL_FORCE_INLINE constexpr f128_s sub_mul_scalar_exact(const f128_s& r, const f128_s& b, double q) noexcept
    {
        double p{}, e{};
        two_prod_precise(b.hi, q, p, e);
        e += b.lo * q;

        double s{}, t{};
        two_sum_precise(r.hi, -p, s, t);
        t += r.lo - e;

        return renorm(s, t);
    }

    [[nodiscard]] inline BL_NO_INLINE f128_s div_f128_double_runtime(const f128_s& a, double b) noexcept
    {
        const double q0 = a.hi / b;
        const f128_s r = sub_mul_scalar_exact(a, f128_s{ b, 0.0 }, q0);
        const double q1 = r.hi / b;
        return renorm(q0, q1);
    }

    /// ------------------ scalar (fast inline) ------------------

    BL_PUSH_PRECISE;
    BL_FORCE_INLINE constexpr void mul_expansion_inline(const f128_s& a, const f128_s& b, double& p, double& e) noexcept
    {
        detail::_f128::two_prod_precise(a.hi, b.hi, p, e);

        e += a.hi * b.lo + a.lo * b.hi;
        e += a.lo * b.lo;
    }
    BL_POP_PRECISE;

    BL_PUSH_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s add_inline(const f128_s& a, const f128_s& b) noexcept
    {
        // accurate sum of the high parts
        double s{}, e{};
        detail::_f128::two_sum_precise(a.hi, b.hi, s, e);

        // fold low parts into the error
        double t = a.lo + b.lo;
        e += t;

        // renormalize
        return detail::_f128::renorm(s, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double s{}, e{};
        detail::_f128::two_sum_precise(a.hi, -b.hi, s, e);

        double t = a.lo - b.lo;
        e += t;

        return detail::_f128::renorm(s, e);
    }
    BL_POP_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_inline(const f128_s& a, const f128_s& b) noexcept
    {
        double p, e;
        detail::_f128::two_prod_precise(a.hi, b.hi, p, e);

        e += a.hi * b.lo + a.lo * b.hi;
        e += a.lo * b.lo;
        return detail::_f128::renorm(p, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s div_inline(const f128_s& a, const f128_s& b) noexcept
    {
        if (b.lo == 0.0)
            return a / b.hi;

        const double inv_b0 = 1.0 / b.hi;

        const double q0 = a.hi * inv_b0;
        f128_s r = detail::_f128::sub_mul_scalar_exact(a, b, q0);

        const double q1 = r.hi * inv_b0;

        return detail::_f128::renorm(q0, q1);
    }

    BL_PUSH_PRECISE;
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_add_inline(const f128_s& a, const f128_s& b, const f128_s& c) noexcept
    {
        double p{}, e{};
        detail::_f128::mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        detail::_f128::two_sum_precise(p, c.hi, s, t);
        t += e + c.lo;
        return detail::_f128::renorm(s, t);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s mul_sub_inline(const f128_s& a, const f128_s& b, const f128_s& c) noexcept
    {
        double p{}, e{};
        detail::_f128::mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        detail::_f128::two_sum_precise(p, -c.hi, s, t);
        t += e - c.lo;
        return detail::_f128::renorm(s, t);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sub_mul_inline(const f128_s& c, const f128_s& a, const f128_s& b) noexcept
    {
        double p{}, e{};
        detail::_f128::mul_expansion_inline(a, b, p, e);

        double s{}, t{};
        detail::_f128::two_sum_precise(c.hi, -p, s, t);
        t += c.lo - e;
        return detail::_f128::renorm(s, t);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s sum_products_inline(const f128_s& a, const f128_s& b, const f128_s& c, const f128_s& d) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};
        detail::_f128::mul_expansion_inline(a, b, p0, e0);
        detail::_f128::mul_expansion_inline(c, d, p1, e1);

        double s{}, t{};
        detail::_f128::two_sum_precise(p0, p1, s, t);
        t += e0 + e1;
        return detail::_f128::renorm(s, t);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f128_s diff_products_inline(const f128_s& a, const f128_s& b, const f128_s& c, const f128_s& d) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};
        detail::_f128::mul_expansion_inline(a, b, p0, e0);
        detail::_f128::mul_expansion_inline(c, d, p1, e1);

        double s{}, t{};
        detail::_f128::two_sum_precise(p0, -p1, s, t);
        t += e0 - e1;
        return detail::_f128::renorm(s, t);
    }
    BL_POP_PRECISE;
}

/// ------------------ scalar ------------------

BL_PUSH_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator+(const f128_s& a, const f128_s& b) noexcept
{
    // accurate sum of the high parts
    double s{}, e{};
    detail::_f128::two_sum_precise(a.hi, b.hi, s, e);

    // fold low parts into the error
    double t = a.lo + b.lo;
    e += t;

    // renormalize
    return detail::_f128::renorm(s, e);
}
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator-(const f128_s& a, const f128_s& b) noexcept
{
    double s{}, e{};
    detail::_f128::two_sum_precise(a.hi, -b.hi, s, e);

    double t = a.lo - b.lo;
    e += t;

    return detail::_f128::renorm(s, e);
}
BL_POP_PRECISE;
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept
{
    double p, e;
    detail::_f128::two_prod_precise(a.hi, b.hi, p, e);

    e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;
    return detail::_f128::renorm(p, e);
}
[[nodiscard]] FLTX_CORE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept
{
    if (b.lo == 0.0)
        return a / b.hi;

    const double inv_b0 = 1.0 / b.hi;

    const double q0 = a.hi * inv_b0;
    f128_s r = detail::_f128::sub_mul_scalar_exact(a, b, q0);

    const double q1 = r.hi * inv_b0;

    return detail::_f128::renorm(q0, q1);
}

// f128 <=> double
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    detail::_f128::two_sum_precise(a.hi, b, s, e);
    e += a.lo;
    return detail::_f128::renorm(s, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    detail::_f128::two_sum_precise(a.hi, -b, s, e);
    e += a.lo;
    return detail::_f128::renorm(s, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept
{
    double p{}, e{};
    detail::_f128::two_prod_precise(a.hi, b, p, e);
	
	e += a.lo * b;
    return detail::_f128::renorm(p, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
    {
        if (isnan(a) || detail::_f128::isnan(b)) [[unlikely]]
            return std::numeric_limits<f128_s>::quiet_NaN();

        if (detail::_f128::isinf(b))
        {
            if (isinf(a))
                return std::numeric_limits<f128_s>::quiet_NaN();

            const bool neg = detail::_f128::signbit_constexpr(a.hi) ^ detail::_f128::signbit_constexpr(b);
            return f128_s{ neg ? -0.0 : 0.0, 0.0 };
        }

        if (b == 0.0) [[unlikely]]
        {
            if (iszero(a))
                return std::numeric_limits<f128_s>::quiet_NaN();

            const bool neg = detail::_f128::signbit_constexpr(a.hi) ^ detail::_f128::signbit_constexpr(b);
            return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
        }

        if (isinf(a)) [[unlikely]]
        {
            const bool neg = detail::_f128::signbit_constexpr(a.hi) ^ detail::_f128::signbit_constexpr(b);
            return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
        }
    }

    if (!bl::use_constexpr_math())
        return detail::_f128::div_f128_double_runtime(a, b);

    const double q0 = a.hi / b;
    const f128_s r = detail::_f128::sub_mul_scalar_exact(a, f128_s{ b, 0.0 }, q0);
    const double q1 = r.hi / b;

    return detail::_f128::renorm(q0, q1);
}

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(double a, const f128_s& b) noexcept { return b + a; }
[[nodiscard]] BL_NO_INLINE constexpr f128_s operator-(double a, const f128_s& b) noexcept
{
    double s{}, e{};
    detail::_f128::two_sum_precise(a, -b.hi, s, e);
    e -= b.lo;
    return detail::_f128::renorm(s, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(double a, const f128_s& b) noexcept { return b * a; }
[[nodiscard]] BL_NO_INLINE constexpr f128_s operator/(double a, const f128_s& b) noexcept
{
    const double q0 = a / b.hi;
    const f128_s r = detail::_f128::sub_mul_scalar_exact(f128_s{ a, 0.0 }, b, q0);
    const double q1 = r.hi / b.hi;
    return detail::_f128::renorm(q0, q1);
}

// f128 <=> float
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(const f128_s& a, float b) noexcept { return a + (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(const f128_s& a, float b) noexcept { return a - (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(const f128_s& a, float b) noexcept { return a * (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(const f128_s& a, float b) noexcept { return a / (double)b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator+(float a, const f128_s& b) noexcept { return (double)a + b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator-(float a, const f128_s& b) noexcept { return (double)a - b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator*(float a, const f128_s& b) noexcept { return (double)a * b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f128_s operator/(float a, const f128_s& b) noexcept { return (double)a / b; }


} // namespace bl

#endif
