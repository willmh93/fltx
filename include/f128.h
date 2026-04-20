#ifndef F128_INCLUDED
#define F128_INCLUDED
#include "fltx_common_math.h"

#include <numbers>

namespace bl {

struct f128_s;
struct f256_s;
struct f256;

namespace _f128_detail
{
    using fltx::common::fp::absd;
    using fltx::common::fp::isnan;
    using fltx::common::fp::isinf;
    using fltx::common::fp::isfinite;
    using fltx::common::fp::magnitude_u64;
    using fltx::common::fp::split_uint64_to_doubles;
    using fltx::common::fp::two_prod_precise;
    using fltx::common::fp::two_sum_precise;

    using fltx::common::fp::signbit_constexpr;
    using fltx::common::fp::fabs_constexpr;
    using fltx::common::fp::floor_constexpr;
    using fltx::common::fp::ceil_constexpr;
    using fltx::common::fp::double_integer_is_odd;
    using fltx::common::fp::fmod_constexpr;
    using fltx::common::fp::sqrt_seed_constexpr;
    using fltx::common::fp::nearbyint_ties_even;
}

FORCE_INLINE constexpr f128_s operator+(const f128_s& a, const f128_s& b) noexcept;
FORCE_INLINE constexpr f128_s operator-(const f128_s& a, const f128_s& b) noexcept;
FORCE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept;
FORCE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept;

FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept;
FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept;
FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept;
FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept;

FORCE_INLINE constexpr f128_s operator+(const f128_s& a, float b) noexcept;
FORCE_INLINE constexpr f128_s operator-(const f128_s& a, float b) noexcept;
FORCE_INLINE constexpr f128_s operator*(const f128_s& a, float b) noexcept;
FORCE_INLINE constexpr f128_s operator/(const f128_s& a, float b) noexcept;

struct f128_s
{
    double hi, lo;

    FORCE_INLINE constexpr f128_s& operator=(f256_s x) noexcept;
    FORCE_INLINE constexpr f128_s& operator=(double x) noexcept {
        hi = x; lo = 0.0; return *this;
    }
    FORCE_INLINE constexpr f128_s& operator=(float x) noexcept {
        hi = static_cast<double>(x); lo = 0.0; return *this;
    }

    FORCE_INLINE constexpr f128_s& operator=(uint64_t u) noexcept;
    FORCE_INLINE constexpr f128_s& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f128_s& operator=(T v) noexcept {
        return (*this = static_cast<int64_t>(v));
    }
    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f128_s& operator=(T v) noexcept {
        return (*this = static_cast<uint64_t>(v));
    }

    // f128 ops
    FORCE_INLINE constexpr f128_s& operator+=(f128_s rhs) noexcept { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator-=(f128_s rhs) noexcept { *this = *this - rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator*=(f128_s rhs) noexcept { *this = *this * rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator/=(f128_s rhs) noexcept { *this = *this / rhs; return *this; }

    // f64 ops
    FORCE_INLINE constexpr f128_s& operator+=(double rhs) noexcept { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator-=(double rhs) noexcept { *this = *this - rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator*=(double rhs) noexcept { *this = *this * rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator/=(double rhs) noexcept { *this = *this / rhs; return *this; }

    // f32 ops
    FORCE_INLINE constexpr f128_s& operator+=(float rhs) noexcept { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator-=(float rhs) noexcept { *this = *this - rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator*=(float rhs) noexcept { *this = *this * rhs; return *this; }
    FORCE_INLINE constexpr f128_s& operator/=(float rhs) noexcept { *this = *this / rhs; return *this; }

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

namespace _f128_detail
{
    FORCE_INLINE constexpr f128_s renorm(double hi, double lo)
    {
        double s{}, e{};
        _f128_detail::two_sum_precise(hi, lo, s, e);
        return { s, e };
    }
    FORCE_INLINE constexpr f128_s canonicalize_math_result(f128_s value) noexcept
    {
        value.lo = fltx::common::fp::zero_low_fraction_bits_finite<8>(value.lo);
        return value;
    }
}

FORCE_INLINE constexpr f128_s& f128_s::operator=(uint64_t u) noexcept {
    double a{}, b{};
    _f128_detail::split_uint64_to_doubles(u, a, b);
    double s{}, e{}; _f128_detail::two_sum_precise(a, b, s, e);
    f128_s r = _f128_detail::renorm(s, e);
    hi = r.hi; lo = r.lo; return *this;
}
FORCE_INLINE constexpr f128_s& f128_s::operator=(int64_t v) noexcept {
    uint64_t u = _f128_detail::magnitude_u64(v);
    f128_s r{}; r = u;                       // reuse uint64_t path
    if (v < 0) { r.hi = -r.hi; r.lo = -r.lo; }
    hi = r.hi; lo = r.lo; return *this;
}

[[nodiscard]] constexpr f128_s to_f128(double x) noexcept { return f128_s{ x, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(float x) noexcept { return f128_s{ (double)x, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(int32_t v) noexcept { return f128_s{ (double)v, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(uint32_t v) noexcept { return f128_s{ (double)v, 0.0 }; }
[[nodiscard]] constexpr f128_s to_f128(int64_t v) noexcept {
    uint64_t u = _f128_detail::magnitude_u64(v);
    f128_s r{}; r = u; // reuse uint64_t path
    if (v < 0) { r.hi = -r.hi; r.lo = -r.lo; }
    return r;
}
[[nodiscard]] constexpr f128_s to_f128(uint64_t u)  noexcept {
    double a{}, b{};
    _f128_detail::split_uint64_to_doubles(u, a, b);
    double s{}, e{}; _f128_detail::two_sum_precise(a, b, s, e);
    return _f128_detail::renorm(s, e);
}

/// ======== Comparisons ========

namespace _f128_detail
{
    FORCE_INLINE constexpr void uint64_compare_pair(uint64_t value, double& hi, double& lo) noexcept
    {
        double a{}, b{};
        _f128_detail::split_uint64_to_doubles(value, a, b);
        _f128_detail::two_sum_precise(a, b, hi, lo);
    }

    FORCE_INLINE constexpr void int64_compare_pair(int64_t value, double& hi, double& lo) noexcept
    {
        _f128_detail::uint64_compare_pair(_f128_detail::magnitude_u64(value), hi, lo);
        if (value < 0) { hi = -hi; lo = -lo; }
    }
}
// ------------------ f128 <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, const f128_s& b)  { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, const f128_s& b)  { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, const f128_s& b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, const f128_s& b) { return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, const f128_s& b) { return a.hi == b.hi && a.lo == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, const f128_s& b) { return a.hi != b.hi || a.lo != b.lo; }

// ------------------ double <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, double b)  { return (a.hi < b) || (a.hi == b && a.lo <  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(double a, const f128_s& b)  { return (a < b.hi) || (a == b.hi && 0.0 <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, double b)  { return (a.hi > b) || (a.hi == b && a.lo >  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(double a, const f128_s& b)  { return (a > b.hi) || (a == b.hi && 0.0 >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, double b) { return (a.hi < b) || (a.hi == b && a.lo <= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(double a, const f128_s& b) { return (a < b.hi) || (a == b.hi && 0.0 <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, double b) { return (a.hi > b) || (a.hi == b && a.lo >= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(double a, const f128_s& b) { return (a > b.hi) || (a == b.hi && 0.0 >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, double b) { return a.hi == b && a.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(double a, const f128_s& b) { return a == b.hi && 0.0 == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, double b) { return a.hi != b || a.lo != 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(double a, const f128_s& b) { return a != b.hi || 0.0 != b.lo; }

// ------------------ float <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, float b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(float a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, float b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(float a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, float b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(float a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, float b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(float a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, float b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(float a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, float b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(float a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }

// --------------- ints <=> f128 ---------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, int32_t b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(int32_t a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, int32_t b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(int32_t a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, int32_t b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(int32_t a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, int32_t b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(int32_t a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, int32_t b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(int32_t a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, int32_t b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(int32_t a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, uint32_t b)  { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(uint32_t a, const f128_s& b)  { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, uint32_t b)  { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >  0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(uint32_t a, const f128_s& b)  { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint32_t b) { const double bd = (double)b; return (a.hi < bd) || (a.hi == bd && a.lo <= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(uint32_t a, const f128_s& b) { const double ad = (double)a; return (ad < b.hi) || (ad == b.hi && 0.0 <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint32_t b) { const double bd = (double)b; return (a.hi > bd) || (a.hi == bd && a.lo >= 0.0); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(uint32_t a, const f128_s& b) { const double ad = (double)a; return (ad > b.hi) || (ad == b.hi && 0.0 >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, uint32_t b) { const double bd = (double)b; return a.hi == bd && a.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(uint32_t a, const f128_s& b) { const double ad = (double)a; return ad == b.hi && 0.0 == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint32_t b) { const double bd = (double)b; return a.hi != bd || a.lo != 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(uint32_t a, const f128_s& b) { const double ad = (double)a; return ad != b.hi || 0.0 != b.lo; }

// ------------------ int64_t/uint64_t <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, int64_t b)  { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(int64_t a, const f128_s& b)  { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, int64_t b)  { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(int64_t a, const f128_s& b)  { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, int64_t b) { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(int64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, int64_t b) { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(int64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, int64_t b) { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(int64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, int64_t b) { double bhi{}, blo{}; _f128_detail::int64_compare_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(int64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::int64_compare_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <  blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, uint64_t b)  { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >  blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(uint64_t a, const f128_s& b)  { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >  b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return (a.hi < bhi) || (a.hi == bhi && a.lo <= blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return (ahi < b.hi) || (ahi == b.hi && alo <= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return (a.hi > bhi) || (a.hi == bhi && a.lo >= blo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return (ahi > b.hi) || (ahi == b.hi && alo >= b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, uint64_t b) { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return a.hi == bhi && a.lo == blo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(uint64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return ahi == b.hi && alo == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint64_t b) { double bhi{}, blo{}; _f128_detail::uint64_compare_pair(b, bhi, blo); return a.hi != bhi || a.lo != blo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(uint64_t a, const f128_s& b) { double ahi{}, alo{}; _f128_detail::uint64_compare_pair(a, ahi, alo); return ahi != b.hi || alo != b.lo; }

// ------------------ classification ------------------

NO_INLINE    constexpr f128_s pow10_128(int k);
FORCE_INLINE constexpr f128_s abs(const f128_s& a);

[[nodiscard]] FORCE_INLINE constexpr bool isnan(const f128_s& x) noexcept      { return fltx::common::fp::isnan(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool isinf(const f128_s& x) noexcept      { return fltx::common::fp::isinf(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool isfinite(const f128_s& x) noexcept   { return fltx::common::fp::isfinite(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool iszero(const f128_s& x) noexcept     { return x.hi == 0.0 && x.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool ispositive(const f128_s& x) noexcept { return x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0); }

[[nodiscard]] FORCE_INLINE constexpr bool signbit(const f128_s& x) noexcept
{
    return _f128_detail::signbit_constexpr(x.hi) || (x.hi == 0.0 && _f128_detail::signbit_constexpr(x.lo));
}
[[nodiscard]] FORCE_INLINE constexpr int  fpclassify(const f128_s& x) noexcept
{
    if (isnan(x))  return FP_NAN;
    if (isinf(x))  return FP_INFINITE;
    if (iszero(x)) return FP_ZERO;
    return abs(x) < std::numeric_limits<f128_s>::min() ? FP_SUBNORMAL : FP_NORMAL;
}
[[nodiscard]] FORCE_INLINE constexpr bool isnormal(const f128_s& x) noexcept
{
    return fpclassify(x) == FP_NORMAL;
}
[[nodiscard]] FORCE_INLINE constexpr bool isunordered(const f128_s& a, const f128_s& b) noexcept
{
    return isnan(a) || isnan(b);
}
[[nodiscard]] FORCE_INLINE constexpr bool isgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a > b;
}
[[nodiscard]] FORCE_INLINE constexpr bool isgreaterequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a >= b;
}
[[nodiscard]] FORCE_INLINE constexpr bool isless(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a < b;
}
[[nodiscard]] FORCE_INLINE constexpr bool islessequal(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a <= b;
}
[[nodiscard]] FORCE_INLINE constexpr bool islessgreater(const f128_s& a, const f128_s& b) noexcept
{
    return !isunordered(a, b) && a != b;
}

/// ------------------ arithmetic operators ------------------

[[nodiscard]] FORCE_INLINE constexpr f128_s clamp(const f128_s& v, const f128_s& lo, const f128_s& hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
[[nodiscard]] FORCE_INLINE constexpr f128_s abs(const f128_s& a)
{
    return (a.hi < 0.0) ? -a : a;
}
[[nodiscard]] NO_INLINE constexpr f128_s floor(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    constexpr double integer_hi_threshold = 4503599627370496.0;

    if (_f128_detail::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return _f128_detail::renorm(a.hi, _f128_detail::floor_constexpr(a.lo));
    }

    f128_s r{ _f128_detail::floor_constexpr(a.hi), 0.0 };
    if (r > a)
        r -= 1.0;
    return r;
}
[[nodiscard]] NO_INLINE constexpr f128_s ceil(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    constexpr double integer_hi_threshold = 4503599627370496.0;

    if (_f128_detail::absd(a.hi) >= integer_hi_threshold)
    {
        if (a.lo == 0.0)
            return f128_s{ a.hi, 0.0 };

        return _f128_detail::renorm(a.hi, _f128_detail::ceil_constexpr(a.lo));
    }

    f128_s r{ _f128_detail::ceil_constexpr(a.hi), 0.0 };
    if (r < a)
        r += 1.0;
    return r;
}
[[nodiscard]] NO_INLINE constexpr f128_s trunc(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    return (a.hi < 0.0) ? ceil(a) : floor(a);
}

[[nodiscard]] FORCE_INLINE constexpr f128_s recip(f128_s b)
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

namespace _f128_detail
{
    BL_PUSH_PRECISE
    FORCE_INLINE constexpr f128_s quick_two_sum(double a, double b)
    {
        double s = a + b;
        double err = b - (s - a);
        return { s, err };
    }
    BL_POP_PRECISE
}

/// ------------------ scalar ------------------

BL_PUSH_PRECISE
[[nodiscard]] FORCE_INLINE constexpr        f128_s operator+(const f128_s& a, const f128_s& b) noexcept
{
    // accurate sum of the high parts
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, b.hi, s, e);

    // fold low parts into the error
    double t = a.lo + b.lo;
    e += t;

    // renormalize
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] FORCE_INLINE constexpr        f128_s operator-(const f128_s& a, const f128_s& b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, -b.hi, s, e);

    double t = a.lo - b.lo;
    e += t;

    return _f128_detail::renorm(s, e);
}
BL_POP_PRECISE

[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(const f128_s& a, const f128_s& b) noexcept
{
    double p, e;
    _f128_detail::two_prod_precise(a.hi, b.hi, p, e);

	e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;
    return _f128_detail::renorm(p, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator/(const f128_s& a, const f128_s& b) noexcept
{
    if (b.lo == 0.0)
        return a / b.hi;

    const double inv_b0 = 1.0 / b.hi;

    const double q0 = a.hi * inv_b0;
    f128_s r = a - b * q0;

    const double q1 = r.hi * inv_b0;

    return _f128_detail::renorm(q0, q1);
}

// f128 <=> double
[[nodiscard]] NO_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, b, s, e);
    e += a.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] NO_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, -b, s, e);
    e += a.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] NO_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept
{
    double p{}, e{};
    _f128_detail::two_prod_precise(a.hi, b, p, e);
	
	e += a.lo * b;
    return _f128_detail::renorm(p, e);
}
[[nodiscard]] NO_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept
{
    if (bl::is_constant_evaluated())
    {
        if (isnan(a) || _f128_detail::isnan(b))
            return std::numeric_limits<f128_s>::quiet_NaN();

        if (_f128_detail::isinf(b))
        {
            if (isinf(a))
                return std::numeric_limits<f128_s>::quiet_NaN();

            const bool neg = _f128_detail::signbit_constexpr(a.hi) ^ _f128_detail::signbit_constexpr(b);
            return f128_s{ neg ? -0.0 : 0.0, 0.0 };
        }

        if (b == 0.0)
        {
            if (iszero(a))
                return std::numeric_limits<f128_s>::quiet_NaN();

            const bool neg = _f128_detail::signbit_constexpr(a.hi) ^ _f128_detail::signbit_constexpr(b);
            return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
        }

        if (isinf(a))
        {
            const bool neg = _f128_detail::signbit_constexpr(a.hi) ^ _f128_detail::signbit_constexpr(b);
            return f128_s{ neg ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity(), 0.0 };
        }
    }

    const double q0 = a.hi / b;
    const f128_s r = a - f128_s{ b, 0.0 } * q0;
    const double q1 = r.hi / b;

    return _f128_detail::renorm(q0, q1);
}

[[nodiscard]] FORCE_INLINE constexpr f128_s operator+(double a, const f128_s& b) noexcept { return b + a; }
[[nodiscard]] NO_INLINE constexpr f128_s operator-(double a, const f128_s& b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a, -b.hi, s, e);
    e -= b.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(double a, const f128_s& b) noexcept { return b * a; }
[[nodiscard]] NO_INLINE constexpr f128_s operator/(double a, const f128_s& b) noexcept
{
    const double q0 = a / b.hi;
    const f128_s r = f128_s{ a, 0.0 } - b * q0;
    const double q1 = r.hi / b.hi;
    return _f128_detail::renorm(q0, q1);
}

// f128 <=> float
[[nodiscard]] FORCE_INLINE constexpr f128_s operator+(const f128_s& a, float b) noexcept { return a + (double)b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator-(const f128_s& a, float b) noexcept { return a - (double)b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(const f128_s& a, float b) noexcept { return a * (double)b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator/(const f128_s& a, float b) noexcept { return a / (double)b; }

[[nodiscard]] FORCE_INLINE constexpr f128_s operator+(float a, const f128_s& b) noexcept { return (double)a + b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator-(float a, const f128_s& b) noexcept { return (double)a - b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(float a, const f128_s& b) noexcept { return (double)a * b; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator/(float a, const f128_s& b) noexcept { return (double)a / b; }


} // namespace bl

#endif