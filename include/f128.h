#pragma once
#define F128_INCLUDED
#include "fltx_common.h"

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
    using fltx::common::fp::two_prod_precise_dekker;
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
    static constexpr bl::f128_s highest()        noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon() }; }

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

template<>
struct std::numeric_limits<bl::f128>
{
    using base = std::numeric_limits<bl::f128_s>;

    static constexpr bool is_specialized = base::is_specialized;

    static constexpr bl::f128 min() noexcept             { return bl::f128{ base::min() }; }
    static constexpr bl::f128 max() noexcept             { return bl::f128{ base::max() }; }
    static constexpr bl::f128 lowest() noexcept          { return bl::f128{ base::lowest() }; }
    static constexpr bl::f128 highest() noexcept         { return bl::f128{ base::highest() }; }

    static constexpr bl::f128 epsilon() noexcept         { return bl::f128{ base::epsilon() }; }
    static constexpr bl::f128 round_error() noexcept     { return bl::f128{ base::round_error() }; }
    static constexpr bl::f128 infinity() noexcept        { return bl::f128{ base::infinity() }; }
    static constexpr bl::f128 quiet_NaN() noexcept       { return bl::f128{ base::quiet_NaN() }; }
    static constexpr bl::f128 signaling_NaN() noexcept   { return bl::f128{ base::signaling_NaN() }; }
    static constexpr bl::f128 denorm_min() noexcept      { return bl::f128{ base::denorm_min() }; }

    static constexpr bool has_infinity       = base::has_infinity;
    static constexpr bool has_quiet_NaN      = base::has_quiet_NaN;
    static constexpr bool has_signaling_NaN  = base::has_signaling_NaN;
                                             
    static constexpr int digits              = base::digits;
    static constexpr int digits10            = base::digits10;
    static constexpr int max_digits10        = base::max_digits10;
    static constexpr bool is_signed          = base::is_signed;
    static constexpr bool is_integer         = base::is_integer;
    static constexpr bool is_exact           = base::is_exact;
    static constexpr int radix               = base::radix;
                                             
    static constexpr int min_exponent        = base::min_exponent;
    static constexpr int max_exponent        = base::max_exponent;
    static constexpr int min_exponent10      = base::min_exponent10;
    static constexpr int max_exponent10      = base::max_exponent10;
                                             
    static constexpr bool is_iec559          = base::is_iec559;
    static constexpr bool is_bounded         = base::is_bounded;
    static constexpr bool is_modulo          = base::is_modulo;
                                             
    static constexpr bool traps              = base::traps;
    static constexpr bool tinyness_before    = base::tinyness_before;
    static constexpr std::float_round_style round_style = base::round_style;
};

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
    // same limb path you already use in f128(u)
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

// ------------------ f128 <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator <(const f128_s& a, const f128_s& b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator >(const f128_s& a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, const f128_s& b) { return (a < b) || (a.hi == b.hi && a.lo == b.lo); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, const f128_s& b) { return b <= a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, const f128_s& b) { return a.hi == b.hi && a.lo == b.lo; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, const f128_s& b) { return !(a == b); }

// ------------------ double <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, double b)  { return a < f128_s{b}; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(double a, const f128_s& b)  { return f128_s{a} < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, double b)  { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(double a, const f128_s& b)  { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, double b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(double a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, double b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(double a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, double b) { return a == f128_s{b}; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(double a, const f128_s& b) { return f128_s{a} == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, double b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(double a, const f128_s& b) { return !(a == b); }

// ------------------ float <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, float b) { return a < f128_s{b}; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(float a, const f128_s& b) { return f128_s{a} < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, float b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(float a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, float b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(float a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, float b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(float a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, float b) { return a == f128_s{b}; }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(float a, const f128_s& b) { return f128_s{a} == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, float b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(float a, const f128_s& b) { return !(a == b); }

// --------------- ints <=> f128 ---------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, int32_t b) { return a < to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(int32_t a, const f128_s& b) { return to_f128(a) < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, int32_t b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(int32_t a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, int32_t b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(int32_t a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, int32_t b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(int32_t a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, int32_t b) { return a == to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(int32_t a, const f128_s& b) { return to_f128(a) == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, int32_t b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(int32_t a, const f128_s& b) { return !(a == b); }

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, uint32_t b) { return a < to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(uint32_t a, const f128_s& b) { return to_f128(a) < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, uint32_t b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(uint32_t a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint32_t b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(uint32_t a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint32_t b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(uint32_t a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, uint32_t b) { return a == to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(uint32_t a, const f128_s& b) { return to_f128(a) == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint32_t b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(uint32_t a, const f128_s& b) { return !(a == b); }

// ------------------ int64_t/uint64_t <=> f128 ------------------

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, int64_t b) { return a < to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(int64_t a, const f128_s& b) { return to_f128(a) < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, int64_t b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(int64_t a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, int64_t b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(int64_t a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, int64_t b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(int64_t a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, int64_t b) { return a == to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(int64_t a, const f128_s& b) { return to_f128(a) == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, int64_t b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(int64_t a, const f128_s& b) { return !(a == b); }

[[nodiscard]] FORCE_INLINE constexpr bool operator<(const f128_s& a, uint64_t b) { return a < to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<(uint64_t a, const f128_s& b) { return to_f128(a) < b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(const f128_s& a, uint64_t b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator>(uint64_t a, const f128_s& b) { return b < a; }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(const f128_s& a, uint64_t b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator<=(uint64_t a, const f128_s& b) { return !(b < a); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(const f128_s& a, uint64_t b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator>=(uint64_t a, const f128_s& b) { return !(a < b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(const f128_s& a, uint64_t b) { return a == to_f128(b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator==(uint64_t a, const f128_s& b) { return to_f128(a) == b; }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(const f128_s& a, uint64_t b) { return !(a == b); }
[[nodiscard]] FORCE_INLINE constexpr bool operator!=(uint64_t a, const f128_s& b) { return !(a == b); }

// ------------------ classification ------------------

[[nodiscard]] FORCE_INLINE constexpr bool isnan(const f128_s& x) noexcept      { return fltx::common::fp::isnan(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool isinf(const f128_s& x) noexcept      { return fltx::common::fp::isinf(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool isfinite(const f128_s& x) noexcept   { return fltx::common::fp::isfinite(x.hi); }
[[nodiscard]] FORCE_INLINE constexpr bool iszero(const f128_s& x) noexcept     { return x.hi == 0.0 && x.lo == 0.0; }
[[nodiscard]] FORCE_INLINE constexpr bool ispositive(const f128_s& x) noexcept { return x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0); }

/// ------------------ arithmetic operators ------------------

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
BL_POP_PRECISE
[[nodiscard]] FORCE_INLINE constexpr        f128_s operator-(const f128_s& a, const f128_s& b) noexcept
{
    return a + f128_s{ -b.hi, -b.lo };
}

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
[[nodiscard]] FORCE_INLINE constexpr f128_s operator+(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, b, s, e);
    e += a.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator-(const f128_s& a, double b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a.hi, -b, s, e);
    e += a.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(const f128_s& a, double b) noexcept
{
    double p{}, e{};
    _f128_detail::two_prod_precise(a.hi, b, p, e);
	
	e += a.lo * b;
    return _f128_detail::renorm(p, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator/(const f128_s& a, double b) noexcept
{
    if (std::is_constant_evaluated())
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
[[nodiscard]] FORCE_INLINE constexpr f128_s operator-(double a, const f128_s& b) noexcept
{
    double s{}, e{};
    _f128_detail::two_sum_precise(a, -b.hi, s, e);
    e -= b.lo;
    return _f128_detail::renorm(s, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s operator*(double a, const f128_s& b) noexcept { return b * a; }
[[nodiscard]] FORCE_INLINE constexpr f128_s operator/(double a, const f128_s& b) noexcept
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

/// ------------------ math ------------------

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

namespace _f128_detail
{
    FORCE_INLINE constexpr bool f128_try_get_int64(const f128_s& x, int64_t& out)
    {
        const f128_s xi = trunc(x);
        if (xi != x)
            return false;

        if (_f128_detail::absd(xi.hi) >= 0x1p63)
            return false;

        const int64_t hi_part = static_cast<int64_t>(xi.hi);
        const f128_s rem = xi - to_f128(hi_part);
        out = hi_part + static_cast<int64_t>(rem.hi + rem.lo);
        return true;
    }
    FORCE_INLINE constexpr f128_s powi(f128_s base, int64_t exp)
    {
        if (exp == 0)
            return f128_s{ 1.0 };

        const bool invert = exp < 0;
        uint64_t n = invert ? _f128_detail::magnitude_u64(exp) : static_cast<uint64_t>(exp);
        f128_s result{ 1.0 };

        while (n != 0)
        {
            if ((n & 1u) != 0)
                result *= base;

            n >>= 1;
            if (n != 0)
                base *= base;
        }

        return invert ? (f128_s{ 1.0 } / result) : result;
    }
    FORCE_INLINE constexpr bool f128_try_exact_binary_log2(const f128_s& x, int& out) noexcept
    {
        if (!(x.hi > 0.0) || x.lo != 0.0)
            return false;

        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x.hi);
        const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
        const std::uint64_t frac_bits = bits & ((std::uint64_t{ 1 } << 52) - 1);

        if (exp_bits == 0 || frac_bits != 0)
            return false;

        out = static_cast<int>(exp_bits) - 1023;
        return true;
    }

    struct fmod_u128
    {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
    };

    struct exact_dyadic_fmod
    {
        bool neg = false;
        int exp2 = 0;
        fmod_u128 mant{};
    };

    FORCE_INLINE constexpr bool fmod_u128_is_zero(const fmod_u128& value)
    {
        return value.lo == 0 && value.hi == 0;
    }
    FORCE_INLINE constexpr bool fmod_u128_is_odd(const fmod_u128& value)
    {
        return (value.lo & 1u) != 0;
    }
    FORCE_INLINE constexpr int fmod_u128_compare(const fmod_u128& a, const fmod_u128& b)
    {
        if (a.hi < b.hi) return -1;
        if (a.hi > b.hi) return 1;
        if (a.lo < b.lo) return -1;
        if (a.lo > b.lo) return 1;
        return 0;
    }
    FORCE_INLINE constexpr int fmod_u128_bit_length(const fmod_u128& value)
    {
        if (value.hi != 0)
            return 128 - static_cast<int>(std::countl_zero(value.hi));
        if (value.lo != 0)
            return 64 - static_cast<int>(std::countl_zero(value.lo));
        return 0;
    }
    FORCE_INLINE constexpr int fmod_u128_trailing_zero_bits(const fmod_u128& value)
    {
        if (value.lo != 0)
            return static_cast<int>(std::countr_zero(value.lo));
        if (value.hi != 0)
            return 64 + static_cast<int>(std::countr_zero(value.hi));
        return 0;
    }
    FORCE_INLINE constexpr bool fmod_u128_get_bit(const fmod_u128& value, int index)
    {
        if (index < 0 || index >= 128)
            return false;
        if (index < 64)
            return ((value.lo >> index) & 1u) != 0;
        return ((value.hi >> (index - 64)) & 1u) != 0;
    }
    FORCE_INLINE constexpr std::uint64_t fmod_u128_get_bits(const fmod_u128& value, int start, int count)
    {
        std::uint64_t out = 0;
        for (int i = 0; i < count; ++i)
        {
            if (fmod_u128_get_bit(value, start + i))
                out |= (std::uint64_t{ 1 } << i);
        }
        return out;
    }
    FORCE_INLINE constexpr bool fmod_u128_any_low_bits_set(const fmod_u128& value, int count)
    {
        if (count <= 0)
            return false;

        if (count >= 64)
        {
            if (value.lo != 0)
                return true;
            count -= 64;
            if (count >= 64)
                return value.hi != 0;
            return (value.hi & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
        }

        return (value.lo & ((std::uint64_t{ 1 } << count) - 1u)) != 0;
    }
    FORCE_INLINE constexpr void fmod_u128_add_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += b.lo;
        a.hi += b.hi + (a.lo < old_lo ? 1u : 0u);
    }
    FORCE_INLINE constexpr void fmod_u128_add_small(fmod_u128& a, std::uint32_t value)
    {
        const std::uint64_t old_lo = a.lo;
        a.lo += value;
        if (a.lo < old_lo)
            ++a.hi;
    }
    FORCE_INLINE constexpr void fmod_u128_sub_inplace(fmod_u128& a, const fmod_u128& b)
    {
        const std::uint64_t borrow = (a.lo < b.lo) ? 1u : 0u;
        a.lo -= b.lo;
        a.hi -= b.hi + borrow;
    }
    FORCE_INLINE constexpr fmod_u128 fmod_u128_shl_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { 0, value.lo << (bits - 64) };

        return {
            value.lo << bits,
            (value.hi << bits) | (value.lo >> (64 - bits))
        };
    }
    FORCE_INLINE constexpr fmod_u128 fmod_u128_shr_bits(fmod_u128 value, int bits)
    {
        if (bits <= 0 || fmod_u128_is_zero(value))
            return value;
        if (bits >= 128)
            return {};
        if (bits >= 64)
            return { value.hi >> (bits - 64), 0 };

        return {
            (value.lo >> bits) | (value.hi << (64 - bits)),
            value.hi >> bits
        };
    }
    FORCE_INLINE constexpr fmod_u128 fmod_u128_shl1(fmod_u128 value)
    {
        return { value.lo << 1, (value.hi << 1) | (value.lo >> 63) };
    }
    FORCE_INLINE constexpr fmod_u128 fmod_u128_mod_shift_subtract(fmod_u128 numerator, const fmod_u128& denominator)
    {
        if (fmod_u128_is_zero(denominator))
            return {};
        if (fmod_u128_compare(numerator, denominator) < 0)
            return numerator;

        int shift = fmod_u128_bit_length(numerator) - fmod_u128_bit_length(denominator);
        fmod_u128 shifted = fmod_u128_shl_bits(denominator, shift);

        for (; shift >= 0; --shift)
        {
            if (fmod_u128_compare(numerator, shifted) >= 0)
                fmod_u128_sub_inplace(numerator, shifted);
            shifted = fmod_u128_shr_bits(shifted, 1);
        }

        return numerator;
    }
    FORCE_INLINE constexpr fmod_u128 fmod_u128_double_mod(fmod_u128 value, const fmod_u128& modulus)
    {
        value = fmod_u128_shl1(value);
        if (fmod_u128_compare(value, modulus) >= 0)
            fmod_u128_sub_inplace(value, modulus);
        return value;
    }
    FORCE_INLINE constexpr exact_dyadic_fmod exact_from_double_fmod(double value)
    {
        exact_dyadic_fmod out;
        if (value == 0.0)
            return out;

        int exponent = 0;
        bool neg = false;
        const std::uint64_t mantissa = fltx::common::exact_decimal::decompose_double_mantissa(value, exponent, neg);
        if (mantissa == 0)
            return out;

        out.neg = neg;
        out.exp2 = exponent;
        out.mant.lo = mantissa;
        return out;
    }
    FORCE_INLINE constexpr void normalize_exact_dyadic_fmod(exact_dyadic_fmod& value)
    {
        if (fmod_u128_is_zero(value.mant))
        {
            value.neg = false;
            value.exp2 = 0;
            return;
        }

        const int tz = fmod_u128_trailing_zero_bits(value.mant);
        if (tz != 0)
        {
            value.mant = fmod_u128_shr_bits(value.mant, tz);
            value.exp2 += tz;
        }
    }
    FORCE_INLINE constexpr exact_dyadic_fmod exact_from_f128_fmod(const f128_s& value)
    {
        exact_dyadic_fmod hi = exact_from_double_fmod(value.hi);
        exact_dyadic_fmod lo = exact_from_double_fmod(value.lo);

        if (fmod_u128_is_zero(hi.mant))
            return lo;
        if (fmod_u128_is_zero(lo.mant))
            return hi;

        const int common_exp = std::min(hi.exp2, lo.exp2);
        const fmod_u128 hi_scaled = fmod_u128_shl_bits(hi.mant, hi.exp2 - common_exp);
        const fmod_u128 lo_scaled = fmod_u128_shl_bits(lo.mant, lo.exp2 - common_exp);

        exact_dyadic_fmod out;
        out.exp2 = common_exp;

        if (hi.neg == lo.neg)
        {
            out.neg = hi.neg;
            out.mant = hi_scaled;
            fmod_u128_add_inplace(out.mant, lo_scaled);
        }
        else
        {
            const int cmp = fmod_u128_compare(hi_scaled, lo_scaled);
            if (cmp >= 0)
            {
                out.neg = hi.neg;
                out.mant = hi_scaled;
                fmod_u128_sub_inplace(out.mant, lo_scaled);
            }
            else
            {
                out.neg = lo.neg;
                out.mant = lo_scaled;
                fmod_u128_sub_inplace(out.mant, hi_scaled);
            }
        }

        normalize_exact_dyadic_fmod(out);
        return out;
    }
    FORCE_INLINE constexpr bool fmod_fast_double_divisor_abs(const f128_s& ax, double ay, f128_s& out)
    {
        if (!(ay > 0.0) || !_f128_detail::isfinite(ay))
            return false;

        const f128_s mod{ ay, 0.0 };

        if (ax.lo == 0.0)
        {
            out = f128_s{ _f128_detail::fmod_constexpr(ax.hi, ay), 0.0 };
            return true;
        }

        const double rh = (ax.hi < ay) ? ax.hi : _f128_detail::fmod_constexpr(ax.hi, ay);
        const double rl = (_f128_detail::absd(ax.lo) < ay) ? ax.lo : _f128_detail::fmod_constexpr(ax.lo, ay);

        f128_s r = f128_s{ rh, 0.0 } + f128_s{ rl, 0.0 };

        if (r < 0.0)
            r += mod;
        if (r >= mod)
            r -= mod;

        if (r < 0.0)
            r += mod;
        if (r >= mod)
            r -= mod;

        if (r < 0.0 || r >= mod)
            return false;

        out = r;
        return true;
    }
    FORCE_INLINE constexpr f128_s exact_dyadic_to_f128_fmod(const fmod_u128& coeff, int exp2, bool neg)
    {
        if (fmod_u128_is_zero(coeff))
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        int ratio_exp = fmod_u128_bit_length(coeff) - 1;
        fmod_u128 q = coeff;

        if (ratio_exp > 105)
        {
            const int right_shift = ratio_exp - 105;
            const bool round_bit = fmod_u128_get_bit(q, right_shift - 1);
            const bool sticky = fmod_u128_any_low_bits_set(q, right_shift - 1);

            q = fmod_u128_shr_bits(q, right_shift);

            if (round_bit && (sticky || fmod_u128_is_odd(q)))
                fmod_u128_add_small(q, 1u);

            if (fmod_u128_bit_length(q) > 106)
            {
                q = fmod_u128_shr_bits(q, 1);
                ++ratio_exp;
            }
        }
        else if (ratio_exp < 105)
        {
            q = fmod_u128_shl_bits(q, 105 - ratio_exp);
        }

        const int e2 = exp2 + ratio_exp;
        if (e2 > 1023)
            return neg ? -std::numeric_limits<f128_s>::infinity() : std::numeric_limits<f128_s>::infinity();
        if (e2 < -1074)
            return neg ? f128_s{ -0.0, 0.0 } : f128_s{ 0.0, 0.0 };

        const std::uint64_t c1 = fmod_u128_get_bits(q, 0, 53);
        const std::uint64_t c0 = fmod_u128_get_bits(q, 53, 53);
        const double hi = c0 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
        const double lo = c1 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;

        f128_s out = _f128_detail::renorm(hi, lo);
        return neg ? -out : out;
    }
    FORCE_INLINE constexpr f128_s fmod_exact_fixed_limb(const f128_s& x, const f128_s& y)
    {
        const exact_dyadic_fmod dx = exact_from_f128_fmod(abs(x));
        const exact_dyadic_fmod dy = exact_from_f128_fmod(abs(y));

        fmod_u128 remainder{};
        int out_exp = 0;

        if (dx.exp2 < dy.exp2)
        {
            const int shift = dy.exp2 - dx.exp2;
            const fmod_u128 denominator = fmod_u128_shl_bits(dy.mant, shift);
            remainder = fmod_u128_mod_shift_subtract(dx.mant, denominator);
            out_exp = dx.exp2;
        }
        else
        {
            remainder = fmod_u128_mod_shift_subtract(dx.mant, dy.mant);
            const int shift = dx.exp2 - dy.exp2;
            for (int i = 0; i < shift && !fmod_u128_is_zero(remainder); ++i)
                remainder = fmod_u128_double_mod(remainder, dy.mant);
            out_exp = dy.exp2;
        }

        f128_s out = exact_dyadic_to_f128_fmod(remainder, out_exp, !ispositive(x));
        if (iszero(out))
            return f128_s{ _f128_detail::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return out;
    }
}

[[nodiscard]] inline NO_INLINE constexpr f128_s fmod(const f128_s& x, const f128_s& y)
{
    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    const f128_s ay = abs(y);

    if (ax < ay)
        return x;

    f128_s fast{};
    if (y.lo == 0.0 && _f128_detail::fmod_fast_double_divisor_abs(ax, ay.hi, fast))
    {
        if (iszero(fast))
            return f128_s{ _f128_detail::signbit_constexpr(x.hi) ? -0.0 : 0.0 };
        return ispositive(x) ? fast : -fast;
    }

    return _f128_detail::fmod_exact_fixed_limb(x, y);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s round(const f128_s& a)
{
    f128_s t = floor(a + f128_s{ 0.5 });
    if ((t - a) == f128_s{ 0.5 } && fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t -= f128_s{ 1.0 };
    return t;
}
[[nodiscard]] inline NO_INLINE f128_s round_to_decimals(f128_s v, int prec)
{
    if (prec <= 0) return v;

    static constexpr f128_s INV10_DD{
        0.1000000000000000055511151231257827021181583404541015625,  // hi (double rounded)
       -0.0000000000000000055511151231257827021181583404541015625   // lo = 0.1 - hi
    };

    // Sign
    const bool neg = v < 0.0;
    if (neg) v = -v;

    // Split
    f128_s ip = floor(v);
    f128_s frac = v - ip;

    // Extract digits with one look-ahead
    std::string dig; dig.reserve((size_t)prec);
    f128_s w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;
        int di = (int)floor(w).hi;
        if (di < 0) di = 0; else if (di > 9) di = 9;
        dig.push_back(char('0' + di));
        w = w - f128_s{ (double)di };
    }

    // Look-ahead digit
    f128_s la = w * 10.0;
    int next = (int)floor(la).hi;
    if (next < 0) next = 0; else if (next > 9) next = 9;
    f128_s rem = la - f128_s{ (double)next };

    // ties-to-even on last printed digit
    const int last = dig.empty() ? 0 : (dig.back() - '0');
    const bool round_up =
        (next > 5) ||
        (next == 5 && (rem.hi > 0.0 || rem.lo > 0.0 || (last & 1)));

    if (round_up) {
        // propagate carry over fractional digits; if overflow, bump integer part
        int i = prec - 1;
        for (; i >= 0; --i) {
            if (dig[(size_t)i] == '9') dig[(size_t)i] = '0';
            else { ++dig[(size_t)i]; break; }
        }
        if (i < 0) ip = ip + 1.0;
    }

    // Rebuild fractional value backward
    f128_s frac_val{ 0.0, 0.0 };
    for (int i = prec - 1; i >= 0; --i) {
        frac_val = frac_val + f128_s{ (double)(dig[(size_t)i] - '0') };
        frac_val = frac_val * INV10_DD;
    }

    f128_s out = ip + frac_val;
    return neg ? -out : out;
}

[[nodiscard]] NO_INLINE constexpr f128_s remainder(const f128_s& x, const f128_s& y)
{
    // Domain checks (match std::remainder)
    if (isnan(x) || isnan(y)) return std::numeric_limits<f128_s>::quiet_NaN();
    if (iszero(y))            return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(x))             return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))             return x;

    // n = nearest integer to q = x/y, ties to even
    const f128_s q = x / y;
    f128_s n = trunc(q);
    f128_s rfrac = q - n;                // fractional part with sign of q
    const f128_s half = f128_s{0.5};
    const f128_s one{ 1 };

    if (abs(rfrac) > half) {
        n += (rfrac.hi >= 0.0 ? one : -one);
    }
    else if (abs(rfrac) == half) {
        // tie: choose even n
        const f128_s n_mod2 = fmod(n, f128_s{ 2 });
        if (n_mod2 != 0)
            n += (rfrac.hi >= 0.0 ? one : -one);
    }

    f128_s r = x - n * y;

    // If result is zero, sign should match x (std::remainder semantics)
    if (iszero(r))
        return f128_s{ _f128_detail::signbit_constexpr(x.hi) ? -0.0 : 0.0 };

    return r;
}
[[nodiscard]] NO_INLINE constexpr f128_s sqrt(f128_s a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128_s{ 0.0 };
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };
    }

    double y0;
    if (std::is_constant_evaluated()) {
        y0 = _f128_detail::sqrt_seed_constexpr(a.hi);
    } else {
        y0 = std::sqrt(a.hi);
    }
    f128_s y{ y0 };

    // Newton refinements
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    return _f128_detail::canonicalize_math_result(y);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s nearbyint(const f128_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f128_s t = floor(a);
    f128_s frac = a - t;

    if (frac < f128_s{ 0.5 })
        return t;

    if (frac > f128_s{ 0.5 })
    {
        t += f128_s{ 1.0 };
        if (iszero(t))
            return f128_s{ _f128_detail::signbit_constexpr(a.hi) ? -0.0 : 0.0 };
        return t;
    }

    if (fmod(t, f128_s{ 2.0 }) != f128_s{ 0.0 })
        t += f128_s{ 1.0 };

    if (iszero(t))
        return f128_s{ _f128_detail::signbit_constexpr(a.hi) ? -0.0 : 0.0 };

    return t;
}

/// ------------------ transcendentals ------------------

[[nodiscard]] FORCE_INLINE constexpr double log_as_double(f128_s a)
{
    const double hi = a.hi;
    if (hi <= 0.0)
        return bl::fltx::common::fp::log_constexpr(static_cast<double>(a));

    return bl::fltx::common::fp::log_constexpr(hi) + bl::fltx::common::fp::log1p_constexpr(a.lo / hi);
}

namespace _f128_const
{
    inline constexpr f128_s e          = std::numbers::e_v<f128_s>;
    inline constexpr f128_s log2e      = std::numbers::log2e_v<f128_s>;
    inline constexpr f128_s log10e     = std::numbers::log10e_v<f128_s>;
    inline constexpr f128_s pi         = std::numbers::pi_v<f128_s>;
    inline constexpr f128_s inv_pi     = std::numbers::inv_pi_v<f128_s>;
    inline constexpr f128_s inv_sqrtpi = std::numbers::inv_sqrtpi_v<f128_s>;
    inline constexpr f128_s ln2        = std::numbers::ln2_v<f128_s>;
    inline constexpr f128_s ln10       = std::numbers::ln10_v<f128_s>;
    inline constexpr f128_s sqrt2      = std::numbers::sqrt2_v<f128_s>;
    inline constexpr f128_s sqrt3      = std::numbers::sqrt3_v<f128_s>;
    inline constexpr f128_s inv_sqrt3  = std::numbers::inv_sqrt3_v<f128_s>;
    inline constexpr f128_s egamma     = std::numbers::egamma_v<f128_s>;
    inline constexpr f128_s phi        = std::numbers::phi_v<f128_s>;

    inline constexpr f128_s pi_2      = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54 };
    inline constexpr f128_s pi_4      = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55 };
    inline constexpr f128_s invpi2    = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55 };
    inline constexpr f128_s inv_ln2   = log2e;
    inline constexpr f128_s inv_ln10  = log10e;
    inline constexpr f128_s sqrt_half = { 0x1.6a09e667f3bcdp-1, -0x1.bdd3413b26456p-55 };
}
namespace _f128_detail
{
    inline constexpr double pi_4_hi = _f128_const::pi_4.hi;

    using fltx::common::fp::signbit_constexpr;
    using fltx::common::fp::fabs_constexpr;
    using fltx::common::fp::floor_constexpr;
    using fltx::common::fp::ceil_constexpr;
    using fltx::common::fp::double_integer_is_odd;
    using fltx::common::fp::fmod_constexpr;
    using fltx::common::fp::sqrt_seed_constexpr;
    using fltx::common::fp::nearbyint_ties_even;

    FORCE_INLINE constexpr f128_s f128_exp_kernel_ln2_half(const f128_s& r)
    {
        f128_s p = f128_s{ 8.89679139245057408e-22 };
        p *= r + f128_s{ 1.95729410633912626e-20 };
        p *= r + f128_s{ 4.11031762331216484e-19 };
        p *= r + f128_s{ 8.22063524662432950e-18 };
        p *= r + f128_s{ 1.56192069685862253e-16 };
        p *= r + f128_s{ 2.81145725434552060e-15 };
        p *= r + f128_s{ 4.77947733238738525e-14 };
        p *= r + f128_s{ 7.64716373181981641e-13 };
        p *= r + f128_s{ 1.14707455977297245e-11 };
        p *= r + f128_s{ 1.60590438368216133e-10 };
        p *= r + f128_s{ 2.08767569878681002e-09 };
        p *= r + f128_s{ 2.50521083854417202e-08 };
        p *= r + f128_s{ 2.75573192239858883e-07 };
        p *= r + f128_s{ 2.75573192239858925e-06 };
        p *= r + f128_s{ 2.48015873015873016e-05 };
        p *= r + f128_s{ 1.98412698412698413e-04 };
        p *= r + f128_s{ 1.38888888888888894e-03 };
        p *= r + f128_s{ 8.33333333333333322e-03 };
        p *= r + f128_s{ 4.16666666666666644e-02 };
        p *= r + f128_s{ 1.66666666666666657e-01 };
        p *= r + f128_s{ 5.00000000000000000e-01 };
        p *= r + f128_s{ 1.0 };
        return (p * r) + f128_s{ 1.0 };
    }
    FORCE_INLINE constexpr f128_s f128_expm1_tiny(const f128_s& r)
    {
        f128_s p =    f128_s{1.0} / f128_s{6227020800.0};
        p = p * r + f128_s{1.0} / f128_s{479001600.0};
        p = p * r + f128_s{1.0} / f128_s{39916800.0};
        p = p * r + f128_s{1.0} / f128_s{3628800.0};
        p = p * r + f128_s{1.0} / f128_s{362880.0};
        p = p * r + f128_s{1.0} / f128_s{40320.0};
        p = p * r + f128_s{1.0} / f128_s{5040.0};
        p = p * r + f128_s{1.0} / f128_s{720.0};
        p = p * r + f128_s{1.0} / f128_s{120.0};
        p = p * r + f128_s{1.0} / f128_s{24.0};
        p = p * r + f128_s{1.0} / f128_s{6.0};
        p = p * r + f128_s{0.5};
        return r + (r * r) * p;
    }

    FORCE_INLINE constexpr bool f128_remainder_pio2(const f128_s& x, long long& n_out, f128_s& r_out)
    {
        const double ax = _f128_detail::fabs_constexpr(x.hi);
        if (!_f128_detail::isfinite(ax))
            return false;

        if (ax > 7.0e15)
            return false;

        const f128_s t = x * _f128_const::invpi2;

        double qd = _f128_detail::nearbyint_ties_even(t.hi);
        if (!_f128_detail::isfinite(qd) ||
            qd < (double)std::numeric_limits<long long>::min() ||
            qd >(double)std::numeric_limits<long long>::max())
            return false;

        const f128_s delta = t - f128_s{ qd };
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            qd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            qd -= 1.0;

        if (qd < (double)std::numeric_limits<long long>::min() ||
            qd >(double)std::numeric_limits<long long>::max())
            return false;

        long long n = (long long)qd;
        f128_s r = x - f128_s{ qd } * _f128_const::pi_2;

        if (r > f128_s{ _f128_detail::pi_4_hi })
        {
            r -= _f128_const::pi_2;
            ++n;
        }
        else if (r < f128_s{ -_f128_detail::pi_4_hi })
        {
            r += _f128_const::pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }
    FORCE_INLINE constexpr f128_s f128_sin_kernel_pi4(const f128_s& x)
    {
        const f128_s t = x * x;

        f128_s ps = f128_s{  1.13099628864477159e-31,  1.04980154129595057e-47 };
        ps = ps * t + f128_s{ -9.18368986379554615e-29, -1.43031503967873224e-45 };
        ps = ps * t + f128_s{  6.44695028438447391e-26, -1.93304042337034642e-42 };
        ps = ps * t + f128_s{ -3.86817017063068404e-23,  8.84317765548234382e-40 };
        ps = ps * t + f128_s{  1.95729410633912625e-20, -1.36435038300879076e-36 };
        ps = ps * t + f128_s{ -8.22063524662432972e-18, -2.21418941196042654e-34 };
        ps = ps * t + f128_s{  2.81145725434552060e-15,  1.65088427308614330e-31 };
        ps = ps * t + f128_s{ -7.64716373181981648e-13, -7.03872877733452971e-30 };
        ps = ps * t + f128_s{  1.60590438368216146e-10,  1.25852945887520981e-26 };
        ps = ps * t + f128_s{ -2.50521083854417188e-08,  1.44881407093591197e-24 };
        ps = ps * t + f128_s{  2.75573192239858907e-06, -1.85839327404647208e-22 };
        ps = ps * t + f128_s{ -1.98412698412698413e-04, -1.72095582934207053e-22 };
        ps = ps * t + f128_s{  8.33333333333333322e-03,  1.15648231731787140e-19 };
        ps = ps * t + f128_s{ -1.66666666666666657e-01, -9.25185853854297066e-18 };
        return x + x * t * ps;
    }
    FORCE_INLINE constexpr f128_s f128_cos_kernel_pi4(const f128_s& x)
    {
        const f128_s t = x * x;

        f128_s pc = f128_s{  3.27988923706983791e-30,  1.51175427440298786e-46 };
        pc = pc * t + f128_s{ -2.47959626322479746e-27,  1.29537309647652292e-43 };
        pc = pc * t + f128_s{  1.61173757109611835e-24, -3.68465735645097656e-41 };
        pc = pc * t + f128_s{ -8.89679139245057329e-22,  7.91140261487237594e-38 };
        pc = pc * t + f128_s{  4.11031762331216486e-19,  1.44129733786595266e-36 };
        pc = pc * t + f128_s{ -1.56192069685862265e-16, -1.19106796602737541e-32 };
        pc = pc * t + f128_s{  4.77947733238738530e-14,  4.39920548583408094e-31 };
        pc = pc * t + f128_s{ -1.14707455977297247e-11, -2.06555127528307454e-28 };
        pc = pc * t + f128_s{  2.08767569878680990e-09, -1.20734505911325997e-25 };
        pc = pc * t + f128_s{ -2.75573192239858907e-07, -2.37677146222502973e-23 };
        pc = pc * t + f128_s{  2.48015873015873016e-05,  2.15119478667758816e-23 };
        pc = pc * t + f128_s{ -1.38888888888888894e-03,  5.30054395437357706e-20 };
        pc = pc * t + f128_s{  4.16666666666666644e-02,  2.31296463463574269e-18 };
        pc = pc * t + f128_s{ -5.00000000000000000e-01,  0.0 };
        return f128_s{ 1.0 } + t * pc;
    }
    FORCE_INLINE constexpr void f128_sincos_kernel_pi4(const f128_s& x, f128_s& s_out, f128_s& c_out)
    {
        s_out = f128_sin_kernel_pi4(x);
        c_out = f128_cos_kernel_pi4(x);
    }

    FORCE_INLINE constexpr f128_s canonicalize_exp_result(f128_s value) noexcept
    {
        value.lo = fltx::common::fp::zero_low_fraction_bits_finite<6>(value.lo);
        return value;
    }

    FORCE_INLINE constexpr f128_s _ldexp(const f128_s& x, int e)
    {
        if (std::is_constant_evaluated())
        {
            return canonicalize_exp_result(_f128_detail::renorm(
                fltx::common::fp::ldexp_constexpr2(x.hi, e),
                fltx::common::fp::ldexp_constexpr2(x.lo, e)
            ));
        }
        else
        {
            return canonicalize_exp_result(_f128_detail::renorm(
                std::ldexp(x.hi, e),
                std::ldexp(x.lo, e)
            ));
        }
    }
    FORCE_INLINE constexpr f128_s _exp(const f128_s& x)
    {
        if (isnan(x))
            return x;
        if (isinf(x))
            return (x.hi < 0.0) ? f128_s{ 0.0 } : std::numeric_limits<f128_s>::infinity();

        if (x.hi > 709.782712893384)
            return std::numeric_limits<f128_s>::infinity();

        if (x.hi < -745.133219101941)
            return f128_s{ 0.0 };

        if (iszero(x))
            return f128_s{ 1.0 };

        const f128_s t = x * _f128_const::inv_ln2;

        double kd = _f128_detail::nearbyint_ties_even(t.hi);
        const f128_s delta = t - f128_s{ kd };
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            kd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            kd -= 1.0;

        const int k = static_cast<int>(kd);
        const f128_s r = (x - f128_s{ kd } * _f128_const::ln2) * f128_s{ 0.0009765625 };

        f128_s e = _f128_detail::f128_expm1_tiny(r);
        for (int i = 0; i < 10; ++i)
            e = e * (e + 2.0);

        return _ldexp(e + 1.0, k);
    }
    FORCE_INLINE constexpr f128_s _log(const f128_s& a)
    {
        if (isnan(a))
            return a;
        if (iszero(a))
            return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
        if (a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0))
            return std::numeric_limits<f128_s>::quiet_NaN();
        if (isinf(a))
            return a;

        int exp2 = 0;
        if (std::is_constant_evaluated()) {
            exp2 = fltx::common::fp::frexp_exponent_constexpr(a.hi);
        }
        else {
            (void)std::frexp(a.hi, &exp2);
        }

        f128_s m = _ldexp(a, -exp2);
        if (m < _f128_const::sqrt_half)
        {
            m *= 2.0;
            --exp2;
        }

        const f128_s exp2_ln2 = f128_s{ static_cast<double>(exp2) } * _f128_const::ln2;
        f128_s y = exp2_ln2 + f128_s{ log_as_double(m) };
        y += m * _exp(exp2_ln2 - y) - 1.0;
        y += m * _exp(exp2_ln2 - y) - 1.0;
        y += m * _exp(exp2_ln2 - y) - 1.0;
        return y;
    }
}

NO_INLINE constexpr f128_s pow10_128(int k);

// exp
[[nodiscard]] NO_INLINE constexpr f128_s ldexp(const f128_s& x, int e)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_ldexp(x, e));
}
[[nodiscard]] NO_INLINE constexpr f128_s exp(const f128_s& x)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_exp(x));
}
[[nodiscard]] NO_INLINE constexpr f128_s exp2(const f128_s& x)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_exp(x * _f128_const::ln2));
}

// log
[[nodiscard]] NO_INLINE constexpr f128_s log(const f128_s& a)
{
    return _f128_detail::canonicalize_math_result(_f128_detail::_log(a));
}
[[nodiscard]] NO_INLINE constexpr f128_s log2(const f128_s& a)
{
    int exact_exp2{};
    if (_f128_detail::f128_try_exact_binary_log2(a, exact_exp2))
        return f128_s{ static_cast<double>(exact_exp2), 0.0 };

    return _f128_detail::canonicalize_math_result(_f128_detail::_log(a) * _f128_const::inv_ln2);
}
[[nodiscard]] NO_INLINE constexpr f128_s log10(const f128_s& x)
{
    if (x.hi > 0.0)
    {
        const int exp2 =
            fltx::common::fp::frexp_exponent_constexpr(x.hi);
        const int k0 =
            static_cast<int>(fltx::common::fp::floor_constexpr((exp2 - 1) * 0.30102999566398114));

        for (int k = k0 - 2; k <= k0 + 2; ++k)
        {
            if (x == pow10_128(k))
                return f128_s{ static_cast<double>(k), 0.0 };
        }
    }

    return _f128_detail::canonicalize_math_result(_f128_detail::_log(x) * _f128_const::inv_ln10);
}

// pow
[[nodiscard]] NO_INLINE constexpr f128_s pow(const f128_s& x, const f128_s& y)
{
    if (iszero(y))
        return f128_s{ 1.0 };

    if (isnan(x) || isnan(y))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s yi = trunc(y);
    const bool y_is_int = (yi == y);

    int64_t yi64{};
    if (y_is_int && _f128_detail::f128_try_get_int64(yi, yi64))
        return _f128_detail::powi(x, yi64);

    if (x.hi < 0.0 || (x.hi == 0.0 && _f128_detail::signbit_constexpr(x.hi)))
    {
        if (!y_is_int)
            return std::numeric_limits<f128_s>::quiet_NaN();

        const f128_s magnitude = exp(y * log(-x));
        const f128_s parity = fmod(abs(yi), f128_s{ 2.0 });
        return _f128_detail::canonicalize_math_result((parity == f128_s{ 1.0 }) ? -magnitude : magnitude);
    }

    return _f128_detail::canonicalize_math_result(exp(y * log(x)));
}
[[nodiscard]] NO_INLINE constexpr f128_s pow10_128(int k)
{
    if (k == 0) return f128_s{ 1.0 };

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

// trig
[[nodiscard]] NO_INLINE constexpr bool sincos(const f128_s& x, f128_s& s_out, f128_s& c_out)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
    {
        s_out = f128_s{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= _f128_detail::pi_4_hi)
    {
        _f128_detail::f128_sincos_kernel_pi4(x, s_out, c_out);
        s_out = _f128_detail::canonicalize_math_result(s_out);
        c_out = _f128_detail::canonicalize_math_result(c_out);
        return true;
    }

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
        return false;

    f128_s sr{}, cr{};
    _f128_detail::f128_sincos_kernel_pi4(r, sr, cr);

    switch ((int)(n & 3))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    s_out = _f128_detail::canonicalize_math_result(s_out);
    c_out = _f128_detail::canonicalize_math_result(c_out);
    return true;
}
[[nodiscard]] NO_INLINE constexpr f128_s sin(const f128_s& x)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= _f128_detail::pi_4_hi)
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
        if (std::is_constant_evaluated()) {
            return _f128_detail::canonicalize_math_result(f128_s{ fltx::common::fp::sin_constexpr(static_cast<double>(x)) });
        } else {
            return _f128_detail::canonicalize_math_result(f128_s{ std::sin((double)x) });
        }

    switch ((int)(n & 3))
    {
    case 0: return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(r));
    case 1: return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(r));
    case 2: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_sin_kernel_pi4(r));
    default: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_cos_kernel_pi4(r));
    }
}
[[nodiscard]] NO_INLINE constexpr f128_s cos(const f128_s& x)
{
    const double ax = _f128_detail::fabs_constexpr(x.hi);
    if (!_f128_detail::isfinite(ax))
        return f128_s{ std::numeric_limits<double>::quiet_NaN() };

    if (ax <= _f128_detail::pi_4_hi)
        return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(x));

    long long n = 0;
    f128_s r{};
    if (!_f128_detail::f128_remainder_pio2(x, n, r))
    {
        if (std::is_constant_evaluated())
        {
            return _f128_detail::canonicalize_math_result(f128_s{ fltx::common::fp::cos_constexpr(static_cast<double>(x)) });
        }
        else 
        {
            return _f128_detail::canonicalize_math_result(f128_s{ std::cos((double)x) });
        }
    }

    switch ((int)(n & 3))
    {
    case 0: return _f128_detail::canonicalize_math_result(_f128_detail::f128_cos_kernel_pi4(r));
    case 1: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_sin_kernel_pi4(r));
    case 2: return _f128_detail::canonicalize_math_result(-_f128_detail::f128_cos_kernel_pi4(r));
    default: return _f128_detail::canonicalize_math_result(_f128_detail::f128_sin_kernel_pi4(r));
    }
}
[[nodiscard]] NO_INLINE constexpr f128_s tan(const f128_s& x)
{
    f128_s s{}, c{};
    if (sincos(x, s, c))
        return s / c;
    const double xd = (double)x;
    if (std::is_constant_evaluated()) {
        return f128_s{ fltx::common::fp::tan_constexpr(xd) };
    } else {
        return f128_s{ std::tan(xd) };
    }
}
[[nodiscard]] NO_INLINE constexpr f128_s atan2(const f128_s& y, const f128_s& x)
{
    if (iszero(x))
    {
        if (iszero(y))
            return f128_s{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? _f128_const::pi_2 : -_f128_const::pi_2;
    }

    const f128_s scale = std::max(abs(x), abs(y));
    const f128_s xs = x / scale;
    const f128_s ys = y / scale;

    f128_s v{ fltx::common::fp::atan2_constexpr(y.hi, x.hi) };

    for (int i = 0; i < 2; ++i)
    {
        f128_s sv{}, cv{};
        if (!sincos(v, sv, cv))
        {
            const double vd = (double)v;
            if (std::is_constant_evaluated()) {
                double sd{}, cd{};
                fltx::common::fp::sincos_constexpr(vd, sd, cd);
                sv = f128_s{ sd };
                cv = f128_s{ cd };
            } else {
                sv = f128_s{ std::sin(vd) };
                cv = f128_s{ std::cos(vd) };
            }
        }

        const f128_s f = xs * sv - ys * cv;
        const f128_s fp = xs * cv + ys * sv;

        v = v - f / fp;
    }

    return _f128_detail::canonicalize_math_result(v);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s atan(const f128_s& x)
{
    return atan2(x, f128_s{ 1.0 });
}
[[nodiscard]] FORCE_INLINE constexpr f128_s asin(const f128_s& x)
{
    return atan2(x, sqrt(f128_s{ 1.0 } - x * x));
}
[[nodiscard]] FORCE_INLINE constexpr f128_s acos(const f128_s& x)
{
    return atan2(sqrt(f128_s{ 1.0 } - x * x), x);
}


[[nodiscard]] FORCE_INLINE constexpr f128_s fabs(const f128_s& a) noexcept
{
    return abs(a);
}

[[nodiscard]] FORCE_INLINE constexpr bool signbit(const f128_s& x) noexcept
{
    return _f128_detail::signbit_constexpr(x.hi) || (x.hi == 0.0 && _f128_detail::signbit_constexpr(x.lo));
}
[[nodiscard]] FORCE_INLINE constexpr int fpclassify(const f128_s& x) noexcept
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

namespace _f128_detail
{
    FORCE_INLINE constexpr f128_s round_half_away_zero(const f128_s& x) noexcept
    {
        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        if (signbit(x))
        {
            f128_s y = -floor((-x) + f128_s{ 0.5 });
            if (iszero(y))
                return f128_s{ -0.0, 0.0 };
            return y;
        }

        return floor(x + f128_s{ 0.5 });
    }

    FORCE_INLINE constexpr double nextafter_double_constexpr(double from, double to) noexcept
    {
        if (fltx::common::fp::isnan(from) || fltx::common::fp::isnan(to))
            return std::numeric_limits<double>::quiet_NaN();

        if (from == to)
            return to;

        if (from == 0.0)
            return fltx::common::fp::signbit_constexpr(to)
                ? -std::numeric_limits<double>::denorm_min()
                :  std::numeric_limits<double>::denorm_min();

        std::uint64_t bits = std::bit_cast<std::uint64_t>(from);
        if ((from > 0.0) == (from < to))
            ++bits;
        else
            --bits;

        return std::bit_cast<double>(bits);
    }

    template<typename SignedInt>
    FORCE_INLINE constexpr SignedInt to_signed_integer_or_zero(const f128_s& x) noexcept
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);
        if (isnan(x) || isinf(x))
            return 0;

        const f128_s lo = to_f128(static_cast<int64_t>(std::numeric_limits<SignedInt>::lowest()));
        const f128_s hi = to_f128(static_cast<int64_t>(std::numeric_limits<SignedInt>::max()));
        if (x < lo || x > hi)
            return 0;

        int64_t out = 0;
        if (!_f128_detail::f128_try_get_int64(x, out))
            return 0;

        return static_cast<SignedInt>(out);
    }

    FORCE_INLINE constexpr f128_s nearest_integer_ties_even(const f128_s& q) noexcept
    {
        f128_s n = trunc(q);
        const f128_s frac = q - n;
        const f128_s half{ 0.5 };
        const f128_s one{ 1.0 };

        if (abs(frac) > half)
        {
            n += signbit(frac) ? -one : one;
        }
        else if (abs(frac) == half)
        {
            if (fmod(n, f128_s{ 2.0 }) != f128_s{ 0.0 })
                n += signbit(frac) ? -one : one;
        }

        return n;
    }

    template<class Func>
    NO_INLINE constexpr f128_s adaptive_simpson_recursive(
        const Func& f,
        const f128_s& a,
        const f128_s& b,
        const f128_s& fa,
        const f128_s& fm,
        const f128_s& fb,
        const f128_s& whole,
        const f128_s& eps,
        int depth)
    {
        const f128_s m = (a + b) * f128_s{ 0.5 };
        const f128_s lm = (a + m) * f128_s{ 0.5 };
        const f128_s rm = (m + b) * f128_s{ 0.5 };

        const f128_s flm = f(lm);
        const f128_s frm = f(rm);

        const f128_s left  = (m - a) * (fa + f128_s{ 4.0 } * flm + fm) / f128_s{ 6.0 };
        const f128_s right = (b - m) * (fm + f128_s{ 4.0 } * frm + fb) / f128_s{ 6.0 };
        const f128_s delta = left + right - whole;

        if (depth <= 0 || abs(delta) <= f128_s{ 15.0 } * eps)
            return left + right + delta / f128_s{ 15.0 };

        return adaptive_simpson_recursive(f, a, m, fa, flm, fm, left, eps * f128_s{ 0.5 }, depth - 1)
             + adaptive_simpson_recursive(f, m, b, fm, frm, fb, right, eps * f128_s{ 0.5 }, depth - 1);
    }

    template<class Func>
    NO_INLINE constexpr f128_s adaptive_simpson(const Func& f, const f128_s& a, const f128_s& b, const f128_s& eps, int depth = 18)
    {
        const f128_s m = (a + b) * f128_s{ 0.5 };
        const f128_s fa = f(a);
        const f128_s fm = f(m);
        const f128_s fb = f(b);
        const f128_s whole = (b - a) * (fa + f128_s{ 4.0 } * fm + fb) / f128_s{ 6.0 };
        return adaptive_simpson_recursive(f, a, b, fa, fm, fb, whole, eps, depth);
    }

    inline constexpr int spouge_a = 40;
    inline constexpr f128_s spouge_coeffs[spouge_a] =
    {
        f128_s{ 0x1.40d931ff62706p+1, -0x1.a6a0d6f814637p-53 },
        f128_s{ 0x1.e04e2378eac2dp+58, 0x1.441c3d822c54bp+4 },
        f128_s{ -0x1.9e3bf3da3102bp+62, -0x1.6b7cace6ee2f9p+7 },
        f128_s{ 0x1.52944c7567b90p+65, -0x1.2be216bee5795p+11 },
        f128_s{ -0x1.5cee3e673ca16p+67, 0x1.aede5e8e1dfe1p+13 },
        f128_s{ 0x1.fcdd4a88643f0p+68, 0x1.6f23cbeb7834ep+7 },
        f128_s{ -0x1.17527b9557c3ap+70, 0x1.6df29f5855351p+16 },
        f128_s{ 0x1.df95f4ba97ec4p+70, 0x1.e8d3fc67ed729p+14 },
        f128_s{ -0x1.4a29a61693badp+71, -0x1.f27c26ccb1d1bp+17 },
        f128_s{ 0x1.72ee538d2f060p+71, 0x1.12bd3e2f702c9p+17 },
        f128_s{ -0x1.5837d0f145407p+71, -0x1.4dbe5ec4d627ep+17 },
        f128_s{ 0x1.0a1cf1ae9efb4p+71, 0x1.23aac3bac3d82p+17 },
        f128_s{ -0x1.58c8c3deae811p+70, -0x1.265ee19944c46p+16 },
        f128_s{ 0x1.77b105d51ada7p+69, -0x1.3684fda97e8f6p+14 },
        f128_s{ -0x1.58eb2c4b76468p+68, -0x1.ea121cb78b86fp+13 },
        f128_s{ 0x1.0ae0f912a0748p+67, -0x1.61364f0f1cad8p+12 },
        f128_s{ -0x1.5ba4650c489ffp+65, 0x1.5126088373a47p+10 },
        f128_s{ 0x1.7c33676c51d29p+63, -0x1.6ba1af9e9f596p+7 },
        f128_s{ -0x1.5bb5c52dd1f12p+61, 0x1.26149af8697afp+7 },
        f128_s{ 0x1.08775cced0c59p+59, 0x1.05380fd87fedbp+4 },
        f128_s{ -0x1.4c3b222a5b897p+56, 0x1.da2f06988f8acp+2 },
        f128_s{ 0x1.55a4026248906p+53, 0x1.0444a07fc3104p-1 },
        f128_s{ -0x1.1c7b8fcb44898p+50, 0x1.62258e547baafp-5 },
        f128_s{ 0x1.7a9e09bfb1327p+46, 0x1.94100dcfbd9afp-8 },
        f128_s{ -0x1.8c4b272a7678bp+42, 0x1.80266d5b62afcp-13 },
        f128_s{ 0x1.3fecadbe602bep+38, -0x1.b9192dbcab9bbp-17 },
        f128_s{ -0x1.8509be037b532p+33, -0x1.4858129b85445p-23 },
        f128_s{ 0x1.5a068a1689b0cp+28, 0x1.9541b5f07fab9p-27 },
        f128_s{ -0x1.b220223bd7630p+22, 0x1.0700bd994c314p-33 },
        f128_s{ 0x1.6ee6e752f31fap+16, 0x1.b4ee42088fc51p-39 },
        f128_s{ -0x1.89e41554e879cp+9, 0x1.94a72d327c730p-45 },
        f128_s{ 0x1.f146712eabbbcp+1, -0x1.01304a6157022p-53 },
        f128_s{ -0x1.4cc01b7e248f8p-7, 0x1.8dec5ddb41371p-61 },
        f128_s{ 0x1.98a0368dca14dp-17, -0x1.6a259f14d92f2p-71 },
        f128_s{ -0x1.7572340daacbep-28, -0x1.96e8c21e59997p-82 },
        f128_s{ 0x1.7055b4e6080b9p-41, -0x1.e1ffb043a9773p-96 },
        f128_s{ -0x1.cc4861a33136dp-57, 0x1.42693d10cc93bp-111 },
        f128_s{ 0x1.0f83d308726f7p-76, -0x1.04bd62417c2e4p-131 },
        f128_s{ -0x1.0ee5fdd16f12cp-103, 0x1.cf88d83fc73a9p-160 },
        f128_s{ 0x1.dabcbe465bfb2p-148, 0x1.5f7085ca81db0p-202 }
    };

    FORCE_INLINE constexpr f128_s spouge_sum(const f128_s& z) noexcept
    {
        const f128_s zm1 = z - f128_s{ 1.0 };
        f128_s sum = spouge_coeffs[0];
        for (int k = 1; k < spouge_a; ++k)
            sum += spouge_coeffs[k] / (zm1 + f128_s{ static_cast<double>(k) });
        return sum;
    }

    FORCE_INLINE constexpr f128_s spouge_log_gamma_positive(const f128_s& z) noexcept
    {
        const f128_s zm1 = z - f128_s{ 1.0 };
        const f128_s u = zm1 + f128_s{ static_cast<double>(spouge_a) };
        const f128_s sum = spouge_sum(z);
        return log(sum) + (zm1 + f128_s{ 0.5 }) * log(u) - u;
    }

    FORCE_INLINE constexpr f128_s spouge_gamma_positive(const f128_s& z) noexcept
    {
        const f128_s zm1 = z - f128_s{ 1.0 };
        const f128_s u = zm1 + f128_s{ static_cast<double>(spouge_a) };
        const f128_s sum = spouge_sum(z);
        return exp(((zm1 + f128_s{ 0.5 }) * log(u)) - u) * sum;
    }
}

[[nodiscard]] NO_INLINE constexpr f128_s expm1(const f128_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    f128_s r = x;
    int squarings = 0;
    while (abs(r) > f128_s{ 0.125 })
    {
        r *= f128_s{ 0.5 };
        ++squarings;
    }

    f128_s e = _f128_detail::f128_expm1_tiny(r);
    for (int i = 0; i < squarings; ++i)
        e = e * (e + f128_s{ 2.0 });

    return _f128_detail::canonicalize_math_result(e);
}
[[nodiscard]] NO_INLINE constexpr f128_s log1p(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (x == f128_s{ -1.0 })
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (x < f128_s{ -1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(x))
        return x;
    if (iszero(x))
        return x;

    if (abs(x) < f128_s{ 0.5 })
    {
        f128_s y{ fltx::common::fp::log1p_constexpr(static_cast<double>(x)) };
        for (int i = 0; i < 3; ++i)
            y += (x - expm1(y)) / exp(y);
        return _f128_detail::canonicalize_math_result(y);
    }

    return _f128_detail::canonicalize_math_result(log(f128_s{ 1.0 } + x));
}

[[nodiscard]] NO_INLINE constexpr f128_s sinh(const f128_s& x)
{
    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const f128_s em1 = expm1(x);
    const f128_s out = em1 * (f128_s{ 1.0 } + f128_s{ 1.0 } / (em1 + f128_s{ 1.0 })) * f128_s{ 0.5 };
    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] NO_INLINE constexpr f128_s cosh(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s ax = abs(x);
    const f128_s ex = exp(ax);
    return _f128_detail::canonicalize_math_result((ex + f128_s{ 1.0 } / ex) * f128_s{ 0.5 });
}
[[nodiscard]] NO_INLINE constexpr f128_s tanh(const f128_s& x)
{
    if (isnan(x) || iszero(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s ax = abs(x);
    if (ax > f128_s{ 20.0 })
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };

    const f128_s em1 = expm1(ax + ax);
    f128_s out = em1 / (em1 + f128_s{ 2.0 });
    if (signbit(x))
        out = -out;
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] NO_INLINE constexpr f128_s asinh(const f128_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const f128_s ax = abs(x);
    f128_s out{};
    if (ax > f128_s{ 0x1p500 })
        out = log(ax) + _f128_const::ln2;
    else
        out = log(ax + sqrt(ax * ax + f128_s{ 1.0 }));

    if (signbit(x))
        out = -out;
    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] NO_INLINE constexpr f128_s acosh(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (x < f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (x == f128_s{ 1.0 })
        return f128_s{ 0.0 };
    if (isinf(x))
        return x;

    f128_s out{};
    if (x > f128_s{ 0x1p500 })
        out = log(x) + _f128_const::ln2;
    else
        out = log(x + sqrt((x - f128_s{ 1.0 }) * (x + f128_s{ 1.0 })));

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] NO_INLINE constexpr f128_s atanh(const f128_s& x)
{
    if (isnan(x) || iszero(x))
        return x;

    const f128_s ax = abs(x);
    if (ax > f128_s{ 1.0 })
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (ax == f128_s{ 1.0 })
        return signbit(x)
            ? f128_s{ -std::numeric_limits<double>::infinity(), 0.0 }
            : f128_s{  std::numeric_limits<double>::infinity(), 0.0 };

    const f128_s out = (log1p(x) - log1p(-x)) * f128_s{ 0.5 };
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] NO_INLINE constexpr f128_s cbrt(const f128_s& x)
{
    if (isnan(x) || iszero(x) || isinf(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s y = std::is_constant_evaluated()
        ? exp(log(ax) / f128_s{ 3.0 })
        : f128_s{ std::cbrt(static_cast<double>(ax)) };

    for (int i = 0; i < 5; ++i)
        y = (y + y + ax / (y * y)) / f128_s{ 3.0 };

    if (neg)
        y = -y;

    return _f128_detail::canonicalize_math_result(y);
}
[[nodiscard]] NO_INLINE constexpr f128_s hypot(const f128_s& x, const f128_s& y)
{
    if (isinf(x) || isinf(y))
        return std::numeric_limits<f128_s>::infinity();
    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    f128_s ax = abs(x);
    f128_s ay = abs(y);
    if (ax < ay)
        std::swap(ax, ay);

    if (iszero(ax))
        return f128_s{ 0.0 };

    const f128_s r = ay / ax;
    return _f128_detail::canonicalize_math_result(ax * sqrt(f128_s{ 1.0 } + r * r));
}

[[nodiscard]] FORCE_INLINE constexpr f128_s rint(const f128_s& x)
{
    return nearbyint(x);
}
[[nodiscard]] FORCE_INLINE constexpr long lround(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long>(_f128_detail::round_half_away_zero(x));
}
[[nodiscard]] FORCE_INLINE constexpr long long llround(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long long>(_f128_detail::round_half_away_zero(x));
}
[[nodiscard]] FORCE_INLINE constexpr long lrint(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long>(nearbyint(x));
}
[[nodiscard]] FORCE_INLINE constexpr long long llrint(const f128_s& x)
{
    return _f128_detail::to_signed_integer_or_zero<long long>(nearbyint(x));
}

[[nodiscard]] NO_INLINE constexpr f128_s remquo(const f128_s& x, const f128_s& y, int* quo)
{
    if (quo)
        *quo = 0;

    if (isnan(x) || isnan(y) || iszero(y) || isinf(x))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (isinf(y))
        return x;

    const f128_s n = _f128_detail::nearest_integer_ties_even(x / y);
    f128_s r = x - n * y;

    if (quo)
    {
        const f128_s qbits = fmod(abs(n), f128_s{ 2147483648.0 });
        int bits = static_cast<int>(trunc(qbits).hi);
        if (signbit(n))
            bits = -bits;
        *quo = bits;
    }

    if (iszero(r))
        return f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };

    return _f128_detail::canonicalize_math_result(r);
}

[[nodiscard]] FORCE_INLINE constexpr f128_s fma(const f128_s& x, const f128_s& y, const f128_s& z)
{
    return _f128_detail::canonicalize_math_result(x * y + z);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s fmin(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? a : b;
    return a;
}
[[nodiscard]] FORCE_INLINE constexpr f128_s fmax(const f128_s& a, const f128_s& b)
{
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    if (iszero(a) && iszero(b))
        return signbit(a) ? b : a;
    return a;
}
[[nodiscard]] FORCE_INLINE constexpr f128_s fdim(const f128_s& x, const f128_s& y)
{
    return (x > y) ? _f128_detail::canonicalize_math_result(x - y) : f128_s{ 0.0 };
}
[[nodiscard]] FORCE_INLINE constexpr f128_s copysign(const f128_s& x, const f128_s& y)
{
    return signbit(x) == signbit(y) ? x : -x;
}

[[nodiscard]] NO_INLINE constexpr f128_s frexp(const f128_s& x, int* exp) noexcept
{
    if (exp)
        *exp = 0;

    if (isnan(x) || isinf(x) || iszero(x))
        return x;

    const double lead = (x.hi != 0.0) ? x.hi : x.lo;
    int e = 0;

    if (std::is_constant_evaluated())
        e = fltx::common::fp::frexp_exponent_constexpr(lead);
    else
        (void)std::frexp(lead, &e);

    f128_s m = ldexp(x, -e);
    const f128_s am = abs(m);

    if (am < f128_s{ 0.5 })
    {
        m *= f128_s{ 2.0 };
        --e;
    }
    else if (am >= f128_s{ 1.0 })
    {
        m *= f128_s{ 0.5 };
        ++e;
    }

    if (exp)
        *exp = e;

    return m;
}
[[nodiscard]] NO_INLINE constexpr f128_s modf(const f128_s& x, f128_s* iptr) noexcept
{
    const f128_s i = trunc(x);
    if (iptr)
        *iptr = i;

    f128_s frac = x - i;
    if (iszero(frac))
        frac = f128_s{ signbit(x) ? -0.0 : 0.0, 0.0 };
    return frac;
}
[[nodiscard]] FORCE_INLINE constexpr int ilogb(const f128_s& x) noexcept
{
    if (isnan(x))
        return FP_ILOGBNAN;
    if (iszero(x))
        return FP_ILOGB0;
    if (isinf(x))
        return std::numeric_limits<int>::max();

    int e = 0;
    (void)frexp(abs(x), &e);
    return e - 1;
}
[[nodiscard]] FORCE_INLINE constexpr f128_s logb(const f128_s& x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return f128_s{ -std::numeric_limits<double>::infinity(), 0.0 };
    if (isinf(x))
        return std::numeric_limits<f128_s>::infinity();

    return f128_s{ static_cast<double>(ilogb(x)), 0.0 };
}
[[nodiscard]] FORCE_INLINE constexpr f128_s scalbn(const f128_s& x, int e) noexcept
{
    return ldexp(x, e);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s scalbln(const f128_s& x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] NO_INLINE constexpr f128_s nextafter(const f128_s& from, const f128_s& to) noexcept
{
    if (isnan(from) || isnan(to))
        return std::numeric_limits<f128_s>::quiet_NaN();
    if (from == to)
        return to;
    if (iszero(from))
        return signbit(to)
            ? f128_s{ -std::numeric_limits<double>::denorm_min(), 0.0 }
            : f128_s{  std::numeric_limits<double>::denorm_min(), 0.0 };
    if (isinf(from))
        return signbit(from)
            ? -std::numeric_limits<f128_s>::max()
            :  std::numeric_limits<f128_s>::max();

    const double toward = (from < to)
        ? std::numeric_limits<double>::infinity()
        : -std::numeric_limits<double>::infinity();

    return _f128_detail::renorm(
        from.hi,
        _f128_detail::nextafter_double_constexpr(from.lo, toward)
    );
}
[[nodiscard]] FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, long double to) noexcept
{
    return nextafter(from, f128_s{ static_cast<double>(to) });
}
[[nodiscard]] FORCE_INLINE constexpr f128_s nexttoward(const f128_s& from, const f128_s& to) noexcept
{
    return nextafter(from, to);
}

[[nodiscard]] NO_INLINE constexpr f128_s erfc(const f128_s& x);
[[nodiscard]] NO_INLINE constexpr f128_s erf(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x) ? f128_s{ -1.0 } : f128_s{ 1.0 };
    if (iszero(x))
        return x;

    const bool neg = signbit(x);
    const f128_s ax = neg ? -x : x;

    f128_s out{ 0.0 };

    if (ax < f128_s{ 2.0 })
    {
        const f128_s xx = ax * ax;
        f128_s power = ax;
        f128_s sum = ax;

        for (int n = 1; n < 256; ++n)
        {
            power *= -xx / f128_s{ static_cast<double>(n) };
            const f128_s term = power / f128_s{ static_cast<double>(2 * n + 1) };
            sum += term;
            if (abs(term) < f128_s::eps())
                break;
        }

        out = f128_s{ 2.0 } * _f128_const::inv_sqrtpi * sum;
    }
    else
    {
        out = f128_s{ 1.0 } - erfc(ax);
    }

    if (neg)
        out = -out;

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] NO_INLINE constexpr f128_s erfc(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (x == f128_s{ 0.0 })
        return f128_s{ 1.0 };
    if (isinf(x))
        return signbit(x) ? f128_s{ 2.0 } : f128_s{ 0.0 };

    if (signbit(x))
        return f128_s{ 2.0 } - erfc(-x);

    if (x < f128_s{ 1.0 })
        return _f128_detail::canonicalize_math_result(f128_s{ 1.0 } - erf(x));

    constexpr f128_s upper = f128_s{ 1.0 - 0x1p-20, 0.0 };
    const auto integrand = [x](const f128_s& u) constexpr -> f128_s
    {
        const f128_s one_minus_u = f128_s{ 1.0 } - u;
        const f128_s t = x + u / one_minus_u;
        return exp(-(t * t)) / (one_minus_u * one_minus_u);
    };

    const f128_s integral = _f128_detail::adaptive_simpson(integrand, f128_s{ 0.0 }, upper, f128_s{ 1.0e-31 });
    const f128_s out = f128_s{ 2.0 } * _f128_const::inv_sqrtpi * integral;
    return _f128_detail::canonicalize_math_result(out);
}

[[nodiscard]] NO_INLINE constexpr f128_s lgamma(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return _f128_detail::canonicalize_math_result(_f128_detail::spouge_log_gamma_positive(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::infinity();

    const f128_s sinpix = sin(_f128_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::infinity();

    const f128_s out =
        log(_f128_const::pi)
        - log(abs(sinpix))
        - _f128_detail::spouge_log_gamma_positive(f128_s{ 1.0 } - x);

    return _f128_detail::canonicalize_math_result(out);
}
[[nodiscard]] NO_INLINE constexpr f128_s tgamma(const f128_s& x)
{
    if (isnan(x))
        return x;
    if (isinf(x))
        return signbit(x)
            ? std::numeric_limits<f128_s>::quiet_NaN()
            : std::numeric_limits<f128_s>::infinity();

    if (x > f128_s{ 0.0 })
        return _f128_detail::canonicalize_math_result(_f128_detail::spouge_gamma_positive(x));

    const f128_s xi = trunc(x);
    if (xi == x)
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s sinpix = sin(_f128_const::pi * x);
    if (iszero(sinpix))
        return std::numeric_limits<f128_s>::quiet_NaN();

    const f128_s out = _f128_const::pi / (sinpix * _f128_detail::spouge_gamma_positive(f128_s{ 1.0 } - x));
    return _f128_detail::canonicalize_math_result(out);
}


/// ======== Public string conversion wrappers ========

namespace _f128_detail
{
    FORCE_INLINE void normalize10(const f128_s& x, f128_s& m, int& exp10)
    {
        if (x.hi == 0.0 && x.lo == 0.0) { m = f128_s{ 0.0 }; exp10 = 0; return; }

        f128_s ax = abs(x);

        int e2 = fltx::common::fp::frexp_exponent_constexpr(ax.hi); // ax.hi = f * 2^(e2-1)
        int e10 = (int)fltx::common::fp::floor_constexpr((e2 - 1) * 0.30102999566398114); // ≈ log10(2)

        m = ax * pow10_128(-e10);
        while (m >= f128_s{ 10.0 }) { m = m / f128_s{ 10.0 }; ++e10; }
        while (m < f128_s{ 1.0 }) { m = m * f128_s{ 10.0 }; --e10; }
        exp10 = e10;
    }

    BL_PUSH_PRECISE
        FORCE_INLINE constexpr f128_s mul_by_double_print(f128_s a, double b) noexcept
    {
        double p, err;
        _f128_detail::two_prod_precise(a.hi, b, p, err);
        err += a.lo * b;

        double s, e;
        _f128_detail::two_sum_precise(p, err, s, e);
        return f128_s{ s, e };
    }
    FORCE_INLINE f128_s sub_by_double_print(f128_s a, double b) noexcept
    {
        double s, e;
        _f128_detail::two_sum_precise(a.hi, -b, s, e);
        e += a.lo;

        double ss, ee;
        _f128_detail::two_sum_precise(s, e, ss, ee);
        return f128_s{ ss, ee };
    }
    BL_POP_PRECISE

        struct f128_chars_result
    {
        char* ptr = nullptr;
        bool ok = false;
    };

    FORCE_INLINE int emit_uint_rev_buf(char* dst, f128_s n)
    {
        // n is a non-negative integer in f128
        const f128_s base = f128_s{ 1000000000.0 }; // 1e9

        int len = 0;

        if (n < f128_s{ 10.0 }) {
            int d = (int)n.hi;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            dst[len++] = char('0' + d);
            return len;
        }

        while (n >= base) {
            f128_s q = floor(n / base);
            f128_s r = n - q * base;

            long long chunk = (long long)std::floor(r.hi);
            if (chunk >= 1000000000LL) { chunk -= 1000000000LL; q = q + f128_s{ 1.0 }; }
            if (chunk < 0) chunk = 0;

            for (int i = 0; i < 9; ++i) {
                int d = int(chunk % 10);
                dst[len++] = char('0' + d);
                chunk /= 10;
            }

            n = q;
        }

        long long last = (long long)std::floor(n.hi);
        if (last == 0) {
            dst[len++] = '0';
        }
        else {
            while (last > 0) {
                int d = int(last % 10);
                dst[len++] = char('0' + d);
                last /= 10;
            }
        }

        return len;
    }
    FORCE_INLINE f128_chars_result append_exp10_to_chars(char* p, char* end, int e10) noexcept
    {
        if (p >= end) return { p, false };
        *p++ = 'e';

        if (p >= end) return { p, false };
        if (e10 < 0) { *p++ = '-'; e10 = -e10; }
        else { *p++ = '+'; }

        char buf[8];
        int n = 0;
        do {
            buf[n++] = char('0' + (e10 % 10));
            e10 /= 10;
        } while (e10);

        if (n < 2) buf[n++] = '0';

        if (p + n > end) return { p, false };
        for (int i = n - 1; i >= 0; --i) *p++ = buf[i];

        return { p, true };
    }

    using biguint = fltx::common::exact_decimal::biguint;

    struct exact_traits
    {
        using value_type = f128_s;
        static constexpr int limb_count = 2;
        static constexpr int significand_bits = 106;

        static double limb(const value_type& x, int index) noexcept
        {
            return index == 0 ? x.hi : x.lo;
        }
        static constexpr value_type zero(bool neg = false) noexcept
        {
            return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 };
        }
        static constexpr value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static constexpr value_type pack_from_significand(const biguint& q, int e2, bool neg) noexcept
        {
            const std::uint64_t c1 = q.get_bits(0, 53);
            const std::uint64_t c0 = q.get_bits(53, 53);
            const double hi = c0 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c0), e2 - 52) : 0.0;
            const double lo = c1 ? fltx::common::fp::ldexp_constexpr2(static_cast<double>(c1), e2 - 105) : 0.0;
            f128_s out = _f128_detail::renorm(hi, lo);
            if (neg)
                out = -out;
            return out;
        }
    };

    inline bool exact_scientific_digits(const f128_s& x, int sig, std::string& digits, int& exp10)
    {
        return fltx::common::exact_decimal::exact_scientific_digits<exact_traits>(x, sig, digits, exp10);
    }
    constexpr inline f128_s exact_decimal_to_f128(const biguint& coeff, int dec_exp, bool neg) noexcept
    {
        return fltx::common::exact_decimal::exact_decimal_to_value<exact_traits>(coeff, dec_exp, neg);
    }

    FORCE_INLINE f128_chars_result emit_fixed_dec_to_chars(char* first, char* last, f128_s x, int prec, bool strip_trailing_zeros) noexcept
    {
        if (x.hi == 0.0 && x.lo == 0.0) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (prec < 0) prec = 0;

        const bool neg = (x.hi < 0.0);
        if (neg) x = f128_s{ -x.hi, -x.lo };
        x = _f128_detail::renorm(x.hi, x.lo);

        f128_s ip = floor(x);
        f128_s fp = sub_by_double_print(x, ip.hi);

        if (fp >= f128_s{ 1.0 }) { fp = fp - f128_s{ 1.0 }; ip = ip + f128_s{ 1.0 }; }
        else if (fp < f128_s{ 0.0 }) { fp = f128_s{ 0.0 }; }

        constexpr int kFracStack = 2048;
        char frac_stack[kFracStack];
        char* frac = frac_stack;

        std::string frac_dyn;
        if (prec > kFracStack) {
            frac_dyn.resize((size_t)prec);
            frac = (char*)frac_dyn.data();
        }

        int frac_len = (prec > 0) ? prec : 0;

        if (prec > 0) {
            static constexpr double kPow10[10] = {
                1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                1000000.0, 10000000.0, 100000000.0, 1000000000.0
            };
            static constexpr uint32_t kPow10u32[10] = {
                1u, 10u, 100u, 1000u, 10000u, 100000u,
                1000000u, 10000000u, 100000000u, 1000000000u
            };

            int written = 0;
            const int full = prec / 9;
            const int rem = prec - full * 9;

            for (int c = 0; c < full; ++c) {
                fp = mul_by_double_print(fp, kPow10[9]);

                uint32_t chunk = 0;
                if (fp.hi > 0.0) {
                    const double hi_floor = std::floor(fp.hi);
                    if (hi_floor >= (double)kPow10u32[9])
                        chunk = kPow10u32[9] - 1u;
                    else
                        chunk = (uint32_t)hi_floor;
                }

                fp = sub_by_double_print(fp, (double)chunk);

                if (fp < f128_s{ 0.0 }) {
                    if (chunk > 0u) {
                        --chunk;
                        fp = sub_by_double_print(fp, -1.0);
                    }
                    else {
                        fp = f128_s{ 0.0 };
                    }
                }

                for (int i = 8; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += 9;
            }

            if (rem > 0) {
                fp = mul_by_double_print(fp, kPow10[rem]);

                uint32_t chunk = 0;
                const uint32_t chunk_limit = kPow10u32[rem] - 1u;
                if (fp.hi > 0.0) {
                    const double hi_floor = std::floor(fp.hi);
                    if (hi_floor >= (double)kPow10u32[rem])
                        chunk = chunk_limit;
                    else
                        chunk = (uint32_t)hi_floor;
                }

                fp = sub_by_double_print(fp, (double)chunk);

                if (fp < f128_s{ 0.0 }) {
                    if (chunk > 0u) {
                        --chunk;
                        fp = sub_by_double_print(fp, -1.0);
                    }
                    else {
                        fp = f128_s{ 0.0 };
                    }
                }

                for (int i = rem - 1; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += rem;
            }

            f128_s la = mul_by_double_print(fp, 10.0);
            int next = (int)la.hi;
            if (next < 0) next = 0; else if (next > 9) next = 9;
            f128_s remv = sub_by_double_print(la, (double)next);

            const int last_digit = frac[prec - 1] - '0';
            bool round_up = false;
            if (next > 5) round_up = true;
            else if (next < 5) round_up = false;
            else {
                const bool gt_half = (remv.hi > 0.0) || (remv.lo > 0.0);
                round_up = gt_half || ((last_digit & 1) != 0);
            }

            if (round_up) {
                int i = prec - 1;
                for (; i >= 0; --i) {
                    char& c = frac[i];
                    if (c == '9') c = '0';
                    else { c = char(c + 1); break; }
                }
                if (i < 0) {
                    ip = ip + f128_s{ 1.0 };
                    for (int j = 0; j < prec; ++j) frac[j] = '0';
                }
            }

            if (strip_trailing_zeros) {
                while (frac_len > 0 && frac[frac_len - 1] == '0') --frac_len;
            }
        }

        char int_rev[320];
        int int_len = emit_uint_rev_buf(int_rev, ip);

        if (neg && int_len == 1 && int_rev[0] == '0' && frac_len == 0) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        const size_t needed = (size_t)(neg ? 1 : 0) + (size_t)int_len + (frac_len ? (size_t)(1 + frac_len) : 0u);
        if ((size_t)(last - first) < needed) return { first, false };

        char* p = first;
        if (neg) *p++ = '-';

        for (int i = int_len - 1; i >= 0; --i) *p++ = int_rev[i];

        if (frac_len > 0) {
            *p++ = '.';
            std::memcpy(p, frac, (size_t)frac_len);
            p += frac_len;
        }

        return { p, true };
    }
    FORCE_INLINE f128_chars_result emit_scientific_sig_to_chars_f128(char* first, char* last, const f128_s& x, std::streamsize sig_digits, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        if (sig_digits < 1) sig_digits = 1;
        const bool neg = (x.hi < 0.0);
        const f128_s v = neg ? -x : x;
        const int sig = static_cast<int>(sig_digits);
        std::string digits;
        int e = 0;
        if (!_f128_detail::exact_scientific_digits(v, sig, digits, e)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        int last_frac = sig - 1;
        if (sig > 1 && strip_trailing_zeros) {
            while (last_frac >= 1 && digits[last_frac] == '0') --last_frac;
        }
        char exp_buf[16];
        char* ep = exp_buf;
        char* eend = exp_buf + sizeof(exp_buf);
        auto er = append_exp10_to_chars(ep, eend, e);
        if (!er.ok) return { first, false };
        const int exp_len = static_cast<int>(er.ptr - ep);
        const bool has_frac = (sig > 1) && (last_frac >= 1);
        const size_t needed = static_cast<size_t>(neg ? 1 : 0) + 1u + (has_frac ? static_cast<size_t>(1 + last_frac) : 0u) + static_cast<size_t>(exp_len);
        if (static_cast<size_t>(last - first) < needed) return { first, false };
        char* p = first;
        if (neg) *p++ = '-';
        *p++ = digits[0];
        if (has_frac) {
            *p++ = '.';
            std::memcpy(p, digits.data() + 1, static_cast<size_t>(last_frac));
            p += last_frac;
        }
        std::memcpy(p, exp_buf, static_cast<size_t>(exp_len));
        p += exp_len;
        return { p, true };
    }
    FORCE_INLINE f128_chars_result emit_scientific_to_chars(char* first, char* last, const f128_s& x, std::streamsize frac_digits, bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0) frac_digits = 0;
        if (iszero(x)) {
            const bool neg = _f128_detail::signbit_constexpr(x.hi);
            int frac_len = strip_trailing_zeros ? 0 : static_cast<int>(frac_digits);
            char exp_buf[16];
            char* ep = exp_buf;
            char* eend = exp_buf + sizeof(exp_buf);
            auto er = append_exp10_to_chars(ep, eend, 0);
            if (!er.ok) return { first, false };
            const int exp_len = static_cast<int>(er.ptr - ep);
            const size_t needed = static_cast<size_t>(neg ? 1 : 0) + 1u + (frac_len ? static_cast<size_t>(1 + frac_len) : 0u) + static_cast<size_t>(exp_len);
            if (static_cast<size_t>(last - first) < needed) return { first, false };
            char* p = first;
            if (neg) *p++ = '-';
            *p++ = '0';
            if (frac_len > 0) {
                *p++ = '.';
                for (int i = 0; i < frac_len; ++i) *p++ = '0';
            }
            std::memcpy(p, exp_buf, static_cast<size_t>(exp_len));
            p += exp_len;
            return { p, true };
        }
        return emit_scientific_sig_to_chars_f128(first, last, x, frac_digits + 1, strip_trailing_zeros);
    }
    FORCE_INLINE f128_chars_result to_chars(char* first, char* last, const f128_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false) noexcept
    {
        if (precision < 0) precision = 0;
        if (fixed && !scientific)
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        if (scientific && !fixed)
            return emit_scientific_to_chars(first, last, x, precision, strip_trailing_zeros);
        const int sig = (precision == 0) ? 1 : precision;
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        f128_s ax = (x.hi < 0.0) ? -x : x;
        f128_s m; int e10 = 0;
        normalize10(ax, m, e10);
        if (e10 >= -4 && e10 < sig) {
            const int frac = std::max(0, sig - (e10 + 1));
            return emit_fixed_dec_to_chars(first, last, x, frac, strip_trailing_zeros);
        }
        return emit_scientific_sig_to_chars_f128(first, last, x, sig, strip_trailing_zeros);
    }

    using f128_format_kind = fltx::common::format_kind;
    using f128_parse_token = fltx::common::parse_token<_f128_detail::biguint>;

    struct f128_io_traits
    {
        using value_type = f128_s;
        using chars_result = f128_chars_result;
        using parse_token = f128_parse_token;

        static constexpr int max_parse_order = 330;
        static constexpr int min_parse_order = -400;

        static bool isnan(const value_type& x) noexcept { return bl::isnan(x); }
        static bool isinf(const value_type& x) noexcept { return bl::isinf(x); }
        static bool iszero(const value_type& x) noexcept { return bl::iszero(x); }
        static bool is_negative(const value_type& x) noexcept { return x.hi < 0.0; }
        static value_type abs(const value_type& x) noexcept { return (x.hi < 0.0) ? -x : x; }
        static constexpr value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 }; }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }
        static void normalize10(const value_type& x, value_type& m, int& e10) { _f128_detail::normalize10(x, m, e10); }
        static chars_result to_chars_general(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return to_chars(first, last, x, precision, false, false, strip_trailing_zeros);
        }
        static chars_result to_chars_fixed(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_fixed_dec_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static chars_result to_chars_scientific_frac(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static chars_result to_chars_scientific_sig(char* first, char* last, const value_type& x, int precision, bool strip_trailing_zeros)
        {
            return emit_scientific_sig_to_chars_f128(first, last, x, precision, strip_trailing_zeros);
        }
        static constexpr value_type exact_decimal_to_value(const parse_token::coeff_type& coeff, int dec_exp, bool neg)
        {
            return _f128_detail::exact_decimal_to_f128(coeff, dec_exp, neg);
        }
    };

    template<typename Writer>
    FORCE_INLINE void write_chars_to_string_f128(std::string& out, std::size_t cap, Writer writer)
    {
        fltx::common::write_chars_to_string<f128_chars_result>(out, cap, writer);
    }
    FORCE_INLINE const char* special_text_f128(const f128_s& x, bool uppercase = false) noexcept
    {
        return fltx::common::special_text<f128_io_traits>(x, uppercase);
    }
    FORCE_INLINE bool assign_special_string_f128(std::string& out, const f128_s& x, bool uppercase = false) noexcept
    {
        return fltx::common::assign_special_string<f128_io_traits>(out, x, uppercase);
    }
    FORCE_INLINE void ensure_decimal_point_f128(std::string& s)
    {
        fltx::common::ensure_decimal_point(s);
    }
    FORCE_INLINE void apply_stream_decorations_f128(std::string& s, bool showpos, bool uppercase)
    {
        fltx::common::apply_stream_decorations(s, showpos, uppercase);
    }
    FORCE_INLINE bool write_stream_special_f128(std::ostream& os, const f128_s& x, bool showpos, bool uppercase)
    {
        return fltx::common::write_stream_special<f128_io_traits>(os, x, showpos, uppercase);
    }
    FORCE_INLINE void format_to_string_f128(std::string& out, const f128_s& x, int precision, f128_format_kind kind, bool strip_trailing_zeros = false)
    {
        fltx::common::format_to_string<f128_io_traits>(out, x, precision, kind, strip_trailing_zeros);
    }
    FORCE_INLINE void to_string_into(std::string& out, const f128_s& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
    {
        fltx::common::to_string_into<f128_io_traits>(out, x, precision, fixed, scientific, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific(std::string& os, const f128_s& x, std::streamsize prec, bool strip_trailing_zeros)
    {
        fltx::common::emit_scientific<f128_io_traits>(os, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_fixed_dec(std::string& os, f128_s x, int prec, bool strip_trailing_zeros)
    {
        fltx::common::emit_fixed_dec<f128_io_traits>(os, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific_sig_f128(std::string& os, const f128_s& x, std::streamsize sig_digits, bool strip_trailing_zeros)
    {
        fltx::common::emit_scientific_sig<f128_io_traits>(os, x, sig_digits, strip_trailing_zeros);
    }

    /// ======== Parsing helpers ========

    FORCE_INLINE bool valid_flt128_string(const char* s) noexcept
    {
        return fltx::common::valid_float_string(s);
    }
    FORCE_INLINE unsigned char ascii_lower_f128(char c) noexcept
    {
        return fltx::common::ascii_lower(c);
    }
    FORCE_INLINE const char* skip_ascii_space_f128(const char* p) noexcept
    {
        return fltx::common::skip_ascii_space(p);
    }

}

FORCE_INLINE constexpr bool parse_flt128(const char* s, f128_s& out, const char** endptr = nullptr) noexcept
{
    return fltx::common::parse_flt<_f128_detail::f128_io_traits>(s, out, endptr);
}
[[nodiscard]] FORCE_INLINE constexpr f128_s to_f128(const char* s) noexcept
{
    f128_s ret;
    if (parse_flt128(s, ret))
        return ret;
    return f128_s{ 0 };
}
[[nodiscard]] FORCE_INLINE constexpr f128_s to_f128(const std::string& s) noexcept
{
    return to_f128(s.c_str());
}
[[nodiscard]] FORCE_INLINE std::string to_string(const f128_s& x, int precision = std::numeric_limits<f128_s>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    std::string out;
    _f128_detail::to_string_into(out, x, precision, fixed, scientific, strip_trailing_zeros);
    return out;
}

/// ======== Stream output ========

FORCE_INLINE std::ostream& operator<<(std::ostream& os, const f128_s& x)
{
    return fltx::common::write_to_stream<_f128_detail::f128_io_traits>(os, x);
}

/// ======== Literals ========
namespace literals
{
    [[nodiscard]] constexpr f128_s operator""_dd(unsigned long long v) noexcept {
        return to_f128(static_cast<uint64_t>(v));
    }
    [[nodiscard]] constexpr f128_s operator""_dd(long double v) noexcept {
        return f128_s{ static_cast<double>(v) };
    }
    [[nodiscard]] consteval f128_s operator""_dd(const char* text, std::size_t len) noexcept
    {
        f128_s out{};
        const char* end = text;
        if (!(parse_flt128(text, out, &end) && (static_cast<std::size_t>(end - text) == len)))
            throw "invalid _dd literal";
        return out;
    }
}
#define DD(x) bl::to_f128(#x)

} // namespace bl
