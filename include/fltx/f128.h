#pragma once
#define F128_INCLUDED
#include "fltx_common.h"

namespace bl {

struct f128;
struct f256;

namespace _f128_detail
{
    using fltx_common::fp::absd;
    using fltx_common::fp::isnan;
    using fltx_common::fp::isinf;
    using fltx_common::fp::isfinite;
    using fltx_common::fp::magnitude_u64;
    using fltx_common::fp::split_uint64_to_doubles;
    using fltx_common::fp::two_prod_precise;
    using fltx_common::fp::two_prod_precise_dekker;
    using fltx_common::fp::two_sum_precise;
}

constexpr f128 operator+(const f128& a, const f128& b);
constexpr f128 operator-(const f128& a, const f128& b);
constexpr f128 operator+(const f128& a, double b);
constexpr f128 operator-(const f128& a, double b);

CONSTEXPR_NO_FMA f128 operator*(const f128& a, const f128& b);
CONSTEXPR_NO_FMA f128 operator/(const f128& a, const f128& b);
CONSTEXPR_NO_FMA f128 operator*(const f128& a, double b);
CONSTEXPR_NO_FMA f128 operator/(const f128& a, double b);

struct f128
{
    double hi; // leading component
    double lo; // trailing error
    
    FORCE_INLINE CONSTEXPR_NO_FMA f128& operator*=(f128 rhs) { *this = *this * rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f128& operator/=(f128 rhs) { *this = *this / rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f128& operator*=(double rhs) { *this = *this * rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f128& operator/=(double rhs) { *this = *this / rhs; return *this; }

    FORCE_INLINE constexpr f128& operator=(f256 x) noexcept;
    FORCE_INLINE constexpr f128& operator=(double x) noexcept {
        hi = x; lo = 0.0; return *this;
    }
    FORCE_INLINE constexpr f128& operator=(float x) noexcept {
        hi = static_cast<double>(x); lo = 0.0; return *this;
    }

    FORCE_INLINE constexpr f128& operator=(uint64_t u) noexcept;
    FORCE_INLINE constexpr f128& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f128& operator=(T v) noexcept {
        return (*this = static_cast<int64_t>(v));
    }
    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f128& operator=(T v) noexcept {
        return (*this = static_cast<uint64_t>(v));
    }

    FORCE_INLINE constexpr f128& operator+=(f128 rhs) { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f128& operator-=(f128 rhs) { *this = *this - rhs; return *this; }
    FORCE_INLINE constexpr f128& operator+=(double rhs) { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f128& operator-=(double rhs) { *this = *this - rhs; return *this; }


    /// ======== Conversions ========
    explicit constexpr operator f256() const noexcept;
    explicit constexpr operator double() const noexcept { return hi + lo; }
    explicit constexpr operator float() const noexcept { return static_cast<float>(hi + lo); }
    explicit constexpr operator int() const noexcept { return static_cast<int>(hi + lo); }

    constexpr f128 operator+() const { return *this; }
    constexpr f128 operator-() const noexcept { return f128{ -hi, -lo }; }

    /// ======== Utility ========
    static constexpr f128 eps() { return { 1.232595164407831e-32, 0.0 }; }
};

struct f128_t : public f128
{
    f128_t() = default;
    constexpr f128_t(double _hi, double _lo) noexcept : f128{ _hi, _lo } {}
    constexpr f128_t(float  x) noexcept : f128{ ((double)x), 0.0 } {}
    constexpr f128_t(double x) noexcept : f128{ ((double)x), 0.0 } {}
    constexpr f128_t(int64_t  v) noexcept : f128{} { static_cast<f128&>(*this) = static_cast<int64_t>(v); }
    constexpr f128_t(uint64_t u) noexcept : f128{} { static_cast<f128&>(*this) = static_cast<uint64_t>(u); }
    constexpr f128_t(int32_t  v) noexcept : f128_t((int64_t)v) {}
    constexpr f128_t(uint32_t u) noexcept : f128_t((int64_t)u) {}
    constexpr f128_t(const f128& f) noexcept : f128{ f.hi, f.lo } {}
    inline operator f128& () { return static_cast<f128&>(*this); }
};

} // end bl

namespace std
{
    using bl::f128;

    template<> struct numeric_limits<f128>
    {
        static constexpr bool is_specialized = true;

        // limits
        static constexpr f128 min()            noexcept { return {  numeric_limits<double>::min(), 0.0 }; }
        static constexpr f128 max()            noexcept { return {  numeric_limits<double>::max(), -numeric_limits<double>::epsilon() }; }
        static constexpr f128 lowest()         noexcept { return { -numeric_limits<double>::max(),  numeric_limits<double>::epsilon() }; }
        static constexpr f128 highest()        noexcept { return {  numeric_limits<double>::max(), -numeric_limits<double>::epsilon() }; }
                                                 
		// special values                        
        static constexpr f128 epsilon()        noexcept { return { 1.232595164407831e-32, 0.0 }; } // ~2^-106, a single ulp of double-double
        static constexpr f128 round_error()    noexcept { return { 0.5, 0.0 }; }
        static constexpr f128 infinity()       noexcept { return { numeric_limits<double>::infinity(), 0.0 }; }
        static constexpr f128 quiet_NaN()      noexcept { return { numeric_limits<double>::quiet_NaN(), 0.0 }; }
        static constexpr f128 signaling_NaN()  noexcept { return { numeric_limits<double>::signaling_NaN(), 0.0 }; }
        static constexpr f128 denorm_min()     noexcept { return { numeric_limits<double>::denorm_min(), 0.0 }; }
                                                 
        static constexpr bool has_infinity       = true;
        static constexpr bool has_quiet_NaN      = true;
        static constexpr bool has_signaling_NaN  = true;
                                                 
		// properties                            
        static constexpr int  digits             = 106;  // ~53 bits * 2
        static constexpr int  digits10           = 31;   // log10(2^106) ≈ 31.9
        static constexpr int  max_digits10       = 33;
        static constexpr bool is_signed          = true;
        static constexpr bool is_integer         = false;
        static constexpr bool is_exact           = false;
        static constexpr int  radix              = 2;
                                                 
		// exponent range                        
        static constexpr int  min_exponent       = numeric_limits<double>::min_exponent;
        static constexpr int  max_exponent       = numeric_limits<double>::max_exponent;
        static constexpr int  min_exponent10     = numeric_limits<double>::min_exponent10;
        static constexpr int  max_exponent10     = numeric_limits<double>::max_exponent10;
                                                 
		// properties                            
        static constexpr bool is_iec559          = false; // not IEEE-754 compliant
        static constexpr bool is_bounded         = true;
        static constexpr bool is_modulo          = false;
                                                 
		// rounding                              
        static constexpr bool traps              = false;
        static constexpr bool tinyness_before    = false;

        static constexpr float_round_style round_style = round_to_nearest;
    };
}

namespace bl {

/// ======== Representation helpers and scalar conversions ========

namespace _f128_detail
{
    FORCE_INLINE constexpr f128 renorm(double hi, double lo)
    {
        double s{}, e{};
        _f128_detail::two_sum_precise(hi, lo, s, e);
        return { s, e };
    }
}

FORCE_INLINE constexpr f128& f128::operator=(uint64_t u) noexcept {
    // same limb path you already use in f128(u)
    double a{}, b{};
    _f128_detail::split_uint64_to_doubles(u, a, b);
    double s{}, e{}; _f128_detail::two_sum_precise(a, b, s, e);
    f128 r = _f128_detail::renorm(s, e);
    hi = r.hi; lo = r.lo; return *this;
}
FORCE_INLINE constexpr f128& f128::operator=(int64_t v) noexcept {
    uint64_t u = _f128_detail::magnitude_u64(v);
    f128 r{}; r = u;                       // reuse uint64_t path
    if (v < 0) { r.hi = -r.hi; r.lo = -r.lo; }
    hi = r.hi; lo = r.lo; return *this;
}

constexpr f128 to_f128(double x) noexcept { return f128{ x, 0.0 }; }
constexpr f128 to_f128(float x) noexcept { return f128{ (double)x, 0.0 }; }
constexpr f128 to_f128(int32_t v) noexcept { return f128{ (double)v, 0.0 }; }
constexpr f128 to_f128(uint32_t v) noexcept { return f128{ (double)v, 0.0 }; }
constexpr f128 to_f128(int64_t v) noexcept {
    uint64_t u = _f128_detail::magnitude_u64(v);
    f128 r{}; r = u; // reuse uint64_t path
    if (v < 0) { r.hi = -r.hi; r.lo = -r.lo; }
    return r;
}
constexpr f128 to_f128(uint64_t u)  noexcept {
    double a{}, b{};
    _f128_detail::split_uint64_to_doubles(u, a, b);
    double s{}, e{}; _f128_detail::two_sum_precise(a, b, s, e);
    return _f128_detail::renorm(s, e);
}

/// ======== Comparisons ========

// ------------------ f128 <=> f128 ------------------

FORCE_INLINE constexpr bool operator <(const f128& a, const f128& b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo); }
FORCE_INLINE constexpr bool operator >(const f128& a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, const f128& b) { return (a < b) || (a.hi == b.hi && a.lo == b.lo); }
FORCE_INLINE constexpr bool operator>=(const f128& a, const f128& b) { return b <= a; }
FORCE_INLINE constexpr bool operator==(const f128& a, const f128& b) { return a.hi == b.hi && a.lo == b.lo; }
FORCE_INLINE constexpr bool operator!=(const f128& a, const f128& b) { return !(a == b); }

/// ------------------ double <=> f128 ------------------

FORCE_INLINE constexpr bool operator<(const f128& a, double b)  { return a < f128{b}; }
FORCE_INLINE constexpr bool operator<(double a, const f128& b)  { return f128{a} < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, double b)  { return b < a; }
FORCE_INLINE constexpr bool operator>(double a, const f128& b)  { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, double b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(double a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, double b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(double a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, double b) { return a == f128{b}; }
FORCE_INLINE constexpr bool operator==(double a, const f128& b) { return f128{a} == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, double b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(double a, const f128& b) { return !(a == b); }

/// ------------------ float <=> f128 ------------------

FORCE_INLINE constexpr bool operator<(const f128& a, float b) { return a < f128{b}; }
FORCE_INLINE constexpr bool operator<(float a, const f128& b) { return f128{a} < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, float b) { return b < a; }
FORCE_INLINE constexpr bool operator>(float a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, float b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(float a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, float b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(float a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, float b) { return a == f128{b}; }
FORCE_INLINE constexpr bool operator==(float a, const f128& b) { return f128{a} == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, float b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(float a, const f128& b) { return !(a == b); }

// --------------- ints <=> f128 ---------------

FORCE_INLINE constexpr bool operator<(const f128& a, int32_t b) { return a < to_f128(b); }
FORCE_INLINE constexpr bool operator<(int32_t a, const f128& b) { return to_f128(a) < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, int32_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(int32_t a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, int32_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(int32_t a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, int32_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(int32_t a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, int32_t b) { return a == to_f128(b); }
FORCE_INLINE constexpr bool operator==(int32_t a, const f128& b) { return to_f128(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, int32_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(int32_t a, const f128& b) { return !(a == b); }

FORCE_INLINE constexpr bool operator<(const f128& a, uint32_t b) { return a < to_f128(b); }
FORCE_INLINE constexpr bool operator<(uint32_t a, const f128& b) { return to_f128(a) < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, uint32_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(uint32_t a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, uint32_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(uint32_t a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, uint32_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(uint32_t a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, uint32_t b) { return a == to_f128(b); }
FORCE_INLINE constexpr bool operator==(uint32_t a, const f128& b) { return to_f128(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, uint32_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(uint32_t a, const f128& b) { return !(a == b); }

/// ------------------ int64_t/uint64_t <=> f128 ------------------

FORCE_INLINE constexpr bool operator<(const f128& a, int64_t b) { return a < to_f128(b); }
FORCE_INLINE constexpr bool operator<(int64_t a, const f128& b) { return to_f128(a) < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, int64_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(int64_t a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, int64_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(int64_t a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, int64_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(int64_t a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, int64_t b) { return a == to_f128(b); }
FORCE_INLINE constexpr bool operator==(int64_t a, const f128& b) { return to_f128(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, int64_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(int64_t a, const f128& b) { return !(a == b); }

FORCE_INLINE constexpr bool operator<(const f128& a, uint64_t b) { return a < to_f128(b); }
FORCE_INLINE constexpr bool operator<(uint64_t a, const f128& b) { return to_f128(a) < b; }
FORCE_INLINE constexpr bool operator>(const f128& a, uint64_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(uint64_t a, const f128& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f128& a, uint64_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(uint64_t a, const f128& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f128& a, uint64_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(uint64_t a, const f128& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f128& a, uint64_t b) { return a == to_f128(b); }
FORCE_INLINE constexpr bool operator==(uint64_t a, const f128& b) { return to_f128(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f128& a, uint64_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(uint64_t a, const f128& b) { return !(a == b); }


/// ======== Arithmetic operators ========

namespace _f128_detail
{
    BL_PUSH_PRECISE
    FORCE_INLINE constexpr f128 quick_two_sum(double a, double b)
    {
        double s = a + b;
        double err = b - (s - a);
        return { s, err };
    }
    BL_POP_PRECISE
}

FORCE_INLINE CONSTEXPR_NO_FMA f128 recip(f128 b)
{
    constexpr f128 one = f128{ 1.0 };
    f128 y = f128{ 1.0 / b.hi };
    f128 e = one - b * y;

    y += y * e;
    e = one - b * y;
    y += y * e;

    return y;
}

/// ------------------ f128 <=> f128 ------------------

BL_PUSH_PRECISE
FORCE_INLINE constexpr f128 operator+(const f128& a, const f128& b)
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

FORCE_INLINE constexpr f128 operator-(const f128& a, const f128& b)
{
    return a + f128{ -b.hi, -b.lo };
}
FORCE_INLINE CONSTEXPR_NO_FMA f128 operator*(const f128& a, const f128& b)
{
    double p, e;
#ifdef FMA_AVAILABLE
    _f128_detail::two_prod_precise(a.hi, b.hi, p, e);   // p ≈ a.hi*b.hi, e = exact error

    e = fltx_common::fp::fma1(a.hi, b.lo, e);
    e = fltx_common::fp::fma1(a.lo, b.hi, e);
    e = fltx_common::fp::fma1(a.lo, b.lo, e);

    return _f128_detail::renorm(p, e);
#else
    _f128_detail::two_prod_precise(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;

    return _f128_detail::quick_two_sum(p, e);
#endif
}
FORCE_INLINE CONSTEXPR_NO_FMA f128 operator/(const f128& a, const f128& b)
{
    if (b.lo == 0.0)
        return a / b.hi;

    const double inv_b0 = 1.0 / b.hi;

    const double q0 = a.hi * inv_b0;
    f128 r = a - b * q0;

    const double q1 = r.hi * inv_b0;

    return _f128_detail::renorm(q0, q1);
}
FORCE_INLINE CONSTEXPR_NO_FMA f128 operator/(const f128& a, double b)
{
    const double q0 = a.hi / b;

    double p0, p1;
    bl::fltx_common::fp::two_prod_precise(b, q0, p0, p1);

    const f128 r = a - f128{ p0, p1 };
    const double q1 = r.hi / b;

    return _f128_detail::renorm(q0, q1);
}

/// ------------------ double <=> f128 ------------------

FORCE_INLINE CONSTEXPR_NO_FMA f128 operator*(const f128& a, double b) { return a * f128{ b }; }
FORCE_INLINE CONSTEXPR_NO_FMA f128 operator*(double a, const f128& b) { return f128{ a } * b; }
//FORCE_INLINE CONSTEXPR_NO_FMA f128 operator/(const f128& a, double b) { return a / f128{ b }; }
//FORCE_INLINE CONSTEXPR_NO_FMA f128 operator/(double a, const f128& b) { return f128{ a } / b; }
FORCE_INLINE constexpr f128 operator+(const f128& a, double b) { return a + f128{ b }; }
FORCE_INLINE constexpr f128 operator+(double a, const f128& b) { return f128{ a } + b; }
FORCE_INLINE constexpr f128 operator-(const f128& a, double b) { return a - f128{ b }; }
FORCE_INLINE constexpr f128 operator-(double a, const f128& b) { return f128{ a } - b; }

/// ======== Additional math functions ========

FORCE_INLINE constexpr bool isnan(const f128& x) noexcept { return _f128_detail::isnan(x.hi); }
FORCE_INLINE constexpr bool isinf(const f128& x) noexcept { return _f128_detail::isinf(x.hi); }
FORCE_INLINE constexpr bool isfinite(const f128& x) noexcept { return _f128_detail::isfinite(x.hi); }
FORCE_INLINE constexpr bool iszero(const f128& x) noexcept { return x.hi == 0.0 && x.lo == 0.0; }
FORCE_INLINE constexpr bool ispositive(const f128& x)
{
    return x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0);
}

FORCE_INLINE constexpr f128 clamp(const f128& v, const f128& lo, const f128& hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
FORCE_INLINE constexpr f128 abs(const f128& a)
{
    return (a.hi < 0.0) ? -a : a;
}
FORCE_INLINE f128 floor(const f128& a)
{
    f128 r{ std::floor(a.hi), 0.0 };
    if (r > a) r -= 1.0;
    return r;
}
FORCE_INLINE f128 ceil(const f128& a)
{
    f128 r{ std::ceil(a.hi), 0.0 };
    if (r < a) r += 1.0;
    return r;
}
FORCE_INLINE f128 trunc(const f128& a)
{
    return (a.hi < 0.0) ? ceil(a) : floor(a);
}
FORCE_INLINE f128 fmod(const f128& x, const f128& y)
{
    if (y.hi == 0.0 && y.lo == 0.0)
        return std::numeric_limits<f128>::quiet_NaN();
    return x - trunc(x / y) * y;
}
FORCE_INLINE f128 round(const f128& a)
{
    f128 t = floor(a + f128{ 0.5 });
    if ((t - a) == f128{ 0.5 } && fmod(t, f128{ 2.0 }) != f128{ 0.0 })
        t -= f128{ 1.0 };
    return t;
}
inline f128 round_to_decimals(f128 v, int prec)
{
    if (prec <= 0) return v;

    static constexpr f128 INV10_DD{
        0.1000000000000000055511151231257827021181583404541015625,  // hi (double rounded)
       -0.0000000000000000055511151231257827021181583404541015625   // lo = 0.1 - hi
    };

    // Sign
    const bool neg = v < 0.0;
    if (neg) v = -v;

    // Split
    f128 ip = floor(v);
    f128 frac = v - ip;

    // Extract digits with one look-ahead
    std::string dig; dig.reserve((size_t)prec);
    f128 w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;
        int di = (int)floor(w).hi;
        if (di < 0) di = 0; else if (di > 9) di = 9;
        dig.push_back(char('0' + di));
        w = w - f128{ (double)di };
    }

    // Look-ahead digit
    f128 la = w * 10.0;
    int next = (int)floor(la).hi;
    if (next < 0) next = 0; else if (next > 9) next = 9;
    f128 rem = la - f128{ (double)next };

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
        if (i < 0) ip = ip + 1;
    }

    // Rebuild fractional value backward
    f128 frac_val{ 0.0, 0.0 };
    for (int i = prec - 1; i >= 0; --i) {
        frac_val = frac_val + f128{ (double)(dig[(size_t)i] - '0') };
        frac_val = frac_val * INV10_DD;
    }

    f128 out = ip + frac_val;
    return neg ? -out : out;
}

FORCE_INLINE f128 remainder(const f128& x, const f128& y)
{
    // Domain checks (match std::remainder)
    if (isnan(x) || isnan(y)) return std::numeric_limits<f128>::quiet_NaN();
    if (iszero(y))            return std::numeric_limits<f128>::quiet_NaN();
    if (isinf(x))             return std::numeric_limits<f128>::quiet_NaN();
    if (isinf(y))             return x;

    // n = nearest integer to q = x/y, ties to even
    const f128 q = x / y;
    f128 n = trunc(q);
    f128 rfrac = q - n;                // fractional part with sign of q
    const f128 half = f128{0.5};
    const f128 one{ 1 };

    if (abs(rfrac) > half) {
        n += (rfrac.hi >= 0.0 ? one : -one);
    }
    else if (abs(rfrac) == half) {
        // tie: choose even n
        const f128 n_mod2 = fmod(n, f128{ 2 });
        if (n_mod2 != 0)
            n += (rfrac.hi >= 0.0 ? one : -one);
    }

    f128 r = x - n * y;

    // If result is zero, sign should match x (std::remainder semantics)
    if (iszero(r))
        return f128{ std::signbit(x.hi) ? -0.0 : 0.0 };

    return r;
}
FORCE_INLINE f128 sqrt(f128 a)
{
    // Match std semantics for negative / zero quickly.
    if (a.hi <= 0.0)
    {
        if (a.hi == 0.0 && a.lo == 0.0) return f128{ 0.0 };
        return f128{ std::numeric_limits<double>::quiet_NaN() };
    }

    double y0 = std::sqrt(a.hi);
    f128 y{ y0 };

    // Two Newton refinements
    y = y + (a - y * y) / (y + y);
    y = y + (a - y * y) / (y + y);
    return y;
}
FORCE_INLINE f128 nearbyint(const f128& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    f128 t = floor(a);
    f128 frac = a - t;

    if (frac < f128{ 0.5 })
        return t;

    if (frac > f128{ 0.5 })
    {
        t += f128{ 1.0 };
        if (iszero(t))
            return f128{ std::signbit(a.hi) ? -0.0 : 0.0 };
        return t;
    }

    if (fmod(t, f128{ 2.0 }) != f128{ 0.0 })
        t += f128{ 1.0 };

    if (iszero(t))
        return f128{ std::signbit(a.hi) ? -0.0 : 0.0 };

    return t;
}

/// ======== Transcendental functions ========
namespace _f128_const
{
    // high-precision constants
    inline constexpr f128 pi      = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53 };
    inline constexpr f128 pi_2    = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54 };
    inline constexpr f128 pi_4    = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55 };
    inline constexpr f128 invpi2  = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55 };
    inline constexpr f128 ln2     = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56 };
    inline constexpr f128 inv_ln2 = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56 };
    inline constexpr f128 ln10    = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53 };
}
namespace _f128_detail
{
    inline constexpr double pi_4_hi = 0x1.921fb54442d18p-1;

    FORCE_INLINE f128 f128_exp_kernel_ln2_half(const f128& r)
    {
        f128 p = f128{ 8.89679139245057408e-22 };
        p = p * r + f128{ 1.95729410633912626e-20 };
        p = p * r + f128{ 4.11031762331216484e-19 };
        p = p * r + f128{ 8.22063524662432950e-18 };
        p = p * r + f128{ 1.56192069685862253e-16 };
        p = p * r + f128{ 2.81145725434552060e-15 };
        p = p * r + f128{ 4.77947733238738525e-14 };
        p = p * r + f128{ 7.64716373181981641e-13 };
        p = p * r + f128{ 1.14707455977297245e-11 };
        p = p * r + f128{ 1.60590438368216133e-10 };
        p = p * r + f128{ 2.08767569878681002e-09 };
        p = p * r + f128{ 2.50521083854417202e-08 };
        p = p * r + f128{ 2.75573192239858883e-07 };
        p = p * r + f128{ 2.75573192239858925e-06 };
        p = p * r + f128{ 2.48015873015873016e-05 };
        p = p * r + f128{ 1.98412698412698413e-04 };
        p = p * r + f128{ 1.38888888888888894e-03 };
        p = p * r + f128{ 8.33333333333333322e-03 };
        p = p * r + f128{ 4.16666666666666644e-02 };
        p = p * r + f128{ 1.66666666666666657e-01 };
        p = p * r + f128{ 5.00000000000000000e-01 };
        p = p * r + f128{ 1.0 };
        return (p * r) + f128{ 1.0 };
    }

    FORCE_INLINE bool f128_rem_pio2(const f128& x, long long& n_out, f128& r_out)
    {
        const double ax = std::fabs(x.hi);
        if (!std::isfinite(ax))
            return false;

        if (ax > 7.0e15)
            return false;

        const f128 t = x * _f128_const::invpi2;

        double qd = std::nearbyint(t.hi);
        if (!std::isfinite(qd) ||
            qd < (double)std::numeric_limits<long long>::min() ||
            qd >(double)std::numeric_limits<long long>::max())
            return false;

        const f128 delta = t - f128{ qd };
        if (delta.hi > 0.5 || (delta.hi == 0.5 && delta.lo > 0.0))
            qd += 1.0;
        else if (delta.hi < -0.5 || (delta.hi == -0.5 && delta.lo < 0.0))
            qd -= 1.0;

        if (qd < (double)std::numeric_limits<long long>::min() ||
            qd >(double)std::numeric_limits<long long>::max())
            return false;

        long long n = (long long)qd;
        f128 r = x - f128{ qd } * _f128_const::pi_2;

        if (r > f128{ _f128_detail::pi_4_hi })
        {
            r -= _f128_const::pi_2;
            ++n;
        }
        else if (r < f128{ -_f128_detail::pi_4_hi })
        {
            r += _f128_const::pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }
    FORCE_INLINE void f128_sincos_kernel_pio4(const f128& x, f128& s_out, f128& c_out)
    {
        f128 t = x * x;

        f128 ps = f128{ 1.13099628864477159e-31,  1.04980154129595057e-47 };
        ps = ps * t + f128{ -9.18368986379554615e-29, -1.43031503967873224e-45 };
        ps = ps * t + f128{ 6.44695028438447391e-26, -1.93304042337034642e-42 };
        ps = ps * t + f128{ -3.86817017063068404e-23,  8.84317765548234382e-40 };
        ps = ps * t + f128{ 1.95729410633912625e-20, -1.36435038300879076e-36 };
        ps = ps * t + f128{ -8.22063524662432972e-18, -2.21418941196042654e-34 };
        ps = ps * t + f128{ 2.81145725434552060e-15,  1.65088427308614330e-31 };
        ps = ps * t + f128{ -7.64716373181981648e-13, -7.03872877733452971e-30 };
        ps = ps * t + f128{ 1.60590438368216146e-10,  1.25852945887520981e-26 };
        ps = ps * t + f128{ -2.50521083854417188e-08,  1.44881407093591197e-24 };
        ps = ps * t + f128{ 2.75573192239858907e-06, -1.85839327404647208e-22 };
        ps = ps * t + f128{ -1.98412698412698413e-04, -1.72095582934207053e-22 };
        ps = ps * t + f128{ 8.33333333333333322e-03,  1.15648231731787140e-19 };
        ps = ps * t + f128{ -1.66666666666666657e-01, -9.25185853854297066e-18 };
        s_out = x + x * t * ps;

        f128 pc = f128{ 3.27988923706983791e-30,  1.51175427440298786e-46 };
        pc = pc * t + f128{ -2.47959626322479746e-27,  1.29537309647652292e-43 };
        pc = pc * t + f128{ 1.61173757109611835e-24, -3.68465735645097656e-41 };
        pc = pc * t + f128{ -8.89679139245057329e-22,  7.91140261487237594e-38 };
        pc = pc * t + f128{ 4.11031762331216486e-19,  1.44129733786595266e-36 };
        pc = pc * t + f128{ -1.56192069685862265e-16, -1.19106796602737541e-32 };
        pc = pc * t + f128{ 4.77947733238738530e-14,  4.39920548583408094e-31 };
        pc = pc * t + f128{ -1.14707455977297247e-11, -2.06555127528307454e-28 };
        pc = pc * t + f128{ 2.08767569878680990e-09, -1.20734505911325997e-25 };
        pc = pc * t + f128{ -2.75573192239858907e-07, -2.37677146222502973e-23 };
        pc = pc * t + f128{ 2.48015873015873016e-05,  2.15119478667758816e-23 };
        pc = pc * t + f128{ -1.38888888888888894e-03,  5.30054395437357706e-20 };
        pc = pc * t + f128{ 4.16666666666666644e-02,  2.31296463463574269e-18 };
        pc = pc * t + f128{ -5.00000000000000000e-01,  0.0 };
        c_out = f128{ 1.0 } + t * pc;
    }
}

// exp
FORCE_INLINE f128 ldexp(const f128& x, int e)
{
    return f128{ std::ldexp(x.hi, e), std::ldexp(x.lo, e) };
}
FORCE_INLINE f128 exp(const f128& x)
{
    if (!std::isfinite(x.hi))
        return f128{ std::exp(x.hi) };

    if (x.hi > 709.782712893384)
        return f128{ std::numeric_limits<double>::max() };

    if (x.hi < -745.133219101941)
        return f128{ 0.0 };

    const f128 t = x * _f128_const::inv_ln2;
    const double kd = std::nearbyint((double)t);

    if (!std::isfinite(kd) || std::fabs(kd) > 9.0e15)
        return f128{ std::exp((double)x) };

    const int k = (int)kd;
    const f128 r = x - f128{ kd } * _f128_const::ln2;

    if (std::fabs(r.hi) > 0.40)
        return f128{ std::exp((double)x) };

    return ldexp(_f128_detail::f128_exp_kernel_ln2_half(r), k);
}
FORCE_INLINE f128 exp2(const f128& x)
{
    return exp(x * _f128_const::ln2);
}

// log
FORCE_INLINE double log_as_double(f128 a)
{
    return std::log(a.hi) + std::log1p(a.lo / a.hi);
}
FORCE_INLINE f128 log(const f128& a)
{
    double log_hi = std::log(a.hi); // first guess
    f128 exp_log_hi{ std::exp(log_hi) }; // exp(guess)
    f128 r = (a - exp_log_hi) / exp_log_hi; // (a - e^g) / e^g ≈ error
    return f128{ log_hi } + r; // refined log
}
FORCE_INLINE f128 log2(const f128& a)
{
    constexpr f128 LOG2_RECIPROCAL{ 1.442695040888963407359924681001892137 }; // 1/log(2)
    return log(a) * LOG2_RECIPROCAL;
}
FORCE_INLINE f128 log10(const f128& x)
{
    // 1 / ln(10) to full-double accuracy
    static constexpr double INV_LN10 = 0.434294481903251827651128918916605082;
    return log(x) * f128 { INV_LN10 };   // log(x) already returns float128
}

// pow
FORCE_INLINE f128 pow(const f128& x, const f128& y) { return exp(y * log(x)); }
FORCE_INLINE f128 pow10_128(int k)
{
    if (k == 0) return f128{ 1.0 };

    int n = (k >= 0) ? k : -k;

    // fast small-exponent path
    if (n <= 16) {
        f128 r = f128{ 1.0 };
        const f128 ten = f128{ 10.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f128{ 1.0 } / r);
    }

    f128 r = f128{ 1.0 };
    f128 base = f128{ 10.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f128{ 1.0 } / r);
}

// trig
inline bool sincos(const f128& x, f128& s_out, f128& c_out)
{
    const double ax = std::fabs(x.hi);
    if (!std::isfinite(ax))
    {
        s_out = f128{ std::numeric_limits<double>::quiet_NaN() };
        c_out = s_out;
        return false;
    }

    if (ax <= _f128_detail::pi_4_hi)
    {
        _f128_detail::f128_sincos_kernel_pio4(x, s_out, c_out);
        return true;
    }

    long long n = 0;
    f128 r{};
    if (!_f128_detail::f128_rem_pio2(x, n, r))
        return false;

    f128 sr{}, cr{};
    _f128_detail::f128_sincos_kernel_pio4(r, sr, cr);

    switch ((int)(n & 3))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    return true;
}
inline f128 sin(const f128& x)
{
    f128 s{}, c{};
    if (sincos(x, s, c))
        return s;
    return f128{ std::sin((double)x) };
}
inline f128 cos(const f128& x)
{
    f128 s{}, c{};
    if (sincos(x, s, c))
        return c;
    return f128{ std::cos((double)x) };
}
inline f128 tan(const f128& x)
{
    f128 s{}, c{};
    if (sincos(x, s, c))
        return s / c;
    const double xd = (double)x;
    return f128{ std::tan(xd) };
}
inline f128 atan2(const f128& y, const f128& x)
{
    if (iszero(x))
    {
        if (iszero(y))
            return f128{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? _f128_const::pi_2 : -_f128_const::pi_2;
    }

    const f128 scale = std::max(abs(x), abs(y));
    const f128 xs = x / scale;
    const f128 ys = y / scale;

    f128 v{ std::atan2(y.hi, x.hi) };

    for (int i = 0; i < 2; ++i)
    {
        f128 sv{}, cv{};
        if (!sincos(v, sv, cv))
        {
            const double vd = (double)v;
            sv = f128{ std::sin(vd) };
            cv = f128{ std::cos(vd) };
        }

        const f128 f = xs * sv - ys * cv;
        const f128 fp = xs * cv + ys * sv;

        v = v - f / fp;
    }

    return v;
}
inline f128 atan(const f128& x)
{
    return atan2(x, f128{ 1.0 });
}
inline f128 asin(const f128& x)
{
    return atan2(x, sqrt(f128{ 1.0 } - x * x));
}
inline f128 acos(const f128& x)
{
    return atan2(sqrt(f128{ 1.0 } - x * x), x);
}

/// ======== Public string conversion wrappers ========

namespace _f128_detail
{
    FORCE_INLINE void normalize10(const f128& x, f128& m, int& exp10)
    {
        if (x.hi == 0.0 && x.lo == 0.0) { m = f128{0.0}; exp10 = 0; return; }

        f128 ax = abs(x);

        int e2 = 0;
        (void)std::frexp(ax.hi, &e2); // ax.hi = f * 2^(e2-1)
        int e10 = (int)std::floor((e2 - 1) * 0.30102999566398114); // ≈ log10(2)

        m = ax * pow10_128(-e10);
        while (m >= f128{10.0}) { m = m / f128{10.0}; ++e10; }
        while (m <  f128{1.0})  { m = m * f128{10.0}; --e10; }
        exp10 = e10;
    }

    FORCE_INLINE f128 round_scaled(f128 x, int prec) noexcept
    {
        /// round x to an integer at scale = 10^prec (ties-to-even)
        if (prec <= 0) return x;
        const f128 scale = pow10_128(prec);
        f128 y = x * scale;

        f128 n = floor(y); // integer below
        f128 f = y - n;    // fraction in [0,1]

        const f128 half = f128{0.5};
        bool tie = (f == half);
        if (f > half || (tie && fmod(n, f128{ 2 }) != 0))
            n = n + 1;

        return n;
    }

    BL_PUSH_PRECISE
    FORCE_INLINE CONSTEXPR_NO_FMA f128 mul_by_double_print(f128 a, double b) noexcept
    {
        double p, err;
        _f128_detail::two_prod_precise(a.hi, b, p, err);
        err += a.lo * b;

        double s, e;
        _f128_detail::two_sum_precise(p, err, s, e);
        return f128{s, e};
    }
    FORCE_INLINE f128 sub_by_double_print(f128 a, double b) noexcept
    {
        double s, e;
        _f128_detail::two_sum_precise(a.hi, -b, s, e);
        e += a.lo;

        double ss, ee;
        _f128_detail::two_sum_precise(s, e, ss, ee);
        return f128{ss, ee};
    }
    BL_POP_PRECISE

    struct f128_chars_result
    {
        char* ptr = nullptr;
        bool ok = false;
    };

    FORCE_INLINE int emit_uint_rev_buf(char* dst, f128 n)
    {
        // n is a non-negative integer in f128
        const f128 base = f128{ 1000000000.0 }; // 1e9

        int len = 0;

        if (n < f128{10.0}) {
            int d = (int)n.hi;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            dst[len++] = char('0' + d);
            return len;
        }

        while (n >= base) {
            f128 q = floor(n / base);
            f128 r = n - q * base;

            long long chunk = (long long)std::floor(r.hi);
            if (chunk >= 1000000000LL) { chunk -= 1000000000LL; q = q + f128{1.0}; }
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
        } else {
            while (last > 0) {
                int d = int(last % 10);
                dst[len++] = char('0' + d);
                last /= 10;
            }
        }

        return len;
    }
    FORCE_INLINE void emit_uint_rev(std::string& out, f128 n)
    {
        char tmp[320];
        int len = emit_uint_rev_buf(tmp, n);
        out.assign(tmp, tmp + len);
    }
    FORCE_INLINE f128_chars_result append_exp10_to_chars(char* p, char* end, int e10) noexcept
    {
        if (p >= end) return {p, false};
        *p++ = 'e';

        if (p >= end) return {p, false};
        if (e10 < 0) { *p++ = '-'; e10 = -e10; }
        else         { *p++ = '+'; }

        char buf[8];
        int n = 0;
        do {
            buf[n++] = char('0' + (e10 % 10));
            e10 /= 10;
        } while (e10);

        if (n < 2) buf[n++] = '0';

        if (p + n > end) return {p, false};
        for (int i = n - 1; i >= 0; --i) *p++ = buf[i];

        return {p, true};
    }

    using biguint = fltx_common::exact_decimal::biguint;

    struct exact_traits
    {
        using value_type = f128;
        static constexpr int limb_count = 2;
        static constexpr int significand_bits = 106;

        static double limb(const value_type& x, int index) noexcept
        {
            return index == 0 ? x.hi : x.lo;
        }
        static value_type zero(bool neg = false) noexcept
        {
            return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 };
        }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static value_type pack_from_significand(const biguint& q, int e2, bool neg) noexcept
        {
            const std::uint64_t c1 = q.get_bits(0, 53);
            const std::uint64_t c0 = q.get_bits(53, 53);
            const double hi = c0 ? std::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
            const double lo = c1 ? std::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
            f128 out = _f128_detail::renorm(hi, lo);
            if (neg)
                out = -out;
            return out;
        }
    };

    inline bool exact_scientific_digits(const f128& x, int sig, std::string& digits, int& exp10)
    {
        return fltx_common::exact_decimal::exact_scientific_digits<exact_traits>(x, sig, digits, exp10);
    }
    inline f128 exact_decimal_to_f128(const biguint& coeff, int dec_exp, bool neg) noexcept
    {
        return fltx_common::exact_decimal::exact_decimal_to_value<exact_traits>(coeff, dec_exp, neg);
    }

    FORCE_INLINE f128_chars_result emit_fixed_dec_to_chars(char* first, char* last, f128 x, int prec, bool strip_trailing_zeros) noexcept
    {
        if (x.hi == 0.0 && x.lo == 0.0) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (prec < 0) prec = 0;

        const bool neg = (x.hi < 0.0);
        if (neg) x = f128{ -x.hi, -x.lo };
        x = _f128_detail::renorm(x.hi, x.lo);

        f128 ip = floor(x);
        f128 fp = sub_by_double_print(x, ip.hi);

        // compensate for non-canonical hi/lo splits where floor based on hi underestimates the integer part
        if (fp >= f128{ 1.0 }) { fp = fp - f128{ 1.0 }; ip = ip + f128{ 1.0 }; }
        else if (fp < f128{ 0.0 }) { fp = f128{ 0.0 }; }

        // fractional digits scratch (rounded in-place)
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

            int written = 0;
            const int full = prec / 9;
            const int rem = prec - full * 9;

            for (int c = 0; c < full; ++c) {
                fp = mul_by_double_print(fp, kPow10[9]);
                uint32_t chunk = (uint32_t)fp.hi;
                fp = sub_by_double_print(fp, (double)chunk);

                for (int i = 8; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += 9;
            }

            if (rem > 0) {
                fp = mul_by_double_print(fp, kPow10[rem]);
                uint32_t chunk = (uint32_t)fp.hi;
                fp = sub_by_double_print(fp, (double)chunk);

                for (int i = rem - 1; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += rem;
            }

            // look-ahead digit for ties-to-even
            f128 la = mul_by_double_print(fp, 10.0);
            int next = (int)la.hi;
            if (next < 0) next = 0; else if (next > 9) next = 9;
            f128 remv = sub_by_double_print(la, (double)next);

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
                    ip = ip + f128{ 1.0 };
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
    FORCE_INLINE f128_chars_result emit_scientific_sig_to_chars_f128(char* first, char* last, const f128& x, std::streamsize sig_digits, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }
        if (sig_digits < 1) sig_digits = 1;
        const bool neg = (x.hi < 0.0);
        const f128 v = neg ? -x : x;
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
    FORCE_INLINE f128_chars_result emit_scientific_to_chars(char* first, char* last, const f128& x, std::streamsize frac_digits, bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0) frac_digits = 0;
        if (iszero(x)) {
            const bool neg = std::signbit(x.hi);
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
    FORCE_INLINE f128_chars_result to_chars(char* first, char* last, const f128& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false) noexcept
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
        f128 ax = (x.hi < 0.0) ? -x : x;
        f128 m; int e10 = 0;
        normalize10(ax, m, e10);
        if (e10 >= -4 && e10 < sig) {
            const int frac = std::max(0, sig - (e10 + 1));
            return emit_fixed_dec_to_chars(first, last, x, frac, strip_trailing_zeros);
        }
        return emit_scientific_sig_to_chars_f128(first, last, x, sig, strip_trailing_zeros);
    }

    using f128_format_kind = fltx_common::format_kind;
    using f128_parse_token = fltx_common::parse_token<_f128_detail::biguint>;

    struct f128_io_traits
    {
        using value_type = f128;
        using chars_result = f128_chars_result;
        using parse_token = f128_parse_token;

        static constexpr int max_parse_order = 330;
        static constexpr int min_parse_order = -400;

        static bool isnan(const value_type& x) noexcept { return bl::isnan(x); }
        static bool isinf(const value_type& x) noexcept { return bl::isinf(x); }
        static bool iszero(const value_type& x) noexcept { return bl::iszero(x); }
        static bool is_negative(const value_type& x) noexcept { return x.hi < 0.0; }
        static value_type abs(const value_type& x) noexcept { return (x.hi < 0.0) ? -x : x; }
        static value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0 } : value_type{ 0.0, 0.0 }; }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static value_type quiet_nan() noexcept { return std::numeric_limits<value_type>::quiet_NaN(); }
        static void normalize10(const value_type& x, value_type& m, int& e10) { normalize10(x, m, e10); }
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
        static value_type exact_decimal_to_value(const parse_token::coeff_type& coeff, int dec_exp, bool neg)
        {
            return _f128_detail::exact_decimal_to_f128(coeff, dec_exp, neg);
        }
    };

    template<typename Writer>
    FORCE_INLINE void write_chars_to_string_f128(std::string& out, std::size_t cap, Writer writer)
    {
        fltx_common::write_chars_to_string<f128_chars_result>(out, cap, writer);
    }
    FORCE_INLINE const char* special_text_f128(const f128& x, bool uppercase = false) noexcept
    {
        return fltx_common::special_text<f128_io_traits>(x, uppercase);
    }
    FORCE_INLINE bool assign_special_string_f128(std::string& out, const f128& x, bool uppercase = false) noexcept
    {
        return fltx_common::assign_special_string<f128_io_traits>(out, x, uppercase);
    }
    FORCE_INLINE void ensure_decimal_point_f128(std::string& s)
    {
        fltx_common::ensure_decimal_point(s);
    }
    FORCE_INLINE void apply_stream_decorations_f128(std::string& s, bool showpos, bool uppercase)
    {
        fltx_common::apply_stream_decorations(s, showpos, uppercase);
    }
    FORCE_INLINE bool write_stream_special_f128(std::ostream& os, const f128& x, bool showpos, bool uppercase)
    {
        return fltx_common::write_stream_special<f128_io_traits>(os, x, showpos, uppercase);
    }
    FORCE_INLINE void format_to_string_f128(std::string& out, const f128& x, int precision, f128_format_kind kind, bool strip_trailing_zeros = false)
    {
        fltx_common::format_to_string<f128_io_traits>(out, x, precision, kind, strip_trailing_zeros);
    }
    FORCE_INLINE void to_string_into(std::string& out, const f128& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
    {
        fltx_common::to_string_into<f128_io_traits>(out, x, precision, fixed, scientific, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific(std::string& os, const f128& x, std::streamsize prec, bool strip_trailing_zeros)
    {
        fltx_common::emit_scientific<f128_io_traits>(os, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_fixed_dec(std::string& os, f128 x, int prec, bool strip_trailing_zeros)
    {
        fltx_common::emit_fixed_dec<f128_io_traits>(os, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific_sig_f128(std::string& os, const f128& x, std::streamsize sig_digits, bool strip_trailing_zeros)
    {
        fltx_common::emit_scientific_sig<f128_io_traits>(os, x, sig_digits, strip_trailing_zeros);
    }

    /// ======== Parsing helpers ========

    FORCE_INLINE bool valid_flt128_string(const char* s) noexcept
    {
        return fltx_common::valid_float_string(s);
    }
    FORCE_INLINE unsigned char ascii_lower_f128(char c) noexcept
    {
        return fltx_common::ascii_lower(c);
    }
    FORCE_INLINE const char* skip_ascii_space_f128(const char* p) noexcept
    {
        return fltx_common::skip_ascii_space(p);
    }
    
}

FORCE_INLINE bool parse_flt128(const char* s, f128& out, const char** endptr = nullptr) noexcept
{
    return fltx_common::parse_flt<_f128_detail::f128_io_traits>(s, out, endptr);
}
FORCE_INLINE f128 to_f128(const char* s)
{
    f128 ret;
    if (parse_flt128(s, ret))
        return ret;
    return f128{0};
}
FORCE_INLINE f128 to_f128(const std::string& s) noexcept
{
    return to_f128(s.c_str());
}
FORCE_INLINE std::string to_string(const f128& x, int precision = std::numeric_limits<f128>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    std::string out;
    _f128_detail::to_string_into(out, x, precision, fixed, scientific, strip_trailing_zeros);
    return out;
}

/// ======== Stream output ========

FORCE_INLINE std::ostream& operator<<(std::ostream& os, const f128& x)
{
    return fltx_common::write_to_stream<_f128_detail::f128_io_traits>(os, x);
}

/// ======== Literals ========

FORCE_INLINE constexpr f128 operator""_dd(unsigned long long v) noexcept {
    return to_f128(static_cast<uint64_t>(v));
}
FORCE_INLINE constexpr f128 operator""_dd(long double v) noexcept {
    return f128{ static_cast<double>(v) };
}
//FORCE_INLINE constexpr f128 operator"" _dd(const char* s, std::size_t) {
//    return to_f128(s);
//}

} // namespace bl
