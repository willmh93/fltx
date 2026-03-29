#pragma once
#define F256_INCLUDED
#include "fltx_common.h"

namespace bl {

struct f128;
struct f256;

namespace _f256_detail
{
    using fltx_common::fp::absd;
    using fltx_common::fp::isnan;
    using fltx_common::fp::isinf;
    using fltx_common::fp::isfinite;
    using fltx_common::fp::magnitude_u64;
    using fltx_common::fp::quick_two_sum_precise;
    using fltx_common::fp::split_uint64_to_doubles;
    using fltx_common::fp::two_prod_precise;
    using fltx_common::fp::two_prod_precise_dekker;
    using fltx_common::fp::two_sum_precise;

    // Shewchuk-style expansion sum, expansions sorted by increasing magnitude (small -> large)
    FORCE_INLINE constexpr int fast_expansion_sum_zeroelim(
        int elen, const double* e,
        int flen, const double* f,
        double* h) noexcept
    {
        int eindex = 0;
        int findex = 0;
        int hindex = 0;

        if (elen == 0) {
            for (int i = 0; i < flen; ++i) if (f[i] != 0.0) h[hindex++] = f[i];
            return hindex;
        }
        if (flen == 0) {
            for (int i = 0; i < elen; ++i) if (e[i] != 0.0) h[hindex++] = e[i];
            return hindex;
        }

        double Q{};
        double Qnew{};
        double hh{};

        double enow = e[eindex];
        double fnow = f[findex];

        if (fltx_common::fp::absd(enow) < fltx_common::fp::absd(fnow)) { Q = enow; ++eindex; enow = (eindex < elen) ? e[eindex] : 0.0; }
        else { Q = fnow; ++findex; fnow = (findex < flen) ? f[findex] : 0.0; }

        while (eindex < elen && findex < flen) {
            if (fltx_common::fp::absd(enow) < fltx_common::fp::absd(fnow)) {
                fltx_common::fp::two_sum_precise(Q, enow, Qnew, hh);
                ++eindex;
                enow = (eindex < elen) ? e[eindex] : 0.0;
            }
            else {
                fltx_common::fp::two_sum_precise(Q, fnow, Qnew, hh);
                ++findex;
                fnow = (findex < flen) ? f[findex] : 0.0;
            }

            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (eindex < elen) {
            fltx_common::fp::two_sum_precise(Q, e[eindex], Qnew, hh);
            ++eindex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (findex < flen) {
            fltx_common::fp::two_sum_precise(Q, f[findex], Qnew, hh);
            ++findex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }

    FORCE_INLINE CONSTEXPR_NO_FMA int scale_expansion_zeroelim(int elen, const double* e, double b, double* h) noexcept
    {
        int hindex = 0;
        if (elen == 0 || b == 0.0) return 0;

        double Q{}, sum{}, hh{};
        double product1{}, product0{};

        fltx_common::fp::two_prod_precise(e[0], b, product1, product0);
        Q = product1;
        if (product0 != 0.0) h[hindex++] = product0;

        for (int i = 1; i < elen; ++i) {
            fltx_common::fp::two_prod_precise(e[i], b, product1, product0);

            fltx_common::fp::two_sum_precise(Q, product0, sum, hh);
            if (hh != 0.0) h[hindex++] = hh;

            fltx_common::fp::quick_two_sum_precise(product1, sum, Q, hh);
            if (hh != 0.0) h[hindex++] = hh;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }
    FORCE_INLINE constexpr int compress_expansion_zeroelim(int elen, const double* e, double* h) noexcept
    {
        double g[40]{};

        if (elen <= 0) return 0;

        double Q = e[elen - 1];
        for (int i = elen - 2; i >= 0; --i) {
            double Qnew{}, q{};
            fltx_common::fp::two_sum_precise(Q, e[i], Qnew, q);
            Q = Qnew;
            g[i + 1] = q;
        }
        g[0] = Q;

        int hindex = 0;
        Q = g[0];
        for (int i = 1; i < elen; ++i) {
            double Qnew{}, q{};
            fltx_common::fp::two_sum_precise(Q, g[i], Qnew, q);
            if (q != 0.0) h[hindex++] = q;
            Q = Qnew;
        }
        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }
}

inline constexpr f256 operator+(const f256& a, const f256& b) noexcept;
inline constexpr f256 operator-(const f256& a, const f256& b) noexcept;
inline constexpr f256 operator+(const f256& a, double b) noexcept;
inline constexpr f256 operator-(const f256& a, double b) noexcept;

inline CONSTEXPR_NO_FMA f256 operator*(const f256& a, const f256& b);
inline CONSTEXPR_NO_FMA f256 operator/(const f256& a, const f256& b);
inline CONSTEXPR_NO_FMA f256 operator*(const f256& a, double b);
inline CONSTEXPR_NO_FMA f256 operator/(const f256& a, double b);

struct f256
{
    double x0, x1, x2, x3; // largest -> smallest

    FORCE_INLINE CONSTEXPR_NO_FMA f256& operator*=(f256 rhs) { *this = *this * rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f256& operator/=(f256 rhs) { *this = *this / rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f256& operator*=(double rhs) { *this = *this * rhs; return *this; }
    FORCE_INLINE CONSTEXPR_NO_FMA f256& operator/=(double rhs) { *this = *this / rhs; return *this; }

    FORCE_INLINE constexpr f256& operator=(f128 x) noexcept;
    FORCE_INLINE constexpr f256& operator=(double x) noexcept {
        x0 = x; x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this;
    }
    FORCE_INLINE constexpr f256& operator=(float x) noexcept {
        x0 = static_cast<double>(x); x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this;
    }

    FORCE_INLINE constexpr f256& operator=(uint64_t u) noexcept;
    FORCE_INLINE constexpr f256& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f256& operator=(T v) noexcept {
        return (*this = static_cast<int64_t>(v));
    }
    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    FORCE_INLINE constexpr f256& operator=(T v) noexcept {
        return (*this = static_cast<uint64_t>(v));
    }

    FORCE_INLINE constexpr f256& operator+=(f256 rhs) { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f256& operator-=(f256 rhs) { *this = *this - rhs; return *this; }
    FORCE_INLINE constexpr f256& operator+=(double rhs) { *this = *this + rhs; return *this; }
    FORCE_INLINE constexpr f256& operator-=(double rhs) { *this = *this - rhs; return *this; }

    /// ======== Conversions ========
    explicit constexpr operator f128() const noexcept;
    explicit constexpr operator double() const noexcept { return ((x0 + x1) + (x2 + x3)); }
    explicit constexpr operator float() const noexcept { return static_cast<float>(((x0 + x1) + (x2 + x3))); }
    explicit constexpr operator int() const noexcept { return static_cast<int>(((x0 + x1) + (x2 + x3))); }

    constexpr f256 operator+() const { return *this; }
    constexpr f256 operator-() const noexcept { return f256{ -x0, -x1, -x2, -x3 }; }

    /// ======== Utility ========
    static constexpr f256 eps() { return { 3.038581678643134e-64, 0.0, 0.0, 0.0 }; } // ~2^-211
};

struct f256_t : public f256
{
    f256_t() = default;
    constexpr f256_t(double _x0, double _x1, double _x2, double _x3) noexcept : f256{ _x0, _x1, _x2, _x3 } {}
    constexpr f256_t(float  x) noexcept : f256{ ((double)x), 0.0 } {}
    constexpr f256_t(double x) noexcept : f256{ ((double)x), 0.0 } {}
    constexpr f256_t(int64_t  v) noexcept : f256{} { static_cast<f256&>(*this) = static_cast<int64_t>(v); }
    constexpr f256_t(uint64_t u) noexcept : f256{} { static_cast<f256&>(*this) = static_cast<uint64_t>(u); }
    constexpr f256_t(int32_t  v) noexcept : f256_t((int64_t)v) {}
    constexpr f256_t(uint32_t u) noexcept : f256_t((int64_t)u) {}
    constexpr f256_t(const f256& f) noexcept : f256{ f.x0, f.x1, f.x2, f.x3 } {}
    inline operator f256& () { return static_cast<f256&>(*this); }
};
}

namespace std
{
    using bl::f256;

    template<> struct numeric_limits<f256>
    {
        static constexpr bool is_specialized = true;

        // limits
        static constexpr f256 min()            noexcept { return { numeric_limits<double>::min(), 0.0, 0.0, 0.0 }; }
        static constexpr f256 max()            noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon(), 0.0, 0.0 }; }
        static constexpr f256 lowest()         noexcept { return { -numeric_limits<double>::max(),  numeric_limits<double>::epsilon(), 0.0, 0.0 }; }

        static constexpr int  digits = 212;
        static constexpr int  digits10 = 63;
        static constexpr int  max_digits10 = 66;

        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr int  radix = 2;

        static constexpr f256 epsilon()        noexcept { return f256::eps(); }
        static constexpr f256 round_error()    noexcept { return { 0.5, 0.0, 0.0, 0.0 }; }

        static constexpr int  min_exponent = numeric_limits<double>::min_exponent;
        static constexpr int  min_exponent10 = numeric_limits<double>::min_exponent10;
        static constexpr int  max_exponent = numeric_limits<double>::max_exponent;
        static constexpr int  max_exponent10 = numeric_limits<double>::max_exponent10;

        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        //static constexpr float_denorm_style has_denorm = denorm_present;
        static constexpr bool has_denorm_loss = false;

        static constexpr f256 infinity()       noexcept { return { numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }; }
        static constexpr f256 quiet_NaN()      noexcept { return { numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 }; }
        static constexpr f256 signaling_NaN()  noexcept { return { numeric_limits<double>::signaling_NaN(), 0.0, 0.0, 0.0 }; }
        static constexpr f256 denorm_min()     noexcept { return { numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 }; }

        static constexpr bool is_iec559 = numeric_limits<double>::is_iec559;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;

        static constexpr bool traps = numeric_limits<double>::traps;
        static constexpr bool tinyness_before = numeric_limits<double>::tinyness_before;
        static constexpr float_round_style round_style = round_to_nearest;
    };
}

namespace bl
{
/// ======== Representation helpers and scalar conversions ========

namespace _f256_detail
{
    FORCE_INLINE constexpr void three_sum(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        _f256_detail::two_sum_precise(a, b, t1, t2);
        _f256_detail::two_sum_precise(c, t1, a, t3);
        _f256_detail::two_sum_precise(t2, t3, b, c);
    }
    FORCE_INLINE constexpr void three_sum2(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        _f256_detail::two_sum_precise(a, b, t1, t2);
        _f256_detail::two_sum_precise(c, t1, a, t3);
        b = t2 + t3;
    }

    FORCE_INLINE constexpr f256 renorm(double c0, double c1, double c2, double c3) noexcept
    {
        //if (_f256_detail::isinf(c0))
        //    return { c0, c1, c2, c3 };

        double s0{}, s1{}, s2 = 0.0, s3 = 0.0;

        _f256_detail::quick_two_sum_precise(c2, c3, s0, c3);
        _f256_detail::quick_two_sum_precise(c1, s0, s0, c2);
        _f256_detail::quick_two_sum_precise(c0, s0, c0, c1);

        s0 = c0;
        s1 = c1;

        if (s1 != 0.0)
        {
            _f256_detail::quick_two_sum_precise(s1, c2, s1, s2);
            if (s2 != 0.0)
                _f256_detail::quick_two_sum_precise(s2, c3, s2, s3);
            else
                _f256_detail::quick_two_sum_precise(s1, c3, s1, s2);
        }
        else
        {
            _f256_detail::quick_two_sum_precise(s0, c2, s0, s1);
            if (s1 != 0.0)
                _f256_detail::quick_two_sum_precise(s1, c3, s1, s2);
            else
                _f256_detail::quick_two_sum_precise(s0, c3, s0, s1);
        }

        return { s0, s1, s2, s3 };
    }
    FORCE_INLINE constexpr f256 renorm4(double c0, double c1, double c2, double c3) noexcept
    {
        double s = c2 + c3;
        double e = c3 - (s - c2);
        c2 = s;
        c3 = e;

        s = c1 + c2;
        e = c2 - (s - c1);
        c1 = s;
        c2 = e;

        s = c0 + c1;
        e = c1 - (s - c0);
        c0 = s;
        c1 = e;

        double s0 = c0;
        double s1 = c1;
        double s2 = 0.0;
        double s3 = 0.0;

        if (c2 != 0.0)
        {
            if (s1 != 0.0)
            {
                s = s1 + c2;
                e = c2 - (s - s1);
                s1 = s;
                s2 = e;
            }
            else
            {
                s = s0 + c2;
                e = c2 - (s - s0);
                s0 = s;
                s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0)
            {
                s = s2 + c3;
                e = c3 - (s - s2);
                s2 = s;
                s3 = e;
            }
            else if (s1 != 0.0)
            {
                s = s1 + c3;
                e = c3 - (s - s1);
                s1 = s;
                s2 = e;
            }
            else
            {
                s = s0 + c3;
                e = c3 - (s - s0);
                s0 = s;
                s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }
    FORCE_INLINE constexpr f256 renorm5(double c0, double c1, double c2, double c3, double c4) noexcept
    {
        double s, e;

        s = c3 + c4;
        e = c4 - (s - c3);
        c3 = s;
        c4 = e;

        s = c2 + c3;
        e = c3 - (s - c2);
        c2 = s;
        c3 = e;

        s = c1 + c2;
        e = c2 - (s - c1);
        c1 = s;
        c2 = e;

        s = c0 + c1;
        e = c1 - (s - c0);
        c0 = s;
        c1 = e;

        double s0 = c0;
        double s1 = c1;
        double s2 = 0.0;
        double s3 = 0.0;

        if (c2 != 0.0)
        {
            if (s1 != 0.0)
            {
                s = s1 + c2;
                e = c2 - (s - s1);
                s1 = s;
                s2 = e;
            }
            else
            {
                s = s0 + c2;
                e = c2 - (s - s0);
                s0 = s;
                s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0)
            {
                s = s2 + c3;
                e = c3 - (s - s2);
                s2 = s;
                s3 = e;
            }
            else if (s1 != 0.0)
            {
                s = s1 + c3;
                e = c3 - (s - s1);
                s1 = s;
                s2 = e;
            }
            else
            {
                s = s0 + c3;
                e = c3 - (s - s0);
                s0 = s;
                s1 = e;
            }
        }

        if (c4 != 0.0)
        {
            if (s3 != 0.0)
            {
                s3 += c4;
            }
            else if (s2 != 0.0)
            {
                s = s2 + c4;
                e = c4 - (s - s2);
                s2 = s;
                s3 = e;
            }
            else if (s1 != 0.0)
            {
                s = s1 + c4;
                e = c4 - (s - s1);
                s1 = s;
                s2 = e;
            }
            else
            {
                s = s0 + c4;
                e = c4 - (s - s0);
                s0 = s;
                s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }
}

FORCE_INLINE constexpr f256 to_f256(uint64_t u) noexcept
{
    f256 r{}; r = u;
    return r;
}
FORCE_INLINE constexpr f256 to_f256(int64_t v) noexcept
{
    f256 r{}; r = v;
    return r;
}

FORCE_INLINE constexpr f256& f256::operator=(uint64_t u) noexcept
{
    double a{}, b{};
    _f256_detail::split_uint64_to_doubles(u, a, b);
    double s{}, e{}; _f256_detail::two_sum_precise(a, b, s, e);
    *this = _f256_detail::renorm(s, e, 0.0, 0.0);
    return *this;
}
FORCE_INLINE constexpr f256& f256::operator=(int64_t v) noexcept
{
    if (v >= 0) return (*this = static_cast<uint64_t>(v));

    uint64_t mag = _f256_detail::magnitude_u64(v);
    f256 tmp = to_f256(mag);
    *this = -tmp;
    return *this;
}

/// ======== Comparisons ========

/// ------------------ f256 <=> f256 ------------------

FORCE_INLINE constexpr bool operator<(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return false;

    if (a.x0 < b.x0) return true;
    if (a.x0 > b.x0) return false;

    if (a.x1 < b.x1) return true;
    if (a.x1 > b.x1) return false;

    if (a.x2 < b.x2) return true;
    if (a.x2 > b.x2) return false;

    return a.x3 < b.x3;
}
FORCE_INLINE constexpr bool operator>(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return false;
    return b < a;
}
FORCE_INLINE constexpr bool operator<=(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return false;
    return !(b < a);
}
FORCE_INLINE constexpr bool operator>=(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return false;
    return !(a < b);
}
FORCE_INLINE constexpr bool operator==(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return false;
    return a.x0 == b.x0 && a.x1 == b.x1 && a.x2 == b.x2 && a.x3 == b.x3;
}
FORCE_INLINE constexpr bool operator!=(const f256& a, const f256& b)
{
    if (_f256_detail::isnan(a.x0) || _f256_detail::isnan(b.x0))
        return true;
    return !(a == b);
}

/// ------------------ double <=> f256 ------------------

FORCE_INLINE constexpr bool operator<(const f256& a, double b) { return a < f256{b}; }
FORCE_INLINE constexpr bool operator<(double a, const f256& b) { return f256{a} < b; }
FORCE_INLINE constexpr bool operator>(const f256& a, double b) { return b < a; }
FORCE_INLINE constexpr bool operator>(double a, const f256& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f256& a, double b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(double a, const f256& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f256& a, double b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(double a, const f256& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f256& a, double b) { return a == f256{b}; }
FORCE_INLINE constexpr bool operator==(double a, const f256& b) { return f256{a} == b; }
FORCE_INLINE constexpr bool operator!=(const f256& a, double b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(double a, const f256& b) { return !(a == b); }

/// ------------------ int64_t/uint64_t <=> f256 ------------------

FORCE_INLINE constexpr bool operator<(const f256& a, int64_t b) { return a < to_f256(b); }
FORCE_INLINE constexpr bool operator<(int64_t a, const f256& b) { return to_f256(a) < b; }
FORCE_INLINE constexpr bool operator>(const f256& a, int64_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(int64_t a, const f256& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f256& a, int64_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(int64_t a, const f256& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f256& a, int64_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(int64_t a, const f256& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f256& a, int64_t b) { return a == to_f256(b); }
FORCE_INLINE constexpr bool operator==(int64_t a, const f256& b) { return to_f256(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f256& a, int64_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(int64_t a, const f256& b) { return !(a == b); }

FORCE_INLINE constexpr bool operator<(const f256& a, uint64_t b) { return a < to_f256(b); }
FORCE_INLINE constexpr bool operator<(uint64_t a, const f256& b) { return to_f256(a) < b; }
FORCE_INLINE constexpr bool operator>(const f256& a, uint64_t b) { return b < a; }
FORCE_INLINE constexpr bool operator>(uint64_t a, const f256& b) { return b < a; }
FORCE_INLINE constexpr bool operator<=(const f256& a, uint64_t b) { return !(b < a); }
FORCE_INLINE constexpr bool operator<=(uint64_t a, const f256& b) { return !(b < a); }
FORCE_INLINE constexpr bool operator>=(const f256& a, uint64_t b) { return !(a < b); }
FORCE_INLINE constexpr bool operator>=(uint64_t a, const f256& b) { return !(a < b); }
FORCE_INLINE constexpr bool operator==(const f256& a, uint64_t b) { return a == to_f256(b); }
FORCE_INLINE constexpr bool operator==(uint64_t a, const f256& b) { return to_f256(a) == b; }
FORCE_INLINE constexpr bool operator!=(const f256& a, uint64_t b) { return !(a == b); }
FORCE_INLINE constexpr bool operator!=(uint64_t a, const f256& b) { return !(a == b); }

/// ======== Arithmetic operators ========

FORCE_INLINE CONSTEXPR_NO_FMA f256 recip(f256 b) noexcept
{
    constexpr f256 one = f256{ 1.0 };

    double q0 = 1.0 / b.x0;
    f256 r = one - (b * q0);

    double q1 = r.x0 / b.x0;
    r -= (b * q1);

    double q2 = r.x0 / b.x0;
    r -= (b * q2);

    double q3 = r.x0 / b.x0;
    r -= (b * q3);

    double q4 = r.x0 / b.x0;

    return _f256_detail::renorm5(q0, q1, q2, q3, q4);
}
FORCE_INLINE CONSTEXPR_NO_FMA f256 inv(const f256& a) { return recip(a); }


/// ------------------ core helpers ------------------

namespace _f256_detail
{
    FORCE_INLINE constexpr f256 add_scalar_precise(const f256& a, double b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        _f256_detail::two_sum_precise(a.x0, b, s0, e0);
        _f256_detail::two_sum_precise(a.x1, e0, s1, e1);
        _f256_detail::two_sum_precise(a.x2, e1, s2, e2);
        _f256_detail::two_sum_precise(a.x3, e2, s3, e3);

        return renorm5(s0, s1, s2, s3, e3);
    }
    FORCE_INLINE constexpr f256 from_expansion_fast(const double* h, int n) noexcept
    {
        if (n <= 0) return {};

        double comp[40]{};
        const int m = _f256_detail::compress_expansion_zeroelim(n, h, comp);

        f256 sum{};
        for (int i = 0; i < m; ++i)
            sum = add_scalar_precise(sum, comp[i]);

        return sum;
    }

    FORCE_INLINE constexpr f256 add_dd_qd(const f256& a, const f256& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};

        _f256_detail::two_sum_precise(a.x0, b.x0, s0, e0);
        _f256_detail::two_sum_precise(a.x1, b.x1, s1, e1);
        _f256_detail::two_sum_precise(s1, e0, s1, e0);

        e0 += e1;

        if (e0 == 0.0)
            return _f256_detail::renorm4(s0, s1, 0.0, 0.0);

        return _f256_detail::renorm5(s0, s1, e0, 0.0, 0.0);
    }
    FORCE_INLINE constexpr f256 sub_dd_qd(const f256& a, const f256& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};

        _f256_detail::two_sum_precise(a.x0, -b.x0, s0, e0);
        _f256_detail::two_sum_precise(a.x1, -b.x1, s1, e1);
        _f256_detail::two_sum_precise(s1, e0, s1, e0);

        e0 += e1;

        if (e0 == 0.0)
            return _f256_detail::renorm4(s0, s1, 0.0, 0.0);

        return _f256_detail::renorm5(s0, s1, e0, 0.0, 0.0);
    }
    FORCE_INLINE constexpr f256 add_qd_qd(const f256& a, const f256& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        _f256_detail::two_sum_precise(a.x0, b.x0, s0, e0);
        _f256_detail::two_sum_precise(a.x1, b.x1, s1, e1);
        _f256_detail::two_sum_precise(a.x2, b.x2, s2, e2);
        _f256_detail::two_sum_precise(a.x3, b.x3, s3, e3);

        _f256_detail::two_sum_precise(s1, e0, s1, e0);
        _f256_detail::three_sum(s2, e0, e1);
        _f256_detail::three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return _f256_detail::renorm4(s0, s1, s2, s3);

        return _f256_detail::renorm5(s0, s1, s2, s3, e0);
    }
    FORCE_INLINE constexpr f256 sub_qd_qd(const f256& a, const f256& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

        _f256_detail::two_sum_precise(a.x0, -b.x0, s0, e0);
        _f256_detail::two_sum_precise(a.x1, -b.x1, s1, e1);
        _f256_detail::two_sum_precise(a.x2, -b.x2, s2, e2);
        _f256_detail::two_sum_precise(a.x3, -b.x3, s3, e3);

        _f256_detail::two_sum_precise(s1, e0, s1, e0);
        _f256_detail::three_sum(s2, e0, e1);
        _f256_detail::three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return _f256_detail::renorm4(s0, s1, s2, s3);

        return _f256_detail::renorm5(s0, s1, s2, s3, e0);
    }

    FORCE_INLINE CONSTEXPR_NO_FMA f256 sub_mul_scalar_fast(const f256& r, const f256& b, double q) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};
        double p2{}, e2{};
        double p3{}, e3{};

        _f256_detail::two_prod_precise(b.x0, q, p0, e0);
        _f256_detail::two_prod_precise(b.x1, q, p1, e1);
        _f256_detail::two_prod_precise(b.x2, q, p2, e2);
        _f256_detail::two_prod_precise(b.x3, q, p3, e3);

        double s0{}, t0{};
        double s1{}, t1{};
        double s2{}, t2{};
        double s3{}, t3{};

        s0 = r.x0 - p0;
        s1 = r.x1 - p1;
        s2 = r.x2 - p2;
        s3 = r.x3 - p3;

        double v0 = s0 - r.x0;
        double v1 = s1 - r.x1;
        double v2 = s2 - r.x2;
        double v3 = s3 - r.x3;

        double u0 = s0 - v0;
        double u1 = s1 - v1;
        double u2 = s2 - v2;
        double u3 = s3 - v3;

        double w0 = r.x0 - u0;
        double w1 = r.x1 - u1;
        double w2 = r.x2 - u2;
        double w3 = r.x3 - u3;

        u0 = -p0 - v0;
        u1 = -p1 - v1;
        u2 = -p2 - v2;
        u3 = -p3 - v3;

        t0 = w0 + u0;
        t1 = w1 + u1;
        t2 = w2 + u2;
        t3 = w3 + u3;

        double tail0 = t0 - e0;
        _f256_detail::two_sum_precise(s1, tail0, s1, t0);

        double tail1 = t1 - e1;
        three_sum(s2, t0, tail1);

        double tail2 = t2 - e2;
        three_sum2(s3, t0, tail2);

        t0 = t0 + tail1 + t3 - e3;

        return renorm5(s0, s1, s2, s3, t0);
    }
    FORCE_INLINE CONSTEXPR_NO_FMA f256 sub_mul_scalar_exact(const f256& r, const f256& b, double q) noexcept
    {
        const f256 prod = b * q;
        return sub_qd_qd(r, prod);
    }
}

/// ------------------ scalar arithmetic ------------------

inline constexpr f256 operator+(const f256& a, double b) noexcept
{
    double c0{}, c1{}, c2{}, c3{};
    double e{};

    _f256_detail::two_sum_precise(a.x0, b, c0, e);
    if (e == 0.0)
        return { c0, a.x1, a.x2, a.x3 };

    _f256_detail::two_sum_precise(a.x1, e, c1, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, a.x2, a.x3);

    _f256_detail::two_sum_precise(a.x2, e, c2, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, c2, a.x3);

    _f256_detail::two_sum_precise(a.x3, e, c3, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, c2, c3);

    return _f256_detail::renorm5(c0, c1, c2, c3, e);
}
inline constexpr f256 operator-(const f256& a, double b) noexcept
{
    double c0{}, c1{}, c2{}, c3{};
    double e{};

    _f256_detail::two_sum_precise(a.x0, -b, c0, e);
    if (e == 0.0)
        return { c0, a.x1, a.x2, a.x3 };

    _f256_detail::two_sum_precise(a.x1, e, c1, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, a.x2, a.x3);

    _f256_detail::two_sum_precise(a.x2, e, c2, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, c2, a.x3);

    _f256_detail::two_sum_precise(a.x3, e, c3, e);
    if (e == 0.0)
        return _f256_detail::renorm4(c0, c1, c2, c3);

    return _f256_detail::renorm5(c0, c1, c2, c3, e);
}
inline constexpr f256 operator+(double a, const f256& b) noexcept
{
    return b + a;
}
inline constexpr f256 operator-(double a, const f256& b) noexcept
{
    return -(b - a);
}
inline constexpr f256 operator+(const f256& a, const f256& b) noexcept
{
    if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0) return b;
    if (b.x0 == 0.0 && b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) return a;

    if (a.x2 == 0.0 && a.x3 == 0.0 &&
        b.x2 == 0.0 && b.x3 == 0.0)
    {
        return _f256_detail::add_dd_qd(a, b);
    }

    return _f256_detail::add_qd_qd(a, b);
}
inline constexpr f256 operator-(const f256& a, const f256& b) noexcept
{
    if (b.x0 == 0.0 && b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) return a;
    if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0) return -b;

    if (a.x2 == 0.0 && a.x3 == 0.0 &&
        b.x2 == 0.0 && b.x3 == 0.0)
    {
        return _f256_detail::sub_dd_qd(a, b);
    }

    return _f256_detail::sub_qd_qd(a, b);
}

/// ------------------ multiply and divide ------------------

inline CONSTEXPR_NO_FMA f256 operator*(const f256& a, double b)
{
    double p0{}, p1{}, p2{}, p3{};
    double q0{}, q1{}, q2{};
    double s0{}, s1{}, s2{}, s3{}, s4{};

    _f256_detail::two_prod_precise(a.x0, b, p0, q0);
    _f256_detail::two_prod_precise(a.x1, b, p1, q1);
    _f256_detail::two_prod_precise(a.x2, b, p2, q2);
    p3 = a.x3 * b;

    s0 = p0;
    _f256_detail::two_sum_precise(q0, p1, s1, s2);
    _f256_detail::three_sum(s2, q1, p2);
    _f256_detail::three_sum2(q1, q2, p3);
    s3 = q1;
    s4 = q2 + p2;

    return _f256_detail::renorm5(s0, s1, s2, s3, s4);
}
inline CONSTEXPR_NO_FMA f256 operator*(const f256& a, const f256& b)
{
    double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
    double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
    double p6{}, p7{}, p8{}, p9{};
    double q6{}, q7{}, q8{}, q9{};
    double r0{}, r1{};
    double t0{}, t1{};
    double s0{}, s1{}, s2{};

    _f256_detail::two_prod_precise(a.x0, b.x0, p0, q0);
    _f256_detail::two_prod_precise(a.x0, b.x1, p1, q1);
    _f256_detail::two_prod_precise(a.x1, b.x0, p2, q2);
    _f256_detail::two_prod_precise(a.x0, b.x2, p3, q3);
    _f256_detail::two_prod_precise(a.x1, b.x1, p4, q4);
    _f256_detail::two_prod_precise(a.x2, b.x0, p5, q5);

    _f256_detail::three_sum(p1, p2, q0);
    _f256_detail::three_sum(p2, q1, q2);
    _f256_detail::three_sum(p3, p4, p5);

    _f256_detail::two_sum_precise(p2, p3, s0, t0);
    _f256_detail::two_sum_precise(q1, p4, s1, t1);
    s2 = q2 + p5;
    _f256_detail::two_sum_precise(s1, t0, s1, t0);
    s2 += (t0 + t1);

    _f256_detail::two_prod_precise(a.x0, b.x3, p6, q6);
    _f256_detail::two_prod_precise(a.x1, b.x2, p7, q7);
    _f256_detail::two_prod_precise(a.x2, b.x1, p8, q8);
    _f256_detail::two_prod_precise(a.x3, b.x0, p9, q9);

    _f256_detail::two_sum_precise(q0, q3, q0, q3);
    _f256_detail::two_sum_precise(q4, q5, q4, q5);
    _f256_detail::two_sum_precise(p6, p7, p6, p7);
    _f256_detail::two_sum_precise(p8, p9, p8, p9);

    _f256_detail::two_sum_precise(q0, q4, t0, t1);
    t1 += (q3 + q5);

    _f256_detail::two_sum_precise(p6, p8, r0, r1);
    r1 += (p7 + p9);

    _f256_detail::two_sum_precise(t0, r0, q3, q4);
    q4 += (t1 + r1);

    _f256_detail::two_sum_precise(q3, s1, t0, t1);
    t1 += q4;

    t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1
        + q6 + q7 + q8 + q9 + s2;

    return _f256_detail::renorm5(p0, p1, s0, t0, t1);
}

// divide
inline CONSTEXPR_NO_FMA f256 operator/(const f256& a, const f256& b)
{
    if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0)
        return a / b.x0;

    const double inv_b0 = 1.0 / b.x0;

    const double q0 = a.x0 * inv_b0;
    f256 r = _f256_detail::sub_mul_scalar_exact(a, b, q0);

    const double q1 = r.x0 * inv_b0;
    r = _f256_detail::sub_mul_scalar_fast(r, b, q1);

    const double q2 = r.x0 * inv_b0;
    r = _f256_detail::sub_mul_scalar_fast(r, b, q2);

    const double q3 = r.x0 * inv_b0;
    r = _f256_detail::sub_mul_scalar_fast(r, b, q3);

    const double q4 = r.x0 * inv_b0;

    return _f256_detail::renorm5(q0, q1, q2, q3, q4);
}
inline CONSTEXPR_NO_FMA f256 operator/(const f256& a, double b)
{
    const double inv_b = 1.0 / b;
    const f256 divisor{ b, 0.0, 0.0, 0.0 };

    const double q0 = a.x0 * inv_b;
    f256 r = _f256_detail::sub_mul_scalar_exact(a, divisor, q0);

    const double q1 = r.x0 * inv_b;
    r = _f256_detail::sub_mul_scalar_fast(r, divisor, q1);

    const double q2 = r.x0 * inv_b;
    r = _f256_detail::sub_mul_scalar_fast(r, divisor, q2);

    const double q3 = r.x0 * inv_b;
    r = _f256_detail::sub_mul_scalar_fast(r, divisor, q3);

    const double q4 = r.x0 * inv_b;

    return _f256_detail::renorm5(q0, q1, q2, q3, q4);
}
FORCE_INLINE CONSTEXPR_NO_FMA f256 operator*(double a, const f256& b) { return b * a; }
FORCE_INLINE CONSTEXPR_NO_FMA f256 operator/(double a, const f256& b) { return f256{ a } / b; }

/// ======== Additional math functions ========

FORCE_INLINE constexpr bool isnan(const f256& a) noexcept { return _f256_detail::isnan(a.x0); }
FORCE_INLINE constexpr bool isinf(const f256& a) noexcept { return _f256_detail::isinf(a.x0); }
FORCE_INLINE constexpr bool isfinite(const f256& x) noexcept { return _f256_detail::isfinite(x.x0); }
FORCE_INLINE constexpr bool iszero(const f256& a) noexcept { return a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0; }
FORCE_INLINE constexpr bool ispositive(const f256& x)
{
    return x.x0 > 0.0 ||
        (x.x0 == 0.0 &&
        (x.x1 > 0.0 ||
        (x.x1 == 0.0 &&
        (x.x2 > 0.0 ||
        (x.x2 == 0.0 && x.x3 > 0.0)))));
}

FORCE_INLINE constexpr f256 clamp(const f256& v, const f256& lo, const f256& hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
FORCE_INLINE constexpr f256 abs(const f256& a) noexcept
{
    return (a.x0 < 0.0) ? -a : a;
}
FORCE_INLINE f256 floor(const f256& a)
{
    f256 r{ std::floor(a.x0) };
    if (r > a) r -= 1.0;
    return r;
}
FORCE_INLINE f256 ceil(const f256& a)
{
    f256 r{ std::ceil(a.x0) };
    if (r < a) r += 1.0;
    return r;
}
FORCE_INLINE f256 trunc(const f256& a)
{
    return (a.x0 < 0.0) ? ceil(a) : floor(a);
}
FORCE_INLINE f256 fmod(const f256& x, const f256& y)
{
    if (iszero(y))
        return std::numeric_limits<f256>::quiet_NaN();
    return x - trunc(x / y) * y;
}
FORCE_INLINE f256 round(const f256& a)
{
    f256 t = floor(a + f256{ 0.5 });
    if ((t - a) == f256{ 0.5 } && fmod(t, f256{ 2.0 }) != f256{ 0.0 })
        t -= f256{ 1.0 };
    return t;
}
FORCE_INLINE f256 round_to_decimals(f256 v, int prec)
{
    if (prec <= 0) return v;

    static constexpr f256 inv10_qd{
         0x1.999999999999ap-4,
        -0x1.999999999999ap-58,
         0x1.999999999999ap-112,
        -0x1.999999999999ap-166
    };

    const bool neg = v < 0.0;
    if (neg) v = -v;

    f256 ip = floor(v);
    f256 frac = v - ip;

    std::string dig;
    dig.reserve((size_t)prec);

    f256 w = frac;
    for (int i = 0; i < prec; ++i)
    {
        w = w * 10.0;
        int di = (int)floor(w).x0;
        if (di < 0) di = 0;
        else if (di > 9) di = 9;
        dig.push_back(char('0' + di));
        w = w - f256{ (double)di };
    }

    f256 la = w * 10.0;
    int next = (int)floor(la).x0;
    if (next < 0) next = 0;
    else if (next > 9) next = 9;
    f256 rem = la - f256{ (double)next };

    const int last = dig.empty() ? 0 : (dig.back() - '0');
    const bool round_up =
        (next > 5) ||
        (next == 5 && (rem.x0 > 0.0 || rem.x1 > 0.0 || rem.x2 > 0.0 || rem.x3 > 0.0 || (last & 1)));

    if (round_up)
    {
        int i = prec - 1;
        for (; i >= 0; --i)
        {
            if (dig[(size_t)i] == '9') dig[(size_t)i] = '0';
            else
            {
                ++dig[(size_t)i];
                break;
            }
        }

        if (i < 0)
            ip = ip + 1.0;
    }

    f256 frac_val{ 0.0 };
    for (int i = prec - 1; i >= 0; --i)
    {
        frac_val = frac_val + f256{ (double)(dig[(size_t)i] - '0') };
        frac_val = frac_val * inv10_qd;
    }

    f256 out = ip + frac_val;
    return neg ? -out : out;
}
FORCE_INLINE f256 sqrt(const f256& a)
{
    if (a.x0 <= 0.0)
    {
        if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256{};
        return f256{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    f256 y{ std::sqrt(a.x0), 0.0, 0.0, 0.0 };
    for (int i = 0; i < 6; ++i)
        y = (y + a / y) * 0.5;

    return y;
}
FORCE_INLINE f256 nearbyint(const f256& a)
{
    f256 flo = floor(a);
    const f256 frac = a - flo;

    if (frac < 0.5)
        return flo;
    if (frac > 0.5)
        return flo + 1.0;

    const long long n = (long long)flo.x0;
    return (n & 1LL) ? (flo + 1.0) : flo;
}

/// ======== Transcendental functions ========
namespace _f256_const
{
    // high-precision constants
    inline constexpr f256 pi      = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53, -0x1.f1976b7ed8fbcp-109,  0x1.4cf98e804177dp-163 };
    inline constexpr f256 pi_2    = { 0x1.921fb54442d18p+0,  0x1.1a62633145c07p-54, -0x1.f1976b7ed8fbcp-110,  0x1.4cf98e804177dp-164 };
    inline constexpr f256 pi_4    = { 0x1.921fb54442d18p-1,  0x1.1a62633145c07p-55, -0x1.f1976b7ed8fbcp-111,  0x1.4cf98e804177dp-165 };
    inline constexpr f256 invpi2  = { 0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55, -0x1.6447e493ad4cep-109,  0x1.e21c820ff28b2p-163 };
    inline constexpr f256 ln2     = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56,  0x1.7b57a079a1934p-111, -0x1.ace93a4ebe5d1p-165 };
    inline constexpr f256 inv_ln2 = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56, -0x1.60bb8a5442ab9p-110, -0x1.4b52d3ba6d74dp-166 };
    inline constexpr f256 ln10    = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53, -0x1.9ebae3ae0260cp-107, -0x1.2d10378be1cf1p-161 };
}
namespace _f256_detail
{
    FORCE_INLINE f256 f256_exp_kernel_ln2_half(const f256& r)
    {
        f256 term{ 1.0 };
        f256 sum{ 1.0 };

        double eps = std::ldexp(1.0, -240);

        for (int n = 1; n <= 64; ++n)
        {
            term = term * (r / (double)n);
            sum = sum + term;

            if (std::fabs(static_cast<double>(term)) <= std::fabs(static_cast<double>(sum)) * eps)
                break;
        }

        return sum;
    }
    FORCE_INLINE bool f256_rem_pio2(const f256& x, long long& n_out, f256& r_out)
    {
        if (!std::isfinite(x.x0))
            return false;

        if (abs(x) <= _f256_const::pi_4)
        {
            n_out = 0;
            r_out = x;
            return true;
        }

        const f256 q = nearbyint(x * _f256_const::invpi2);
        const double qd = q.x0;

        if (!std::isfinite(qd) || std::fabs(qd) > 9.0e15)
        {
            const double xd = static_cast<double>(x);
            const double fallback_qd = (double)std::llround(xd * static_cast<double>(_f256_const::invpi2));

            if (!std::isfinite(fallback_qd) || std::fabs(fallback_qd) > 9.0e15)
                return false;

            const long long n = (long long)fallback_qd;
            const f256 qf{ (double)n };

            f256 r = x;
            r -= qf * _f256_const::pi_2.x0;
            r -= qf * _f256_const::pi_2.x1;
            r -= qf * _f256_const::pi_2.x2;
            r -= qf * _f256_const::pi_2.x3;

            if (r > _f256_const::pi_4)
            {
                r -= _f256_const::pi_2;
                n_out = n + 1;
            }
            else if (r < -_f256_const::pi_4)
            {
                r += _f256_const::pi_2;
                n_out = n - 1;
            }
            else
            {
                n_out = n;
            }

            r_out = r;
            return true;
        }

        long long n = (long long)qd;
        f256 r = x;
        r -= q * _f256_const::pi_2.x0;
        r -= q * _f256_const::pi_2.x1;
        r -= q * _f256_const::pi_2.x2;
        r -= q * _f256_const::pi_2.x3;

        if (r > _f256_const::pi_4)
        {
            r -= _f256_const::pi_2;
            ++n;
        }
        else if (r < -_f256_const::pi_4)
        {
            r += _f256_const::pi_2;
            --n;
        }

        n_out = n;
        r_out = r;
        return true;
    }
    FORCE_INLINE void f256_sincos_kernel_pio4(const f256& r, f256& s_out, f256& c_out)
    {
        static constexpr f256 sin_coeffs[] = {
            {  0x1.5a42f0dfeb086p-209, -0x1.35ae015f78f6ep-264, -0x1.c71a521ce2e79p-318,  0x1.6a300230ce998p-372 },
            { -0x1.8da8e0a127ebap-198,  0x1.21d2eac9d275cp-252,  0x1.ad541d26964afp-306,  0x1.1c066ebdf95dep-360 },
            {  0x1.a3cb872220648p-187, -0x1.c7f4e85b8e6cdp-241, -0x1.413a0bc5fc28ap-295, -0x1.16ae534063fabp-352 },
            { -0x1.95db45257e512p-176, -0x1.6e5d72b6f79b9p-231, -0x1.b830cf0b5b5c6p-291, -0x1.29276833f5728p-345 },
            {  0x1.65e61c39d0241p-165, -0x1.c0ed181727269p-220, -0x1.abbd2f56bbc2fp-276, -0x1.18ff57fdc2e4ep-330 },
            { -0x1.1e99449a4bacep-154,  0x1.fefbb89514b3cp-210,  0x1.53433f743a2d9p-264, -0x1.25f70d1395dd7p-320 },
            {  0x1.9ec8d1c94e85bp-144, -0x1.670e9d4784ec6p-201,  0x1.79fe5954939a2p-255,  0x1.82e418d9b0c9ep-311 },
            { -0x1.0dc59c716d91fp-133, -0x1.419e3fad3f031p-188, -0x1.d9d7ed1981ffcp-244,  0x1.345ea5d66a84bp-300 },
            {  0x1.3981254dd0d52p-123, -0x1.2b1f4c8015a2fp-177, -0x1.d82af23edb6dbp-231,  0x1.a1cd20123a99bp-285 },
            { -0x1.434d2e783f5bcp-113, -0x1.0b87b91be9affp-167, -0x1.c89db1796db75p-224,  0x1.8923b7699c8bep-278 },
            {  0x1.259f98b4358adp-103,  0x1.eaf8c39dd9bc5p-157, -0x1.6e29990a26fb6p-211, -0x1.2d867809b5568p-267 },
            { -0x1.d1ab1c2dccea3p-94,  -0x1.054d0c78aea14p-149,  0x1.196bf16c33a56p-203, -0x1.f0e65ed04d346p-257 },
            {  0x1.3f3ccdd165fa9p-84,  -0x1.58ddadf344487p-139, -0x1.e8ed8001ad67ep-193,  0x1.80a5edffcced7p-247 },
            { -0x1.761b41316381ap-75,   0x1.3423c7d91404fp-130, -0x1.e6135bfc1194ap-185,  0x1.ba7b1a3077b39p-239 },
            {  0x1.71b8ef6dcf572p-66,  -0x1.d043ae40c4647p-120,  0x1.486121e81d5fep-176, -0x1.2d4ba8e1e64c7p-230 },
            { -0x1.2f49b46814157p-57,  -0x1.2650f61dbdcb4p-112,  0x1.69502917cbf3bp-166, -0x1.e35fbddac4553p-223 },
            {  0x1.952c77030ad4ap-49,   0x1.ac981465ddc6cp-103, -0x1.588b72e53bc5fp-165,  0x1.7079e8909271ap-221 },
            { -0x1.ae7f3e733b81fp-41,  -0x1.1d8656b0ee8cbp-97,   0x1.6e142a138f825p-157, -0x1.43c0c38ccdcc6p-212 },
            {  0x1.6124613a86d09p-33,   0x1.f28e0cc748ebep-87,  -0x1.7b2c4c8a840bcp-141,  0x1.c71cca1034c07p-195 },
            { -0x1.ae64567f544e4p-26,   0x1.c062e06d1f209p-80,  -0x1.c7880adcbc46ep-136,  0x1.5553a6f0fed60p-190 },
            {  0x1.71de3a556c734p-19,  -0x1.c154f8ddc6c00p-73,   0x1.71de3a556c734p-127, -0x1.c154f8ddc6c00p-181 },
            { -0x1.a01a01a01a01ap-13,  -0x1.a01a01a01a01ap-73,  -0x1.a01a01a01a01ap-133, -0x1.a01a01a01a01ap-193 },
            {  0x1.1111111111111p-7,    0x1.1111111111111p-63,   0x1.1111111111111p-119,  0x1.1111111111111p-175 },
            { -0x1.5555555555555p-3,   -0x1.5555555555555p-57,  -0x1.5555555555555p-111, -0x1.5555555555555p-165 }
        };

        static constexpr f256 cos_coeffs[] = {
            {  0x1.091b406b6ff26p-203,  0x1.e973637973b18p-257, -0x1.1e38136f0edcap-311, -0x1.7ab33e52a1d28p-366 },
            { -0x1.240804f659510p-192, -0x1.8b291b93c9718p-246, -0x1.096c752f5341fp-301,  0x1.c12972a70641ep-355 },
            {  0x1.272b1b03fec6ap-181,  0x1.3f67cc9f9fdb8p-235, -0x1.71dcd047354c9p-289, -0x1.c3f29289464c4p-346 },
            { -0x1.10af527530de8p-170, -0x1.b626c912ee5c8p-225, -0x1.349f032c6e859p-279,  0x1.ec616617f45c6p-333 },
            {  0x1.ca8ed42a12ae3p-160,  0x1.a07244abad2abp-224,  0x1.facdac6fb71b7p-278, -0x1.ca2f486d514e1p-339 },
            { -0x1.5d4acb9c0c3abp-149,  0x1.6ec2c8f5b13b2p-205, -0x1.e2860aaa59188p-259,  0x1.866eba0408569p-313 },
            {  0x1.df983290c2ca9p-139,  0x1.5835c6895393bp-194, -0x1.0578f45b1aaaep-249, -0x1.281508688972dp-303 },
            { -0x1.2710231c0fd7ap-128, -0x1.3f8a2b4af9d6bp-184, -0x1.c32215a9f317ep-238,  0x1.d451e158a1205p-293 },
            {  0x1.434d2e783f5bcp-118,  0x1.0b87b91be9affp-172,  0x1.c89db1796db75p-229, -0x1.8923b7699c8bep-283 },
            { -0x1.3932c5047d60ep-108, -0x1.832b7b530a627p-162, -0x1.5d2c61f6d124cp-218, -0x1.f192b328d82c4p-272 },
            {  0x1.0a18a2635085dp-98,   0x1.b9e2e28e1aa54p-153,  0x1.a8549a9d99586p-207, -0x1.141dcc8cc5668p-266 },
            { -0x1.88e85fc6a4e5ap-89,   0x1.71c37ebd16540p-143, -0x1.494676265a364p-197,  0x1.397b40007db79p-253 },
            {  0x1.f2cf01972f578p-80,  -0x1.9ada5fcc1ab14p-135,  0x1.440ce7fd610dcp-189, -0x1.26fcbc204fcd1p-243 },
            { -0x1.0ce396db7f853p-70,   0x1.aebcdbd20331cp-124,  0x1.38a88578b4d75p-178, -0x1.c0fbc29694fb8p-233 },
            {  0x1.e542ba4020225p-62,   0x1.ea72b4afe3c2fp-120, -0x1.44020dfd65c8cp-174, -0x1.6e69b50fc88abp-231 },
            { -0x1.6827863b97d97p-53,  -0x1.eec01221a8b0bp-107,  0x1.568798662118bp-161, -0x1.f00d8b9e49291p-222 },
            {  0x1.ae7f3e733b81fp-45,   0x1.1d8656b0ee8cbp-101, -0x1.6e142a138f825p-161,  0x1.43c0c38ccdcc6p-216 },
            { -0x1.93974a8c07c9dp-37,  -0x1.05d6f8a2efd1fp-92,  -0x1.3aa3346236a5dp-147, -0x1.d75f096ea801ep-201 },
            {  0x1.1eed8eff8d898p-29,  -0x1.2aec959e14c06p-83,   0x1.2fb0073dd2d9ep-139,  0x1.c71d90b4ab715p-193 },
            { -0x1.27e4fb7789f5cp-22,  -0x1.cbbc05b4fa99ap-76,   0x1.c6d278883e8f5p-132, -0x1.95567d3a50ccep-186 },
            {  0x1.a01a01a01a01ap-16,   0x1.a01a01a01a01ap-76,   0x1.a01a01a01a01ap-136,  0x1.a01a01a01a01ap-196 },
            { -0x1.6c16c16c16c17p-10,   0x1.f49f49f49f49fp-65,   0x1.27d27d27d27d2p-119,  0x1.f49f49f49f49fp-173 },
            {  0x1.5555555555555p-5,    0x1.5555555555555p-59,   0x1.5555555555555p-113,  0x1.5555555555555p-167 },
            { -0x1.0000000000000p-1,    0x0.0p+0,                0x0.0p+0,                0x0.0p+0 }
        };

        const f256 t = r * r;

        f256 ps = sin_coeffs[0];
        for (std::size_t i = 1; i < sizeof(sin_coeffs) / sizeof(sin_coeffs[0]); ++i)
            ps = ps * t + sin_coeffs[i];
        s_out = r + r * t * ps;

        f256 pc = cos_coeffs[0];
        for (std::size_t i = 1; i < sizeof(cos_coeffs) / sizeof(cos_coeffs[0]); ++i)
            pc = pc * t + cos_coeffs[i];
        c_out = f256{ 1.0 } + t * pc;
    }
}

// exp
FORCE_INLINE f256 ldexp(const f256& a, int e)
{
    const double s = std::ldexp(1.0, e);
    return _f256_detail::renorm(a.x0 * s, a.x1 * s, a.x2 * s, a.x3 * s);
}
FORCE_INLINE f256 exp(const f256& x)
{
    if (!std::isfinite(x.x0))
        return f256{ std::exp(x.x0), 0.0, 0.0, 0.0 };

    if (x.x0 > 709.782712893384)
        return f256{ std::numeric_limits<double>::max(), 0.0, 0.0, 0.0 };

    if (x.x0 < -745.133219101941)
        return f256{ 0.0, 0.0, 0.0, 0.0 };

    const f256 t = x * _f256_const::inv_ln2;
    const f256 kf = nearbyint(t);
    const double kd = kf.x0;

    if (!std::isfinite(kd) || std::fabs(kd) > 9.0e15)
        return f256{ std::exp(static_cast<double>(x)), 0.0, 0.0, 0.0 };

    const int k = (int)kd;
    const f256 r = x - kf * _f256_const::ln2;

    if (abs(r) > _f256_const::ln2 * 0.5)
        return f256{ std::exp(static_cast<double>(x)), 0.0, 0.0, 0.0 };

    return ldexp(_f256_detail::f256_exp_kernel_ln2_half(r), k);
}
FORCE_INLINE f256 exp2(const f256& x)
{
    return exp(x * _f256_const::ln2);
}

// log
FORCE_INLINE double log_as_double(f256 a) noexcept
{
    const double hi = a.x0;
    if (!(hi > 0.0))
        return std::log(static_cast<double>(a));

    const double lo = (a.x1 + a.x2) + a.x3;
    return std::log(hi) + std::log1p(lo / hi);
}
FORCE_INLINE f256 log(const f256& a)
{
    if (a.x0 <= 0.0)
    {
        if (a.x0 == 0.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256{ -std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
        return f256{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    // Newton solve exp(y)=a
    f256 y{ std::log(a.x0), 0.0, 0.0, 0.0 };
    for (int i = 0; i < 8; ++i)
    {
        const f256 ey = exp(y);
        y = y + (a - ey) / ey;
    }
    return y;
}
FORCE_INLINE f256 log2(const f256& a)
{
    return log(a) / _f256_const::ln2;
}
FORCE_INLINE f256 log10(const f256& a)
{
    return log(a) / _f256_const::ln10;
}

// pow
inline f256 pow(const f256& a, const f256& b)
{
    // handle integer exponent for negative bases
    if (a.x0 < 0.0)
    {
        const double bd = static_cast<double>(b);
        const double rd = std::round(bd);
        if (std::fabs(bd - rd) <= std::ldexp(1.0, -40))
        {
            const long long n = (long long)rd;
            f256 base = -a;
            f256 res{ 1.0 };
            long long e = (n < 0) ? -n : n;
            while (e)
            {
                if (e & 1LL) res *= base;
                base *= base;
                e >>= 1LL;
            }
            if (n < 0) res = inv(res);
            return (n & 1LL) ? -res : res;
        }
        return f256{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
    }

    return exp(b * log(a));
}
inline f256 pow10_256(int k)
{
    if (k == 0) return f256{ 1.0 };

    int n = (k >= 0) ? k : -k;

    if (n <= 16) {
        f256 r = f256{ 1.0 };
        const f256 ten = f256{ 10.0, 0.0, 0.0, 0.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f256{ 1.0 } / r);
    }

    f256 r = f256{ 1.0 };
    f256 base = f256{ 10.0, 0.0, 0.0, 0.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f256{ 1.0 } / r);
}

// trig
inline bool sincos(const f256& x, f256& s_out, f256& c_out)
{
    if (!std::isfinite(x.x0))
    {
        s_out = f256{ std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 };
        c_out = s_out;
        return false;
    }

    long long n = 0;
    f256 r{};
    if (!_f256_detail::f256_rem_pio2(x, n, r))
        return false;

    f256 sr{}, cr{};
    _f256_detail::f256_sincos_kernel_pio4(r, sr, cr);

    switch ((int)(n & 3LL))
    {
    case 0: s_out = sr;  c_out = cr;  break;
    case 1: s_out = cr;  c_out = -sr; break;
    case 2: s_out = -sr; c_out = -cr; break;
    default: s_out = -cr; c_out = sr;  break;
    }

    return true;
}
inline f256 sin(const f256& x)
{
    f256 s{}, c{};
    if (sincos(x, s, c))
        return s;
    return f256{ std::sin(static_cast<double>(x)) };
}
inline f256 cos(const f256& x)
{
    f256 s{}, c{};
    if (sincos(x, s, c))
        return c;
    return f256{ std::cos(static_cast<double>(x)) };
}
inline f256 tan(const f256& x)
{
    f256 s{}, c{};
    if (sincos(x, s, c))
        return s / c;
    return f256{ std::tan(static_cast<double>(x)) };
}
inline f256 atan2(const f256& y, const f256& x)
{
    if (iszero(x))
    {
        if (iszero(y))
            return f256{ std::numeric_limits<double>::quiet_NaN() };

        return ispositive(y) ? _f256_const::pi_2 : -_f256_const::pi_2;
    }

    const f256 scale = std::max(abs(x), abs(y));
    const f256 xs = x / scale;
    const f256 ys = y / scale;

    f256 v{ std::atan2(static_cast<double>(y), static_cast<double>(x)) };

    for (int i = 0; i < 3; ++i)
    {
        f256 sv{}, cv{};
        if (!sincos(v, sv, cv))
        {
            const double vd = static_cast<double>(v);
            sv = f256{ std::sin(vd) };
            cv = f256{ std::cos(vd) };
        }

        const f256 f = xs * sv - ys * cv;
        const f256 fp = xs * cv + ys * sv;

        v = v - f / fp;
    }

    return v;
}
inline f256 atan(const f256& x)
{
    return atan2(x, f256{ 1.0 });
}
inline f256 asin(const f256& x)
{
    return atan2(x, sqrt(f256{ 1.0 } - x * x));
}
inline f256 acos(const f256& x)
{
    return atan2(sqrt(f256{ 1.0 } - x * x), x);
}

/// ======== Printing helpers ========

namespace _f256_detail
{
    BL_PUSH_PRECISE
    BL_PRINT_NOINLINE FORCE_INLINE f256 mul_by_double_print(f256 a, double b) noexcept
    {
        return a * b;
    }
    BL_PRINT_NOINLINE FORCE_INLINE f256 sub_by_double_print(f256 a, double b) noexcept
    {
        return a - b;
    }
    BL_POP_PRECISE

    struct f256_chars_result
    {
        char* ptr = nullptr;
        bool ok = false;
    };
    struct f256_print_expansion
    {
        double terms[64]{}; // small -> large
        int n = 0;
    };

    inline void print_expansion_set(f256_print_expansion& st, const f256& x) noexcept
    {
        double tmp[4] = { x.x3, x.x2, x.x1, x.x0 };
        st.n = _f256_detail::compress_expansion_zeroelim(4, tmp, st.terms);
    }
    inline bool print_expansion_is_zero(const f256_print_expansion& st) noexcept
    {
        return st.n <= 0;
    }
    inline f256 print_expansion_to_f256(const f256_print_expansion& st) noexcept
    {
        return from_expansion_fast(st.terms, st.n);
    }
    inline void print_expansion_scale(f256_print_expansion& st, double b) noexcept
    {
        double tmp[128]{};
        int n = _f256_detail::scale_expansion_zeroelim(st.n, st.terms, b, tmp);

        double comp[64]{};
        st.n = _f256_detail::compress_expansion_zeroelim(n, tmp, comp);
        for (int i = 0; i < st.n; ++i)
            st.terms[i] = comp[i];
    }
    inline void print_expansion_add_double(f256_print_expansion& st, double b) noexcept
    {
        double term = b;
        double tmp[128]{};
        int n = _f256_detail::fast_expansion_sum_zeroelim(st.n, st.terms, 1, &term, tmp);

        double comp[64]{};
        st.n = _f256_detail::compress_expansion_zeroelim(n, tmp, comp);
        for (int i = 0; i < st.n; ++i)
            st.terms[i] = comp[i];
    }
    inline uint32_t print_expansion_take_uint(f256_print_expansion& st, uint32_t max_value) noexcept
    {
        f256 approx = print_expansion_to_f256(st);
        long long value = (long long)floor(approx).x0;

        if (value < 0) value = 0;
        else if (value > (long long)max_value) value = (long long)max_value;

        print_expansion_add_double(st, -(double)value);
        return (uint32_t)value;
    }

    inline void normalize10(const f256& x, f256& m, int& exp10)
    {
        if (iszero(x)) { m = f256{}; exp10 = 0; return; }

        f256 ax = abs(x);

        int e2 = 0;
        (void)std::frexp(ax.x0, &e2);
        int e10 = (int)std::floor((e2 - 1) * 0.30102999566398114);

        m = ax * pow10_256(-e10);
        while (m >= f256{ 10.0, 0.0, 0.0, 0.0 }) { m = m / f256{ 10.0, 0.0, 0.0, 0.0 }; ++e10; }
        while (m < f256{1.0}) { m = m * f256{ 10.0, 0.0, 0.0, 0.0 }; --e10; }
        exp10 = e10;
    }
    inline int emit_uint_rev_buf(char* dst, f256 n)
    {
        const f256 base = f256{ 1000000000.0, 0.0, 0.0, 0.0 };

        int len = 0;

        if (n < f256{ 10.0, 0.0, 0.0, 0.0 }) {
            int d = (int)n.x0;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            dst[len++] = char('0' + d);
            return len;
        }

        while (n >= base) {
            f256 q = floor(n / base);
            f256 r = n - q * base;

            long long chunk = (long long)std::floor(r.x0);
            if (chunk >= 1000000000LL) { chunk -= 1000000000LL; q += 1.0; }
            if (chunk < 0) chunk = 0;

            for (int i = 0; i < 9; ++i) {
                int d = int(chunk % 10);
                dst[len++] = char('0' + d);
                chunk /= 10;
            }

            n = q;
        }

        long long last = (long long)std::floor(n.x0);
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
    inline f256_chars_result append_exp10_to_chars_f256(char* p, char* end, int e10) noexcept
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


    using biguint = fltx_common::exact_decimal::biguint;

    struct exact_traits
    {
        using value_type = f256;
        static constexpr int limb_count = 4;
        static constexpr int significand_bits = 212;

        static double limb(const value_type& x, int index) noexcept
        {
            switch (index)
            {
            case 0: return x.x0;
            case 1: return x.x1;
            case 2: return x.x2;
            default: return x.x3;
            }
        }
        static value_type zero(bool neg = false) noexcept
        {
            return neg ? value_type{ -0.0, 0.0, 0.0, 0.0 } : value_type{ 0.0, 0.0, 0.0, 0.0 };
        }
        static value_type infinity(bool neg = false) noexcept
        {
            const value_type inf = std::numeric_limits<value_type>::infinity();
            return neg ? -inf : inf;
        }
        static value_type pack_from_significand(const biguint& q, int e2, bool neg) noexcept
        {
            const std::uint64_t c3 = q.get_bits(0, 53);
            const std::uint64_t c2 = q.get_bits(53, 53);
            const std::uint64_t c1 = q.get_bits(106, 53);
            const std::uint64_t c0 = q.get_bits(159, 53);

            const double x0 = c0 ? std::ldexp(static_cast<double>(c0), e2 - 52) : 0.0;
            const double x1 = c1 ? std::ldexp(static_cast<double>(c1), e2 - 105) : 0.0;
            const double x2 = c2 ? std::ldexp(static_cast<double>(c2), e2 - 158) : 0.0;
            const double x3 = c3 ? std::ldexp(static_cast<double>(c3), e2 - 211) : 0.0;

            f256 out = renorm(x0, x1, x2, x3);
            if (neg)
                out = -out;
            return out;
        }
    };

    inline bool exact_scientific_digits(const f256& x, int sig, std::string& digits, int& exp10)
    {
        return fltx_common::exact_decimal::exact_scientific_digits<exact_traits>(x, sig, digits, exp10);
    }
    inline f256 exact_decimal_to_f256(const biguint& coeff, int dec_exp, bool neg) noexcept
    {
        return fltx_common::exact_decimal::exact_decimal_to_value<exact_traits>(coeff, dec_exp, neg);
    }


    inline f256_chars_result emit_fixed_dec_to_chars(char* first, char* last, f256 x, int prec, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (prec < 0) prec = 0;

        const bool neg = (x.x0 < 0.0);
        if (neg) x = -x;
        x = renorm(x.x0, x.x1, x.x2, x.x3);

        f256 ip = floor(x);
        f256 fp = x - ip;

        if (fp >= f256{ 1.0 }) { fp -= 1.0; ip += 1.0; }
        else if (fp < f256{}) { fp = f256{}; }

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

            f256_print_expansion fp_exp;
            print_expansion_set(fp_exp, fp);

            int written = 0;
            const int full = prec / 9;
            const int rem = prec - full * 9;

            for (int c = 0; c < full; ++c) {
                print_expansion_scale(fp_exp, kPow10[9]);
                uint32_t chunk = print_expansion_take_uint(fp_exp, 999999999u);

                for (int i = 8; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += 9;
            }

            if (rem > 0) {
                print_expansion_scale(fp_exp, kPow10[rem]);
                uint32_t chunk = print_expansion_take_uint(fp_exp, (uint32_t)kPow10[rem] - 1u);

                for (int i = rem - 1; i >= 0; --i) {
                    frac[written + i] = char('0' + (chunk % 10u));
                    chunk /= 10u;
                }
                written += rem;
            }

            f256_print_expansion round_exp = fp_exp;
            print_expansion_scale(round_exp, 10.0);
            int next = (int)print_expansion_take_uint(round_exp, 9u);

            const int last_digit = frac[prec - 1] - '0';
            bool round_up = false;
            if (next > 5) round_up = true;
            else if (next < 5) round_up = false;
            else {
                const bool gt_half = !print_expansion_is_zero(round_exp);
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
                    ip += 1.0;
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
    inline f256_chars_result emit_scientific_sig_to_chars(char* first, char* last, const f256& x, std::streamsize sig_digits, bool strip_trailing_zeros) noexcept
    {
        if (iszero(x)) {
            if (first >= last) return { first, false };
            *first = '0';
            return { first + 1, true };
        }

        if (sig_digits < 1) sig_digits = 1;

        const bool neg = (x.x0 < 0.0);
        const f256 v = neg ? -x : x;
        const int sig = static_cast<int>(sig_digits);

        std::string digits;
        int e = 0;
        if (!_f256_detail::exact_scientific_digits(v, sig, digits, e)) {
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
        auto er = append_exp10_to_chars_f256(ep, eend, e);
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
    inline f256_chars_result emit_scientific_to_chars(char* first, char* last, const f256& x, std::streamsize frac_digits, bool strip_trailing_zeros) noexcept
    {
        if (frac_digits < 0) frac_digits = 0;

        if (iszero(x)) {
            const bool neg = std::signbit(x.x0);
            int frac_len = strip_trailing_zeros ? 0 : (int)frac_digits;

            char exp_buf[16];
            char* ep = exp_buf;
            char* eend = exp_buf + sizeof(exp_buf);
            auto er = append_exp10_to_chars_f256(ep, eend, 0);
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

        return emit_scientific_sig_to_chars(first, last, x, frac_digits + 1, strip_trailing_zeros);
    }
    inline f256_chars_result to_chars(char* first, char* last, const f256& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false) noexcept
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

        f256 ax = (x.x0 < 0.0) ? -x : x;
        f256 m; int e10 = 0;
        normalize10(ax, m, e10);

        if (e10 >= -4 && e10 < sig) {
            const int frac = std::max(0, sig - (e10 + 1));
            return emit_fixed_dec_to_chars(first, last, x, frac, strip_trailing_zeros);
        }

        return emit_scientific_sig_to_chars(first, last, x, sig, strip_trailing_zeros);
    }

    using f256_format_kind = fltx_common::format_kind;
    using f256_parse_token = fltx_common::parse_token<_f256_detail::biguint>;

    struct f256_io_traits
    {
        using value_type = f256;
        using chars_result = f256_chars_result;
        using parse_token = f256_parse_token;

        static constexpr int max_parse_order = 330;
        static constexpr int min_parse_order = -400;

        static bool isnan(const value_type& x) noexcept { return bl::isnan(x); }
        static bool isinf(const value_type& x) noexcept { return bl::isinf(x); }
        static bool iszero(const value_type& x) noexcept { return bl::iszero(x); }
        static bool is_negative(const value_type& x) noexcept { return x.x0 < 0.0; }
        static value_type abs(const value_type& x) noexcept { return (x.x0 < 0.0) ? -x : x; }
        static value_type zero(bool neg = false) noexcept { return neg ? value_type{ -0.0, 0.0, 0.0, 0.0 } : value_type{ 0.0, 0.0, 0.0, 0.0 }; }
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
            return emit_scientific_sig_to_chars(first, last, x, precision, strip_trailing_zeros);
        }
        static value_type exact_decimal_to_value(const parse_token::coeff_type& coeff, int dec_exp, bool neg)
        {
            return _f256_detail::exact_decimal_to_f256(coeff, dec_exp, neg);
        }
    };

    template<typename Writer>
    FORCE_INLINE void write_chars_to_string(std::string& out, std::size_t cap, Writer writer)
    {
        fltx_common::write_chars_to_string<f256_chars_result>(out, cap, writer);
    }
    FORCE_INLINE const char* special_text_f256(const f256& x, bool uppercase = false) noexcept
    {
        return fltx_common::special_text<f256_io_traits>(x, uppercase);
    }
    FORCE_INLINE bool assign_special_string(std::string& out, const f256& x, bool uppercase = false) noexcept
    {
        return fltx_common::assign_special_string<f256_io_traits>(out, x, uppercase);
    }
    FORCE_INLINE void ensure_decimal_point(std::string& s)
    {
        fltx_common::ensure_decimal_point(s);
    }
    FORCE_INLINE void apply_stream_decorations(std::string& s, bool showpos, bool uppercase)
    {
        fltx_common::apply_stream_decorations(s, showpos, uppercase);
    }
    FORCE_INLINE bool write_stream_special(std::ostream& os, const f256& x, bool showpos, bool uppercase)
    {
        return fltx_common::write_stream_special<f256_io_traits>(os, x, showpos, uppercase);
    }
    FORCE_INLINE void format_to_string(std::string& out, const f256& x, int precision, f256_format_kind kind, bool strip_trailing_zeros = false)
    {
        fltx_common::format_to_string<f256_io_traits>(out, x, precision, kind, strip_trailing_zeros);
    }
    FORCE_INLINE void to_string_into(std::string& out, const f256& x, int precision, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
    {
        fltx_common::to_string_into<f256_io_traits>(out, x, precision, fixed, scientific, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific(std::string& out, const f256& x, std::streamsize prec, bool strip_trailing_zeros)
    {
        fltx_common::emit_scientific<f256_io_traits>(out, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_fixed_dec(std::string& out, const f256& x, int prec, bool strip_trailing_zeros)
    {
        fltx_common::emit_fixed_dec<f256_io_traits>(out, x, prec, strip_trailing_zeros);
    }
    FORCE_INLINE void emit_scientific_sig(std::string& out, const f256& x, std::streamsize sig_digits, bool strip_trailing_zeros)
    {
        fltx_common::emit_scientific_sig<f256_io_traits>(out, x, sig_digits, strip_trailing_zeros);
    }

    /// ======== Parsing helpers ========

    FORCE_INLINE bool valid_flt256_string(const char* s) noexcept
    {
        return fltx_common::valid_float_string(s);
    }
    FORCE_INLINE unsigned char ascii_lower_f256(char c) noexcept
    {
        return fltx_common::ascii_lower(c);
    }
    FORCE_INLINE const char* skip_ascii_space_f256(const char* p) noexcept
    {
        return fltx_common::skip_ascii_space(p);
    }
    
}

/// ======== Public string conversion wrappers ========

FORCE_INLINE bool parse_flt256(const char* s, f256& out, const char** endptr = nullptr) noexcept
{
    return fltx_common::parse_flt<_f256_detail::f256_io_traits>(s, out, endptr);
}
FORCE_INLINE f256 to_f256(const char* s) noexcept
{
    f256 ret;
    if (parse_flt256(s, ret))
        return ret;
    return f256{ 0.0 };
}
FORCE_INLINE f256 to_f256(const std::string& s) noexcept
{
    return to_f256(s.c_str());
}
FORCE_INLINE std::string to_string(const f256& x, int precision = std::numeric_limits<f256>::digits10, bool fixed = false, bool scientific = false, bool strip_trailing_zeros = false)
{
    std::string out;
    _f256_detail::to_string_into(out, x, precision, fixed, scientific, strip_trailing_zeros);
    return out;
}

/// ======== Stream output ========

inline std::ostream& operator<<(std::ostream& os, const f256& x)
{
    return fltx_common::write_to_stream<_f256_detail::f256_io_traits>(os, x);
}

/// ======== Literals ========

FORCE_INLINE constexpr f256 operator""_qd(unsigned long long v) noexcept {
    return to_f256(static_cast<uint64_t>(v));
}
FORCE_INLINE constexpr f256 operator""_qd(long double v) noexcept {
    return f256{ static_cast<double>(v) };
}

} // namespace bl
