/**
 * f256.h - f256 quad-double type and core arithmetic.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_INCLUDED
#define F256_INCLUDED

#include "fltx_common_math.h"
#include <numbers>

#if !defined(BL_F256_HAS_SSE2)
#  if !defined(__EMSCRIPTEN__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && (_M_IX86_FP >= 2)))
#    define BL_F256_HAS_SSE2 1
#  else
#    define BL_F256_HAS_SSE2 0
#  endif
#endif

#if !defined(BL_F256_HAS_NEON)
#  if !defined(__EMSCRIPTEN__) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#    define BL_F256_HAS_NEON 1
#  else
#    define BL_F256_HAS_NEON 0
#  endif
#endif

#if !defined(BL_F256_ENABLE_SIMD)
#  if defined(BL_F256_ENABLE_TRIG_SIMD)
#    define BL_F256_ENABLE_SIMD BL_F256_ENABLE_TRIG_SIMD
#  elif BL_F256_HAS_SSE2 || BL_F256_HAS_NEON
#    define BL_F256_ENABLE_SIMD 1
#  else
#    define BL_F256_ENABLE_SIMD 0
#  endif
#endif

#if !defined(BL_F256_ENABLE_TRIG_SIMD)
#  define BL_F256_ENABLE_TRIG_SIMD BL_F256_ENABLE_SIMD
#endif

#if BL_F256_ENABLE_SIMD
#  if BL_F256_HAS_SSE2
#    include <emmintrin.h>
#  elif BL_F256_HAS_NEON
#    include <arm_neon.h>
#  else
#    error "BL_F256_ENABLE_SIMD requires SSE2 or AArch64 NEON support."
#  endif
#endif

namespace bl {

struct f128_s;
struct f128;
struct f256_s;

namespace detail::_f256
{
    using detail::fp::absd;
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::magnitude_u64;
    using detail::fp::quick_two_sum_precise;
    using detail::fp::uint64_to_exact_double_pair;
    using detail::fp::two_prod_precise;
    using detail::fp::two_sum_precise;
    using detail::fp::signbit_constexpr;
    using detail::fp::fabs_constexpr;
    using detail::fp::floor_constexpr;
    using detail::fp::ceil_constexpr;
    using detail::fp::fmod_constexpr;
    using detail::fp::sqrt_seed_constexpr;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::frexp_exponent_constexpr;

    BL_FORCE_INLINE constexpr bool f256_runtime_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_SIMD
        return !bl::use_constexpr_math();
        #else
        return false;
        #endif
    }
    BL_FORCE_INLINE constexpr bool f256_runtime_trig_simd_enabled() noexcept
    {
        #if BL_F256_ENABLE_TRIG_SIMD
        return !bl::use_constexpr_math();
        #else
        return false;
        #endif
    }

#if BL_F256_ENABLE_SIMD
    #if BL_F256_HAS_SSE2
    using f256_simd2d = __m128d;
    #elif BL_F256_HAS_NEON
    using f256_simd2d = float64x2_t;
    #endif

    BL_FORCE_INLINE f256_simd2d f256_simd_set(double lane0, double lane1) noexcept
    {
        #if BL_F256_HAS_SSE2
        return _mm_set_pd(lane1, lane0);
        #else
        return vsetq_lane_f64(lane1, vdupq_n_f64(lane0), 1);
        #endif
    }
    BL_FORCE_INLINE f256_simd2d f256_simd_splat(double value) noexcept
    {
        #if BL_F256_HAS_SSE2
        return _mm_set1_pd(value);
        #else
        return vdupq_n_f64(value);
        #endif
    }
    BL_FORCE_INLINE void f256_simd_store(f256_simd2d value, double& lane0, double& lane1) noexcept
    {
        alignas(16) double lanes[2];
        #if BL_F256_HAS_SSE2
        _mm_storeu_pd(lanes, value);
        #else
        vst1q_f64(lanes, value);
        #endif
        lane0 = lanes[0];
        lane1 = lanes[1];
    }
    BL_FORCE_INLINE f256_simd2d f256_simd_add(f256_simd2d a, f256_simd2d b) noexcept
    {
        #if BL_F256_HAS_SSE2
        return _mm_add_pd(a, b);
        #else
        return vaddq_f64(a, b);
        #endif
    }
    BL_FORCE_INLINE f256_simd2d f256_simd_sub(f256_simd2d a, f256_simd2d b) noexcept
    {
        #if BL_F256_HAS_SSE2
        return _mm_sub_pd(a, b);
        #else
        return vsubq_f64(a, b);
        #endif
    }
    BL_FORCE_INLINE f256_simd2d f256_simd_mul(f256_simd2d a, f256_simd2d b) noexcept
    {
        #if BL_F256_HAS_SSE2
        return _mm_mul_pd(a, b);
        #else
        return vmulq_f64(a, b);
        #endif
    }
    BL_FORCE_INLINE void f256_simd_two_prod_precise(f256_simd2d a, f256_simd2d b, f256_simd2d& p, f256_simd2d& e) noexcept
    {
        p = f256_simd_mul(a, b);

        #if BL_F256_HAS_NEON
        e = vfmaq_f64(f256_simd_sub(f256_simd_splat(0.0), p), a, b);
        #else
        const f256_simd2d split = f256_simd_splat(134217729.0);
        const f256_simd2d a_scaled = f256_simd_mul(a, split);
        const f256_simd2d b_scaled = f256_simd_mul(b, split);

        const f256_simd2d a_hi = f256_simd_sub(a_scaled, f256_simd_sub(a_scaled, a));
        const f256_simd2d b_hi = f256_simd_sub(b_scaled, f256_simd_sub(b_scaled, b));
        const f256_simd2d a_lo = f256_simd_sub(a, a_hi);
        const f256_simd2d b_lo = f256_simd_sub(b, b_hi);

        e = f256_simd_add(
            f256_simd_add(f256_simd_sub(f256_simd_mul(a_hi, b_hi), p), f256_simd_mul(a_hi, b_lo)),
            f256_simd_add(f256_simd_mul(a_lo, b_hi), f256_simd_mul(a_lo, b_lo))
        );
        #endif
    }
    BL_FORCE_INLINE void f256_simd_store_array(f256_simd2d value, double* lanes) noexcept
    {
        #if BL_F256_HAS_SSE2
        _mm_storeu_pd(lanes, value);
        #else
        vst1q_f64(lanes, value);
        #endif
    }
    BL_FORCE_INLINE void f256_simd_two_sum(f256_simd2d a, f256_simd2d b, f256_simd2d& s, f256_simd2d& e) noexcept
    {
        s = f256_simd_add(a, b);
        const f256_simd2d bb = f256_simd_sub(s, a);
        e = f256_simd_add(f256_simd_sub(a, f256_simd_sub(s, bb)), f256_simd_sub(b, bb));
    }
#endif

    // Shewchuk-style expansion sum, expansions sorted by increasing magnitude (small -> large)
    BL_FORCE_INLINE constexpr int fast_expansion_sum_zeroelim(int elen, const double* e, int flen, const double* f, double* h) noexcept
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

        if (detail::fp::absd(enow) < detail::fp::absd(fnow)) { Q = enow; ++eindex; enow = (eindex < elen) ? e[eindex] : 0.0; }
        else { Q = fnow; ++findex; fnow = (findex < flen) ? f[findex] : 0.0; }

        while (eindex < elen && findex < flen) {
            if (detail::fp::absd(enow) < detail::fp::absd(fnow)) {
                detail::fp::two_sum_precise(Q, enow, Qnew, hh);
                ++eindex;
                enow = (eindex < elen) ? e[eindex] : 0.0;
            }
            else {
                detail::fp::two_sum_precise(Q, fnow, Qnew, hh);
                ++findex;
                fnow = (findex < flen) ? f[findex] : 0.0;
            }

            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (eindex < elen) {
            detail::fp::two_sum_precise(Q, e[eindex], Qnew, hh);
            ++eindex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        while (findex < flen) {
            detail::fp::two_sum_precise(Q, f[findex], Qnew, hh);
            ++findex;
            if (hh != 0.0) h[hindex++] = hh;
            Q = Qnew;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }
    BL_FORCE_INLINE constexpr int scale_expansion_zeroelim(int elen, const double* e, double b, double* h) noexcept
    {
        int hindex = 0;
        if (elen == 0 || b == 0.0) return 0;

        double Q{}, sum{}, hh{};
        double product1{}, product0{};

        detail::fp::two_prod_precise(e[0], b, product1, product0);
        Q = product1;
        if (product0 != 0.0) h[hindex++] = product0;

        for (int i = 1; i < elen; ++i) {
            detail::fp::two_prod_precise(e[i], b, product1, product0);

            detail::fp::two_sum_precise(Q, product0, sum, hh);
            if (hh != 0.0) h[hindex++] = hh;

            detail::fp::quick_two_sum_precise(product1, sum, Q, hh);
            if (hh != 0.0) h[hindex++] = hh;
        }

        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }
    BL_FORCE_INLINE constexpr int compress_expansion_zeroelim(int elen, const double* e, double* h) noexcept
    {
        double g[40]{};

        if (elen <= 0) return 0;

        double Q = e[elen - 1];
        for (int i = elen - 2; i >= 0; --i) {
            double Qnew{}, q{};
            detail::fp::two_sum_precise(Q, e[i], Qnew, q);
            Q = Qnew;
            g[i + 1] = q;
        }
        g[0] = Q;

        int hindex = 0;
        Q = g[0];
        for (int i = 1; i < elen; ++i) {
            double Qnew{}, q{};
            detail::fp::two_sum_precise(Q, g[i], Qnew, q);
            if (q != 0.0) h[hindex++] = q;
            Q = Qnew;
        }
        if (Q != 0.0 || hindex == 0) h[hindex++] = Q;
        return hindex;
    }
}

BL_MSVC_NOINLINE constexpr f256_s operator+(const f256_s& a, const f256_s& b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator-(const f256_s& a, const f256_s& b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator*(const f256_s& a, const f256_s& b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator/(const f256_s& a, const f256_s& b) noexcept;

BL_MSVC_NOINLINE constexpr f256_s operator+(const f256_s& a, double b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator-(const f256_s& a, double b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator*(const f256_s& a, double b) noexcept;
BL_MSVC_NOINLINE constexpr f256_s operator/(const f256_s& a, double b) noexcept;

BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, float b) noexcept;
BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, float b) noexcept;

// raw storage type
struct f256_s
{
    double x0, x1, x2, x3; // largest -> smallest

    BL_FORCE_INLINE constexpr f256_s& operator=(f128_s x) noexcept;
    BL_FORCE_INLINE constexpr f256_s& operator=(double x) noexcept { x0 = x; x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this; }
    BL_FORCE_INLINE constexpr f256_s& operator=(float x) noexcept { x0 = static_cast<double>(x); x1 = 0.0; x2 = 0.0; x3 = 0.0; return *this; }

    BL_MSVC_NOINLINE constexpr f256_s& operator=(uint64_t u) noexcept;
    BL_MSVC_NOINLINE constexpr f256_s& operator=(int64_t v) noexcept;

    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_signed_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator=(T v) noexcept {
        return (*this = static_cast<int64_t>(v));
    }
    template<class T, std::enable_if_t<std::is_integral_v<T>&& std::is_unsigned_v<T> && (sizeof(T) < 8), int> = 0>
    BL_FORCE_INLINE constexpr f256_s& operator=(T v) noexcept {
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

    [[nodiscard]] explicit constexpr operator f128() const noexcept;
    [[nodiscard]] explicit constexpr operator f128_s()   const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return ((x0 + x1) + (x2 + x3)); }
    [[nodiscard]] explicit constexpr operator float()  const noexcept { return static_cast<float>(((x0 + x1) + (x2 + x3))); }
    [[nodiscard]] explicit constexpr operator int()    const noexcept { return static_cast<int>(((x0 + x1) + (x2 + x3))); }

    [[nodiscard]] constexpr f256_s operator+() const { return *this; }
    [[nodiscard]] constexpr f256_s operator-() const noexcept { return f256_s{ -x0, -x1, -x2, -x3 }; }

    [[nodiscard]] static constexpr f256_s eps() { return { 3.038581678643134e-64, 0.0, 0.0, 0.0 }; } // ~2^-211
};

// complete value type
struct f256 : public f256_s
{
    f256() = default;
    constexpr f256(double _x0, double _x1, double _x2, double _x3) noexcept : f256_s{ _x0, _x1, _x2, _x3 } {}
    constexpr f256(float  x) noexcept : f256_s{ ((double)x), 0.0 } {}
    constexpr f256(double x) noexcept : f256_s{ ((double)x), 0.0 } {}
    constexpr f256(int64_t  v) noexcept : f256_s{} { static_cast<f256_s&>(*this) = static_cast<int64_t>(v); }
    constexpr f256(uint64_t u) noexcept : f256_s{} { static_cast<f256_s&>(*this) = static_cast<uint64_t>(u); }
    constexpr f256(int32_t  v) noexcept : f256((int64_t)v) {}
    constexpr f256(uint32_t u) noexcept : f256((int64_t)u) {}
    constexpr f256(const char*);

    constexpr f256(f128_s f) noexcept;
    constexpr f256(const f256_s& f) noexcept : f256_s{ f.x0, f.x1, f.x2, f.x3 } {}

    using f256_s::operator=;

    [[nodiscard]] explicit constexpr operator f128_s() const noexcept;
    [[nodiscard]] explicit constexpr operator f128() const noexcept;
    [[nodiscard]] explicit constexpr operator double() const noexcept { return ((x0 + x1) + (x2 + x3)); }
    [[nodiscard]] explicit constexpr operator float() const noexcept { return static_cast<float>(((x0 + x1) + (x2 + x3))); }
};

}  // namespace bl

/// ============= Std::numeric_limits specialization =============

template<>
struct std::numeric_limits<bl::f256_s>
{
    static constexpr bool is_specialized = true;

    static constexpr bl::f256_s min()            noexcept { return { numeric_limits<double>::min(), 0.0, 0.0, 0.0 }; }
    static constexpr bl::f256_s max()            noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon(), 0.0, 0.0 }; }
    static constexpr bl::f256_s lowest()         noexcept { return { -numeric_limits<double>::max(), numeric_limits<double>::epsilon(), 0.0, 0.0 }; }
    static constexpr bl::f256_s highest()        noexcept { return { numeric_limits<double>::max(), -numeric_limits<double>::epsilon(), 0.0, 0.0 }; }

    static constexpr int  digits = 212;
    static constexpr int  digits10 = 63;
    static constexpr int  max_digits10 = 66;

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int  radix = 2;

    static constexpr bl::f256_s epsilon()        noexcept { return bl::f256_s::eps(); }
    static constexpr bl::f256_s round_error()    noexcept { return { 0.5, 0.0, 0.0, 0.0 }; }

    static constexpr int  min_exponent   = numeric_limits<double>::min_exponent;
    static constexpr int  min_exponent10 = numeric_limits<double>::min_exponent10;
    static constexpr int  max_exponent   = numeric_limits<double>::max_exponent;
    static constexpr int  max_exponent10 = numeric_limits<double>::max_exponent10;

    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool has_denorm_loss = false;

    static constexpr bl::f256_s infinity()       noexcept { return { numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 }; }
    static constexpr bl::f256_s quiet_NaN()      noexcept { return { numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0 }; }
    static constexpr bl::f256_s signaling_NaN()  noexcept { return { numeric_limits<double>::signaling_NaN(), 0.0, 0.0, 0.0 }; }
    static constexpr bl::f256_s denorm_min()     noexcept { return { numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0 }; }

    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;

    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

// apply same limits to f256
BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS(bl::f256, bl::f256_s)

/// ============= Std::numbers specialization =============

namespace std::numbers
{
    template<> inline constexpr bl::f256_s e_v<bl::f256_s>          = { 0x1.5bf0a8b145769p+1,  0x1.4d57ee2b1013ap-53, -0x1.618713a31d3e2p-109,  0x1.c5a6d2b53c26dp-163 };
    template<> inline constexpr bl::f256_s log2e_v<bl::f256_s>      = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56, -0x1.60bb8a5442ab9p-110, -0x1.4b52d3ba6d74dp-166 };
    template<> inline constexpr bl::f256_s log10e_v<bl::f256_s>     = { 0x1.bcb7b1526e50ep-2,  0x1.95355baaafad3p-57,  0x1.ee191f71a3012p-112,  0x1.7268808e8fcb5p-167 };
    template<> inline constexpr bl::f256_s pi_v<bl::f256_s>         = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53, -0x1.f1976b7ed8fbcp-109,  0x1.4cf98e804177dp-163 };
    template<> inline constexpr bl::f256_s inv_pi_v<bl::f256_s>     = { 0x1.45f306dc9c883p-2, -0x1.6b01ec5417056p-56, -0x1.6447e493ad4cep-110,  0x1.e21c820ff28b2p-164 };
    template<> inline constexpr bl::f256_s inv_sqrtpi_v<bl::f256_s> = { 0x1.20dd750429b6dp-1,  0x1.1ae3a914fed80p-57, -0x1.3cbbebf65f145p-112, -0x1.e0c574632f53ep-167 };
    template<> inline constexpr bl::f256_s ln2_v<bl::f256_s>        = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56,  0x1.7b57a079a1934p-111, -0x1.ace93a4ebe5d1p-165 };
    template<> inline constexpr bl::f256_s ln10_v<bl::f256_s>       = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53, -0x1.9ebae3ae0260cp-107, -0x1.2d10378be1cf1p-161 };
    template<> inline constexpr bl::f256_s sqrt2_v<bl::f256_s>      = { 0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54,  0x1.57d3e3adec175p-108,  0x1.2775099da2f59p-164 };
    template<> inline constexpr bl::f256_s sqrt3_v<bl::f256_s>      = { 0x1.bb67ae8584caap+0,  0x1.cec95d0b5c1e3p-54, -0x1.f11db689f2ccfp-110,  0x1.3da4798c720a6p-164 };
    template<> inline constexpr bl::f256_s inv_sqrt3_v<bl::f256_s>  = { 0x1.279a74590331cp-1,  0x1.34863e0792bedp-55, -0x1.a82f9e6c53222p-109, -0x1.cb0f41134253ap-163 };
    template<> inline constexpr bl::f256_s egamma_v<bl::f256_s>     = { 0x1.2788cfc6fb619p-1, -0x1.6cb90701fbfabp-58, -0x1.34a95e3133c51p-112,  0x1.9730064300f7dp-166 };
    template<> inline constexpr bl::f256_s phi_v<bl::f256_s>        = { 0x1.9e3779b97f4a8p+0, -0x1.f506319fcfd19p-55,  0x1.b906821044ed8p-109, -0x1.8bb1b5c0f272cp-165 };

    template<> inline constexpr bl::f256 e_v<bl::f256>          = bl::f256{ e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 log2e_v<bl::f256>      = bl::f256{ log2e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 log10e_v<bl::f256>     = bl::f256{ log10e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 pi_v<bl::f256>         = bl::f256{ pi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_pi_v<bl::f256>     = bl::f256{ inv_pi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_sqrtpi_v<bl::f256> = bl::f256{ inv_sqrtpi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 ln2_v<bl::f256>        = bl::f256{ ln2_v<bl::f256_s> };
    template<> inline constexpr bl::f256 ln10_v<bl::f256>       = bl::f256{ ln10_v<bl::f256_s> };
    template<> inline constexpr bl::f256 sqrt2_v<bl::f256>      = bl::f256{ sqrt2_v<bl::f256_s> };
    template<> inline constexpr bl::f256 sqrt3_v<bl::f256>      = bl::f256{ sqrt3_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_sqrt3_v<bl::f256>  = bl::f256{ inv_sqrt3_v<bl::f256_s> };
    template<> inline constexpr bl::f256 egamma_v<bl::f256>     = bl::f256{ egamma_v<bl::f256_s> };
    template<> inline constexpr bl::f256 phi_v<bl::f256>        = bl::f256{ phi_v<bl::f256_s> };
}


namespace bl {

// std-style conveniences: bl::numeric_limits and bl::numbers
using std::numeric_limits;
namespace numbers = std::numbers;

/// ============= Representation helpers and scalar conversions =============

namespace detail::_f256
{
    // quad-double normalization
    BL_FORCE_INLINE constexpr void three_sum(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        two_sum_precise(a, b, t1, t2);
        two_sum_precise(c, t1, a, t3);
        two_sum_precise(t2, t3, b, c);
    }
    BL_FORCE_INLINE constexpr void three_sum2(double& a, double& b, double& c) noexcept
    {
        double t1{}, t2{}, t3{};
        two_sum_precise(a, b, t1, t2);
        two_sum_precise(c, t1, a, t3);
        b = t2 + t3;
    }

    BL_FORCE_INLINE constexpr f256_s renorm(double c0, double c1, double c2, double c3) noexcept
    {
        double s0{}, s1{}, s2 = 0.0, s3 = 0.0;

        quick_two_sum_precise(c2, c3, s0, c3);
        quick_two_sum_precise(c1, s0, s0, c2);
        quick_two_sum_precise(c0, s0, c0, c1);

        s0 = c0;
        s1 = c1;

        if (s1 != 0.0)
        {
            quick_two_sum_precise(s1, c2, s1, s2);
            if (s2 != 0.0)
                quick_two_sum_precise(s2, c3, s2, s3);
            else
                quick_two_sum_precise(s1, c3, s1, s2);
        }
        else
        {
            quick_two_sum_precise(s0, c2, s0, s1);
            if (s1 != 0.0)
                quick_two_sum_precise(s1, c3, s1, s2);
            else
                quick_two_sum_precise(s0, c3, s0, s1);
        }

        return { s0, s1, s2, s3 };
    }
    BL_FORCE_INLINE constexpr f256_s renorm4(double c0, double c1, double c2, double c3) noexcept
    {
        double s, e;
        s = c2 + c3,  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        double s0 = c0, s1 = c1, s2 = 0.0, s3 = 0.0;

        if (c2 != 0.0)
        {
            if (s1 != 0.0) {
                s = s1 + c2; e = c2 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c2; e = c2 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0) {
                s = s2 + c3; e = c3 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c3; e = c3 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c3; e = c3 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }
    BL_FORCE_INLINE constexpr f256_s renorm5(double c0, double c1, double c2, double c3, double c4) noexcept
    {
        double s, e;

        s = c3 + c4;  e = c4 - (s - c3);  c3 = s;  c4 = e;
        s = c2 + c3;  e = c3 - (s - c2);  c2 = s;  c3 = e;
        s = c1 + c2;  e = c2 - (s - c1);  c1 = s;  c2 = e;
        s = c0 + c1;  e = c1 - (s - c0);  c0 = s;  c1 = e;

        double s0 = c0, s1 = c1, s2 = 0.0, s3 = 0.0;
        if (c2 != 0.0)
        {
            if (s1 != 0.0) {
                s = s1 + c2;
                e = c2 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c2;
                e = c2 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c3 != 0.0)
        {
            if (s2 != 0.0) {
                s = s2 + c3; e = c3 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c3; e = c3 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c3; e = c3 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        if (c4 != 0.0)
        {
            if (s3 != 0.0) {
                s3 += c4;
            }
            else if (s2 != 0.0) {
                s = s2 + c4; e = c4 - (s - s2);
                s2 = s; s3 = e;
            }
            else if (s1 != 0.0) {
                s = s1 + c4; e = c4 - (s - s1);
                s1 = s; s2 = e;
            }
            else {
                s = s0 + c4; e = c4 - (s - s0);
                s0 = s; s1 = e;
            }
        }

        return { s0, s1, s2, s3 };
    }

    // canonicalization helper
    BL_FORCE_INLINE constexpr f256_s canonicalize_math_result(f256_s value) noexcept
    {
        value.x3 = detail::fp::zero_low_fraction_bits_finite<8>(value.x3);
        return value;
    }
}

// constexpr/runtime declarations needed by public conversion wrappers
namespace detail::_f256_constexpr
{
    using namespace detail::_f256;

    // scalar conversion
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(uint64_t u) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s to_f256(int64_t v) noexcept;

    BL_FORCE_INLINE constexpr f256_s& assign(f256_s& out, uint64_t u) noexcept;
    BL_FORCE_INLINE constexpr f256_s& assign(f256_s& out, int64_t v) noexcept;

    // rounding and powers
    [[nodiscard]] inline constexpr f256_s floor(const f256_s& a);
    [[nodiscard]] inline constexpr f256_s ceil(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s trunc(const f256_s& a);
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s pow10_256(int k);

    // f256 <=> f256 arithmetic
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div(const f256_s& a, const f256_s& b) noexcept;

    // f256 <=> f64 arithmetic
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double(const f256_s& a, double b) noexcept;
}

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_NO_INLINE f256_s to_f256(uint64_t u) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s to_f256(int64_t v) noexcept;
    BL_NO_INLINE f256_s& assign(f256_s& out, uint64_t u) noexcept;
    BL_NO_INLINE f256_s& assign(f256_s& out, int64_t v) noexcept;
}

BL_MSVC_NOINLINE constexpr f256_s& f256_s::operator=(uint64_t u) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::assign(*this, u);

    return detail::_f256_runtime::assign(*this, u);
}
BL_MSVC_NOINLINE constexpr f256_s& f256_s::operator=(int64_t v) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::assign(*this, v);

    return detail::_f256_runtime::assign(*this, v);
}

[[nodiscard]] constexpr f256_s to_f256(double x) noexcept { return f256_s{ x, 0.0, 0.0, 0.0 }; }
[[nodiscard]] constexpr f256_s to_f256(float x) noexcept { return f256_s{ (double)x, 0.0, 0.0, 0.0 }; }
[[nodiscard]] constexpr f256_s to_f256(int32_t v) noexcept { return f256_s{ (double)v, 0.0, 0.0, 0.0 }; }
[[nodiscard]] constexpr f256_s to_f256(uint32_t v) noexcept { return f256_s{ (double)v, 0.0, 0.0, 0.0 }; }

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s to_f256(uint64_t u) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::to_f256(u);

    return detail::_f256_runtime::to_f256(u);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s to_f256(int64_t v) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::to_f256(v);

    return detail::_f256_runtime::to_f256(v);
}

// Remaining public f256 core wrappers are defined together after the constexpr implementations.
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s floor(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s ceil(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s trunc(const f256_s& a);
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow10_256(int k);

/// ============= Comparisons =============

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
        uint64_compare_terms(magnitude_u64(value), x0, x1, x2, x3);
        if (value < 0)
        {
            x0 = -x0;
            x1 = -x1;
        }
    }
}

// f256 <=> f256

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

// f64 <=> f256

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

// f32 <=> f256

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

// i32/u32 <=> f256

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

// i64/u64 <=> f256

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

/// ============= Classification =============

[[nodiscard]] BL_FORCE_INLINE constexpr bool isnan(const f256_s& a) noexcept { return detail::_f256::isnan(a.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isinf(const f256_s& a) noexcept { return detail::_f256::isinf(a.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool isfinite(const f256_s& x) noexcept { return detail::_f256::isfinite(x.x0); }
[[nodiscard]] BL_FORCE_INLINE constexpr bool iszero(const f256_s& a) noexcept { return a.x0 == 0 && a.x1 == 0 && a.x2 == 0 && a.x3 == 0; }
[[nodiscard]] BL_FORCE_INLINE constexpr bool ispositive(const f256_s& x) noexcept { return x.x0 > 0 || (x.x0 == 0 && (x.x1 > 0 || (x.x1 == 0 && (x.x2 > 0 || (x.x2 == 0 && x.x3 > 0))))); }

/// ============= Arithmetic operators =============

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s clamp(const f256_s& v, const f256_s& lo, const f256_s& hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s abs(const f256_s& a) noexcept
{
    return (a.x0 < 0.0) ? -a : a;
}

namespace detail::_f256_runtime
{
    [[nodiscard]] BL_NO_INLINE f256_s floor(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s ceil(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s trunc(const f256_s& a);
    [[nodiscard]] BL_NO_INLINE f256_s pow10_256(int k);
}


[[nodiscard]] BL_FORCE_INLINE constexpr f256_s recip(f256_s b) noexcept
{
    using namespace detail::_f256;

    constexpr f256_s one = f256_s{ 1.0 };

    double q0 = 1.0 / b.x0;
    f256_s r = one - (b * q0);

    double q1 = r.x0 / b.x0; r -= (b * q1);
    double q2 = r.x0 / b.x0; r -= (b * q2);
    double q3 = r.x0 / b.x0; r -= (b * q3);
    double q4 = r.x0 / b.x0;

    return renorm5(q0, q1, q2, q3, q4);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s inv(const f256_s& a) { return recip(a); }

/// ============= Core helpers =============

namespace detail::_f256
{
    BL_FORCE_INLINE constexpr f256_s add_scalar_precise(const f256_s& a, double b) noexcept
    {
        double s0{}, e0{};  two_sum_precise(a.x0, b, s0, e0);
        double s1{}, e1{};  two_sum_precise(a.x1, e0, s1, e1);
        double s2{}, e2{};  two_sum_precise(a.x2, e1, s2, e2);
        double s3{}, e3{};  two_sum_precise(a.x3, e2, s3, e3);

        return renorm5(s0, s1, s2, s3, e3);
    }
    BL_FORCE_INLINE constexpr f256_s from_expansion_fast(const double* h, int n) noexcept
    {
        if (n <= 0) return {};

        double comp[40]{};
        const int m = compress_expansion_zeroelim(n, h, comp);

        f256_s sum{};
        for (int i = 0; i < m; ++i)
            sum = add_scalar_precise(sum, comp[i]);

        return sum;
    }

    BL_FORCE_INLINE constexpr f256_s add_dd_qd(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};

        #if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            const f256_simd2d av = f256_simd_set(a.x0, a.x1);
            const f256_simd2d bv = f256_simd_set(b.x0, b.x1);
            f256_simd2d sv{}, ev{};
            f256_simd_two_sum(av, bv, sv, ev);
            f256_simd_store(sv, s0, s1);
            f256_simd_store(ev, e0, e1);
        }
        else
        #endif
        {
            two_sum_precise(a.x0, b.x0, s0, e0);
            two_sum_precise(a.x1, b.x1, s1, e1);
        }
        two_sum_precise(s1, e0, s1, e0);

        e0 += e1;

        if (e0 == 0.0)
            return renorm4(s0, s1, 0.0, 0.0);

        return renorm5(s0, s1, e0, 0.0, 0.0);
    }
    BL_FORCE_INLINE constexpr f256_s sub_dd_qd(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};

#if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            const f256_simd2d av = f256_simd_set(a.x0, a.x1);
            const f256_simd2d bv = f256_simd_set(-b.x0, -b.x1);
            f256_simd2d sv{}, ev{};
            f256_simd_two_sum(av, bv, sv, ev);
            f256_simd_store(sv, s0, s1);
            f256_simd_store(ev, e0, e1);
        }
        else
#endif
        {
            two_sum_precise(a.x0, -b.x0, s0, e0);
            two_sum_precise(a.x1, -b.x1, s1, e1);
        }
        two_sum_precise(s1, e0, s1, e0);

        e0 += e1;

        if (e0 == 0.0)
            return renorm4(s0, s1, 0.0, 0.0);

        return renorm5(s0, s1, e0, 0.0, 0.0);
    }
    BL_FORCE_INLINE constexpr f256_s add_qd_qd(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

#if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            const f256_simd2d a01 = f256_simd_set(a.x0, a.x1);
            const f256_simd2d b01 = f256_simd_set(b.x0, b.x1);
            const f256_simd2d a23 = f256_simd_set(a.x2, a.x3);
            const f256_simd2d b23 = f256_simd_set(b.x2, b.x3);
            f256_simd2d s01{}, e01{}, s23{}, e23{};
            f256_simd_two_sum(a01, b01, s01, e01);
            f256_simd_two_sum(a23, b23, s23, e23);
            f256_simd_store(s01, s0, s1);
            f256_simd_store(e01, e0, e1);
            f256_simd_store(s23, s2, s3);
            f256_simd_store(e23, e2, e3);
        }
        else
#endif
        {
            two_sum_precise(a.x0, b.x0, s0, e0);
            two_sum_precise(a.x1, b.x1, s1, e1);
            two_sum_precise(a.x2, b.x2, s2, e2);
            two_sum_precise(a.x3, b.x3, s3, e3);
        }
        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }
    BL_FORCE_INLINE constexpr f256_s sub_qd_qd(const f256_s& a, const f256_s& b) noexcept
    {
        double s0{}, e0{};
        double s1{}, e1{};
        double s2{}, e2{};
        double s3{}, e3{};

#if BL_F256_ENABLE_SIMD
        if (f256_runtime_simd_enabled())
        {
            const f256_simd2d a01 = f256_simd_set(a.x0, a.x1);
            const f256_simd2d b01 = f256_simd_set(-b.x0, -b.x1);
            const f256_simd2d a23 = f256_simd_set(a.x2, a.x3);
            const f256_simd2d b23 = f256_simd_set(-b.x2, -b.x3);
            f256_simd2d s01{}, e01{}, s23{}, e23{};
            f256_simd_two_sum(a01, b01, s01, e01);
            f256_simd_two_sum(a23, b23, s23, e23);
            f256_simd_store(s01, s0, s1);
            f256_simd_store(e01, e0, e1);
            f256_simd_store(s23, s2, s3);
            f256_simd_store(e23, e2, e3);
        }
        else
#endif
        {
            two_sum_precise(a.x0, -b.x0, s0, e0);
            two_sum_precise(a.x1, -b.x1, s1, e1);
            two_sum_precise(a.x2, -b.x2, s2, e2);
            two_sum_precise(a.x3, -b.x3, s3, e3);
        }
        two_sum_precise(s1, e0, s1, e0);
        three_sum(s2, e0, e1);
        three_sum2(s3, e0, e2);

        e0 += e1 + e3;

        if (e0 == 0.0)
            return renorm4(s0, s1, s2, s3);

        return renorm5(s0, s1, s2, s3, e0);
    }

    BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_fast(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double p0{}, e0{};
        double p1{}, e1{};
        double p2{}, e2{};
        double p3{}, e3{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, e01{};
            f256_simd2d p23{}, e23{};
            const f256_simd2d qv = f256_simd_splat(q);
            f256_simd_two_prod_precise(f256_simd_set(b.x0, b.x1), qv, p01, e01);
            f256_simd_two_prod_precise(f256_simd_set(b.x2, b.x3), qv, p23, e23);
            f256_simd_store(p01, p0, p1);
            f256_simd_store(e01, e0, e1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(e23, e2, e3);
        }
        else
        #endif
        {
            two_prod_precise(b.x0, q, p0, e0);
            two_prod_precise(b.x1, q, p1, e1);
            two_prod_precise(b.x2, q, p2, e2);
            two_prod_precise(b.x3, q, p3, e3);
        }

        double s0 = r.x0-p0; double v0 = s0-r.x0; double u0 = s0-v0; double w0 = r.x0-u0;  u0 = -p0 - v0;
        double s1 = r.x1-p1; double v1 = s1-r.x1; double u1 = s1-v1; double w1 = r.x1-u1;  u1 = -p1 - v1;
        double s2 = r.x2-p2; double v2 = s2-r.x2; double u2 = s2-v2; double w2 = r.x2-u2;  u2 = -p2 - v2;
        double s3 = r.x3-p3; double v3 = s3-r.x3; double u3 = s3-v3; double w3 = r.x3-u3;  u3 = -p3 - v3;

        double t0 = w0 + u0;
        double t1 = w1 + u1;
        double t2 = w2 + u2;
        double t3 = w3 + u3;

        double tail0 = t0 - e0; two_sum_precise(s1, tail0, s1, t0);
        double tail1 = t1 - e1; three_sum(s2, t0, tail1);
        double tail2 = t2 - e2; three_sum2(s3, t0, tail2);

        t0 = t0 + tail1 + t3 - e3;

        return renorm5(s0, s1, s2, s3, t0);
    }
    BL_FORCE_INLINE constexpr f256_s sub_mul_scalar_exact(const f256_s& r, const f256_s& b, double q) noexcept
    {
        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            const f256_simd2d qv = f256_simd_splat(q);
            f256_simd_two_prod_precise(f256_simd_set(b.x0, b.x1), qv, p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(b.x2, b.x3), qv, p23, q23);
            double ignored{};
            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, ignored);
        }
        else
        #endif
        {
            two_prod_precise(b.x0, q, p0, q0);
            two_prod_precise(b.x1, q, p1, q1);
            two_prod_precise(b.x2, q, p2, q2);
            p3 = b.x3 * q;
        }

        s0 = p0;
        two_sum_precise(q0, p1, s1, s2);
        three_sum(s2, q1, p2);
        three_sum2(q1, q2, p3);
        s3 = q1;
        s4 = q2 + p2;

        double c0{}, e0{};
        double c1{}, e1{};
        double c2{}, e2{};
        double c3{}, e3{};

        two_sum_precise(r.x0, -s0, c0, e0);
        two_sum_precise(r.x1, -s1, c1, e1);
        two_sum_precise(r.x2, -s2, c2, e2);
        two_sum_precise(r.x3, -s3, c3, e3);

        two_sum_precise(c1, e0, c1, e0);
        three_sum(c2, e0, e1);
        three_sum2(c3, e0, e2);

        e0 += e1 + e3 - s4;

        return renorm5(c0, c1, c2, c3, e0);
    }
}

namespace detail::_f256_runtime
{
    // f256 <=> f256 operations
    [[nodiscard]] BL_NO_INLINE f256_s add(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul(const f256_s& a, const f256_s& b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div(const f256_s& a, const f256_s& b) noexcept;

    // f256 <=> f64 operations
    [[nodiscard]] BL_NO_INLINE f256_s add_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s sub_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s mul_double(const f256_s& a, double b) noexcept;
    [[nodiscard]] BL_NO_INLINE f256_s div_double(const f256_s& a, double b) noexcept;
}

namespace detail::_f256
{
    // f256 <=> f256 force inlined operations
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (a.x2 == 0.0 && a.x3 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return add_dd_qd(a, b);

        return add_qd_qd(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (a.x2 == 0.0 && a.x3 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return sub_dd_qd(a, b);

        return sub_qd_qd(a, b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            f256_simd2d p45{}, q45{};
            f256_simd2d p67{}, q67{};
            f256_simd2d p89{}, q89{};

            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x0), f256_simd_set(b.x0, b.x1), p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x0), f256_simd_set(b.x0, b.x2), p23, q23);
            f256_simd_two_prod_precise(f256_simd_set(a.x1, a.x2), f256_simd_set(b.x1, b.x0), p45, q45);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(b.x3, b.x2), p67, q67);
            f256_simd_two_prod_precise(f256_simd_set(a.x2, a.x3), f256_simd_set(b.x1, b.x0), p89, q89);

            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, q3);
            f256_simd_store(p45, p4, p5);
            f256_simd_store(q45, q4, q5);
            f256_simd_store(p67, p6, p7);
            f256_simd_store(q67, q6, q7);
            f256_simd_store(p89, p8, p9);
            f256_simd_store(q89, q8, q9);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, b.x0, p0, q0);
            two_prod_precise(a.x0, b.x1, p1, q1);
            two_prod_precise(a.x1, b.x0, p2, q2);
            two_prod_precise(a.x0, b.x2, p3, q3);
            two_prod_precise(a.x1, b.x1, p4, q4);
            two_prod_precise(a.x2, b.x0, p5, q5);
            two_prod_precise(a.x0, b.x3, p6, q6);
            two_prod_precise(a.x1, b.x2, p7, q7);
            two_prod_precise(a.x2, b.x1, p8, q8);
            two_prod_precise(a.x3, b.x0, p9, q9);
        }

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + s2;

        return renorm5(p0, p1, s0, t0, t1);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_inline(const f256_s& a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return a / b.x0;

        const double inv_b0 = 1.0 / b.x0;

        const double q0 = a.x0 * inv_b0;
        f256_s r = sub_mul_scalar_exact(a, b, q0);

        const double q1 = r.x0 * inv_b0;
        r = sub_mul_scalar_fast(r, b, q1);

        const double q2 = r.x0 * inv_b0;
        r = sub_mul_scalar_fast(r, b, q2);

        const double q3 = r.x0 * inv_b0;
        r = sub_mul_scalar_fast(r, b, q3);

        const double q4 = r.x0 * inv_b0;

        return renorm5(q0, q1, q2, q3, q4);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sqr_inline(const f256_s& a) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
        double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
        double p6{}, p7{}, p8{}, p9{};
        double q6{}, q7{}, q8{}, q9{};
        double r0{}, r1{};
        double t0{}, t1{};
        double s0{}, s1{}, s2{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p34{}, q34{};
            f256_simd2d p67{}, q67{};

            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x0), f256_simd_set(a.x0, a.x1), p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(a.x2, a.x1), p34, q34);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), f256_simd_set(a.x3, a.x2), p67, q67);

            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p34, p3, p4);
            f256_simd_store(q34, q3, q4);
            f256_simd_store(p67, p6, p7);
            f256_simd_store(q67, q6, q7);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, a.x0, p0, q0);
            two_prod_precise(a.x0, a.x1, p1, q1);
            two_prod_precise(a.x0, a.x2, p3, q3);
            two_prod_precise(a.x1, a.x1, p4, q4);
            two_prod_precise(a.x0, a.x3, p6, q6);
            two_prod_precise(a.x1, a.x2, p7, q7);
        }
        p2 = p1;
        q2 = q1;
        p5 = p3;
        q5 = q3;

        three_sum(p1, p2, q0);
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);

        two_sum_precise(p2, p3, s0, t0);
        two_sum_precise(q1, p4, s1, t1);
        s2 = q2 + p5;
        two_sum_precise(s1, t0, s1, t0);
        s2 += (t0 + t1);

        p8 = p7;
        q8 = q7;
        p9 = p6;
        q9 = q6;

        two_sum_precise(q0, q3, q0, q3);
        two_sum_precise(q4, q5, q4, q5);
        two_sum_precise(p6, p7, p6, p7);
        two_sum_precise(p8, p9, p8, p9);

        two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
        two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
        two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

        two_sum_precise(q3, s1, t0, t1);
        t1 += q4;
        t1 += a.x1 * a.x3 + a.x2 * a.x2 + a.x3 * a.x1 + q6 + q7 + q8 + q9 + s2;

        return renorm5(p0, p1, s0, t0, t1);
    }

    // f256 <=> f64 force inlined operations
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_inline(const f256_s& a, double b) noexcept
    {
        double c0{}, c1{}, c2{}, c3{}, e{};

        two_sum_precise(a.x0, b, c0, e);
        if (e == 0.0) return renorm4(c0, a.x1, a.x2, a.x3);

        two_sum_precise(a.x1, e, c1, e);
        if (e == 0.0) return renorm4(c0, c1, a.x2, a.x3);

        two_sum_precise(a.x2, e, c2, e);
        if (e == 0.0) return renorm4(c0, c1, c2, a.x3);

        two_sum_precise(a.x3, e, c3, e);
        if (e == 0.0) return renorm4(c0, c1, c2, c3);

        return renorm5(c0, c1, c2, c3, e);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s add_double_inline(double a, const f256_s& b) noexcept
    {
        return add_double_inline(b, a);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_inline(const f256_s& a, double b) noexcept
    {
        return add_double_inline(a, -b);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s sub_double_inline(double a, const f256_s& b) noexcept
    {
        return add_double_inline(-b, a);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_inline(const f256_s& a, double b) noexcept
    {
        using namespace detail::_f256;

        double p0{}, p1{}, p2{}, p3{};
        double q0{}, q1{}, q2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        #if BL_F256_ENABLE_SIMD && BL_F256_HAS_NEON
        if (f256_runtime_simd_enabled())
        {
            f256_simd2d p01{}, q01{};
            f256_simd2d p23{}, q23{};
            const f256_simd2d bv = f256_simd_splat(b);
            f256_simd_two_prod_precise(f256_simd_set(a.x0, a.x1), bv, p01, q01);
            f256_simd_two_prod_precise(f256_simd_set(a.x2, a.x3), bv, p23, q23);
            double ignored{};
            f256_simd_store(p01, p0, p1);
            f256_simd_store(q01, q0, q1);
            f256_simd_store(p23, p2, p3);
            f256_simd_store(q23, q2, ignored);
        }
        else
        #endif
        {
            two_prod_precise(a.x0, b, p0, q0);
            two_prod_precise(a.x1, b, p1, q1);
            two_prod_precise(a.x2, b, p2, q2);
            p3 = a.x3 * b;
        }

        s0 = p0;
        two_sum_precise(q0, p1, s1, s2);
        three_sum(s2, q1, p2);
        three_sum2(q1, q2, p3);
        s3 = q1;
        s4 = q2 + p2;

        return renorm5(s0, s1, s2, s3, s4);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s mul_double_inline(double a, const f256_s& b) noexcept
    {
        return mul_double_inline(b, a);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_inline(const f256_s& a, double b) noexcept
    {
        using namespace detail::_f256;

        if (bl::use_constexpr_math())
        {
            if (isnan(a) || isnan(b))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (isinf(b))
            {
                if (isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
                return f256_s{ neg ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
            }

            if (b == 0.0)
            {
                if (iszero(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (isinf(a))
            {
                const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        const double inv_b = 1.0 / b;
        const f256_s divisor{ b, 0.0, 0.0, 0.0 };

        const double q0 = a.x0 * inv_b;
        f256_s r = sub_mul_scalar_exact(a, divisor, q0);

        const double q1 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q1);
        const double q2 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q2);
        const double q3 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q3);
        const double q4 = r.x0 * inv_b;

        return renorm5(q0, q1, q2, q3, q4);
    }
    [[nodiscard]] BL_FORCE_INLINE constexpr f256_s div_double_inline(double a, const f256_s& b) noexcept
    {
        using namespace detail::_f256;

        if (bl::use_constexpr_math())
        {
            if (isnan(a) || isnan(b))
                return std::numeric_limits<f256_s>::quiet_NaN();

            if (isinf(b))
            {
                if (isinf(a))
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit_constexpr(a) ^ signbit_constexpr(b.x0);
                return f256_s{ neg ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
            }

            if (iszero(b))
            {
                if (a == 0.0)
                    return std::numeric_limits<f256_s>::quiet_NaN();

                const bool neg = signbit_constexpr(a) ^ signbit_constexpr(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }

            if (isinf(a))
            {
                const bool neg = signbit_constexpr(a) ^ signbit_constexpr(b.x0);
                return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
            }
        }

        if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
            return div_double_inline(f256_s{ a, 0.0, 0.0, 0.0 }, b.x0);

        const double inv_b0 = 1.0 / b.x0;
        const double q0 = a * inv_b0;

        double p0{}, p1{}, p2{}, p3{};
        double e0{}, e1{}, e2{};
        double s0{}, s1{}, s2{}, s3{}, s4{};

        two_prod_precise(b.x0, q0, p0, e0);
        two_prod_precise(b.x1, q0, p1, e1);
        two_prod_precise(b.x2, q0, p2, e2);
        p3 = b.x3 * q0;

        s0 = p0;
        two_sum_precise(e0, p1, s1, s2);
        three_sum(s2, e1, p2);
        three_sum2(e1, e2, p3);
        s3 = e1;
        s4 = e2 + p2;

        double c0{}, t0{};
        two_sum_precise(a, -s0, c0, t0);

        double c1 = -s1;
        double c2 = -s2;
        double c3 = -s3;
        double t1 = 0.0;
        double t2 = 0.0;

        two_sum_precise(c1, t0, c1, t0);
        three_sum(c2, t0, t1);
        three_sum2(c3, t0, t2);
        t0 += t1 - s4;

        f256_s r = renorm5(c0, c1, c2, c3, t0);

        const double q1 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q1);
        const double q2 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q2);
        const double q3 = r.x0 * inv_b0; r = sub_mul_scalar_fast(r, b, q3);
        const double q4 = r.x0 * inv_b0;

        return renorm5(q0, q1, q2, q3, q4);
    }
}

/// ============= Scalar =============

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(double a, const f256_s& b) noexcept { return b + a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(double a, const f256_s& b) noexcept { return -(b - a); }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(double a, const f256_s& b) noexcept { return b * a; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(double a, const f256_s& b) noexcept { return f256_s{ a } / b; }

// f256 <=> float
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(const f256_s& a, float b) noexcept { return a + (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(const f256_s& a, float b) noexcept { return a - (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(const f256_s& a, float b) noexcept { return a * (double)b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(const f256_s& a, float b) noexcept { return a / (double)b; }

[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator+(float a, const f256_s& b) noexcept { return (double)a + b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator-(float a, const f256_s& b) noexcept { return (double)a - b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator*(float a, const f256_s& b) noexcept { return (double)a * b; }
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s operator/(float a, const f256_s& b) noexcept { return (double)a / b; }

/// ============= f256 constexpr core implementations =============

// scalar conversion
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::to_f256(uint64_t u) noexcept
{
    f256_s r{};
    double s{}, e{};
    uint64_to_exact_double_pair(u, s, e);
    r = renorm(s, e, 0.0, 0.0);
    return r;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::to_f256(int64_t v) noexcept
{
    if (v >= 0)
        return to_f256(static_cast<uint64_t>(v));

    const uint64_t mag = magnitude_u64(v);
    return -to_f256(mag);
}

BL_FORCE_INLINE constexpr f256_s& detail::_f256_constexpr::assign(f256_s& out, uint64_t u) noexcept
{
    out = to_f256(u);
    return out;
}
BL_FORCE_INLINE constexpr f256_s& detail::_f256_constexpr::assign(f256_s& out, int64_t v) noexcept
{
    out = to_f256(v);
    return out;
}

// rounding and powers
[[nodiscard]] inline constexpr f256_s detail::_f256_constexpr::floor(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    constexpr double integer_x0_threshold = 4503599627370496.0;

    if (absd(a.x0) >= integer_x0_threshold)
    {
        if (a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256_s{ a.x0, 0.0, 0.0, 0.0 };

        return f256_s{ a.x0, 0.0, 0.0, 0.0 } + detail::_f256_constexpr::floor(f256_s{ a.x1, a.x2, a.x3, 0.0 });
    }

    f256_s r{ floor_constexpr(a.x0), 0.0, 0.0, 0.0 };
    if (r > a)
        r -= 1.0;
    return r;
}
[[nodiscard]] inline constexpr f256_s detail::_f256_constexpr::ceil(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    constexpr double integer_x0_threshold = 4503599627370496.0;

    if (absd(a.x0) >= integer_x0_threshold)
    {
        if (a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0)
            return f256_s{ a.x0, 0.0, 0.0, 0.0 };

        return f256_s{ a.x0, 0.0, 0.0, 0.0 } + detail::_f256_constexpr::ceil(f256_s{ a.x1, a.x2, a.x3, 0.0 });
    }

    f256_s r{ ceil_constexpr(a.x0), 0.0, 0.0, 0.0 };
    if (r < a)
        r += 1.0;
    return r;
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::trunc(const f256_s& a)
{
    if (isnan(a) || isinf(a) || iszero(a))
        return a;

    return (a.x0 < 0.0) ? detail::_f256_constexpr::ceil(a) : detail::_f256_constexpr::floor(a);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::pow10_256(int k)
{
    if (k == 0) return f256_s{ 1.0 };

    int n = (k >= 0) ? k : -k;

    if (n <= 16) {
        f256_s r = f256_s{ 1.0 };
        const f256_s ten = f256_s{ 10.0, 0.0, 0.0, 0.0 };
        for (int i = 0; i < n; ++i) r = r * ten;
        return (k >= 0) ? r : (f256_s{ 1.0 } / r);
    }

    f256_s r = f256_s{ 1.0 };
    f256_s base = f256_s{ 10.0, 0.0, 0.0, 0.0 };

    while (n) {
        if (n & 1) r = r * base;
        n >>= 1;
        if (n) base = base * base;
    }

    return (k >= 0) ? r : (f256_s{ 1.0 } / r);
}

// f256 <=> f256 arithmetic
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::add(const f256_s& a, const f256_s& b) noexcept
{
    // optimize for double-double precision
    if (a.x2 == 0.0 && a.x3 == 0.0 &&
        b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
    {
        return add_dd_qd(a, b);
    }

    return add_qd_qd(a, b);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::sub(const f256_s& a, const f256_s& b) noexcept
{
    // optimize for double-double precision
    if (a.x2 == 0.0 && a.x3 == 0.0 &&
        b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
    {
        return sub_dd_qd(a, b);
    }

    return sub_qd_qd(a, b);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::mul(const f256_s& a, const f256_s& b) noexcept
{
    double p0{}, p1{}, p2{}, p3{}, p4{}, p5{};
    double q0{}, q1{}, q2{}, q3{}, q4{}, q5{};
    double p6{}, p7{}, p8{}, p9{};
    double q6{}, q7{}, q8{}, q9{};
    double r0{}, r1{};
    double t0{}, t1{};
    double s0{}, s1{}, s2{};

    two_prod_precise(a.x0, b.x0, p0, q0);
    two_prod_precise(a.x0, b.x1, p1, q1);
    two_prod_precise(a.x1, b.x0, p2, q2);
    two_prod_precise(a.x0, b.x2, p3, q3);
    two_prod_precise(a.x1, b.x1, p4, q4);
    two_prod_precise(a.x2, b.x0, p5, q5);

    three_sum(p1, p2, q0);
    three_sum(p2, q1, q2);
    three_sum(p3, p4, p5);

    two_sum_precise(p2, p3, s0, t0);
    two_sum_precise(q1, p4, s1, t1);  s2 = q2 + p5;
    two_sum_precise(s1, t0, s1, t0);  s2 += (t0 + t1);

    two_prod_precise(a.x0, b.x3, p6, q6);
    two_prod_precise(a.x1, b.x2, p7, q7);
    two_prod_precise(a.x2, b.x1, p8, q8);
    two_prod_precise(a.x3, b.x0, p9, q9);

    two_sum_precise(q0, q3, q0, q3);
    two_sum_precise(q4, q5, q4, q5);
    two_sum_precise(p6, p7, p6, p7);
    two_sum_precise(p8, p9, p8, p9);

    two_sum_precise(q0, q4, t0, t1);  t1 += (q3 + q5);
    two_sum_precise(p6, p8, r0, r1);  r1 += (p7 + p9);
    two_sum_precise(t0, r0, q3, q4);  q4 += (t1 + r1);

    two_sum_precise(q3, s1, t0, t1);
    t1 += q4;
    t1 += a.x1 * b.x3 + a.x2 * b.x2 + a.x3 * b.x1 + q6 + q7 + q8 + q9 + s2;

    return renorm5(p0, p1, s0, t0, t1);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::div(const f256_s& a, const f256_s& b) noexcept
{
    if (b.x1 == 0.0 && b.x2 == 0.0 && b.x3 == 0.0) [[unlikely]]
        return a / b.x0;

    const double inv_b0 = 1.0 / b.x0;

    const double q0 = a.x0 * inv_b0;
    f256_s r = sub_mul_scalar_exact(a, b, q0);

    const double q1 = r.x0 * inv_b0;
    r = sub_mul_scalar_fast(r, b, q1);

    const double q2 = r.x0 * inv_b0;
    r = sub_mul_scalar_fast(r, b, q2);

    const double q3 = r.x0 * inv_b0;
    r = sub_mul_scalar_fast(r, b, q3);

    const double q4 = r.x0 * inv_b0;

    return renorm5(q0, q1, q2, q3, q4);
}

// f256 <=> f64 arithmetic
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::add_double(const f256_s& a, double b) noexcept
{
    double c0{}, c1{}, c2{}, c3{};
    double e{};

    two_sum_precise(a.x0, b, c0, e);
    if (e == 0.0) return { c0, a.x1, a.x2, a.x3 };

    two_sum_precise(a.x1, e, c1, e);
    if (e == 0.0) return renorm4(c0, c1, a.x2, a.x3);

    two_sum_precise(a.x2, e, c2, e);
    if (e == 0.0) return renorm4(c0, c1, c2, a.x3);

    two_sum_precise(a.x3, e, c3, e);
    if (e == 0.0) return renorm4(c0, c1, c2, c3);

    return renorm5(c0, c1, c2, c3, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::sub_double(const f256_s& a, double b) noexcept
{
    double c0{}, c1{}, c2{}, c3{}, e{};

    two_sum_precise(a.x0, -b, c0, e);
    if (e == 0.0) return { c0, a.x1, a.x2, a.x3 };

    two_sum_precise(a.x1, e, c1, e);
    if (e == 0.0) return renorm4(c0, c1, a.x2, a.x3);

    two_sum_precise(a.x2, e, c2, e);
    if (e == 0.0) return renorm4(c0, c1, c2, a.x3);

    two_sum_precise(a.x3, e, c3, e);
    if (e == 0.0) return renorm4(c0, c1, c2, c3);

    return renorm5(c0, c1, c2, c3, e);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::mul_double(const f256_s& a, double b) noexcept
{
    double p0{}, p1{}, p2{}, p3{};
    double q0{}, q1{}, q2{};
    double s0{}, s1{}, s2{}, s3{}, s4{};

    two_prod_precise(a.x0, b, p0, q0);
    two_prod_precise(a.x1, b, p1, q1);
    two_prod_precise(a.x2, b, p2, q2);
    p3 = a.x3 * b;

    s0 = p0;
    two_sum_precise(q0, p1, s1, s2);
    three_sum(s2, q1, p2);
    three_sum2(q1, q2, p3);
    s3 = q1;
    s4 = q2 + p2;

    return renorm5(s0, s1, s2, s3, s4);
}
[[nodiscard]] BL_FORCE_INLINE constexpr f256_s detail::_f256_constexpr::div_double(const f256_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
    {
        if (isnan(a) || isnan(b))
            return std::numeric_limits<f256_s>::quiet_NaN();

        if (isinf(b))
        {
            if (isinf(a))
                return std::numeric_limits<f256_s>::quiet_NaN();

            const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
            return f256_s{ neg ? -0.0 : 0.0, 0.0, 0.0, 0.0 };
        }

        if (b == 0.0)
        {
            if (iszero(a))
                return std::numeric_limits<f256_s>::quiet_NaN();

            const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
            return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                               :  std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
        }

        if (isinf(a))
        {
            const bool neg = signbit_constexpr(a.x0) ^ signbit_constexpr(b);
            return f256_s{ neg ? -std::numeric_limits<double>::infinity()
                               :  std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0 };
        }
    }

    const double inv_b = 1.0 / b;
    const f256_s divisor{ b, 0.0, 0.0, 0.0 };

    const double q0 = a.x0 * inv_b;
    f256_s r = sub_mul_scalar_exact(a, divisor, q0);

    const double q1 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q1);
    const double q2 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q2);
    const double q3 = r.x0 * inv_b; r = sub_mul_scalar_fast(r, divisor, q3);
    const double q4 = r.x0 * inv_b;

    return renorm5(q0, q1, q2, q3, q4);
}

/// ============= f256 public core wrappers =============

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s floor(const f256_s& a)
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::floor(a);

    return detail::_f256_runtime::floor(a);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s ceil(const f256_s& a)
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::ceil(a);

    return detail::_f256_runtime::ceil(a);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s trunc(const f256_s& a)
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::trunc(a);

    return detail::_f256_runtime::trunc(a);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s pow10_256(int k)
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::pow10_256(k);

    return detail::_f256_runtime::pow10_256(k);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator+(const f256_s& a, const f256_s& b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::add(a, b);

    return detail::_f256::add_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator-(const f256_s& a, const f256_s& b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::sub(a, b);

    return detail::_f256::sub_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator*(const f256_s& a, const f256_s& b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::mul(a, b);

    return detail::_f256::mul_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator/(const f256_s& a, const f256_s& b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::div(a, b);

    return detail::_f256::div_inline(a, b);
}

[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator+(const f256_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::add_double(a, b);

    return detail::_f256::add_double_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator-(const f256_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::sub_double(a, b);

    return detail::_f256::sub_double_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator*(const f256_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::mul_double(a, b);

    return detail::_f256::mul_double_inline(a, b);
}
[[nodiscard]] BL_MSVC_NOINLINE constexpr f256_s operator/(const f256_s& a, double b) noexcept
{
    if (bl::use_constexpr_math())
        return detail::_f256_constexpr::div_double(a, b);

    return detail::_f256::div_double_inline(a, b);
}

} // namespace bl

#endif
