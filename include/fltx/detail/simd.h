/**
 * fltx/detail/simd.h - Shared SIMD platform detection and intrinsic includes.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_SIMD_INCLUDED
#define FLTX_SIMD_INCLUDED
#include "fltx/config.h"

#if !defined(BL_FLTX_HAS_SSE2)
#  if !defined(__EMSCRIPTEN__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && (_M_IX86_FP >= 2)))
#    define BL_FLTX_HAS_SSE2 1
#  else
#    define BL_FLTX_HAS_SSE2 0
#  endif
#endif

#if !defined(BL_FLTX_HAS_NEON)
#  if !defined(__EMSCRIPTEN__) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#    define BL_FLTX_HAS_NEON 1
#  else
#    define BL_FLTX_HAS_NEON 0
#  endif
#endif

#if !defined(BL_FLTX_HAS_WASM_SIMD)
#  if defined(__wasm_simd128__)
#    define BL_FLTX_HAS_WASM_SIMD 1
#  else
#    define BL_FLTX_HAS_WASM_SIMD 0
#  endif
#endif

#if !defined(BL_FLTX_HAS_X86_FMA)
#  if BL_FLTX_HAS_SSE2 && (defined(__FMA__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))))
#    define BL_FLTX_HAS_X86_FMA 1
#  else
#    define BL_FLTX_HAS_X86_FMA 0
#  endif
#endif

#if BL_FLTX_HAS_SSE2
#  include <emmintrin.h>
#endif

#if BL_FLTX_HAS_X86_FMA
#  include <immintrin.h>
#endif

#if BL_FLTX_HAS_NEON
#  include <arm_neon.h>
#endif

#if BL_FLTX_HAS_WASM_SIMD
#  include <wasm_simd128.h>
#endif

#if !defined(BL_FLTX_SIMD_USE_FMA_TWO_PROD)
#  if defined(FLTX_DISABLE_FMA_AVAILABLE)
#    define BL_FLTX_SIMD_USE_FMA_TWO_PROD 0
#  elif BL_FLTX_HAS_X86_FMA || BL_FLTX_HAS_NEON
#    define BL_FLTX_SIMD_USE_FMA_TWO_PROD 1
#  else
#    define BL_FLTX_SIMD_USE_FMA_TWO_PROD 0
#  endif
#endif

#if BL_FLTX_HAS_SSE2 || BL_FLTX_HAS_NEON || BL_FLTX_HAS_WASM_SIMD
namespace bl::detail::simd
{
    #if BL_FLTX_HAS_SSE2
    using f64x2 = __m128d;
    #elif BL_FLTX_HAS_NEON
    using f64x2 = float64x2_t;
    #else
    using f64x2 = v128_t;
    #endif

    BL_FORCE_INLINE f64x2 f64x2_set(double lane0, double lane1) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return _mm_set_pd(lane1, lane0);
        #elif BL_FLTX_HAS_NEON
        return vsetq_lane_f64(lane1, vdupq_n_f64(lane0), 1);
        #else
        return wasm_f64x2_make(lane0, lane1);
        #endif
    }

    BL_FORCE_INLINE f64x2 f64x2_splat(double value) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return _mm_set1_pd(value);
        #elif BL_FLTX_HAS_NEON
        return vdupq_n_f64(value);
        #else
        return wasm_f64x2_splat(value);
        #endif
    }

    BL_FORCE_INLINE void f64x2_store_array(f64x2 value, double* lanes) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        _mm_storeu_pd(lanes, value);
        #elif BL_FLTX_HAS_NEON
        vst1q_f64(lanes, value);
        #else
        wasm_v128_store(lanes, value);
        #endif
    }

    BL_FORCE_INLINE void f64x2_store(f64x2 value, double& lane0, double& lane1) noexcept
    {
        alignas(16) double lanes[2];
        f64x2_store_array(value, lanes);
        lane0 = lanes[0];
        lane1 = lanes[1];
    }

    BL_FORCE_INLINE f64x2 f64x2_add(f64x2 a, f64x2 b) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return _mm_add_pd(a, b);
        #elif BL_FLTX_HAS_NEON
        return vaddq_f64(a, b);
        #else
        return wasm_f64x2_add(a, b);
        #endif
    }

    BL_FORCE_INLINE f64x2 f64x2_sub(f64x2 a, f64x2 b) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return _mm_sub_pd(a, b);
        #elif BL_FLTX_HAS_NEON
        return vsubq_f64(a, b);
        #else
        return wasm_f64x2_sub(a, b);
        #endif
    }

    BL_FORCE_INLINE f64x2 f64x2_mul(f64x2 a, f64x2 b) noexcept
    {
        #if BL_FLTX_HAS_SSE2
        return _mm_mul_pd(a, b);
        #elif BL_FLTX_HAS_NEON
        return vmulq_f64(a, b);
        #else
        return wasm_f64x2_mul(a, b);
        #endif
    }

    BL_FORCE_INLINE void f64x2_two_prod_precise(f64x2 a, f64x2 b, f64x2& p, f64x2& e) noexcept
    {
        p = f64x2_mul(a, b);

        #if BL_FLTX_SIMD_USE_FMA_TWO_PROD && BL_FLTX_HAS_X86_FMA
        e = _mm_fmsub_pd(a, b, p);
        #elif BL_FLTX_SIMD_USE_FMA_TWO_PROD && BL_FLTX_HAS_NEON
        e = vfmaq_f64(f64x2_sub(f64x2_splat(0.0), p), a, b);
        #else
        const f64x2 split    = f64x2_splat(134217729.0);
        const f64x2 a_scaled = f64x2_mul(a, split);
        const f64x2 b_scaled = f64x2_mul(b, split);

        const f64x2 a_hi = f64x2_sub(a_scaled, f64x2_sub(a_scaled, a));
        const f64x2 b_hi = f64x2_sub(b_scaled, f64x2_sub(b_scaled, b));
        const f64x2 a_lo = f64x2_sub(a, a_hi);
        const f64x2 b_lo = f64x2_sub(b, b_hi);

        e = f64x2_add(f64x2_sub(f64x2_mul(a_hi, b_hi), p), f64x2_mul(a_hi, b_lo));
        e = f64x2_add(e, f64x2_mul(a_lo, b_hi));
        e = f64x2_add(e, f64x2_mul(a_lo, b_lo));
        #endif
    }

    BL_FORCE_INLINE void f64x2_two_sum(f64x2 a, f64x2 b, f64x2& s, f64x2& e) noexcept
    {
        s = f64x2_add(a, b);
        const f64x2 bb = f64x2_sub(s, a);
        e = f64x2_add(f64x2_sub(a, f64x2_sub(s, bb)), f64x2_sub(b, bb));
    }

} // namespace bl::detail::simd

#endif

#endif
