/**
 * fltx/config.h - Core macros, precision controls, and constant-evaluation helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_CONFIG_INCLUDED
#define FLTX_CONFIG_INCLUDED
#include <cstdint>
#include <limits>

static_assert(sizeof(double) == sizeof(std::uint64_t),
    "fltx requires double to be 64 bits.");

static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::radix == 2 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::min_exponent == -1021 &&
              std::numeric_limits<double>::max_exponent == 1024,
    "fltx requires double to use the IEEE 754 binary64 format.");

#ifndef BL_FORCE_INLINE
#if defined(_MSC_VER)
#define BL_FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define BL_FORCE_INLINE inline __attribute__((always_inline))
#else
#define BL_FORCE_INLINE inline
#endif
#endif

#ifndef BL_NO_INLINE
#if defined(_MSC_VER)
#define BL_NO_INLINE __declspec(noinline)
#elif defined(__clang__) || defined(__GNUC__)
#define BL_NO_INLINE __attribute__((noinline))
#else
#define BL_NO_INLINE
#endif
#endif

// Use when pairing BL_NO_INLINE with constexpr
#if defined(_MSC_VER)
#define BL_MSVC_NOINLINE BL_NO_INLINE
#else
#define BL_MSVC_NOINLINE
#endif

#ifndef FLTX_INLINE_LEVEL
#define FLTX_INLINE_LEVEL 2
#endif

// Base inline aggressiveness (adjustable)
#if FLTX_INLINE_LEVEL >= 2
#define FLTX_CORE_INLINE BL_FORCE_INLINE
#elif FLTX_INLINE_LEVEL == 1
#define FLTX_CORE_INLINE
#else
#define FLTX_CORE_INLINE BL_MSVC_NOINLINE
#endif

#ifndef BL_FAST_MATH
#if defined(__FAST_MATH__)
#define BL_FAST_MATH
#elif defined(_MSC_VER) && defined(_M_FP_FAST)
#define BL_FAST_MATH
#endif
#endif

#ifndef BL_PRINT_NOINLINE
#if defined(_MSC_VER) && defined(BL_FAST_MATH)
#define BL_PRINT_NOINLINE __declspec(noinline)
#else
#define BL_PRINT_NOINLINE
#endif
#endif

#if !defined(FLTX_DISABLE_FMA_AVAILABLE)
#ifndef FMA_AVAILABLE
#ifndef __EMSCRIPTEN__
#if defined(__FMA__) || defined(__FMA4__) || defined(_MSC_VER) || defined(__clang__)
#define FMA_AVAILABLE
#endif
#endif
#endif
#endif

#if defined(_MSC_VER)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE __pragma(float_control(precise, on, push)) \
                         __pragma(fp_contract(off))
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE __pragma(float_control(pop))
#endif
#elif defined(__EMSCRIPTEN__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE _Pragma("clang fp reassociate(on)")  \
                         _Pragma("clang fp contract(fast)")
#endif
#elif defined(__clang__)
// Clang's fp pragmas here do not restore a previous stack state. Keep strict
// FP semantics after protected blocks; expansion arithmetic relies on it.
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#elif defined(__GNUC__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE _Pragma("GCC push_options")               \
                         _Pragma("GCC optimize(\"no-fast-math\")") \
                         _Pragma("STDC FP_CONTRACT OFF")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE _Pragma("GCC pop_options")
#endif
#else
#define BL_PUSH_PRECISE
#define BL_POP_PRECISE
#endif

namespace bl
{
    #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
    namespace _fltx_debug {

        inline bool simulate_consteval_path = false;

        BL_FORCE_INLINE void set_simulated_consteval_path(bool enabled) noexcept { simulate_consteval_path = enabled; }
        BL_FORCE_INLINE void set_forced_constexpr_path() noexcept { set_simulated_consteval_path(true); }
        BL_FORCE_INLINE void set_forced_runtime_path() noexcept { set_simulated_consteval_path(false); }

    } // namespace _fltx_debug

    #endif

    [[nodiscard]] BL_FORCE_INLINE constexpr bool is_constant_evaluated() noexcept
    {
        // In simulated-consteval mode, tests can run ordinary runtime calls
        // through the branches that would be selected during constant
        // evaluation. FLTX_CONSTEXPR_PARITY itself is intentionally not part of
        // this decision; it requests bitwise-compatible results, not forced
        // constexpr-path execution.
        if consteval
        {
            return true;
        }

        #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
        return _fltx_debug::simulate_consteval_path;
        #else
        return false;
        #endif
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool use_constexpr_parity() noexcept
    {
        // Result-parity policy only. Callers may use this to decide whether an
        // otherwise optional canonicalization step is worth paying for.
        #if defined(FLTX_CONSTEXPR_PARITY)
        return true;
        #else
        return false;
        #endif
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr bool use_constexpr_math() noexcept
    {
        // Select constexpr-safe algorithms. In normal builds this tracks actual
        // constant evaluation. In simulated-consteval mode, tests can force
        // this at runtime to compare predicted consteval results with runtime
        // results. FLTX_CONSTEXPR_PARITY does not change is_constant_evaluated(),
        // but it may choose constexpr-safe runtime algorithms when that is the
        // cleanest way to guarantee bitwise-identical results.
        #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
        return is_constant_evaluated() || _fltx_debug::simulate_consteval_path || use_constexpr_parity();
        #else
        return is_constant_evaluated() || use_constexpr_parity();
        #endif
    }

} // namespace bl

#ifndef BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS
#define BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS(wrapper_type, storage_type)                                   \
template<>                                                                                                   \
struct std::numeric_limits<wrapper_type>                                                                     \
{                                                                                                            \
    using base = std::numeric_limits<storage_type>;                                                          \
                                                                                                             \
    static constexpr bool is_specialized = base::is_specialized;                                             \
                                                                                                             \
    static constexpr wrapper_type min()           noexcept { return wrapper_type{ base::min() }; }           \
    static constexpr wrapper_type max()           noexcept { return wrapper_type{ base::max() }; }           \
    static constexpr wrapper_type lowest()        noexcept { return wrapper_type{ base::lowest() }; }        \
    static constexpr wrapper_type epsilon()       noexcept { return wrapper_type{ base::epsilon() }; }       \
    static constexpr wrapper_type round_error()   noexcept { return wrapper_type{ base::round_error() }; }   \
    static constexpr wrapper_type infinity()      noexcept { return wrapper_type{ base::infinity() }; }      \
    static constexpr wrapper_type quiet_NaN()     noexcept { return wrapper_type{ base::quiet_NaN() }; }     \
    static constexpr wrapper_type signaling_NaN() noexcept { return wrapper_type{ base::signaling_NaN() }; } \
    static constexpr wrapper_type denorm_min()    noexcept { return wrapper_type{ base::denorm_min() }; }    \
                                                                                                             \
    static constexpr bool has_infinity      = base::has_infinity;                                            \
    static constexpr bool has_quiet_NaN     = base::has_quiet_NaN;                                           \
    static constexpr bool has_signaling_NaN = base::has_signaling_NaN;                                       \
                                                                                                             \
    static constexpr int digits       = base::digits;                                                        \
    static constexpr int digits10     = base::digits10;                                                      \
    static constexpr int max_digits10 = base::max_digits10;                                                  \
                                                                                                             \
    static constexpr bool is_signed  = base::is_signed;                                                      \
    static constexpr bool is_integer = base::is_integer;                                                     \
    static constexpr bool is_exact   = base::is_exact;                                                       \
    static constexpr int  radix      = base::radix;                                                          \
                                                                                                             \
    static constexpr int min_exponent   = base::min_exponent;                                                \
    static constexpr int max_exponent   = base::max_exponent;                                                \
    static constexpr int min_exponent10 = base::min_exponent10;                                              \
    static constexpr int max_exponent10 = base::max_exponent10;                                              \
                                                                                                             \
    static constexpr bool is_iec559       = base::is_iec559;                                                 \
    static constexpr bool is_bounded      = base::is_bounded;                                                \
    static constexpr bool is_modulo       = base::is_modulo;                                                 \
    static constexpr bool traps           = base::traps;                                                     \
    static constexpr bool tinyness_before = base::tinyness_before;                                           \
                                                                                                             \
    static constexpr std::float_round_style round_style = base::round_style;                                 \
};
#endif


#ifndef BL_CXX_LANGUAGE_VERSION
#if defined(_MSVC_LANG) && (!defined(__cplusplus) || (_MSVC_LANG > __cplusplus))
#define BL_CXX_LANGUAGE_VERSION _MSVC_LANG
#else
#define BL_CXX_LANGUAGE_VERSION __cplusplus
#endif
#endif

#if BL_CXX_LANGUAGE_VERSION <= 202002L
#error fltx requires C++23 or newer.
#endif

#if !defined(__cpp_if_consteval) || (__cpp_if_consteval < 202106L)
#error fltx requires C++23 if consteval support.
#endif

#ifndef BL_CONSTEXPR_RUNTIME_DISPATCH
  #if !defined(FLTX_CONSTEXPR_PARITY) && !defined(FLTX_SIMULATE_CONSTEVAL_MODE)
    #define BL_CONSTEXPR_RUNTIME_DISPATCH_USES_CONSTEVAL 1
    #define BL_CONSTEXPR_RUNTIME_DISPATCH(CONSTEVAL_EXPR, RUNTIME_EXPR) \
        do                                                              \
        {                                                               \
            if consteval                                                \
            {                                                           \
                return (CONSTEVAL_EXPR);                                \
            }                                                           \
            else                                                        \
            {                                                           \
                return (RUNTIME_EXPR);                                  \
            }                                                           \
        } while (false)
  #else
    #define BL_CONSTEXPR_RUNTIME_DISPATCH_USES_CONSTEVAL 0
    #define BL_CONSTEXPR_RUNTIME_DISPATCH(CONSTEVAL_EXPR, RUNTIME_EXPR) \
        do                                                              \
        {                                                               \
            if consteval                                                \
            {                                                           \
                return (CONSTEVAL_EXPR);                                \
            }                                                           \
            else                                                        \
            {                                                           \
                if (bl::use_constexpr_math())                           \
                    return (CONSTEVAL_EXPR);                            \
                return (RUNTIME_EXPR);                                  \
            }                                                           \
        } while (false)
  #endif
#endif

#ifndef BL_CONSTEXPR_RUNTIME_DISPATCH_USES_CONSTEVAL
#define BL_CONSTEXPR_RUNTIME_DISPATCH_USES_CONSTEVAL 0
#endif

#endif
