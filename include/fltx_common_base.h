#ifndef FLTX_COMMON_BASE_INCLUDED
#define FLTX_COMMON_BASE_INCLUDED

#include <limits>
#include <type_traits>

#ifndef FORCE_INLINE
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif
#endif

#ifndef NO_INLINE
#if defined(_MSC_VER)
#define NO_INLINE __declspec(noinline)
#elif defined(__clang__) || defined(__GNUC__)
#define NO_INLINE __attribute__((noinline))
#else
#define NO_INLINE
#endif
#endif

#ifndef BL_LIKELY
#if defined(__clang__) || defined(__GNUC__)
#define BL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define BL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BL_LIKELY(x)   (x)
#define BL_UNLIKELY(x) (x)
#endif
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

#ifndef FMA_AVAILABLE
#ifndef __EMSCRIPTEN__
#if defined(__FMA__) || defined(__FMA4__) || defined(_MSC_VER) || defined(__clang__)
#define FMA_AVAILABLE
#endif
#endif
#endif

#if defined(_MSC_VER)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  __pragma(float_control(precise, on, push)) \
                         __pragma(fp_contract(off))
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   __pragma(float_control(pop))
#endif
#elif defined(__EMSCRIPTEN__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("clang fp reassociate(on)")  \
                         _Pragma("clang fp contract(fast)")
#endif
#elif defined(__clang__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("clang fp reassociate(off)") \
                         _Pragma("clang fp contract(off)")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("clang fp reassociate(on)")  \
                         _Pragma("clang fp contract(fast)")
#endif
#elif defined(__GNUC__)
#ifndef BL_PUSH_PRECISE
#define BL_PUSH_PRECISE  _Pragma("GCC push_options")               \
                         _Pragma("GCC optimize(\"no-fast-math\")") \
                         _Pragma("STDC FP_CONTRACT OFF")
#endif
#ifndef BL_POP_PRECISE
#define BL_POP_PRECISE   _Pragma("GCC pop_options")
#endif
#else
#define BL_PUSH_PRECISE
#define BL_POP_PRECISE
#endif

#ifdef FLTX_CONSTEXPR_PARITY_TEST_MODE
#define BL_CONSTEXPR
#else
#define BL_CONSTEXPR constexpr
#endif

namespace bl
{
    namespace _fltx_debug {
        inline bool always_constexpr_path = false;
        FORCE_INLINE void set_forced_constexpr_path() noexcept { always_constexpr_path = true; }
        FORCE_INLINE void set_forced_runtime_path() noexcept { always_constexpr_path = false; }
    }

    [[nodiscard]] FORCE_INLINE constexpr bool is_constant_evaluated() noexcept
    {
#ifdef FLTX_CONSTEXPR_PARITY_TEST_MODE
        return std::is_constant_evaluated() ? true : _fltx_debug::always_constexpr_path;
#else
        return std::is_constant_evaluated();
#endif
    }
}

#ifndef BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS
#define BL_DEFINE_FLOAT_WRAPPER_NUMERIC_LIMITS(wrapper_type, storage_type) \
template<>                                                                \
struct std::numeric_limits<wrapper_type>                                  \
{                                                                         \
    using base = std::numeric_limits<storage_type>;                       \
                                                                          \
    static constexpr bool is_specialized = base::is_specialized;          \
                                                                          \
    static constexpr wrapper_type min() noexcept           { return wrapper_type{ base::min() }; } \
    static constexpr wrapper_type max() noexcept           { return wrapper_type{ base::max() }; } \
    static constexpr wrapper_type lowest() noexcept        { return wrapper_type{ base::lowest() }; } \
    static constexpr wrapper_type epsilon() noexcept       { return wrapper_type{ base::epsilon() }; } \
    static constexpr wrapper_type round_error() noexcept   { return wrapper_type{ base::round_error() }; } \
    static constexpr wrapper_type infinity() noexcept      { return wrapper_type{ base::infinity() }; } \
    static constexpr wrapper_type quiet_NaN() noexcept     { return wrapper_type{ base::quiet_NaN() }; } \
    static constexpr wrapper_type signaling_NaN() noexcept { return wrapper_type{ base::signaling_NaN() }; } \
    static constexpr wrapper_type denorm_min() noexcept    { return wrapper_type{ base::denorm_min() }; } \
                                                                          \
    static constexpr bool has_infinity      = base::has_infinity;         \
    static constexpr bool has_quiet_NaN     = base::has_quiet_NaN;        \
    static constexpr bool has_signaling_NaN = base::has_signaling_NaN;    \
                                                                          \
    static constexpr int digits         = base::digits;                   \
    static constexpr int digits10       = base::digits10;                 \
    static constexpr int max_digits10   = base::max_digits10;             \
                                                                          \
    static constexpr bool is_signed     = base::is_signed;                \
    static constexpr bool is_integer    = base::is_integer;               \
    static constexpr bool is_exact      = base::is_exact;                 \
    static constexpr int radix          = base::radix;                    \
                                                                          \
    static constexpr int min_exponent   = base::min_exponent;             \
    static constexpr int max_exponent   = base::max_exponent;             \
    static constexpr int min_exponent10 = base::min_exponent10;           \
    static constexpr int max_exponent10 = base::max_exponent10;           \
                                                                          \
    static constexpr bool is_iec559       = base::is_iec559;              \
    static constexpr bool is_bounded      = base::is_bounded;             \
    static constexpr bool is_modulo       = base::is_modulo;              \
    static constexpr bool traps           = base::traps;                  \
    static constexpr bool tinyness_before = base::tinyness_before;        \
                                                                          \
    static constexpr std::float_round_style round_style = base::round_style; \
};
#endif

#endif