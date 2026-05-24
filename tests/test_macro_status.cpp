#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fltx/f256.h"

#ifndef FLTX_TEST_NAME
#define FLTX_TEST_NAME "unknown_tests"
#endif

#define FLTX_STRINGIZE_IMPL(x) #x
#define FLTX_STRINGIZE(x) FLTX_STRINGIZE_IMPL(x)

#if defined(__aarch64__) || defined(_M_ARM64)
#define FLTX_ARCH_ARM64_STATUS "defined"
#else
#define FLTX_ARCH_ARM64_STATUS "not defined"
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define FLTX_ARCH_NEON_STATUS "defined"
#else
#define FLTX_ARCH_NEON_STATUS "not defined"
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && (_M_IX86_FP >= 2))
#define FLTX_ARCH_SSE2_STATUS "defined"
#else
#define FLTX_ARCH_SSE2_STATUS "not defined"
#endif

#if defined(__wasm_simd128__)
#define FLTX_ARCH_WASM_SIMD128_STATUS "defined"
#else
#define FLTX_ARCH_WASM_SIMD128_STATUS "not defined"
#endif

namespace
{
    [[nodiscard]] bool env_requests_simulated_consteval(bool fallback) noexcept
    {
        const char* value = std::getenv("FLTX_SIMULATE_CONSTEVAL");
        if (value == nullptr || *value == '\0')
            return fallback;

        if (std::strcmp(value, "0") == 0 ||
            std::strcmp(value, "false") == 0 ||
            std::strcmp(value, "FALSE") == 0 ||
            std::strcmp(value, "off") == 0 ||
            std::strcmp(value, "OFF") == 0)
        {
            return false;
        }

        return true;
    }

    struct fltx_test_macro_status_printer
    {
        fltx_test_macro_status_printer() noexcept
        {
            #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
            bool simulated_consteval_enabled = false;
            #if defined(FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
            simulated_consteval_enabled = true;
            #endif
            simulated_consteval_enabled = env_requests_simulated_consteval(simulated_consteval_enabled);
            bl::_fltx_debug::set_simulated_consteval_path(simulated_consteval_enabled);
            #endif

            std::fputs("[fltx ", stderr);
            std::fputs(FLTX_TEST_NAME, stderr);
            std::fputs("] BL_FAST_MATH = ", stderr);
            #if defined(BL_FAST_MATH)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs("\nFMA_AVAILABLE = ", stderr);
            #if defined(FMA_AVAILABLE)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs("\nFLTX_CONSTEXPR_PARITY = ", stderr);
            #if defined(FLTX_CONSTEXPR_PARITY)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs("\nFLTX_SIMULATE_CONSTEVAL_MODE = ", stderr);
            #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs("\nFLTX_TEST_FORCE_SIMULATED_CONSTEVAL = ", stderr);
            #if defined(FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs("\nsimulated_consteval = ", stderr);
            #if defined(FLTX_SIMULATE_CONSTEVAL_MODE)
            std::fputs(simulated_consteval_enabled ? "on" : "off", stderr);
            #else
            std::fputs("unavailable", stderr);
            #endif

            std::fputs("\nBL_FLTX_HAS_SSE2 = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_FLTX_HAS_SSE2), stderr);

            std::fputs("\nBL_FLTX_HAS_NEON = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_FLTX_HAS_NEON), stderr);

            std::fputs("\nBL_FLTX_HAS_WASM_SIMD = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_FLTX_HAS_WASM_SIMD), stderr);

            std::fputs("\nBL_F256_ENABLE_SIMD = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_ENABLE_SIMD), stderr);

            std::fputs("\nBL_F256_ENABLE_TRIG_SIMD = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_ENABLE_TRIG_SIMD), stderr);

            std::fputs("\nBL_FLTX_SIMD_USE_FMA_TWO_PROD = ", stderr);
            std::fputs(FLTX_STRINGIZE(BL_FLTX_SIMD_USE_FMA_TWO_PROD), stderr);

            std::fputs("\narm64 = ", stderr);
            std::fputs(FLTX_ARCH_ARM64_STATUS, stderr);

            std::fputs("\nneon = ", stderr);
            std::fputs(FLTX_ARCH_NEON_STATUS, stderr);

            std::fputs("\nsse2 = ", stderr);
            std::fputs(FLTX_ARCH_SSE2_STATUS, stderr);

            std::fputs("\nwasm_simd128 = ", stderr);
            std::fputs(FLTX_ARCH_WASM_SIMD128_STATUS, stderr);

            std::fputc('\n', stderr);
            std::fflush(stderr);
        }
    };

    const fltx_test_macro_status_printer fltx_test_macro_status_printer_instance{};

} // namespace
