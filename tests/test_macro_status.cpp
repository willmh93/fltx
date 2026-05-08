#include "f256.h"

#include <cstdio>

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

namespace
{
    struct fltx_test_macro_status_printer
    {
        fltx_test_macro_status_printer() noexcept
        {
            std::fputs("[fltx ", stderr);
            std::fputs(FLTX_TEST_NAME, stderr);
            std::fputs("] BL_FAST_MATH=", stderr);
            #if defined(BL_FAST_MATH)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs(", FLTX_CONSTEXPR_PARITY=", stderr);
            #if defined(FLTX_CONSTEXPR_PARITY)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs(", FLTX_CONSTEXPR_PARITY_TEST_MODE=", stderr);
            #if defined(FLTX_CONSTEXPR_PARITY_TEST_MODE)
            std::fputs("defined", stderr);
            #else
            std::fputs("not defined", stderr);
            #endif

            std::fputs(", BL_F256_HAS_SSE2=", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_HAS_SSE2), stderr);

            std::fputs(", BL_F256_HAS_NEON=", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_HAS_NEON), stderr);

            std::fputs(", BL_F256_ENABLE_SIMD=", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_ENABLE_SIMD), stderr);

            std::fputs(", BL_F256_ENABLE_TRIG_SIMD=", stderr);
            std::fputs(FLTX_STRINGIZE(BL_F256_ENABLE_TRIG_SIMD), stderr);

            std::fputs(", arm64=", stderr);
            std::fputs(FLTX_ARCH_ARM64_STATUS, stderr);

            std::fputs(", neon=", stderr);
            std::fputs(FLTX_ARCH_NEON_STATUS, stderr);

            std::fputs(", sse2=", stderr);
            std::fputs(FLTX_ARCH_SSE2_STATUS, stderr);

            std::fputc('\n', stderr);
            std::fflush(stderr);
        }
    };

    const fltx_test_macro_status_printer fltx_test_macro_status_printer_instance{};
}
