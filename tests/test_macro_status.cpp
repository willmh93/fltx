#include "fltx_common_base.h"

#include <cstdio>

#ifndef FLTX_TEST_NAME
#define FLTX_TEST_NAME "unknown_tests"
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

            std::fputc('\n', stderr);
            std::fflush(stderr);
        }
    };

    const fltx_test_macro_status_printer fltx_test_macro_status_printer_instance{};
}
