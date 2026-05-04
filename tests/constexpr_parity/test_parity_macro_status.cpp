#include <cstdio>

namespace
{
    struct parity_macro_status_printer
    {
        parity_macro_status_printer() noexcept
        {
            std::fputs("[fltx parity_tests] FLTX_CONSTEXPR_PARITY=", stderr);
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

    const parity_macro_status_printer parity_macro_status_printer_instance{};
}
