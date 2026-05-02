// isolated_runtime.cpp
#include "isolated_runtime.h"

#include <f128.h>
#include <f256.h>

#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#endif

#if defined(_MSC_VER)
#define BL_ISOLATED_NOINLINE __declspec(noinline)
#elif defined(__clang__) || defined(__GNUC__)
#define BL_ISOLATED_NOINLINE __attribute__((noinline))
#else
#define BL_ISOLATED_NOINLINE
#endif

namespace bl::isolated
{
    BL_ISOLATED_NOINLINE double runtime_f64(double value) noexcept
    {
        static volatile double sink;
        sink = value;
        return sink;
    }

    BL_ISOLATED_NOINLINE int runtime_i32(int value) noexcept
    {
        static volatile int sink;
        sink = value;
        return sink;
    }

    BL_ISOLATED_NOINLINE long runtime_long(long value) noexcept
    {
        static volatile long sink;
        sink = value;
        return sink;
    }

    BL_ISOLATED_NOINLINE long double runtime_long_double(long double value) noexcept
    {
        static volatile long double sink;
        sink = value;
        return sink;
    }

    BL_ISOLATED_NOINLINE bl::f128 runtime_f128(double value) noexcept
    {
        return runtime_f128(value, 0.0);
    }

    BL_ISOLATED_NOINLINE bl::f128 runtime_f128(double hi, double lo) noexcept
    {
        static volatile double sink[2];
        sink[0] = hi;
        sink[1] = lo;

        return bl::f128{ sink[0], sink[1] };
    }

    BL_ISOLATED_NOINLINE bl::f256 runtime_f256(double value) noexcept
    {
        return runtime_f256(value, 0.0, 0.0, 0.0);
    }

    BL_ISOLATED_NOINLINE bl::f256 runtime_f256(double x0, double x1, double x2, double x3) noexcept
    {
        static volatile double sink[4];
        sink[0] = x0;
        sink[1] = x1;
        sink[2] = x2;
        sink[3] = x3;

        return bl::f256{ sink[0], sink[1], sink[2], sink[3] };
    }

    BL_ISOLATED_NOINLINE void consume_address(void const* address) noexcept
    {
        #if defined(__clang__) || defined(__GNUC__)
        asm volatile("" : : "g"(address) : "memory");
        #elif defined(_MSC_VER)
        auto volatile consumed_address = address;
        (void)consumed_address;
        _ReadWriteBarrier();
        #else
        static volatile void const* sink;
        sink = address;
        #endif
    }
}
