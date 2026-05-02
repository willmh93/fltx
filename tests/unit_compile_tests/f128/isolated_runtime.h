// isolated_runtime.h
#pragma once

namespace bl
{
    struct f128;
    struct f256;
}

namespace bl::isolated
{
    double runtime_f64(double value) noexcept;
    int runtime_i32(int value) noexcept;
    long runtime_long(long value) noexcept;
    long double runtime_long_double(long double value) noexcept;
    bl::f128 runtime_f128(double value) noexcept;
    bl::f128 runtime_f128(double hi, double lo) noexcept;
    bl::f256 runtime_f256(double value) noexcept;
    bl::f256 runtime_f256(double x0, double x1, double x2, double x3) noexcept;

    void consume_address(void const* address) noexcept;

    template<class T>
    void keep_value(T const& value) noexcept
    {
        consume_address(static_cast<void const*>(&value));
    }
}
