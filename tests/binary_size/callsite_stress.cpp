#include <fltx/f128/math.h>
#include <fltx/f256/math.h>
#include "isolated_runtime.h"

#if !defined(FLTX_BINARY_SIZE_STRESS_CASE)
#  error "FLTX_BINARY_SIZE_STRESS_CASE must be defined."
#endif

#if !defined(FLTX_BINARY_SIZE_STRESS_REPEAT)
#  define FLTX_BINARY_SIZE_STRESS_REPEAT 64
#endif

#if defined(FLTX_BINARY_SIZE_STRESS_F128)
using fltx_binary_size_value = bl::f128;

BL_FORCE_INLINE fltx_binary_size_value fltx_binary_size_runtime(double value) noexcept
{
    return bl::isolated::runtime_f128(value);
}
#elif defined(FLTX_BINARY_SIZE_STRESS_F256)
using fltx_binary_size_value = bl::f256;

BL_FORCE_INLINE fltx_binary_size_value fltx_binary_size_runtime(double value) noexcept
{
    return bl::isolated::runtime_f256(value);
}
#else
#  error "Define FLTX_BINARY_SIZE_STRESS_F128 or FLTX_BINARY_SIZE_STRESS_F256."
#endif

template<int I>
BL_FORCE_INLINE fltx_binary_size_value fltx_binary_size_bias() noexcept
{
    return fltx_binary_size_value{ 1.0 + static_cast<double>((I % 11) + 1) * 0.0001 };
}

template<int I>
BL_FORCE_INLINE fltx_binary_size_value fltx_binary_size_step(
    fltx_binary_size_value x,
    fltx_binary_size_value a,
    fltx_binary_size_value b,
    fltx_binary_size_value c) noexcept
{
    if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 1)
        return x + a + fltx_binary_size_bias<I>();
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 2)
        return x * a * fltx_binary_size_bias<I>();
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 3)
        return x / (a + fltx_binary_size_bias<I>());
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 4)
        return ((x + a) * b - c) / (a + fltx_binary_size_bias<I>());
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 5)
        return bl::sqrt((x * x) + fltx_binary_size_bias<I>());
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 6)
        return bl::sin((x + fltx_binary_size_bias<I>()) * 0.001);
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 7)
        return bl::cos((x + fltx_binary_size_bias<I>()) * 0.001);
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 8)
        return bl::tan((x + fltx_binary_size_bias<I>()) * 0.001);
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 9)
        return bl::exp((x + fltx_binary_size_bias<I>()) * 0.001);
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 10)
        return bl::log((x * x) + fltx_binary_size_bias<I>());
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 11)
        return bl::pow((x * 0.001) + fltx_binary_size_bias<I>(), b);
    else if constexpr (FLTX_BINARY_SIZE_STRESS_CASE == 12)
    {
        fltx_binary_size_value s;
        fltx_binary_size_value c_out;
        (void)bl::sincos((x + fltx_binary_size_bias<I>()) * 0.001, s, c_out);
        return s + c_out;
    }
    else
    {
        static_assert(FLTX_BINARY_SIZE_STRESS_CASE >= 1 && FLTX_BINARY_SIZE_STRESS_CASE <= 12);
        return x;
    }
}

template<int I>
BL_FORCE_INLINE fltx_binary_size_value fltx_binary_size_run(
    fltx_binary_size_value x,
    fltx_binary_size_value a,
    fltx_binary_size_value b,
    fltx_binary_size_value c) noexcept
{
    if constexpr (I == 0)
        return x;
    else
        return fltx_binary_size_run<I - 1>(fltx_binary_size_step<I>(x, a, b, c), a, b, c);
}

int main()
{
    auto x = fltx_binary_size_runtime(1.125);
    auto a = fltx_binary_size_runtime(1.0001);
    auto b = fltx_binary_size_runtime(1.25);
    auto c = fltx_binary_size_runtime(0.0625);

    auto value = fltx_binary_size_run<FLTX_BINARY_SIZE_STRESS_REPEAT>(x, a, b, c);
    bl::isolated::keep_value(value);
    return 0;
}
