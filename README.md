<img src="res/logo.webp" alt="fltx logo" width="90">

A C++20 header-only library for fixed-width extended-precision floating-point work.

The goal is simple: **more numerical headroom, `double`-like ergonomics, and strong performance for fixed precision workloads** without moving all the way to arbitrary precision.

## Highlights

- C++20 header-only design
- `constexpr`-capable math functions for `f32`, `f64`, `f128`, and `f256`, with a `<cmath>`-like API for writing generic numeric code.
- Literal support, operators, conversions, comparisons, and common math functions
- Standard-library integration: `std::ostream`, `std::numeric_limits`, `std::numbers`, and stream manipulators such as `std::setprecision`
- **Optional:** Bitwise parity between constexpr and runtime results. To enable, define `FLTX_CONSTEXPR_PARITY` prior to including math headers (impacts performance, only use if required)
- **Optional:** Runtime-to-compile-time dispatch helpers
- Benchmarked against `boost::multiprecision::mpfr_float_backend<digits>` at comparable precision (results below)
 
## core types

| Type | Backing representation |
|---|---|
| `f32` | Alias for native `float` |
| `f64` | Alias for native `double` |
| `f128` | Double-double precision, stored as two `double` limbs |
| `f256` | Quad-double precision, stored as four `double` limbs |

## constexpr support

| Function | `bl::f32` | `bl::f64` | `bl::f128` | `bl::f256` |
|---|---:|---:|---:|---:|
| `bl::abs` | ✅ | ✅ | ✅ | ✅ |
| `bl::floor` | ✅ | ✅ | ✅ | ✅ |
| `bl::ceil` | ✅ | ✅ | ✅ | ✅ |
| `bl::trunc` | ✅ | ✅ | ✅ | ✅ |
| `bl::round` | ✅ | ✅ | ✅ | ✅ |
| `bl::lround` | ✅ | ✅ | ✅ | ✅ |
| `bl::llround` | ✅ | ✅ | ✅ | ✅ |
| `bl::nearbyint` | ✅ | ✅ | ✅ | ✅ |
| `bl::rint` | ✅ | ✅ | ✅ | ✅ |
| `bl::lrint` | ✅ | ✅ | ✅ | ✅ |
| `bl::llrint` | ✅ | ✅ | ✅ | ✅ |
| `bl::fmod` | ✅ | ✅ | ✅ | ✅ |
| `bl::remainder` | ✅ | ✅ | ✅ | ✅ |
| `bl::remquo` | ✅ | ✅ | ✅ | ✅ |
| `bl::fma` | ✅ | ✅ | ✅ | ✅ |
| `bl::fmin` | ✅ | ✅ | ✅ | ✅ |
| `bl::fmax` | ✅ | ✅ | ✅ | ✅ |
| `bl::fdim` | ✅ | ✅ | ✅ | ✅ |
| `bl::copysign` | ✅ | ✅ | ✅ | ✅ |
| `bl::sqrt` | ✅ | ✅ | ✅ | ✅ |
| `bl::cbrt` | ✅ | ✅ | ✅ | ✅ |
| `bl::hypot` | ✅ | ✅ | ✅ | ✅ |
| `bl::exp` | ✅ | ✅ | ✅ | ✅ |
| `bl::exp2` | ✅ | ✅ | ✅ | ✅ |
| `bl::expm1` | ✅ | ✅ | ✅ | ✅ |
| `bl::log` | ✅ | ✅ | ✅ | ✅ |
| `bl::log2` | ✅ | ✅ | ✅ | ✅ |
| `bl::log10` | ✅ | ✅ | ✅ | ✅ |
| `bl::log1p` | ✅ | ✅ | ✅ | ✅ |
| `bl::pow` | ✅ | ✅ | ✅ | ✅ |
| `bl::sin` | ✅ | ✅ | ✅ | ✅ |
| `bl::cos` | ✅ | ✅ | ✅ | ✅ |
| `bl::tan` | ✅ | ✅ | ✅ | ✅ |
| `bl::asin` | ✅ | ✅ | ✅ | ✅ |
| `bl::acos` | ✅ | ✅ | ✅ | ✅ |
| `bl::atan` | ✅ | ✅ | ✅ | ✅ |
| `bl::atan2` | ✅ | ✅ | ✅ | ✅ |
| `bl::sinh` | ✅ | ✅ | ✅ | ✅ |
| `bl::cosh` | ✅ | ✅ | ✅ | ✅ |
| `bl::tanh` | ✅ | ✅ | ✅ | ✅ |
| `bl::asinh` | ✅ | ✅ | ✅ | ✅ |
| `bl::acosh` | ✅ | ✅ | ✅ | ✅ |
| `bl::atanh` | ✅ | ✅ | ✅ | ✅ |
| `bl::erf` | ✅ | ✅ | ✅ | ✅ |
| `bl::erfc` | ✅ | ✅ | ✅ | ✅ |
| `bl::lgamma` | ✅ | ✅ | ✅ | ✅ |
| `bl::tgamma` | ✅ | ✅ | ✅ | ✅ |
| `bl::ldexp` | ✅ | ✅ | ✅ | ✅ |
| `bl::scalbn` | ✅ | ✅ | ✅ | ✅ |
| `bl::scalbln` | ✅ | ✅ | ✅ | ✅ |
| `bl::frexp` | ✅ | ✅ | ✅ | ✅ |
| `bl::modf` | ✅ | ✅ | ✅ | ✅ |
| `bl::ilogb` | ✅ | ✅ | ✅ | ✅ |
| `bl::logb` | ✅ | ✅ | ✅ | ✅ |
| `bl::nextafter` | ✅ | ✅ | ✅ | ✅ |
| `bl::nexttoward` | ✅ | ✅ | ✅ | ✅ |

## Use case

`fltx` is aimed at workloads where **both speed and precision matter**:

- simulations and iterative numerical kernels
- geometric transforms
- numerically sensitive reference code
- constexpr-heavy validation

It is a good fit when you want an extended-precision type that can still be passed around like a normal scalar in ordinary C++ code.

## Quick example

```cpp
#include <iostream>
#include <iomanip>

#include <fltx.h>
using namespace bl;
using namespace bl::literals;

int main()
{
    constexpr f256 a = 1_qd / 3_qd;
    constexpr f256 b = 2_qd / 3_qd;
    constexpr f256 c = a + b;

    std::cout << std::setprecision(std::numeric_limits<f256>::digits10)
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "a + b = " << c << "\n";
}
```

Output:

```text
a = 0.333333333333333333333333333333333333333333333333333333333333333
b = 0.666666666666666666666666666666666666666666666666666666666666667
a + b = 1
```

## Public headers

| Umbrella Headers | Provides |
|---|---|
| `fltx.h` | full library |
| `fltx_core.h` | `f32`, `f64`, `f128`, `f256`, storage types (`f128_s`, `f256_s`)<br>arithmetic, conversions, and standard numeric integration |
| `fltx_math.h` | `f32`, `f64`, `f128`, `f256` constexpr-capable `<cmath>` math interface |
| `fltx_io.h` | `f32`, `f64`, `f128`, `f256` parsing, formatting, string conversion, stream output, and literals |

| Individual Headers | Provides |
|---|---|
| `f128.h`<br>`f256.h` | individual extended-precision types and their core operations |
| `f32_math.h`<br>`f64_math.h`<br>`f128_math.h`<br>`f256_math.h` | constexpr-capable `<cmath>` math interface for individual floating-point types |
| `f128_io.h`<br>`f256_io.h` | IO and literals for one extended-precision type |
| `fltx_types.h` | aliases, concepts, `FloatType`, and enum helpers |
| `constexpr_dispatch.h` | standalone constexpr dispatch utility |
| `fltx_dispatch.h` | includes `constexpr_dispatch.h`<br>and `bl::enum_type(FloatType) -> [f32, f64, f128, f256]` type mapping |

## Numeric types

The main user-facing types are:

```
bl::f32
bl::f64
bl::f128
bl::f256
```

`f32` and `f64` are aliases for native `float` and `double`. `f128` and `f256` are fixed-width extended-precision types.

There are also trivial storage forms:

```cpp
bl::f128_s
bl::f256_s
```

These are useful when standard-layout storage matters, such as packed structures, unions, binary buffers, or interop boundaries. They can be used interchangeably. 

```cpp
f128_s a { 5.0 };
f128   b = 5.0f;

f128   c = a + b;
f128_s d = c;
```

The common type traits are available from `fltx_types.h`:

```cpp
bl::is_f32_v<T>
bl::is_f64_v<T>
bl::is_f128_v<T>
bl::is_f256_v<T>

bl::is_fltx_v<T>
bl::is_floating_point_v<T>
bl::is_arithmetic_v<T>
```

## IO and literals

`fltx_io.h` adds constexpr-capable parsing, formatting, stream output, string conversion, and the `_dd` / `_qd` literals:

```cpp
using namespace bl;
using namespace bl::literals;

constexpr f128 a = 1.25_dd;
constexpr f256 b = "3.1415926535897932384626433832795028841971"_qd;

constexpr auto s1 = bl::to_string(b);
std::string s2    = bl::to_std_string(b);
```

## Math

`fltx_math.h` provides a `<cmath>`-style API for `bl::f32`, `bl::f64`, `bl::f128`, and `bl::f256`.

This lets generic numeric code switch precision without changing its math calls:
```cpp
template<class T>
constexpr T radius(T x, T y)
{
    return bl::sqrt(x * x + y * y);
}

constexpr f32   a = radius(1.0f, 2.0f);
constexpr f64   b = radius(1.0,  2.0);
constexpr f128  c = radius(1.0_dd, 2.0_dd);
constexpr f256  d = radius(1.0_qd, 2.0_qd);
```

## constexpr dispatch

`fltx_dispatch.h` includes a small runtime-to-compile-time dispatch layer.

This is useful when a user setting, file format, UI option, or benchmark parameter chooses the precision at runtime, but the actual kernel should still compile as a type-specialized template.

```cpp
#include <fltx_dispatch.h>

using namespace bl;

template<class T>
void run_kernel(int width, int height)
{
    T scale = 1.0 / T{ width + height };
    // ...
}

int main()
{
    FloatType type = FloatType::F256;

    table_invoke(
        dispatch_table(run_kernel, 1920, 1080),
        enum_type(type)
    );
}
```

`bl::enum_type(type)` maps `FloatType::F32`, `FloatType::F64`, `FloatType::F128`, and `FloatType::F256` to `bl::f32`, `bl::f64`, `bl::f128`, and `bl::f256`.

You can also dispatch on enum or bool values as compile-time non-type template arguments, which makes it useful for generating optimized variants of numerical kernels without hand-writing large switch blocks.

## Precision model

`f128` and `f256` are multi-limb floating-point types built from `double` components:

```text
sizeof(f128) == 16
sizeof(f256) == 32
```

The names refer to storage size, not IEEE binary128 or binary256 semantics.

These types give a large precision increase over native `double`, while preserving a familiar floating-point programming model. They are still floating-point approximations, not exact values. Conditioning, cancellation, argument reduction, and algorithm design still matter.

## What fltx is not

`fltx` is not an arbitrary-precision library, a symbolic math package, decimal arithmetic package, exact rational type, or true IEEE binary128 / binary256 implementation.

If you need unbounded precision or exact arithmetic, use a multiprecision or symbolic package instead. `fltx` is about a different trade-off: **fixed-size extended precision with practical ergonomics and strong performance**.

## vcpkg

Add the bitloop registry to `vcpkg-configuration.json`:

```json
{
  "default-registry": { ... },
  "registries": [
    {
      "kind": "git",
      "baseline": "45fca757b3ddbcbe804ea7d84b3699a469fda448",
      "repository": "https://github.com/willmh93/bitloop-registry.git",
      "packages": ["fltx"]
    }
  ]
}
```

Then add `fltx` to `vcpkg.json`:

```json
{
  "name": "myapp",
  "version": "1.0.0",
  "dependencies": [
    "fltx"
  ]
}
```

## CMake

With vcpkg:

```cmake
find_package(fltx CONFIG REQUIRED)
target_link_libraries(main PRIVATE fltx::fltx)
```

Or include it directly:

```cmake
target_include_directories(main PRIVATE /path/to/fltx/include)
```

## Benchmarks

`fltx` is tested and benchmarked against `boost::multiprecision::mpfr_float_backend<digits>` at comparable precision levels.

<img src="res/f128_hard_ratios.svg" alt="f128 benchmark ratio" width="600">
<img src="res/f256_hard_ratios.svg" alt="f256 benchmark ratio" width="600">

## License

This project is licensed under the MIT Licence.
