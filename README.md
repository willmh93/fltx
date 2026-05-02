<p align="center">
  <img src="res/logo.webp" alt="fltx logo" width="80">
</p>

<p align="center">
  A header-only C++20 library for fixed-width extended-precision floating-point types.
</p>

`fltx` is for C++ code that needs more precision than `double`, while still keeping fixed-size scalar types, predictable performance, `constexpr` support, and ordinary C++ ergonomics.

Accuracy and performance are tested against equivalent-precision MPFR. MPFR is used as the reference oracle for validation and benchmarking, not as the dependency model `fltx` is trying to wrap or replace.

## Highlights

- Header-only C++20 library with no required runtime dependencies
- Fixed-size extended-precision scalar types [`bl::f128`](include/f128.h), [`bl::f256`](include/f256.h)
- `constexpr` arithmetic, comparisons, conversions, parsing and formatting
- `constexpr` math interface modeled after C++ [`<cmath>`](https://en.cppreference.com/w/cpp/header/cmath)
- Standard-library integration for streams, `std::numeric_limits`, `std::numbers`, and common stream manipulators
- Bitwise runtime/`constexpr` parity for [`bl::f128`](include/f128.h) and [`bl::f256`](include/f256.h) by default
- Native `std::` runtime performance for [`bl::f32`](include/fltx_types.h) and [`bl::f64`](include/fltx_types.h) by default, with optional parity mode via `FLTX_CONSTEXPR_PARITY`
- Suitable for lightweight native builds and WebAssembly / Emscripten targets
- Optional runtime-to-compile-time dispatch helpers
- Tested and Benchmarked against [`boost::multiprecision::mpfr_float_backend<>`](https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/mpfr_float.html) at comparable precision

## Core Types

| Type | Representation | Accuracy |
|---|---|---|
| [`bl::f32`](include/fltx_types.h) | Alias for native [`float`](https://en.cppreference.com/w/cpp/language/types) | Native [`float`](https://en.cppreference.com/w/cpp/language/types) precision |
| [`bl::f64`](include/fltx_types.h) | Alias for native [`double`](https://en.cppreference.com/w/cpp/language/types) | Native [`double`](https://en.cppreference.com/w/cpp/language/types) precision |
| [`bl::f128`](include/f128.h) | Double-double, stored as two [`double`](https://en.cppreference.com/w/cpp/language/types) limbs | Minimum 29 decimal digits across arithmetic and math functions |
| [`bl::f256`](include/f256.h) | Quad-double, stored as four [`double`](https://en.cppreference.com/w/cpp/language/types) limbs | Minimum 59 decimal digits across arithmetic and math functions |

The names refer to storage size:

```text
sizeof(bl::f128) == 16
sizeof(bl::f256) == 32
```

[`bl::f128`](include/f128.h) and [`bl::f256`](include/f256.h) are not IEEE [binary128](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#IEEE_754_quadruple-precision_binary_floating-point_format:_binary128) / [binary256](https://en.wikipedia.org/wiki/Octuple-precision_floating-point_format#IEEE_754_octuple-precision_binary_floating-point_format:_binary256) types, and they are not arbitrary-precision numbers. They provide more precision than native `double` while preserving a fixed-width, scalar-friendly floating-point model. Values are still approximate, so conditioning, cancellation, argument reduction, and algorithm design still matter.

## Quick Start

```cpp
#include <iomanip>
#include <iostream>
#include <limits>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

int main()
{
    constexpr f256 a = 1_qd / 3_qd;
    constexpr f256 b = 2_qd / 3_qd;
    constexpr f256 c = a + b;

    std::cout << std::setprecision(std::numeric_limits<f256>::digits10)
              << "a = " << a << '\n'
              << "b = " << b << '\n'
              << "a + b = " << c << '\n';
}
```

Output:

```text
a = 0.333333333333333333333333333333333333333333333333333333333333333
b = 0.666666666666666666666666666666666666666666666666666666666666667
a + b = 1
```

More examples are available in [examples/](examples/).

## Use Cases

`fltx` is aimed at workloads where both precision and speed matter:

- simulations and iterative numerical kernels
- geometric transforms
- numerically sensitive reference code
- compile-time validation and test generation
- fixed-size interop boundaries that cannot use arbitrary-precision types

It is a good fit when you want an extended-precision type that can still be passed around like a normal scalar in ordinary C++ code.

## Public Headers

| Header | Provides |
|---|---|
| [`fltx.h`](include/fltx.h) | Full library interface |
| [`fltx_core.h`](include/fltx_core.h) | Core numeric types, storage types, arithmetic, conversions, and standard numeric integration |
| [`fltx_math.h`](include/fltx_math.h) | Core types plus the `constexpr` `<cmath>`-style API |
| [`fltx_io.h`](include/fltx_io.h) | Core types plus parsing, formatting, string conversion, stream output, and literals |

Individual headers are also available when you want a smaller include surface:

| Header | Provides |
|---|---|
| [`f128.h`](include/f128.h), [`f256.h`](include/f256.h) | Individual extended-precision types, storage types, and core operations |
| [`f32_math.h`](include/f32_math.h), [`f64_math.h`](include/f64_math.h), [`f128_math.h`](include/f128_math.h), [`f256_math.h`](include/f256_math.h) | Math APIs for individual floating-point types |
| [`f128_io.h`](include/f128_io.h), [`f256_io.h`](include/f256_io.h) | IO and literals for individual extended-precision types |
| [`fltx_types.h`](include/fltx_types.h) | Type aliases, concepts, `FloatType`, and enum helpers |
| [`constexpr_dispatch.h`](include/constexpr_dispatch.h) | Standalone runtime-to-compile-time dispatch utility |
| [`fltx_dispatch.h`](include/fltx_dispatch.h) | Dispatch helpers for mapping `FloatType` values to `f32`, `f64`, `f128`, or `f256` |

## Numeric Types

The main user-facing types are:

```text
bl::f32     // float alias
bl::f64     // double alias
bl::f128    // double-double scalar type
bl::f256    // quad-double scalar type

bl::f128_s  // trivial storage form
bl::f256_s  // trivial storage form
```

The trivial storage forms are useful when standard-layout storage matters, such as packed structures, unions, binary buffers, or interop boundaries. They convert cleanly to and from the full scalar types:

```cpp
bl::f128_s a { 5.0 };
bl::f128   b = 5.0f;

bl::f128   c = a + b;
bl::f128_s d = c;
```

Common type traits are available from [`fltx_types.h`](include/fltx_types.h):

```cpp
bl::is_f32_v<T>
bl::is_f64_v<T>
bl::is_f128_v<T>
bl::is_f256_v<T>

bl::is_fltx_v<T>
bl::is_floating_point_v<T>
bl::is_arithmetic_v<T>
```

## Math API

[`fltx_math.h`](include/fltx_math.h) provides a `constexpr` math interface modeled after C++ <cmath> API across [`bl::f32`](include/fltx_types.h), [`bl::f64`](include/fltx_types.h), [`bl::f128`](include/f128.h), and [`bl::f256`](include/f256.h).

This lets generic numeric code switch precision without changing its math calls:

```cpp
template<class T>
constexpr T radius(T x, T y)
{
    return bl::sqrt(x * x + y * y);
}

using namespace bl::literals;

constexpr bl::f32   a = radius(1.0f, 2.0f);
constexpr bl::f64   b = radius(1.0,  2.0);
constexpr bl::f128  c = radius(1.0_dd, 2.0_dd);
constexpr bl::f256  d = radius(1.0_qd, 2.0_qd);
```

Supported function groups:

| constexpr | Category | Functions |
|---|---|---|
| ✅ | Arithmetic | `abs`, `fma` |
| ✅ | Rounding | `floor`, `ceil`, `trunc`, `round`, `lround`, `llround`, `nearbyint`, `rint`, `lrint`, `llrint` |
| ✅ | Remainders | `fmod`, `remainder`, `remquo` |
| ✅ | Min / max / sign | `fmin`, `fmax`, `fdim`, `copysign` |
| ✅ | Roots / powers | `sqrt`, `cbrt`, `hypot`, `pow` |
| ✅ | Exp / log | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logb`, `ilogb` |
| ✅ | Trigonometry | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| ✅ | Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| ✅ | Special functions | `erf`, `erfc`, `lgamma`, `tgamma` |
| ✅ | Scaling / layout | `ldexp`, `scalbn`, `scalbln`, `frexp`, `modf`, `nextafter`, `nexttoward` |


## IO and Literals

[`fltx_io.h`](include/fltx_io.h) adds `constexpr`-capable parsing, formatting, stream output, string conversion, and the `_dd` / `_qd` literals:

```cpp
using namespace bl;
using namespace bl::literals;

constexpr f128 a = 1.25_dd;
constexpr f256 b = "3.1415926535897932384626433832795028841971"_qd;

constexpr auto s1 = bl::to_string(b);      // static_string<512>
std::string s2    = bl::to_std_string(b);
```

Stream output supports `std::setprecision`, `std::fixed`, `std::scientific`, `std::showpoint`, `std::showpos`, and `std::uppercase`.

## Constexpr Dispatch

[`fltx_dispatch.h`](include/fltx_dispatch.h) includes a small runtime-to-compile-time dispatch layer.

This lets runtime values such as [`FloatType::F128`](include/fltx_types.h) or [`FloatType::F256`](include/fltx_types.h) select a compile-time type, so the called function still compiles as a normal template specialization.

```cpp
#include <iostream>

#include <fltx.h>

using namespace bl;

template<class T>
void run_kernel(int width, int height)
{
    T x = T{ width } / T{ height };
    T y = bl::sqrt(x) + bl::sin(x);

    std::cout << y << '\n';
}

int main()
{
    FloatType precision = FloatType::F256;

    table_invoke(
        dispatch_table(run_kernel, 1920, 1080),
        enum_type(precision)
    );
}
```

<details>
<summary>Mapping a custom enum to a compile-time type</summary>

[`constexpr_dispatch.h`](include/constexpr_dispatch.h) is the lower-level dispatch utility used by [`fltx_dispatch.h`](include/fltx_dispatch.h). It can also be used directly when you want your own runtime enum to select one of several compile-time types:

```cpp
#include <constexpr_dispatch.h>

enum struct Backend { Cpu, Gpu, COUNT };

struct CpuBackend {};
struct GpuBackend {};

bl_map_enum_to_type(Backend::Cpu, CpuBackend);
bl_map_enum_to_type(Backend::Gpu, GpuBackend);

template<class BackendT, bool Debug>
void run()
{
    if constexpr (Debug)
    {
        // Debug-only path.
    }
}

int main()
{
    Backend backend = Backend::Gpu;
    bool debug = true;

    bl::table_invoke(
        bl::dispatch_table(run),
        bl::enum_type(backend),
        debug
    );
}
```

</details>

## Installation

### vcpkg

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

### CMake

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

`fltx` is tested and benchmarked against [`boost::multiprecision::mpfr_float_backend<>`](https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/mpfr_float.html) at comparable precision levels.

<table>
<tr>
<td><img src="res/f128_typical_ratios.svg" alt="f128 benchmark ratio"></td>
<td><img src="res/f256_typical_ratios.svg" alt="f256 benchmark ratio"></td>
</tr>
</table>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
