<p align="center">
  <img src="res/logo.webp" alt="fltx logo" width="80"><br>
  Extended-precision floating point and constexpr math for C++20<br>
  Fast · Precise · Lightweight
</p>

[![Precision Linux](https://github.com/willmh93/fltx/actions/workflows/precision-tests-linux.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/precision-tests-linux.yml)
[![Precision Windows](https://github.com/willmh93/fltx/actions/workflows/precision-tests-windows.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/precision-tests-windows.yml)
[![Precision macOS](https://github.com/willmh93/fltx/actions/workflows/precision-tests-mac.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/precision-tests-mac.yml)
[![Parity Linux](https://github.com/willmh93/fltx/actions/workflows/parity-tests-linux.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/parity-tests-linux.yml)
[![Parity Windows](https://github.com/willmh93/fltx/actions/workflows/parity-tests-windows.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/parity-tests-windows.yml)
[![Parity macOS](https://github.com/willmh93/fltx/actions/workflows/parity-tests-mac.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/parity-tests-mac.yml)
[![IO Linux](https://github.com/willmh93/fltx/actions/workflows/io-tests-linux.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/io-tests-linux.yml)
[![IO Windows](https://github.com/willmh93/fltx/actions/workflows/io-tests-windows.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/io-tests-windows.yml)
[![IO macOS](https://github.com/willmh93/fltx/actions/workflows/io-tests-mac.yml/badge.svg)](https://github.com/willmh93/fltx/actions/workflows/io-tests-mac.yml)

`fltx` is for code that needs more precision than `double`, without giving up fixed-size scalar types, predictable performance, `constexpr` support, or familiar C++ ergonomics.

## Highlights

- Fixed-size extended-precision scalar types: [`bl::f128`](include/fltx/f128.h) and [`bl::f256`](include/fltx/f256.h)
- `constexpr` arithmetic, comparisons, conversions, parsing, formatting, and [`<cmath>`](https://en.cppreference.com/w/cpp/header/cmath)-style math
- Accuracy and performance validated against [`boost::multiprecision::mpfr_float_backend<>`](https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/mpfr_float.html) at equivalent precision
- [`bl::f256`](include/fltx/f256.h) has an expression node system which recognises and fuses common arithmetic shapes, including product sums, dot-product-plus-bias expressions, and scaled linear combinations. This reduces intermediate rounding and temporary value materialisation, allowing those shapes to run through specialised fused evaluation paths.
- Selected [`bl::f128`](include/fltx/f128.h) and [`bl::f256`](include/fltx/f256.h) kernels use platform SIMD by default where supported: x86/x64 SSE2 with FMA when available, AArch64 NEON, and WebAssembly `wasm128` SIMD via Emscripten
- Compatible with fast-math builds; compiled runtime sources default to fast-math for speed, while `FLTX_CONSTEXPR_PARITY` remains available when bitwise-identical runtime and `constexpr` results matter
- No required runtime dependencies
- Standard-library integration for streams, `std::numeric_limits`, `std::numbers`, and common stream manipulators
- Runtime code favors native performance by default
- Test builds can define `FLTX_SIMULATE_CONSTEVAL_MODE` to run runtime samples through the branches used during constant evaluation without enabling `FLTX_CONSTEXPR_PARITY`
- Suitable for lightweight native builds, WebAssembly / Emscripten
- Optional runtime-to-compile-time dispatch helper for template-specialized kernels

## Core Types

| Type | Representation | Accuracy |
|---|---|---|
| [`bl::f32`](include/fltx/types.h) | Alias for native [`float`](https://en.cppreference.com/w/cpp/language/types) | Native [`float`](https://en.cppreference.com/w/cpp/language/types) precision |
| [`bl::f64`](include/fltx/types.h) | Alias for native [`double`](https://en.cppreference.com/w/cpp/language/types) | Native [`double`](https://en.cppreference.com/w/cpp/language/types) precision |
| [`bl::f128`](include/fltx/f128.h) | Double-double, stored as two [`double`](https://en.cppreference.com/w/cpp/language/types) limbs | Minimum 29 decimal digits across arithmetic and math functions |
| [`bl::f256`](include/fltx/f256.h) | Quad-double, stored as four [`double`](https://en.cppreference.com/w/cpp/language/types) limbs | Minimum 59 decimal digits across arithmetic and math functions |

The names refer to storage size:

```text
sizeof(bl::f128) == 16
sizeof(bl::f256) == 32
```

[`bl::f128`](include/fltx/f128.h) and [`bl::f256`](include/fltx/f256.h) are not IEEE [binary128](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#IEEE_754_quadruple-precision_binary_floating-point_format:_binary128) / [binary256](https://en.wikipedia.org/wiki/Octuple-precision_floating-point_format#IEEE_754_octuple-precision_binary_floating-point_format:_binary256) types, and they are not arbitrary-precision numbers. They provide more precision than native `double` while preserving a fixed-width, scalar-friendly floating-point model. Values are still approximate, so conditioning, cancellation, argument reduction, and algorithm design still matter.

## f256 Expression Fusion

[`bl::f256`](include/fltx/f256.h) has an expression node system which recognises common arithmetic shapes, including product sums, dot-product-plus-bias expressions, and scaled linear combinations.

For example:

```cpp
f256 r = a * b + c * d + e;
```

is kept as a small compile-time expression and lowered to the existing fused product-sum body. This avoids materialising and normalising each intermediate quad-double result before the final value is needed.

The matcher also accepts selected equivalent spellings, such as reordered product/value terms and scaled linear forms. That gives common kernels like affine transforms, small matrix-vector operations, polynomial-style updates, and recurrence relations a faster path without requiring users to call specialised helpers or the implementation to maintain a full symbolic optimiser.

The fused bodies remain explicit and bounded, which keeps compile-time and code-size costs under control.

## Quick Start

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
    constexpr f256 d = bl::sin(a + b);

    std::cout
        << std::setprecision(std::numeric_limits<f256>::digits10)
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "c = " << c << "\n"
        << "d = " << d << "\n\n";
}
```

Output:

```text
a = 0.333333333333333333333333333333333333333333333333333333333333333
b = 0.666666666666666666666666666666666666666666666666666666666666667
c = 1
d = 0.84147098480789650665250232163029899962256306079837106567275171
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
| [`fltx/core.h`](include/fltx/core.h) | Core numeric types, storage types, arithmetic, conversions, and standard numeric integration |
| [`fltx/math.h`](include/fltx/math.h) | Core types plus the `constexpr` `<cmath>`-style API |
| [`fltx/io.h`](include/fltx/io.h) | Core types plus parsing, formatting, string conversion, stream output, and literals |

Individual headers are also available when you want a smaller include surface:

| Header | Provides |
|---|---|
| [`fltx/f128.h`](include/fltx/f128.h), [`fltx/f256.h`](include/fltx/f256.h) | Individual extended-precision types, storage types, and core operations |
| [`fltx/f32/math.h`](include/fltx/f32/math.h), [`fltx/f64/math.h`](include/fltx/f64/math.h), [`fltx/f128/math.h`](include/fltx/f128/math.h), [`fltx/f256/math.h`](include/fltx/f256/math.h) | Math APIs for individual floating-point types |
| [`fltx/f128/io.h`](include/fltx/f128/io.h), [`fltx/f256/io.h`](include/fltx/f256/io.h) | IO and literals for individual extended-precision types |
| [`fltx/types.h`](include/fltx/types.h) | Type aliases, concepts, `FloatType`, and enum helpers |
| [`fltx/template_dispatch.h`](include/fltx/template_dispatch.h) | Standalone runtime-to-compile-time dispatch-table utility |
| [`fltx/dispatch.h`](include/fltx/dispatch.h) | Dispatch helpers for mapping `FloatType` values to `f32`, `f64`, `f128`, or `f256` |

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

Common type traits are available from [`fltx/types.h`](include/fltx/types.h):

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

[`fltx/math.h`](include/fltx/math.h) provides a `constexpr` math interface modeled after C++ <cmath> API across [`bl::f32`](include/fltx/types.h), [`bl::f64`](include/fltx/types.h), [`bl::f128`](include/fltx/f128.h), and [`bl::f256`](include/fltx/f256.h).

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

[`fltx/io.h`](include/fltx/io.h) adds `constexpr`-capable parsing, formatting, stream output, string conversion, and the `_dd` / `_qd` literals:

```cpp
using namespace bl;
using namespace bl::literals;

constexpr f128 a = 1.25_dd;
constexpr f256 b = "3.1415926535897932384626433832795028841971"_qd;

constexpr auto s1 = bl::to_string(b);      // static_string<512>
std::string s2    = bl::to_std_string(b);
```

Stream output supports `std::setprecision`, `std::fixed`, `std::scientific`, `std::showpoint`, `std::showpos`, and `std::uppercase`.

## Template Dispatch

[`fltx/dispatch.h`](include/fltx/dispatch.h) includes a small runtime-to-template dispatch layer.

This lets runtime values such as [`FloatType::F128`](include/fltx/types.h) or [`FloatType::F256`](include/fltx/types.h) select a compile-time type, so the called function still compiles as a normal template specialization.

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

[`fltx/template_dispatch.h`](include/fltx/template_dispatch.h) is the lower-level dispatch utility used by [`fltx/dispatch.h`](include/fltx/dispatch.h). It can also be used directly when you want your own runtime enum to select one of several compile-time types:

```cpp
#include <fltx/template_dispatch.h>

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
add_subdirectory(/path/to/fltx fltx-build)
target_link_libraries(main PRIVATE fltx::fltx)
```

If you need the old include-only style, define `FLTX_HEADER_ONLY` before including the headers. That keeps constexpr and runtime calls in headers, but gives up the precompiled runtime helpers:

```cmake
target_compile_definitions(main PRIVATE FLTX_HEADER_ONLY)
target_include_directories(main PRIVATE /path/to/fltx/include)
```

## Building fltx

These steps are for contributors who want to build the `fltx` repository itself, including tests and benchmarks. They install the repo-local vcpkg dependencies used by the test suite, such as Catch2, Boost.Multiprecision, GMP, and MPFR.

If you only want to use `fltx` from your own project, you do not need this full setup. Use the vcpkg or CMake installation instructions above instead.

<details>
<summary>Windows</summary>

Install Visual Studio with the C++ desktop workload, then use a Developer PowerShell or Developer Command Prompt:

```powershell
git clone --recurse-submodules https://github.com/willmh93/fltx.git
cd fltx

.\vcpkg\bootstrap-vcpkg.bat

cmake --preset vs2026
cmake --build build\vs2026 --config Release
```

The `vs2026` preset uses Visual Studio/MSBuild with the MSVC compiler. If your Visual Studio version differs, create or select the matching CMake preset before configuring.

</details>

<details>
<summary>macOS</summary>

This is the fresh Apple Silicon macOS setup used for contributor builds on macOS Sequoia:

```bash
# 1. Install Homebrew, if missing
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Add Homebrew to zsh PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Install tools needed by CMake and vcpkg ports
brew install cmake pkg-config autoconf autoconf-archive automake libtool m4

# 4. Clone fltx with submodules
cd ~/Documents
git clone --recurse-submodules https://github.com/willmh93/fltx.git
cd fltx

# 5. Bootstrap repo-local vcpkg
./vcpkg/bootstrap-vcpkg.sh

# 6. Configure; this installs vcpkg dependencies like Catch2, GMP, and MPFR
cmake --preset macos-release

# 7. Build
cmake --build --preset macos-release --config Release
```

For an already-cloned repo:

```bash
cd ~/Documents/fltx
git pull
git submodule update --init --recursive
./vcpkg/bootstrap-vcpkg.sh
cmake --preset macos-release
cmake --build --preset macos-release --config Release
```

</details>

<details>
<summary>Linux</summary>

Linux setup has not been manually validated yet. The expected Ubuntu/Debian flow is:

```bash
sudo apt update
sudo apt install -y git build-essential cmake ninja-build pkg-config autoconf autoconf-archive automake libtool m4

git clone --recurse-submodules https://github.com/willmh93/fltx.git
cd fltx

./vcpkg/bootstrap-vcpkg.sh

cmake --preset ninja-release
cmake --build --preset ninja-release
```

Other distributions should use equivalent packages for a C++20 compiler, CMake, Ninja, pkg-config, and the autotools used by the GMP/MPFR vcpkg ports.

</details>

## Benchmarks

<img src="res/bench/benchmark_table.svg" alt="fltx benchmark table" width="100%">

`fltx` is tested and benchmarked against [`boost::multiprecision::mpfr_float_backend<>`](https://www.boost.org/doc/libs/release/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/mpfr_float.html) at comparable precision levels.

<details>
<summary>Windows benchmark charts</summary>

<img src="res/bench/windows/MSVC_f128_typical_ratios.svg" alt="Windows MSVC f128 benchmark ratios" width="100%">

<img src="res/bench/windows/MSVC_f256_typical_ratios.svg" alt="Windows MSVC f256 benchmark ratios" width="100%">

<img src="res/bench/windows/MinGW_f128_typical_ratios.svg" alt="Windows MinGW f128 benchmark ratios" width="100%">

<img src="res/bench/windows/MinGW_f256_typical_ratios.svg" alt="Windows MinGW f256 benchmark ratios" width="100%">

</details>

<details>
<summary>Linux benchmark charts</summary>

<img src="res/bench/linux/GCC_f128_typical_ratios.svg" alt="Linux GCC f128 benchmark ratios" width="100%">

<img src="res/bench/linux/GCC_f256_typical_ratios.svg" alt="Linux GCC f256 benchmark ratios" width="100%">

<img src="res/bench/linux/Clang_f128_typical_ratios.svg" alt="Linux Clang f128 benchmark ratios" width="100%">

<img src="res/bench/linux/Clang_f256_typical_ratios.svg" alt="Linux Clang f256 benchmark ratios" width="100%">

</details>

<details>
<summary>WebAssembly benchmark charts</summary>

<img src="res/bench/wasm32/Nodejs_f128_typical_ratios.svg" alt="WebAssembly Node.js f128 benchmark ratios" width="100%">

<img src="res/bench/wasm32/Chrome_f128_typical_ratios.svg" alt="WebAssembly Chrome f128 benchmark ratios" width="100%">

<img src="res/bench/wasm32/Chrome_f256_typical_ratios.svg" alt="WebAssembly Chrome f256 benchmark ratios" width="100%">

</details>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
