<img src="res/logo.webp" alt="fltx logo" width="90">

A modern C++ header-only library for extended-precision floating-point work.

It provides high-precision numeric types designed for code that needs **substantially more precision than `double`**, while staying faster and more light-weight than arbitrary-precision libraries.

At its core, **`fltx`** focuses on two types:

- **`f128`** — a double-double type built from two `double` limbs
- **`f256`** — a quad-double type built from four `double` limbs

## Highlights
- Modern header-only C++ design (literals, operators, conversions, etc)
- Precision and performance tested against `boost::multiprecision::mpfr_float_backend<digits>` (see results below)
- Complete `constexpr` support (arithmetic, math functions, parsing/serializing, classification, etc)
- bitwise parity between constexpr and runtime results (not guaranteed, but high-confidence based on internal testing)

## Use case

The library is aimed at workloads where *both* speed and precision matters: fractals, simulations, geometric transforms, numerically sensitive kernels, reference-quality math experiments, and other precision-heavy systems where `double` is not enough.

`fltx` is a good fit when you need one or more of the following:

- Cross-platform uniformity
- An extended-precision type you can pass around in ordinary C++ code without redesigning your whole project
- A collection of constexpr-capable math functions that perform quickly and accurately
- Overall better performance than an arbitrary-precision library when targeting the same level of precision

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

    std::cout << std::setprecision( std::numeric_limits<f256>::digits10 )
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "a + b = " << c << "\n";
}

// output:   a = 0.333333333333333333333333333333333333333333333333333333333333333
//           b = 0.666666666666666666666666666666666666666666666666666666666666667
//           a + b = 1
```
## Usage

- full types: `f128` `f256` — user-facing numeric types with implicit converting constructors from supported scalar types
- base types: `f128_s` `f256_s` — trivial standard-layout storage types, suitable for unions and low-level storage. They are typically initialized with braces, and scalar brace initialization works naturally (`f128_s{5.0}`, `f256_s{5.0}`), with remaining limbs zero-initialized


```cpp
f128_s a { 5.0 };  // brace initialization of the trivial storage type
f128 b = 5.0f;     // convenient scalar construction via the full type

// after construction, both are usable in nearly the same way in normal library code
f128   c = a + b;
f128_s d = c;
```

## vcpkg

Add the bitloop-registry to your `vcpkg-configuration.json`:
```
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
Then add `"fltx"` to your `vcpkg.json` dependencies:
```
{
  "name": "myapp",
  "version-string": "1.0",
  "dependencies": [
    "fltx"
  ]
}
```

## CMake

```cmake
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE /path/to/fltx/include)
```

## Content Overview

- **`fltx.h`** — umbrella header
- **`fltx_core.h`** — includes f128.h / f256.h types (and necessary conversions)
- **`f128.h`** — includes `f128` / `f128_s` and associated math functions
- **`f256.h`** — includes `f256` / `f256_s` and associated math functions
- **`numeric_types.h`** — includes type-alias and type-traits, e.g.

Trivial types:

```
bl::f32   // (float typedef)
bl::f64   // (double typedef)
bl::f128
bl::f256
```

Type traits:

```
bl::is_f32_v<T>             // true if T is f32  (float)
bl::is_f64_v<T>             // true if T is f64  (double)
bl::is_f128_v<T>            // true if T is f128 
bl::is_f256_v<T>            // true if T is f256
bl::is_fltx_v<T>            // true if T is f128 / f256
bl::is_floating_point_v<>   // true if T is f32 / f64 / f128 / f256
```

## Design goals

`fltx` is built around a few explicit goals:

### 1. Fixed-width extended precision
The library provides practical scalar types with substantially more precision than `double`, while keeping value representation explicit and bounded.

### 2. Performance-conscious implementation
The implementation favors approaches that remain suitable for heavy arithmetic, iterative kernels, and precision-sensitive pipelines.

### 3. Compile-time and runtime parity
A large part of the library is designed to work both at compile time and at runtime, making it useful for static validation, deterministic tests, and constexpr-heavy code.

### 4. Seamless integration with the STL
`fltx` aims to behave like a natural extension of the standard C++ numeric model.

It includes support for common standard-library integration points such as:

- **`std::ostream`** (with compatibility for standard stream manipulators such as **`std::setprecision`**)
- **`std::numeric_limits`**
- **`std::numbers`**

This keeps `f128` and `f256` easy to use in existing codebases, generic code, and debugging workflows without forcing a separate "special-case" programming style.

## What `fltx` is not

- `fltx` is **not** an arbitrary-precision library.
- A true IEEE binary128 / binary256 floating-point type. f128/f256 use [double-double](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) / [quad-double](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format) implementations. The sizes in f128/f256 represent memory size, i.e.
```
sizeof(f128) == 16
sizeof(f256) == 32
```

If you need unbounded precision, decimal arithmetic semantics, symbolic manipulation, or exact rational arithmetic, a multiprecision package is still the better tool.

`fltx` is about a different trade-off: **fixed-size extended precision with strong performance and practical ergonomics**.

## Precision model

`fltx` uses multi-limb floating-point representations based on `double` components:

- **`f128`** stores two `double` limbs
- **`f256`** stores four `double` limbs

That gives you a large precision increase over native `double`, while preserving a familiar floating-point programming model.

These are still floating-point approximations, not exact values. Precision is improved dramatically, but the normal concerns of numerical computing still apply: conditioning, cancellation, argument reduction, and algorithm design still matter.


### Test results

<img src="res/fltx_f128_vs_mpfr_ratio.png" alt="fltx logo" width="600">
<img src="res/fltx_f256_vs_mpfr_ratio.png" alt="fltx logo" width="600">

## License

This project is licensed under the MIT Licence.

## Summary

`fltx` is a focused C++ library for developers who need **fixed-width extended precision** without jumping all the way to arbitrary precision.

If you want `double`-like ergonomics with substantially more numerical headroom, `f128` and `f256` are the point of the library.
