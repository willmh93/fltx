<img src="res/logo.webp" alt="fltx logo" width="90">

A modern C++ header-only library for extended-precision floating-point work.

It provides fixed-width high-precision numeric types designed for code that needs **substantially more precision than `double`**, while staying faster and more light-weight than arbitrary-precision libraries.

At its core, **`fltx`** focuses on two trivial types:

- **`f128`** — a double-double type built from two `double` limbs
- **`f256`** — a quad-double type built from four `double` limbs

The library is aimed at workloads where both precision and speed matters: fractals, simulations, geometric transforms, numerically sensitive kernels, reference-quality math experiments, and other precision-heavy systems where `double` is not enough.

The public types and functions are intended to be used from the `bl` (bitloop) namespace.

---

## Highlights
- Header-only
- Eloquent C++ oriented design (literals, operators, conversions, etc)
- Precision tested and performance benchmarked against `boost::multiprecision::mpfr_float_backend<digits>` (results below)
- Full `constexpr` support (arithmetic, math functions, parsing/serializing, classification, etc)
- bitwise parity between constexpr and runtime results (not guaranteed, but high-confidence based on test results)
---

## What `fltx` is for

`fltx` is a good fit when you need one or more of the following:

- Cross-platform uniformity
- An extended-precision type you can pass around in ordinary C++ code without redesigning your whole project
- A collection of constexpr-capable math functions that perform quickly and accurately even on edge-cases

Typical use cases include:

- Fractal rendering and deep zoom exploration
- Scientific and mathematical visualization
- High-precision coordinate transforms
- Geometry and simulation code sensitive to accumulated error
- Testing numeric kernels against higher-precision references
- Custom math libraries and numerical research projects

---

## What `fltx` is not

- `fltx` is **not** an arbitrary-precision library.
- A true IEEE binary128 / binary256 floating-point type. f128/f256 use [double-double](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) / [quad-double](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format) implementations. The sizes in f128/f256 represent memory size, i.e.
```
sizeof(f128) == 16
sizeof(f256) == 32
```

If you need unbounded precision, decimal arithmetic semantics, symbolic manipulation, or exact rational arithmetic, a multiprecision package is still the better tool.

`fltx` is about a different trade-off: **fixed-size extended precision with strong performance and practical ergonomics**.

---

## Usage

**`f128_t`** / **`f256_t`** are optinal wrappers providing implicit converting constructors from scalar types (can be used interchangably with the base types), e.g.

```cpp
f128   a { 5.0 };  // brace initialization of the trivial type
f128_t b = 5.0f;   // convenient scalar construction via wrapper

// after construction, both behave effectively the same in normal use
f128_t c = a + b;
f128   d = c;
```

## Quick example

```cpp
#include <iostream>
#include <iomanip>

#include <fltx/fltx.h>
using namespace bl::literals;

int main()
{
    constexpr bl::f256_t a = 1_qd / 3_qd;
    constexpr bl::f256_t b = 2_qd / 3_qd;
    bl::f256_t c = a + b;

    std::cout << std::setprecision(std::numeric_limits<bl::f256>::digits10)
        << "a = " << a << "\n"
        << "b = " << b << "\n"
        << "a + b = " << c << "\n";

    // output:
    //   a = 0.333333333333333333333333333333333333333333333333333333333333333
    //   b = 0.666666666666666666666666666666666666666666666666666666666666667
    //   a + b = 1
}
```

---

## Integration

Because `fltx` is header-only, the simplest integration path is to add the library's `include/` directory to your target include paths.

### CMake

```cmake
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE /path/to/fltx/include)
```

Then include the header:

```cpp
#include <fltx/fltx.h>
```

---

### Public header overview

- **`fltx.h`** — umbrella header
- **`fltx_core.h`** — includes f128.h / f256.h types and necessary conversions
- **`f128.h`** — includes `f128` / `f128_t` and associated math functions
- **`f256.h`** — includes `f256` / `f256_t` and associated math functions
- **`numeric_types.h`** — type-alias and type-traits header, e.g.

`f32`, `f64`, `f128`, `f256`
`is_f32<>`, `is_f64<>`, `is_f128<>`, `is_f256<>`
`is_fltx_v<>`, `is_floating_point_v<>`, 
`enum FloatType`

---

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

- **`std::ostream`** for text output (with compatibility for standard stream manipulators such as **`std::setprecision`**)
- **`std::numeric_limits`** for numeric traits and limits

This keeps `f128` and `f256` easy to use in existing codebases, generic code, and debugging workflows without forcing a separate “special-case” programming style.

---

## Precision model

`fltx` uses multi-limb floating-point representations based on `double` components:

- **`f128`** stores two `double` limbs
- **`f256`** stores four `double` limbs

That gives you a large precision increase over native `double`, while preserving a familiar floating-point programming model.

These are still floating-point approximations, not exact values. Precision is improved dramatically, but the normal concerns of numerical computing still apply: conditioning, cancellation, argument reduction, and algorithm design still matter.

---

## License

This project is licensed under the MIT Licence.

---

## Summary

`fltx` is a focused C++ library for developers who need **fixed-width extended precision** without jumping all the way to arbitrary precision.

If you want `double`-like ergonomics with substantially more numerical headroom, `f128` and `f256` are the point of the library.
