# fltx

`fltx` is a modern C++ header-only library for extended-precision floating-point work.

It provides fixed-width high-precision numeric types designed for code that needs **substantially more precision than `double`**, while staying much lighter-weight and more predictable than arbitrary-precision arithmetic.

At its core, `fltx` focuses on two types:

- **`f128`** — a double-double style type built from two `double` limbs
- **`f256`** — a quad-double style type built from four `double` limbs

The library is aimed at workloads where precision, determinism, and integration into ordinary C++ code matter: fractals, simulation, geometric transforms, numerically sensitive kernels, reference-quality math experiments, and other precision-heavy systems where `double` is not enough.

The public types and functions are intended to be used from the `bl` namespace.

---

## Requirements

`fltx` is currently being developed as a **modern C++ library**.

If you are adopting it, assume a recent compiler and standard library are expected. If your branch specifically targets **C++23**, it is worth stating that explicitly in the repository once your packaging story is finalized.

---

## Why fltx?

`fltx` sits in the gap between standard floating-point and arbitrary precision:

- **more precision than `double`**, without committing to a big-number runtime model
- **fixed-size value types** with predictable storage and calling conventions
- **header-only integration** for straightforward adoption
- **constexpr-oriented design** so a large amount of logic can run both at compile time and at runtime
- **high-precision arithmetic and math support** for real numerical work, not just toy operators
- **string parsing and formatting support** so values can be tested, serialized, and compared in practical workflows

This makes it useful when you want high precision that still behaves like a normal numeric library rather than a general-purpose arbitrary-precision framework.

---

## Highlights

- Header-only
- Modern C++ design
- Extended-precision scalar types: `f128` and `f256`
- Arithmetic operators for precision-preserving workflows
- Classification helpers such as finite / infinite / NaN / zero checks
- Math-oriented functionality for numerically sensitive code paths
- Parsing and formatting support for exacting test and validation workflows
- Suitable for both runtime and compile-time evaluation in much of the API surface

---

## What `fltx` is for

`fltx` is a good fit when you need one or more of the following:

- deeper numerical stability than `double`
- more headroom for iterative or cancellation-heavy algorithms
- fixed-width types that are easier to reason about than arbitrary precision
- reproducible precision experiments and comparison against reference implementations
- an extended-precision type you can pass around in ordinary C++ code without redesigning your whole project

Typical use cases include:

- fractal rendering and deep zoom exploration
- scientific and mathematical visualization
- high-precision coordinate transforms
- geometry and simulation code sensitive to accumulated error
- testing numeric kernels against higher-precision references
- custom math libraries and numerical research projects

---

## What `fltx` is not

`fltx` is **not** an arbitrary-precision library.

If you need unbounded precision, decimal arithmetic semantics, symbolic manipulation, or exact rational arithmetic, a multiprecision package is still the better tool.

`fltx` is about a different trade-off: **fixed-size extended precision with strong performance and practical ergonomics**.

---

## Quick example

```cpp
#include <fltx/fltx_core.h>

int main()
{
    bl::f128 a{ 1.0 };
    bl::f128 b{ 3.0 };
    bl::f128 c = a / b;

    bl::f256 x{ 1.0 };
    bl::f256 y{ 10.0 };
    bl::f256 z = x / y;

    auto sum = c + bl::f128{ 2.0 };
    auto product = z * bl::f256{ 4.0 };

    (void)sum;
    (void)product;
}
```

---

## Integration

Because `fltx` is header-only, the simplest integration path is to add the library's `include/` directory to your target include paths.

### CMake

```cmake
add_executable(my_app main.cpp)

target_include_directories(my_app
    PRIVATE
        /path/to/fltx/include
)
```

Then include the umbrella header:

```cpp
#include <fltx/fltx_core.h>
```

If your project configures include paths differently, you may choose a different include style, but `fltx_core.h` is intended to act as the main entry point.

---

## Repository layout

```text
fltx/
+- include/
¦  +- fltx/
¦     +- f128.h
¦     +- f256.h
¦     +- fltx_core.h
¦     +- fltx_common.h
¦     +- numeric_types.h
```

### Header overview

- **`fltx_core.h`** — umbrella header for the library
- **`f128.h`** — `f128` extended-precision type and related functionality
- **`f256.h`** — `f256` extended-precision type and related functionality
- **`fltx_common.h`** — shared helpers and common infrastructure
- **`numeric_types.h`** — core type plumbing and numeric support definitions

---

## Design goals

`fltx` is built around a few explicit goals:

### 1. Fixed-width extended precision
The library provides practical scalar types with substantially more precision than `double`, while keeping value representation explicit and bounded.

### 2. Performance-conscious implementation
This is intended for real numeric workloads, not just textbook demonstrations. The implementation favors approaches that remain suitable for heavy arithmetic, iterative kernels, and precision-sensitive pipelines.

### 3. Compile-time and runtime parity
A large part of the library is designed to work both at compile time and at runtime, making it useful for static validation, deterministic tests, and constant-expression-heavy code.

### 4. Strong testing and validation workflows
Extended precision is only useful if you can trust it. Parsing, formatting, classification, and reproducibility matter just as much as operator overloads.

### 5. Practical integration
`fltx` is meant to drop into ordinary C++ projects without dragging the rest of your architecture into a heavyweight numeric stack.

---

## Precision model

`fltx` uses multi-limb floating-point representations based on `double` components:

- **`f128`** stores two `double` limbs
- **`f256`** stores four `double` limbs

That gives you a large precision increase over native `double`, while preserving a familiar floating-point programming model.

These are still floating-point approximations, not exact values. Precision is improved dramatically, but the normal concerns of numerical computing still apply: conditioning, cancellation, argument reduction, and algorithm design still matter.

---

## Status

`fltx` is best thought of as a **serious precision library under active refinement**.

Its direction is toward:

- robust extended-precision arithmetic
- reliable math behavior across runtime and constexpr evaluation
- practical parsing and formatting
- clean packaging and integration into modern C++ projects

If you are evaluating it for production use, test it against your own numerical requirements and tolerance model, especially for the specific math functions and edge cases your workload depends on.

---

## Roadmap direction

Planned or natural next-step areas for a library like `fltx` include:

- cleaner package/distribution workflows
- stronger CMake packaging and install support
- registry/package-manager integration
- expanded documentation and worked examples
- additional benchmarks and validation suites
- continued refinement of compile-time support and performance characteristics

---

## When to choose `f128` vs `f256`

### Choose `f128` when:
- `double` is too weak
- you need a meaningful precision boost with lower overhead
- the algorithm is sensitive, but not extreme
- you want a strong default extended-precision type

### Choose `f256` when:
- you are pushing much deeper into precision-sensitive territory
- your workload involves extreme zoom, long iterative chains, or severe cancellation
- `f128` still loses too much information
- you can justify the extra cost for materially better numerical headroom

A common strategy is to start with `f128`, validate the numerical behavior of the algorithm, and only move to `f256` where the extra precision is genuinely required.

---

## Documentation philosophy

The library is easiest to understand if you think of it as three layers:

1. **value types** — `f128` and `f256`
2. **numeric behavior** — arithmetic, classification, math support
3. **validation tooling** — parsing, formatting, deterministic comparison, and testability

That is the perspective this README is written from: not as a wrapper around one or two types, but as a precision-oriented numeric toolkit.

---

## Contributing

If you want to contribute, the most useful changes are typically:

- correctness fixes for difficult edge cases
- tighter tests for runtime / constexpr consistency
- benchmark-backed performance improvements
- API cleanup that improves clarity without weakening numeric guarantees
- packaging and integration improvements

For numerical changes, include tests and explain the precision or performance trade-off being made.

---

## License

No license is stated in this README yet. Add a `LICENSE` file and update this section when you decide how the project will be distributed.

---

## Summary

`fltx` is a focused C++ library for developers who need **fixed-width extended precision** without jumping all the way to arbitrary precision.

If you want `double`-like ergonomics with substantially more numerical headroom, `f128` and `f256` are the point of the library.
