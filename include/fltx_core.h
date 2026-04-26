#ifndef FLTX_CORE_INCLUDED
#define FLTX_CORE_INCLUDED

#include "f128.h"
#include "f256.h"

namespace bl
{
    using f32 = float;
    using f64 = double;

    // todo: Move to "fltx_conversions.h" and check which headers were actually included?
    BL_FORCE_INLINE constexpr f256_s::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f256_s::operator f128() const noexcept   { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f128_s::operator f256_s() const noexcept { return f256_s{ hi, lo }; }

    BL_FORCE_INLINE constexpr f128::operator f256_s() const noexcept { return f256_s{ hi, lo }; }
    BL_FORCE_INLINE constexpr f128::operator f256() const noexcept   { return f256_s{ hi, lo }; }

    BL_FORCE_INLINE constexpr f256::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    BL_FORCE_INLINE constexpr f256::operator f128() const noexcept   { return f128_s{ x0, x1 }; }

    BL_FORCE_INLINE constexpr f256::f256(f128_s x) noexcept {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
    }

    BL_FORCE_INLINE constexpr f128_s& f128_s::operator=(f256_s x) noexcept {
        hi = x.x0; lo = x.x1;
        return *this;
    }
    BL_FORCE_INLINE constexpr f256_s& f256_s::operator=(f128_s x) noexcept {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
        return *this;
    }
}

#endif