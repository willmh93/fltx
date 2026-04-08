#pragma once

#include "f128.h"
#include "f256.h"

namespace bl
{
    using std::abs;
    using std::fabs;

    using std::sin;
    using std::cos;
    using std::tan;
    using std::asin;
    using std::acos;
    using std::atan;
    using std::atan2;

    using std::sinh;
    using std::cosh;
    using std::tanh;
    using std::asinh;
    using std::acosh;
    using std::atanh;

    using std::exp;
    using std::exp2;
    using std::expm1;

    using std::log;
    using std::log10;
    using std::log2;
    using std::log1p;

    using std::pow;
    using std::sqrt;
    using std::cbrt;
    using std::hypot;

    using std::ceil;
    using std::floor;
    using std::trunc;
    using std::round;
    using std::nearbyint;
    using std::rint;
    using std::lround;
    using std::llround;
    using std::lrint;
    using std::llrint;

    using std::fmod;
    using std::remainder;
    using std::remquo;

    using std::fma;
    using std::fmin;
    using std::fmax;
    using std::fdim;

    using std::copysign;
    using std::frexp;
    using std::ldexp;
    using std::modf;
    using std::ilogb;
    using std::logb;
    using std::scalbn;
    using std::scalbln;
    using std::nextafter;
    using std::nexttoward;

    using std::erf;
    using std::erfc;
    using std::tgamma;
    using std::lgamma;

    using std::fpclassify;
    using std::isfinite;
    using std::isinf;
    using std::isnan;
    using std::isnormal;
    using std::signbit;

    using std::isgreater;
    using std::isgreaterequal;
    using std::isless;
    using std::islessequal;
    using std::islessgreater;
    using std::isunordered;

    // for consistecy with invoking with template type
    FORCE_INLINE constexpr double log_as_double(float a) { return fltx::common::fp::log_constexpr(a); }
    FORCE_INLINE constexpr double log_as_double(double a) { return fltx::common::fp::log_constexpr(a); }

    // todo: Move to "fltx_conversions.h" and check which headers were actually included?
    FORCE_INLINE constexpr f256_s::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    FORCE_INLINE constexpr f256_s::operator f128() const noexcept   { return f128_s{ x0, x1 }; }
    FORCE_INLINE constexpr f128_s::operator f256_s() const noexcept { return f256_s{ hi, lo }; }

    FORCE_INLINE constexpr f128::operator f256_s() const noexcept { return f256_s{ hi, lo }; }
    FORCE_INLINE constexpr f128::operator f256() const noexcept   { return f256_s{ hi, lo }; }

    FORCE_INLINE constexpr f256::operator f128_s() const noexcept { return f128_s{ x0, x1 }; }
    FORCE_INLINE constexpr f256::operator f128() const noexcept   { return f128_s{ x0, x1 }; }

    FORCE_INLINE constexpr f256::f256(f128_s x) noexcept {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
    }

    FORCE_INLINE constexpr f128_s& f128_s::operator=(f256_s x) noexcept {
        hi = x.x0; lo = x.x1;
        return *this;
    }
    FORCE_INLINE constexpr f256_s& f256_s::operator=(f128_s x) noexcept {
        x0 = x.hi; x1 = x.lo; x2 = 0.0; x3 = 0.0;
        return *this;
    }

    FORCE_INLINE std::string to_string(float a,
        int max_precision = std::numeric_limits<float>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        std::string out;
        _f128_detail::to_string_into(
            out,
            f128_s{ static_cast<double>(a), 0.0 },
            max_precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    FORCE_INLINE std::string to_string(double a,
        int max_precision = std::numeric_limits<double>::digits10,
        bool fixed = false,
        bool scientific = false,
        bool strip_trailing_zeros = false)
    {
        std::string out;
        _f128_detail::to_string_into(
            out,
            f128_s{ a, 0.0 },
            max_precision,
            fixed,
            scientific,
            strip_trailing_zeros
        );
        return out;
    }

    template<typename T>
    std::string to_string_collapsed(T value, int max_precision, int peek_front = 4, int peek_back = 10)
    {
        if (max_precision < 0) max_precision = 0;
        if (peek_front < 0) peek_front = 0;
        if (peek_back < 0) peek_back = 0;

        peek_front = std::min(peek_front, max_precision);
        peek_back = std::min(peek_back, max_precision);

        std::string s = bl::to_string(value, max_precision, true, false);
        size_t period_i = s.find('.');

        if (period_i == std::string::npos)
            return s;

        int front_end = (int)period_i + 1 + peek_front;
        int back_start = (int)period_i + 1 + (max_precision - peek_back);

        // ensure we're collapsing more than 3 characters, otherwise there's little point
        if (back_start <= front_end + 3)
            return s;

        std::string ret;
        ret += s.substr(0, front_end);
        ret += "...";
        ret += s.substr(back_start, peek_back);

        return ret;
    }
}
