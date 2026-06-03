#ifndef FLTX_TESTS_METRICS_QDPP_INCLUDED
#define FLTX_TESTS_METRICS_QDPP_INCLUDED

#include <cmath>

#include <qd/dd.h>
#include <qd/qd_real.h>

namespace bl::test::metrics::qdpp
{
    [[nodiscard]] inline dd_real make(double x0, double x1)
    {
        if (x0 == 0.0 && x1 == 0.0)
            return dd_real{ std::copysign(0.0, x0), 0.0 };
        return std::isfinite(x0) ? dd_real::add(x0, x1) : dd_real{ x0, x1 };
    }

    [[nodiscard]] inline qd_real make(double x0, double x1, double x2, double x3)
    {
        if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0)
            return qd_real{ std::copysign(0.0, x0), 0.0, 0.0, 0.0 };

        qd_real value{ x0, x1, x2, x3 };
        if (std::isfinite(x0))
            value.renorm();
        return value;
    }

    [[nodiscard]] inline dd_real cbrt(const dd_real& value)
    {
        return value < 0.0 ? -::nroot(-value, 3) : ::nroot(value, 3);
    }

    [[nodiscard]] inline qd_real cbrt(const qd_real& value)
    {
        return value < 0.0 ? -::nroot(-value, 3) : ::nroot(value, 3);
    }

    [[nodiscard]] inline dd_real trunc(const dd_real& value)
    {
        return ::aint(value);
    }

    [[nodiscard]] inline qd_real trunc(const qd_real& value)
    {
        return ::aint(value);
    }

    [[nodiscard]] inline dd_real nearbyint(const dd_real& value)
    {
        return ::nint(value);
    }

    [[nodiscard]] inline qd_real nearbyint(const qd_real& value)
    {
        return ::nint(value);
    }

    [[nodiscard]] inline dd_real rint(const dd_real& value)
    {
        return ::nint(value);
    }

    [[nodiscard]] inline qd_real rint(const qd_real& value)
    {
        return ::nint(value);
    }

    [[nodiscard]] inline dd_real remainder(const dd_real& x, const dd_real& y)
    {
        return ::drem(x, y);
    }

    [[nodiscard]] inline qd_real remainder(const qd_real& x, const qd_real& y)
    {
        return ::drem(x, y);
    }

    [[nodiscard]] inline dd_real scalbn(const dd_real& value, int exponent)
    {
        return ::ldexp(value, exponent);
    }

    [[nodiscard]] inline qd_real scalbn(const qd_real& value, int exponent)
    {
        return ::ldexp(value, exponent);
    }

    [[nodiscard]] inline dd_real scalbln(const dd_real& value, long exponent)
    {
        return ::ldexp(value, static_cast<int>(exponent));
    }

    [[nodiscard]] inline qd_real scalbln(const qd_real& value, long exponent)
    {
        return ::ldexp(value, static_cast<int>(exponent));
    }
}

#endif
