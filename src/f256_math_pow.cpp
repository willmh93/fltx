#include "fltx/detail/f256/math/pow.h"

namespace bl::detail::_f256_runtime
{
    namespace
    {
        using namespace detail::_f256;

        BL_NO_INLINE f256_s pow_positive_eighth_fraction_runtime(const f256_s& x, int numerator)
        {
            const f256_s r2 = detail::_f256_runtime::sqrt(x);
            if (numerator == 4)
                return r2;

            const f256_s r4 = detail::_f256_runtime::sqrt(r2);
            if (numerator == 2)
                return r4;

            f256_s out{ 1.0 };
            if ((numerator & 4) != 0)
                out = mul_inline(out, r2);
            if ((numerator & 2) != 0)
                out = mul_inline(out, r4);
            if ((numerator & 1) != 0)
            {
                const f256_s r8 = polish_eighth_root(x, detail::_f256_runtime::sqrt(r4));
                if (numerator == 1)
                    return r8;
                out = mul_inline(out, r8);
            }
            return out;
        }

        BL_NO_INLINE f256_s pow_dyadic_eighth_runtime(const f256_s& x, int64_t n)
        {
            if (n == 0)
                return f256_s{ 1.0 };

            const bool neg = n < 0;
            const uint64_t magnitude = neg ? static_cast<uint64_t>(-n) : static_cast<uint64_t>(n);
            const uint64_t whole     = magnitude / 8u;
            const int rem = static_cast<int>(magnitude & 7u);

            f256_s result = (whole == 0u) ? f256_s{ 1.0 } : powi(x, static_cast<int64_t>(whole));
            if (rem != 0)
                result = mul_inline(result, pow_positive_eighth_fraction_runtime(x, rem));
            if (neg)
                result = recip(result);

            return result;
        }

        BL_FORCE_INLINE f256_s exp_for_pow_runtime(const f256_s& x) noexcept
        {
            return detail::_f256_runtime::exp(x);
        }

    } // namespace

    BL_NO_INLINE f256_s pow(const f256_s& x, const f256_s& y)
    {
        using namespace detail::_f256;

        if (iszero(y))
            return f256_s{ 1.0 };

        if (isnan(x) || isnan(y))
            return std::numeric_limits<f256_s>::quiet_NaN();

        const f256_s yi = detail::_f256_runtime::trunc(y);
        const bool y_is_int = (yi == y);

        int64_t yi64{};
        if (y_is_int && try_get_int64(yi, yi64))
            return powi(x, yi64);

        int64_t dyadic_exponent{};
        if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
            return canonicalize_math_result(pow_dyadic_eighth_runtime(x, dyadic_exponent));

        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit_constexpr(x.x0)))
        {
            if (!y_is_int)
                return std::numeric_limits<f256_s>::quiet_NaN();

            const f256_s magnitude = exp_for_pow_runtime(mul_inline(y, detail::_f256_runtime::log(-x)));
            return is_odd_integer(yi) ? -magnitude : magnitude;
        }

        return canonicalize_math_result(exp_for_pow_runtime(mul_inline(y, detail::_f256_runtime::log(x))));
    }

    BL_NO_INLINE f256_s pow(const f256_s& x, double y)
    {
        using namespace detail::_f256;

        if (y == 0.0)
            return f256_s{ 1.0 };

        if (isnan(x) || isnan(y))
            return std::numeric_limits<f256_s>::quiet_NaN();

        if (y == 1.0) return x;
        if (y == 2.0) return canonicalize_math_result(sqr_inline(x));
        if (y == -1.0) return canonicalize_math_result(f256_s{ 1.0 } / x);
        if (y == 0.5) return canonicalize_math_result(detail::_f256_runtime::sqrt(x));

        const double yi = std::trunc(y);
        const bool y_is_int = (yi == y);

        if (y_is_int && absd(yi) < 0x1p63)
            return powi(x, static_cast<int64_t>(yi));

        int64_t dyadic_exponent{};
        if (try_get_pow_dyadic_eighth_exponent(x, y, dyadic_exponent))
            return canonicalize_math_result(pow_dyadic_eighth_runtime(x, dyadic_exponent));

        if (x.x0 < 0.0 || (x.x0 == 0.0 && signbit_constexpr(x.x0)))
        {
            if (!y_is_int)
                return std::numeric_limits<f256_s>::quiet_NaN();

            const f256_s magnitude = exp_for_pow_runtime(mul_double_inline(detail::_f256_runtime::log(-x), y));
            const bool y_is_odd =
                (absd(yi) < 0x1p53) &&
                ((static_cast<int64_t>(yi) & 1ll) != 0);

            return canonicalize_math_result(y_is_odd ? -magnitude : magnitude);
        }

        return canonicalize_math_result(exp_for_pow_runtime(mul_double_inline(detail::_f256_runtime::log(x), y)));
    }

} // namespace bl::detail::_f256_runtime
