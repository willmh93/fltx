#include "fltx/detail/f256/math_support.h"

namespace bl::detail::_f256_runtime
{
    BL_NO_INLINE f256_s mul_add_horner_step(const f256_s& a, const f256_s& b, const f256_s& c) noexcept
    {
        return detail::_f256::mul_add_horner_step_constexpr(a, b, c);
    }

    BL_NO_INLINE f256_s horner_forward(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f256_s horner_reverse(const f256_s* coeffs, std::size_t count, const f256_s& x) noexcept
    {
        if (count == 0)
            return {};

        f256_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f256::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f256_s* left_coeffs,
        const f256_s* right_coeffs,
        std::size_t count,
        const f256_s& x,
        f256_s& left_out,
        f256_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f256_s{};
            right_out = f256_s{};
            return;
        }

        f256_s left  = left_coeffs[0];
        f256_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            left = detail::_f256::mul_add_inline(left, x, left_coeffs[i]);
            right = detail::_f256::mul_add_inline(right, x, right_coeffs[i]);
        }

        left_out = left;
        right_out = right;
    }

    BL_NO_INLINE f256_s cheb_eval(const f256_s& x, const f256_s* coeffs, std::size_t count, double shift) noexcept
    {
        if (count == 0)
            return {};

        const f256_s t = detail::_f256::sub_inline(
            detail::_f256::mul_double_inline(x, 2.0),
            f256_s{ shift });
        f256_s b1{ 0.0 };
        f256_s b2{ 0.0 };

        for (std::size_t i = count - 1; i >= 1; --i)
        {
            const f256_s b0 = detail::_f256::add_inline(
                detail::_f256::mul_double_sub_inline(detail::_f256::mul_inline(t, b1), 2.0, b2),
                coeffs[i]);
            b2 = b1;
            b1 = b0;
        }

        return detail::_f256::add_inline(detail::_f256::mul_sub_inline(t, b1, b2), coeffs[0]);
    }

    BL_NO_INLINE f256_s log1p_series_reduced(const f256_s& x) noexcept
    {
        const f256_s z = detail::_f256::div_add_double_inline(x, x, 2.0);
        const f256_s z2 = detail::_f256::sqr_inline(z);

        f256_s term = z;
        f256_s sum  = z;

        for (int k = 3; k <= 257; k += 2)
        {
            term = detail::_f256::mul_inline(term, z2);
            const f256_s add = detail::_f256::div_double_inline(term, static_cast<double>(k));
            sum = detail::_f256::add_inline(sum, add);

            const f256_s asum  = abs(sum);
            const f256_s scale = (asum > f256_s{ 1.0 }) ? asum : f256_s{ 1.0 };
            if (abs(add) <= detail::_f256::mul_inline(f256_s::eps(), scale))
                break;
        }

        return detail::_f256::add_inline(sum, sum);
    }

} // namespace bl::detail::_f256_runtime
