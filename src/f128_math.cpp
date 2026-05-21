#include "fltx/detail/f128/math_support.h"

namespace bl::detail::_f128_runtime
{
    BL_NO_INLINE f128_s horner_forward(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i]);
        return p;
    }

    BL_NO_INLINE f128_s horner_reverse(const f128_s* coeffs, std::size_t count, const f128_s& x) noexcept
    {
        if (count == 0)
            return {};

        f128_s p = coeffs[count - 1];
        for (std::size_t i = count - 1; i > 0; --i)
            p = detail::_f128::mul_add_inline(p, x, coeffs[i - 1]);
        return p;
    }

    BL_NO_INLINE void horner_pair_forward(
        const f128_s* left_coeffs,
        const f128_s* right_coeffs,
        std::size_t count,
        const f128_s& x,
        f128_s& left_out,
        f128_s& right_out) noexcept
    {
        if (count == 0)
        {
            left_out = f128_s{};
            right_out = f128_s{};
            return;
        }

        f128_s left  = left_coeffs[0];
        f128_s right = right_coeffs[0];
        for (std::size_t i = 1; i < count; ++i)
            detail::_f128::mul_add_pair_same_rhs_inline(left, right, x, left_coeffs[i], right_coeffs[i], left, right);

        left_out = left;
        right_out = right;
    }

} // namespace bl::detail::_f128_runtime
