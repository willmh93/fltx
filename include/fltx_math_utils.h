#ifndef FLTX_MATH_UTILS
#define FLTX_MATH_UTILS

#include <concepts>
#include <type_traits>

namespace bl::fltx::common::fp
{
    template<class T>
    concept non_bool_integral =
        std::integral<std::remove_cvref_t<T>> &&
        !std::same_as<std::remove_cvref_t<T>, bool>;

    template<class U>
    requires non_bool_integral<U>
    constexpr std::make_unsigned_t<std::remove_cvref_t<U>> unsigned_abs(U value) noexcept
    {
        using S = std::remove_cvref_t<U>;
        using Uu = std::make_unsigned_t<S>;

        if constexpr (std::signed_integral<S>)
        {
            return (value < 0)
                ? static_cast<Uu>(-(value + 1)) + Uu{ 1 }
            : static_cast<Uu>(value);
        }
        else
        {
            return static_cast<Uu>(value);
        }
    }

    template<class R, class ExpUnsigned>
    constexpr R ipow_nonneg(R base, ExpUnsigned exp)
    {
        R result = R{ 1 };

        while (exp != ExpUnsigned{ 0 })
        {
            if ((exp & ExpUnsigned{ 1 }) != ExpUnsigned{ 0 })
                result *= base;

            exp >>= 1;

            if (exp != ExpUnsigned{ 0 })
                base *= base;
        }

        return result;
    }
}

namespace bl
{
    template<class T>
    [[nodiscard]] constexpr T sq(T x) noexcept(noexcept(x* x))
    {
        return x * x;
    }

    template<class T>
    [[nodiscard]] constexpr T clamp(T x, T low, T high) noexcept(noexcept(x < low) && noexcept(high < x))
    {
        return (x < low) ? low : ((high < x) ? high : x);
    }

    template<class T, class Exp>
    requires (!std::integral<std::remove_cvref_t<T>> && fltx::common::fp::non_bool_integral<Exp>)
    [[nodiscard]] constexpr std::remove_cvref_t<T> pow(T base, Exp exp)
    {
        using R = std::remove_cvref_t<T>;
        using U = std::make_unsigned_t<std::remove_cvref_t<Exp>>;

        const U magnitude = fltx::common::fp::unsigned_abs(exp);
        const R powered = fltx::common::fp::ipow_nonneg<R, U>(static_cast<R>(base), magnitude);

        if constexpr (std::signed_integral<std::remove_cvref_t<Exp>>)
        {
            if (exp < 0)
                return R{ 1 } / powered;
        }

        return powered;
    }

    template<class T, class U>
    requires (fltx::common::fp::non_bool_integral<T> && fltx::common::fp::non_bool_integral<U>)
    [[nodiscard]] constexpr std::common_type_t<T, U> pow(T x, U y)
    {
        using R = std::common_type_t<T, U>;
        using Exp = std::remove_cvref_t<U>;
        using ExpUnsigned = std::make_unsigned_t<Exp>;

        const R base = static_cast<R>(x);

        if constexpr (std::signed_integral<Exp>)
        {
            if (y < 0)
            {
                if (base == R{ 1 })
                    return R{ 1 };

                if constexpr (std::signed_integral<R>)
                {
                    if (base == R{ -1 })
                        return (y % 2 != 0) ? R{ -1 } : R{ 1 };
                }

                return R{ 0 };
            }
        }

        const ExpUnsigned exp = fltx::common::fp::unsigned_abs(y);
        return fltx::common::fp::ipow_nonneg<R, ExpUnsigned>(base, exp);
    }
}

#endif