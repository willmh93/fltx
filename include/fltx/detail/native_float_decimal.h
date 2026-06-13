/**
 * fltx/detail/native_float_decimal.h - Shared native f32/f64 decimal conversion helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_NATIVE_FLOAT_DECIMAL_INCLUDED
#define FLTX_DETAIL_NATIVE_FLOAT_DECIMAL_INCLUDED

#include <bit>
#include <cstdint>
#include <limits>

#include "fltx/config.h"
#include "fltx/detail/common_decimal.h"
#include "fltx/detail/common_fp.h"

namespace bl::detail::_native_float_decimal
{
    [[nodiscard]] BL_FORCE_INLINE constexpr std::uint64_t rounded_shr_u64(std::uint64_t value, int bits) noexcept
    {
        if (bits <= 0)
            return value;

        const bool round_bit = bits <= 64 && ((value >> (bits - 1)) & 1u) != 0;
        const bool sticky = bits > 1
            ? (bits < 64 ? (value & ((std::uint64_t{ 1 } << (bits - 1)) - 1u)) != 0 : (value << 1) != 0)
            : false;
        const std::uint64_t shifted = bits >= 64 ? 0 : (value >> bits);
        return shifted + (round_bit && (sticky || (shifted & 1u) != 0) ? 1u : 0u);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double exact_dyadic_to_double(std::uint64_t coeff, int exp2, bool neg) noexcept
    {
        const std::uint64_t sign = neg ? (std::uint64_t{ 1 } << 63) : 0;
        if (coeff == 0)
            return std::bit_cast<double>(sign);

        const int top_bit = detail::fp::bit_length_u64(coeff) - 1;
        int unbiased_exp = exp2 + top_bit;
        if (unbiased_exp > 1023)
            return neg ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();

        if (unbiased_exp < -1022)
        {
            const int scale = exp2 + 1074;
            const std::uint64_t subnormal = scale >= 0
                ? (scale >= 64 ? 0 : (coeff << scale))
                : rounded_shr_u64(coeff, -scale);

            if (subnormal == 0)
                return std::bit_cast<double>(sign);
            if (subnormal >= (std::uint64_t{ 1 } << 52))
                return std::bit_cast<double>(sign | (std::uint64_t{ 1 } << 52));
            return std::bit_cast<double>(sign | subnormal);
        }

        const int shift = top_bit - 52;
        std::uint64_t significand = shift > 0
            ? rounded_shr_u64(coeff, shift)
            : (coeff << -shift);

        if (detail::fp::bit_length_u64(significand) > 53)
        {
            significand >>= 1;
            ++unbiased_exp;
            if (unbiased_exp > 1023)
                return neg ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        }

        const std::uint64_t exp_bits  = static_cast<std::uint64_t>(unbiased_exp + 1023);
        const std::uint64_t frac_bits = significand & ((std::uint64_t{ 1 } << 52) - 1u);
        return std::bit_cast<double>(sign | (exp_bits << 52) | frac_bits);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr float exact_dyadic_to_float(std::uint64_t coeff, int exp2, bool neg) noexcept
    {
        const std::uint32_t sign = neg ? (std::uint32_t{ 1 } << 31) : 0;
        if (coeff == 0)
            return std::bit_cast<float>(sign);

        const int top_bit = detail::fp::bit_length_u64(coeff) - 1;
        int unbiased_exp = exp2 + top_bit;
        if (unbiased_exp > 127)
            return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

        if (unbiased_exp < -126)
        {
            const int scale = exp2 + 149;
            const std::uint64_t subnormal = scale >= 0
                ? (scale >= 64 ? 0 : (coeff << scale))
                : rounded_shr_u64(coeff, -scale);

            if (subnormal == 0)
                return std::bit_cast<float>(sign);
            if (subnormal >= (std::uint64_t{ 1 } << 23))
                return std::bit_cast<float>(sign | (std::uint32_t{ 1 } << 23));
            return std::bit_cast<float>(sign | static_cast<std::uint32_t>(subnormal));
        }

        const int shift = top_bit - 23;
        std::uint64_t significand = shift > 0
            ? rounded_shr_u64(coeff, shift)
            : (coeff << -shift);

        if (detail::fp::bit_length_u64(significand) > 24)
        {
            significand >>= 1;
            ++unbiased_exp;
            if (unbiased_exp > 127)
                return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }

        const std::uint32_t exp_bits  = static_cast<std::uint32_t>(unbiased_exp + 127);
        const std::uint32_t frac_bits = static_cast<std::uint32_t>(significand) & ((std::uint32_t{ 1 } << 23) - 1u);
        return std::bit_cast<float>(sign | (exp_bits << 23) | frac_bits);
    }

    struct f64_decimal_round_traits
    {
        using value_type = double;

        static constexpr int significand_bits = 53;
        static constexpr int max_binary_exponent = 1023;
        static constexpr int min_binary_exponent = -1074;
        static constexpr int limb_count = 1;

        static constexpr bool isfinite(value_type x) noexcept { return detail::fp::isfinite(x); }
        static constexpr bool signbit(value_type x) noexcept { return detail::fp::signbit(x); }
        static constexpr value_type abs(value_type x) noexcept { return detail::fp::fabs(x); }
        static constexpr value_type zero(bool neg) noexcept { return neg ? -0.0 : 0.0; }
        static constexpr double limb(value_type x, int) noexcept { return x; }
        static constexpr value_type infinity(bool neg) noexcept
        {
            return neg ? -std::numeric_limits<value_type>::infinity() : std::numeric_limits<value_type>::infinity();
        }

        static constexpr value_type pack_from_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return exact_dyadic_to_double(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

    struct f32_decimal_round_traits
    {
        using value_type = float;

        static constexpr int significand_bits = 24;
        static constexpr int max_binary_exponent = 127;
        static constexpr int min_binary_exponent = -149;
        static constexpr int limb_count = 1;

        static constexpr bool isfinite(value_type x) noexcept { return detail::fp::isfinite(x); }
        static constexpr bool signbit(value_type x) noexcept { return detail::fp::signbit(x); }
        static constexpr value_type abs(value_type x) noexcept { return detail::fp::fabs(x); }
        static constexpr value_type zero(bool neg) noexcept { return neg ? -0.0f : 0.0f; }
        static constexpr double limb(value_type x, int) noexcept { return static_cast<double>(x); }
        static constexpr value_type infinity(bool neg) noexcept
        {
            return neg ? -std::numeric_limits<value_type>::infinity() : std::numeric_limits<value_type>::infinity();
        }

        static constexpr value_type pack_from_significand(const detail::exact_decimal::biguint& q, int e2, bool neg) noexcept
        {
            return exact_dyadic_to_float(q.get_bits(0, significand_bits), e2 - (significand_bits - 1), neg);
        }
    };

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr typename Traits::value_type exact_decimal_to_native_value(
        const detail::exact_decimal::biguint& coeff,
        int dec_exp,
        bool neg) noexcept
    {
        using detail::exact_decimal::biguint;

        if (coeff.is_zero())
            return Traits::zero(neg);

        biguint numerator = coeff;
        biguint denominator{ 1 };
        int bin_exp = 0;

        if (dec_exp >= 0)
        {
            numerator = detail::exact_decimal::mul_big(coeff, detail::exact_decimal::pow5_big(dec_exp));
            bin_exp = dec_exp;
        }
        else
        {
            denominator = detail::exact_decimal::pow5_big(-dec_exp);
            bin_exp = dec_exp;
        }

        int ratio_exp = detail::exact_decimal::floor_log2_ratio(numerator, denominator);
        biguint q = detail::exact_decimal::extract_rounded_significand_chunks(
            numerator,
            denominator,
            ratio_exp,
            Traits::significand_bits);

        if (q.bit_length() > Traits::significand_bits)
        {
            q.shr1();
            ++ratio_exp;
        }

        const int e2 = bin_exp + ratio_exp;
        if (e2 > Traits::max_binary_exponent)
            return Traits::infinity(neg);
        if (e2 < Traits::min_binary_exponent)
            return Traits::zero(neg);

        return Traits::pack_from_significand(q, e2, neg);
    }
}

#endif
