/**
 * fltx/random.h - Standard-shaped deterministic random facilities.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_RANDOM_INCLUDED
#define FLTX_RANDOM_INCLUDED
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <ios>
#include <istream>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "fltx/config.h"
#include "fltx/math.h"
#include "fltx/f128_io.h"
#include "fltx/f256_io.h"

namespace bl::detail::random
{
    template<class RealType>
    struct real_traits
    {
        static constexpr bool enabled =
            std::is_same_v<RealType, float> ||
            std::is_same_v<RealType, double> ||
            std::is_same_v<RealType, long double>;

        static constexpr int digits = std::numeric_limits<RealType>::digits;

        [[nodiscard]] BL_FORCE_INLINE static constexpr RealType zero() noexcept
        {
            return static_cast<RealType>(0);
        }

        [[nodiscard]] BL_FORCE_INLINE static constexpr RealType one() noexcept
        {
            return static_cast<RealType>(1);
        }

        template<class UInt>
        [[nodiscard]] BL_FORCE_INLINE static constexpr RealType from_uint(UInt value) noexcept
        {
            return static_cast<RealType>(value);
        }
    };
}

#include "fltx/detail/f128_random.h"
#include "fltx/detail/f256_random.h"

namespace bl
{
    template<class G>
    concept uniform_random_bit_generator =
        std::invocable<G&> &&
        std::unsigned_integral<typename G::result_type> &&
        std::same_as<std::invoke_result_t<G&>, typename G::result_type> &&
        requires
        {
            { G::min() } -> std::same_as<typename G::result_type>;
            { G::max() } -> std::same_as<typename G::result_type>;
            requires G::min() < G::max();
        };

namespace detail::random
{
    template<class UIntType>
    inline constexpr int uint_digits_v = std::numeric_limits<UIntType>::digits;

    template<class UIntType>
    [[nodiscard]] BL_FORCE_INLINE constexpr UIntType word_mask(int bits) noexcept
    {
        if (bits >= uint_digits_v<UIntType>)
            return std::numeric_limits<UIntType>::max();
        return (UIntType{ 1 } << bits) - UIntType{ 1 };
    }

    template<class UIntType>
    [[nodiscard]] BL_FORCE_INLINE constexpr int bit_width(UIntType value) noexcept
    {
        int width = 0;
        while (value != 0)
        {
            value >>= 1;
            ++width;
        }
        return width;
    }

    template<class UIntType>
    [[nodiscard]] BL_FORCE_INLINE constexpr bool is_all_low_bits_set(UIntType value) noexcept
    {
        return value == std::numeric_limits<UIntType>::max() ||
               ((value & (value + UIntType{ 1 })) == UIntType{ 0 });
    }

    template<class UIntType>
    [[nodiscard]] BL_FORCE_INLINE constexpr int conservative_range_bits(UIntType inclusive_range) noexcept
    {
        const int width = bit_width(inclusive_range);
        if (width <= 1)
            return 1;
        return is_all_low_bits_set(inclusive_range) ? width : width - 1;
    }

    template<class IntType>
    concept supported_integer =
        std::is_integral_v<IntType> &&
        !std::is_same_v<std::remove_cv_t<IntType>, bool>;

    template<class RealType>
    concept supported_real =
        real_traits<RealType>::enabled &&
        !std::numeric_limits<RealType>::is_integer;

    template<class RealType, class UInt>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_from_uint(UInt value) noexcept
    {
        return real_traits<RealType>::from_uint(value);
    }

    template<class RealType>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_zero() noexcept
    {
        return real_traits<RealType>::zero();
    }

    template<class RealType>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_one() noexcept
    {
        return real_traits<RealType>::one();
    }

    template<class RealType>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_log(const RealType& value)
    {
        if constexpr (std::is_same_v<RealType, long double>)
            return std::log(value);
        else
            return bl::log(value);
    }

    template<class RealType>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_sqrt(const RealType& value)
    {
        if constexpr (std::is_same_v<RealType, long double>)
            return std::sqrt(value);
        else
            return bl::sqrt(value);
    }

    template<class RealType>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType real_exp(const RealType& value)
    {
        if constexpr (std::is_same_v<RealType, long double>)
            return std::exp(value);
        else
            return bl::exp(value);
    }

    template<class UInt, uniform_random_bit_generator URBG>
    [[nodiscard]] BL_FORCE_INLINE constexpr UInt uniform_bits(URBG& g, int bit_count) noexcept
    {
        static_assert(std::unsigned_integral<UInt>);

        using generator_result = typename URBG::result_type;
        constexpr generator_result inclusive_range = URBG::max() - URBG::min();
        constexpr int max_chunk_bits = conservative_range_bits(inclusive_range);
        static_assert(max_chunk_bits > 0);

        UInt value = 0;
        int filled_bits = 0;
        while (filled_bits < bit_count)
        {
            const int take_bits = (bit_count - filled_bits) < max_chunk_bits
                ? (bit_count - filled_bits)
                : max_chunk_bits;
            const generator_result mask = word_mask<generator_result>(take_bits);

            generator_result sample = 0;
            if constexpr (is_all_low_bits_set(inclusive_range))
            {
                sample = static_cast<generator_result>(g() - URBG::min());
                sample &= mask;
            }
            else
            {
                const generator_result bucket_size = static_cast<generator_result>(mask + generator_result{ 1 });
                const generator_result bucket_count = static_cast<generator_result>(
                    (inclusive_range + generator_result{ 1 }) / bucket_size);
                const generator_result limit = static_cast<generator_result>(bucket_count * bucket_size - 1);
                do
                {
                    sample = static_cast<generator_result>(g() - URBG::min());
                }
                while (sample > limit);
                sample &= mask;
            }

            value = static_cast<UInt>(value | (static_cast<UInt>(sample) << filled_bits));
            filled_bits += take_bits;
        }

        return value;
    }

    template<class UInt, uniform_random_bit_generator URBG>
    [[nodiscard]] BL_FORCE_INLINE constexpr UInt uniform_unsigned(URBG& g, UInt inclusive_max) noexcept
    {
        static_assert(std::unsigned_integral<UInt>);

        if (inclusive_max == 0)
            return 0;

        const int bits = bit_width(inclusive_max);
        UInt value = 0;
        do
        {
            value = uniform_bits<UInt>(g, bits);
        }
        while (value > inclusive_max);

        return value;
    }

    template<class IntType, class UInt>
    [[nodiscard]] BL_FORCE_INLINE constexpr IntType add_unsigned_offset(IntType lower, UInt offset) noexcept
    {
        if constexpr (std::unsigned_integral<IntType>)
        {
            return static_cast<IntType>(lower + static_cast<IntType>(offset));
        }
        else
        {
            if (lower >= IntType{ 0 })
                return static_cast<IntType>(lower + static_cast<IntType>(offset));

            const UInt negative_count = static_cast<UInt>(-(lower + IntType{ 1 })) + UInt{ 1 };
            if (offset < negative_count)
                return static_cast<IntType>(lower + static_cast<IntType>(offset));

            return static_cast<IntType>(offset - negative_count);
        }
    }

    template<class Value>
    BL_FORCE_INLINE std::istream& read_value(std::istream& is, Value& value)
    {
        return is >> value;
    }

    BL_FORCE_INLINE std::istream& read_value(std::istream& is, f128_s& value)
    {
        double hi = 0.0;
        double lo = 0.0;
        if (is >> hi >> lo)
            value = f128_s{ hi, lo };
        return is;
    }

    BL_FORCE_INLINE std::istream& read_value(std::istream& is, f128& value)
    {
        f128_s parsed{};
        read_value(is, parsed);
        if (is)
            value = parsed;
        return is;
    }

    BL_FORCE_INLINE std::istream& read_value(std::istream& is, f256_s& value)
    {
        double x0 = 0.0;
        double x1 = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;
        if (is >> x0 >> x1 >> x2 >> x3)
            value = f256_s{ x0, x1, x2, x3 };
        return is;
    }

    BL_FORCE_INLINE std::istream& read_value(std::istream& is, f256& value)
    {
        f256_s parsed{};
        read_value(is, parsed);
        if (is)
            value = parsed;
        return is;
    }

    class ostream_precision_guard
    {
    public:
        explicit ostream_precision_guard(std::ostream& stream) noexcept
            : os(stream), precision(stream.precision())
        {
        }

        ~ostream_precision_guard()
        {
            os.precision(precision);
        }

        ostream_precision_guard(const ostream_precision_guard&) = delete;
        ostream_precision_guard& operator=(const ostream_precision_guard&) = delete;

    private:
        std::ostream& os;
        std::streamsize precision;
    };

    template<class Value>
    BL_FORCE_INLINE std::ostream& write_value(std::ostream& os, const Value& value)
    {
        ostream_precision_guard guard{ os };
        os.precision(std::numeric_limits<Value>::max_digits10);
        return os << value;
    }

    BL_FORCE_INLINE std::ostream& write_value(std::ostream& os, const f128_s& value)
    {
        ostream_precision_guard guard{ os };
        os.precision(std::numeric_limits<double>::max_digits10);
        return os << value.hi << ' ' << value.lo;
    }

    BL_FORCE_INLINE std::ostream& write_value(std::ostream& os, const f128& value)
    {
        return write_value(os, static_cast<const f128_s&>(value));
    }

    BL_FORCE_INLINE std::ostream& write_value(std::ostream& os, const f256_s& value)
    {
        ostream_precision_guard guard{ os };
        os.precision(std::numeric_limits<double>::max_digits10);
        return os << value.x0 << ' ' << value.x1 << ' ' << value.x2 << ' ' << value.x3;
    }

    BL_FORCE_INLINE std::ostream& write_value(std::ostream& os, const f256& value)
    {
        return write_value(os, static_cast<const f256_s&>(value));
    }
}

    class seed_seq
    {
    public:
        using result_type = std::uint_least32_t;

        seed_seq() = default;

        template<class InputIt>
        BL_FORCE_INLINE constexpr seed_seq(InputIt first, InputIt last)
        {
            for (; first != last; ++first)
                seeds.push_back(static_cast<std::uint32_t>(*first));
        }

        BL_FORCE_INLINE constexpr seed_seq(std::initializer_list<result_type> values)
            : seed_seq(values.begin(), values.end())
        {
        }

        seed_seq(const seed_seq&) = delete;
        seed_seq& operator=(const seed_seq&) = delete;

        template<class RandomAccessIt>
        BL_FORCE_INLINE constexpr void generate(RandomAccessIt first, RandomAccessIt last) const
        {
            const auto n_signed = last - first;
            if (n_signed <= 0)
                return;

            const auto n = static_cast<std::size_t>(n_signed);
            for (std::size_t index = 0; index < n; ++index)
                first[static_cast<typename std::iterator_traits<RandomAccessIt>::difference_type>(index)] =
                    static_cast<typename std::iterator_traits<RandomAccessIt>::value_type>(0x8b8b8b8bu);

            const std::size_t s = seeds.size();
            const std::size_t t =
                n >= 623 ? std::size_t{ 11 } :
                n >= 68 ? std::size_t{ 7 } :
                n >= 39 ? std::size_t{ 5 } :
                n >= 7 ? std::size_t{ 3 } :
                (n - 1) / 2;
            const std::size_t p = (n - t) / 2;
            const std::size_t q = p + t;
            const std::size_t m = std::max(s + 1, n);

            auto at = [&](std::size_t index) constexpr -> std::uint32_t
            {
                return static_cast<std::uint32_t>(
                    first[static_cast<typename std::iterator_traits<RandomAccessIt>::difference_type>(index)]);
            };
            auto set = [&](std::size_t index, std::uint32_t value) constexpr
            {
                first[static_cast<typename std::iterator_traits<RandomAccessIt>::difference_type>(index)] =
                    static_cast<typename std::iterator_traits<RandomAccessIt>::value_type>(value);
            };
            auto add = [&](std::size_t index, std::uint32_t value) constexpr
            {
                set(index, static_cast<std::uint32_t>(at(index) + value));
            };
            auto mix = [](std::uint32_t value) constexpr -> std::uint32_t
            {
                return static_cast<std::uint32_t>(value ^ (value >> 27));
            };

            for (std::size_t k = 0; k < m; ++k)
            {
                const std::size_t k_mod = k % n;
                const std::size_t p_mod = (k + p) % n;
                const std::size_t q_mod = (k + q) % n;
                const std::size_t previous_mod = (k + n - 1) % n;
                const std::uint32_t r1 = static_cast<std::uint32_t>(
                    1664525u * mix(static_cast<std::uint32_t>(at(k_mod) ^ at(p_mod) ^ at(previous_mod))));
                const std::uint32_t r2 = static_cast<std::uint32_t>(
                    r1 + static_cast<std::uint32_t>(k == 0 ? s : k <= s ? k_mod + seeds[k - 1] : k_mod));

                add(p_mod, r1);
                add(q_mod, r2);
                set(k_mod, r2);
            }

            for (std::size_t k = m; k < m + n; ++k)
            {
                const std::size_t k_mod = k % n;
                const std::size_t p_mod = (k + p) % n;
                const std::size_t q_mod = (k + q) % n;
                const std::size_t previous_mod = (k + n - 1) % n;
                const std::uint32_t r3 = static_cast<std::uint32_t>(
                    1566083941u * mix(static_cast<std::uint32_t>(at(k_mod) + at(p_mod) + at(previous_mod))));
                const std::uint32_t r4 = static_cast<std::uint32_t>(r3 - static_cast<std::uint32_t>(k_mod));

                set(p_mod, static_cast<std::uint32_t>(at(p_mod) ^ r3));
                set(q_mod, static_cast<std::uint32_t>(at(q_mod) ^ r4));
                set(k_mod, r4);
            }
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr std::size_t size() const noexcept
        {
            return seeds.size();
        }

        template<class OutputIt>
        BL_FORCE_INLINE constexpr void param(OutputIt dest) const
        {
            for (std::uint32_t seed : seeds)
                *dest++ = static_cast<result_type>(seed);
        }

    private:
        std::vector<std::uint32_t> seeds;
    };

    template<
        class UIntType,
        std::size_t w,
        std::size_t n,
        std::size_t m,
        std::size_t r,
        UIntType a,
        std::size_t u,
        UIntType d,
        std::size_t s,
        UIntType b,
        std::size_t t,
        UIntType c,
        std::size_t l,
        UIntType f>
    class mersenne_twister_engine
    {
        static_assert(std::unsigned_integral<UIntType>, "UIntType must be an unsigned integer type.");
        static_assert(1 <= m && m <= n, "mersenne_twister_engine requires 1 <= m <= n.");
        static_assert(2 <= w && w <= static_cast<std::size_t>(std::numeric_limits<UIntType>::digits),
            "word size must fit in UIntType.");
        static_assert(r <= w, "mask_bits must not exceed word_size.");
        static_assert(u <= w && s <= w && t <= w && l <= w, "tempering shifts must not exceed word_size.");

    public:
        using result_type = UIntType;

        static constexpr std::size_t word_size = w;
        static constexpr std::size_t state_size = n;
        static constexpr std::size_t shift_size = m;
        static constexpr std::size_t mask_bits = r;
        static constexpr result_type xor_mask = a;
        static constexpr std::size_t tempering_u = u;
        static constexpr result_type tempering_d = d;
        static constexpr std::size_t tempering_s = s;
        static constexpr result_type tempering_b = b;
        static constexpr std::size_t tempering_t = t;
        static constexpr result_type tempering_c = c;
        static constexpr std::size_t tempering_l = l;
        static constexpr result_type initialization_multiplier = f;
        static constexpr result_type default_seed = 5489u;

        [[nodiscard]] BL_FORCE_INLINE static constexpr result_type min() noexcept { return 0; }
        [[nodiscard]] BL_FORCE_INLINE static constexpr result_type max() noexcept { return word_mask_value; }

        BL_FORCE_INLINE constexpr mersenne_twister_engine() noexcept
            : mersenne_twister_engine(default_seed)
        {
        }

        BL_FORCE_INLINE constexpr explicit mersenne_twister_engine(result_type value) noexcept
        {
            seed(value);
        }

        template<class Sseq>
            requires requires(Sseq& sequence, std::uint_least32_t* first, std::uint_least32_t* last)
            {
                sequence.generate(first, last);
            }
        BL_FORCE_INLINE constexpr explicit mersenne_twister_engine(Sseq& sequence)
        {
            seed(sequence);
        }

        BL_FORCE_INLINE constexpr void seed(result_type value = default_seed) noexcept
        {
            state[0] = value & word_mask_value;
            for (std::size_t state_index = 1; state_index < state_size; ++state_index)
            {
                const result_type previous = state[state_index - 1];
                state[state_index] =
                    (initialization_multiplier * (previous ^ (previous >> (word_size - 2))) +
                     static_cast<result_type>(state_index)) & word_mask_value;
            }
            index = state_size;
        }

        template<class Sseq>
            requires requires(Sseq& sequence, std::uint_least32_t* first, std::uint_least32_t* last)
            {
                sequence.generate(first, last);
            }
        BL_FORCE_INLINE constexpr void seed(Sseq& sequence)
        {
            constexpr std::size_t word_count = (word_size + 31) / 32;
            std::array<std::uint_least32_t, state_size * word_count> seed_words{};
            sequence.generate(seed_words.begin(), seed_words.end());

            for (std::size_t state_index = 0; state_index < state_size; ++state_index)
            {
                result_type value = 0;
                for (std::size_t word_index = 0; word_index < word_count; ++word_index)
                {
                    const std::size_t shift = 32 * word_index;
                    value |= static_cast<result_type>(
                        static_cast<std::uint32_t>(seed_words[state_index * word_count + word_index])) << shift;
                }
                state[state_index] = value & word_mask_value;
            }

            bool all_zero = (state[0] & upper_mask) == 0;
            for (std::size_t state_index = 1; state_index < state_size; ++state_index)
                all_zero = all_zero && state[state_index] == 0;

            if (all_zero)
                state[0] = word_mask_value;

            index = state_size;
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()() noexcept
        {
            if (index >= state_size)
                twist();

            result_type value = state[index++];
            value ^= ((value >> tempering_u) & tempering_d);
            value ^= ((value << tempering_s) & tempering_b);
            value ^= ((value << tempering_t) & tempering_c);
            value ^= (value >> tempering_l);
            return value & word_mask_value;
        }

        BL_FORCE_INLINE constexpr void discard(unsigned long long z) noexcept
        {
            while (z-- != 0)
                static_cast<void>((*this)());
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const mersenne_twister_engine& lhs,
            const mersenne_twister_engine& rhs) noexcept
        {
            return lhs.index == rhs.index && lhs.state == rhs.state;
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const mersenne_twister_engine& lhs,
            const mersenne_twister_engine& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const mersenne_twister_engine& engine)
        {
            os << engine.index;
            for (result_type value : engine.state)
                os << ' ' << value;
            return os;
        }

        friend std::istream& operator>>(std::istream& is, mersenne_twister_engine& engine)
        {
            std::size_t new_index = 0;
            std::array<result_type, state_size> new_state{};

            if (!(is >> new_index))
                return is;
            if (new_index > state_size)
            {
                is.setstate(std::ios_base::failbit);
                return is;
            }

            for (result_type& value : new_state)
            {
                if (!(is >> value))
                    return is;
            }

            engine.index = new_index;
            engine.state = new_state;
            return is;
        }

    private:
        static constexpr result_type word_mask_value =
            detail::random::word_mask<result_type>(static_cast<int>(word_size));
        static constexpr result_type lower_mask = detail::random::word_mask<result_type>(static_cast<int>(mask_bits));
        static constexpr result_type upper_mask = word_mask_value & ~lower_mask;

        std::array<result_type, state_size> state{};
        std::size_t index = state_size;

        BL_FORCE_INLINE constexpr void twist() noexcept
        {
            for (std::size_t state_index = 0; state_index < state_size; ++state_index)
            {
                const result_type y =
                    (state[state_index] & upper_mask) |
                    (state[(state_index + 1) % state_size] & lower_mask);
                const result_type y_a = (y >> 1) ^ ((y & result_type{ 1 }) != 0 ? xor_mask : result_type{ 0 });
                state[state_index] = (state[(state_index + shift_size) % state_size] ^ y_a) & word_mask_value;
            }
            index = 0;
        }
    };

    using mt19937 = mersenne_twister_engine<
        std::uint_fast32_t,
        32, 624, 397, 31,
        0x9908b0dfu,
        11, 0xffffffffu,
        7, 0x9d2c5680u,
        15, 0xefc60000u,
        18, 1812433253u>;

    using mt19937_64 = mersenne_twister_engine<
        std::uint_fast64_t,
        64, 312, 156, 31,
        0xb5026f5aa96619e9ull,
        29, 0x5555555555555555ull,
        17, 0x71d67fffeda60000ull,
        37, 0xfff7eee000000000ull,
        43, 6364136223846793005ull>;

    using default_random_engine = mt19937;

    template<detail::random::supported_integer IntType = int>
    class uniform_int_distribution
    {
    public:
        using result_type = IntType;

        class param_type
        {
        public:
            using distribution_type = uniform_int_distribution;

            BL_FORCE_INLINE constexpr param_type() noexcept
                : param_type(0)
            {
            }

            BL_FORCE_INLINE constexpr explicit param_type(
                result_type a,
                result_type b = std::numeric_limits<result_type>::max()) noexcept
                : a_value(a), b_value(b)
            {
            }

            [[nodiscard]] BL_FORCE_INLINE constexpr result_type a() const noexcept { return a_value; }
            [[nodiscard]] BL_FORCE_INLINE constexpr result_type b() const noexcept { return b_value; }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return lhs.a_value == rhs.a_value && lhs.b_value == rhs.b_value;
            }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            result_type a_value;
            result_type b_value;
        };

        BL_FORCE_INLINE constexpr uniform_int_distribution() noexcept
            : uniform_int_distribution(0)
        {
        }

        BL_FORCE_INLINE constexpr explicit uniform_int_distribution(
            result_type a,
            result_type b = std::numeric_limits<result_type>::max()) noexcept
            : params(a, b)
        {
        }

        BL_FORCE_INLINE constexpr explicit uniform_int_distribution(const param_type& _params) noexcept
            : params(_params)
        {
        }

        BL_FORCE_INLINE constexpr void reset() noexcept {}

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g) noexcept
        {
            return (*this)(g, params);
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g, const param_type& _params) noexcept
        {
            using unsigned_type = std::make_unsigned_t<result_type>;
            const auto lower = static_cast<unsigned_type>(_params.a());
            const auto upper = static_cast<unsigned_type>(_params.b());
            const auto span = static_cast<unsigned_type>(upper - lower);
            const unsigned_type offset = detail::random::uniform_unsigned<unsigned_type>(g, span);
            return detail::random::add_unsigned_offset<result_type>(_params.a(), offset);
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type a() const noexcept { return params.a(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type b() const noexcept { return params.b(); }

        [[nodiscard]] BL_FORCE_INLINE constexpr param_type param() const noexcept { return params; }
        BL_FORCE_INLINE constexpr void param(const param_type& _params) noexcept { params = _params; }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type min() const noexcept { return params.a(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type max() const noexcept { return params.b(); }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const uniform_int_distribution& lhs,
            const uniform_int_distribution& rhs) noexcept
        {
            return lhs.params == rhs.params;
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const uniform_int_distribution& lhs,
            const uniform_int_distribution& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const uniform_int_distribution& distribution)
        {
            return os << distribution.params.a() << ' ' << distribution.params.b();
        }

        friend std::istream& operator>>(std::istream& is, uniform_int_distribution& distribution)
        {
            result_type a{};
            result_type b{};
            if (is >> a >> b)
                distribution.params = param_type{ a, b };
            return is;
        }

    private:
        param_type params;
    };

    template<detail::random::supported_real RealType, std::size_t bits, uniform_random_bit_generator URBG>
    [[nodiscard]] BL_FORCE_INLINE constexpr RealType generate_canonical(URBG& g) noexcept
    {
        using traits = detail::random::real_traits<RealType>;
        using result_type = typename URBG::result_type;

        constexpr std::size_t requested_bits = bits < static_cast<std::size_t>(traits::digits)
            ? bits
            : static_cast<std::size_t>(traits::digits);
        constexpr result_type inclusive_range = URBG::max() - URBG::min();
        constexpr int bits_per_call = detail::random::conservative_range_bits(inclusive_range);
        constexpr std::size_t call_count =
            requested_bits == 0 ? std::size_t{ 1 } :
            (requested_bits + static_cast<std::size_t>(bits_per_call) - 1) /
                static_cast<std::size_t>(bits_per_call);

        const RealType range =
            detail::random::real_from_uint<RealType>(URBG::max()) -
            detail::random::real_from_uint<RealType>(URBG::min()) +
            detail::random::real_one<RealType>();

        for (;;)
        {
            RealType sum = detail::random::real_zero<RealType>();
            RealType factor = detail::random::real_one<RealType>();

            for (std::size_t index = 0; index < call_count; ++index)
            {
                const result_type sample = static_cast<result_type>(g() - URBG::min());
                sum = sum + detail::random::real_from_uint<RealType>(sample) * factor;
                factor = factor * range;
            }

            const RealType value = sum / factor;
            if (value < detail::random::real_one<RealType>())
                return value;
        }
    }

    template<detail::random::supported_real RealType = double>
    class uniform_real_distribution
    {
    public:
        using result_type = RealType;

        class param_type
        {
        public:
            using distribution_type = uniform_real_distribution;

            BL_FORCE_INLINE constexpr param_type() noexcept
                : param_type(result_type{ 0.0 }, result_type{ 1.0 })
            {
            }

            BL_FORCE_INLINE constexpr explicit param_type(result_type a, result_type b = result_type{ 1.0 }) noexcept
                : a_value(a), b_value(b)
            {
            }

            [[nodiscard]] BL_FORCE_INLINE constexpr result_type a() const noexcept { return a_value; }
            [[nodiscard]] BL_FORCE_INLINE constexpr result_type b() const noexcept { return b_value; }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return lhs.a_value == rhs.a_value && lhs.b_value == rhs.b_value;
            }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            result_type a_value;
            result_type b_value;
        };

        BL_FORCE_INLINE constexpr uniform_real_distribution() noexcept
            : uniform_real_distribution(result_type{ 0.0 })
        {
        }

        BL_FORCE_INLINE constexpr explicit uniform_real_distribution(
            result_type a,
            result_type b = result_type{ 1.0 }) noexcept
            : params(a, b)
        {
        }

        BL_FORCE_INLINE constexpr explicit uniform_real_distribution(const param_type& _params) noexcept
            : params(_params)
        {
        }

        BL_FORCE_INLINE constexpr void reset() noexcept {}

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g) noexcept
        {
            return (*this)(g, params);
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g, const param_type& _params) noexcept
        {
            const result_type u = bl::generate_canonical<result_type, std::numeric_limits<result_type>::digits>(g);
            return _params.a() + (_params.b() - _params.a()) * u;
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type a() const noexcept { return params.a(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type b() const noexcept { return params.b(); }

        [[nodiscard]] BL_FORCE_INLINE constexpr param_type param() const noexcept { return params; }
        BL_FORCE_INLINE constexpr void param(const param_type& _params) noexcept { params = _params; }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type min() const noexcept { return params.a(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type max() const noexcept { return params.b(); }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const uniform_real_distribution& lhs,
            const uniform_real_distribution& rhs) noexcept
        {
            return lhs.params == rhs.params;
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const uniform_real_distribution& lhs,
            const uniform_real_distribution& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const uniform_real_distribution& distribution)
        {
            detail::random::write_value(os, distribution.params.a());
            os << ' ';
            detail::random::write_value(os, distribution.params.b());
            return os;
        }

        friend std::istream& operator>>(std::istream& is, uniform_real_distribution& distribution)
        {
            result_type a{};
            result_type b{};
            if (detail::random::read_value(is, a) && detail::random::read_value(is, b))
                distribution.params = param_type{ a, b };
            return is;
        }

    private:
        param_type params;
    };

    template<detail::random::supported_real RealType = double>
    class exponential_distribution
    {
    public:
        using result_type = RealType;

        class param_type
        {
        public:
            using distribution_type = exponential_distribution;

            BL_FORCE_INLINE constexpr explicit param_type(result_type lambda = result_type{ 1.0 }) noexcept
                : lambda_value(lambda)
            {
            }

            [[nodiscard]] BL_FORCE_INLINE constexpr result_type lambda() const noexcept { return lambda_value; }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return lhs.lambda_value == rhs.lambda_value;
            }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            result_type lambda_value;
        };

        BL_FORCE_INLINE constexpr exponential_distribution() noexcept
            : exponential_distribution(result_type{ 1.0 })
        {
        }

        BL_FORCE_INLINE constexpr explicit exponential_distribution(result_type lambda) noexcept
            : params(lambda)
        {
        }

        BL_FORCE_INLINE constexpr explicit exponential_distribution(const param_type& _params) noexcept
            : params(_params)
        {
        }

        BL_FORCE_INLINE constexpr void reset() noexcept {}

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g) noexcept
        {
            return (*this)(g, params);
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g, const param_type& _params) noexcept
        {
            const result_type u =
                bl::generate_canonical<result_type, std::numeric_limits<result_type>::digits>(g);
            const result_type one_minus_u = detail::random::real_one<result_type>() - u;
            return -detail::random::real_log(one_minus_u) / _params.lambda();
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type lambda() const noexcept { return params.lambda(); }

        [[nodiscard]] BL_FORCE_INLINE constexpr param_type param() const noexcept { return params; }
        BL_FORCE_INLINE constexpr void param(const param_type& _params) noexcept { params = _params; }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type min() const noexcept
        {
            return detail::random::real_zero<result_type>();
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type max() const noexcept
        {
            return std::numeric_limits<result_type>::max();
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const exponential_distribution& lhs,
            const exponential_distribution& rhs) noexcept
        {
            return lhs.params == rhs.params;
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const exponential_distribution& lhs,
            const exponential_distribution& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const exponential_distribution& distribution)
        {
            return detail::random::write_value(os, distribution.params.lambda());
        }

        friend std::istream& operator>>(std::istream& is, exponential_distribution& distribution)
        {
            result_type lambda{};
            if (detail::random::read_value(is, lambda))
                distribution.params = param_type{ lambda };
            return is;
        }

    private:
        param_type params;
    };

    template<detail::random::supported_real RealType = double>
    class normal_distribution
    {
    public:
        using result_type = RealType;

        class param_type
        {
        public:
            using distribution_type = normal_distribution;

            BL_FORCE_INLINE constexpr explicit param_type(
                result_type mean = result_type{ 0.0 },
                result_type stddev = result_type{ 1.0 }) noexcept
                : mean_value(mean), stddev_value(stddev)
            {
            }

            [[nodiscard]] BL_FORCE_INLINE constexpr result_type mean() const noexcept { return mean_value; }
            [[nodiscard]] BL_FORCE_INLINE constexpr result_type stddev() const noexcept { return stddev_value; }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return lhs.mean_value == rhs.mean_value && lhs.stddev_value == rhs.stddev_value;
            }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            result_type mean_value;
            result_type stddev_value;
        };

        BL_FORCE_INLINE constexpr normal_distribution() noexcept
            : normal_distribution(result_type{ 0.0 })
        {
        }

        BL_FORCE_INLINE constexpr explicit normal_distribution(
            result_type mean,
            result_type stddev = result_type{ 1.0 }) noexcept
            : params(mean, stddev)
        {
        }

        BL_FORCE_INLINE constexpr explicit normal_distribution(const param_type& _params) noexcept
            : params(_params)
        {
        }

        BL_FORCE_INLINE constexpr void reset() noexcept
        {
            has_saved = false;
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g) noexcept
        {
            return (*this)(g, params);
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g, const param_type& _params) noexcept
        {
            if (has_saved)
            {
                has_saved = false;
                return _params.mean() + _params.stddev() * saved_standard;
            }

            uniform_real_distribution<result_type> unit(
                -detail::random::real_one<result_type>(),
                detail::random::real_one<result_type>());

            result_type x{};
            result_type y{};
            result_type radius_squared{};
            do
            {
                x = unit(g);
                y = unit(g);
                radius_squared = x * x + y * y;
            }
            while (radius_squared <= detail::random::real_zero<result_type>() ||
                   radius_squared >= detail::random::real_one<result_type>());

            const result_type multiplier_argument =
                (result_type{ -2.0 } * detail::random::real_log(radius_squared)) / radius_squared;
            const result_type multiplier = detail::random::real_sqrt<result_type>(multiplier_argument);
            saved_standard = y * multiplier;
            has_saved = true;
            return _params.mean() + _params.stddev() * (x * multiplier);
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type mean() const noexcept { return params.mean(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type stddev() const noexcept { return params.stddev(); }

        [[nodiscard]] BL_FORCE_INLINE constexpr param_type param() const noexcept { return params; }
        BL_FORCE_INLINE constexpr void param(const param_type& _params) noexcept { params = _params; }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type min() const noexcept
        {
            return std::numeric_limits<result_type>::lowest();
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type max() const noexcept
        {
            return std::numeric_limits<result_type>::max();
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const normal_distribution& lhs,
            const normal_distribution& rhs) noexcept
        {
            return lhs.params == rhs.params &&
                   lhs.has_saved == rhs.has_saved &&
                   (!lhs.has_saved || lhs.saved_standard == rhs.saved_standard);
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const normal_distribution& lhs,
            const normal_distribution& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const normal_distribution& distribution)
        {
            detail::random::write_value(os, distribution.params.mean());
            os << ' ';
            detail::random::write_value(os, distribution.params.stddev());
            os << ' ' << distribution.has_saved;
            if (distribution.has_saved)
            {
                os << ' ';
                detail::random::write_value(os, distribution.saved_standard);
            }
            return os;
        }

        friend std::istream& operator>>(std::istream& is, normal_distribution& distribution)
        {
            result_type mean{};
            result_type stddev{};
            bool loaded_has_saved = false;
            result_type loaded_saved{};

            if (!(detail::random::read_value(is, mean) &&
                  detail::random::read_value(is, stddev) &&
                  (is >> loaded_has_saved)))
                return is;

            if (loaded_has_saved && !detail::random::read_value(is, loaded_saved))
                return is;

            distribution.params = param_type{ mean, stddev };
            distribution.has_saved = loaded_has_saved;
            distribution.saved_standard = loaded_saved;
            return is;
        }

    private:
        param_type params;
        bool has_saved = false;
        result_type saved_standard{};
    };

    template<detail::random::supported_real RealType = double>
    class lognormal_distribution
    {
    public:
        using result_type = RealType;

        class param_type
        {
        public:
            using distribution_type = lognormal_distribution;

            BL_FORCE_INLINE constexpr explicit param_type(
                result_type m = result_type{ 0.0 },
                result_type s = result_type{ 1.0 }) noexcept
                : m_value(m), s_value(s)
            {
            }

            [[nodiscard]] BL_FORCE_INLINE constexpr result_type m() const noexcept { return m_value; }
            [[nodiscard]] BL_FORCE_INLINE constexpr result_type s() const noexcept { return s_value; }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return lhs.m_value == rhs.m_value && lhs.s_value == rhs.s_value;
            }

            [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
                const param_type& lhs,
                const param_type& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            result_type m_value;
            result_type s_value;
        };

        BL_FORCE_INLINE constexpr lognormal_distribution() noexcept
            : lognormal_distribution(result_type{ 0.0 })
        {
        }

        BL_FORCE_INLINE constexpr explicit lognormal_distribution(
            result_type m,
            result_type s = result_type{ 1.0 }) noexcept
            : normal(m, s)
        {
        }

        BL_FORCE_INLINE constexpr explicit lognormal_distribution(const param_type& _params) noexcept
            : normal(_params.m(), _params.s())
        {
        }

        BL_FORCE_INLINE constexpr void reset() noexcept
        {
            normal.reset();
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g) noexcept
        {
            return detail::random::real_exp(normal(g));
        }

        template<uniform_random_bit_generator URBG>
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type operator()(URBG& g, const param_type& _params) noexcept
        {
            return detail::random::real_exp(
                normal(g, typename normal_distribution<result_type>::param_type{ _params.m(), _params.s() }));
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type m() const noexcept { return normal.mean(); }
        [[nodiscard]] BL_FORCE_INLINE constexpr result_type s() const noexcept { return normal.stddev(); }

        [[nodiscard]] BL_FORCE_INLINE constexpr param_type param() const noexcept
        {
            return param_type{ normal.mean(), normal.stddev() };
        }

        BL_FORCE_INLINE constexpr void param(const param_type& _params) noexcept
        {
            normal.param(typename normal_distribution<result_type>::param_type{ _params.m(), _params.s() });
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type min() const noexcept
        {
            return detail::random::real_zero<result_type>();
        }

        [[nodiscard]] BL_FORCE_INLINE constexpr result_type max() const noexcept
        {
            return std::numeric_limits<result_type>::max();
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator==(
            const lognormal_distribution& lhs,
            const lognormal_distribution& rhs) noexcept
        {
            return lhs.normal == rhs.normal;
        }

        [[nodiscard]] BL_FORCE_INLINE friend constexpr bool operator!=(
            const lognormal_distribution& lhs,
            const lognormal_distribution& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const lognormal_distribution& distribution)
        {
            return os << distribution.normal;
        }

        friend std::istream& operator>>(std::istream& is, lognormal_distribution& distribution)
        {
            normal_distribution<result_type> loaded;
            if (is >> loaded)
                distribution.normal = loaded;
            return is;
        }

    private:
        normal_distribution<result_type> normal;
    };

    template<class Distribution, class Engine>
    concept random_distribution_for =
        requires(Distribution& distribution, Engine& engine)
        {
            typename std::remove_cvref_t<Distribution>::result_type;
            {
                distribution(engine)
            } -> std::same_as<typename std::remove_cvref_t<Distribution>::result_type>;
        };

    template<std::size_t Count, uniform_random_bit_generator Engine, class Distribution>
        requires random_distribution_for<Distribution, Engine>
    [[nodiscard]] BL_FORCE_INLINE constexpr auto random_array(Engine engine, Distribution distribution)
    {
        using result_type = typename std::remove_cvref_t<Distribution>::result_type;

        std::array<result_type, Count> values{};
        for (auto& value : values)
            value = distribution(engine);

        return values;
    }

    class random_device
    {
    public:
        using result_type = unsigned int;

        [[nodiscard]] BL_FORCE_INLINE static constexpr result_type min() noexcept
        {
            return std::random_device::min();
        }

        [[nodiscard]] BL_FORCE_INLINE static constexpr result_type max() noexcept
        {
            return std::random_device::max();
        }

        random_device() = default;
        explicit random_device(const std::string& token) : device(token) {}

        [[nodiscard]] result_type operator()() { return device(); }
        [[nodiscard]] double entropy() const noexcept { return device.entropy(); }

        random_device(const random_device&) = delete;
        random_device& operator=(const random_device&) = delete;

    private:
        std::random_device device;
    };

}

#endif
