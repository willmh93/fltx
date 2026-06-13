#ifndef FLTX_TESTS_METRICS_IO_INCLUDED
#define FLTX_TESTS_METRICS_IO_INCLUDED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <ios>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <fltx/io.h>

#include "metrics_case_output.h"
#include "metrics_config.h"
#include "metrics_f128_primary.h"
#include "metrics_f256_primary.h"
#include "metrics_reference.h"

namespace bl::test::metrics::io_metrics
{
    inline constexpr domain_id io_domain{ "io", domain_role::primary };
    inline constexpr std::size_t samples_per_kind = config::scale_accuracy_sample_count(10000);
    inline constexpr std::size_t benchmark_min_iterations = config::scale_mixed_iterations(200000);

    template<class Float>
    struct io_profile;

    template<class T>
    [[nodiscard]] constexpr double expansion_storage_bits() noexcept
    {
        using value_type = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<value_type, bl::f128> ||
                      std::is_same_v<value_type, bl::f128_s> ||
                      std::is_same_v<value_type, boost::multiprecision::cpp_double_double> ||
                      std::is_same_v<value_type, dd_real>)
        {
            return 106.0;
        }
        else if constexpr (std::is_same_v<value_type, bl::f256> ||
                           std::is_same_v<value_type, bl::f256_s> ||
                           std::is_same_v<value_type, qd_real>)
        {
            return 212.0;
        }
        else
        {
            return 0.0;
        }
    }

    template<class PerfectRef>
    [[nodiscard]] int floor_log2_abs_ref(const PerfectRef& value)
    {
        const PerfectRef magnitude = value < 0 ? -value : value;
        if (magnitude == 0)
            return 0;

        using std::floor;
        using std::log2;
        return floor(log2(magnitude)).template convert_to<int>();
    }

    template<class Value, class PerfectRef>
    [[nodiscard]] double effective_domain_ideal_bits(const PerfectRef& expected)
    {
        constexpr double storage_bits = expansion_storage_bits<Value>();
        if constexpr (storage_bits <= 0.0)
        {
            return 0.0;
        }
        else
        {
            if (expected == 0)
                return storage_bits;

            constexpr int min_double_bit_exponent =
                std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits;
            const double available_bits =
                static_cast<double>(floor_log2_abs_ref(expected) - min_double_bit_exponent + 1);

            return std::clamp(available_bits, 1.0, storage_bits);
        }
    }

    template<>
    struct io_profile<bl::f128>
    {
        using references = reference_types<bl::f128>;
        using fltx_type = references::fltx_type;
        using perfect_ref = references::perfect_ref;
        using competitor_ref = references::competitor_ref;
        using extra_competitor_ref = references::extra_competitor_ref;

        static constexpr precision_type precision = precision_type::f128;
        static constexpr std::string_view title = "f128 IO metrics results";
        static constexpr double ideal_bits = f128_primary::domain_ideal_bits;

        [[nodiscard]] static perfect_ref to_perfect(const fltx_type& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const competitor_ref& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const extra_competitor_ref& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static double matching_bits(const perfect_ref& actual, const perfect_ref& expected)
        {
            const perfect_ref error = f128_primary::abs_ref(actual - expected);
            if (error == 0)
                return std::numeric_limits<double>::infinity();

            const perfect_ref scale = expected == 0 ? perfect_ref{ 1 } : f128_primary::abs_ref(expected);
            using std::log2;
            return static_cast<double>(-log2(error / scale));
        }
        [[nodiscard]] static double finite_for_mean(double bits) noexcept { return f128_primary::finite_for_mean(bits); }
        template<class Value>
        [[nodiscard]] static double domain_ideal_bits_for(const perfect_ref& expected)
        {
            const double effective_bits = effective_domain_ideal_bits<Value>(expected);
            return effective_bits > 0.0 ? effective_bits : ideal_bits;
        }

        static void consume(const fltx_type& value) { f128_primary::consume_benchmark_value(value); }
        static void consume(const competitor_ref& value) { f128_primary::consume_benchmark_value(value); }
        static void consume(const extra_competitor_ref& value) { f128_primary::consume_benchmark_value(value); }
        static void consume_text(std::string_view text)
        {
            f128_primary::consume_benchmark_limb(static_cast<double>(text.size()));
            if (!text.empty())
                f128_primary::consume_benchmark_limb(static_cast<unsigned char>(text.front()));
        }
    };

    template<>
    struct io_profile<bl::f256>
    {
        using references = reference_types<bl::f256>;
        using fltx_type = references::fltx_type;
        using perfect_ref = references::perfect_ref;
        using competitor_ref = references::competitor_ref;
        using extra_competitor_ref = references::extra_competitor_ref;

        static constexpr precision_type precision = precision_type::f256;
        static constexpr std::string_view title = "f256 IO metrics results";
        static constexpr double ideal_bits = f256_primary::domain_ideal_bits;

        [[nodiscard]] static perfect_ref to_perfect(const fltx_type& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const competitor_ref& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const extra_competitor_ref& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static double matching_bits(const perfect_ref& actual, const perfect_ref& expected)
        {
            const perfect_ref error = f256_primary::abs_ref(actual - expected);
            if (error == 0)
                return std::numeric_limits<double>::infinity();

            const perfect_ref scale = expected == 0 ? perfect_ref{ 1 } : f256_primary::abs_ref(expected);
            using std::log2;
            return static_cast<double>(-log2(error / scale));
        }
        [[nodiscard]] static double finite_for_mean(double bits) noexcept { return f256_primary::finite_for_mean(bits); }
        template<class Value>
        [[nodiscard]] static double domain_ideal_bits_for(const perfect_ref& expected)
        {
            const double effective_bits = effective_domain_ideal_bits<Value>(expected);
            return effective_bits > 0.0 ? effective_bits : ideal_bits;
        }

        static void consume(const fltx_type& value) { f256_primary::consume_benchmark_value(value); }
        static void consume(const competitor_ref& value) { f256_primary::consume_benchmark_value(value); }
        static void consume(const extra_competitor_ref& value) { f256_primary::consume_benchmark_value(value); }
        static void consume_text(std::string_view text)
        {
            f256_primary::consume_benchmark_limb(static_cast<double>(text.size()));
            if (!text.empty())
                f256_primary::consume_benchmark_limb(static_cast<unsigned char>(text.front()));
        }
    };

    template<class Profile>
    struct io_sample
    {
        std::string text;
        typename Profile::perfect_ref oracle;
        int precision = -1;
    };

    template<class Profile>
    struct io_sample_group
    {
        std::string_view label;
        std::vector<io_sample<Profile>> defaultfloat;
        std::vector<io_sample<Profile>> fixed;
        std::vector<io_sample<Profile>> scientific;
        std::vector<io_sample<Profile>> hexfloat;
    };

    struct io_format_case
    {
        std::string_view label;
        std::ios_base::fmtflags flags;
        bool hexfloat = false;
    };

    [[nodiscard]] inline std::string_view intern_operation_name(std::string text)
    {
        static std::deque<std::string> storage;
        storage.push_back(std::move(text));
        return storage.back();
    }

    [[nodiscard]] inline std::array<io_format_case, 4> format_cases() noexcept
    {
        return {
            io_format_case{ "default", std::ios_base::fmtflags{}, false },
            io_format_case{ "fixed", std::ios_base::fixed, false },
            io_format_case{ "scientific", std::ios_base::scientific, false },
            io_format_case{ "hexfloat", std::ios_base::fixed | std::ios_base::scientific, true }
        };
    }

    template<class Profile>
    [[nodiscard]] constexpr std::array<int, 3> precision_cases() noexcept
    {
        constexpr int digits = std::numeric_limits<typename Profile::fltx_type>::digits10;
        if constexpr (digits > 40)
            return { digits, digits / 2, 15 };
        else
            return { digits, digits / 2, 7 };
    }

    class io_text_rng
    {
    public:
        explicit io_text_rng(std::uint64_t seed) noexcept
            : engine(seed)
        {
        }

        [[nodiscard]] int integer(int min, int max)
        {
            std::uniform_int_distribution<int> distribution(min, max);
            return distribution(engine);
        }

        [[nodiscard]] int normal_integer(int min, int max)
        {
            if (min >= max)
                return min;

            const double center = 0.5 * static_cast<double>(min + max);
            const double sigma = std::max(1.0, static_cast<double>(max - min) / 6.0);
            std::normal_distribution<double> distribution(center, sigma);

            for (int attempt = 0; attempt < 16; ++attempt)
            {
                const int value = static_cast<int>(std::lround(distribution(engine)));
                if (value >= min && value <= max)
                    return value;
            }

            return std::clamp(static_cast<int>(std::lround(distribution(engine))), min, max);
        }

        [[nodiscard]] bool negative()
        {
            return integer(0, 1) != 0;
        }

        [[nodiscard]] char digit(bool nonzero = false)
        {
            return static_cast<char>('0' + integer(nonzero ? 1 : 0, 9));
        }

        [[nodiscard]] char hex_digit(bool nonzero = false)
        {
            constexpr std::string_view digits = "0123456789abcdef";
            return digits[static_cast<std::size_t>(integer(nonzero ? 1 : 0, 15))];
        }

        [[nodiscard]] std::string digits(std::size_t count, bool first_nonzero = true)
        {
            std::string out;
            out.reserve(count);
            for (std::size_t index = 0; index < count; ++index)
                out.push_back(digit(first_nonzero && index == 0));
            return out;
        }

        [[nodiscard]] std::string hex_digits(std::size_t count)
        {
            std::string out;
            out.reserve(count);
            for (std::size_t index = 0; index < count; ++index)
                out.push_back(hex_digit());
            return out;
        }

    private:
        std::mt19937_64 engine;
    };

    template<class Profile>
    [[nodiscard]] typename Profile::perfect_ref parse_decimal_oracle(std::string_view text)
    {
        return typename Profile::perfect_ref{ std::string(text) };
    }

    [[nodiscard]] inline int hex_value(char ch)
    {
        if (ch >= '0' && ch <= '9')
            return ch - '0';
        if (ch >= 'a' && ch <= 'f')
            return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F')
            return ch - 'A' + 10;
        return -1;
    }

    template<class Profile>
    [[nodiscard]] typename Profile::perfect_ref parse_hex_oracle(std::string_view text)
    {
        using boost::multiprecision::cpp_int;
        using boost::multiprecision::ldexp;
        using perfect_ref = typename Profile::perfect_ref;

        std::size_t index = 0;
        bool negative = false;
        if (index < text.size() && (text[index] == '+' || text[index] == '-'))
            negative = text[index++] == '-';

        if (index + 2 <= text.size() && text[index] == '0' && (text[index + 1] == 'x' || text[index + 1] == 'X'))
            index += 2;

        cpp_int significand = 0;
        int fractional_hex_digits = 0;
        bool after_point = false;
        bool consumed_digit = false;

        while (index < text.size())
        {
            const char ch = text[index];
            if (ch == '.')
            {
                after_point = true;
                ++index;
                continue;
            }
            if (ch == 'p' || ch == 'P')
                break;

            const int digit = hex_value(ch);
            if (digit < 0)
                throw std::invalid_argument("invalid hexfloat oracle");

            significand <<= 4;
            significand += digit;
            fractional_hex_digits += after_point ? 1 : 0;
            consumed_digit = true;
            ++index;
        }

        if (!consumed_digit || index >= text.size() || (text[index] != 'p' && text[index] != 'P'))
            throw std::invalid_argument("invalid hexfloat oracle");

        ++index;
        bool negative_exponent = false;
        if (index < text.size() && (text[index] == '+' || text[index] == '-'))
            negative_exponent = text[index++] == '-';

        int exponent = 0;
        bool consumed_exponent = false;
        while (index < text.size())
        {
            const char ch = text[index++];
            if (ch < '0' || ch > '9')
                throw std::invalid_argument("invalid hexfloat exponent");
            exponent = exponent * 10 + (ch - '0');
            consumed_exponent = true;
        }

        if (!consumed_exponent)
            throw std::invalid_argument("invalid hexfloat exponent");

        if (negative_exponent)
            exponent = -exponent;

        perfect_ref value{ significand };
        value = ldexp(value, exponent - 4 * fractional_hex_digits);
        return negative ? -value : value;
    }

    template<class Profile>
    [[nodiscard]] typename Profile::perfect_ref parse_oracle(std::string_view text, bool hexfloat)
    {
        return hexfloat
            ? parse_hex_oracle<Profile>(text)
            : parse_decimal_oracle<Profile>(text);
    }

    [[nodiscard]] inline std::string with_sign(io_text_rng& rng, std::string text)
    {
        if (rng.negative())
            text.insert(text.begin(), '-');
        return text;
    }

    [[nodiscard]] inline bool is_brute_label(std::string_view label) noexcept
    {
        return label == "brute-huge" || label == "brute-med" || label == "brute-tiny";
    }

    template<class Profile>
    [[nodiscard]] constexpr int brute_decimal_exponent_limit() noexcept
    {
        return std::numeric_limits<typename Profile::fltx_type>::max_exponent10 - 2;
    }

    template<class Profile>
    [[nodiscard]] constexpr int brute_medium_exponent_limit() noexcept
    {
        constexpr int limit = brute_decimal_exponent_limit<Profile>();
        return std::max(4, (limit + 99) / 100);
    }

    struct signed_exponent_range
    {
        int min_abs;
        int max_abs;
        bool allow_zero = false;
    };

    template<class Profile>
    [[nodiscard]] constexpr signed_exponent_range brute_exponent_range(std::string_view label) noexcept
    {
        constexpr int max_exp = brute_decimal_exponent_limit<Profile>();
        constexpr int med_exp = brute_medium_exponent_limit<Profile>();

        if (label == "brute-tiny")
            return { 0, 3, true };
        if (label == "brute-med")
            return { 4, med_exp, false };
        return { med_exp + 1, max_exp, false };
    }

    template<class Profile>
    [[nodiscard]] int brute_decimal_exponent(std::string_view label, io_text_rng& rng)
    {
        const signed_exponent_range range = brute_exponent_range<Profile>(label);
        const int magnitude = rng.normal_integer(range.min_abs, range.max_abs);
        if (range.allow_zero && magnitude == 0)
            return 0;
        return rng.negative() ? -magnitude : magnitude;
    }

    [[nodiscard]] inline std::string make_decimal_scientific_text(
        io_text_rng& rng,
        int exponent,
        int digit_count)
    {
        std::string text;
        if (rng.negative())
            text.push_back('-');
        text.push_back(rng.digit(true));
        if (digit_count > 1)
        {
            text.push_back('.');
            text += rng.digits(static_cast<std::size_t>(digit_count - 1), false);
        }
        text.push_back('e');
        text.push_back(exponent < 0 ? '-' : '+');
        text += std::to_string(std::abs(exponent));
        return text;
    }

    [[nodiscard]] inline std::string make_decimal_fixed_text(
        io_text_rng& rng,
        int exponent,
        int digit_count)
    {
        std::string digits = rng.digits(static_cast<std::size_t>(digit_count));
        std::string text;

        if (exponent >= 0)
        {
            const int integer_digits = exponent + 1;
            if (integer_digits <= digit_count)
            {
                text = digits.substr(0, static_cast<std::size_t>(integer_digits));
                if (integer_digits < digit_count)
                {
                    text.push_back('.');
                    text += digits.substr(static_cast<std::size_t>(integer_digits));
                }
            }
            else
            {
                text = std::move(digits);
                text.append(static_cast<std::size_t>(integer_digits - digit_count), '0');
            }
        }
        else
        {
            text = "0.";
            text.append(static_cast<std::size_t>(-exponent - 1), '0');
            text += digits;
        }

        return with_sign(rng, std::move(text));
    }

    template<class Profile>
    [[nodiscard]] std::string make_brute_fixed_text(std::string_view label, io_text_rng& rng)
    {
        const int exponent = brute_decimal_exponent<Profile>(label, rng);
        const int digit_count = rng.normal_integer(
            1,
            std::numeric_limits<typename Profile::fltx_type>::digits10 + 8);
        return make_decimal_fixed_text(rng, exponent, digit_count);
    }

    template<class Profile>
    [[nodiscard]] std::string make_brute_defaultfloat_text(std::string_view label, io_text_rng& rng, int precision)
    {
        const int exponent = brute_decimal_exponent<Profile>(label, rng);
        const int digit_count = std::max(1, precision);
        if (exponent < -4 || exponent >= digit_count)
            return make_decimal_scientific_text(rng, exponent, digit_count);
        return make_decimal_fixed_text(rng, exponent, digit_count);
    }

    template<class Profile>
    [[nodiscard]] std::string make_brute_scientific_text(std::string_view label, io_text_rng& rng)
    {
        const int exponent = brute_decimal_exponent<Profile>(label, rng);
        const int digit_count = rng.normal_integer(
            1,
            std::numeric_limits<typename Profile::fltx_type>::digits10 + 8);
        return make_decimal_scientific_text(rng, exponent, digit_count);
    }

    template<class Profile>
    [[nodiscard]] std::string make_brute_hex_text(std::string_view label, io_text_rng& rng)
    {
        constexpr double log2_10 = 3.32192809488736234787;
        const int decimal_exponent = brute_decimal_exponent<Profile>(label, rng);
        const int binary_exponent = static_cast<int>(std::lround(static_cast<double>(decimal_exponent) * log2_10));

        std::string text;
        if (rng.negative())
            text.push_back('-');
        text += "0x";
        text.push_back(rng.hex_digit(true));
        text.push_back('.');
        const int hex_digits = rng.normal_integer(
            1,
            std::max(8, std::numeric_limits<typename Profile::fltx_type>::digits10 / 2));
        text += rng.hex_digits(static_cast<std::size_t>(hex_digits));
        text.push_back('p');
        text.push_back(binary_exponent < 0 ? '-' : '+');
        text += std::to_string(std::abs(binary_exponent));
        return text;
    }

    [[nodiscard]] inline std::string make_fixed_text(std::string_view label, io_text_rng& rng)
    {
        if (label == "high-prec")
            return with_sign(rng, "0." + rng.digits(static_cast<std::size_t>(rng.integer(70, 105))));
        if (label == "large-exp")
            return with_sign(rng, rng.digits(static_cast<std::size_t>(rng.integer(145, 165))) +
                "." + rng.digits(static_cast<std::size_t>(rng.integer(10, 24)), false));
        if (label == "small-exp")
            return with_sign(rng, "0." + std::string(static_cast<std::size_t>(rng.integer(145, 165)), '0') +
                rng.digits(static_cast<std::size_t>(rng.integer(20, 44))));
        if (label == "big-dec")
            return with_sign(rng, rng.digits(static_cast<std::size_t>(rng.integer(13, 17))) +
                "." + rng.digits(static_cast<std::size_t>(rng.integer(22, 58)), false));
        if (label == "low-prec")
        {
            static constexpr std::array<std::string_view, 8> values{
                "0.125", "-0.5", "1.25", "-2.75", "16.5", "-32.125", "1024", "-4096.25"
            };
            return std::string(values[static_cast<std::size_t>(rng.integer(0, static_cast<int>(values.size() - 1)))]);
        }
        if (label == "tiny")
            return with_sign(rng, "0." + std::string(static_cast<std::size_t>(rng.integer(17, 23)), '0') +
                rng.digits(static_cast<std::size_t>(rng.integer(20, 48))));

        return with_sign(rng, rng.digits(static_cast<std::size_t>(rng.integer(21, 26))));
    }

    [[nodiscard]] inline std::string make_scientific_text(std::string_view label, io_text_rng& rng)
    {
        int exponent = 0;
        if (label == "high-prec")
            exponent = rng.integer(-4, 4);
        else if (label == "large-exp")
            exponent = rng.integer(145, 170);
        else if (label == "small-exp")
            exponent = -rng.integer(145, 170);
        else if (label == "big-dec")
            exponent = rng.integer(12, 18);
        else if (label == "low-prec")
            exponent = rng.integer(-3, 5);
        else if (label == "tiny")
            exponent = -rng.integer(18, 24);
        else
            exponent = rng.integer(21, 26);

        std::string text;
        if (rng.negative())
            text.push_back('-');
        text.push_back(rng.digit(true));
        text.push_back('.');
        const int digit_count = label == "low-prec" ? rng.integer(1, 6) : rng.integer(24, 82);
        text += rng.digits(static_cast<std::size_t>(digit_count), false);
        text.push_back('e');
        text.push_back(exponent < 0 ? '-' : '+');
        text += std::to_string(std::abs(exponent));
        return text;
    }

    [[nodiscard]] inline std::string make_hex_text(std::string_view label, io_text_rng& rng)
    {
        int exponent = 0;
        if (label == "high-prec")
            exponent = rng.integer(-16, 16);
        else if (label == "large-exp")
            exponent = rng.integer(480, 565);
        else if (label == "small-exp")
            exponent = -rng.integer(480, 565);
        else if (label == "big-dec")
            exponent = rng.integer(42, 60);
        else if (label == "low-prec")
            exponent = rng.integer(-8, 12);
        else if (label == "tiny")
            exponent = -rng.integer(58, 78);
        else
            exponent = rng.integer(70, 90);

        std::string text;
        if (rng.negative())
            text.push_back('-');
        text += "0x";
        text.push_back(rng.hex_digit(true));
        text.push_back('.');
        text += rng.hex_digits(static_cast<std::size_t>(label == "low-prec" ? rng.integer(1, 5) : rng.integer(24, 58)));
        text.push_back('p');
        text.push_back(exponent < 0 ? '-' : '+');
        text += std::to_string(std::abs(exponent));
        return text;
    }

    template<class Profile>
    [[nodiscard]] io_sample<Profile> make_sample(std::string text, bool hexfloat, int precision = -1)
    {
        typename Profile::perfect_ref oracle = parse_oracle<Profile>(text, hexfloat);
        return { std::move(text), std::move(oracle), precision };
    }

    template<class Profile>
    [[nodiscard]] std::vector<io_sample_group<Profile>> make_sample_groups()
    {
        static constexpr std::array<std::string_view, 7> labels{
            "high-prec", "large-exp", "small-exp", "big-dec", "low-prec", "tiny", "huge"
        };
        static constexpr std::array<std::string_view, 3> brute_labels{
            "brute-huge", "brute-med", "brute-tiny"
        };

        io_text_rng rng{ Profile::precision == precision_type::f128 ? 0x12810f00dull : 0x25610f00dull };
        std::vector<io_sample_group<Profile>> groups;
        groups.reserve(labels.size() + brute_labels.size());

        for (std::string_view label : labels)
        {
            io_sample_group<Profile> group{ label };
            group.defaultfloat.reserve(samples_per_kind);
            group.fixed.reserve(samples_per_kind);
            group.scientific.reserve(samples_per_kind);
            group.hexfloat.reserve(samples_per_kind);

            for (std::size_t index = 0; index < samples_per_kind; ++index)
            {
                group.fixed.push_back(make_sample<Profile>(make_fixed_text(label, rng), false));
                group.scientific.push_back(make_sample<Profile>(make_scientific_text(label, rng), false));
                group.hexfloat.push_back(make_sample<Profile>(make_hex_text(label, rng), true));
            }

            groups.push_back(std::move(group));
        }

        for (std::string_view label : brute_labels)
        {
            io_sample_group<Profile> group{ label };
            group.defaultfloat.reserve(samples_per_kind);
            group.fixed.reserve(samples_per_kind);
            group.scientific.reserve(samples_per_kind);
            group.hexfloat.reserve(samples_per_kind);

            for (std::size_t index = 0; index < samples_per_kind; ++index)
            {
                const int default_precision = rng.integer(0, std::numeric_limits<typename Profile::fltx_type>::digits10);
                const int fixed_precision = rng.integer(0, std::numeric_limits<typename Profile::fltx_type>::digits10);
                const int scientific_precision = rng.integer(0, std::numeric_limits<typename Profile::fltx_type>::digits10);
                const int hex_precision = rng.integer(0, std::numeric_limits<typename Profile::fltx_type>::digits10);
                group.defaultfloat.push_back(make_sample<Profile>(
                    make_brute_defaultfloat_text<Profile>(label, rng, default_precision),
                    false,
                    default_precision));
                group.fixed.push_back(make_sample<Profile>(
                    make_brute_fixed_text<Profile>(label, rng),
                    false,
                    fixed_precision));
                group.scientific.push_back(make_sample<Profile>(
                    make_brute_scientific_text<Profile>(label, rng),
                    false,
                    scientific_precision));
                group.hexfloat.push_back(make_sample<Profile>(
                    make_brute_hex_text<Profile>(label, rng),
                    true,
                    hex_precision));
            }

            groups.push_back(std::move(group));
        }

        return groups;
    }

    [[nodiscard]] inline std::size_t benchmark_repetitions(std::size_t sample_count) noexcept
    {
        if (sample_count == 0)
            return 0;
        return std::max<std::size_t>(3, (benchmark_min_iterations + sample_count - 1) / sample_count);
    }

    template<class Profile, class Values, class EvalFn, class ConsumeFn>
    [[nodiscard]] benchmark_result benchmark_values(const Values& values, EvalFn eval, ConsumeFn consume)
    {
        const std::size_t repetitions = benchmark_repetitions(values.size());
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (const auto& value : values)
                consume(eval(value));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    template<class Profile, class Values, class EvalFn, class ConsumeFn>
    [[nodiscard]] benchmark_result benchmark_indexed_values(const Values& values, EvalFn eval, ConsumeFn consume)
    {
        const std::size_t repetitions = benchmark_repetitions(values.size());
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        {
            for (std::size_t index = 0; index < values.size(); ++index)
                consume(eval(index, values[index]));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const std::size_t iterations = repetitions * values.size();
        const double ns = std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
        return { ns, iterations };
    }

    template<class Profile, class Samples, class EvalFn, class ExpectedFn, class IdealBitsFn>
    [[nodiscard]] accuracy_result measure_accuracy(
        const Samples& samples,
        EvalFn eval,
        ExpectedFn expected_value,
        IdealBitsFn ideal_bits)
    {
        double total_bits = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::vector<double> domain_scores;
        domain_scores.reserve(samples.size());

        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            double bits = 0.0;
            double sample_ideal_bits = Profile::ideal_bits;
            try
            {
                const typename Profile::perfect_ref expected = expected_value(index);
                sample_ideal_bits = ideal_bits(index, expected);
                const typename Profile::perfect_ref actual = eval(index);
                bits = Profile::matching_bits(actual, expected);
                if (std::isnan(bits))
                    bits = 0.0;
            }
            catch (...)
            {
                bits = 0.0;
            }

            worst_bits = std::min(worst_bits, bits);
            total_bits += Profile::finite_for_mean(bits);
            domain_scores.push_back(domain_sample_score(bits, sample_ideal_bits));
        }

        return {
            worst_bits,
            total_bits / static_cast<double>(samples.size()),
            samples.size(),
            domain_score(std::move(domain_scores))
        };
    }

    [[nodiscard]] inline bool is_decimal_digit(char ch) noexcept
    {
        return ch >= '0' && ch <= '9';
    }

    [[nodiscard]] inline std::size_t exponent_marker_pos(std::string_view text) noexcept
    {
        const std::size_t e = text.find_first_of("eE");
        return e == std::string_view::npos ? text.size() : e;
    }

    [[nodiscard]] inline int fractional_digit_count(std::string_view text) noexcept
    {
        const std::size_t exponent = exponent_marker_pos(text);
        const std::size_t point = text.substr(0, exponent).find('.');
        if (point == std::string_view::npos)
            return 0;
        return static_cast<int>(exponent - point - 1);
    }

    [[nodiscard]] inline int significant_digit_count(std::string_view text) noexcept
    {
        const std::size_t exponent = exponent_marker_pos(text);
        int count = 0;
        bool seen_nonzero = false;
        bool saw_digit = false;

        for (std::size_t index = 0; index < exponent; ++index)
        {
            const char ch = text[index];
            if (!is_decimal_digit(ch))
                continue;

            saw_digit = true;
            if (ch != '0')
                seen_nonzero = true;
            if (seen_nonzero)
                ++count;
        }

        return count == 0 && saw_digit ? 1 : count;
    }

    [[nodiscard]] inline bool formatted_decimal_respects_precision(
        std::string_view text,
        int precision,
        const io_format_case& format) noexcept
    {
        if (format.hexfloat)
            return true;

        const int requested = precision < 0 ? 6 : precision;
        const std::ios_base::fmtflags floatfield = format.flags & std::ios_base::floatfield;
        if (floatfield == std::ios_base::fixed)
            return text.find_first_of("eE") == std::string_view::npos &&
                   fractional_digit_count(text) == requested;

        if (floatfield == std::ios_base::scientific)
            return text.find_first_of("eE") != std::string_view::npos &&
                   fractional_digit_count(text) == requested;

        const int max_significant_digits = requested == 0 ? 1 : requested;
        return significant_digit_count(text) <= max_significant_digits;
    }

    template<class Profile, class Samples, class Value, class FormatFn>
    [[nodiscard]] accuracy_result measure_to_string_accuracy(
        const Samples& samples,
        const std::vector<Value>& values,
        int precision,
        const io_format_case& format,
        FormatFn format_value)
    {
        return measure_accuracy<Profile>(
            samples,
            [&](std::size_t index)
            {
                const int sample_precision = samples[index].precision >= 0 ? samples[index].precision : precision;
                const std::string text = format_value(values[index], sample_precision, format.flags);
                if (!formatted_decimal_respects_precision(text, sample_precision, format))
                    throw std::invalid_argument("formatter ignored requested precision");
                return parse_oracle<Profile>(text, format.hexfloat);
            },
            [&](std::size_t index)
            {
                return Profile::to_perfect(values[index]);
            },
            [](std::size_t, const typename Profile::perfect_ref& expected)
            {
                return Profile::template domain_ideal_bits_for<Value>(expected);
            });
    }

    template<class Profile, class Value, class Samples, class ParseFn>
    [[nodiscard]] accuracy_result measure_parse_accuracy(
        const Samples& samples,
        ParseFn parse_value)
    {
        return measure_accuracy<Profile>(
            samples,
            [&](std::size_t index)
            {
                const Value value = parse_value(samples[index].text);
                return Profile::to_perfect(value);
            },
            [&](std::size_t index)
            {
                return samples[index].oracle;
            },
            [](std::size_t, const typename Profile::perfect_ref& expected)
            {
                return Profile::template domain_ideal_bits_for<Value>(expected);
            });
    }

    template<class T>
    [[nodiscard]] T parse_stream_value(std::string_view text)
    {
        std::istringstream stream{ std::string(text) };
        T value{};
        stream >> value;
        if (!stream || stream.peek() != std::char_traits<char>::eof())
            throw std::invalid_argument("stream parse failed");
        return value;
    }

    template<class T>
    [[nodiscard]] T parse_qdpp_value(std::string_view text)
    {
        std::string copy(text);
        T value{};
        if (T::read(copy.c_str(), value) < 0)
            throw std::invalid_argument("qdpp parse failed");
        return value;
    }

    template<class T>
    [[nodiscard]] std::string format_stream_value(const T& value, int precision, std::ios_base::fmtflags flags)
    {
        std::ostringstream stream;
        stream.precision(precision);
        stream.setf(flags & std::ios_base::floatfield, std::ios_base::floatfield);
        stream << value;
        return stream.str();
    }

    template<class T>
    [[nodiscard]] std::string format_qdpp_value(const T& value, int precision, std::ios_base::fmtflags flags)
    {
        return value.to_string(
            precision,
            0,
            flags,
            (flags & std::ios_base::showpos) != std::ios_base::fmtflags{},
            (flags & std::ios_base::uppercase) != std::ios_base::fmtflags{});
    }

    template<class Profile>
    [[nodiscard]] std::vector<typename Profile::fltx_type> make_fltx_values(const std::vector<io_sample<Profile>>& samples)
    {
        std::vector<typename Profile::fltx_type> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back(bl::parse<typename Profile::fltx_type>(sample.text));
        return values;
    }

    template<class Profile>
    [[nodiscard]] std::vector<typename Profile::competitor_ref> make_competitor_values(const std::vector<io_sample<Profile>>& samples)
    {
        std::vector<typename Profile::competitor_ref> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back(parse_stream_value<typename Profile::competitor_ref>(sample.text));
        return values;
    }

    template<class Profile>
    [[nodiscard]] std::vector<typename Profile::extra_competitor_ref> make_extra_competitor_values(const std::vector<io_sample<Profile>>& samples)
    {
        std::vector<typename Profile::extra_competitor_ref> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back(parse_qdpp_value<typename Profile::extra_competitor_ref>(sample.text));
        return values;
    }

    template<class Profile>
    [[nodiscard]] bool qdpp_fixed_format_is_unsafe(
        const std::vector<io_sample<Profile>>& samples,
        const io_format_case& format)
    {
        if ((format.flags & std::ios_base::floatfield) != std::ios_base::fixed)
            return false;

        // qdpp's fixed formatter can hit a fatal path on extremely large decimal expansions.
        const typename Profile::perfect_ref limit{ "1e100" };
        for (const io_sample<Profile>& sample : samples)
        {
            const typename Profile::perfect_ref magnitude =
                sample.oracle < 0 ? -sample.oracle : sample.oracle;
            if (magnitude > limit)
                return true;
        }
        return false;
    }

    [[nodiscard]] inline bool qdpp_format_is_unsupported(const io_format_case& format) noexcept
    {
        if (format.hexfloat)
            return true;

        // qdpp treats "no floatfield" like scientific output rather than the
        // standard defaultfloat rules, so it is not a comparable reference.
        return (format.flags & std::ios_base::floatfield) == std::ios_base::fmtflags{};
    }

    [[nodiscard]] inline suite_id make_suite(precision_type precision, std::string_view operation) noexcept
    {
        return {
            precision,
            operation_id{ operation, operation },
            io_domain
        };
    }

    [[nodiscard]] inline std::string make_to_string_operation(
        std::string_view sample_label,
        std::string_view precision_label,
        std::string_view format_label)
    {
        std::string operation = "to_string(";
        operation += sample_label;
        operation += ", ";
        operation += precision_label;
        operation += ", ";
        operation += format_label;
        operation += ")";
        return operation;
    }

    [[nodiscard]] inline std::string make_to_string_operation(
        std::string_view sample_label,
        int precision,
        std::string_view format_label)
    {
        return make_to_string_operation(sample_label, std::to_string(precision), format_label);
    }

    [[nodiscard]] inline std::string make_parse_operation(
        std::string_view sample_label,
        std::string_view style_label)
    {
        std::string operation = "parse(";
        operation += sample_label;
        operation += ", ";
        operation += style_label;
        operation += ")";
        return operation;
    }

    [[nodiscard]] inline std::string make_brute_parse_operation(
        std::string_view sample_label,
        std::string_view style_label)
    {
        std::string operation = "parse(rand_prec, ";
        operation += style_label;
        operation += ", ";
        operation += sample_label;
        operation += ")";
        return operation;
    }

    template<class Profile>
    [[nodiscard]] metrics_record make_record(std::string_view operation)
    {
        metrics_record record;
        record.suite = make_suite(Profile::precision, operation);
        record.competitor_name = Profile::references::competitor_name;
        record.extra_competitors.push_back({ Profile::references::extra_competitor_name });
        return record;
    }

    template<class Profile>
    [[nodiscard]] metrics_record make_to_string_record(
        std::string_view operation,
        const std::vector<io_sample<Profile>>& samples,
        int precision,
        const io_format_case& format)
    {
        metrics_record record = make_record<Profile>(operation);

        const auto fltx_values = make_fltx_values<Profile>(samples);
        const auto competitor_values = make_competitor_values<Profile>(samples);
        const auto extra_values = make_extra_competitor_values<Profile>(samples);

        record.fltx_accuracy = measure_to_string_accuracy<Profile>(
            samples,
            fltx_values,
            precision,
            format,
            [](const auto& value, int digits, std::ios_base::fmtflags flags)
            {
                return bl::to_string(value, digits, flags);
            });
        record.fltx_benchmark = benchmark_indexed_values<Profile>(
            fltx_values,
            [&samples, precision, format](std::size_t index, const auto& value)
            {
                const int sample_precision = samples[index].precision >= 0 ? samples[index].precision : precision;
                return bl::to_string(value, sample_precision, format.flags);
            },
            [](const std::string& text)
            {
                Profile::consume_text(text);
            });

        if (format.hexfloat)
        {
            record.competitor_supported = false;
            record.extra_competitors.front().supported = false;
            return record;
        }

        record.competitor_accuracy = measure_to_string_accuracy<Profile>(
            samples,
            competitor_values,
            precision,
            format,
            [](const auto& value, int digits, std::ios_base::fmtflags flags)
            {
                return format_stream_value(value, digits, flags);
            });
        record.competitor_benchmark = benchmark_indexed_values<Profile>(
            competitor_values,
            [&samples, precision, format](std::size_t index, const auto& value)
            {
                const int sample_precision = samples[index].precision >= 0 ? samples[index].precision : precision;
                return format_stream_value(value, sample_precision, format.flags);
            },
            [](const std::string& text)
            {
                Profile::consume_text(text);
            });

        competitor_result& extra = record.extra_competitors.front();
        if (qdpp_format_is_unsupported(format))
        {
            extra.supported = false;
            return record;
        }

        if (qdpp_fixed_format_is_unsafe(samples, format))
        {
            extra.supported = false;
            return record;
        }

        extra.accuracy = measure_to_string_accuracy<Profile>(
            samples,
            extra_values,
            precision,
            format,
            [](const auto& value, int digits, std::ios_base::fmtflags flags)
            {
                return format_qdpp_value(value, digits, flags);
            });
        extra.benchmark = benchmark_indexed_values<Profile>(
            extra_values,
            [&samples, precision, format](std::size_t index, const auto& value)
            {
                const int sample_precision = samples[index].precision >= 0 ? samples[index].precision : precision;
                return format_qdpp_value(value, sample_precision, format.flags);
            },
            [](const std::string& text)
            {
                Profile::consume_text(text);
            });

        return record;
    }

    template<class Profile>
    [[nodiscard]] metrics_record make_parse_record(
        std::string_view operation,
        const std::vector<io_sample<Profile>>& samples,
        bool hexfloat)
    {
        metrics_record record = make_record<Profile>(operation);

        std::vector<std::string> texts;
        texts.reserve(samples.size());
        for (const auto& sample : samples)
            texts.push_back(sample.text);

        record.fltx_accuracy = measure_parse_accuracy<Profile, typename Profile::fltx_type>(
            samples,
            [](std::string_view text)
            {
                return bl::parse<typename Profile::fltx_type>(text);
            });
        record.fltx_benchmark = benchmark_values<Profile>(
            texts,
            [](const std::string& text)
            {
                return bl::parse<typename Profile::fltx_type>(text);
            },
            [](const auto& value)
            {
                Profile::consume(value);
            });

        if (hexfloat)
        {
            record.competitor_supported = false;
            record.extra_competitors.front().supported = false;
            return record;
        }

        record.competitor_accuracy = measure_parse_accuracy<Profile, typename Profile::competitor_ref>(
            samples,
            [](std::string_view text)
            {
                return parse_stream_value<typename Profile::competitor_ref>(text);
            });
        record.competitor_benchmark = benchmark_values<Profile>(
            texts,
            [](const std::string& text)
            {
                return parse_stream_value<typename Profile::competitor_ref>(text);
            },
            [](const auto& value)
            {
                Profile::consume(value);
            });

        competitor_result& extra = record.extra_competitors.front();
        extra.accuracy = measure_parse_accuracy<Profile, typename Profile::extra_competitor_ref>(
            samples,
            [](std::string_view text)
            {
                return parse_qdpp_value<typename Profile::extra_competitor_ref>(text);
            });
        extra.benchmark = benchmark_values<Profile>(
            texts,
            [](const std::string& text)
            {
                return parse_qdpp_value<typename Profile::extra_competitor_ref>(text);
            },
            [](const auto& value)
            {
                Profile::consume(value);
            });

        return record;
    }

    [[nodiscard]] inline accuracy_result average_accuracy(const std::vector<accuracy_result>& values)
    {
        double total_bits = 0.0;
        double total_domain = 0.0;
        double worst_bits = std::numeric_limits<double>::infinity();
        std::size_t sample_count = 0;

        for (const accuracy_result& value : values)
        {
            total_bits += value.mean_bits * static_cast<double>(value.sample_count);
            total_domain += value.domain_score * static_cast<double>(value.sample_count);
            worst_bits = std::min(worst_bits, value.worst_bits);
            sample_count += value.sample_count;
        }

        if (sample_count == 0)
            return {};

        return {
            worst_bits,
            total_bits / static_cast<double>(sample_count),
            sample_count,
            total_domain / static_cast<double>(sample_count)
        };
    }

    [[nodiscard]] inline benchmark_result average_benchmark(const std::vector<benchmark_result>& values)
    {
        double total_ns = 0.0;
        std::size_t count = 0;
        std::size_t iterations = 0;
        for (const benchmark_result& value : values)
        {
            if (value.iteration_count == 0)
                continue;
            total_ns += value.ns_per_iter;
            iterations += value.iteration_count;
            ++count;
        }

        if (count == 0)
            return {};

        return { total_ns / static_cast<double>(count), iterations };
    }

    template<class Profile>
    [[nodiscard]] metrics_record make_average_record(
        std::string_view operation,
        const std::vector<metrics_record>& records)
    {
        metrics_record average = make_record<Profile>(operation);
        std::vector<accuracy_result> fltx_accuracy;
        std::vector<accuracy_result> primary_supported_fltx_accuracy;
        std::vector<accuracy_result> competitor_accuracy;
        std::vector<accuracy_result> extra_accuracy;
        std::vector<benchmark_result> fltx_benchmark;
        std::vector<benchmark_result> primary_supported_fltx_benchmark;
        std::vector<benchmark_result> competitor_benchmark;
        std::vector<benchmark_result> extra_benchmark;

        fltx_accuracy.reserve(records.size());
        primary_supported_fltx_accuracy.reserve(records.size());
        competitor_accuracy.reserve(records.size());
        extra_accuracy.reserve(records.size());
        fltx_benchmark.reserve(records.size());
        primary_supported_fltx_benchmark.reserve(records.size());
        competitor_benchmark.reserve(records.size());
        extra_benchmark.reserve(records.size());

        bool competitor_has_any_support = false;
        bool extra_has_any_support = false;

        for (const metrics_record& record : records)
        {
            fltx_accuracy.push_back(record.fltx_accuracy);
            fltx_benchmark.push_back(record.fltx_benchmark);

            if (record.competitor_supported)
            {
                competitor_has_any_support = true;
                primary_supported_fltx_accuracy.push_back(record.fltx_accuracy);
                primary_supported_fltx_benchmark.push_back(record.fltx_benchmark);
                competitor_accuracy.push_back(record.competitor_accuracy);
                competitor_benchmark.push_back(record.competitor_benchmark);
            }

            if (!record.extra_competitors.empty() && record.extra_competitors.front().supported)
            {
                extra_has_any_support = true;
                extra_accuracy.push_back(record.extra_competitors.front().accuracy);
                extra_benchmark.push_back(record.extra_competitors.front().benchmark);
            }
        }

        average.fltx_accuracy = competitor_has_any_support
            ? average_accuracy(primary_supported_fltx_accuracy)
            : average_accuracy(fltx_accuracy);
        average.fltx_benchmark = competitor_has_any_support
            ? average_benchmark(primary_supported_fltx_benchmark)
            : average_benchmark(fltx_benchmark);
        average.competitor_supported = competitor_has_any_support;
        average.competitor_accuracy = average_accuracy(competitor_accuracy);
        average.competitor_benchmark = average_benchmark(competitor_benchmark);
        average.extra_competitors.front().supported = extra_has_any_support;
        average.extra_competitors.front().accuracy = average_accuracy(extra_accuracy);
        average.extra_competitors.front().benchmark = average_benchmark(extra_benchmark);
        return average;
    }

    template<class Profile, class Sink>
    void emit_to_string_records(const std::vector<io_sample_group<Profile>>& groups, Sink&& sink)
    {
        std::vector<metrics_record> records;
        const auto precisions = precision_cases<Profile>();
        const auto formats = format_cases();
        records.reserve(groups.size() * precisions.size() * formats.size() + 1u);

        for (const io_sample_group<Profile>& group : groups)
        {
            if (is_brute_label(group.label))
            {
                for (const io_format_case& format : formats)
                {
                    const std::string_view operation = intern_operation_name(
                        make_to_string_operation(group.label, "rand", format.label));
                    records.push_back(make_to_string_record<Profile>(
                        operation,
                        group.scientific,
                        std::numeric_limits<typename Profile::fltx_type>::digits10,
                        format));
                    sink(records.back());
                }
                continue;
            }

            for (int precision : precisions)
            {
                for (const io_format_case& format : formats)
                {
                    const std::string_view operation = intern_operation_name(
                        make_to_string_operation(group.label, precision, format.label));
                    records.push_back(make_to_string_record<Profile>(operation, group.scientific, precision, format));
                    sink(records.back());
                }
            }
        }

        records.push_back(make_average_record<Profile>(
            intern_operation_name("to_string (average)"),
            records));
        sink(records.back());
    }

    template<class Profile>
    [[nodiscard]] std::vector<metrics_record> make_to_string_records(const std::vector<io_sample_group<Profile>>& groups)
    {
        std::vector<metrics_record> records;
        emit_to_string_records<Profile>(
            groups,
            [&records](const metrics_record& record)
            {
                records.push_back(record);
            });
        return records;
    }

    template<class Profile, class Sink>
    void emit_parse_records(const std::vector<io_sample_group<Profile>>& groups, Sink&& sink)
    {
        struct parse_case
        {
            std::string_view label;
            bool hexfloat;
            std::vector<io_sample<Profile>> io_sample_group<Profile>::* samples;
        };

        const std::array<parse_case, 3> cases{
            parse_case{ "fixed", false, &io_sample_group<Profile>::fixed },
            parse_case{ "scientific", false, &io_sample_group<Profile>::scientific },
            parse_case{ "hexfloat", true, &io_sample_group<Profile>::hexfloat }
        };
        const std::array<parse_case, 4> brute_cases{
            parse_case{ "default", false, &io_sample_group<Profile>::defaultfloat },
            parse_case{ "fixed", false, &io_sample_group<Profile>::fixed },
            parse_case{ "scientific", false, &io_sample_group<Profile>::scientific },
            parse_case{ "hexfloat", true, &io_sample_group<Profile>::hexfloat }
        };

        std::vector<metrics_record> records;
        records.reserve(groups.size() * brute_cases.size() + 1u);
        for (const io_sample_group<Profile>& group : groups)
        {
            const bool brute = is_brute_label(group.label);
            if (brute)
            {
                for (const parse_case& parse : brute_cases)
                {
                    const std::string_view operation = intern_operation_name(
                        make_brute_parse_operation(group.label, parse.label));
                    records.push_back(make_parse_record<Profile>(operation, group.*(parse.samples), parse.hexfloat));
                    sink(records.back());
                }
                continue;
            }

            for (const parse_case& parse : cases)
            {
                const std::string_view operation = intern_operation_name(
                    make_parse_operation(group.label, parse.label));
                records.push_back(make_parse_record<Profile>(operation, group.*(parse.samples), parse.hexfloat));
                sink(records.back());
            }
        }

        records.push_back(make_average_record<Profile>(
            intern_operation_name("parse (average)"),
            records));
        sink(records.back());
    }

    template<class Profile>
    [[nodiscard]] std::vector<metrics_record> make_parse_records(const std::vector<io_sample_group<Profile>>& groups)
    {
        std::vector<metrics_record> records;
        emit_parse_records<Profile>(
            groups,
            [&records](const metrics_record& record)
            {
                records.push_back(record);
            });
        return records;
    }

    template<class Profile, class Sink>
    void emit_io_records(Sink&& sink)
    {
        const std::vector<io_sample_group<Profile>> groups = make_sample_groups<Profile>();
        emit_to_string_records<Profile>(groups, sink);
        emit_parse_records<Profile>(groups, sink);
    }

    template<class Profile>
    [[nodiscard]] std::vector<metrics_record> make_io_records()
    {
        std::vector<metrics_record> records;
        emit_io_records<Profile>(
            [&records](const metrics_record& record)
            {
                records.push_back(record);
            });
        return records;
    }
}

#endif
