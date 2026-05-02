#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fltx.h>

namespace
{
    constexpr std::size_t kSampleCount = 100000;
    constexpr int kSignificantDigits = 80;
    constexpr int kExponentLimit = 300;
    constexpr std::uint64_t kSeed = 0x6c8e9cf570932bd1ULL;

    void append_zero_padded_unsigned(std::string& text, std::uint64_t value, int width)
    {
        char buf[20]{};
        int len = 0;
        do
        {
            buf[len++] = static_cast<char>('0' + (value % 10));
            value /= 10;
        } while (value != 0);

        for (int i = len; i < width; ++i)
            text.push_back('0');
        for (int i = len - 1; i >= 0; --i)
            text.push_back(buf[i]);
    }

    std::uint64_t pow10_u64(int exponent)
    {
        std::uint64_t value = 1;
        for (int i = 0; i < exponent; ++i)
            value *= 10;
        return value;
    }

    std::vector<std::string> make_random_f256_strings()
    {
        std::mt19937_64 rng(kSeed);
        std::bernoulli_distribution sign_dist(0.5);
        std::uniform_int_distribution<int> first_digit_dist(1, 9);
        std::uniform_int_distribution<int> digit_dist(0, 9);
        std::uniform_int_distribution<int> exponent_dist(-kExponentLimit, kExponentLimit);

        std::vector<std::string> samples;
        samples.reserve(kSampleCount);

        for (std::size_t i = 0; i < kSampleCount; ++i)
        {
            std::string text;
            text.reserve(static_cast<std::size_t>(kSignificantDigits + 12));

            if (sign_dist(rng))
                text.push_back('-');

            text.push_back(static_cast<char>('0' + first_digit_dist(rng)));
            text.push_back('.');

            for (int digit_index = 1; digit_index < kSignificantDigits; ++digit_index)
                text.push_back(static_cast<char>('0' + digit_dist(rng)));

            const int exponent = exponent_dist(rng);
            text.push_back('e');
            text.push_back(exponent < 0 ? '-' : '+');

            int exponent_abs = exponent < 0 ? -exponent : exponent;
            if (exponent_abs >= 100)
            {
                text.push_back(static_cast<char>('0' + (exponent_abs / 100)));
                exponent_abs %= 100;
                text.push_back(static_cast<char>('0' + (exponent_abs / 10)));
                text.push_back(static_cast<char>('0' + (exponent_abs % 10)));
            }
            else
            {
                text.push_back(static_cast<char>('0' + (exponent_abs / 10)));
                text.push_back(static_cast<char>('0' + (exponent_abs % 10)));
            }

            samples.push_back(std::move(text));
        }

        return samples;
    }

    std::vector<std::string> make_common_qd_literal_strings()
    {
        std::mt19937_64 rng(kSeed ^ 0xa51b6c3d94f2807eULL);
        std::uniform_int_distribution<int> style_dist(0, 4);
        std::uniform_int_distribution<std::uint64_t> integer_dist(0, 1000000);
        std::uniform_int_distribution<int> frac_digit_count_dist(1, 6);
        std::uniform_int_distribution<int> exponent_dist(-8, 8);
        std::uniform_int_distribution<int> small_exponent_dist(0, 6);

        std::vector<std::string> samples;
        samples.reserve(kSampleCount);

        samples.push_back("123.456");
        samples.push_back("1024.0");
        samples.push_back("1e6");
        samples.push_back("3.14e-5");

        while (samples.size() < kSampleCount)
        {
            std::string text;
            const int style = style_dist(rng);

            if (style == 0)
            {
                text = std::to_string(integer_dist(rng));
            }
            else if (style == 1)
            {
                const int frac_digits = frac_digit_count_dist(rng);
                text = std::to_string(integer_dist(rng));
                text.push_back('.');
                append_zero_padded_unsigned(text, rng() % pow10_u64(frac_digits), frac_digits);
            }
            else if (style == 2)
            {
                text = std::to_string(integer_dist(rng));
                text += ".0";
            }
            else if (style == 3)
            {
                text = std::to_string((integer_dist(rng) % 9999) + 1);
                text.push_back('e');
                text += std::to_string(small_exponent_dist(rng));
            }
            else
            {
                const int frac_digits = frac_digit_count_dist(rng);
                text = std::to_string((integer_dist(rng) % 9999) + 1);
                text.push_back('.');
                append_zero_padded_unsigned(text, rng() % pow10_u64(frac_digits), frac_digits);
                text.push_back('e');

                const int exponent = exponent_dist(rng);
                if (exponent >= 0)
                    text.push_back('+');
                text += std::to_string(exponent);
            }

            samples.push_back(std::move(text));
        }

        return samples;
    }

    std::uint64_t parse_all_and_hash(const std::vector<std::string>& samples)
    {
        std::uint64_t hash = 0xcbf29ce484222325ULL;

        for (const std::string& text : samples)
        {
            const bl::f256_s value = bl::to_f256(text.c_str());

            hash ^= std::bit_cast<std::uint64_t>(value.x0);
            hash *= 1099511628211ULL;

            hash ^= std::bit_cast<std::uint64_t>(value.x1);
            hash *= 1099511628211ULL;

            hash ^= std::bit_cast<std::uint64_t>(value.x2);
            hash *= 1099511628211ULL;

            hash ^= std::bit_cast<std::uint64_t>(value.x3);
            hash *= 1099511628211ULL;
        }

        return hash;
    }
}

TEST_CASE("f256 parse benchmark uses common short qd literal strings", "[f256][bench_parse]")
{
    static const std::vector<std::string> samples = make_common_qd_literal_strings();

    REQUIRE(samples.size() == kSampleCount);

    for (const std::string& text : samples)
    {
        bl::f256_s value{};
        const char* end = nullptr;

        REQUIRE(bl::parse_flt256(text.c_str(), value, &end));
        REQUIRE(end != nullptr);
        REQUIRE(*end == '\0');
    }

    BENCHMARK("parse common short qd literal strings")
    {
        return parse_all_and_hash(samples);
    };
}

TEST_CASE("f256 parse benchmark random full-length strings", "[f256][bench_parse]")
{
    static const std::vector<std::string> samples = make_random_f256_strings();

    REQUIRE(samples.size() == kSampleCount);

    for (const std::string& text : samples)
    {
        bl::f256_s value{};
        const char* end = nullptr;

        REQUIRE(bl::parse_flt256(text.c_str(), value, &end));
        REQUIRE(end != nullptr);
        REQUIRE(*end == '\0');
    }

    BENCHMARK("parse random full-length f256 strings")
    {
        return parse_all_and_hash(samples);
    };
}
