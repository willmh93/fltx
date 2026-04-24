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

TEST_CASE("f256 parse benchmark uses 100000 random full-length strings", "[f256][bench][parse]")
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

    BENCHMARK("parse 100000 random full-length f256 strings")
    {
        return parse_all_and_hash(samples);
    };
}
