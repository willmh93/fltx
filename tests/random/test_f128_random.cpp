#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include <fltx/f128_comparison.h>
#include <fltx/f128_io.h>
#include <fltx/random.h>

#include "random_print_utils.h"

namespace
{
    [[nodiscard]] constexpr bl::f128 constexpr_unit_sample()
    {
        bl::mt19937_64 rng{ 0x123456789abcdef0ull };
        bl::uniform_real_distribution<bl::f128> dist{ bl::f128{ 0.0 }, bl::f128{ 1.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f128 constexpr_interval_sample()
    {
        bl::mt19937_64 rng{ 0x1020304050607080ull };
        bl::uniform_real_distribution<bl::f128> dist{ bl::f128{ -2.0 }, bl::f128{ 3.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr auto constexpr_random_array_sample()
    {
        return bl::random_array<4>(
            bl::mt19937_64{ 0x1020304050607080ull },
            bl::uniform_real_distribution<bl::f128>{ bl::f128{ -2.0 }, bl::f128{ 3.0 } });
    }

    [[nodiscard]] constexpr bl::f128 constexpr_canonical_sample()
    {
        bl::mt19937 rng{ 1234u };
        return bl::generate_canonical<bl::f128, std::numeric_limits<bl::f128>::digits>(rng);
    }

    [[nodiscard]] constexpr float constexpr_float_sample()
    {
        bl::mt19937_64 rng{ 0x1111222233334444ull };
        bl::uniform_real_distribution<float> dist{ -0.5f, 0.5f };
        return dist(rng);
    }

    [[nodiscard]] constexpr double constexpr_double_sample()
    {
        bl::mt19937_64 rng{ 0x5555666677778888ull };
        bl::uniform_real_distribution<double> dist{ -0.5, 0.5 };
        return dist(rng);
    }

    [[nodiscard]] constexpr int constexpr_int_sample()
    {
        bl::mt19937 rng{ 0x13579bdfu };
        bl::uniform_int_distribution<int> dist{ -17, 29 };
        return dist(rng);
    }

    [[nodiscard]] constexpr std::uint_fast32_t constexpr_seed_seq_sample()
    {
        bl::seed_seq seq{ 1u, 2u, 3u, 4u };
        bl::mt19937 rng{ seq };
        return rng();
    }

    [[nodiscard]] constexpr bl::f128 constexpr_exponential_sample()
    {
        bl::mt19937_64 rng{ 0x2468ace02468ace0ull };
        bl::exponential_distribution<bl::f128> dist{ bl::f128{ 1.25 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f128 constexpr_normal_sample()
    {
        bl::mt19937_64 rng{ 0x3141592653589793ull };
        bl::normal_distribution<bl::f128> dist{ bl::f128{ 0.5 }, bl::f128{ 2.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f128 constexpr_lognormal_sample()
    {
        bl::mt19937_64 rng{ 0x2718281828459045ull };
        bl::lognormal_distribution<bl::f128> dist{ bl::f128{ 0.0 }, bl::f128{ 0.25 } };
        return dist(rng);
    }

    struct modulo_six_urbg
    {
        using result_type = unsigned int;

        unsigned int value = 0;

        [[nodiscard]] static constexpr result_type min() noexcept { return 0; }
        [[nodiscard]] static constexpr result_type max() noexcept { return 5; }

        [[nodiscard]] constexpr result_type operator()() noexcept
        {
            const result_type result = value;
            value = (value + 1) % 6;
            return result;
        }
    };

    [[nodiscard]] constexpr double constexpr_non_power_canonical_sample()
    {
        modulo_six_urbg rng{};
        return bl::generate_canonical<double, 4>(rng);
    }

    [[nodiscard]] constexpr float constexpr_float_exponential_sample()
    {
        bl::mt19937_64 rng{ 0x1111222233334444ull };
        bl::exponential_distribution<float> dist{ 1.5f };
        return dist(rng);
    }

    [[nodiscard]] constexpr double constexpr_double_normal_sample()
    {
        bl::mt19937_64 rng{ 0x5555666677778888ull };
        bl::normal_distribution<double> dist{ -1.0, 0.25 };
        return dist(rng);
    }

    [[nodiscard]] constexpr double constexpr_double_lognormal_sample()
    {
        bl::mt19937_64 rng{ 0x9999aaaabbbbccccull };
        bl::lognormal_distribution<double> dist{ 0.1, 0.5 };
        return dist(rng);
    }

    static_assert(bl::uniform_random_bit_generator<bl::mt19937>);
    static_assert(bl::uniform_random_bit_generator<bl::mt19937_64>);
    static_assert(bl::uniform_random_bit_generator<modulo_six_urbg>);
    static_assert(bl::random_distribution_for<bl::uniform_real_distribution<bl::f128>, bl::mt19937_64>);
    static_assert(std::is_same_v<bl::default_random_engine, bl::mt19937>);
    static_assert(std::is_same_v<bl::uniform_real_distribution<bl::f128>::result_type, bl::f128>);
    static_assert(std::is_same_v<bl::uniform_int_distribution<int>::result_type, int>);
    static_assert(!std::is_copy_constructible_v<bl::random_device>);
    static_assert(!std::is_copy_assignable_v<bl::random_device>);
    static_assert(bl::random_device::min() < bl::random_device::max());
    static_assert(constexpr_unit_sample() >= bl::f128{ 0.0 });
    static_assert(constexpr_unit_sample() < bl::f128{ 1.0 });
    static_assert(constexpr_interval_sample() >= bl::f128{ -2.0 });
    static_assert(constexpr_interval_sample() < bl::f128{ 3.0 });
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(constexpr_random_array_sample())>> == 4);
    static_assert(constexpr_random_array_sample()[0] >= bl::f128{ -2.0 });
    static_assert(constexpr_random_array_sample()[0] < bl::f128{ 3.0 });
    static_assert(constexpr_random_array_sample()[3] >= bl::f128{ -2.0 });
    static_assert(constexpr_random_array_sample()[3] < bl::f128{ 3.0 });
    static_assert(constexpr_canonical_sample() >= bl::f128{ 0.0 });
    static_assert(constexpr_canonical_sample() < bl::f128{ 1.0 });
    static_assert(constexpr_float_sample() >= -0.5f);
    static_assert(constexpr_float_sample() < 0.5f);
    static_assert(constexpr_double_sample() >= -0.5);
    static_assert(constexpr_double_sample() < 0.5);
    static_assert(constexpr_int_sample() >= -17);
    static_assert(constexpr_int_sample() <= 29);
    static_assert(constexpr_seed_seq_sample() != 0u);
    static_assert(constexpr_exponential_sample() >= bl::f128{ 0.0 });
    static_assert(constexpr_normal_sample() == constexpr_normal_sample());
    static_assert(constexpr_lognormal_sample() > bl::f128{ 0.0 });
    static_assert(constexpr_non_power_canonical_sample() >= 0.0);
    static_assert(constexpr_non_power_canonical_sample() < 1.0);
    static_assert(constexpr_float_exponential_sample() >= 0.0f);
    static_assert(constexpr_double_normal_sample() == constexpr_double_normal_sample());
    static_assert(constexpr_double_lognormal_sample() > 0.0);

}

TEST_CASE("bl mersenne twister engines match std output", "[fltx][random][f128][constexpr]")
{
    bl::mt19937 bl32{ 5489u };
    std::mt19937 std32{ 5489u };
    for (int index = 0; index < 1000; ++index)
        REQUIRE(bl32() == std32());

    bl::mt19937_64 bl64{ 5489u };
    std::mt19937_64 std64{ 5489u };
    for (int index = 0; index < 1000; ++index)
        REQUIRE(bl64() == std64());
}

TEST_CASE("bl random_device exposes the runtime-only standard-shaped API", "[fltx][random]")
{
    bl::random_device device;
    const bl::random_device::result_type sample = device();

    REQUIRE(sample >= bl::random_device::min());
    REQUIRE(sample <= bl::random_device::max());
    REQUIRE(device.entropy() >= 0.0);
}

TEST_CASE("bl random_array generates deterministic distribution samples", "[fltx][random][f128][constexpr]")
{
    bl::mt19937_64 rng{ 0x1020304050607080ull };
    bl::uniform_real_distribution<bl::f128> dist{ bl::f128{ -2.0 }, bl::f128{ 3.0 } };

    const auto values = bl::random_array<4>(rng, dist);
    for (const bl::f128& value : values)
    {
        REQUIRE(value >= bl::f128{ -2.0 });
        REQUIRE(value < bl::f128{ 3.0 });
    }

    for (const bl::f128& value : values)
        REQUIRE(value == dist(rng));
}

TEST_CASE("bl seed_seq and mersenne twister seed sequence match std output", "[fltx][random][f128][constexpr]")
{
    std::seed_seq std_seq{ 1u, 2u, 3u, 4u, 5u };
    bl::seed_seq bl_seq{ 1u, 2u, 3u, 4u, 5u };

    std::array<std::uint32_t, 16> std_words{};
    std::array<std::uint32_t, 16> bl_words{};
    std_seq.generate(std_words.begin(), std_words.end());
    bl_seq.generate(bl_words.begin(), bl_words.end());
    REQUIRE(bl_words == std_words);

    std::vector<std::uint_least32_t> params;
    bl_seq.param(std::back_inserter(params));
    REQUIRE(params == std::vector<std::uint_least32_t>{ 1u, 2u, 3u, 4u, 5u });
    REQUIRE(bl_seq.size() == params.size());

    std::seed_seq std_empty_seq;
    bl::seed_seq bl_empty_seq;
    std::array<std::uint32_t, 8> std_empty_words{};
    std::array<std::uint32_t, 8> bl_empty_words{};
    std_empty_seq.generate(std_empty_words.begin(), std_empty_words.end());
    bl_empty_seq.generate(bl_empty_words.begin(), bl_empty_words.end());
    REQUIRE(bl_empty_words == std_empty_words);
    bl_empty_seq.generate(bl_empty_words.begin(), bl_empty_words.begin());
    REQUIRE(bl_empty_seq.size() == 0u);

    std::array<unsigned int, 4> constructor_words{ 0x10u, 0x20u, 0x30u, 0x40u };
    std::seed_seq std_iter_seq{ constructor_words.begin(), constructor_words.end() };
    bl::seed_seq bl_iter_seq{ constructor_words.begin(), constructor_words.end() };
    std::array<std::uint32_t, 16> std_iter_words{};
    std::array<std::uint32_t, 16> bl_iter_words{};
    std_iter_seq.generate(std_iter_words.begin(), std_iter_words.end());
    bl_iter_seq.generate(bl_iter_words.begin(), bl_iter_words.end());
    REQUIRE(bl_iter_words == std_iter_words);

    std::seed_seq std_engine_seq{ 8u, 6u, 7u, 5u, 3u, 0u, 9u };
    bl::seed_seq bl_engine_seq{ 8u, 6u, 7u, 5u, 3u, 0u, 9u };
    std::mt19937 std_rng{ std_engine_seq };
    bl::mt19937 bl_rng{ bl_engine_seq };
    for (int index = 0; index < 1000; ++index)
        REQUIRE(bl_rng() == std_rng());

    std::seed_seq std_engine_seq64{ 11u, 22u, 33u, 44u, 55u, 66u };
    bl::seed_seq bl_engine_seq64{ 11u, 22u, 33u, 44u, 55u, 66u };
    std::mt19937_64 std_rng64{ std_engine_seq64 };
    bl::mt19937_64 bl_rng64{ bl_engine_seq64 };
    for (int index = 0; index < 1000; ++index)
        REQUIRE(bl_rng64() == std_rng64());
}

TEST_CASE("bl random engines round-trip through streams", "[fltx][random][f128][constexpr]")
{
    bl::mt19937_64 rng{ 42u };
    rng.discard(37);

    std::stringstream stream;
    stream << rng;

    bl::mt19937_64 restored;
    stream >> restored;

    REQUIRE(restored == rng);
    for (int index = 0; index < 32; ++index)
        REQUIRE(restored() == rng());

    bl::mt19937 discarded{ 123u };
    bl::mt19937 stepped{ 123u };
    discarded.discard(100);
    for (int index = 0; index < 100; ++index)
        static_cast<void>(stepped());
    REQUIRE(discarded == stepped);

    std::stringstream bad_stream;
    bad_stream << (bl::mt19937::state_size + 1u);
    bl::mt19937 bad_target;
    bad_stream >> bad_target;
    REQUIRE(bad_stream.fail());
}

TEST_CASE("bl uniform_int_distribution is deterministic and bounded", "[fltx][random][f128][constexpr]")
{
    bl::mt19937_64 left_rng{ 42u };
    bl::mt19937_64 right_rng{ 42u };
    bl::uniform_int_distribution<int> dist{ -128, 513 };

    for (int index = 0; index < 256; ++index)
    {
        const int left = dist(left_rng);
        const int right = dist(right_rng);
        REQUIRE(left == right);
        REQUIRE(left >= -128);
        REQUIRE(left <= 513);
    }

    bl::uniform_int_distribution<std::int64_t> full_range{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max()
    };
    const std::int64_t full_sample = full_range(left_rng);
    (void)full_sample;
}

TEST_CASE("bl uniform_int_distribution exposes standard-shaped params", "[fltx][random][f128][constexpr]")
{
    using distribution = bl::uniform_int_distribution<int>;

    constexpr distribution::param_type first{ -7, 13 };
    constexpr distribution::param_type second{ -7, 13 };
    static_assert(first == second);
    static_assert(first.a() == -7);
    static_assert(first.b() == 13);

    distribution dist{ first };
    REQUIRE(dist.a() == -7);
    REQUIRE(dist.b() == 13);
    REQUIRE(dist.min() == -7);
    REQUIRE(dist.max() == 13);
    REQUIRE(dist.param() == first);

    dist.param(distribution::param_type{ 10, 12 });
    REQUIRE(dist.param().a() == 10);
    REQUIRE(dist.param().b() == 12);
    dist.reset();

    bl::mt19937 rng{ 321u };
    const int override_sample = dist(rng, distribution::param_type{ -2, 2 });
    REQUIRE(override_sample >= -2);
    REQUIRE(override_sample <= 2);
    REQUIRE(dist.a() == 10);
    REQUIRE(dist.b() == 12);

    bl::uniform_int_distribution<unsigned int> one_value{ 7u, 7u };
    for (int index = 0; index < 8; ++index)
        REQUIRE(one_value(rng) == 7u);

    bl::uniform_int_distribution<unsigned int> unsigned_dist{ 0u, 11u };
    const unsigned int unsigned_sample = unsigned_dist(rng);
    REQUIRE(unsigned_sample <= 11u);
}

TEST_CASE("bl integer and canonical generation handle non-power-of-two URBG ranges", "[fltx][random][f128][constexpr]")
{
    modulo_six_urbg canonical_rng{};
    for (int index = 0; index < 16; ++index)
    {
        const double sample = bl::generate_canonical<double, 4>(canonical_rng);
        REQUIRE(sample >= 0.0);
        REQUIRE(sample < 1.0);
    }

    modulo_six_urbg int_rng{};
    bl::uniform_int_distribution<int> dist{ 0, 3 };
    for (int index = 0; index < 16; ++index)
    {
        const int sample = dist(int_rng);
        REQUIRE(sample >= 0);
        REQUIRE(sample <= 3);
    }
}

TEST_CASE("bl f128 uniform_real_distribution is deterministic and bounded", "[fltx][random][f128][constexpr]")
{
    bl::mt19937_64 left_rng{ 42u };
    bl::mt19937_64 right_rng{ 42u };
    bl::uniform_real_distribution<bl::f128> dist{ bl::f128{ -1.25 }, bl::f128{ 2.5 } };

    auto& out = Catch::cout();
    bl::test::random_print::ostream_state_guard out_state{ out };
    out << "\n[bl::uniform_real_distribution<bl::f128> samples]\n"
        << "seed = 42, range = [-1.25, 2.5), precision = "
        << std::numeric_limits<bl::f128>::max_digits10 << "\n";

    std::vector<std::string> samples;
    samples.reserve(128);
    for (int index = 0; index < 128; ++index)
    {
        const bl::f128 left = dist(left_rng);
        const bl::f128 right = dist(right_rng);

        samples.push_back(bl::test::random_print::to_max_digits_string(left));

        REQUIRE(left == right);
        REQUIRE(left >= bl::f128{ -1.25 });
        REQUIRE(left < bl::f128{ 2.5 });
    }
    bl::test::random_print::print_aligned_samples(out, samples);
}

TEST_CASE("bl f128 uniform_real_distribution exposes standard-shaped params", "[fltx][random][f128][constexpr]")
{
    using distribution = bl::uniform_real_distribution<bl::f128>;

    constexpr distribution::param_type first{ bl::f128{ 1.0 }, bl::f128{ 2.0 } };
    constexpr distribution::param_type second{ bl::f128{ 1.0 }, bl::f128{ 2.0 } };
    static_assert(first == second);

    distribution dist{ first };
    REQUIRE(dist.a() == bl::f128{ 1.0 });
    REQUIRE(dist.b() == bl::f128{ 2.0 });
    REQUIRE(dist.min() == bl::f128{ 1.0 });
    REQUIRE(dist.max() == bl::f128{ 2.0 });

    dist.param(distribution::param_type{ bl::f128{ -4.0 }, bl::f128{ -3.0 } });
    REQUIRE(dist.param().a() == bl::f128{ -4.0 });
    REQUIRE(dist.param().b() == bl::f128{ -3.0 });

    bl::mt19937_64 rng{ 7u };
    const bl::f128 override_sample =
        dist(rng, distribution::param_type{ bl::f128{ 10.0 }, bl::f128{ 12.0 } });
    REQUIRE(override_sample >= bl::f128{ 10.0 });
    REQUIRE(override_sample < bl::f128{ 12.0 });
    REQUIRE(dist.a() == bl::f128{ -4.0 });
    REQUIRE(dist.b() == bl::f128{ -3.0 });

    dist.reset();
    REQUIRE(dist.param().a() == bl::f128{ -4.0 });
}

TEST_CASE("bl f128 real distributions are constexpr-capable and bounded", "[fltx][random][f128][constexpr]")
{
    bl::mt19937_64 rng{ 123u };

    bl::exponential_distribution<bl::f128> exponential{ bl::f128{ 0.75 } };
    const bl::f128 exponential_sample = exponential(rng);
    REQUIRE(exponential_sample >= bl::f128{ 0.0 });

    bl::normal_distribution<bl::f128> normal{ bl::f128{ 1.0 }, bl::f128{ 0.5 } };
    const bl::f128 normal_sample = normal(rng);
    REQUIRE(normal_sample == normal_sample);

    bl::lognormal_distribution<bl::f128> lognormal{ bl::f128{ 0.0 }, bl::f128{ 0.5 } };
    const bl::f128 lognormal_sample = lognormal(rng);
    REQUIRE(lognormal_sample > bl::f128{ 0.0 });
}

TEST_CASE("bl real distributions expose standard-shaped params and cached state", "[fltx][random][f128][constexpr]")
{
    using exponential = bl::exponential_distribution<double>;
    using normal = bl::normal_distribution<double>;
    using lognormal = bl::lognormal_distribution<double>;

    constexpr exponential::param_type exp_params{ 1.5 };
    constexpr normal::param_type normal_params{ -1.0, 0.25 };
    constexpr lognormal::param_type lognormal_params{ 0.1, 0.75 };
    static_assert(exp_params == exponential::param_type{ 1.5 });
    static_assert(normal_params == normal::param_type{ -1.0, 0.25 });
    static_assert(lognormal_params == lognormal::param_type{ 0.1, 0.75 });

    bl::mt19937_64 rng{ 777u };

    exponential exp_dist{ exp_params };
    REQUIRE(exp_dist.lambda() == 1.5);
    REQUIRE(exp_dist.min() == 0.0);
    REQUIRE(exp_dist.max() == std::numeric_limits<double>::max());
    REQUIRE(exp_dist.param() == exp_params);
    const double exp_override = exp_dist(rng, exponential::param_type{ 0.5 });
    REQUIRE(exp_override >= 0.0);
    REQUIRE(exp_dist.lambda() == 1.5);
    exp_dist.param(exponential::param_type{ 2.0 });
    REQUIRE(exp_dist.lambda() == 2.0);
    exp_dist.reset();

    normal normal_dist{ normal_params };
    normal normal_fresh{ normal_params };
    REQUIRE(normal_dist.mean() == -1.0);
    REQUIRE(normal_dist.stddev() == 0.25);
    REQUIRE(normal_dist.min() == std::numeric_limits<double>::lowest());
    REQUIRE(normal_dist.max() == std::numeric_limits<double>::max());
    REQUIRE(normal_dist.param() == normal_params);
    const double normal_override = normal_dist(rng, normal::param_type{ 10.0, 2.0 });
    REQUIRE(normal_override == normal_override);
    REQUIRE(normal_dist.mean() == -1.0);
    REQUIRE(normal_dist.stddev() == 0.25);
    REQUIRE(normal_dist != normal_fresh);
    normal_dist.reset();
    REQUIRE(normal_dist == normal_fresh);
    normal_dist.param(normal::param_type{ 3.0, 4.0 });
    REQUIRE(normal_dist.mean() == 3.0);
    REQUIRE(normal_dist.stddev() == 4.0);

    lognormal lognormal_dist{ lognormal_params };
    lognormal lognormal_fresh{ lognormal_params };
    REQUIRE(lognormal_dist.m() == 0.1);
    REQUIRE(lognormal_dist.s() == 0.75);
    REQUIRE(lognormal_dist.min() == 0.0);
    REQUIRE(lognormal_dist.max() == std::numeric_limits<double>::max());
    REQUIRE(lognormal_dist.param() == lognormal_params);
    const double lognormal_override = lognormal_dist(rng, lognormal::param_type{ 0.2, 0.5 });
    REQUIRE(lognormal_override > 0.0);
    REQUIRE(lognormal_dist.m() == 0.1);
    REQUIRE(lognormal_dist.s() == 0.75);
    REQUIRE(lognormal_dist != lognormal_fresh);
    lognormal_dist.reset();
    REQUIRE(lognormal_dist == lognormal_fresh);
    lognormal_dist.param(lognormal::param_type{ -0.5, 1.25 });
    REQUIRE(lognormal_dist.m() == -0.5);
    REQUIRE(lognormal_dist.s() == 1.25);
}

TEST_CASE("bl real distributions support the standard floating types at runtime", "[fltx][random]")
{
    bl::mt19937_64 rng{ 1001u };

    bl::exponential_distribution<float> exponential{ 1.5f };
    const float exponential_sample = exponential(rng);
    REQUIRE(exponential_sample >= 0.0f);

    bl::normal_distribution<double> normal{ 1.0, 2.0 };
    const double normal_sample = normal(rng);
    REQUIRE(normal_sample == normal_sample);

    bl::lognormal_distribution<long double> lognormal{ 0.0L, 0.5L };
    const long double lognormal_sample = lognormal(rng);
    REQUIRE(lognormal_sample > 0.0L);
}

TEST_CASE("bl distributions round-trip through streams", "[fltx][random][f128][constexpr]")
{
    bl::uniform_int_distribution<int> int_dist{ -9, 15 };
    std::stringstream int_stream;
    int_stream << int_dist;
    bl::uniform_int_distribution<int> restored_int;
    int_stream >> restored_int;
    REQUIRE(restored_int == int_dist);

    bl::uniform_real_distribution<bl::f128> real_dist{ bl::f128{ -1.5 }, bl::f128{ 2.25 } };
    std::stringstream real_stream;
    real_stream << real_dist;
    bl::uniform_real_distribution<bl::f128> restored_real;
    real_stream >> restored_real;
    REQUIRE(restored_real.a() == real_dist.a());
    REQUIRE(restored_real.b() == real_dist.b());

    bl::mt19937_64 rng{ 99u };
    bl::normal_distribution<bl::f128> normal{ bl::f128{ 1.0 }, bl::f128{ 2.0 } };
    (void)normal(rng);
    std::stringstream normal_stream;
    normal_stream << normal;
    bl::normal_distribution<bl::f128> restored_normal;
    normal_stream >> restored_normal;
    REQUIRE(restored_normal == normal);

    bl::exponential_distribution<bl::f128> exponential{ bl::f128{ 1.75 } };
    std::stringstream exponential_stream;
    exponential_stream << exponential;
    bl::exponential_distribution<bl::f128> restored_exponential;
    exponential_stream >> restored_exponential;
    REQUIRE(restored_exponential == exponential);

    bl::lognormal_distribution<bl::f128> lognormal{ bl::f128{ 0.25 }, bl::f128{ 0.5 } };
    (void)lognormal(rng);
    std::stringstream lognormal_stream;
    lognormal_stream << lognormal;
    bl::lognormal_distribution<bl::f128> restored_lognormal;
    lognormal_stream >> restored_lognormal;
    REQUIRE(restored_lognormal == lognormal);

    std::stringstream bad_real_stream;
    bad_real_stream << "not-a-number";
    bl::uniform_real_distribution<bl::f128> bad_real_target{ bl::f128{ 4.0 }, bl::f128{ 5.0 } };
    bad_real_stream >> bad_real_target;
    REQUIRE(bad_real_stream.fail());
    REQUIRE(bad_real_target.a() == bl::f128{ 4.0 });
    REQUIRE(bad_real_target.b() == bl::f128{ 5.0 });
}
