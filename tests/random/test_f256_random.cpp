#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include <fltx/f256_comparison.h>
#include <fltx/f256_io.h>
#include <fltx/random.h>

#include "random_print_utils.h"

namespace
{
    [[nodiscard]] constexpr bl::f256 constexpr_unit_sample()
    {
        bl::mt19937_64 rng{ 0x123456789abcdef0ull };
        bl::uniform_real_distribution<bl::f256> dist{ bl::f256{ 0.0 }, bl::f256{ 1.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f256 constexpr_interval_sample()
    {
        bl::mt19937_64 rng{ 0x1020304050607080ull };
        bl::uniform_real_distribution<bl::f256> dist{ bl::f256{ -2.0 }, bl::f256{ 3.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f256 constexpr_canonical_sample()
    {
        bl::mt19937 rng{ 1234u };
        return bl::generate_canonical<bl::f256, std::numeric_limits<bl::f256>::digits>(rng);
    }

    [[nodiscard]] constexpr bl::f256 constexpr_exponential_sample()
    {
        bl::mt19937_64 rng{ 0x2468ace02468ace0ull };
        bl::exponential_distribution<bl::f256> dist{ bl::f256{ 1.25 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f256 constexpr_normal_sample()
    {
        bl::mt19937_64 rng{ 0x3141592653589793ull };
        bl::normal_distribution<bl::f256> dist{ bl::f256{ 0.5 }, bl::f256{ 2.0 } };
        return dist(rng);
    }

    [[nodiscard]] constexpr bl::f256 constexpr_lognormal_sample()
    {
        bl::mt19937_64 rng{ 0x2718281828459045ull };
        bl::lognormal_distribution<bl::f256> dist{ bl::f256{ 0.0 }, bl::f256{ 0.25 } };
        return dist(rng);
    }

    static_assert(std::is_same_v<bl::uniform_real_distribution<bl::f256>::result_type, bl::f256>);
    static_assert(constexpr_unit_sample() >= bl::f256{ 0.0 });
    static_assert(constexpr_unit_sample() < bl::f256{ 1.0 });
    static_assert(constexpr_interval_sample() >= bl::f256{ -2.0 });
    static_assert(constexpr_interval_sample() < bl::f256{ 3.0 });
    static_assert(constexpr_canonical_sample() >= bl::f256{ 0.0 });
    static_assert(constexpr_canonical_sample() < bl::f256{ 1.0 });
    static_assert(constexpr_exponential_sample() >= bl::f256{ 0.0 });
    static_assert(constexpr_normal_sample() == constexpr_normal_sample());
    static_assert(constexpr_lognormal_sample() > bl::f256{ 0.0 });

}

TEST_CASE("bl f256 uniform_real_distribution is deterministic and bounded", "[fltx][random][f256][constexpr]")
{
    bl::mt19937_64 left_rng{ 42u };
    bl::mt19937_64 right_rng{ 42u };
    bl::uniform_real_distribution<bl::f256> dist{ bl::f256{ -1.25 }, bl::f256{ 2.5 } };

    auto& out = Catch::cout();
    bl::test::random_print::ostream_state_guard out_state{ out };
    out << "\n[bl::uniform_real_distribution<bl::f256> samples]\n"
        << "seed = 42, range = [-1.25, 2.5), precision = "
        << std::numeric_limits<bl::f256>::max_digits10 << "\n";

    std::vector<std::string> samples;
    samples.reserve(64);
    for (int index = 0; index < 64; ++index)
    {
        const bl::f256 left = dist(left_rng);
        const bl::f256 right = dist(right_rng);

        samples.push_back(bl::test::random_print::to_max_digits_string(left));

        REQUIRE(left == right);
        REQUIRE(left >= bl::f256{ -1.25 });
        REQUIRE(left < bl::f256{ 2.5 });
    }
    bl::test::random_print::print_aligned_samples(out, samples);
}

TEST_CASE("bl f256 uniform_real_distribution exposes standard-shaped params", "[fltx][random][f256][constexpr]")
{
    using distribution = bl::uniform_real_distribution<bl::f256>;

    constexpr distribution::param_type first{ bl::f256{ 1.0 }, bl::f256{ 2.0 } };
    constexpr distribution::param_type second{ bl::f256{ 1.0 }, bl::f256{ 2.0 } };
    static_assert(first == second);

    distribution dist{ first };
    REQUIRE(dist.a() == bl::f256{ 1.0 });
    REQUIRE(dist.b() == bl::f256{ 2.0 });
    REQUIRE(dist.min() == bl::f256{ 1.0 });
    REQUIRE(dist.max() == bl::f256{ 2.0 });

    dist.param(distribution::param_type{ bl::f256{ -4.0 }, bl::f256{ -3.0 } });
    REQUIRE(dist.param().a() == bl::f256{ -4.0 });
    REQUIRE(dist.param().b() == bl::f256{ -3.0 });
}

TEST_CASE("bl f256 real distributions are constexpr-capable and bounded", "[fltx][random][f256][constexpr]")
{
    bl::mt19937_64 rng{ 123u };

    bl::exponential_distribution<bl::f256> exponential{ bl::f256{ 0.75 } };
    const bl::f256 exponential_sample = exponential(rng);
    REQUIRE(exponential_sample >= bl::f256{ 0.0 });

    bl::normal_distribution<bl::f256> normal{ bl::f256{ 1.0 }, bl::f256{ 0.5 } };
    const bl::f256 normal_sample = normal(rng);
    REQUIRE(normal_sample == normal_sample);

    bl::lognormal_distribution<bl::f256> lognormal{ bl::f256{ 0.0 }, bl::f256{ 0.5 } };
    const bl::f256 lognormal_sample = lognormal(rng);
    REQUIRE(lognormal_sample > bl::f256{ 0.0 });
}

TEST_CASE("bl f256 real distributions expose standard-shaped params and cached state", "[fltx][random][f256][constexpr]")
{
    using exponential = bl::exponential_distribution<bl::f256>;
    using normal = bl::normal_distribution<bl::f256>;
    using lognormal = bl::lognormal_distribution<bl::f256>;

    constexpr exponential::param_type exp_params{ bl::f256{ 1.5 } };
    constexpr normal::param_type normal_params{ bl::f256{ -1.0 }, bl::f256{ 0.25 } };
    constexpr lognormal::param_type lognormal_params{ bl::f256{ 0.1 }, bl::f256{ 0.75 } };
    static_assert(exp_params == exponential::param_type{ bl::f256{ 1.5 } });
    static_assert(normal_params == normal::param_type{ bl::f256{ -1.0 }, bl::f256{ 0.25 } });
    static_assert(lognormal_params == lognormal::param_type{ bl::f256{ 0.1 }, bl::f256{ 0.75 } });

    bl::mt19937_64 rng{ 777u };

    exponential exp_dist{ exp_params };
    REQUIRE(exp_dist.lambda() == bl::f256{ 1.5 });
    REQUIRE(exp_dist.min() == bl::f256{ 0.0 });
    REQUIRE(exp_dist.max() == std::numeric_limits<bl::f256>::max());
    const bl::f256 exp_override = exp_dist(rng, exponential::param_type{ bl::f256{ 0.5 } });
    REQUIRE(exp_override >= bl::f256{ 0.0 });
    REQUIRE(exp_dist.lambda() == bl::f256{ 1.5 });

    normal normal_dist{ normal_params };
    normal normal_fresh{ normal_params };
    REQUIRE(normal_dist.mean() == bl::f256{ -1.0 });
    REQUIRE(normal_dist.stddev() == bl::f256{ 0.25 });
    REQUIRE(normal_dist.min() == std::numeric_limits<bl::f256>::lowest());
    REQUIRE(normal_dist.max() == std::numeric_limits<bl::f256>::max());
    const bl::f256 normal_override =
        normal_dist(rng, normal::param_type{ bl::f256{ 10.0 }, bl::f256{ 2.0 } });
    REQUIRE(normal_override == normal_override);
    REQUIRE(normal_dist != normal_fresh);
    normal_dist.reset();
    REQUIRE(normal_dist == normal_fresh);

    lognormal lognormal_dist{ lognormal_params };
    lognormal lognormal_fresh{ lognormal_params };
    REQUIRE(lognormal_dist.m() == bl::f256{ 0.1 });
    REQUIRE(lognormal_dist.s() == bl::f256{ 0.75 });
    REQUIRE(lognormal_dist.min() == bl::f256{ 0.0 });
    REQUIRE(lognormal_dist.max() == std::numeric_limits<bl::f256>::max());
    const bl::f256 lognormal_override =
        lognormal_dist(rng, lognormal::param_type{ bl::f256{ 0.2 }, bl::f256{ 0.5 } });
    REQUIRE(lognormal_override > bl::f256{ 0.0 });
    REQUIRE(lognormal_dist != lognormal_fresh);
    lognormal_dist.reset();
    REQUIRE(lognormal_dist == lognormal_fresh);
}

TEST_CASE("bl f256 distributions round-trip through streams", "[fltx][random][f256][constexpr]")
{
    bl::uniform_real_distribution<bl::f256> real_dist{ bl::f256{ -1.5 }, bl::f256{ 2.25 } };
    std::stringstream real_stream;
    real_stream << real_dist;
    bl::uniform_real_distribution<bl::f256> restored_real;
    real_stream >> restored_real;
    REQUIRE(restored_real.a() == real_dist.a());
    REQUIRE(restored_real.b() == real_dist.b());

    bl::mt19937_64 rng{ 99u };
    bl::normal_distribution<bl::f256> normal{ bl::f256{ 1.0 }, bl::f256{ 2.0 } };
    (void)normal(rng);
    std::stringstream normal_stream;
    normal_stream << normal;
    bl::normal_distribution<bl::f256> restored_normal;
    normal_stream >> restored_normal;
    REQUIRE(restored_normal == normal);

    bl::exponential_distribution<bl::f256> exponential{ bl::f256{ 1.75 } };
    std::stringstream exponential_stream;
    exponential_stream << exponential;
    bl::exponential_distribution<bl::f256> restored_exponential;
    exponential_stream >> restored_exponential;
    REQUIRE(restored_exponential == exponential);

    bl::lognormal_distribution<bl::f256> lognormal{ bl::f256{ 0.25 }, bl::f256{ 0.5 } };
    (void)lognormal(rng);
    std::stringstream lognormal_stream;
    lognormal_stream << lognormal;
    bl::lognormal_distribution<bl::f256> restored_lognormal;
    lognormal_stream >> restored_lognormal;
    REQUIRE(restored_lognormal == lognormal);
}
