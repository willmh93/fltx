#ifndef FLTX_TESTS_METRICS_CONFIG_INCLUDED
#define FLTX_TESTS_METRICS_CONFIG_INCLUDED

#include <cstddef>

namespace bl::test::metrics::config
{
    constexpr double sample_count_scale = 5.0;// 20.0; // precision/domain sample scale (fixed, no function-local scaling)
    constexpr double bench_iters_scale  = 2.0;  // benchmark iterations scale    (on top of function-local scaling)

    [[nodiscard]] constexpr std::size_t scale_count(std::size_t count, double scale) noexcept
    {
        const auto scaled = static_cast<std::size_t>(static_cast<double>(count) * scale);
        return scaled == 0 ? std::size_t{ 1 } : scaled;
    }

    [[nodiscard]] constexpr std::size_t scale_sample_count(std::size_t count) noexcept
    {
        return scale_count(count, sample_count_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_benchmark_iterations(std::size_t count) noexcept
    {
        return scale_count(count, bench_iters_scale);
    }

#ifdef NDEBUG
    constexpr std::size_t f128_primary_random_sample_count       = scale_sample_count(4096);
    constexpr std::size_t f128_primary_benchmark_min_iterations  = scale_benchmark_iterations(1000000);
    constexpr std::size_t f256_primary_random_sample_count       = scale_sample_count(2048);
    constexpr std::size_t f256_primary_benchmark_min_iterations  = scale_benchmark_iterations(200000);
    constexpr std::size_t primary_domain_random_sample_count     = scale_sample_count(4096);
#else
    constexpr std::size_t f128_primary_random_sample_count       = scale_sample_count(128);
    constexpr std::size_t f128_primary_benchmark_min_iterations  = scale_benchmark_iterations(20000);
    constexpr std::size_t f256_primary_random_sample_count       = scale_sample_count(128);
    constexpr std::size_t f256_primary_benchmark_min_iterations  = scale_benchmark_iterations(20000);
    constexpr std::size_t primary_domain_random_sample_count     = scale_sample_count(64);
#endif
}

#endif
