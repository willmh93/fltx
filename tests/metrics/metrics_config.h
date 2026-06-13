#ifndef FLTX_TESTS_METRICS_CONFIG_INCLUDED
#define FLTX_TESTS_METRICS_CONFIG_INCLUDED

#include <cstddef>

namespace bl::test::metrics::config
{
    constexpr double accuracy_sample_count_scale  = 0.1;    // precision sample scale (default 0.1, fixed, no function-local scaling)
    constexpr double domain_sample_count_scale    = 1.0;    // domain sample scale    (default 2.0, fixed, no function-local scaling)

    constexpr double benchmark_sample_count_scale = 2.0;   // global benchmark sample scale (default 20)
    constexpr double bench_iters_scale            = 2.0;   // global benchmark iteration scale (default 20)
    constexpr double mixed_sample_count_scale     = 0.25;   // mixed workload generated sample scale (default 0.25)
    constexpr double mixed_iters_scale            = 0.02;   // mixed workload benchmark iteration scale (default 0.02)

    [[nodiscard]] constexpr std::size_t scale_count(std::size_t count, double scale) noexcept
    {
        const auto scaled = static_cast<std::size_t>(static_cast<double>(count) * scale);
        return scaled == 0 ? std::size_t{ 1 } : scaled;
    }

    [[nodiscard]] constexpr std::size_t scale_accuracy_sample_count(std::size_t count) noexcept
    {
        return scale_count(count, accuracy_sample_count_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_domain_sample_count(std::size_t count) noexcept
    {
        return scale_count(count, domain_sample_count_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_benchmark_sample_count(std::size_t count) noexcept
    {
        return scale_count(count, benchmark_sample_count_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_benchmark_iterations(std::size_t count) noexcept
    {
        return scale_count(count, bench_iters_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_mixed_sample_count(std::size_t count) noexcept
    {
        return scale_count(count, mixed_sample_count_scale);
    }

    [[nodiscard]] constexpr std::size_t scale_mixed_iterations(std::size_t count) noexcept
    {
        return scale_count(count, mixed_iters_scale);
    }

#ifdef NDEBUG
    constexpr std::size_t f128_primary_accuracy_random_sample_count  = scale_accuracy_sample_count(4096);
    constexpr std::size_t f128_primary_benchmark_random_sample_count = scale_benchmark_sample_count(4096);
    constexpr std::size_t f128_primary_benchmark_min_iterations      = scale_benchmark_iterations(1000000);
    constexpr std::size_t f256_primary_accuracy_random_sample_count  = scale_accuracy_sample_count(2048);
    constexpr std::size_t f256_primary_benchmark_random_sample_count = scale_benchmark_sample_count(2048);
    constexpr std::size_t f256_primary_benchmark_min_iterations      = scale_benchmark_iterations(200000);
    constexpr std::size_t primary_domain_random_sample_count         = scale_domain_sample_count(4096);
#else
    constexpr std::size_t f128_primary_accuracy_random_sample_count  = scale_accuracy_sample_count(128);
    constexpr std::size_t f128_primary_benchmark_random_sample_count = scale_benchmark_sample_count(128);
    constexpr std::size_t f128_primary_benchmark_min_iterations      = scale_benchmark_iterations(20000);
    constexpr std::size_t f256_primary_accuracy_random_sample_count  = scale_accuracy_sample_count(128);
    constexpr std::size_t f256_primary_benchmark_random_sample_count = scale_benchmark_sample_count(128);
    constexpr std::size_t f256_primary_benchmark_min_iterations      = scale_benchmark_iterations(20000);
    constexpr std::size_t primary_domain_random_sample_count         = scale_domain_sample_count(64);
#endif
}

#endif
