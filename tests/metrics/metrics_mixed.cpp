#include <catch2/catch_test_macros.hpp>

#include <fltx/f128_math.h>
#include <fltx/f256_math.h>

#include "metrics_case_output.h"
#include "metrics_mixed_workloads.h"

namespace
{
    template<class Profile>
    void write_mixed_record(std::string_view title, const bl::test::metrics::metrics_record& record)
    {
        bl::test::metrics::write_metrics_case_report(title, record);
        CHECK(record.fltx_accuracy.sample_count > 0);
        CHECK(record.competitor_accuracy.sample_count > 0);
        CHECK(record.fltx_benchmark.iteration_count > 0);
        CHECK(record.competitor_benchmark.iteration_count > 0);
    }
}

TEST_CASE("f128 workload 001 mixed arithmetic", "[metrics][fltx][f128][mixed][workload][arithmetic]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f128>;
    write_mixed_record<profile>(
        "f128 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_mixed_arithmetic_record<profile>(true));
}

TEST_CASE("f128 workload 002 affine transform", "[metrics][fltx][f128][mixed][workload][affine]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f128>;
    write_mixed_record<profile>(
        "f128 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_affine_record<profile>(true));
}

TEST_CASE("f128 workload 003 mandelbrot", "[metrics][fltx][f128][mixed][workload][mandelbrot]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f128>;
    write_mixed_record<profile>(
        "f128 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_mandelbrot_record<profile>(true));
}

TEST_CASE("f256 workload 001 mixed arithmetic", "[metrics][fltx][f256][mixed][workload][arithmetic]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f256>;
    write_mixed_record<profile>(
        "f256 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_mixed_arithmetic_record<profile>(true));
}

TEST_CASE("f256 workload 002 affine transform", "[metrics][fltx][f256][mixed][workload][affine]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f256>;
    write_mixed_record<profile>(
        "f256 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_affine_record<profile>(true));
}

TEST_CASE("f256 workload 003 mandelbrot", "[metrics][fltx][f256][mixed][workload][mandelbrot]")
{
    using profile = bl::test::metrics::mixed_workloads::mixed_profile<bl::f256>;
    write_mixed_record<profile>(
        "f256 mixed workload metrics results",
        bl::test::metrics::mixed_workloads::run_mandelbrot_record<profile>(true));
}
