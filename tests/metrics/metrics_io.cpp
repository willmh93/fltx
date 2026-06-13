#include <catch2/catch_test_macros.hpp>

#include "metrics_io.h"

namespace
{
    void write_io_record(std::string_view title, const bl::test::metrics::metrics_record& record)
    {
        bl::test::metrics::write_metrics_case_report(title, record);
        CHECK(record.fltx_accuracy.sample_count > 0);
        CHECK(record.fltx_benchmark.iteration_count > 0);
        if (record.competitor_supported)
        {
            CHECK(record.competitor_accuracy.sample_count > 0);
            CHECK(record.competitor_benchmark.iteration_count > 0);
        }
        for (const bl::test::metrics::competitor_result& competitor : record.extra_competitors)
        {
            if (!competitor.supported)
                continue;
            CHECK(competitor.accuracy.sample_count > 0);
            CHECK(competitor.benchmark.iteration_count > 0);
        }
    }
}

TEST_CASE("f128 IO metrics", "[metrics][precision][accuracy][domain][bench][fltx][f128][io]")
{
    using profile = bl::test::metrics::io_metrics::io_profile<bl::f128>;
    bl::test::metrics::io_metrics::emit_io_records<profile>(
        [](const bl::test::metrics::metrics_record& record)
        {
            write_io_record(profile::title, record);
        });
}

TEST_CASE("f256 IO metrics", "[metrics][precision][accuracy][domain][bench][fltx][f256][io]")
{
    using profile = bl::test::metrics::io_metrics::io_profile<bl::f256>;
    bl::test::metrics::io_metrics::emit_io_records<profile>(
        [](const bl::test::metrics::metrics_record& record)
        {
            write_io_record(profile::title, record);
        });
}
