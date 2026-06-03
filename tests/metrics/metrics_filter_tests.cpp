#include "metrics_case_output.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

namespace
{
    class metrics_filter_restore
    {
    public:
        metrics_filter_restore()
            : filters_{ bl::test::metrics::metrics_filter_arguments() }
        {}

        ~metrics_filter_restore()
        {
            bl::test::metrics::set_metrics_filter_arguments(std::move(filters_));
        }

        metrics_filter_restore(const metrics_filter_restore&) = delete;
        metrics_filter_restore& operator=(const metrics_filter_restore&) = delete;

    private:
        std::vector<std::string> filters_;
    };

    void set_single_metrics_filter(std::string filter)
    {
        bl::test::metrics::set_metrics_filter_arguments({ std::move(filter) });
    }
}

TEST_CASE("metrics complete report filters are recognized exactly", "[metrics][filters]")
{
    const metrics_filter_restore restore;

    set_single_metrics_filter("[precision],[domain],[bench]");
    CHECK(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[domain],[bench],[accuracy]");
    CHECK(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[f128][precision],[f128][domain],[f128][bench]");
    CHECK(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[precision][f256],[domain][f256],[bench][f256]");
    CHECK(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[add][precision],[add][domain],[add][bench]");
    CHECK_FALSE(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[bench]");
    CHECK_FALSE(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));
}
