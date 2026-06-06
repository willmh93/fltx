#include "metrics_case_output.h"

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <sstream>
#include <string>
#include <string_view>
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

    bl::test::metrics::metrics_record make_filter_test_record(
        bl::test::metrics::precision_type precision,
        std::string_view operation,
        bl::test::metrics::domain_role role = bl::test::metrics::domain_role::primary)
    {
        bl::test::metrics::metrics_record record{};
        record.suite = {
            precision,
            { operation, operation },
            { role == bl::test::metrics::domain_role::primary ? "primary" : "mixed", role }
        };
        record.competitor_name = "reference";
        record.fltx_accuracy = { 100.0, 110.0, 16, 75.0 };
        record.competitor_accuracy = { 95.0, 105.0, 16, 70.0 };
        return record;
    }

    std::size_t count_substrings(std::string_view text, std::string_view pattern)
    {
        std::size_t count = 0;
        std::string_view::size_type pos = 0;
        while ((pos = text.find(pattern, pos)) != std::string_view::npos)
        {
            ++count;
            pos += pattern.size();
        }
        return count;
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
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[bench]");
    CHECK_FALSE(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));

    set_single_metrics_filter("[f128][bench]");
    CHECK_FALSE(bl::test::metrics::metrics_filter_is_complete_report_filter());
    CHECK(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f128));
    CHECK_FALSE(bl::test::metrics::metrics_should_generate_report_for(bl::test::metrics::precision_type::f256));
}

TEST_CASE("metrics partial phase reports merge operation rows", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    set_single_metrics_filter("[precision],[domain]");

    auto precision_record = make_filter_test_record(bl::test::metrics::precision_type::f128, "add");
    auto domain_record = precision_record;
    domain_record.fltx_accuracy.domain_score = 99.0;
    domain_record.competitor_accuracy.domain_score = 88.0;

    const std::vector<bl::test::metrics::metrics_case_report_entry> entries{
        { "f128 add precision", precision_record },
        { "f128 add domain", domain_record }
    };

    const std::vector<bl::test::metrics::metrics_record> records =
        bl::test::metrics::make_csv_report_records(entries, bl::test::metrics::precision_type::f128);

    REQUIRE(records.size() == 1);
    CHECK(records.front().suite.operation.name == "add");
    CHECK(records.front().fltx_accuracy.mean_bits == precision_record.fltx_accuracy.mean_bits);
    CHECK(records.front().fltx_accuracy.domain_score == domain_record.fltx_accuracy.domain_score);

    std::ostringstream out;
    bl::test::metrics::write_partial_metrics_case_reports(out, entries);
    const std::string text = out.str();
    CHECK(text.find("[ f128 primary metrics results ]") != std::string::npos);
    CHECK(count_substrings(text, " add ") == 1);
    CHECK(text.find("add precision") == std::string::npos);
    CHECK(text.find("add domain") == std::string::npos);
}

TEST_CASE("metrics realtime partial phase reports print when selected phases complete", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    set_single_metrics_filter("[precision],[domain]");
    bl::test::metrics::clear_realtime_metrics_console();

    auto precision_record = make_filter_test_record(bl::test::metrics::precision_type::f128, "add");
    auto domain_record = precision_record;
    domain_record.fltx_accuracy.domain_score = 99.0;

    std::vector<bl::test::metrics::metrics_case_report_entry> entries;
    std::ostringstream out;

    entries.push_back({ "f128 add precision", precision_record });
    bl::test::metrics::write_realtime_metrics_case_report(
        out,
        entries,
        entries.back().title,
        entries.back().record);
    CHECK(out.str().empty());

    entries.push_back({ "f128 add domain", domain_record });
    bl::test::metrics::write_realtime_metrics_case_report(
        out,
        entries,
        entries.back().title,
        entries.back().record);

    const std::string text = out.str();
    CHECK(text.find("[ f128 primary metrics results ]") != std::string::npos);
    CHECK(count_substrings(text, " add ") == 1);
    CHECK(text.find("add precision") == std::string::npos);
    CHECK(text.find("add domain") == std::string::npos);

    bl::test::metrics::clear_realtime_metrics_console();
}

TEST_CASE("metrics partial phase reports keep mixed workloads separate", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    set_single_metrics_filter("[precision],[domain]");

    const std::vector<bl::test::metrics::metrics_case_report_entry> entries{
        {
            "f128 add precision",
            make_filter_test_record(bl::test::metrics::precision_type::f128, "add")
        },
        {
            "f128 mixed workload metrics results",
            make_filter_test_record(
                bl::test::metrics::precision_type::f128,
                "mixed arithmetic",
                bl::test::metrics::domain_role::stress)
        }
    };

    std::ostringstream out;
    bl::test::metrics::write_partial_metrics_case_reports(out, entries);
    const std::string text = out.str();
    CHECK(text.find("[ f128 primary metrics results ]") != std::string::npos);
    CHECK(text.find("[ f128 mixed workload metrics results ]") != std::string::npos);
}

TEST_CASE("metrics bench-only console reports hide accuracy and domain columns", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    set_single_metrics_filter("[bench]");

    auto record = make_filter_test_record(
        bl::test::metrics::precision_type::f128,
        "mixed arithmetic",
        bl::test::metrics::domain_role::stress);
    record.fltx_benchmark = { 10.0, 1000 };
    record.competitor_benchmark = { 20.0, 1000 };

    std::ostringstream out;
    bl::test::metrics::write_metrics_case_report_group(
        out,
        "f128 primary metrics results",
        "f128 mixed workload metrics results",
        { record });

    const std::string text = out.str();
    CHECK(text.find("[ f128 mixed workload metrics results ]") != std::string::npos);
    CHECK(text.find("bench") != std::string::npos);
    CHECK(text.find("speed") != std::string::npos);
    CHECK(text.find("reference") != std::string::npos);
    CHECK(text.find("bits accurate") == std::string::npos);
    CHECK(text.find("domain") == std::string::npos);
    CHECK(text.find("Inf/") == std::string::npos);
    CHECK(text.find("NaN") == std::string::npos);
}

TEST_CASE("metrics exact comparison formatting avoids literal infinities", "[metrics][filters]")
{
    auto record = make_filter_test_record(bl::test::metrics::precision_type::f128, "round");
    record.fltx_accuracy.worst_bits = std::numeric_limits<double>::infinity();
    record.fltx_accuracy.mean_bits = std::numeric_limits<double>::infinity();

    CHECK(bl::test::metrics::format_metrics_number(record.fltx_accuracy.mean_bits, 1) == "exact");

    auto exact_reference = record.fltx_accuracy;
    CHECK(bl::test::metrics::precision_gap_bits(record.fltx_accuracy, exact_reference) == 0.0);

    std::ostringstream out;
    bl::test::metrics::write_csv_record(out, record);
    const std::string csv = out.str();
    CHECK(csv.find("inf") == std::string::npos);
    CHECK(csv.find("exact") != std::string::npos);
}
