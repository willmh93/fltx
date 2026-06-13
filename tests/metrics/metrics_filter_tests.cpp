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

    std::string strip_ansi_sequences(std::string_view text)
    {
        std::string out;
        out.reserve(text.size());
        for (std::size_t index = 0; index < text.size(); ++index)
        {
            if (text[index] == '\033' && index + 1 < text.size() && text[index + 1] == '[')
            {
                index += 2;
                while (index < text.size() &&
                       (text[index] < '@' || text[index] > '~'))
                {
                    ++index;
                }
                continue;
            }

            out.push_back(text[index]);
        }
        return out;
    }

    std::vector<std::size_t> separator_positions(std::string_view line)
    {
        std::vector<std::size_t> positions;
        for (std::size_t index = 0; index < line.size(); ++index)
        {
            if (line[index] == '|')
                positions.push_back(index);
        }
        return positions;
    }

    std::string find_line_containing(std::string_view text, std::string_view needle)
    {
        std::size_t start = 0;
        while (start < text.size())
        {
            const std::size_t end = text.find('\n', start);
            const std::string_view line =
                end == std::string_view::npos
                    ? text.substr(start)
                    : text.substr(start, end - start);
            if (line.find(needle) != std::string_view::npos)
                return std::string(line);
            if (end == std::string_view::npos)
                break;
            start = end + 1;
        }
        return {};
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

TEST_CASE("metrics unfiltered run enables every phase", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    bl::test::metrics::set_metrics_filter_arguments({});

    CHECK(bl::test::metrics::metrics_case_phase_enabled("precision"));
    CHECK(bl::test::metrics::metrics_case_phase_enabled("domain"));
    CHECK(bl::test::metrics::metrics_case_phase_enabled("bench"));
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

TEST_CASE("metrics realtime clear preserves run-level streamed marker", "[metrics][filters]")
{
    struct streamed_marker_restore
    {
        bool previous = bl::test::metrics::metrics_console_streamed_this_run();

        ~streamed_marker_restore()
        {
            bl::test::metrics::metrics_console_streamed_this_run_state() = previous;
        }
    } restore;

    bl::test::metrics::clear_metrics_console_streamed_this_run();
    bl::test::metrics::mark_metrics_console_streamed_this_run();
    bl::test::metrics::clear_realtime_metrics_console();

    CHECK(bl::test::metrics::metrics_console_streamed_this_run());
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

TEST_CASE("metrics console report keeps IO rows out of primary results", "[metrics][filters]")
{
    const metrics_filter_restore restore;
    set_single_metrics_filter("[precision],[domain]");

    auto math_record = make_filter_test_record(bl::test::metrics::precision_type::f128, "add");
    auto io_record = make_filter_test_record(bl::test::metrics::precision_type::f128, "parse(tiny, default)");

    std::ostringstream out;
    bl::test::metrics::write_metrics_case_report_group(
        out,
        "f128 primary metrics results",
        "f128 mixed workload metrics results",
        { math_record, io_record });

    const std::string text = out.str();
    CHECK(text.find("[ f128 primary metrics results ]") != std::string::npos);
    CHECK(text.find("[ f128 IO metrics results ]") != std::string::npos);

    const std::size_t primary_title = text.find("[ f128 primary metrics results ]");
    const std::size_t io_title = text.find("[ f128 IO metrics results ]");
    REQUIRE(primary_title != std::string::npos);
    REQUIRE(io_title != std::string::npos);
    CHECK(bl::test::metrics::metrics_csv_group(io_record) == "IO");

    const std::size_t add_row = text.find(" add ", primary_title);
    const std::size_t parse_row = text.find("parse(tiny, default)", primary_title);
    REQUIRE(add_row != std::string::npos);
    REQUIRE(parse_row != std::string::npos);
    CHECK(add_row < io_title);
    CHECK(parse_row > io_title);
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

TEST_CASE("metrics console report keeps wide values aligned", "[metrics][filters]")
{
    auto normal = make_filter_test_record(
        bl::test::metrics::precision_type::f128,
        "normal alignment row");
    normal.fltx_benchmark = { 10.0, 1000 };
    normal.competitor_benchmark = { 20.0, 1000 };

    auto wide = make_filter_test_record(
        bl::test::metrics::precision_type::f128,
        "wide alignment row");
    wide.fltx_accuracy = { 104.2, 106.3, 16, 99.0 };
    wide.competitor_accuracy = {
        std::numeric_limits<double>::infinity(),
        160.0,
        16,
        100.0
    };
    wide.fltx_benchmark = { 1.0, 1000 };
    wide.competitor_benchmark = { 123456.78, 1000 };

    std::ostringstream out;
    bl::test::metrics::write_console_report(
        out,
        "alignment metrics results",
        { normal, wide });

    const std::string text = strip_ansi_sequences(out.str());
    const std::string normal_line = find_line_containing(text, "normal alignment row");
    const std::string wide_line = find_line_containing(text, "wide alignment row");

    REQUIRE_FALSE(normal_line.empty());
    REQUIRE_FALSE(wide_line.empty());
    CHECK(separator_positions(wide_line) == separator_positions(normal_line));
}
