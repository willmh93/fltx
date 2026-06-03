#ifndef FLTX_TESTS_METRICS_CASE_OUTPUT_INCLUDED
#define FLTX_TESTS_METRICS_CASE_OUTPUT_INCLUDED

#include <algorithm>
#include <catch2/internal/catch_stdstreams.hpp>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "metrics_report.h"

#ifndef VERBOSE_TESTS
#define VERBOSE_TESTS 1
#endif

namespace bl::test::metrics
{
    [[nodiscard]] inline bool metrics_verbose_enabled() noexcept
    {
        return VERBOSE_TESTS != 0;
    }

    [[nodiscard]] inline std::vector<std::string>& metrics_filter_arguments()
    {
        static std::vector<std::string> filters;
        return filters;
    }

    inline void set_metrics_filter_arguments(std::vector<std::string> filters)
    {
        metrics_filter_arguments() = std::move(filters);
    }

    [[nodiscard]] inline bool metrics_filter_has_tag(std::string_view tag)
    {
        std::string bracketed;
        bracketed.reserve(tag.size() + 2);
        bracketed += "[";
        bracketed += tag;
        bracketed += "]";

        for (const std::string& filter : metrics_filter_arguments())
        {
            if (filter.find(bracketed) != std::string::npos)
                return true;
        }
        return false;
    }

    [[nodiscard]] inline bool metrics_filter_has_explicit_phase()
    {
        return metrics_filter_has_tag("precision") ||
            metrics_filter_has_tag("accuracy") ||
            metrics_filter_has_tag("bench") ||
            metrics_filter_has_tag("domain");
    }

    [[nodiscard]] inline bool metrics_case_phase_enabled(std::string_view phase)
    {
        if (!metrics_filter_has_explicit_phase())
            return true;

        if (phase == "precision")
            return metrics_filter_has_tag("precision") || metrics_filter_has_tag("accuracy");
        if (phase == "bench")
            return metrics_filter_has_tag("bench");
        if (phase == "domain")
            return metrics_filter_has_tag("domain");

        return true;
    }

    [[nodiscard]] inline bool metrics_case_assertions_enabled()
    {
        return true;
    }

    [[nodiscard]] inline std::string normalized_metrics_filter(std::string_view filter)
    {
        std::string normalized;
        normalized.reserve(filter.size());
        for (char ch : filter)
        {
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n')
                normalized += ch;
        }
        return normalized;
    }

    [[nodiscard]] inline bool metrics_filter_is_exact_tag(std::string_view tag)
    {
        if (metrics_filter_arguments().size() != 1)
            return false;

        std::string expected;
        expected.reserve(tag.size() + 2);
        expected += "[";
        expected += tag;
        expected += "]";

        return normalized_metrics_filter(metrics_filter_arguments().front()) == expected;
    }

    [[nodiscard]] inline bool metrics_should_generate_report_for(precision_type precision)
    {
        if (metrics_filter_arguments().empty())
            return true;
        return precision == precision_type::f128
            ? metrics_filter_is_exact_tag("f128")
            : metrics_filter_is_exact_tag("f256");
    }

    struct metrics_case_report_entry
    {
        std::string title;
        metrics_record record;
    };

    [[nodiscard]] inline std::vector<metrics_case_report_entry>& pending_metrics_case_reports()
    {
        static std::vector<metrics_case_report_entry> entries;
        return entries;
    }

    [[nodiscard]] inline std::string_view metrics_case_phase(std::string_view title) noexcept
    {
        if (title.find("precision") != std::string_view::npos)
            return "precision";
        if (title.find("benchmark") != std::string_view::npos)
            return "bench";
        if (title.find("domain") != std::string_view::npos)
            return "domain";
        return {};
    }

    [[nodiscard]] inline int metrics_case_phase_rank(std::string_view title) noexcept
    {
        const std::string_view phase = metrics_case_phase(title);
        if (phase == "precision")
            return 0;
        if (phase == "bench")
            return 1;
        if (phase == "domain")
            return 2;
        return 3;
    }

    inline void clear_pending_metrics_case_reports()
    {
        pending_metrics_case_reports().clear();
    }

    struct metrics_complete_record_group
    {
        precision_type precision = precision_type::f128;
        std::string operation;
        const metrics_record* precision_record = nullptr;
        const metrics_record* benchmark_record = nullptr;
        const metrics_record* domain_record = nullptr;
    };

    [[nodiscard]] inline metrics_complete_record_group& find_or_add_complete_record_group(
        std::vector<metrics_complete_record_group>& groups,
        const metrics_record& record)
    {
        for (metrics_complete_record_group& group : groups)
        {
            if (group.precision == record.suite.precision &&
                group.operation == record.suite.operation.name)
            {
                return group;
            }
        }

        groups.push_back(metrics_complete_record_group{
            record.suite.precision,
            std::string{ record.suite.operation.name }
        });
        return groups.back();
    }

    inline void copy_extra_domain_scores(
        metrics_record& output,
        const metrics_record& domain_record)
    {
        for (competitor_result& output_competitor : output.extra_competitors)
        {
            if (const competitor_result* domain_competitor =
                    find_extra_competitor(domain_record, output_competitor.name))
            {
                output_competitor.supported =
                    output_competitor.supported && domain_competitor->supported;
                if (domain_competitor->supported)
                {
                    output_competitor.accuracy.domain_score =
                        domain_competitor->accuracy.domain_score;
                }
            }
        }
    }

    inline void copy_extra_benchmarks(
        metrics_record& output,
        const metrics_record& benchmark_record)
    {
        for (competitor_result& output_competitor : output.extra_competitors)
        {
            if (const competitor_result* benchmark_competitor =
                    find_extra_competitor(benchmark_record, output_competitor.name))
            {
                output_competitor.supported =
                    output_competitor.supported && benchmark_competitor->supported;
                if (benchmark_competitor->supported)
                    output_competitor.benchmark = benchmark_competitor->benchmark;
            }
        }
    }

    [[nodiscard]] inline std::vector<metrics_record> make_complete_report_records(
        const std::vector<metrics_case_report_entry>& entries,
        precision_type precision)
    {
        std::vector<metrics_complete_record_group> groups;
        std::vector<metrics_record> records;
        for (const metrics_case_report_entry& entry : entries)
        {
            if (entry.record.suite.precision != precision)
                continue;

            const std::string_view phase = metrics_case_phase(entry.title);
            if (phase.empty())
            {
                records.push_back(entry.record);
                continue;
            }

            metrics_complete_record_group& group =
                find_or_add_complete_record_group(groups, entry.record);
            if (phase == "precision")
                group.precision_record = &entry.record;
            else if (phase == "bench")
                group.benchmark_record = &entry.record;
            else if (phase == "domain")
                group.domain_record = &entry.record;
        }

        if (groups.empty())
            return records;

        records.reserve(records.size() + groups.size());
        for (const metrics_complete_record_group& group : groups)
        {
            if (group.precision_record == nullptr ||
                group.benchmark_record == nullptr ||
                group.domain_record == nullptr)
            {
                return {};
            }

            metrics_record record = *group.precision_record;
            record.fltx_accuracy.domain_score =
                group.domain_record->fltx_accuracy.domain_score;
            record.competitor_supported =
                record.competitor_supported &&
                group.domain_record->competitor_supported &&
                group.benchmark_record->competitor_supported;
            if (group.domain_record->competitor_supported)
            {
                record.competitor_accuracy.domain_score =
                    group.domain_record->competitor_accuracy.domain_score;
            }
            copy_extra_domain_scores(record, *group.domain_record);

            record.fltx_benchmark = group.benchmark_record->fltx_benchmark;
            if (group.benchmark_record->competitor_supported)
                record.competitor_benchmark = group.benchmark_record->competitor_benchmark;
            copy_extra_benchmarks(record, *group.benchmark_record);

            records.push_back(std::move(record));
        }

        std::sort(
            records.begin(),
            records.end(),
            metrics_csv_record_less);
        return records;
    }

    inline void write_automatic_metrics_report(
        std::ostream& out,
        const std::vector<metrics_case_report_entry>& entries,
        precision_type precision)
    {
        if (!metrics_should_generate_report_for(precision))
            return;

        std::vector<metrics_record> records = make_complete_report_records(entries, precision);
        if (records.empty())
            return;

        const std::string_view precision_name = precision == precision_type::f128 ? "f128" : "f256";
        const std::string output_path = metrics_output_path(precision_name, "", "csv");
        write_csv_report(output_path, records);

        if (metrics_verbose_enabled())
        {
            out << "\n[metrics report]\n"
                << precision_name << " csv = " << output_path << '\n';
        }
    }

    [[nodiscard]] inline std::size_t metrics_phase_entry_count_for_precision(
        const std::vector<metrics_case_report_entry>& entries,
        precision_type precision) noexcept
    {
        std::size_t count = 0;
        for (const metrics_case_report_entry& entry : entries)
        {
            if (entry.record.suite.precision == precision &&
                !metrics_case_phase(entry.title).empty())
            {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] inline std::size_t metrics_primary_record_count(
        const std::vector<metrics_record>& records) noexcept
    {
        std::size_t count = 0;
        for (const metrics_record& record : records)
        {
            if (record.suite.domain.role == domain_role::primary)
                ++count;
        }
        return count;
    }

    inline void write_complete_metrics_case_reports(
        std::ostream& out,
        const std::vector<metrics_record>& f128_records,
        const std::vector<metrics_record>& f256_records)
    {
        if (!f128_records.empty())
            write_console_report(out, "f128 primary metrics results", f128_records);
        if (!f256_records.empty())
            write_console_report(out, "f256 primary metrics results", f256_records);
    }

    [[nodiscard]] inline bool try_make_complete_report_record(
        const std::vector<metrics_case_report_entry>& entries,
        precision_type precision,
        std::string_view operation,
        metrics_record& output)
    {
        const metrics_record* precision_record = nullptr;
        const metrics_record* benchmark_record = nullptr;
        const metrics_record* domain_record = nullptr;

        for (const metrics_case_report_entry& entry : entries)
        {
            if (entry.record.suite.precision != precision ||
                entry.record.suite.operation.name != operation)
            {
                continue;
            }

            const std::string_view phase = metrics_case_phase(entry.title);
            if (phase == "precision")
                precision_record = &entry.record;
            else if (phase == "bench")
                benchmark_record = &entry.record;
            else if (phase == "domain")
                domain_record = &entry.record;
        }

        if (precision_record == nullptr ||
            benchmark_record == nullptr ||
            domain_record == nullptr)
        {
            return false;
        }

        output = *precision_record;
        output.fltx_accuracy.domain_score =
            domain_record->fltx_accuracy.domain_score;
        output.competitor_supported =
            output.competitor_supported &&
            domain_record->competitor_supported &&
            benchmark_record->competitor_supported;
        if (domain_record->competitor_supported)
        {
            output.competitor_accuracy.domain_score =
                domain_record->competitor_accuracy.domain_score;
        }
        copy_extra_domain_scores(output, *domain_record);

        output.fltx_benchmark = benchmark_record->fltx_benchmark;
        if (benchmark_record->competitor_supported)
            output.competitor_benchmark = benchmark_record->competitor_benchmark;
        copy_extra_benchmarks(output, *benchmark_record);
        return true;
    }

    [[nodiscard]] inline std::string metrics_realtime_complete_key(
        precision_type precision,
        std::string_view operation)
    {
        std::string key = std::string(to_string(precision));
        key += "|";
        key += operation;
        return key;
    }

    struct metrics_realtime_table
    {
        std::string key;
        std::string title;
        std::string primary_competitor;
        std::vector<std::string> extra_competitors;
        std::string fltx_backend;
        int operation_column_width = 10;
        std::unique_ptr<metrics_console_report_writer> writer;
    };

    struct metrics_realtime_console_state
    {
        std::vector<std::unique_ptr<metrics_realtime_table>> tables;
        std::vector<std::string> printed_complete_keys;
        std::string current_table_key;
        bool printed_any = false;
    };

    [[nodiscard]] inline metrics_realtime_console_state& realtime_metrics_console()
    {
        static metrics_realtime_console_state state;
        return state;
    }

    inline void clear_realtime_metrics_console()
    {
        realtime_metrics_console() = {};
    }

    [[nodiscard]] inline std::string metrics_realtime_table_key(
        std::string_view title,
        const metrics_record& record)
    {
        std::string key = std::string(title);
        key += "|";
        key += to_string(record.suite.precision);
        key += "|";
        key += record.competitor_name;
        for (const competitor_result& competitor : record.extra_competitors)
        {
            key += "|";
            key += competitor.name;
        }
        return key;
    }

    [[nodiscard]] inline std::vector<std::string_view> metrics_realtime_extra_views(
        const metrics_realtime_table& table)
    {
        std::vector<std::string_view> views;
        views.reserve(table.extra_competitors.size());
        for (const std::string& name : table.extra_competitors)
            views.push_back(name);
        return views;
    }

    inline void reset_realtime_metrics_writer(
        std::ostream& out,
        metrics_realtime_table& table)
    {
        const std::vector<std::string_view> extra_views =
            metrics_realtime_extra_views(table);
        table.writer = std::make_unique<metrics_console_report_writer>(
            out,
            table.title,
            table.primary_competitor,
            extra_views,
            table.fltx_backend,
            table.operation_column_width);
    }

    [[nodiscard]] inline metrics_realtime_table& find_or_add_realtime_metrics_table(
        std::ostream& out,
        std::string_view title,
        const metrics_record& record)
    {
        metrics_realtime_console_state& state = realtime_metrics_console();
        const std::string key = metrics_realtime_table_key(title, record);
        for (const std::unique_ptr<metrics_realtime_table>& table : state.tables)
        {
            if (table->key == key)
            {
                if (state.current_table_key != key)
                    reset_realtime_metrics_writer(out, *table);
                state.current_table_key = key;
                return *table;
            }
        }

        auto table = std::make_unique<metrics_realtime_table>();
        table->key = key;
        table->title = std::string(title);
        table->primary_competitor = std::string(record.competitor_name);
        table->fltx_backend = metrics_fltx_backend_name(record.suite.precision);
        table->operation_column_width = std::max(
            10,
            static_cast<int>(record.suite.operation.name.size()));
        table->extra_competitors.reserve(record.extra_competitors.size());
        for (const competitor_result& competitor : record.extra_competitors)
            table->extra_competitors.emplace_back(competitor.name);

        reset_realtime_metrics_writer(out, *table);
        state.current_table_key = key;

        state.tables.push_back(std::move(table));
        return *state.tables.back();
    }

    inline void write_realtime_metrics_record(
        std::ostream& out,
        std::string_view title,
        const metrics_record& record)
    {
        if (!metrics_verbose_enabled())
            return;

        metrics_realtime_table& table =
            find_or_add_realtime_metrics_table(out, title, record);
        table.writer->write_record(record);
        realtime_metrics_console().printed_any = true;
    }

    [[nodiscard]] inline bool metrics_complete_record_already_printed(
        precision_type precision,
        std::string_view operation)
    {
        const std::string key = metrics_realtime_complete_key(precision, operation);
        const auto& printed_keys = realtime_metrics_console().printed_complete_keys;
        return std::find(printed_keys.begin(), printed_keys.end(), key) != printed_keys.end();
    }

    inline void mark_metrics_complete_record_printed(
        precision_type precision,
        std::string_view operation)
    {
        realtime_metrics_console().printed_complete_keys.push_back(
            metrics_realtime_complete_key(precision, operation));
    }

    inline void write_realtime_metrics_case_report(
        std::ostream& out,
        const std::vector<metrics_case_report_entry>& entries,
        std::string_view title,
        const metrics_record& record)
    {
        if (!metrics_verbose_enabled())
            return;

        if (metrics_filter_has_explicit_phase())
        {
            write_realtime_metrics_record(out, "selected metrics results", record);
            return;
        }

        if (metrics_case_phase(title).empty())
        {
            write_realtime_metrics_record(out, title, record);
            return;
        }

        const precision_type precision = record.suite.precision;
        const std::string_view operation = record.suite.operation.name;
        if (metrics_complete_record_already_printed(precision, operation))
            return;

        metrics_record complete_record;
        if (!try_make_complete_report_record(entries, precision, operation, complete_record))
            return;

        const std::string primary_title = std::string(to_string(precision)) + " primary metrics results";
        write_realtime_metrics_record(out, primary_title, complete_record);
        mark_metrics_complete_record_printed(precision, operation);
    }

    inline void write_pending_metrics_case_reports(std::ostream& out)
    {
        auto& entries = pending_metrics_case_reports();
        if (entries.empty())
            return;

        bool has_precision = false;
        bool has_benchmark = false;
        bool has_domain = false;
        std::stable_sort(
            entries.begin(),
            entries.end(),
            [](const metrics_case_report_entry& lhs, const metrics_case_report_entry& rhs)
            {
                if (lhs.record.suite.operation.name != rhs.record.suite.operation.name)
                    return lhs.record.suite.operation.name < rhs.record.suite.operation.name;
                return metrics_case_phase_rank(lhs.title) < metrics_case_phase_rank(rhs.title);
            });

        for (const metrics_case_report_entry& entry : entries)
        {
            const std::string_view phase = metrics_case_phase(entry.title);
            has_precision = has_precision || phase == "precision";
            has_benchmark = has_benchmark || phase == "bench";
            has_domain = has_domain || phase == "domain";
        }

        const bool mixed_phases =
            (has_precision ? 1 : 0) +
            (has_benchmark ? 1 : 0) +
            (has_domain ? 1 : 0) > 1;

        const std::vector<metrics_record> complete_f128_records =
            make_complete_report_records(entries, precision_type::f128);
        const std::vector<metrics_record> complete_f256_records =
            make_complete_report_records(entries, precision_type::f256);
        const std::size_t f128_entry_count =
            metrics_phase_entry_count_for_precision(entries, precision_type::f128);
        const std::size_t f256_entry_count =
            metrics_phase_entry_count_for_precision(entries, precision_type::f256);
        const bool f128_entries_are_complete =
            f128_entry_count == 0 || metrics_primary_record_count(complete_f128_records) * 3 == f128_entry_count;
        const bool f256_entries_are_complete =
            f256_entry_count == 0 || metrics_primary_record_count(complete_f256_records) * 3 == f256_entry_count;
        const bool can_print_complete_records =
            (!complete_f128_records.empty() || !complete_f256_records.empty()) &&
            f128_entries_are_complete &&
            f256_entries_are_complete;

        if (metrics_verbose_enabled() &&
            !realtime_metrics_console().printed_any &&
            can_print_complete_records)
        {
            write_complete_metrics_case_reports(out, complete_f128_records, complete_f256_records);
        }
        else if (metrics_verbose_enabled() && !realtime_metrics_console().printed_any)
        {
            std::vector<std::string> operation_names;
            operation_names.reserve(entries.size());

            std::vector<metrics_record> records;
            records.reserve(entries.size());
            for (const metrics_case_report_entry& entry : entries)
            {
                records.push_back(entry.record);
                operation_names.emplace_back(records.back().suite.operation.name);
                if (mixed_phases)
                {
                    const std::string_view phase = metrics_case_phase(entry.title);
                    if (!phase.empty())
                    {
                        operation_names.back() += " ";
                        operation_names.back() += phase;
                    }
                }
                records.back().suite.operation.name = operation_names.back();
            }

            write_console_report(out, "selected metrics results", records);
        }

        write_automatic_metrics_report(out, entries, precision_type::f128);
        write_automatic_metrics_report(out, entries, precision_type::f256);
        entries.clear();
        clear_realtime_metrics_console();
    }

    inline void write_metrics_case_report(
        std::string_view title,
        const metrics_record& record)
    {
        pending_metrics_case_reports().push_back(
            metrics_case_report_entry{ std::string{ title }, record });
        write_realtime_metrics_case_report(
            Catch::cout(),
            pending_metrics_case_reports(),
            title,
            pending_metrics_case_reports().back().record);
    }

    inline void write_metrics_report_now(
        std::string_view title,
        const std::vector<metrics_record>& records)
    {
        if (!metrics_verbose_enabled())
            return;

        write_console_report(std::cout, title, records);
    }
}

#endif
