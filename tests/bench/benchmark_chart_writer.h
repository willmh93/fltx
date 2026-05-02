#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bl::bench
{
    struct benchmark_chart_entry
    {
        std::string group{};
        std::string label{};
        double candidate_ns_per_iter = 0.0;
        double reference_ns_per_iter = 0.0;
        double ratio = 0.0;
    };

    [[nodiscard]] inline double calculate_ratio(double candidate_ns_per_iter, double reference_ns_per_iter)
    {
        if (candidate_ns_per_iter <= 0.0)
            return 0.0;

        return reference_ns_per_iter / candidate_ns_per_iter;
    }

    [[nodiscard]] inline std::string escape_csv(const std::string& text_value)
    {
        bool needs_quotes = false;
        for (const char ch : text_value)
        {
            if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r')
            {
                needs_quotes = true;
                break;
            }
        }

        if (!needs_quotes)
            return text_value;

        std::string out;
        out.reserve(text_value.size() + 2);
        out.push_back('"');
        for (const char ch : text_value)
        {
            if (ch == '"')
                out += "\"\"";
            else
                out.push_back(ch);
        }
        out.push_back('"');
        return out;
    }

    [[nodiscard]] inline std::string escape_svg(std::string_view text_value)
    {
        std::string out;
        out.reserve(text_value.size());

        for (const char ch : text_value)
        {
            switch (ch)
            {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out.push_back(ch); break;
            }
        }

        return out;
    }

    [[nodiscard]] inline std::string format_decimal(double value, int precision)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << value;
        return out.str();
    }

    [[nodiscard]] inline std::string format_ns_per_iter(double ns_per_iter)
    {
        std::ostringstream out;

        if (ns_per_iter < 1.0)
            out << std::fixed << std::setprecision(2) << ns_per_iter;
        else
            out << std::fixed << std::setprecision(0) << ns_per_iter;

        out << "ns";
        return out.str();
    }

    [[nodiscard]] inline int benchmark_group_rank(std::string_view group)
    {
        if (group == "Arithmetic")
            return 0;
        if (group == "Rounding")
            return 1;
        if (group == "Remainders")
            return 2;
        if (group == "Floating-point utilities")
            return 3;
        if (group == "Roots & powers")
            return 4;
        if (group == "Exponentials")
            return 5;
        if (group == "Logarithms")
            return 6;
        if (group == "Trigonometric")
            return 7;
        if (group == "Hyperbolic")
            return 8;
        if (group == "Inverse hyperbolic")
            return 9;
        if (group == "Special functions")
            return 10;
        return 100;
    }

    class benchmark_chart_writer
    {
    public:
        benchmark_chart_writer(
            std::string candidate_name_value,
            std::string reference_name_value,
            std::string title_value,
            std::string csv_output_path_value,
            std::string svg_output_path_value,
            double visible_ratio_cap_value=10.0
        )
            : candidate_name(std::move(candidate_name_value)),
              reference_name(std::move(reference_name_value)),
              title(std::move(title_value)),
              csv_output_path(std::move(csv_output_path_value)),
              svg_output_path(std::move(svg_output_path_value)),
            visible_ratio_cap(visible_ratio_cap_value)
        {
        }

        benchmark_chart_writer(const benchmark_chart_writer&) = delete;
        benchmark_chart_writer& operator=(const benchmark_chart_writer&) = delete;

        ~benchmark_chart_writer() noexcept
        {
            try
            {
                write_outputs();
            }
            catch (...)
            {
            }
        }

        void record_result(
            const char* group,
            const char* label,
            double candidate_ns_per_iter,
            double reference_ns_per_iter)
        {
            benchmark_chart_entry entry{};
            entry.group = group;
            entry.label = label;
            entry.candidate_ns_per_iter = candidate_ns_per_iter;
            entry.reference_ns_per_iter = reference_ns_per_iter;
            entry.ratio = calculate_ratio(candidate_ns_per_iter, reference_ns_per_iter);

            const auto found = std::find_if(entries.begin(), entries.end(), [&](const benchmark_chart_entry& item)
            {
                return item.group == entry.group && item.label == entry.label;
            });

            if (found != entries.end())
                *found = std::move(entry);
            else
                entries.push_back(std::move(entry));
        }

        void write_outputs() const
        {
            if (entries.empty())
                return;

            if (const std::filesystem::path parent = std::filesystem::path(csv_output_path).parent_path(); !parent.empty())
                std::filesystem::create_directories(parent);
            if (const std::filesystem::path parent = std::filesystem::path(svg_output_path).parent_path(); !parent.empty())
                std::filesystem::create_directories(parent);

            const auto sorted_entries = make_sorted_entries();
            write_csv(sorted_entries);
            write_svg(sorted_entries);
        }

    private:
        struct grouped_layout_entry
        {
            bool is_group_heading = false;
            std::string text{};
            benchmark_chart_entry value{};
        };

        std::string candidate_name{};
        std::string reference_name{};
        std::string title{};
        std::string csv_output_path{};
        std::string svg_output_path{};
        std::vector<benchmark_chart_entry> entries{};
        double visible_ratio_cap = 10.0;

        [[nodiscard]] std::vector<benchmark_chart_entry> make_sorted_entries() const
        {
            std::vector<benchmark_chart_entry> sorted = entries;
            std::sort(sorted.begin(), sorted.end(), [](const benchmark_chart_entry& lhs, const benchmark_chart_entry& rhs)
            {
                const int lhs_group_rank = benchmark_group_rank(lhs.group);
                const int rhs_group_rank = benchmark_group_rank(rhs.group);
                if (lhs_group_rank != rhs_group_rank)
                    return lhs_group_rank < rhs_group_rank;

                if (lhs.ratio != rhs.ratio)
                    return lhs.ratio > rhs.ratio;

                return lhs.label < rhs.label;
            });
            return sorted;
        }

        [[nodiscard]] std::vector<grouped_layout_entry> make_layout_entries(const std::vector<benchmark_chart_entry>& sorted_entries) const
        {
            std::vector<grouped_layout_entry> layout_entries;
            layout_entries.reserve(sorted_entries.size() * 2);

            std::string current_group;
            for (const auto& entry : sorted_entries)
            {
                if (entry.group != current_group)
                {
                    current_group = entry.group;
                    grouped_layout_entry heading{};
                    heading.is_group_heading = true;
                    heading.text = current_group;
                    layout_entries.push_back(std::move(heading));
                }

                grouped_layout_entry value{};
                value.is_group_heading = false;
                value.value = entry;
                layout_entries.push_back(std::move(value));
            }

            return layout_entries;
        }

        void write_csv(const std::vector<benchmark_chart_entry>& sorted_entries) const
        {
            std::ofstream out(csv_output_path, std::ios::trunc);
            if (!out)
                return;

            out << "group,label," << escape_csv(candidate_name) << "_ns_per_iter,"
                << escape_csv(reference_name) << "_ns_per_iter,"
                << escape_csv(reference_name) << "_to_" << escape_csv(candidate_name) << "_ratio\n";

            for (const auto& item : sorted_entries)
            {
                out
                    << escape_csv(item.group) << ','
                    << escape_csv(item.label) << ','
                    << std::setprecision(17) << item.candidate_ns_per_iter << ','
                    << std::setprecision(17) << item.reference_ns_per_iter << ','
                    << std::setprecision(17) << item.ratio << '\n';
            }
        }

        void write_svg(const std::vector<benchmark_chart_entry>& sorted_entries) const
        {
            std::ofstream out(svg_output_path, std::ios::trunc);
            if (!out)
                return;

            //constexpr double visible_ratio_cap = 6.0;

            double max_ratio = 1.0;
            std::size_t longest_label_size = 0;
            for (const auto& item : sorted_entries)
            {
                max_ratio = std::max(max_ratio, item.ratio);
                longest_label_size = std::max(longest_label_size, item.label.size());
            }

            const auto layout_entries = make_layout_entries(sorted_entries);
            const bool has_capped_ratios = max_ratio > visible_ratio_cap;
            const double visible_max_ratio = has_capped_ratios ? visible_ratio_cap : max_ratio;
            const double padded_max_ratio = visible_max_ratio + std::max(0.2, visible_max_ratio * 0.08);
            const double tick_step = padded_max_ratio <= 1.5 ? 0.25 : (padded_max_ratio <= 3.5 ? 0.5 : 1.0);
            const double tick_max = has_capped_ratios ? visible_ratio_cap : std::ceil(padded_max_ratio / tick_step) * tick_step;

            constexpr int group_heading_height = 32;
            constexpr int group_separator_height = 8;
            constexpr int row_height = 34;
            constexpr int bar_height = 24;
            constexpr int top_margin = 92;
            constexpr int bottom_margin = 24;
            const int left_margin = std::max(250, static_cast<int>(longest_label_size * 11) + 0);
            const int right_margin = has_capped_ratios ? 224 : 160;
            //constexpr int plot_width = 1180;
            constexpr int plot_width = 500;
            const int width = left_margin + plot_width + right_margin;

            int content_height = 0;
            bool has_group_heading = false;
            for (const auto& item : layout_entries)
            {
                if (item.is_group_heading)
                {
                    if (has_group_heading)
                        content_height += group_separator_height;

                    has_group_heading = true;
                    content_height += group_heading_height;
                }
                else
                {
                    content_height += row_height;
                }
            }

            const int height = top_margin + bottom_margin + content_height;
            const int plot_left = left_margin;
            const int plot_right = left_margin + plot_width;
            const int plot_top = top_margin - 10;
            const int plot_bottom = height - bottom_margin + 6;

            const auto map_ratio_to_x = [&](double ratio)
            {
                return static_cast<double>(plot_left) + (ratio / tick_max) * static_cast<double>(plot_width);
            };

            out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
            out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
                << "\" viewBox=\"0 0 " << width << ' ' << height << "\">\n";
            out << "  <rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height << "\" fill=\"#f2f2f2\"/>\n";
            out << "  <text x=\"" << (width / 2) << "\" y=\"38\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"28\" font-weight=\"600\" fill=\"#222222\">"
                << escape_svg(title) << "</text>\n";

            for (double tick_value = 0.0; tick_value <= tick_max + 1e-9; tick_value += tick_step)
            {
                const double x = map_ratio_to_x(tick_value);
                out << "  <line x1=\"" << x << "\" y1=\"" << plot_top << "\" x2=\"" << x << "\" y2=\"" << plot_bottom
                    << "\" stroke=\"#dddddd\" stroke-width=\"1\"/>\n";
            }

            if (1.0 <= tick_max)
            {
                const double x = map_ratio_to_x(1.0);
                out << "  <line x1=\"" << x << "\" y1=\"" << plot_top << "\" x2=\"" << x << "\" y2=\"" << plot_bottom
                    << "\" stroke=\"#6da7d9\" stroke-width=\"2\" stroke-dasharray=\"6,6\"/>\n";
            }

            int y_cursor = plot_top;
            std::string current_group;
            for (const auto& item : layout_entries)
            {
                if (item.is_group_heading)
                {
                    if (!current_group.empty())
                    {
                        out << "  <line x1=\"" << (plot_left - 8) << "\" y1=\"" << y_cursor << "\" x2=\"" << plot_right
                            << "\" y2=\"" << y_cursor << "\" stroke=\"#cfcfcf\" stroke-width=\"1\"/>\n";
                        y_cursor += group_separator_height;
                    }

                    current_group = item.text;
                    out << "  <text x=\"" << (plot_left - 8) << "\" y=\"" << (y_cursor + 22)
                        << "\" text-anchor=\"end\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"20\" font-weight=\"700\" fill=\"#1f1f1f\">"
                        << escape_svg(item.text)
                        << "</text>\n";
                    y_cursor += group_heading_height;
                    continue;
                }

                const auto& value = item.value;
                const double y = static_cast<double>(y_cursor) + (row_height - bar_height) * 0.5;
                const bool is_capped = value.ratio > tick_max;
                const double visible_ratio = is_capped ? tick_max : value.ratio;
                const double x = map_ratio_to_x(visible_ratio);
                const char* fill = value.ratio >= 1.0 ? "#89d88a" : "#2c7fb8";

                out << "  <text x=\"" << (plot_left - 12) << "\" y=\"" << (y + bar_height * 0.5 + 6.0)
                    << "\" text-anchor=\"end\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" fill=\"#222222\">"
                    << escape_svg(value.label)
                    << "</text>\n";

                out << "  <rect x=\"" << plot_left << "\" y=\"" << y << "\" width=\"" << std::max(0.0, x - plot_left) << "\" height=\"" << bar_height
                    << "\" rx=\"2\" ry=\"2\" fill=\"" << fill << "\"/>\n";

                if (is_capped)
                {
                    const double marker_y_mid = y + bar_height * 0.5;
                    out << "  <polygon points=\""
                        << plot_right << ',' << y << ' '
                        << (plot_right + 18) << ',' << marker_y_mid << ' '
                        << plot_right << ',' << (y + bar_height)
                        << "\" fill=\"" << fill << "\"/>\n";
                }

                const double value_label_x = is_capped
                    ? static_cast<double>(plot_right + 26)
                    : std::min<double>(x + 10.0, static_cast<double>(plot_right - 8));
                const std::string value_label =
                    format_decimal(value.ratio, 2) + "x [" +
                    format_ns_per_iter(value.candidate_ns_per_iter) + "]";

                out << "  <text x=\"" << value_label_x
                    << "\" y=\"" << (y + bar_height * 0.5 + 5.5)
                    << "\" text-anchor=\"start\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"15\" font-weight=\"600\" fill=\"#444444\">"
                    << escape_svg(value_label)
                    << "</text>\n";

                y_cursor += row_height;
            }
            out << "</svg>\n";
        }
    };
}
