#ifndef FLTX_TESTS_METRICS_RECORDS_INCLUDED
#define FLTX_TESTS_METRICS_RECORDS_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "metrics_types.h"

namespace bl::test::metrics
{
    enum class special_correctness
    {
        unavailable,
        pass,
        fail
    };

    struct accuracy_result
    {
        double worst_bits = std::numeric_limits<double>::infinity();
        double mean_bits = std::numeric_limits<double>::infinity();
        std::size_t sample_count = 0;
        double domain_score = 0.0;
    };

    struct benchmark_result
    {
        double ns_per_iter = 0.0;
        std::size_t iteration_count = 0;
    };

    struct competitor_result
    {
        std::string_view name;
        bool supported = true;
        accuracy_result accuracy;
        benchmark_result benchmark;
        special_correctness special_values = special_correctness::unavailable;
    };

    struct metrics_record
    {
        suite_id suite;
        accuracy_result fltx_accuracy;
        special_correctness fltx_special_values = special_correctness::unavailable;
        std::string_view competitor_name = "comp";
        bool competitor_supported = true;
        accuracy_result competitor_accuracy;
        special_correctness competitor_special_values = special_correctness::unavailable;
        benchmark_result fltx_benchmark;
        benchmark_result competitor_benchmark;
        std::vector<competitor_result> extra_competitors;
    };

    [[nodiscard]] inline double domain_sample_score(double bits, double ideal_bits) noexcept
    {
        if (std::isnan(bits))
            return 0.0;
        if (std::isinf(bits))
            return bits > 0.0 ? 1.0 : 0.0;
        if (ideal_bits <= 0.0)
            return bits > 0.0 ? 1.0 : 0.0;

        return std::clamp(bits / ideal_bits, 0.0, 1.0);
    }

    [[nodiscard]] inline double domain_score(std::vector<double> sample_scores)
    {
        if (sample_scores.empty())
            return 0.0;

        double total = 0.0;
        double worst = 1.0;
        for (double score : sample_scores)
        {
            total += score;
            worst = std::min(worst, score);
        }

        std::sort(sample_scores.begin(), sample_scores.end());
        const std::size_t lower_tail_index =
            std::min<std::size_t>(
                sample_scores.size() - 1,
                static_cast<std::size_t>(static_cast<double>(sample_scores.size() - 1) * 0.01));
        const double mean = total / static_cast<double>(sample_scores.size());
        const double lower_tail = sample_scores[lower_tail_index];

        return 100.0 * (mean * 0.50 + lower_tail * 0.30 + worst * 0.20);
    }
}

#endif
