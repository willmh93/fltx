#include "metrics_case_output.h"

#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace bl::test::metrics
{
    namespace
    {
        void write_metrics_console_legend(std::ostream& out)
        {
            if (!metrics_verbose_enabled())
                return;

            out << "\n[metrics legend]\n"
                << "bits accurate       = estimated matching binary bits versus the MPFR oracle; exact sampled matches are shown as exact.\n"
                << "Inf/NaN             = whether the backend matches the std/libm oracle on a small Inf/NaN/signed-zero probe set.\n"
                << "domain score        = normal finite-domain magnitude quality: 50% mean sample score, 30% 1st-percentile sample score, 20% worst sample score.\n"
                << "sample score        = clamp(finite-domain bits / target bits, 0, 1); targets: f128=106 bits, f256=212 bits, reduced near expansion underflow.\n\n"

                << "preferred reference = preferred backend picked from faster comparable results.\n"
                << "cppdd               = boost::multiprecision::cpp_double_double\n"
                << "mpfr<64>            = boost::multiprecision::mpfr_float_backend<64>\n"
                << "dd_real             = qdpp double-double type\n"
                << "qd_real             = qdpp quad-double type\n\n";
        }
    }

    class metrics_case_report_listener final : public Catch::EventListenerBase
    {
    public:
        using Catch::EventListenerBase::EventListenerBase;

        static std::string getDescription()
        {
            return "prints consolidated metrics tables and writes complete metrics CSV reports";
        }

        void testRunStarting(const Catch::TestRunInfo&) override
        {
            clear_pending_metrics_case_reports();
            clear_realtime_metrics_console();
            clear_metrics_console_streamed_this_run();
            set_metrics_filter_arguments(m_config->getTestsOrTags());
            write_metrics_console_legend(Catch::cout());
        }

        void testRunEnded(const Catch::TestRunStats&) override
        {
            write_pending_metrics_case_reports(std::cout);
        }
    };
}

CATCH_REGISTER_LISTENER(bl::test::metrics::metrics_case_report_listener)
