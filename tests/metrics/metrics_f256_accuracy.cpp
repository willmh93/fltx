#include "metrics_f256_primary.h"
#include "metrics_f128_primary_cases.h"
#include "metrics_case_output.h"

using namespace bl::test::metrics::f256_primary;

#define FLTX_F256_UNARY_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return CALL(x); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_BINARY_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_binary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_BINARY_BOOL_ACCURACY(NAME, LABEL, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " LABEL " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_binary_bool_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            LABEL, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " LABEL, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_TERNARY_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_ternary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y, const auto& z) { return CALL(x, y, z); }, \
            [](const auto& x, const auto& y, const auto& z) { return REF_CALL(x, y, z); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_UNARY_INT_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_int_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_UNARY_INTEGER_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_integer_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return REF_CALL(x); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

#define FLTX_F256_FREXP_ACCURACY(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " precision", "[metrics][precision][accuracy][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("precision")) { SUCCEED("metrics precision phase not selected"); return; } \
        const bool enforce_assertions = bl::test::metrics::metrics_case_assertions_enabled(); \
        const auto samples = SAMPLES(); \
        const auto record = run_frexp_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            false, enforce_assertions); \
        bl::test::metrics::write_metrics_case_report("f256 primary precision: " #NAME, record); \
        if (enforce_assertions) check_competitor_slack(record); \
    }

FLTX_F128_PRIMARY_CASES(
    FLTX_F256_UNARY_ACCURACY,
    FLTX_F256_BINARY_ACCURACY,
    FLTX_F256_TERNARY_ACCURACY,
    FLTX_F256_UNARY_INT_ACCURACY,
    FLTX_F256_UNARY_INTEGER_ACCURACY,
    FLTX_F256_FREXP_ACCURACY)

FLTX_F128_PRIMARY_COMPARISON_CASES(FLTX_F256_BINARY_BOOL_ACCURACY)

#undef FLTX_F256_UNARY_ACCURACY
#undef FLTX_F256_BINARY_ACCURACY
#undef FLTX_F256_BINARY_BOOL_ACCURACY
#undef FLTX_F256_TERNARY_ACCURACY
#undef FLTX_F256_UNARY_INT_ACCURACY
#undef FLTX_F256_UNARY_INTEGER_ACCURACY
#undef FLTX_F256_FREXP_ACCURACY
