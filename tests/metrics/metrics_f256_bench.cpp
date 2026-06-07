#include "metrics_f256_primary.h"
#include "metrics_f128_primary_cases.h"
#include "metrics_case_output.h"

using namespace bl::test::metrics::f256_primary;

#define FLTX_F256_UNARY_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return CALL(x); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_BINARY_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_binary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_BINARY_BOOL_BENCH(NAME, LABEL, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " LABEL " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ LABEL }; \
        const auto samples = SAMPLES(); \
        const auto record = run_binary_bool_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            LABEL, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " LABEL, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_TERNARY_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_ternary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y, const auto& z) { return CALL(x, y, z); }, \
            [](const auto& x, const auto& y, const auto& z) { return REF_CALL(x, y, z); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_UNARY_INT_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_int_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_UNARY_INTEGER_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_unary_integer_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return REF_CALL(x); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

#define FLTX_F256_FREXP_BENCH(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f256 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " benchmark", "[metrics][bench][fltx][f256][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("bench")) { SUCCEED("metrics benchmark phase not selected"); return; } \
        const scoped_benchmark_sample_count benchmark_samples{ #NAME }; \
        const auto samples = SAMPLES(); \
        const auto record = run_frexp_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            true, false, domain_ideal_bits, false); \
        bl::test::metrics::write_metrics_case_report("f256 primary benchmark: " #NAME, record); \
        check_benchmark_claim_is_meaningful(record); \
    }

FLTX_F128_PRIMARY_CASES(
    FLTX_F256_UNARY_BENCH,
    FLTX_F256_BINARY_BENCH,
    FLTX_F256_TERNARY_BENCH,
    FLTX_F256_UNARY_INT_BENCH,
    FLTX_F256_UNARY_INTEGER_BENCH,
    FLTX_F256_FREXP_BENCH)

FLTX_F128_PRIMARY_COMPARISON_CASES(FLTX_F256_BINARY_BOOL_BENCH)

#undef FLTX_F256_UNARY_BENCH
#undef FLTX_F256_BINARY_BENCH
#undef FLTX_F256_BINARY_BOOL_BENCH
#undef FLTX_F256_TERNARY_BENCH
#undef FLTX_F256_UNARY_INT_BENCH
#undef FLTX_F256_UNARY_INTEGER_BENCH
#undef FLTX_F256_FREXP_BENCH
