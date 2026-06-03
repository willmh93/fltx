#include "metrics_f128_primary.h"
#include "metrics_f128_primary_cases.h"
#include "metrics_case_output.h"

using namespace bl::test::metrics::f128_primary;

#define FLTX_F128_UNARY_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_unary_samples(#NAME); \
        const auto record = run_unary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return CALL(x); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_BINARY_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_binary_samples(#NAME); \
        const auto record = run_binary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_BINARY_BOOL_DOMAIN(NAME, LABEL, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " LABEL " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_binary_samples(LABEL); \
        const auto record = run_binary_bool_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            LABEL, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y) { return CALL(x, y); }, \
            [](const auto& x, const auto& y) { return REF_CALL(x, y); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " LABEL, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_TERNARY_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_ternary_samples(#NAME); \
        const auto record = run_ternary_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, const auto& y, const auto& z) { return CALL(x, y, z); }, \
            [](const auto& x, const auto& y, const auto& z) { return REF_CALL(x, y, z); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_UNARY_INT_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_unary_int_samples(#NAME); \
        const auto record = run_unary_int_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            [](const auto& x, int n) { return CALL(x, n); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_UNARY_INTEGER_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL, REF_CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_unary_samples(#NAME); \
        const auto record = run_unary_integer_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            [](const auto& x) { return REF_CALL(x); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

#define FLTX_F128_FREXP_DOMAIN(NAME, TAGS, REQUIRED_BITS, SAMPLES, CALL) \
    TEST_CASE("f128 primary " FLTX_PRIMARY_CASE_ORDER(NAME) " " #NAME " domain", "[metrics][domain][fltx][f128][primary]" TAGS) \
    { \
        if (!bl::test::metrics::metrics_case_phase_enabled("domain")) { SUCCEED("metrics domain phase not selected"); return; } \
        const auto samples = make_domain_unary_samples(#NAME); \
        const auto record = run_frexp_case<FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME)>( \
            #NAME, REQUIRED_BITS, samples, \
            [](const auto& x) { return CALL(x); }, \
            false, false); \
        bl::test::metrics::write_metrics_case_report("f128 primary domain: " #NAME, record); \
        CHECK(record.fltx_accuracy.sample_count > 0); \
    }

FLTX_F128_PRIMARY_CASES(
    FLTX_F128_UNARY_DOMAIN,
    FLTX_F128_BINARY_DOMAIN,
    FLTX_F128_TERNARY_DOMAIN,
    FLTX_F128_UNARY_INT_DOMAIN,
    FLTX_F128_UNARY_INTEGER_DOMAIN,
    FLTX_F128_FREXP_DOMAIN)

FLTX_F128_PRIMARY_COMPARISON_CASES(FLTX_F128_BINARY_BOOL_DOMAIN)

#undef FLTX_F128_UNARY_DOMAIN
#undef FLTX_F128_BINARY_DOMAIN
#undef FLTX_F128_BINARY_BOOL_DOMAIN
#undef FLTX_F128_TERNARY_DOMAIN
#undef FLTX_F128_UNARY_INT_DOMAIN
#undef FLTX_F128_UNARY_INTEGER_DOMAIN
#undef FLTX_F128_FREXP_DOMAIN
