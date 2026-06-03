#ifndef FLTX_TESTS_METRICS_TYPES_INCLUDED
#define FLTX_TESTS_METRICS_TYPES_INCLUDED

#include <string_view>

namespace bl::test::metrics
{
    enum class precision_type
    {
        f128,
        f256
    };

    enum class domain_role
    {
        primary,
        stress
    };

    struct operation_id
    {
        std::string_view name;
        std::string_view standard_name;
    };

    struct domain_id
    {
        std::string_view name;
        domain_role role = domain_role::primary;
    };

    struct suite_id
    {
        precision_type precision;
        operation_id operation;
        domain_id domain;
    };
}

#endif
