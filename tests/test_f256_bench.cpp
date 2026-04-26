#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <f256_math.h>
#include <f256_io.h>
#include "benchmark_chart_writer.h"

using namespace bl;

namespace
{
    constexpr unsigned mpfr_digits10 = std::numeric_limits<f256>::digits10;
    using mpfr_ref = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<mpfr_digits10>>;
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 5;
    constexpr std::size_t bucket_value_count = 8;
    constexpr std::size_t bucket_count = 3;

    struct bench_result
    {
        double total_ms = 0.0;
        double ns_per_iter = 0.0;
        std::int64_t iteration_count = 0;
    };

    struct comparison_result
    {
        bench_result f256{};
        bench_result mpfr{};
    };

    struct bucketed_comparison_result
    {
        comparison_result easy{};
        comparison_result medium{};
        comparison_result hard{};
        comparison_result typical{};
    };


    bl::bench::benchmark_chart_writer chart_writer{
        "f256",
        "mpfr",
        "f256 vs MPFR hard benchmark ratios",
        "benchmark_charts/f256_hard_ratios.csv",
        "benchmark_charts/f256_hard_ratios.svg"
    };

    template<typename Spec>
    struct bucket_array_set
    {
        std::array<Spec, bucket_value_count> easy{};
        std::array<Spec, bucket_value_count> medium{};
        std::array<Spec, bucket_value_count> hard{};
    };

    struct binary_value_spec
    {
        const char* lhs = "0";
        const char* rhs = "0";
    };

    struct recurrence_value_spec
    {
        const char* x = "0";
        const char* y = "0";
        const char* a = "0";
        const char* b = "0";
        const char* c = "1";
        const char* d = "1";
    };

    struct ldexp_value_spec
    {
        const char* value = "0";
        int exponent = 0;
    };

    volatile double benchmark_sink = 0.0;
    volatile std::int64_t benchmark_integer_sink = 0;

    void consume_result(const f256& value)
    {
        benchmark_sink += value.x0;
    }

    void consume_result(const mpfr_ref& value)
    {
        benchmark_sink += static_cast<double>(value);
    }

    void consume_result(std::int64_t value)
    {
        benchmark_integer_sink += value;
    }

    template<typename T, typename U>
    void consume_result(const std::pair<T, U>& value)
    {
        consume_result(value.first);
        consume_result(value.second);
    }

    template<typename T>
    [[nodiscard]] T parse_benchmark_value(const char* text);

    template<>
    [[nodiscard]] f256 parse_benchmark_value<f256>(const char* text)
    {
        return to_f256(text);
    }

    template<>
    [[nodiscard]] mpfr_ref parse_benchmark_value<mpfr_ref>(const char* text)
    {
        return mpfr_ref(text);
    }

    [[nodiscard]] bucket_array_set<const char*> generic_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-0.75",
            "0.125",
            "3.125",
            "2.5",
            "1.25",
            "0.875",
            "16.5",
            "0.03125"
        }};

        out.medium = {{
            "-0.7902875113057738885487120120465604260562530768860942235",
            "0.1577450190437525533903646435935788464125842783342441464",
            "3.1415926535897932384626433832795028841971",
            "2.7182818284590452353602874713526624977572",
            "1.0000000000000000000000000000000001",
            "0.9999999999999999999999999999999999",
            "12345678901234567890.125",
            "0.000000000000000000000000000000125"
        }};

        out.hard = {{
            "-1.0000000000000000000000000000000000000000000000000001",
            "0.3333333333333333333333333333333333333333333333333333",
            "3.1415926535897932384626433832795028841971693993751058209",
            "2.7182818284590452353602874713526624977572470936999595749",
            "1.0000000000000000000000000000000000000000000000000001",
            "0.9999999999999999999999999999999999999999999999999999",
            "123456789012345678901234567890.125",
            "0.0000000000000000000000000000000000000000000000000125"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> rounding_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-3.75",
            "-2.5",
            "-1.125",
            "-0.5",
            "0.5",
            "1.125",
            "2.5",
            "3.75"
        }};

        out.medium = {{
            "-2.0000000000000000000000000000000001",
            "-1.9999999999999999999999999999999999",
            "-0.5000000000000000000000000000000001",
            "0.4999999999999999999999999999999999",
            "0.9999999999999999999999999999999999",
            "1.0000000000000000000000000000000001",
            "12345678901234567890.5000000000000000000000000000000001",
            "-12345678901234567890.5000000000000000000000000000000001"
        }};

        out.hard = {{
            "-4503599627370496.5000000000000000000000000000000001",
            "-4503599627370495.9999999999999999999999999999999999",
            "-1.0000000000000000000000000000000001",
            "-0.9999999999999999999999999999999999",
            "0.9999999999999999999999999999999999",
            "1.0000000000000000000000000000000001",
            "4503599627370495.9999999999999999999999999999999999",
            "4503599627370496.5000000000000000000000000000000001"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> integer_rounding_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-8.75",
            "-4.5",
            "-1.125",
            "-0.5",
            "0.5",
            "1.125",
            "4.5",
            "8.75"
        }};

        out.medium = {{
            "-1048576.5000000000000000000000000001",
            "-1024.4999999999999999999999999999",
            "-16.5000000000000000000000000001",
            "-0.4999999999999999999999999999",
            "0.4999999999999999999999999999",
            "16.5000000000000000000000000001",
            "1024.4999999999999999999999999999",
            "1048576.5000000000000000000000000001"
        }};

        out.hard = {{
            "-2000000000.4999999999999999999999999999",
            "-1073741824.5000000000000000000000000001",
            "-65536.5000000000000000000000000001",
            "-1.5000000000000000000000000001",
            "1.5000000000000000000000000001",
            "65536.5000000000000000000000000001",
            "1073741824.4999999999999999999999999999",
            "2000000000.4999999999999999999999999999"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> positive_sqrt_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "0.03125",
            "0.125",
            "0.5",
            "1.0",
            "2.0",
            "4.0",
            "16.0",
            "256.0"
        }};

        out.medium = {{
            "0.000000000000000000000000000000125",
            "0.9999999999999999999999999999999999",
            "1.0000000000000000000000000000000001",
            "2.7182818284590452353602874713526624977572",
            "3.1415926535897932384626433832795028841971",
            "12345678901234567890.125",
            "0.0009765625",
            "4294967296.0"
        }};

        out.hard = {{
            "0.0000000000000000000000000000000000000000000000000125",
            "0.3333333333333333333333333333333333333333333333333333",
            "0.9999999999999999999999999999999999999999999999999999",
            "1.0000000000000000000000000000000000000000000000000001",
            "3.1415926535897932384626433832795028841971693993751058209",
            "2.7182818284590452353602874713526624977572470936999595749",
            "123456789012345678901234567890.125",
            "18446744073709551616.0"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> positive_log_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "0.125",
            "0.5",
            "0.75",
            "1.0",
            "1.25",
            "2.0",
            "4.0",
            "16.0"
        }};

        out.medium = {{
            "0.000000000000000000000000000000125",
            "0.125",
            "0.9999999999999999999999999999999999",
            "1.0000000000000000000000000000000001",
            "1.0009765625",
            "2.7182818284590452353602874713526624977572",
            "3.1415926535897932384626433832795028841971",
            "12345678901234567890.125"
        }};

        out.hard = {{
            "0.0000000000000000000000000000000000000000000000000125",
            "0.3333333333333333333333333333333333333333333333333333",
            "0.9999999999999999999999999999999999999999999999999999",
            "1.0000000000000000000000000000000000000000000000000001",
            "1.0000000000000000000000000000000000000000000000005",
            "2.7182818284590452353602874713526624977572470936999595749",
            "3.1415926535897932384626433832795028841971693993751058209",
            "123456789012345678901234567890.125"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> exponent_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-4.0",
            "-1.0",
            "-0.5",
            "0.0",
            "0.5",
            "1.0",
            "2.0",
            "4.0"
        }};

        out.medium = {{
            "-16.0",
            "-2.0",
            "-0.6931471805599453094172321214581765680755",
            "-0.0000000000000000000000000000000001",
            "0.0000000000000000000000000000000001",
            "0.6931471805599453094172321214581765680755",
            "2.7182818284590452353602874713526624977572",
            "16.0"
        }};

        out.hard = {{
            "-64.0",
            "-32.0",
            "-0.6931471805599453094172321214581765680755001343602552",
            "-0.0000000000000000000000000000000000000000000000000001",
            "0.0000000000000000000000000000000000000000000000000001",
            "0.6931471805599453094172321214581765680755001343602552",
            "32.0",
            "64.0"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<ldexp_value_spec> ldexp_value_buckets()
    {
        bucket_array_set<ldexp_value_spec> out{};

        out.easy = {{
            { "0.5", -8 },
            { "1.0", -4 },
            { "1.25", -1 },
            { "2.0", 0 },
            { "0.75", 1 },
            { "3.125", 4 },
            { "-1.5", -6 },
            { "-2.5", 6 }
        }};

        out.medium = {{
            { "0.9999999999999999999999999999999999", -64 },
            { "1.0000000000000000000000000000000001", -32 },
            { "0.125", -16 },
            { "2.7182818284590452353602874713526624977572", 8 },
            { "3.1415926535897932384626433832795028841971", 16 },
            { "12345678901234567890.125", -20 },
            { "-0.000000000000000000000000000000125", 40 },
            { "-16.0", 24 }
        }};

        out.hard = {{
            { "0.9999999999999999999999999999999999999999999999999999", -256 },
            { "1.0000000000000000000000000000000000000000000000000001", -128 },
            { "0.0000000000000000000000000000000000000000000000000125", 320 },
            { "2.7182818284590452353602874713526624977572470936999595749", 48 },
            { "3.1415926535897932384626433832795028841971693993751058209", 96 },
            { "123456789012345678901234567890.125", -40 },
            { "-0.3333333333333333333333333333333333333333333333333333", 160 },
            { "-32.0", 80 }
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> trig_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-2.9137426812345678901234567890123456789012345678901234",
            "-1.3372469102468135791357913579135791357913579135791358",
            "-0.6172839450617283945061728394506172839450617283945062",
            "-0.000000000000000000000000000000000000000000000000000137",
            "0.000000000000000000000000000000000000000000000000000251",
            "0.4839201746502987364510298376451029837645102983764510",
            "1.9283746501928374650192837465019283746501928374650193",
            "2.7764510298374650192837465019283746501928374650192837"
        }};

        out.medium = {{
            "-987.6543210123456789012345678901234567890123456789012345",
            "-271.8281828459045235360287471352662497757247093699959575",
            "-64.1250000000000000000000000000000000000000000000000125",
            "-1.0000000000000000000000000000000000000000000000000125",
            "1.0000000000000000000000000000000000000000000000000250",
            "37.6991118430775174569050479884012319897331299080235173",
            "123.4567890123456789012345678901234567890123456789012345",
            "789.0123456789012345678901234567890123456789012345678901"
        }};

        out.hard = {{
            "-987654321.1234567890123456789012345678901234567890123456789",
            "-12345678.901234567890123456789012345678901234567890123456",
            "-1048576.000000000000000000000000000000000000000000000125",
            "-4096.000000000000000000000000000000000000000000000000125",
            "4096.000000000000000000000000000000000000000000000000375",
            "1048576.000000000000000000000000000000000000000000000625",
            "12345678.901234567890123456789012345678901234567890123456",
            "987654321.9876543210987654321098765432109876543210987654321"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> inverse_trig_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-0.9",
            "-0.5",
            "-0.125",
            "0.0",
            "0.125",
            "0.5",
            "0.75",
            "0.9"
        }};

        out.medium = {{
            "-0.9999999999999999999999999999999999",
            "-0.9990234375",
            "-0.5",
            "-0.0000000000000000000000000000000001",
            "0.0000000000000000000000000000000001",
            "0.5",
            "0.9990234375",
            "0.9999999999999999999999999999999999"
        }};

        out.hard = {{
            "-0.9999999999999999999999999999999999999999999999999999",
            "-0.9999999999999999999999999999995",
            "-0.3333333333333333333333333333333333333333333333333333",
            "-0.0000000000000000000000000000000000000000000000000001",
            "0.0000000000000000000000000000000000000000000000000001",
            "0.3333333333333333333333333333333333333333333333333333",
            "0.9999999999999999999999999999995",
            "0.9999999999999999999999999999999999999999999999999999"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> pow_value_buckets()
    {
        bucket_array_set<binary_value_spec> out{};

        out.easy = {{
            { "0.5", "-2.0" },
            { "0.5", "2.0" },
            { "1.25", "-1.0" },
            { "1.25", "2.5" },
            { "2.0", "-0.5" },
            { "2.0", "3.0" },
            { "4.0", "0.5" },
            { "16.0", "0.25" }
        }};

        out.medium = {{
            { "0.9999999999999999999999999999999999", "8.0" },
            { "1.0000000000000000000000000000000001", "8.0" },
            { "1.0009765625", "16.0" },
            { "1.25", "-7.5" },
            { "2.7182818284590452353602874713526624977572", "0.5" },
            { "3.1415926535897932384626433832795028841971", "1.5" },
            { "12345678901234567890.125", "0.25" },
            { "0.000000000000000000000000000000125", "0.5" }
        }};

        out.hard = {{
            { "0.9999999999999999999999999999999999999999999999999999", "32.0" },
            { "1.0000000000000000000000000000000000000000000000000001", "32.0" },
            { "1.0000000000000000000000000000000000000000000000005", "64.0" },
            { "1.25", "-15.75" },
            { "2.7182818284590452353602874713526624977572470936999595749", "0.3333333333333333333333333333333333333333333333333333" },
            { "3.1415926535897932384626433832795028841971693993751058209", "1.75" },
            { "123456789012345678901234567890.125", "0.125" },
            { "0.0000000000000000000000000000000000000000000000000125", "0.5" }
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> atan2_value_buckets()
    {
        bucket_array_set<binary_value_spec> out{};

        out.easy = {{
            { "1.0", "1.0" },
            { "1.0", "-1.0" },
            { "-1.0", "1.0" },
            { "-1.0", "-1.0" },
            { "0.5", "2.0" },
            { "2.0", "0.5" },
            { "-0.5", "2.0" },
            { "2.0", "-0.5" }
        }};

        out.medium = {{
            { "0.1577450190437525533903646435935788464125842783342441464", "-0.7902875113057738885487120120465604260562530768860942235" },
            { "-0.7902875113057738885487120120465604260562530768860942235", "0.1577450190437525533903646435935788464125842783342441464" },
            { "3.1415926535897932384626433832795028841971", "2.7182818284590452353602874713526624977572" },
            { "2.7182818284590452353602874713526624977572", "3.1415926535897932384626433832795028841971" },
            { "0.000000000000000000000000000000125", "12345678901234567890.125" },
            { "12345678901234567890.125", "0.000000000000000000000000000000125" },
            { "-12345678901234567890.125", "0.000000000000000000000000000000125" },
            { "0.000000000000000000000000000000125", "-12345678901234567890.125" }
        }};

        out.hard = {{
            { "0.3333333333333333333333333333333333333333333333333333", "-1.0000000000000000000000000000000000000000000000000001" },
            { "-1.0000000000000000000000000000000000000000000000000001", "0.3333333333333333333333333333333333333333333333333333" },
            { "3.1415926535897932384626433832795028841971693993751058209", "2.7182818284590452353602874713526624977572470936999595749" },
            { "2.7182818284590452353602874713526624977572470936999595749", "3.1415926535897932384626433832795028841971693993751058209" },
            { "0.0000000000000000000000000000000000000000000000000125", "123456789012345678901234567890.125" },
            { "123456789012345678901234567890.125", "0.0000000000000000000000000000000000000000000000000125" },
            { "-123456789012345678901234567890.125", "0.0000000000000000000000000000000000000000000000000125" },
            { "0.0000000000000000000000000000000000000000000000000125", "-123456789012345678901234567890.125" }
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> fmod_value_buckets()
    {
        bucket_array_set<binary_value_spec> out{};

        out.easy = {{
            { "5.5", "2.0" },
            { "-5.5", "2.0" },
            { "7.25", "0.5" },
            { "-7.25", "0.5" },
            { "16.5", "3.125" },
            { "-16.5", "3.125" },
            { "1.25", "0.875" },
            { "-1.25", "0.875" }
        }};

        out.medium = {{
            { "12345678901234567890.125", "3.1415926535897932384626433832795028841971" },
            { "-12345678901234567890.125", "3.1415926535897932384626433832795028841971" },
            { "2.7182818284590452353602874713526624977572", "0.9999999999999999999999999999999999" },
            { "-2.7182818284590452353602874713526624977572", "0.9999999999999999999999999999999999" },
            { "1.0000000000000000000000000000000001", "0.3333333333333333333333333333333333" },
            { "-1.0000000000000000000000000000000001", "0.3333333333333333333333333333333333" },
            { "0.000000000000000000000000000000125", "0.00000000000000000000000000000003125" },
            { "-0.000000000000000000000000000000125", "0.00000000000000000000000000000003125" }
        }};

        out.hard = {{
            { "123456789012345678901234567890.125", "3.1415926535897932384626433832795028841971693993751058209" },
            { "-123456789012345678901234567890.125", "3.1415926535897932384626433832795028841971693993751058209" },
            { "2.7182818284590452353602874713526624977572470936999595749", "0.9999999999999999999999999999999999999999999999999999" },
            { "-2.7182818284590452353602874713526624977572470936999595749", "0.9999999999999999999999999999999999999999999999999999" },
            { "1.0000000000000000000000000000000000000000000000000001", "0.3333333333333333333333333333333333333333333333333333" },
            { "-1.0000000000000000000000000000000000000000000000000001", "0.3333333333333333333333333333333333333333333333333333" },
            { "0.0000000000000000000000000000000000000000000000000125", "0.000000000000000000000000000000000000000000000000003125" },
            { "-0.0000000000000000000000000000000000000000000000000125", "0.000000000000000000000000000000000000000000000000003125" }
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<recurrence_value_spec> recurrence_specs()
    {
        bucket_array_set<recurrence_value_spec> out{};

        out.easy = {{
            { "-0.75", "0.125", "3.125", "2.5", "1.25", "0.875" },
            { "0.5", "-0.25", "1.125", "0.75", "0.5", "1.0" },
            { "-1.0", "0.5", "0.25", "-0.125", "1.5", "0.75" },
            { "0.25", "0.75", "-0.5", "1.0", "0.625", "1.25" },
            { "-0.125", "-0.5", "0.875", "-0.375", "1.125", "0.625" },
            { "1.0", "0.0", "0.5", "0.25", "0.75", "1.5" },
            { "-0.625", "0.375", "1.25", "-0.75", "0.875", "0.5" },
            { "0.875", "-0.125", "-0.25", "0.625", "1.0", "1.125" }
        }};

        out.medium = {{
            { "-0.7902875113057738885487120120465604260562530768860942235", "0.1577450190437525533903646435935788464125842783342441464", "3.1415926535897932384626433832795028841971", "2.7182818284590452353602874713526624977572", "1.0000000000000000000000000000000001", "0.9999999999999999999999999999999999" },
            { "0.1577450190437525533903646435935788464125842783342441464", "-0.7902875113057738885487120120465604260562530768860942235", "2.7182818284590452353602874713526624977572", "3.1415926535897932384626433832795028841971", "0.9999999999999999999999999999999999", "1.0000000000000000000000000000000001" },
            { "3.1415926535897932384626433832795028841971", "2.7182818284590452353602874713526624977572", "-0.7902875113057738885487120120465604260562530768860942235", "0.1577450190437525533903646435935788464125842783342441464", "1.25", "0.875" },
            { "2.7182818284590452353602874713526624977572", "3.1415926535897932384626433832795028841971", "0.1577450190437525533903646435935788464125842783342441464", "-0.7902875113057738885487120120465604260562530768860942235", "0.875", "1.25" },
            { "1.0000000000000000000000000000000001", "0.9999999999999999999999999999999999", "12345678901234567890.125", "0.000000000000000000000000000000125", "1.5", "0.625" },
            { "0.9999999999999999999999999999999999", "1.0000000000000000000000000000000001", "0.000000000000000000000000000000125", "12345678901234567890.125", "0.625", "1.5" },
            { "12345678901234567890.125", "0.000000000000000000000000000000125", "1.0000000000000000000000000000000001", "0.9999999999999999999999999999999999", "1.125", "0.75" },
            { "0.000000000000000000000000000000125", "12345678901234567890.125", "0.9999999999999999999999999999999999", "1.0000000000000000000000000000000001", "0.75", "1.125" }
        }};

        out.hard = {{
            { "-1.0000000000000000000000000000000000000000000000000001", "0.3333333333333333333333333333333333333333333333333333", "3.1415926535897932384626433832795028841971693993751058209", "2.7182818284590452353602874713526624977572470936999595749", "1.0000000000000000000000000000000001", "0.9999999999999999999999999999999999" },
            { "0.3333333333333333333333333333333333333333333333333333", "-1.0000000000000000000000000000000000000000000000000001", "2.7182818284590452353602874713526624977572470936999595749", "3.1415926535897932384626433832795028841971693993751058209", "0.9999999999999999999999999999999999", "1.0000000000000000000000000000000001" },
            { "3.1415926535897932384626433832795028841971693993751058209", "2.7182818284590452353602874713526624977572470936999595749", "-1.0000000000000000000000000000000000000000000000000001", "0.3333333333333333333333333333333333333333333333333333", "1.25", "0.875" },
            { "2.7182818284590452353602874713526624977572470936999595749", "3.1415926535897932384626433832795028841971693993751058209", "0.3333333333333333333333333333333333333333333333333333", "-1.0000000000000000000000000000000000000000000000000001", "0.875", "1.25" },
            { "1.0000000000000000000000000000000000000000000000000001", "0.9999999999999999999999999999999999999999999999999999", "123456789012345678901234567890.125", "0.0000000000000000000000000000000000000000000000000125", "1.5", "0.625" },
            { "0.9999999999999999999999999999999999999999999999999999", "1.0000000000000000000000000000000000000000000000000001", "0.0000000000000000000000000000000000000000000000000125", "123456789012345678901234567890.125", "0.625", "1.5" },
            { "123456789012345678901234567890.125", "0.0000000000000000000000000000000000000000000000000125", "1.0000000000000000000000000000000000000000000000000001", "0.9999999999999999999999999999999999999999999999999999", "1.125", "0.75" },
            { "0.0000000000000000000000000000000000000000000000000125", "123456789012345678901234567890.125", "0.9999999999999999999999999999999999999999999999999999", "1.0000000000000000000000000000000000000000000000000001", "0.75", "1.125" }
        }};

        return out;
    }

    template<typename Work>
    [[nodiscard]] bench_result run_benchmark(std::int64_t iteration_count, Work&& work)
    {
        const auto start = clock_type::now();
        const auto final_value = work();
        const auto end = clock_type::now();

        consume_result(final_value);

        const std::chrono::duration<double, std::milli> elapsed = end - start;

        bench_result result;
        result.total_ms = elapsed.count();
        result.ns_per_iter = (elapsed.count() * 1'000'000.0) / static_cast<double>(iteration_count);
        result.iteration_count = iteration_count;
        return result;
    }

    void print_result(const char* label, const comparison_result& result)
    {
        const double ratio = result.mpfr.ns_per_iter / result.f256.ns_per_iter;

        std::cout
            << std::fixed << std::setprecision(2)
            << label
            << "\n  f256 : " << result.f256.total_ms << " ms total, " << result.f256.ns_per_iter << " ns/iter" << "  (total_iterations: " << result.f256.iteration_count << ")"
            << "\n  mpfr : " << result.mpfr.total_ms << " ms total, " << result.mpfr.ns_per_iter << " ns/iter" << "  (total_iterations: " << result.mpfr.iteration_count << ")"
            << "\n  mpfr/f256 ratio: " << ratio << "x"
            << "\n";
    }

    void print_bucketed_results(const char* group, const char* label, const bucketed_comparison_result& results)
    {
        if (std::string_view(label) != "fabs")
            chart_writer.record_result(group, label, results.hard.f256.ns_per_iter, results.hard.mpfr.ns_per_iter);
        std::string easy_label = std::string(label) + " [easy]";
        std::string medium_label = std::string(label) + " [medium]";
        std::string hard_label = std::string(label) + " [hard]";
        std::string typical_label = std::string(label) + " [typical]";

        print_result(easy_label.c_str(), results.easy);
        print_result(medium_label.c_str(), results.medium);
        print_result(hard_label.c_str(), results.hard);
        print_result(typical_label.c_str(), results.typical);
    }

    [[nodiscard]] comparison_result combine_typical_results(
        const comparison_result& easy,
        const comparison_result& medium,
        const comparison_result& hard)
    {
        constexpr std::int64_t easy_weight = 1;
        constexpr std::int64_t medium_weight = 4;
        constexpr std::int64_t hard_weight = 3;

        const auto combine = [=](const bench_result& easy_result, const bench_result& medium_result, const bench_result& hard_result)
        {
            bench_result out{};
            out.total_ms =
                easy_result.total_ms * static_cast<double>(easy_weight) +
                medium_result.total_ms * static_cast<double>(medium_weight) +
                hard_result.total_ms * static_cast<double>(hard_weight);
            out.iteration_count =
                easy_result.iteration_count * easy_weight +
                medium_result.iteration_count * medium_weight +
                hard_result.iteration_count * hard_weight;
            out.ns_per_iter = (out.total_ms * 1'000'000.0) / static_cast<double>(out.iteration_count);
            return out;
        };

        comparison_result out{};
        out.f256 = combine(easy.f256, medium.f256, hard.f256);
        out.mpfr = combine(easy.mpfr, medium.mpfr, hard.mpfr);
        return out;
    }

    template<typename T, typename U>
    [[nodiscard]] T blend_result(const U& value, const T& acc)
    {
        return static_cast<T>(value) + acc * T(0.25);
    }

    template<typename T, typename U>
    [[nodiscard]] std::pair<T, U> blend_result(const std::pair<T, U>& value, const std::pair<T, U>& acc)
    {
        return {
            blend_result(value.first, acc.first),
            blend_result(value.second, acc.second)
        };
    }


    template<typename T>
    [[nodiscard]] T apply_floor(const T& x)
    {
        using std::floor;
        return floor(x);
    }

    template<typename T>
    [[nodiscard]] T apply_ceil(const T& x)
    {
        using std::ceil;
        return ceil(x);
    }

    template<typename T>
    [[nodiscard]] T apply_trunc(const T& x)
    {
        using std::trunc;
        return trunc(x);
    }

    template<typename T>
    [[nodiscard]] T apply_round(const T& x)
    {
        using std::round;
        return round(x);
    }

    template<typename T>
    [[nodiscard]] T apply_sqrt(const T& x)
    {
        using std::sqrt;
        return sqrt(x);
    }

    template<typename T>
    [[nodiscard]] T apply_exp(const T& x)
    {
        using std::exp;
        return exp(x);
    }

    template<typename T>
    [[nodiscard]] T apply_exp2(const T& x)
    {
        using std::exp2;
        return exp2(x);
    }

    template<typename T>
    [[nodiscard]] T apply_log(const T& x)
    {
        using std::log;
        return log(x);
    }

    template<typename T>
    [[nodiscard]] T apply_log2(const T& x)
    {
        using std::log2;
        return log2(x);
    }

    template<typename T>
    [[nodiscard]] T apply_log10(const T& x)
    {
        using std::log10;
        return log10(x);
    }

    template<typename T>
    [[nodiscard]] T apply_fmod(const T& x, const T& y)
    {
        using std::fmod;
        return fmod(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_pow(const T& x, const T& y)
    {
        using std::pow;
        return pow(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_sin(const T& x)
    {
        using std::sin;
        return sin(x);
    }

    template<typename T>
    [[nodiscard]] T apply_cos(const T& x)
    {
        using std::cos;
        return cos(x);
    }

    template<typename T>
    [[nodiscard]] T apply_tan(const T& x)
    {
        using std::tan;
        return tan(x);
    }

    template<typename T>
    [[nodiscard]] T apply_atan(const T& x)
    {
        using std::atan;
        return atan(x);
    }

    template<typename T>
    [[nodiscard]] T apply_atan2(const T& y, const T& x)
    {
        using std::atan2;
        return atan2(y, x);
    }

    template<typename T>
    [[nodiscard]] T apply_asin(const T& x)
    {
        using std::asin;
        return asin(x);
    }

    template<typename T>
    [[nodiscard]] T apply_acos(const T& x)
    {
        using std::acos;
        return acos(x);
    }

    template<typename T>
    [[nodiscard]] T apply_ldexp(const T& x, int exponent)
    {
        using std::ldexp;
        return ldexp(x, exponent);
    }


    template<typename T>
    [[nodiscard]] T apply_abs(const T& x)
    {
        using std::abs;
        return abs(x);
    }

    template<typename T>
    [[nodiscard]] T apply_fabs(const T& x)
    {
        using std::fabs;
        return fabs(x);
    }

    template<typename T>
    [[nodiscard]] long apply_lround(const T& x)
    {
        using std::lround;
        return lround(x);
    }

    template<typename T>
    [[nodiscard]] long long apply_llround(const T& x)
    {
        using std::llround;
        return llround(x);
    }

    template<typename T>
    [[nodiscard]] long apply_lrint(const T& x)
    {
        using std::lrint;
        return lrint(x);
    }

    template<typename T>
    [[nodiscard]] long long apply_llrint(const T& x)
    {
        using std::llrint;
        return llrint(x);
    }

    template<typename T>
    [[nodiscard]] std::pair<T, int> apply_remquo(const T& x, const T& y)
    {
        int quotient = 0;
        using std::remquo;
        T remainder = remquo(x, y, &quotient);
        return { remainder, quotient };
    }

    template<typename T>
    [[nodiscard]] T apply_fma(const T& x, const T& y, const T& z)
    {
        using std::fma;
        return fma(x, y, z);
    }

    template<typename T>
    [[nodiscard]] T apply_fmin(const T& x, const T& y)
    {
        using std::fmin;
        return fmin(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_fmax(const T& x, const T& y)
    {
        using std::fmax;
        return fmax(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_fdim(const T& x, const T& y)
    {
        using std::fdim;
        return fdim(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_copysign(const T& x, const T& y)
    {
        using std::copysign;
        return copysign(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_scalbn(const T& x, int exponent)
    {
        using std::scalbn;
        return scalbn(x, exponent);
    }

    template<typename T>
    [[nodiscard]] T apply_scalbln(const T& x, long exponent)
    {
        using std::scalbln;
        return scalbln(x, exponent);
    }

    template<typename T>
    [[nodiscard]] std::pair<T, int> apply_frexp(const T& x)
    {
        int exponent = 0;
        using std::frexp;
        T fraction = frexp(x, &exponent);
        return { fraction, exponent };
    }

    template<typename T>
    [[nodiscard]] std::pair<T, T> apply_modf(const T& x)
    {
        T integral{};
        using std::modf;
        T fractional = modf(x, &integral);
        return { fractional, integral };
    }

    template<typename T>
    [[nodiscard]] int apply_ilogb(const T& x)
    {
        using std::ilogb;
        return ilogb(x);
    }

    template<typename T>
    [[nodiscard]] T apply_logb(const T& x)
    {
        using std::logb;
        return logb(x);
    }

    template<typename T>
    [[nodiscard]] T apply_nextafter(const T& from, const T& to)
    {
        using std::nextafter;
        return nextafter(from, to);
    }

    template<typename T>
    [[nodiscard]] T apply_nexttoward(const T& from, const T& to)
    {
        using std::nexttoward;
        return nexttoward(from, to);
    }

    template<typename T>
    [[nodiscard]] T apply_nexttoward_long_double(const T& from, long double to)
    {
        using std::nexttoward;
        return nexttoward(from, to);
    }

    template<>
    [[nodiscard]] mpfr_ref apply_nexttoward_long_double<mpfr_ref>(const mpfr_ref& from, long double to)
    {
        using boost::multiprecision::nexttoward;
        return nexttoward(from, mpfr_ref(to));
    }

    template<typename T, typename Op>
    [[nodiscard]] bench_result benchmark_value_bucket(
        const std::array<const char*, bucket_value_count>& texts,
        std::int64_t total_iterations,
        Op&& op)
    {
        std::array<T, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = parse_benchmark_value<T>(texts[i]);

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values, i), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Op>
    [[nodiscard]] bench_result benchmark_binary_bucket(
        const std::array<binary_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        std::array<std::pair<T, T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].first = parse_benchmark_value<T>(specs[i].lhs);
            values[i].second = parse_benchmark_value<T>(specs[i].rhs);
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front().first;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i].first, values[i].second), acc);
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_ldexp_bucket(
        const std::array<ldexp_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations)
    {
        struct ldexp_case
        {
            T value{};
            int exponent = 0;
        };

        std::array<ldexp_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].value = parse_benchmark_value<T>(specs[i].value);
            values[i].exponent = specs[i].exponent;
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front().value;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(apply_ldexp(values[i].value, values[i].exponent), acc);
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_mixed_recurrence_bucket(
        const std::array<recurrence_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations)
    {
        struct recurrence_case
        {
            T x{};
            T y{};
            T a{};
            T b{};
            T c{};
            T d{};
        };

        std::array<recurrence_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].x = parse_benchmark_value<T>(specs[i].x);
            values[i].y = parse_benchmark_value<T>(specs[i].y);
            values[i].a = parse_benchmark_value<T>(specs[i].a);
            values[i].b = parse_benchmark_value<T>(specs[i].b);
            values[i].c = parse_benchmark_value<T>(specs[i].c);
            values[i].d = parse_benchmark_value<T>(specs[i].d);
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            auto state = values;
            T acc_x = state.front().x;
            T acc_y = state.front().y;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (auto& item : state)
                {
                    const T xx = item.x * item.x;
                    const T yy = item.y * item.y;
                    const T xy = item.x * item.y;

                    item.x = (xx - yy) + item.a;
                    item.y = (xy + xy) + item.b;
                    item.x = item.x / (item.c + T(0.5));
                    item.y = item.y / (item.d + T(1.5));

                    acc_x = blend_result(item.x, acc_x);
                    acc_y = blend_result(item.y, acc_y);
                }
            }

            return std::pair<T, T>{ acc_x, acc_y };
        });
    }


    template<typename T, typename Result, typename Op>
    [[nodiscard]] bench_result benchmark_value_bucket_result(
        const std::array<const char*, bucket_value_count>& texts,
        std::int64_t total_iterations,
        Op&& op)
    {
        std::array<T, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = parse_benchmark_value<T>(texts[i]);

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            Result acc = op(values.front());

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i]), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Result, typename Op>
    [[nodiscard]] bench_result benchmark_binary_bucket_result(
        const std::array<binary_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        std::array<std::pair<T, T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].first = parse_benchmark_value<T>(specs[i].lhs);
            values[i].second = parse_benchmark_value<T>(specs[i].rhs);
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            Result acc = op(values.front().first, values.front().second);

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i].first, values[i].second), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Result, typename Op>
    [[nodiscard]] bench_result benchmark_binary_long_double_bucket_result(
        const std::array<binary_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        struct nexttoward_case
        {
            T from{};
            long double to = 0.0L;
        };

        std::array<nexttoward_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].from = parse_benchmark_value<T>(specs[i].lhs);
            const T target = parse_benchmark_value<T>(specs[i].rhs);
            values[i].to = static_cast<long double>(static_cast<double>(target));
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            Result acc = op(values.front().from, values.front().to);

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i].from, values[i].to), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Result, typename Exponent, typename Op>
    [[nodiscard]] bench_result benchmark_exponent_bucket_result(
        const std::array<ldexp_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        struct exponent_case
        {
            T value{};
            Exponent exponent{};
        };

        std::array<exponent_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].value = parse_benchmark_value<T>(specs[i].value);
            values[i].exponent = static_cast<Exponent>(specs[i].exponent);
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            Result acc = op(values.front().value, values.front().exponent);

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i].value, values[i].exponent), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Result, typename Op>
    [[nodiscard]] bench_result benchmark_ternary_bucket_result(
        const std::array<recurrence_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        struct ternary_case
        {
            T x{};
            T y{};
            T z{};
        };

        std::array<ternary_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].x = parse_benchmark_value<T>(specs[i].x);
            values[i].y = parse_benchmark_value<T>(specs[i].y);
            values[i].z = parse_benchmark_value<T>(specs[i].a);
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark(iteration_count, [&]()
        {
            Result acc = op(values.front().x, values.front().y, values.front().z);

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i].x, values[i].y, values[i].z), acc);
            }

            return acc;
        });
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_value_benchmark_result(
        const bucket_array_set<const char*>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_value_bucket_result<f256, FltxResult>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_value_bucket_result<f256, FltxResult>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_value_bucket_result<f256, FltxResult>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_benchmark_result(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_binary_bucket_result<f256, FltxResult>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_binary_bucket_result<f256, FltxResult>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_binary_bucket_result<f256, FltxResult>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_long_double_benchmark_result(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_binary_long_double_bucket_result<f256, FltxResult>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_binary_long_double_bucket_result<f256, FltxResult>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_binary_long_double_bucket_result<f256, FltxResult>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Exponent, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_exponent_benchmark_result(
        const bucket_array_set<ldexp_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_exponent_bucket_result<f256, FltxResult, Exponent>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_exponent_bucket_result<f256, FltxResult, Exponent>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_exponent_bucket_result<f256, FltxResult, Exponent>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_ternary_benchmark_result(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_ternary_bucket_result<f256, FltxResult>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_ternary_bucket_result<f256, FltxResult>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_ternary_bucket_result<f256, FltxResult>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_value_benchmark(
        const bucket_array_set<const char*>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_value_bucket<f256>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_value_bucket<mpfr_ref>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_value_bucket<f256>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_value_bucket<mpfr_ref>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_value_bucket<f256>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_value_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_benchmark(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_binary_bucket<f256>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.easy, total_iterations, op);
        out.medium.f256 = benchmark_binary_bucket<f256>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.medium, total_iterations, op);
        out.hard.f256 = benchmark_binary_bucket<f256>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_ldexp_benchmark(
        const bucket_array_set<ldexp_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_ldexp_bucket<f256>(specs.easy, total_iterations);
        out.easy.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.easy, total_iterations);
        out.medium.f256 = benchmark_ldexp_bucket<f256>(specs.medium, total_iterations);
        out.medium.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.medium, total_iterations);
        out.hard.f256 = benchmark_ldexp_bucket<f256>(specs.hard, total_iterations);
        out.hard.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.hard, total_iterations);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_mixed_recurrence_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        out.easy.f256 = benchmark_mixed_recurrence_bucket<f256>(specs.easy, total_iterations);
        out.easy.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.easy, total_iterations);
        out.medium.f256 = benchmark_mixed_recurrence_bucket<f256>(specs.medium, total_iterations);
        out.medium.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.medium, total_iterations);
        out.hard.f256 = benchmark_mixed_recurrence_bucket<f256>(specs.hard, total_iterations);
        out.hard.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.hard, total_iterations);
        out.typical = combine_typical_results(out.easy, out.medium, out.hard);

        return out;
    }

    [[nodiscard]] mpfr_ref ref_floor(const mpfr_ref& value)
    {
        return boost::multiprecision::floor(value);
    }

    [[nodiscard]] mpfr_ref ref_fmod(const mpfr_ref& x, const mpfr_ref& y)
    {
        const mpfr_ref q = x / y;
        mpfr_ref truncated{};
        if (q < 0)
            truncated = boost::multiprecision::ceil(q);
        else
            truncated = boost::multiprecision::floor(q);
        return x - truncated * y;
    }

    [[nodiscard]] mpfr_ref ref_round_to_even(const mpfr_ref& value)
    {
        mpfr_ref rounded = ref_floor(value + mpfr_ref{ "0.5" });
        if ((rounded - value) == mpfr_ref{ "0.5" } && ref_fmod(rounded, mpfr_ref{ 2 }) != mpfr_ref{ 0 })
            rounded -= 1;
        return rounded;
    }

    [[nodiscard]] mpfr_ref ref_remainder_ties_even(const mpfr_ref& x, const mpfr_ref& y)
    {
        return x - ref_round_to_even(x / y) * y;
    }

    template<typename T>
    [[nodiscard]] T apply_nearbyint(const T& x)
    {
        using std::nearbyint;
        return nearbyint(x);
    }

    template<>
    [[nodiscard]] mpfr_ref apply_nearbyint<mpfr_ref>(const mpfr_ref& x)
    {
        return ref_round_to_even(x);
    }

    template<typename T>
    [[nodiscard]] T apply_rint(const T& x)
    {
        using std::rint;
        return rint(x);
    }

    template<>
    [[nodiscard]] mpfr_ref apply_rint<mpfr_ref>(const mpfr_ref& x)
    {
        return ref_round_to_even(x);
    }

    template<typename T>
    [[nodiscard]] T apply_cbrt(const T& x)
    {
        using std::cbrt;
        return cbrt(x);
    }

    template<typename T>
    [[nodiscard]] T apply_hypot(const T& x, const T& y)
    {
        using std::hypot;
        return hypot(x, y);
    }

    template<>
    [[nodiscard]] mpfr_ref apply_hypot<mpfr_ref>(const mpfr_ref& x, const mpfr_ref& y)
    {
        using std::sqrt;
        return sqrt(x * x + y * y);
    }

    template<typename T>
    [[nodiscard]] T apply_expm1(const T& x)
    {
        using std::expm1;
        return expm1(x);
    }

    template<typename T>
    [[nodiscard]] T apply_log1p(const T& x)
    {
        using std::log1p;
        return log1p(x);
    }

    template<typename T>
    [[nodiscard]] T apply_remainder(const T& x, const T& y)
    {
        using std::remainder;
        return remainder(x, y);
    }

    template<>
    [[nodiscard]] mpfr_ref apply_remainder<mpfr_ref>(const mpfr_ref& x, const mpfr_ref& y)
    {
        return ref_remainder_ties_even(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_sinh(const T& x)
    {
        using std::sinh;
        return sinh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_cosh(const T& x)
    {
        using std::cosh;
        return cosh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_tanh(const T& x)
    {
        using std::tanh;
        return tanh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_asinh(const T& x)
    {
        using std::asinh;
        return asinh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_acosh(const T& x)
    {
        using std::acosh;
        return acosh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_atanh(const T& x)
    {
        using std::atanh;
        return atanh(x);
    }

    template<typename T>
    [[nodiscard]] T apply_erf(const T& x)
    {
        using std::erf;
        return erf(x);
    }

    template<typename T>
    [[nodiscard]] T apply_erfc(const T& x)
    {
        using std::erfc;
        return erfc(x);
    }

    template<typename T>
    [[nodiscard]] T apply_lgamma(const T& x)
    {
        using std::lgamma;
        return lgamma(x);
    }

    template<typename T>
    [[nodiscard]] T apply_tgamma(const T& x)
    {
        using std::tgamma;
        return tgamma(x);
    }

    [[nodiscard]] bucket_array_set<const char*> log1p_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-0.875",
            "-0.5",
            "-0.125",
            "-1e-20",
            "1e-20",
            "0.125",
            "1",
            "10"
        }};

        out.medium = {{
            "-0.9500000000000000000000000000000000",
            "-0.5000000000000000000000000000000001",
            "-0.1250000000000000000000000000000000",
            "-0.0000000000000000000000000000000001",
            "0.0000000000000000000000000000000001",
            "0.1250000000000000000000000000000000",
            "1.0000000000000000000000000000000001",
            "15.5"
        }};

        out.hard = {{
            "-0.9999999999999999999999999999999999999999999999999990",
            "-0.8750000000000000000000000000000000000000000000000000",
            "-0.1250000000000000000000000000000000000000000000000000",
            "-0.0000000000000000000000000000000000000000000000000001",
            "0.0000000000000000000000000000000000000000000000000001",
            "0.1250000000000000000000000000000000000000000000000000",
            "1.0000000000000000000000000000000000000000000000000001",
            "31.0"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> hyperbolic_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-2",
            "-1",
            "-0.125",
            "0",
            "0.125",
            "1",
            "2",
            "4"
        }};

        out.medium = {{
            "-8",
            "-4",
            "-1",
            "-0.125",
            "0.125",
            "1",
            "4",
            "8"
        }};

        out.hard = {{
            "-12",
            "-8",
            "-2",
            "-0.125",
            "0.125",
            "2",
            "8",
            "12"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> acosh_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "1",
            "1.125",
            "1.5",
            "2",
            "4",
            "8",
            "16",
            "32"
        }};

        out.medium = {{
            "1.0000000000000000000000000000000001",
            "1.1250000000000000000000000000000000",
            "1.5",
            "2",
            "4",
            "8",
            "16",
            "64"
        }};

        out.hard = {{
            "1.0000000000000000000000000000000000000000000000000001",
            "1.0009765625",
            "1.1250000000000000000000000000000000000000000000000000",
            "1.5",
            "2",
            "4",
            "16",
            "128"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> atanh_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-0.95",
            "-0.5",
            "-0.125",
            "-1e-20",
            "1e-20",
            "0.125",
            "0.5",
            "0.95"
        }};

        out.medium = {{
            "-0.9990234375",
            "-0.95",
            "-0.5",
            "-0.125",
            "0.125",
            "0.5",
            "0.95",
            "0.9990234375"
        }};

        out.hard = {{
            "-0.9999999999999999999999999999999999999999999999999990",
            "-0.9999999999999999999999999999995",
            "-0.5",
            "-0.125",
            "0.125",
            "0.5",
            "0.9999999999999999999999999999995",
            "0.9999999999999999999999999999999999999999999999999990"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> erf_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "-4",
            "-1",
            "-0.125",
            "-1e-20",
            "1e-20",
            "0.125",
            "1",
            "4"
        }};

        out.medium = {{
            "-4",
            "-2",
            "-1",
            "-0.125",
            "0.125",
            "1",
            "2",
            "4"
        }};

        out.hard = {{
            "-6",
            "-4",
            "-2.5",
            "-0.125",
            "0.125",
            "2.5",
            "4",
            "6"
        }};

        return out;
    }

    [[nodiscard]] bucket_array_set<const char*> gamma_value_buckets()
    {
        bucket_array_set<const char*> out{};

        out.easy = {{
            "0.125",
            "0.5",
            "0.75",
            "1",
            "1.5",
            "2.5",
            "5.5",
            "10"
        }};

        out.medium = {{
            "0.125",
            "0.2502917256654617036630365854587596",
            "0.5",
            "0.75",
            "1.5",
            "2.2502917256654617036630365854587596",
            "5.5",
            "15.875"
        }};

        out.hard = {{
            "0.1250000000000000000000000000000000000000000000000000",
            "0.2502917256654617036630365854587596250705075583428090",
            "0.5000000000000000000000000000000000000000000000000000",
            "0.7500000000000000000000000000000000000000000000000000",
            "1.5000000000000000000000000000000000000000000000000000",
            "2.2502917256654617036630365854587596250705075583428090",
            "5.5000000000000000000000000000000000000000000000000000",
            "31.875"
        }};

        return out;
    }
}

TEST_CASE("f256 vs mpfr add performance", "[bench][fltx][f256][arithmetic][add]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(generic_value_buckets(), total_iterations, [](const auto& values, std::size_t i)
    {
        const auto& a = values[i];
        const auto& b = values[(i + 1) % bucket_value_count];
        return a + b;
    });
    print_bucketed_results("Arithmetic", "add", results);
}

TEST_CASE("f256 vs mpfr subtract performance", "[bench][fltx][f256][arithmetic][subtract]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(generic_value_buckets(), total_iterations, [](const auto& values, std::size_t i)
    {
        const auto& a = values[i];
        const auto& b = values[(i + 1) % bucket_value_count];
        return a - b;
    });
    print_bucketed_results("Arithmetic", "subtract", results);
}

TEST_CASE("f256 vs mpfr multiply performance", "[bench][fltx][f256][arithmetic][multiply]")
{
    const std::int64_t total_iterations = 12000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(generic_value_buckets(), total_iterations, [](const auto& values, std::size_t i)
    {
        const auto& a = values[i];
        const auto& b = values[(i + 3) % bucket_value_count];
        return a * b;
    });
    print_bucketed_results("Arithmetic", "multiply", results);
}

TEST_CASE("f256 vs mpfr divide performance", "[bench][fltx][f256][arithmetic][divide]")
{
    const std::int64_t total_iterations = 8000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(generic_value_buckets(), total_iterations, [](const auto& values, std::size_t i)
    {
        const auto& numerator = values[(i + 2) % bucket_value_count];
        const auto& denominator = values[(i + 5) % bucket_value_count];
        return numerator / denominator;
    });
    print_bucketed_results("Arithmetic", "divide", results);
}

TEST_CASE("f256 vs mpfr mixed recurrence performance", "[bench][fltx][f256][arithmetic]")
{
    const std::int64_t total_iterations = 25000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark(recurrence_specs(), total_iterations);
    print_bucketed_results("Arithmetic", "mixed recurrence", results);
}


TEST_CASE("f256 vs mpfr abs performance", "[bench][fltx][f256][arithmetic][abs]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f256, mpfr_ref>(generic_value_buckets(), total_iterations, [](const auto& x) { return apply_abs(x); });
    print_bucketed_results("Floating-point utilities", "abs", results);
}

TEST_CASE("f256 vs mpfr fabs performance", "[bench][fltx][f256][arithmetic][fabs]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f256, mpfr_ref>(generic_value_buckets(), total_iterations, [](const auto& x) { return apply_fabs(x); });
    print_bucketed_results("Floating-point utilities", "fabs", results);
}

TEST_CASE("f256 vs mpfr lround performance", "[bench][fltx][f256][rounding][lround]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long, long>(integer_rounding_value_buckets(), total_iterations, [](const auto& x) { return apply_lround(x); });
    print_bucketed_results("Rounding", "lround", results);
}

TEST_CASE("f256 vs mpfr llround performance", "[bench][fltx][f256][rounding][llround]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long long, long long>(integer_rounding_value_buckets(), total_iterations, [](const auto& x) { return apply_llround(x); });
    print_bucketed_results("Rounding", "llround", results);
}

TEST_CASE("f256 vs mpfr lrint performance", "[bench][fltx][f256][rounding][lrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long, long>(integer_rounding_value_buckets(), total_iterations, [](const auto& x) { return apply_lrint(x); });
    print_bucketed_results("Rounding", "lrint", results);
}

TEST_CASE("f256 vs mpfr llrint performance", "[bench][fltx][f256][rounding][llrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long long, long long>(integer_rounding_value_buckets(), total_iterations, [](const auto& x) { return apply_llrint(x); });
    print_bucketed_results("Rounding", "llrint", results);
}

TEST_CASE("f256 vs mpfr remquo performance", "[bench][fltx][f256][remquo]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<std::pair<f256, int>, std::pair<mpfr_ref, int>>(fmod_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_remquo(x, y); });
    print_bucketed_results("Remainders", "remquo", results);
}

TEST_CASE("f256 vs mpfr fma performance", "[bench][fltx][f256][fma]")
{
    const std::int64_t total_iterations = 20000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_ternary_benchmark_result<f256, mpfr_ref>(recurrence_specs(), total_iterations, [](const auto& x, const auto& y, const auto& z) { return apply_fma(x, y, z); });
    print_bucketed_results("Floating-point utilities", "fma", results);
}

TEST_CASE("f256 vs mpfr fmin performance", "[bench][fltx][f256][fmin]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_fmin(x, y); });
    print_bucketed_results("Floating-point utilities", "fmin", results);
}

TEST_CASE("f256 vs mpfr fmax performance", "[bench][fltx][f256][fmax]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_fmax(x, y); });
    print_bucketed_results("Floating-point utilities", "fmax", results);
}

TEST_CASE("f256 vs mpfr fdim performance", "[bench][fltx][f256][fdim]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_fdim(x, y); });
    print_bucketed_results("Floating-point utilities", "fdim", results);
}

TEST_CASE("f256 vs mpfr copysign performance", "[bench][fltx][f256][copysign]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_copysign(x, y); });
    print_bucketed_results("Floating-point utilities", "copysign", results);
}

TEST_CASE("f256 vs mpfr scalbn performance", "[bench][fltx][f256][scalbn]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_exponent_benchmark_result<f256, mpfr_ref, int>(ldexp_value_buckets(), total_iterations, [](const auto& x, int e) { return apply_scalbn(x, e); });
    print_bucketed_results("Floating-point utilities", "scalbn", results);
}

TEST_CASE("f256 vs mpfr scalbln performance", "[bench][fltx][f256][scalbln]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_exponent_benchmark_result<f256, mpfr_ref, long>(ldexp_value_buckets(), total_iterations, [](const auto& x, long e) { return apply_scalbln(x, e); });
    print_bucketed_results("Floating-point utilities", "scalbln", results);
}

TEST_CASE("f256 vs mpfr frexp performance", "[bench][fltx][f256][frexp]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<std::pair<f256, int>, std::pair<mpfr_ref, int>>(generic_value_buckets(), total_iterations, [](const auto& x) { return apply_frexp(x); });
    print_bucketed_results("Floating-point utilities", "frexp", results);
}

TEST_CASE("f256 vs mpfr modf performance", "[bench][fltx][f256][modf]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<std::pair<f256, f256>, std::pair<mpfr_ref, mpfr_ref>>(rounding_value_buckets(), total_iterations, [](const auto& x) { return apply_modf(x); });
    print_bucketed_results("Floating-point utilities", "modf", results);
}

TEST_CASE("f256 vs mpfr ilogb performance", "[bench][fltx][f256][ilogb]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<int, int>(generic_value_buckets(), total_iterations, [](const auto& x) { return apply_ilogb(x); });
    print_bucketed_results("Floating-point utilities", "ilogb", results);
}

TEST_CASE("f256 vs mpfr logb performance", "[bench][fltx][f256][logb]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f256, mpfr_ref>(generic_value_buckets(), total_iterations, [](const auto& x) { return apply_logb(x); });
    print_bucketed_results("Floating-point utilities", "logb", results);
}

TEST_CASE("f256 vs mpfr nextafter performance", "[bench][fltx][f256][nextafter]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& from, const auto& to) { return apply_nextafter(from, to); });
    print_bucketed_results("Floating-point utilities", "nextafter", results);
}

TEST_CASE("f256 vs mpfr nexttoward(type) performance", "[bench][fltx][f256][nexttoward]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& from, const auto& to) { return apply_nexttoward(from, to); });
    print_bucketed_results("Floating-point utilities", "nexttoward(type)", results);
}

TEST_CASE("f256 vs mpfr nexttoward(long double) performance", "[bench][fltx][f256][nexttoward]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_long_double_benchmark_result<f256, mpfr_ref>(atan2_value_buckets(), total_iterations, [](const auto& from, long double to) { return apply_nexttoward_long_double(from, to); });
    print_bucketed_results("Floating-point utilities", "nexttoward(long double)", results);
}

TEST_CASE("f256 vs mpfr floor performance", "[bench][fltx][f256][rounding]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_floor(values[i]); });
    print_bucketed_results("Rounding", "floor", results);
}

TEST_CASE("f256 vs mpfr ceil performance", "[bench][fltx][f256][rounding]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_ceil(values[i]); });
    print_bucketed_results("Rounding", "ceil", results);
}

TEST_CASE("f256 vs mpfr trunc performance", "[bench][fltx][f256][rounding]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_trunc(values[i]); });
    print_bucketed_results("Rounding", "trunc", results);
}

TEST_CASE("f256 vs mpfr round performance", "[bench][fltx][f256][rounding]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_round(values[i]); });
    print_bucketed_results("Rounding", "round", results);
}

TEST_CASE("f256 vs mpfr sqrt performance", "[bench][fltx][f256][sqrt]")
{
    const std::int64_t total_iterations = 8000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(positive_sqrt_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_sqrt(values[i]); });
    print_bucketed_results("Roots & powers", "sqrt", results);
}

TEST_CASE("f256 vs mpfr exp performance", "[bench][fltx][f256][exp]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(exponent_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_exp(values[i]); });
    print_bucketed_results("Exponentials", "exp", results);
}

TEST_CASE("f256 vs mpfr exp2 performance", "[bench][fltx][f256][exp2]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(exponent_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_exp2(values[i]); });
    print_bucketed_results("Exponentials", "exp2", results);
}

TEST_CASE("f256 vs mpfr ldexp performance", "[bench][fltx][f256][ldexp]")
{
    const std::int64_t total_iterations = 16000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_ldexp_benchmark(ldexp_value_buckets(), total_iterations);
    print_bucketed_results("Floating-point utilities", "ldexp", results);
}

TEST_CASE("f256 vs mpfr log performance", "[bench][fltx][f256][log]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(positive_log_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_log(values[i]); });
    print_bucketed_results("Logarithms", "log", results);
}

TEST_CASE("f256 vs mpfr log2 performance", "[bench][fltx][f256][log2]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(positive_log_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_log2(values[i]); });
    print_bucketed_results("Logarithms", "log2", results);
}

TEST_CASE("f256 vs mpfr log10 performance", "[bench][fltx][f256][log10]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(positive_log_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_log10(values[i]); });
    print_bucketed_results("Logarithms", "log10", results);
}

TEST_CASE("f256 vs mpfr fmod performance", "[bench][fltx][f256][fmod]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(fmod_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_fmod(x, y); });
    print_bucketed_results("Remainders", "fmod", results);
}

TEST_CASE("f256 vs mpfr pow performance", "[bench][fltx][f256][pow]")
{
    const std::int64_t total_iterations = 1500ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(pow_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_pow(x, y); });
    print_bucketed_results("Roots & powers", "pow", results);
}

TEST_CASE("f256 vs mpfr sin performance", "[bench][fltx][f256][transcendental][trig][sin]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_sin(values[i]); });
    print_bucketed_results("Trigonometric", "sin", results);
}

TEST_CASE("f256 vs mpfr cos performance", "[bench][fltx][f256][transcendental][trig][cos]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_cos(values[i]); });
    print_bucketed_results("Trigonometric", "cos", results);
}

TEST_CASE("f256 vs mpfr tan performance", "[bench][fltx][f256][transcendental][trig][tan]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_tan(values[i]); });
    print_bucketed_results("Trigonometric", "tan", results);
}

TEST_CASE("f256 vs mpfr atan performance", "[bench][fltx][f256][transcendental][trig][atan]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_atan(values[i]); });
    print_bucketed_results("Trigonometric", "atan", results);
}

TEST_CASE("f256 vs mpfr atan2 performance", "[bench][fltx][f256][transcendental][trig][atan2]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(atan2_value_buckets(), total_iterations, [](const auto& y, const auto& x) { return apply_atan2(y, x); });
    print_bucketed_results("Trigonometric", "atan2", results);
}

TEST_CASE("f256 vs mpfr asin performance", "[bench][fltx][f256][transcendental][trig][asin]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(inverse_trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_asin(values[i]); });
    print_bucketed_results("Trigonometric", "asin", results);
}

TEST_CASE("f256 vs mpfr acos performance", "[bench][fltx][f256][transcendental][trig][acos]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(inverse_trig_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_acos(values[i]); });
    print_bucketed_results("Trigonometric", "acos", results);
}


TEST_CASE("f256 vs mpfr nearbyint performance", "[bench][fltx][f256][rounding][nearbyint]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_nearbyint(values[i]); });
    print_bucketed_results("Rounding", "nearbyint", results);
}

TEST_CASE("f256 vs mpfr rint performance", "[bench][fltx][f256][rounding][rint]")
{
    const std::int64_t total_iterations = 30000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(rounding_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_rint(values[i]); });
    print_bucketed_results("Rounding", "rint", results);
}

TEST_CASE("f256 vs mpfr cbrt performance", "[bench][fltx][f256][cbrt]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(generic_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_cbrt(values[i]); });
    print_bucketed_results("Roots & powers", "cbrt", results);
}

TEST_CASE("f256 vs mpfr hypot performance", "[bench][fltx][f256][hypot]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(atan2_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_hypot(x, y); });
    print_bucketed_results("Roots & powers", "hypot", results);
}

TEST_CASE("f256 vs mpfr expm1 performance", "[bench][fltx][f256][expm1]")
{
    const std::int64_t total_iterations = 2500ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(exponent_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_expm1(values[i]); });
    print_bucketed_results("Exponentials", "expm1", results);
}

TEST_CASE("f256 vs mpfr log1p performance", "[bench][fltx][f256][log1p]")
{
    const std::int64_t total_iterations = 2500ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(log1p_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_log1p(values[i]); });
    print_bucketed_results("Logarithms", "log1p", results);
}

TEST_CASE("f256 vs mpfr remainder performance", "[bench][fltx][f256][remainder]")
{
    const std::int64_t total_iterations = 1500ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(fmod_value_buckets(), total_iterations, [](const auto& x, const auto& y) { return apply_remainder(x, y); });
    print_bucketed_results("Remainders", "remainder", results);
}

TEST_CASE("f256 vs mpfr sinh performance", "[bench][fltx][f256][transcendental][hyperbolic][sinh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(hyperbolic_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_sinh(values[i]); });
    print_bucketed_results("Hyperbolic", "sinh", results);
}

TEST_CASE("f256 vs mpfr cosh performance", "[bench][fltx][f256][transcendental][hyperbolic][cosh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(hyperbolic_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_cosh(values[i]); });
    print_bucketed_results("Hyperbolic", "cosh", results);
}

TEST_CASE("f256 vs mpfr tanh performance", "[bench][fltx][f256][transcendental][hyperbolic][tanh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(hyperbolic_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_tanh(values[i]); });
    print_bucketed_results("Hyperbolic", "tanh", results);
}

TEST_CASE("f256 vs mpfr asinh performance", "[bench][fltx][f256][transcendental][inverse_hyperbolic][asinh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(hyperbolic_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_asinh(values[i]); });
    print_bucketed_results("Inverse hyperbolic", "asinh", results);
}

TEST_CASE("f256 vs mpfr acosh performance", "[bench][fltx][f256][transcendental][inverse_hyperbolic][acosh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(acosh_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_acosh(values[i]); });
    print_bucketed_results("Inverse hyperbolic", "acosh", results);
}

TEST_CASE("f256 vs mpfr atanh performance", "[bench][fltx][f256][transcendental][inverse_hyperbolic][atanh]")
{
    const std::int64_t total_iterations = 2000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(atanh_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_atanh(values[i]); });
    print_bucketed_results("Inverse hyperbolic", "atanh", results);
}

TEST_CASE("f256 vs mpfr erf performance", "[bench][fltx][f256][transcendental][erf]")
{
    const std::int64_t total_iterations = 800ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(erf_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_erf(values[i]); });
    print_bucketed_results("Special functions", "erf", results);
}

TEST_CASE("f256 vs mpfr erfc performance", "[bench][fltx][f256][transcendental][erfc]")
{
    const std::int64_t total_iterations = 800ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(erf_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_erfc(values[i]); });
    print_bucketed_results("Special functions", "erfc", results);
}

TEST_CASE("f256 vs mpfr lgamma performance", "[bench][fltx][f256][transcendental][gamma][lgamma]")
{
    const std::int64_t total_iterations = 120ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(gamma_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_lgamma(values[i]); });
    print_bucketed_results("Special functions", "lgamma", results);
}

TEST_CASE("f256 vs mpfr tgamma performance", "[bench][fltx][f256][transcendental][gamma][tgamma]")
{
    const std::int64_t total_iterations = 120ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark(gamma_value_buckets(), total_iterations, [](const auto& values, std::size_t i) { return apply_tgamma(values[i]); });
    print_bucketed_results("Special functions", "tgamma", results);
}
