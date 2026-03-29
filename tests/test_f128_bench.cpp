#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>

#include <fltx/f128.h>

using namespace bl;

namespace
{
    using mpfr_ref = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<40>>;
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 75;

    constexpr std::array<const char*, 8> benchmark_value_texts = {{
        "-0.79028751130577388854871201204656",
        "0.15774501904375255339036464359358",
        "3.1415926535897932384626433832795",
        "2.7182818284590452353602874713527",
        "1.0000000000000000000000000000001",
        "0.9999999999999999999999999999999",
        "1234567890123456.125",
        "0.000000000000000000000000000125"
    }};

    struct bench_result
    {
        double total_ms = 0.0;
        double ns_per_iter = 0.0;
    };

    volatile double benchmark_sink = 0.0;

    void consume_result(const f128& value)
    {
        benchmark_sink += static_cast<double>(value);
    }

    void consume_result(const mpfr_ref& value)
    {
        benchmark_sink += static_cast<double>(value);
    }

    template<typename T>
    void consume_result(const std::pair<T, T>& value)
    {
        consume_result(value.first);
        consume_result(value.second);
    }

    template<typename T>
    [[nodiscard]] T parse_benchmark_value(const char* text);

    template<>
    [[nodiscard]] f128 parse_benchmark_value<f128>(const char* text)
    {
        return to_f128(text);
    }

    template<>
    [[nodiscard]] mpfr_ref parse_benchmark_value<mpfr_ref>(const char* text)
    {
        return mpfr_ref(text);
    }

    template<typename T>
    [[nodiscard]] std::array<T, benchmark_value_texts.size()> make_values()
    {
        std::array<T, benchmark_value_texts.size()> values{};

        for (std::size_t i = 0; i < benchmark_value_texts.size(); ++i)
            values[i] = parse_benchmark_value<T>(benchmark_value_texts[i]);

        return values;
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
        return result;
    }

    void print_result(const char* label, const bench_result& f128_result, const bench_result& mpfr_result)
    {
        const double ratio = mpfr_result.ns_per_iter / f128_result.ns_per_iter;

        std::cout
            << std::fixed << std::setprecision(2)
            << label
            << "\n  f128 : " << f128_result.total_ms << " ms total, " << f128_result.ns_per_iter << " ns/iter"
            << "\n  mpfr : " << mpfr_result.total_ms << " ms total, " << mpfr_result.ns_per_iter << " ns/iter"
            << "\n  mpfr/f128 ratio: " << ratio << "x"
            << "\n";
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_add_sub(std::int64_t outer_loops)
    {
        constexpr std::size_t value_count = benchmark_value_texts.size();
        const auto values = make_values<T>();
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < value_count; ++i)
                {
                    const T& a = values[i];
                    const T& b = values[(i + 1) % value_count];

                    acc = ((acc + a) - b) + (a - b);
                }
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_mul(std::int64_t outer_loops)
    {
        constexpr std::size_t value_count = benchmark_value_texts.size();
        const auto values = make_values<T>();
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < value_count; ++i)
                {
                    const T& a = values[i];
                    const T& b = values[(i + 3) % value_count];

                    acc = (acc * a) + (a * b);
                }
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_div(std::int64_t outer_loops)
    {
        constexpr std::size_t value_count = benchmark_value_texts.size();
        const auto values = make_values<T>();
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(value_count);

        return run_benchmark(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < value_count; ++i)
                {
                    const T& numerator = values[(i + 2) % value_count];
                    const T& denominator = values[(i + 5) % value_count];

                    acc = (acc + numerator) / (denominator + T(1));
                }
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_mixed_recurrence(std::int64_t iteration_count)
    {
        const auto values = make_values<T>();

        return run_benchmark(iteration_count, [&]()
        {
            T x = values[0];
            T y = values[1];
            const T a = values[2];
            const T b = values[3];
            const T c = values[4];
            const T d = values[5];

            for (std::int64_t i = 0; i < iteration_count; ++i)
            {
                const T xx = x * x;
                const T yy = y * y;
                const T xy = x * y;

                x = (xx - yy) + a;
                y = (xy + xy) + b;
                x = x / (c + T(0.5));
                y = y / (d + T(1.5));
            }

            return std::pair<T, T>{ x, y };
        });
    }
}

TEST_CASE("f128 vs mpfr add/sub performance", "[bench][fltx][f128]")
{
    const std::int64_t loops = 50000 * benchmark_scale;

    const auto f128_result = benchmark_add_sub<f128>(loops);
    const auto mpfr_result = benchmark_add_sub<mpfr_ref>(loops);

    print_result("add/sub", f128_result, mpfr_result);
}

TEST_CASE("f128 vs mpfr multiply performance", "[bench][fltx][f128]")
{
    const std::int64_t loops = 20000 * benchmark_scale;

    const auto f128_result = benchmark_mul<f128>(loops);
    const auto mpfr_result = benchmark_mul<mpfr_ref>(loops);

    print_result("multiply", f128_result, mpfr_result);
}

TEST_CASE("f128 vs mpfr divide performance", "[bench][fltx][f128]")
{
    const std::int64_t loops = 12000 * benchmark_scale;

    const auto f128_result = benchmark_div<f128>(loops);
    const auto mpfr_result = benchmark_div<mpfr_ref>(loops);

    print_result("divide", f128_result, mpfr_result);
}

TEST_CASE("f128 vs mpfr mixed recurrence performance", "[bench][fltx][f128]")
{
    const std::int64_t iterations = 40000 * benchmark_scale;

    const auto f128_result = benchmark_mixed_recurrence<f128>(iterations);
    const auto mpfr_result = benchmark_mixed_recurrence<mpfr_ref>(iterations);

    print_result("mixed recurrence", f128_result, mpfr_result);
}
