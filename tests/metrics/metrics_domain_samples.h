#ifndef FLTX_TESTS_METRICS_DOMAIN_SAMPLES_INCLUDED
#define FLTX_TESTS_METRICS_DOMAIN_SAMPLES_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "metrics_samples.h"

namespace bl::test::metrics
{
    enum class domain_value_kind
    {
        wide_real,
        reduction_real,
        balanced_real,
        positive,
        positive_balanced,
        nonzero_wide_real,
        nonzero_balanced_real,
        unit_closed,
        unit_open,
        exp_argument,
        exp2_argument,
        hyperbolic_argument,
        log1p_argument,
        acosh_argument,
        gamma_argument,
        tgamma_argument,
        integer_rounding_argument,
        pow_exponent
    };

    class domain_sample_rng
    {
    public:
        explicit constexpr domain_sample_rng(std::uint64_t seed) noexcept
            : state(seed == 0 ? 0x9e3779b97f4a7c15ull : seed)
        {
        }

        [[nodiscard]] std::uint64_t next() noexcept
        {
            std::uint64_t x = state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            state = x;
            return x * 0x2545f4914f6cdd1dull;
        }

        [[nodiscard]] double unit() noexcept
        {
            return static_cast<double>(next() >> 11) * 0x1.0p-53;
        }

        [[nodiscard]] int integer(int min, int max) noexcept
        {
            const auto span = static_cast<std::uint64_t>(max - min + 1);
            return min + static_cast<int>(next() % span);
        }

        [[nodiscard]] double sign() noexcept
        {
            return (next() & 1ull) == 0 ? -1.0 : 1.0;
        }

    private:
        std::uint64_t state;
    };

    [[nodiscard]] inline std::uint64_t domain_seed(std::string_view text, std::uint64_t salt = 0) noexcept
    {
        std::uint64_t value = 1469598103934665603ull ^ salt;
        for (char c : text)
        {
            value ^= static_cast<unsigned char>(c);
            value *= 1099511628211ull;
        }
        return value;
    }

    [[nodiscard]] inline std::size_t configured_domain_random_sample_count(std::size_t fallback) noexcept
    {
        const char* text = std::getenv("FLTX_DOMAIN_RANDOM_SAMPLES");
        if (text == nullptr || *text == '\0')
            return fallback;

        char* end = nullptr;
        const unsigned long long value = std::strtoull(text, &end, 10);
        return end != text && value > 0ull
            ? static_cast<std::size_t>(value)
            : fallback;
    }

    [[nodiscard]] inline bool domain_trace_enabled() noexcept
    {
        static const bool enabled = []
        {
            const char* text = std::getenv("FLTX_DOMAIN_TRACE");
            return text != nullptr && *text != '\0' && *text != '0';
        }();
        return enabled;
    }

    inline void trace_domain_value(const value_sample& value)
    {
        std::cerr << std::hexfloat << value.hi;
        if (value.lo != 0.0 || value.x2 != 0.0 || value.x3 != 0.0)
            std::cerr << " + [" << value.lo << ", " << value.x2 << ", " << value.x3 << ']';
        std::cerr << std::defaultfloat;
    }

    inline void trace_domain_sample(
        std::string_view operation,
        std::string_view backend_name,
        std::size_t index,
        const unary_sample& sample)
    {
        if (!domain_trace_enabled())
            return;
        std::cerr << "[domain trace] " << operation << ' ' << backend_name << " sample " << index << " x=";
        trace_domain_value(sample.x);
        std::cerr << '\n';
    }

    inline void trace_domain_sample(
        std::string_view operation,
        std::string_view backend_name,
        std::size_t index,
        const binary_sample& sample)
    {
        if (!domain_trace_enabled())
            return;
        std::cerr << "[domain trace] " << operation << ' ' << backend_name << " sample " << index << " x=";
        trace_domain_value(sample.x);
        std::cerr << " y=";
        trace_domain_value(sample.y);
        std::cerr << '\n';
    }

    inline void trace_domain_sample(
        std::string_view operation,
        std::string_view backend_name,
        std::size_t index,
        const ternary_sample& sample)
    {
        if (!domain_trace_enabled())
            return;
        std::cerr << "[domain trace] " << operation << ' ' << backend_name << " sample " << index << " x=";
        trace_domain_value(sample.x);
        std::cerr << " y=";
        trace_domain_value(sample.y);
        std::cerr << " z=";
        trace_domain_value(sample.z);
        std::cerr << '\n';
    }

    inline void trace_domain_sample(
        std::string_view operation,
        std::string_view backend_name,
        std::size_t index,
        const unary_int_sample& sample)
    {
        if (!domain_trace_enabled())
            return;
        std::cerr << "[domain trace] " << operation << ' ' << backend_name << " sample " << index << " x=";
        trace_domain_value(sample.x);
        std::cerr << " n=" << sample.n << '\n';
    }

    inline void trace_domain_result(
        std::string_view operation,
        std::string_view backend_name,
        std::size_t index,
        double bits)
    {
        if (!domain_trace_enabled())
            return;
        std::cerr << "[domain trace] " << operation << ' ' << backend_name << " sample " << index
                  << " bits=" << bits << '\n';
    }

    [[nodiscard]] inline bool domain_is_nonzero(domain_value_kind kind) noexcept
    {
        return kind == domain_value_kind::nonzero_wide_real ||
               kind == domain_value_kind::nonzero_balanced_real;
    }

    [[nodiscard]] inline bool domain_supports_residual(domain_value_kind kind) noexcept
    {
        switch (kind)
        {
        case domain_value_kind::wide_real:
        case domain_value_kind::reduction_real:
        case domain_value_kind::balanced_real:
        case domain_value_kind::positive:
        case domain_value_kind::positive_balanced:
        case domain_value_kind::nonzero_wide_real:
        case domain_value_kind::nonzero_balanced_real:
            return true;
        case domain_value_kind::unit_closed:
        case domain_value_kind::unit_open:
        case domain_value_kind::exp_argument:
        case domain_value_kind::exp2_argument:
        case domain_value_kind::hyperbolic_argument:
        case domain_value_kind::log1p_argument:
        case domain_value_kind::acosh_argument:
        case domain_value_kind::gamma_argument:
        case domain_value_kind::tgamma_argument:
        case domain_value_kind::pow_exponent:
        case domain_value_kind::integer_rounding_argument:
            return false;
        }
        return false;
    }

    [[nodiscard]] inline bool domain_accepts(domain_value_kind kind, double value) noexcept
    {
        if (!std::isfinite(value))
            return false;

        const double abs_value = std::fabs(value);
        const double wide_limit = std::ldexp(1.0, 900);
        const double reduction_limit = std::ldexp(1.0, 20);
        const double balanced_limit = std::ldexp(1.0, 450);
        const double balanced_min = std::ldexp(1.0, -450);
        const double ordinary_argument_min = std::ldexp(1.0, -64);

        switch (kind)
        {
        case domain_value_kind::wide_real:
            return abs_value <= wide_limit;
        case domain_value_kind::reduction_real:
            return abs_value <= reduction_limit;
        case domain_value_kind::balanced_real:
            return abs_value <= balanced_limit;
        case domain_value_kind::positive:
            return value > 0.0 && value <= wide_limit;
        case domain_value_kind::positive_balanced:
            return value >= balanced_min && value <= balanced_limit;
        case domain_value_kind::nonzero_wide_real:
            return value != 0.0 && abs_value <= wide_limit;
        case domain_value_kind::nonzero_balanced_real:
            return abs_value >= balanced_min && abs_value <= balanced_limit;
        case domain_value_kind::unit_closed:
            return value >= -1.0 && value <= 1.0;
        case domain_value_kind::unit_open:
            return value > -1.0 && value < 1.0;
        case domain_value_kind::exp_argument:
            return value >= -700.0 && value <= 700.0;
        case domain_value_kind::exp2_argument:
            return value >= -1020.0 && value <= 1020.0;
        case domain_value_kind::hyperbolic_argument:
            return value == 0.0 ||
                (abs_value >= ordinary_argument_min && value >= -700.0 && value <= 700.0);
        case domain_value_kind::log1p_argument:
            return value > -1.0 && value <= wide_limit;
        case domain_value_kind::acosh_argument:
            return value >= 1.0 && value <= wide_limit;
        case domain_value_kind::gamma_argument:
            if (value < -30.0 || value > 170.0)
                return false;
            return value > 0.0 || std::fabs(value - std::round(value)) > 0.000000000001;
        case domain_value_kind::tgamma_argument:
            if (value < -30.0 || value > 170.0)
                return false;
            if (value > 0.0)
                return value >= std::ldexp(1.0, -1023);
            return std::fabs(value - std::round(value)) > 0.000000000001;
        case domain_value_kind::integer_rounding_argument:
            return value >= -1000000.0 && value <= 1000000.0;
        case domain_value_kind::pow_exponent:
            return value >= -16.0 && value <= 16.0;
        }
        return false;
    }

    inline void append_domain_value(
        std::vector<value_sample>& values,
        domain_value_kind kind,
        double hi,
        double lo = 0.0,
        double x2 = 0.0,
        double x3 = 0.0)
    {
        if (!domain_accepts(kind, hi))
            return;
        if (((hi + lo) + (x2 + x3)) == 0.0 && domain_is_nonzero(kind))
            return;
        values.push_back({ "domain", hi, lo, x2, x3 });
    }

    inline void append_domain_neighborhood(
        std::vector<value_sample>& values,
        domain_value_kind kind,
        double center)
    {
        append_domain_value(values, kind, center);
        append_domain_value(values, kind, std::nextafter(center, -std::numeric_limits<double>::infinity()));
        append_domain_value(values, kind, std::nextafter(center, std::numeric_limits<double>::infinity()));

        if (center != 0.0 && std::isfinite(center) && domain_supports_residual(kind))
        {
            const int exponent = std::ilogb(std::fabs(center));
            append_domain_value(
                values,
                kind,
                center,
                std::ldexp(1.0, exponent - 60),
                -std::ldexp(1.0, exponent - 113),
                std::ldexp(1.0, exponent - 166));
            append_domain_value(
                values,
                kind,
                center,
                -std::ldexp(1.0, exponent - 60),
                std::ldexp(1.0, exponent - 113),
                -std::ldexp(1.0, exponent - 166));
        }
    }

    inline void append_domain_landmarks(std::vector<value_sample>& values, domain_value_kind kind)
    {
        constexpr double pi = 3.14159265358979323846264338327950288;
        constexpr double e = 2.71828182845904523536028747135266250;
        constexpr double ln2 = 0.69314718055994530941723212145817657;
        constexpr double sqrt2 = 1.41421356237309504880168872420969808;

        const double centers[] = {
            0.0,
            std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::min(),
            std::numeric_limits<double>::epsilon(),
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            10.0,
            pi / 6.0,
            pi / 4.0,
            pi / 2.0,
            pi,
            2.0 * pi,
            e,
            ln2,
            sqrt2
        };

        for (double center : centers)
        {
            append_domain_neighborhood(values, kind, center);
            append_domain_neighborhood(values, kind, -center);
        }

        const int exponents[] = {
            -1022, -900, -768, -512, -256, -128, -64, -32, -16, -8, -4, -1,
            0, 1, 4, 8, 16, 32, 64, 128, 256, 512, 768, 900, 1023
        };
        for (int exponent : exponents)
        {
            const double value = std::ldexp(1.0, exponent);
            append_domain_neighborhood(values, kind, value);
            append_domain_neighborhood(values, kind, -value);
        }

        for (int integer = -8; integer <= 8; ++integer)
        {
            const double value = static_cast<double>(integer);
            append_domain_neighborhood(values, kind, value);
            append_domain_neighborhood(values, kind, value + 0.5);
        }
    }

    [[nodiscard]] inline double signed_log_value(domain_sample_rng& rng, int min_exp, int max_exp)
    {
        const int exponent = rng.integer(min_exp, max_exp);
        return rng.sign() * std::ldexp(1.0 + rng.unit(), exponent);
    }

    [[nodiscard]] inline double positive_log_value(domain_sample_rng& rng, int min_exp, int max_exp)
    {
        const int exponent = rng.integer(min_exp, max_exp);
        return std::ldexp(1.0 + rng.unit(), exponent);
    }

    [[nodiscard]] inline double bounded_signed_value(domain_sample_rng& rng, double limit)
    {
        if ((rng.next() & 3ull) == 0)
            return rng.sign() * std::ldexp(1.0 + rng.unit(), rng.integer(-30, static_cast<int>(std::log2(limit))));
        return (rng.unit() * 2.0 - 1.0) * limit;
    }

    [[nodiscard]] inline double random_domain_value(domain_value_kind kind, domain_sample_rng& rng)
    {
        switch (kind)
        {
        case domain_value_kind::wide_real:
        case domain_value_kind::nonzero_wide_real:
            return signed_log_value(rng, -900, 900);
        case domain_value_kind::reduction_real:
            return signed_log_value(rng, -20, 20);
        case domain_value_kind::balanced_real:
        case domain_value_kind::nonzero_balanced_real:
            return signed_log_value(rng, -450, 450);
        case domain_value_kind::positive:
            return positive_log_value(rng, -900, 900);
        case domain_value_kind::positive_balanced:
            return positive_log_value(rng, -450, 450);
        case domain_value_kind::unit_closed:
            return rng.unit() * 2.0 - 1.0;
        case domain_value_kind::unit_open:
            return std::nextafter(rng.unit() * 2.0 - 1.0, 0.0);
        case domain_value_kind::exp_argument:
            return bounded_signed_value(rng, 700.0);
        case domain_value_kind::exp2_argument:
            return bounded_signed_value(rng, 1020.0);
        case domain_value_kind::hyperbolic_argument:
            return bounded_signed_value(rng, 700.0);
        case domain_value_kind::log1p_argument:
            if ((rng.next() & 3ull) == 0)
                return -1.0 + positive_log_value(rng, -900, -1);
            return signed_log_value(rng, -900, 900);
        case domain_value_kind::acosh_argument:
            if ((rng.next() & 3ull) == 0)
                return 1.0 + positive_log_value(rng, -900, -1);
            return 1.0 + positive_log_value(rng, -900, 900);
        case domain_value_kind::gamma_argument:
        case domain_value_kind::tgamma_argument:
        {
            double value = rng.unit() * 200.0 - 30.0;
            if (value <= 0.0 && std::fabs(value - std::round(value)) < 0.03125)
                value += 0.125;
            return value;
        }
        case domain_value_kind::integer_rounding_argument:
            return rng.unit() * 2000000.0 - 1000000.0;
        case domain_value_kind::pow_exponent:
            return bounded_signed_value(rng, 16.0);
        }
        return 0.0;
    }

    [[nodiscard]] inline value_sample make_random_domain_sample(
        domain_value_kind kind,
        domain_sample_rng& rng)
    {
        double hi = 0.0;
        do
        {
            hi = random_domain_value(kind, rng);
        }
        while (!domain_accepts(kind, hi) || (hi == 0.0 && domain_is_nonzero(kind)));

        double lo = 0.0;
        double x2 = 0.0;
        double x3 = 0.0;
        if (hi != 0.0 && std::isfinite(hi) && domain_supports_residual(kind))
        {
            const int exponent = std::ilogb(std::fabs(hi));
            lo = rng.sign() * std::ldexp(0.5 + rng.unit() * 0.5, exponent - 60);
            x2 = rng.sign() * std::ldexp(0.5 + rng.unit() * 0.5, exponent - 113);
            x3 = rng.sign() * std::ldexp(0.5 + rng.unit() * 0.5, exponent - 166);
        }

        return { "domain", hi, lo, x2, x3 };
    }

    [[nodiscard]] inline std::vector<value_sample> make_domain_values(
        domain_value_kind kind,
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<value_sample> values;
        values.reserve(random_count + 256);
        append_domain_landmarks(values, kind);

        domain_sample_rng rng{ seed };
        for (std::size_t index = 0; index < random_count; ++index)
            values.push_back(make_random_domain_sample(kind, rng));

        return values;
    }

    [[nodiscard]] inline std::vector<unary_sample> make_domain_unary_values(
        domain_value_kind kind,
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<unary_sample> samples;
        const auto values = make_domain_values(kind, random_count, seed);
        samples.reserve(values.size());
        for (const value_sample& value : values)
            samples.push_back({ "domain", value });
        return samples;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_domain_binary_values(
        domain_value_kind x_kind,
        domain_value_kind y_kind,
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<binary_sample> samples;
        const auto x_values = make_domain_values(x_kind, random_count, seed ^ 0x8da6b343u);
        const auto y_values = make_domain_values(y_kind, random_count, seed ^ 0xd8163841u);
        const std::size_t count = std::min(x_values.size(), y_values.size());
        samples.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            samples.push_back({ "domain", x_values[index], y_values[(index * 37u + 11u) % count] });
        return samples;
    }

    [[nodiscard]] inline long double domain_value_estimate(const value_sample& value) noexcept
    {
        return static_cast<long double>(value.hi) +
               static_cast<long double>(value.lo) +
               static_cast<long double>(value.x2) +
               static_cast<long double>(value.x3);
    }

    [[nodiscard]] inline bool pow_domain_result_is_representable(
        const value_sample& base,
        const value_sample& exponent) noexcept
    {
        const long double base_estimate = domain_value_estimate(base);
        const long double exponent_estimate = domain_value_estimate(exponent);
        if (!(base_estimate > 0.0L) || !std::isfinite(base_estimate) || !std::isfinite(exponent_estimate))
            return false;

        constexpr long double max_finite_result_log2 = 1000.0L;
        const long double result_log2 = std::log2(base_estimate) * exponent_estimate;
        return std::isfinite(result_log2) &&
               result_log2 >= -max_finite_result_log2 &&
               result_log2 <= max_finite_result_log2;
    }

    [[nodiscard]] inline std::vector<binary_sample> make_pow_domain_binary_values(
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<binary_sample> samples;
        samples.reserve(random_count + 128);

        std::vector<value_sample> base_landmarks;
        std::vector<value_sample> exponent_landmarks;
        append_domain_landmarks(base_landmarks, domain_value_kind::positive_balanced);
        append_domain_landmarks(exponent_landmarks, domain_value_kind::pow_exponent);

        const std::size_t landmark_count = std::min(base_landmarks.size(), exponent_landmarks.size());
        for (std::size_t index = 0; index < landmark_count; ++index)
        {
            const value_sample& base = base_landmarks[index];
            const value_sample& exponent = exponent_landmarks[(index * 37u + 11u) % landmark_count];
            if (pow_domain_result_is_representable(base, exponent))
                samples.push_back({ "domain", base, exponent });
        }

        domain_sample_rng rng{ seed ^ 0x6eed0e9da4d94a4bull };
        std::size_t accepted_random = 0;
        std::size_t attempts = 0;
        const std::size_t max_attempts = std::max<std::size_t>(random_count * 128u, 1024u);
        while (accepted_random < random_count && attempts++ < max_attempts)
        {
            const value_sample base = make_random_domain_sample(domain_value_kind::positive_balanced, rng);
            const value_sample exponent = make_random_domain_sample(domain_value_kind::pow_exponent, rng);
            if (!pow_domain_result_is_representable(base, exponent))
                continue;

            samples.push_back({ "domain", base, exponent });
            ++accepted_random;
        }

        return samples;
    }

    [[nodiscard]] inline std::vector<ternary_sample> make_domain_ternary_values(
        domain_value_kind x_kind,
        domain_value_kind y_kind,
        domain_value_kind z_kind,
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<ternary_sample> samples;
        const auto x_values = make_domain_values(x_kind, random_count, seed ^ 0x27d4eb2du);
        const auto y_values = make_domain_values(y_kind, random_count, seed ^ 0x165667b1u);
        const auto z_values = make_domain_values(z_kind, random_count, seed ^ 0x9e3779b9u);
        const std::size_t count = std::min({ x_values.size(), y_values.size(), z_values.size() });
        samples.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            samples.push_back({
                "domain",
                x_values[index],
                y_values[(index * 37u + 11u) % count],
                z_values[(index * 53u + 17u) % count]
            });
        }
        return samples;
    }

    inline void constrain_scaling_shift(double value, int& min_shift, int& max_shift) noexcept
    {
        if (value == 0.0 || !std::isfinite(value))
            return;

        int exponent = 0;
        (void)std::frexp(std::fabs(value), &exponent);
        min_shift = std::max(min_shift, -1073 - exponent);
        max_shift = std::min(max_shift, 1024 - exponent);
    }

    [[nodiscard]] inline bool scaling_shift_bounds(
        const value_sample& value,
        int& min_shift,
        int& max_shift) noexcept
    {
        min_shift = std::numeric_limits<int>::min();
        max_shift = std::numeric_limits<int>::max();
        constrain_scaling_shift(value.hi, min_shift, max_shift);
        constrain_scaling_shift(value.lo, min_shift, max_shift);
        constrain_scaling_shift(value.x2, min_shift, max_shift);
        constrain_scaling_shift(value.x3, min_shift, max_shift);
        return min_shift <= max_shift;
    }

    [[nodiscard]] inline std::vector<unary_int_sample> make_domain_unary_int_values(
        domain_value_kind kind,
        std::size_t random_count,
        std::uint64_t seed)
    {
        std::vector<unary_int_sample> samples;
        const auto values = make_domain_values(kind, random_count, seed);
        samples.reserve(values.size());
        domain_sample_rng rng{ seed ^ 0x94d049bb133111ebull };
        const int fixed_exponents[] = {
            -1024, -900, -512, -128, -32, -1, 0, 1, 32, 128, 512, 900, 1024
        };
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            int min_shift = 0;
            int max_shift = 0;
            if (!scaling_shift_bounds(values[index], min_shift, max_shift))
                continue;

            int n = 0;
            if (index < sizeof(fixed_exponents) / sizeof(fixed_exponents[0]))
            {
                n = std::clamp(fixed_exponents[index], min_shift, max_shift);
            }
            else
            {
                const int random_min = std::max(min_shift, -900);
                const int random_max = std::min(max_shift, 900);
                if (random_min > random_max)
                    continue;
                n = rng.integer(random_min, random_max);
            }
            samples.push_back({ "domain", values[index], n });
        }
        return samples;
    }
}

#endif
