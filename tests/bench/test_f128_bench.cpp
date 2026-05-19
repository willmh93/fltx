#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#define FLTX_INLINE_LEVEL 2
#include <f128_math.h>
#include <f128_io.h>
#include "benchmark_chart_writer.h"

using namespace bl;

namespace
{
    constexpr unsigned mpfr_digits10 = std::numeric_limits<f128>::digits10;
    using mpfr_ref = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<mpfr_digits10>>;
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 10;
    constexpr bool only_bench_typical = true;
    constexpr bool generate_compact_report = true;
    constexpr std::size_t bucket_value_count = 64;
    constexpr std::size_t bucket_count = 3;
    constexpr std::size_t atan_typical_value_count = 4096;
    constexpr double benchmark_pi = 3.141592653589793238462643383279502884;
    constexpr int mandelbrot_kernel_width = 32;
    constexpr int mandelbrot_kernel_height = 32;
    constexpr const char* mixed_recurrence_square_diff_label = "((x*x - y*y) + a) / (c + scalar)";
    constexpr const char* mixed_recurrence_product_sum_label = "((x*a + y*b) + c) / (c + scalar)";
    constexpr const char* mixed_recurrence_scaled_product_sum_label = "((x*sa + y*sb) + a) / (c + scalar)";
    constexpr const char* mixed_recurrence_shifted_sum_label = "((x + a) + b) / ((c + d) + scalar)";
    constexpr const char* mixed_recurrence_three_product_sum_label = "((x*a + y*b) + c*d) / ((c + d) + scalar)";
    constexpr const char* scalar_mixed_recurrence_label = "((x + scalar)*scalar + a) * (scalar*(y + scalar) - b)";
    constexpr const char* fused_mixed_expression_label = "((x*y + a*b) + c) / (c*c + scalar)";
    constexpr const char* affine_trig_transform_label = bl::bench::mixed_affine_trig_transform_label;
    constexpr const char* mandelbrot_kernel_label = bl::bench::mixed_mandelbrot_kernel_label;

    bl::bench::benchmark_chart_writer chart_writer{
        "f128",
        "mpfr",
        "f128 vs MPFR typical benchmark ratios",
        bl::bench::benchmark_output_path("f128", "typical_ratios", "csv"),
        bl::bench::benchmark_output_path("f128", "typical_ratios", "svg"),
        8.0,
        generate_compact_report
    };

    struct bench_result
    {
        double total_ms = 0.0;
        double ns_per_iter = 0.0;
        std::int64_t iteration_count = 0;
    };

    struct comparison_result
    {
        bench_result f128{};
        bench_result mpfr{};
    };

    struct bucketed_comparison_result
    {
        comparison_result easy{};
        comparison_result medium{};
        comparison_result hard{};
        comparison_result typical{};
    };

    struct value_spec
    {
        double hi = 0.0;
        double lo = 0.0;
    };

    struct binary_value_spec
    {
        value_spec lhs{};
        value_spec rhs{};
    };

    struct recurrence_value_spec
    {
        value_spec x{};
        value_spec y{};
        value_spec a{};
        value_spec b{};
        value_spec c{};
        value_spec d{};
    };

    struct affine_transform_value_spec
    {
        value_spec x{};
        value_spec y{};
        value_spec angle{};
        value_spec scale_x{};
        value_spec scale_y{};
        value_spec translate_x{};
        value_spec translate_y{};
    };

    enum class mixed_recurrence_workload
    {
        one,
        two,
        three,
        four,
        five
    };

    struct ldexp_value_spec
    {
        value_spec value{};
        int exponent = 0;
    };

    template<typename Spec>
    struct bucket_array_set
    {
        std::array<Spec, bucket_value_count> easy{};
        std::array<Spec, bucket_value_count> medium{};
        std::array<Spec, bucket_value_count> hard{};
    };

    template<typename T>
    struct recurrence_case
    {
        T x{};
        T y{};
        T a{};
        T b{};
        T denom_x{};
        T denom_y{};
    };

    volatile double benchmark_sink = 0.0;
    volatile std::int64_t benchmark_integer_sink = 0;

    void consume_result(const f128& value)
    {
        benchmark_sink += static_cast<double>(value);
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

    [[nodiscard]] double random_unit(std::mt19937_64& rng)
    {
        return std::generate_canonical<double, 53>(rng);
    }

    [[nodiscard]] double random_real(std::mt19937_64& rng, double lo, double hi)
    {
        return lo + (hi - lo) * random_unit(rng);
    }

    [[nodiscard]] int random_int(std::mt19937_64& rng, int lo, int hi)
    {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    }

    [[nodiscard]] bool random_bool(std::mt19937_64& rng)
    {
        return (rng() & 1ull) != 0ull;
    }

    [[nodiscard]] double random_sign(std::mt19937_64& rng)
    {
        return random_bool(rng) ? -1.0 : 1.0;
    }

    [[nodiscard]] double ulp_of(double x)
    {
        if (x == 0.0)
            return std::numeric_limits<double>::denorm_min();

        int exp2 = 0;
        (void)std::frexp(std::abs(x), &exp2);
        return std::ldexp(1.0, exp2 - 53);
    }

    [[nodiscard]] value_spec renorm_spec(double hi, double lo)
    {
        const f128 r = bl::detail::_f128::renorm(hi, lo);
        return { r.hi, r.lo };
    }

    [[nodiscard]] value_spec spec_from_f128(const f128& value)
    {
        return { value.hi, value.lo };
    }

    [[nodiscard]] value_spec spec_from_text(const char* text)
    {
        return spec_from_f128(f128{ to_f128(text) });
    }

    [[nodiscard]] value_spec make_random_value_spec(
        std::mt19937_64& rng,
        bool positive,
        bool with_lo,
        int exp_lo,
        int exp_hi,
        double lo_scale_lo = 0.05,
        double lo_scale_hi = 0.95)
    {
        const double mantissa = random_real(rng, 0.5, 1.9999999999999998);
        double hi = std::ldexp(mantissa, random_int(rng, exp_lo, exp_hi));
        if (!positive)
            hi *= random_sign(rng);

        double lo = 0.0;
        if (with_lo)
            lo = random_sign(rng) * random_real(rng, lo_scale_lo, lo_scale_hi) * ulp_of(hi);

        const value_spec spec = renorm_spec(hi, lo);
        if (positive && (spec.hi + spec.lo) <= 0.0)
            return renorm_spec(std::abs(hi) + std::abs(lo) + 1.0, 0.0);

        return spec;
    }

    [[nodiscard]] value_spec make_positive_near_one_spec(std::mt19937_64& rng, bool with_lo)
    {
        const double delta = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 18, 60));
        const double hi = 1.0 + random_sign(rng) * delta;
        double lo = 0.0;
        if (with_lo)
            lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(hi);

        const value_spec spec = renorm_spec(hi, lo);
        if ((spec.hi + spec.lo) <= 0.0)
            return renorm_spec(1.0 + delta, 0.0);
        return spec;
    }

    [[nodiscard]] value_spec make_rounding_hard_spec(std::mt19937_64& rng)
    {
        const double hi = random_sign(rng) * std::ldexp(random_real(rng, 0.5, 1.9999999999999998), random_int(rng, 55, 500));
        const double lo = random_sign(rng) * random_real(rng, 0.1, 0.95) * ulp_of(hi);
        return renorm_spec(hi, lo);
    }

    [[nodiscard]] value_spec make_trig_hard_spec(std::mt19937_64& rng)
    {
        const double pi_2 = std::acos(-1.0) * 0.5;
        const double k = std::ldexp(random_real(rng, 0.75, 1.25), random_int(rng, 18, 42));
        const double eps = random_sign(rng) * std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 12, 40));
        const double hi = random_sign(rng) * (k * pi_2 + eps);
        const double lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(hi);
        return renorm_spec(hi, lo);
    }

    [[nodiscard]] value_spec make_inverse_trig_hard_spec(std::mt19937_64& rng)
    {
        const double margin = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 18, 58));
        const double hi = random_sign(rng) * (1.0 - margin);
        double lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(hi);
        value_spec spec = renorm_spec(hi, lo);
        const double sum = spec.hi + spec.lo;
        if (sum >= 1.0)
            spec = renorm_spec(std::nextafter(1.0, 0.0), -std::abs(lo));
        else if (sum <= -1.0)
            spec = renorm_spec(-std::nextafter(1.0, 0.0), std::abs(lo));
        return spec;
    }

    [[nodiscard]] value_spec make_exp_hard_spec(std::mt19937_64& rng)
    {
        const double ln2 = std::log(2.0);
        const double k = static_cast<double>(random_int(rng, -128, 128));
        const double eps = random_sign(rng) * std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 18, 50));
        const double hi = k * ln2 + eps;
        const double lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(hi == 0.0 ? 1.0 : hi);
        return renorm_spec(hi, lo);
    }

    [[nodiscard]] value_spec make_pow_base_hard_spec(std::mt19937_64& rng)
    {
        if (random_bool(rng))
            return make_positive_near_one_spec(rng, true);

        return make_random_value_spec(rng, true, true, -20, 20);
    }

    [[nodiscard]] value_spec make_pow_exponent_hard_spec(std::mt19937_64& rng)
    {
        const double hi = random_sign(rng) * random_real(rng, 0.125, 12.0);
        const double lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(hi);
        return renorm_spec(hi, lo);
    }

    struct random_value_domain
    {
        double min_value = std::numeric_limits<double>::infinity();
        double max_value = -std::numeric_limits<double>::infinity();
        double min_abs = std::numeric_limits<double>::infinity();
        double max_abs = 0.0;
        bool saw_value = false;
        bool saw_negative = false;
        bool saw_positive = false;
    };

    void observe_random_value_domain(random_value_domain& domain, const value_spec& spec)
    {
        const double value = spec.hi + spec.lo;
        if (!std::isfinite(value))
            return;

        domain.saw_value = true;
        domain.min_value = std::min(domain.min_value, value);
        domain.max_value = std::max(domain.max_value, value);
        domain.saw_negative = domain.saw_negative || value < 0.0;
        domain.saw_positive = domain.saw_positive || value > 0.0;

        const double magnitude = std::abs(value);
        domain.max_abs = std::max(domain.max_abs, magnitude);
        if (magnitude > 0.0)
            domain.min_abs = std::min(domain.min_abs, magnitude);
    }

    [[nodiscard]] random_value_domain normalized_random_value_domain(random_value_domain domain)
    {
        if (!domain.saw_value)
        {
            domain.min_value = -1.0;
            domain.max_value = 1.0;
            domain.min_abs = 1.0;
            domain.max_abs = 1.0;
            domain.saw_value = true;
            domain.saw_negative = true;
            domain.saw_positive = true;
            return domain;
        }

        if (!std::isfinite(domain.min_abs))
            domain.min_abs = 1.0;
        if (domain.max_abs == 0.0)
            domain.max_abs = 1.0;
        return domain;
    }

    [[nodiscard]] std::uint64_t double_bits(double value)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    void hash_u64(std::uint64_t& hash, std::uint64_t value)
    {
        constexpr std::uint64_t prime = 1099511628211ull;
        for (int i = 0; i < 8; ++i)
        {
            hash ^= static_cast<unsigned char>((value >> (i * 8)) & 0xffu);
            hash *= prime;
        }
    }

    void hash_text(std::uint64_t& hash, std::string_view text)
    {
        constexpr std::uint64_t prime = 1099511628211ull;
        for (const char ch : text)
        {
            hash ^= static_cast<unsigned char>(ch);
            hash *= prime;
        }

        hash ^= 0xffu;
        hash *= prime;
    }

    void hash_spec(std::uint64_t& hash, const value_spec& spec)
    {
        hash_u64(hash, double_bits(spec.hi));
        hash_u64(hash, double_bits(spec.lo));
    }

    void hash_spec(std::uint64_t& hash, const binary_value_spec& spec)
    {
        hash_spec(hash, spec.lhs);
        hash_spec(hash, spec.rhs);
    }

    void hash_spec(std::uint64_t& hash, const recurrence_value_spec& spec)
    {
        hash_spec(hash, spec.x);
        hash_spec(hash, spec.y);
        hash_spec(hash, spec.a);
        hash_spec(hash, spec.b);
        hash_spec(hash, spec.c);
        hash_spec(hash, spec.d);
    }

    void hash_spec(std::uint64_t& hash, const affine_transform_value_spec& spec)
    {
        hash_spec(hash, spec.x);
        hash_spec(hash, spec.y);
        hash_spec(hash, spec.angle);
        hash_spec(hash, spec.scale_x);
        hash_spec(hash, spec.scale_y);
        hash_spec(hash, spec.translate_x);
        hash_spec(hash, spec.translate_y);
    }

    void hash_spec(std::uint64_t& hash, const ldexp_value_spec& spec)
    {
        hash_spec(hash, spec.value);
        hash_u64(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(spec.exponent)));
    }

    template<typename Spec>
    [[nodiscard]] std::uint64_t make_typical_seed(const bucket_array_set<Spec>& specs, std::string_view salt)
    {
        std::uint64_t hash = 14695981039346656037ull;
        hash_text(hash, salt);

        for (const auto& spec : specs.easy)
            hash_spec(hash, spec);
        for (const auto& spec : specs.medium)
            hash_spec(hash, spec);
        for (const auto& spec : specs.hard)
            hash_spec(hash, spec);

        return hash;
    }

    [[nodiscard]] value_spec clamp_random_value_to_domain(value_spec spec, const random_value_domain& domain)
    {
        const double sum = spec.hi + spec.lo;

        if (!domain.saw_negative && sum <= 0.0)
            return renorm_spec(std::max(domain.min_abs, std::numeric_limits<double>::denorm_min()), 0.0);

        if (domain.min_value >= 1.0 && sum < 1.0)
            return renorm_spec(1.0 + std::max(domain.min_abs - 1.0, 0.0), 0.0);

        if (domain.min_value > -1.0 && sum <= -1.0)
            return renorm_spec(std::nextafter(-1.0, 0.0), 0.0);

        if (domain.max_value <= 1.0 && sum >= 1.0)
            return renorm_spec(std::nextafter(1.0, 0.0), 0.0);

        return spec;
    }

    [[nodiscard]] value_spec make_unit_random_value_spec(std::mt19937_64& rng, const random_value_domain& domain)
    {
        const double lo = domain.saw_negative ? -std::nextafter(1.0, 0.0) : std::numeric_limits<double>::denorm_min();
        const double hi = domain.saw_positive ? std::nextafter(1.0, 0.0) : -std::numeric_limits<double>::denorm_min();
        const double base = random_real(rng, lo, hi);
        const double tail = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(base == 0.0 ? 1.0 : base);
        return clamp_random_value_to_domain(renorm_spec(base, tail), domain);
    }

    [[nodiscard]] value_spec make_fixed_random_value_spec(std::mt19937_64& rng, random_value_domain domain)
    {
        double lo = domain.min_value;
        double hi = domain.max_value;

        if (lo == hi)
        {
            if (lo == 0.0)
            {
                lo = domain.saw_negative ? -1.0 : 0.0;
                hi = domain.saw_positive ? 1.0 : 0.0;
            }
            else
            {
                lo *= 0.75;
                hi *= 1.25;
                if (lo > hi)
                    std::swap(lo, hi);
            }
        }

        if (!domain.saw_negative)
            lo = std::max(lo, std::max(domain.min_abs, std::numeric_limits<double>::denorm_min()));

        const double base = random_real(rng, lo, hi);
        const double tail = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(base == 0.0 ? 1.0 : base);
        return clamp_random_value_to_domain(renorm_spec(base, tail), domain);
    }

    [[nodiscard]] value_spec make_scientific_random_value_spec(std::mt19937_64& rng, random_value_domain domain)
    {
        const double min_abs = std::max(domain.min_abs, std::numeric_limits<double>::denorm_min());
        const double max_abs = std::max(domain.max_abs, min_abs);
        int exp_lo = static_cast<int>(std::floor(std::log2(min_abs))) - 1;
        int exp_hi = static_cast<int>(std::ceil(std::log2(max_abs))) + 1;

        if (!domain.saw_negative && domain.min_value >= 1.0)
            exp_lo = std::max(exp_lo, 0);
        if (exp_lo > exp_hi)
            std::swap(exp_lo, exp_hi);

        exp_lo = std::clamp(exp_lo, -900, 900);
        exp_hi = std::clamp(exp_hi, -900, 900);
        if (exp_lo > exp_hi)
            std::swap(exp_lo, exp_hi);

        const bool positive = !domain.saw_negative;
        return clamp_random_value_to_domain(make_random_value_spec(rng, positive, true, exp_lo, exp_hi), domain);
    }

    [[nodiscard]] value_spec make_random_typical_value_spec(std::mt19937_64& rng, random_value_domain domain)
    {
        domain = normalized_random_value_domain(domain);

        if (domain.min_value >= -1.0 && domain.max_value <= 1.0)
            return make_unit_random_value_spec(rng, domain);

        if (domain.max_abs <= 1'000'000.0)
            return make_fixed_random_value_spec(rng, domain);

        return make_scientific_random_value_spec(rng, domain);
    }

    [[nodiscard]] random_value_domain make_value_domain(const std::array<value_spec, bucket_value_count>& specs)
    {
        random_value_domain domain{};
        for (const value_spec& spec : specs)
            observe_random_value_domain(domain, spec);
        return normalized_random_value_domain(domain);
    }

    template<typename Spec, typename Getter>
    [[nodiscard]] random_value_domain make_spec_domain(const std::array<Spec, bucket_value_count>& specs, Getter&& getter)
    {
        random_value_domain domain{};
        for (const auto& spec : specs)
            observe_random_value_domain(domain, getter(spec));
        return normalized_random_value_domain(domain);
    }

    [[nodiscard]] std::array<value_spec, bucket_value_count> make_typical_value_specs(const bucket_array_set<value_spec>& specs)
    {
        std::array<value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "value")};
        const random_value_domain domain = make_value_domain(specs.medium);

        for (value_spec& spec : out)
            spec = make_random_typical_value_spec(rng, domain);

        return out;
    }

    [[nodiscard]] std::array<affine_transform_value_spec, bucket_value_count> make_typical_affine_transform_specs(const bucket_array_set<affine_transform_value_spec>& specs)
    {
        std::array<affine_transform_value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "affine-transform")};
        const random_value_domain x_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.x; });
        const random_value_domain y_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.y; });
        const random_value_domain angle_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.angle; });
        const random_value_domain scale_x_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.scale_x; });
        const random_value_domain scale_y_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.scale_y; });
        const random_value_domain translate_x_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.translate_x; });
        const random_value_domain translate_y_domain = make_spec_domain(specs.medium, [](const affine_transform_value_spec& spec) { return spec.translate_y; });

        for (affine_transform_value_spec& spec : out)
        {
            spec.x = make_random_typical_value_spec(rng, x_domain);
            spec.y = make_random_typical_value_spec(rng, y_domain);
            spec.angle = make_random_typical_value_spec(rng, angle_domain);
            spec.scale_x = make_random_typical_value_spec(rng, scale_x_domain);
            spec.scale_y = make_random_typical_value_spec(rng, scale_y_domain);
            spec.translate_x = make_random_typical_value_spec(rng, translate_x_domain);
            spec.translate_y = make_random_typical_value_spec(rng, translate_y_domain);
        }

        return out;
    }

    [[nodiscard]] double sample_bounded_normal_weight(std::mt19937_64& rng)
    {
        std::normal_distribution<double> normal{ 0.0, 0.40 };
        return std::clamp(std::abs(normal(rng)), 0.0, 1.0);
    }

    template<typename SignedInt>
    [[nodiscard]] value_spec make_integer_rounding_typical_spec(std::mt19937_64& rng)
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);

        constexpr int max_exp = std::min<int>(62, std::numeric_limits<SignedInt>::digits - 1);
        constexpr int fast_exp_hi = std::min(51, max_exp);
        constexpr bool has_fallback_range = max_exp >= 52;

        const bool use_fallback_range = has_fallback_range && random_real(rng, 0.0, 1.0) >= 0.75;
        const int exp_lo = use_fallback_range ? 52 : 0;
        const int exp_hi = use_fallback_range ? max_exp : fast_exp_hi;
        const double weight = sample_bounded_normal_weight(rng);
        const int exp = exp_lo + std::min(exp_hi - exp_lo, static_cast<int>(std::floor(weight * static_cast<double>(exp_hi - exp_lo + 1))));

        constexpr std::array<double, 6> fractions{{ 0.25, 0.499999999999, 0.5, 0.500000000001, 0.75, 0.9375 }};
        const double fraction = fractions[static_cast<std::size_t>(random_int(rng, 0, static_cast<int>(fractions.size() - 1)))];
        const double whole = std::floor(std::ldexp(random_real(rng, 0.5, 1.75), exp));
        const double sign = random_sign(rng);

        if (use_fallback_range)
            return renorm_spec(sign * whole, sign * fraction);

        const double value = sign * (whole + fraction);
        const double tail = sign * random_real(rng, 0.05, 0.95) * ulp_of(value == 0.0 ? 1.0 : value);
        return renorm_spec(value, tail);
    }

    template<typename SignedInt>
    [[nodiscard]] std::array<value_spec, bucket_value_count> make_integer_rounding_typical_specs(const bucket_array_set<value_spec>& specs)
    {
        std::array<value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "integer-rounding")};

        for (value_spec& spec : out)
            spec = make_integer_rounding_typical_spec<SignedInt>(rng);

        return out;
    }

    [[nodiscard]] std::array<binary_value_spec, bucket_value_count> make_typical_binary_specs(const bucket_array_set<binary_value_spec>& specs)
    {
        std::array<binary_value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "binary")};
        const random_value_domain lhs_domain = make_spec_domain(specs.medium, [](const binary_value_spec& spec) { return spec.lhs; });
        const random_value_domain rhs_domain = make_spec_domain(specs.medium, [](const binary_value_spec& spec) { return spec.rhs; });

        for (binary_value_spec& spec : out)
        {
            spec.lhs = make_random_typical_value_spec(rng, lhs_domain);
            spec.rhs = make_random_typical_value_spec(rng, rhs_domain);
        }

        return out;
    }

    [[nodiscard]] value_spec make_typical_atan_value_spec(double value, std::mt19937_64& rng)
    {
        const double lo = random_sign(rng) * random_real(rng, 0.05, 0.95) * ulp_of(value == 0.0 ? 1.0 : value);
        return renorm_spec(value, lo);
    }

    [[nodiscard]] std::array<value_spec, atan_typical_value_count> make_typical_atan_specs()
    {
        std::array<value_spec, atan_typical_value_count> out{};
        std::mt19937_64 rng{0xf27a6c1e0d9112b5ull};
        std::uniform_real_distribution<double> near_unit{-1.0, 1.0};
        std::uniform_real_distribution<double> wide{-16.0, 16.0};
        std::bernoulli_distribution use_near_unit{0.70};

        for (value_spec& spec : out)
        {
            const double value = use_near_unit(rng) ? near_unit(rng) : wide(rng);
            spec = make_typical_atan_value_spec(value, rng);
        }

        return out;
    }

    [[nodiscard]] std::array<binary_value_spec, atan_typical_value_count> make_typical_atan2_specs()
    {
        std::array<binary_value_spec, atan_typical_value_count> out{};
        std::mt19937_64 rng{0x4bd2f85a99d7260full};
        std::uniform_real_distribution<double> angle_distribution{-benchmark_pi, benchmark_pi};
        std::uniform_real_distribution<double> log_radius_distribution{-12.0, 12.0};

        for (binary_value_spec& spec : out)
        {
            const double angle = angle_distribution(rng);
            const double radius = std::pow(10.0, log_radius_distribution(rng));
            const double x = radius * std::cos(angle);
            const double y = radius * std::sin(angle);

            spec.lhs = make_typical_atan_value_spec(y, rng);
            spec.rhs = make_typical_atan_value_spec(x, rng);
        }

        return out;
    }

    [[nodiscard]] std::array<ldexp_value_spec, bucket_value_count> make_typical_ldexp_specs(const bucket_array_set<ldexp_value_spec>& specs)
    {
        std::array<ldexp_value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "ldexp")};
        const random_value_domain value_domain = make_spec_domain(specs.medium, [](const ldexp_value_spec& spec) { return spec.value; });

        int min_exponent = specs.medium.front().exponent;
        int max_exponent = specs.medium.front().exponent;
        for (const ldexp_value_spec& spec : specs.medium)
        {
            min_exponent = std::min(min_exponent, spec.exponent);
            max_exponent = std::max(max_exponent, spec.exponent);
        }

        for (ldexp_value_spec& spec : out)
        {
            spec.value = make_random_typical_value_spec(rng, value_domain);
            spec.exponent = random_int(rng, min_exponent, max_exponent);
        }

        return out;
    }

    [[nodiscard]] std::array<recurrence_value_spec, bucket_value_count> make_typical_recurrence_specs(const bucket_array_set<recurrence_value_spec>& specs)
    {
        std::array<recurrence_value_spec, bucket_value_count> out{};
        std::mt19937_64 rng{make_typical_seed(specs, "recurrence")};
        const random_value_domain x_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.x; });
        const random_value_domain y_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.y; });
        const random_value_domain a_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.a; });
        const random_value_domain b_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.b; });
        const random_value_domain c_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.c; });
        const random_value_domain d_domain = make_spec_domain(specs.medium, [](const recurrence_value_spec& spec) { return spec.d; });

        for (recurrence_value_spec& spec : out)
        {
            spec.x = make_random_typical_value_spec(rng, x_domain);
            spec.y = make_random_typical_value_spec(rng, y_domain);
            spec.a = make_random_typical_value_spec(rng, a_domain);
            spec.b = make_random_typical_value_spec(rng, b_domain);
            spec.c = make_random_typical_value_spec(rng, c_domain);
            spec.d = make_random_typical_value_spec(rng, d_domain);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_generic_unary_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1001ull };
        std::mt19937_64 medium_rng{ 0x1002ull };
        std::mt19937_64 hard_rng{ 0x1003ull };

        for (auto& value : out.easy)
            value = make_random_value_spec(easy_rng, false, false, -40, 40);
        for (auto& value : out.medium)
            value = make_random_value_spec(medium_rng, false, true, -60, 60);
        for (auto& value : out.hard)
            value = make_random_value_spec(hard_rng, false, true, -240, 240);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_rounding_unary_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1101ull };
        std::mt19937_64 medium_rng{ 0x1102ull };
        std::mt19937_64 hard_rng{ 0x1103ull };

        for (auto& value : out.easy)
            value = make_random_value_spec(easy_rng, false, false, -20, 20);
        for (auto& value : out.medium)
            value = make_random_value_spec(medium_rng, false, true, -20, 20);
        for (auto& value : out.hard)
            value = make_rounding_hard_spec(hard_rng);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_positive_sqrt_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1201ull };
        std::mt19937_64 medium_rng{ 0x1202ull };
        std::mt19937_64 hard_rng{ 0x1203ull };

        for (auto& value : out.easy)
            value = make_random_value_spec(easy_rng, true, false, -30, 30);
        for (auto& value : out.medium)
            value = make_random_value_spec(medium_rng, true, true, -80, 80);
        for (auto& value : out.hard)
            value = make_random_value_spec(hard_rng, true, true, -450, 450);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_positive_log_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1301ull };
        std::mt19937_64 medium_rng{ 0x1302ull };
        std::mt19937_64 hard_rng{ 0x1303ull };

        for (auto& value : out.easy)
            value = make_random_value_spec(easy_rng, true, false, -20, 20);
        for (auto& value : out.medium)
            value = make_random_value_spec(medium_rng, true, true, -80, 80);
        for (auto& value : out.hard)
        {
            value = random_bool(hard_rng)
                ? make_positive_near_one_spec(hard_rng, true)
                : make_random_value_spec(hard_rng, true, true, -300, 300);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_exponent_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1401ull };
        std::mt19937_64 medium_rng{ 0x1402ull };
        std::mt19937_64 hard_rng{ 0x1403ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_sign(easy_rng) * random_real(easy_rng, 0.0, 4.0), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_sign(medium_rng) * random_real(medium_rng, 0.0, 16.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1.0));
        for (auto& value : out.hard)
            value = make_exp_hard_spec(hard_rng);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_log1p_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1411ull };
        std::mt19937_64 medium_rng{ 0x1412ull };
        std::mt19937_64 hard_rng{ 0x1413ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, -0.5, 0.5), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_real(medium_rng, -0.95, 4.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1.0));
        for (auto& value : out.hard)
        {
            if (random_bool(hard_rng))
            {
                const double margin = std::ldexp(random_real(hard_rng, 0.25, 0.95), -random_int(hard_rng, 18, 60));
                const double hi = -1.0 + margin;
                const double lo = random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(1.0);
                value = renorm_spec(hi, lo);
                if ((value.hi + value.lo) <= -1.0)
                    value = renorm_spec(std::nextafter(-1.0, 0.0), 0.0);
            }
            else
            {
                value = make_positive_near_one_spec(hard_rng, true);
            }
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_hyperbolic_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1421ull };
        std::mt19937_64 medium_rng{ 0x1422ull };
        std::mt19937_64 hard_rng{ 0x1423ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_sign(easy_rng) * random_real(easy_rng, 0.0, 2.0), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_sign(medium_rng) * random_real(medium_rng, 0.0, 10.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(10.0));
        for (auto& value : out.hard)
        {
            if (random_bool(hard_rng))
            {
                const double hi = random_sign(hard_rng) * std::ldexp(random_real(hard_rng, 0.25, 0.95), -random_int(hard_rng, 18, 60));
                const double lo = random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(1.0);
                value = renorm_spec(hi, lo);
            }
            else
            {
                value = renorm_spec(random_sign(hard_rng) * random_real(hard_rng, 10.0, 20.0), random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(20.0));
            }
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_acosh_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1431ull };
        std::mt19937_64 medium_rng{ 0x1432ull };
        std::mt19937_64 hard_rng{ 0x1433ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, 1.0, 4.0), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_real(medium_rng, 1.0, 64.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(64.0));
        for (auto& value : out.hard)
        {
            if (random_bool(hard_rng))
            {
                const double margin = std::ldexp(random_real(hard_rng, 0.25, 0.95), -random_int(hard_rng, 18, 60));
                value = renorm_spec(1.0 + margin, random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(1.0));
                if ((value.hi + value.lo) < 1.0)
                    value = renorm_spec(1.0, 0.0);
            }
            else
            {
                value = make_random_value_spec(hard_rng, true, true, -40, 300);
                if ((value.hi + value.lo) < 1.0)
                    value = renorm_spec(1.0 + random_real(hard_rng, 0.25, 4.0), 0.0);
            }
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_atanh_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1441ull };
        std::mt19937_64 medium_rng{ 0x1442ull };
        std::mt19937_64 hard_rng{ 0x1443ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, -0.75, 0.75), 0.0);
        for (auto& value : out.medium)
        {
            value = renorm_spec(random_real(medium_rng, -0.999, 0.999), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1.0));
            const double sum = value.hi + value.lo;
            if (sum >= 1.0)
                value = renorm_spec(std::nextafter(1.0, 0.0), 0.0);
            else if (sum <= -1.0)
                value = renorm_spec(-std::nextafter(1.0, 0.0), 0.0);
        }
        for (auto& value : out.hard)
            value = make_inverse_trig_hard_spec(hard_rng);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_erf_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1461ull };
        std::mt19937_64 medium_rng{ 0x1462ull };
        std::mt19937_64 hard_rng{ 0x1463ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, -1.5, 1.5), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_real(medium_rng, -4.0, 4.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(4.0));
        for (auto& value : out.hard)
            value = renorm_spec(random_sign(hard_rng) * random_real(hard_rng, 4.0, 10.0), random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(10.0));

        return out;
    }

    [[nodiscard]] value_spec make_gamma_hard_spec(std::mt19937_64& rng)
    {
        if (random_bool(rng))
        {
            const double anchor = static_cast<double>(random_int(rng, 1, 32));
            const double delta = std::ldexp(random_real(rng, 0.25, 0.95), -random_int(rng, 18, 58));
            return renorm_spec(anchor + random_sign(rng) * delta, 0.0);
        }

        return make_random_value_spec(rng, true, true, -3, 8);
    }

    [[nodiscard]] bucket_array_set<value_spec> make_gamma_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1471ull };
        std::mt19937_64 medium_rng{ 0x1472ull };
        std::mt19937_64 hard_rng{ 0x1473ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, 0.5, 8.0), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_real(medium_rng, 0.125, 24.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(24.0));
        for (auto& value : out.hard)
        {
            value = make_gamma_hard_spec(hard_rng);
            if ((value.hi + value.lo) <= 0.0)
                value = renorm_spec(0.125, 0.0);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> make_hypot_specs()
    {
        bucket_array_set<binary_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1481ull };
        std::mt19937_64 medium_rng{ 0x1482ull };
        std::mt19937_64 hard_rng{ 0x1483ull };

        for (auto& value : out.easy)
        {
            value.lhs = make_random_value_spec(easy_rng, false, false, -20, 20);
            value.rhs = make_random_value_spec(easy_rng, false, false, -20, 20);
        }

        for (auto& value : out.medium)
        {
            value.lhs = make_random_value_spec(medium_rng, false, true, -80, 80);
            value.rhs = make_random_value_spec(medium_rng, false, true, -80, 80);
        }

        for (auto& value : out.hard)
        {
            if (random_bool(hard_rng))
            {
                value.lhs = make_random_value_spec(hard_rng, false, true, 100, 400);
                value.rhs = make_random_value_spec(hard_rng, false, true, -120, -20);
            }
            else
            {
                value.lhs = make_random_value_spec(hard_rng, false, true, -300, 300);
                value.rhs = make_random_value_spec(hard_rng, false, true, -300, 300);
            }
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<ldexp_value_spec> make_ldexp_specs()
    {
        bucket_array_set<ldexp_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1451ull };
        std::mt19937_64 medium_rng{ 0x1452ull };
        std::mt19937_64 hard_rng{ 0x1453ull };

        for (auto& value : out.easy)
        {
            value.value = make_random_value_spec(easy_rng, false, true, -20, 20);
            value.exponent = random_int(easy_rng, -16, 16);
        }

        for (auto& value : out.medium)
        {
            value.value = make_random_value_spec(medium_rng, false, true, -80, 80);
            value.exponent = random_int(medium_rng, -96, 96);
        }

        for (auto& value : out.hard)
        {
            value.value = make_random_value_spec(hard_rng, false, true, -220, 220);
            value.exponent = random_int(hard_rng, -400, 400);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_trig_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1501ull };
        std::mt19937_64 medium_rng{ 0x1502ull };
        std::mt19937_64 hard_rng{ 0x1503ull };

        const double pi = std::acos(-1.0);
        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, -pi, pi), 0.0);
        for (auto& value : out.medium)
            value = renorm_spec(random_real(medium_rng, -1024.0, 1024.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1024.0));
        for (auto& value : out.hard)
            value = make_trig_hard_spec(hard_rng);

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> make_inverse_trig_specs()
    {
        bucket_array_set<value_spec> out{};

        std::mt19937_64 easy_rng{ 0x1601ull };
        std::mt19937_64 medium_rng{ 0x1602ull };
        std::mt19937_64 hard_rng{ 0x1603ull };

        for (auto& value : out.easy)
            value = renorm_spec(random_real(easy_rng, -0.9, 0.9), 0.0);
        for (auto& value : out.medium)
        {
            value = renorm_spec(random_real(medium_rng, -0.999, 0.999), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1.0));
            const double sum = value.hi + value.lo;
            if (sum >= 1.0)
                value = renorm_spec(std::nextafter(1.0, 0.0), 0.0);
            else if (sum <= -1.0)
                value = renorm_spec(-std::nextafter(1.0, 0.0), 0.0);
        }
        for (auto& value : out.hard)
            value = make_inverse_trig_hard_spec(hard_rng);

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> make_generic_binary_specs()
    {
        bucket_array_set<binary_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x2001ull };
        std::mt19937_64 medium_rng{ 0x2002ull };
        std::mt19937_64 hard_rng{ 0x2003ull };

        for (auto& value : out.easy)
        {
            value.lhs = make_random_value_spec(easy_rng, false, false, -40, 40);
            value.rhs = make_random_value_spec(easy_rng, false, false, -40, 40);
        }

        for (auto& value : out.medium)
        {
            value.lhs = make_random_value_spec(medium_rng, false, false, -60, 60);
            value.rhs = make_random_value_spec(medium_rng, false, true, -60, 60);
        }

        for (auto& value : out.hard)
        {
            value.lhs = make_random_value_spec(hard_rng, false, true, -240, 240);
            value.rhs = make_random_value_spec(hard_rng, false, true, -240, 240);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> make_pow_specs()
    {
        bucket_array_set<binary_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x2101ull };
        std::mt19937_64 medium_rng{ 0x2102ull };
        std::mt19937_64 hard_rng{ 0x2103ull };

        for (auto& value : out.easy)
        {
            value.lhs = renorm_spec(random_real(easy_rng, 0.125, 16.0), 0.0);
            value.rhs = renorm_spec(random_sign(easy_rng) * random_real(easy_rng, 0.0, 4.0), 0.0);
        }

        for (auto& value : out.medium)
        {
            value.lhs = renorm_spec(random_real(medium_rng, 0.125, 32.0), 0.0);
            value.rhs = renorm_spec(random_sign(medium_rng) * random_real(medium_rng, 0.0, 6.0), random_sign(medium_rng) * random_real(medium_rng, 0.05, 0.95) * ulp_of(1.0));
        }

        for (auto& value : out.hard)
        {
            value.lhs = make_pow_base_hard_spec(hard_rng);
            value.rhs = make_pow_exponent_hard_spec(hard_rng);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> make_atan2_specs()
    {
        bucket_array_set<binary_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x2201ull };
        std::mt19937_64 medium_rng{ 0x2202ull };
        std::mt19937_64 hard_rng{ 0x2203ull };

        for (auto& value : out.easy)
        {
            value.lhs = make_random_value_spec(easy_rng, false, false, -6, 6);
            value.rhs = make_random_value_spec(easy_rng, false, false, -6, 6);
            if ((value.rhs.hi + value.rhs.lo) == 0.0)
                value.rhs = renorm_spec(1.0, 0.0);
        }

        for (auto& value : out.medium)
        {
            value.lhs = make_random_value_spec(medium_rng, false, false, -20, 20);
            value.rhs = make_random_value_spec(medium_rng, false, true, -20, 20);
            if ((value.rhs.hi + value.rhs.lo) == 0.0)
                value.rhs = renorm_spec(0.75, ulp_of(0.75) * 0.5);
        }

        for (auto& value : out.hard)
        {
            value.lhs = make_random_value_spec(hard_rng, false, true, -40, 40);
            const double hi = random_sign(hard_rng) * std::ldexp(random_real(hard_rng, 0.5, 1.9999999999999998), random_int(hard_rng, -80, -20));
            const double lo = random_sign(hard_rng) * random_real(hard_rng, 0.05, 0.95) * ulp_of(hi);
            value.rhs = renorm_spec(hi, lo);
            if ((value.rhs.hi + value.rhs.lo) == 0.0)
                value.rhs = renorm_spec(std::numeric_limits<double>::denorm_min(), 0.0);
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> make_fmod_specs()
    {
        bucket_array_set<binary_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x2301ull };
        std::mt19937_64 medium_rng{ 0x2302ull };
        std::mt19937_64 hard_rng{ 0x2303ull };

        for (auto& value : out.easy)
        {
            value.lhs = make_random_value_spec(easy_rng, false, false, -40, 60);
            const double y_hi = std::ldexp(random_real(easy_rng, 0.5, 1.9999999999999998), random_int(easy_rng, -12, 12));
            value.rhs = renorm_spec(y_hi, 0.0);
        }

        for (auto& value : out.medium)
        {
            const value_spec y_spec = make_random_value_spec(medium_rng, true, true, -20, 20);
            const double q = std::ldexp(random_real(medium_rng, 0.5, 1.9999999999999998), random_int(medium_rng, 0, 18));
            const double frac = random_real(medium_rng, 0.0, 0.95);
            double x_hi = (y_spec.hi + y_spec.lo) * (q + frac);
            if (random_bool(medium_rng))
                x_hi = -x_hi;
            value.lhs = renorm_spec(x_hi, 0.0);
            value.rhs = y_spec;
        }

        for (auto& value : out.hard)
        {
            const value_spec y_spec = make_random_value_spec(hard_rng, true, true, -100, 100);
            const f128 y = f128{ y_spec.hi, y_spec.lo };
            const double q = std::ldexp(random_real(hard_rng, 0.5, 1.9999999999999998), random_int(hard_rng, 0, 20));
            const f128 eps = y * std::ldexp(random_sign(hard_rng) * random_real(hard_rng, 0.25, 0.95), -random_int(hard_rng, 40, 90));
            f128 x = y * q + eps;
            if (random_bool(hard_rng))
                x = -x;
            value.lhs = spec_from_f128(x);
            value.rhs = y_spec;
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<recurrence_value_spec> make_recurrence_specs()
    {
        bucket_array_set<recurrence_value_spec> out{};

        std::mt19937_64 easy_rng{ 0x2401ull };
        std::mt19937_64 medium_rng{ 0x2402ull };
        std::mt19937_64 hard_rng{ 0x2403ull };

        for (auto& value : out.easy)
        {
            value.x = make_random_value_spec(easy_rng, false, false, -4, 4);
            value.y = make_random_value_spec(easy_rng, false, false, -4, 4);
            value.a = make_random_value_spec(easy_rng, false, false, -4, 4);
            value.b = make_random_value_spec(easy_rng, false, false, -4, 4);
            value.c = make_random_value_spec(easy_rng, true, false, -2, 2);
            value.d = make_random_value_spec(easy_rng, true, false, -2, 2);
        }

        for (auto& value : out.medium)
        {
            value.x = make_random_value_spec(medium_rng, false, false, -8, 8);
            value.y = make_random_value_spec(medium_rng, false, false, -8, 8);
            value.a = make_random_value_spec(medium_rng, false, true, -8, 8);
            value.b = make_random_value_spec(medium_rng, false, true, -8, 8);
            value.c = make_random_value_spec(medium_rng, true, true, -4, 4);
            value.d = make_random_value_spec(medium_rng, true, true, -4, 4);
        }

        for (auto& value : out.hard)
        {
            value.x = make_random_value_spec(hard_rng, false, true, -16, 16);
            value.y = make_random_value_spec(hard_rng, false, true, -16, 16);
            value.a = make_random_value_spec(hard_rng, false, true, -16, 16);
            value.b = make_random_value_spec(hard_rng, false, true, -16, 16);
            value.c = make_random_value_spec(hard_rng, true, true, -8, 8);
            value.d = make_random_value_spec(hard_rng, true, true, -8, 8);
        }

        return out;
    }

    using scalar_recurrence_text_case = std::array<const char*, 6>;

    [[nodiscard]] recurrence_value_spec make_recurrence_spec_from_text(const scalar_recurrence_text_case& value)
    {
        return recurrence_value_spec{
            spec_from_text(value[0]),
            spec_from_text(value[1]),
            spec_from_text(value[2]),
            spec_from_text(value[3]),
            spec_from_text(value[4]),
            spec_from_text(value[5])
        };
    }

    template<std::size_t SourceCount>
    void fill_scalar_recurrence_specs(
        std::array<recurrence_value_spec, bucket_value_count>& out,
        const std::array<scalar_recurrence_text_case, SourceCount>& source)
    {
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = make_recurrence_spec_from_text(source[i % SourceCount]);
    }

    [[nodiscard]] bucket_array_set<recurrence_value_spec> make_scalar_mixed_recurrence_specs()
    {
        bucket_array_set<recurrence_value_spec> out{};

        static constexpr std::array<scalar_recurrence_text_case, 8> easy{{
            {{ "-0.75", "0.125", "0.875", "-0.375", "1.25", "0.625" }},
            {{ "0.5", "-0.25", "-0.625", "0.75", "0.875", "1.125" }},
            {{ "-0.25", "0.5", "0.375", "0.875", "1.5", "0.75" }},
            {{ "0.125", "-0.625", "-0.875", "0.25", "0.625", "1.25" }},
            {{ "0.875", "0.375", "0.5", "-0.75", "1.125", "0.875" }},
            {{ "-0.5", "-0.125", "-0.25", "0.625", "0.75", "1.5" }},
            {{ "0.625", "-0.875", "0.75", "0.125", "1.375", "0.5" }},
            {{ "-0.125", "0.75", "-0.5", "-0.875", "0.5", "1.375" }}
        }};

        static constexpr std::array<scalar_recurrence_text_case, 8> medium{{
            {{ "-1.1250000000000000000000000000000001", "0.3125000000000000000000000000000001", "0.8750000000000000000000000000000001", "-0.4062500000000000000000000000000001", "1.3125000000000000000000000000000001", "0.6875000000000000000000000000000001" }},
            {{ "0.6875000000000000000000000000000001", "-0.9375000000000000000000000000000001", "-0.5625000000000000000000000000000001", "0.7812500000000000000000000000000001", "0.8125000000000000000000000000000001", "1.1875000000000000000000000000000001" }},
            {{ "-0.3437500000000000000000000000000001", "0.8437500000000000000000000000000001", "0.4687500000000000000000000000000001", "0.6562500000000000000000000000000001", "1.5625000000000000000000000000000001", "0.9062500000000000000000000000000001" }},
            {{ "0.1562500000000000000000000000000001", "-0.7187500000000000000000000000000001", "-0.8437500000000000000000000000000001", "0.3437500000000000000000000000000001", "0.7187500000000000000000000000000001", "1.4062500000000000000000000000000001" }},
            {{ "1.0312500000000000000000000000000001", "0.4062500000000000000000000000000001", "0.5937500000000000000000000000000001", "-0.6875000000000000000000000000000001", "1.2187500000000000000000000000000001", "0.7812500000000000000000000000000001" }},
            {{ "-0.6562500000000000000000000000000001", "-0.1875000000000000000000000000000001", "-0.3125000000000000000000000000000001", "0.7187500000000000000000000000000001", "0.9062500000000000000000000000000001", "1.6250000000000000000000000000000001" }},
            {{ "0.7812500000000000000000000000000001", "-1.0312500000000000000000000000000001", "0.6875000000000000000000000000000001", "0.2187500000000000000000000000000001", "1.4375000000000000000000000000000001", "0.5937500000000000000000000000000001" }},
            {{ "-0.2187500000000000000000000000000001", "0.9375000000000000000000000000000001", "-0.4687500000000000000000000000000001", "-0.8125000000000000000000000000000001", "0.5937500000000000000000000000000001", "1.3125000000000000000000000000000001" }}
        }};

        static constexpr std::array<scalar_recurrence_text_case, 8> hard{{
            {{ "-1.3333333333333333333333333333333333", "0.4142135623730950488016887242096981", "0.7071067811865475244008443621048490", "-0.5773502691896257645091487805019575", "1.6180339887498948482045868343656381", "0.6180339887498948482045868343656381" }},
            {{ "0.5773502691896257645091487805019575", "-1.2247448713915890490986420373529457", "-0.4142135623730950488016887242096981", "0.8660254037844386467637231707529362", "0.7071067811865475244008443621048490", "1.4142135623730950488016887242096981" }},
            {{ "-0.3819660112501051517954131656343619", "1.1180339887498948482045868343656381", "0.6180339887498948482045868343656381", "0.7320508075688772935274463415058724", "1.7320508075688772935274463415058724", "0.8164965809277260327324280249019638" }},
            {{ "0.2679491924311227064725536584941276", "-0.8164965809277260327324280249019638", "-0.8660254037844386467637231707529362", "0.3819660112501051517954131656343619", "0.7639320225002103035908263312687238", "1.6180339887498948482045868343656381" }},
            {{ "1.1547005383792515290182975610039150", "0.3660254037844386467637231707529362", "0.8164965809277260327324280249019638", "-0.7071067811865475244008443621048490", "1.3333333333333333333333333333333333", "0.6666666666666666666666666666666667" }},
            {{ "-0.7639320225002103035908263312687238", "-0.2360679774997896964091736687312762", "-0.3333333333333333333333333333333333", "0.7320508075688772935274463415058724", "0.8660254037844386467637231707529362", "1.7320508075688772935274463415058724" }},
            {{ "0.8660254037844386467637231707529362", "-1.1180339887498948482045868343656381", "0.7639320225002103035908263312687238", "0.2679491924311227064725536584941276", "1.5", "0.5773502691896257645091487805019575" }},
            {{ "-0.2679491924311227064725536584941276", "1.2247448713915890490986420373529457", "-0.6180339887498948482045868343656381", "-0.7639320225002103035908263312687238", "0.6666666666666666666666666666666667", "1.5" }}
        }};

        fill_scalar_recurrence_specs(out.easy, easy);
        fill_scalar_recurrence_specs(out.medium, medium);
        fill_scalar_recurrence_specs(out.hard, hard);
        return out;
    }

    using affine_transform_text_case = std::array<const char*, 7>;

    [[nodiscard]] affine_transform_value_spec make_affine_transform_spec_from_text(const affine_transform_text_case& value)
    {
        return affine_transform_value_spec{
            spec_from_text(value[0]),
            spec_from_text(value[1]),
            spec_from_text(value[2]),
            spec_from_text(value[3]),
            spec_from_text(value[4]),
            spec_from_text(value[5]),
            spec_from_text(value[6])
        };
    }

    template<std::size_t SourceCount>
    void fill_affine_transform_specs(
        std::array<affine_transform_value_spec, bucket_value_count>& out,
        const std::array<affine_transform_text_case, SourceCount>& source)
    {
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = make_affine_transform_spec_from_text(source[i % SourceCount]);
    }

    [[nodiscard]] bucket_array_set<affine_transform_value_spec> make_affine_transform_specs()
    {
        bucket_array_set<affine_transform_value_spec> out{};

        static constexpr std::array<affine_transform_text_case, 8> easy{{
            {{ "-1.0", "0.5", "-0.78539816339744830962", "0.875", "1.125", "0.25", "-0.125" }},
            {{ "0.75", "-0.25", "0.52359877559829887308", "1.0", "0.75", "-0.375", "0.5" }},
            {{ "-0.5", "-0.75", "1.04719755119659774615", "0.625", "1.25", "0.125", "0.375" }},
            {{ "0.25", "1.0", "-1.57079632679489661923", "1.375", "0.875", "-0.5", "-0.25" }},
            {{ "1.125", "-0.625", "0.39269908169872415481", "0.75", "1.0", "0.3125", "-0.4375" }},
            {{ "-0.875", "0.875", "-0.26179938779914943654", "1.25", "0.625", "-0.1875", "0.25" }},
            {{ "0.5", "0.125", "1.30899693899574718269", "0.9375", "1.1875", "0.0625", "-0.3125" }},
            {{ "-0.125", "-1.125", "-1.17809724509617246442", "1.0625", "0.8125", "-0.25", "0.1875" }}
        }};

        static constexpr std::array<affine_transform_text_case, 8> medium{{
            {{ "-1.1250000000000000000000000000000001", "0.3125000000000000000000000000000001", "-3.1415926535897932384626433832795029", "0.8750000000000000000000000000000001", "1.1250000000000000000000000000000001", "0.2500000000000000000000000000000001", "-0.1250000000000000000000000000000001" }},
            {{ "0.6875000000000000000000000000000001", "-0.9375000000000000000000000000000001", "-2.3561944901923449288469825374596272", "1.0312500000000000000000000000000001", "0.7812500000000000000000000000000001", "-0.3750000000000000000000000000000001", "0.5000000000000000000000000000000001" }},
            {{ "-0.3437500000000000000000000000000001", "0.8437500000000000000000000000000001", "-1.0471975511965977461542144610931676", "0.7187500000000000000000000000000001", "1.3125000000000000000000000000000001", "0.1250000000000000000000000000000001", "0.3750000000000000000000000000000001" }},
            {{ "0.1562500000000000000000000000000001", "-0.7187500000000000000000000000000001", "-0.3926990816987241548078304229099379", "1.4062500000000000000000000000000001", "0.9062500000000000000000000000000001", "-0.5000000000000000000000000000000001", "-0.2500000000000000000000000000000001" }},
            {{ "1.0312500000000000000000000000000001", "0.4062500000000000000000000000000001", "0.4487989505128276054946633404685004", "0.8125000000000000000000000000000001", "1.2187500000000000000000000000000001", "0.3125000000000000000000000000000001", "-0.4375000000000000000000000000000001" }},
            {{ "-0.6562500000000000000000000000000001", "-0.1875000000000000000000000000000001", "1.5707963267948966192313216916397514", "1.1875000000000000000000000000000001", "0.6875000000000000000000000000000001", "-0.1875000000000000000000000000000001", "0.2500000000000000000000000000000001" }},
            {{ "0.7812500000000000000000000000000001", "-1.0312500000000000000000000000000001", "2.0943951023931954923084289221863353", "0.9687500000000000000000000000000001", "1.3437500000000000000000000000000001", "0.0625000000000000000000000000000001", "-0.3125000000000000000000000000000001" }},
            {{ "-0.2187500000000000000000000000000001", "0.9375000000000000000000000000000001", "3.1415926535897932384626433832795029", "1.0937500000000000000000000000000001", "0.8437500000000000000000000000000001", "-0.2500000000000000000000000000000001", "0.1875000000000000000000000000000001" }}
        }};

        static constexpr std::array<affine_transform_text_case, 8> hard{{
            {{ "-4.1250000000000000000000000000000001", "2.3125000000000000000000000000000001", "-12.5663706143591729538505735331180115", "0.7812500000000000000000000000000001", "1.4375000000000000000000000000000001", "1.2500000000000000000000000000000001", "-1.1250000000000000000000000000000001" }},
            {{ "3.6875000000000000000000000000000001", "-3.9375000000000000000000000000000001", "-9.4247779607693797153879301498385087", "1.5312500000000000000000000000000001", "0.6562500000000000000000000000000001", "-1.3750000000000000000000000000000001", "1.5000000000000000000000000000000001" }},
            {{ "-2.3437500000000000000000000000000001", "3.8437500000000000000000000000000001", "-6.2831853071795864769252867665590058", "0.5937500000000000000000000000000001", "1.6250000000000000000000000000000001", "1.1250000000000000000000000000000001", "1.3750000000000000000000000000000001" }},
            {{ "2.1562500000000000000000000000000001", "-2.7187500000000000000000000000000001", "-4.7123889803846898576939650749192543", "1.6875000000000000000000000000000001", "0.5625000000000000000000000000000001", "-1.5000000000000000000000000000000001", "-1.2500000000000000000000000000000001" }},
            {{ "4.0312500000000000000000000000000001", "1.4062500000000000000000000000000001", "4.7123889803846898576939650749192543", "0.6875000000000000000000000000000001", "1.5625000000000000000000000000000001", "1.3125000000000000000000000000000001", "-1.4375000000000000000000000000000001" }},
            {{ "-3.6562500000000000000000000000000001", "-1.1875000000000000000000000000000001", "6.2831853071795864769252867665590058", "1.4375000000000000000000000000000001", "0.5937500000000000000000000000000001", "-1.1875000000000000000000000000000001", "1.2500000000000000000000000000000001" }},
            {{ "2.7812500000000000000000000000000001", "-4.0312500000000000000000000000000001", "9.4247779607693797153879301498385087", "0.8437500000000000000000000000000001", "1.6875000000000000000000000000000001", "1.0625000000000000000000000000000001", "-1.3125000000000000000000000000000001" }},
            {{ "-1.2187500000000000000000000000000001", "3.9375000000000000000000000000000001", "12.5663706143591729538505735331180115", "1.2187500000000000000000000000000001", "0.7187500000000000000000000000000001", "-1.2500000000000000000000000000000001", "1.1875000000000000000000000000000001" }}
        }};

        fill_affine_transform_specs(out.easy, easy);
        fill_affine_transform_specs(out.medium, medium);
        fill_affine_transform_specs(out.hard, hard);
        return out;
    }

    template<typename T>
    [[nodiscard]] T make_value(const value_spec& spec);

    template<>
    [[nodiscard]] f128 make_value<f128>(const value_spec& spec)
    {
        return f128{ spec.hi, spec.lo };
    }

    template<>
    [[nodiscard]] mpfr_ref make_value<mpfr_ref>(const value_spec& spec)
    {
        return mpfr_ref(spec.hi) + mpfr_ref(spec.lo);
    }

    template<typename T>
    [[nodiscard]] recurrence_case<T> make_recurrence_case(const recurrence_value_spec& spec)
    {
        recurrence_case<T> out{};
        out.x = make_value<T>(spec.x);
        out.y = make_value<T>(spec.y);
        out.a = make_value<T>(spec.a);
        out.b = make_value<T>(spec.b);
        out.denom_x = make_value<T>(spec.c) + 0.5;
        out.denom_y = make_value<T>(spec.d) + 1.5;
        return out;
    }

    template<typename T, typename Work>
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
        const double ratio = result.mpfr.ns_per_iter / result.f128.ns_per_iter;

        std::cout
            << std::fixed << std::setprecision(2)
            << label
            << "\n  f128 : " << result.f128.total_ms << " ms total, " << result.f128.ns_per_iter << " ns/iter" << "  (total_iterations: " << result.f128.iteration_count << ")"
            << "\n  mpfr : " << result.mpfr.total_ms << " ms total, " << result.mpfr.ns_per_iter << " ns/iter" << "  (total_iterations: " << result.mpfr.iteration_count << ")"
            << "\n  mpfr/f128 ratio: " << ratio << "x"
            << "\n";
    }

    [[nodiscard]] const char* benchmark_group_for_label(std::string_view label)
    {
        if (label == "add" || label == "subtract" || label == "multiply" || label == "divide")
            return "Arithmetic";

        if (label == "floor" || label == "ceil" || label == "trunc" || label == "round" ||
            label == "nearbyint" || label == "rint" || label == "lround" || label == "llround" ||
            label == "lrint" || label == "llrint")
            return "Rounding";

        if (label == "fmod" || label == "remainder" || label == "remquo")
            return "Remainders";

        if (label == "abs" || label == "fabs" ||
            label == "fma" || label == "fmin" || label == "fmax" || label == "fdim" ||
            label == "copysign" || label == "ldexp" || label == "scalbn" || label == "scalbln" ||
            label == "frexp" || label == "modf" || label == "ilogb" || label == "logb" ||
            label == "nextafter" || label.starts_with("nexttoward"))
            return "Floating-point utilities";

        if (label == "sqrt" || label == "cbrt" || label == "hypot" || label == "pow")
            return "Roots & powers";

        if (label == "exp" || label == "exp2" || label == "expm1")
            return "Exponentials";

        if (label == "log" || label == "log2" || label == "log10" || label == "log1p")
            return "Logarithms";

        if (label == "sin" || label == "cos" || label == "tan" || label == "atan" ||
            label == "atan2" || label == "asin" || label == "acos")
            return "Trigonometric";

        if (label == "sinh" || label == "cosh" || label == "tanh")
            return "Hyperbolic";

        if (label == "asinh" || label == "acosh" || label == "atanh")
            return "Inverse hyperbolic";

        if (label == "erf" || label == "erfc" || label == "lgamma" || label == "tgamma")
            return "Special functions";

        return "Other";
    }

    void print_bucketed_results(const char* group, const char* label, const bucketed_comparison_result& results)
    {
        if (std::string_view(label) != "fabs")
            chart_writer.record_result(group, label, results.typical.f128.ns_per_iter, results.typical.mpfr.ns_per_iter);

        const std::string label_prefix = std::string(group) + " / " + label;
        std::string easy_label = label_prefix + " [easy]";
        std::string medium_label = label_prefix + " [medium]";
        std::string hard_label = label_prefix + " [hard]";
        std::string typical_label = label_prefix + " [typical]";

        if constexpr (!only_bench_typical)
        {
            print_result(easy_label.c_str(), results.easy);
            print_result(medium_label.c_str(), results.medium);
            print_result(hard_label.c_str(), results.hard);
        }
        print_result(typical_label.c_str(), results.typical);
    }

    void print_bucketed_results(const char* label, const bucketed_comparison_result& results)
    {
        print_bucketed_results(benchmark_group_for_label(label), label, results);
    }

    void print_mandelbrot_result(const char* label, const comparison_result& result)
    {
        const double ratio = result.mpfr.ns_per_iter / result.f128.ns_per_iter;
        constexpr std::int64_t pixel_count =
            static_cast<std::int64_t>(mandelbrot_kernel_width) * static_cast<std::int64_t>(mandelbrot_kernel_height);
        const double f128_ns_per_pixel = (result.f128.total_ms * 1'000'000.0) / static_cast<double>(pixel_count);
        const double mpfr_ns_per_pixel = (result.mpfr.total_ms * 1'000'000.0) / static_cast<double>(pixel_count);

        std::cout
            << std::fixed << std::setprecision(2)
            << label
            << "\n  f128 : " << result.f128.total_ms << " ms total, " << result.f128.ns_per_iter << " ns/formula-iter, " << f128_ns_per_pixel << " ns/pixel"
            << "  (formula_iterations: " << result.f128.iteration_count << ", pixels: " << pixel_count << ")"
            << "\n  mpfr : " << result.mpfr.total_ms << " ms total, " << result.mpfr.ns_per_iter << " ns/formula-iter, " << mpfr_ns_per_pixel << " ns/pixel"
            << "  (formula_iterations: " << result.mpfr.iteration_count << ", pixels: " << pixel_count << ")"
            << "\n  mpfr/f128 ratio: " << ratio << "x"
            << "\n";
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
    [[nodiscard]] T apply_expm1(const T& x)
    {
        using std::expm1;
        return expm1(x);
    }

    template<typename T>
    [[nodiscard]] T apply_log(const T& x)
    {
        using std::log;
        return log(x);
    }

    //template<typename T>
    //[[nodiscard]] T apply_log_as_double(const T& x)
    //{
    //    return (T)bl::log_as_double(x);
    //}

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
    [[nodiscard]] T apply_log1p(const T& x)
    {
        using std::log1p;
        return log1p(x);
    }

    template<typename T>
    [[nodiscard]] T apply_fmod(const T& x, const T& y)
    {
        using std::fmod;
        return fmod(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_remainder(const T& x, const T& y)
    {
        using std::remainder;
        return remainder(x, y);
    }

    template<typename T>
    [[nodiscard]] T apply_hypot(const T& x, const T& y)
    {
        using std::hypot;
        return hypot(x, y);
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

    // Keep scalar operands non-degenerate so overload benchmarks exercise the advertised precision.
    constexpr std::array<double, 8> arithmetic_f64_scalars{
        0x1.921fb54442d18p+1,
        -0x1.5bf0a8b145769p+1,
        0x1.6a09e667f3bcdp+0,
        -0x1.279a74590331cp-1,
        0x1.9e3779b97f4a8p+0,
        -0x1.62e42fefa39efp-1,
        0x1.26bb1bbb55516p+1,
        -0x1.71547652b82fep+0
    };

    constexpr std::array<float, 8> arithmetic_f32_scalars{
        0x1.921fb6p+1f,
        -0x1.5bf0a8p+1f,
        0x1.6a09e6p+0f,
        -0x1.279a74p-1f,
        0x1.9e377ap+0f,
        -0x1.62e430p-1f,
        0x1.26bb1cp+1f,
        -0x1.715476p+0f
    };

    constexpr std::array<std::int64_t, 8> arithmetic_i64_scalars{
        2ll,
        -3ll,
        5ll,
        -7ll,
        9'007'199'254'740'993ll,
        -9'007'199'254'740'993ll,
        1'234'567'890'123'456'789ll,
        -1'234'567'890'123'456'789ll
    };

    constexpr std::array<std::int32_t, 8> arithmetic_i32_scalars{
        2,
        -3,
        5,
        -7,
        11,
        -13,
        65'537,
        -65'537
    };

    template<typename Value, typename Scalar>
    [[nodiscard]] Value apply_scalar_add(const Value& value, Scalar scalar, bool scalar_left)
    {
        if (scalar_left)
            return Value{ scalar + value };
        return Value{ value + scalar };
    }

    template<typename Value, typename Scalar>
    [[nodiscard]] Value apply_scalar_subtract(const Value& value, Scalar scalar, bool scalar_left)
    {
        if (scalar_left)
            return Value{ scalar - value };
        return Value{ value - scalar };
    }

    template<typename Value, typename Scalar>
    [[nodiscard]] Value apply_scalar_multiply(const Value& value, Scalar scalar, bool scalar_left)
    {
        if (scalar_left)
            return Value{ scalar * value };
        return Value{ value * scalar };
    }

    template<typename Value, typename Scalar>
    [[nodiscard]] Value apply_scalar_divide(const Value& value, Scalar scalar, bool scalar_left)
    {
        if (scalar_left)
            return Value{ scalar / value };
        return Value{ value / scalar };
    }

    template<typename T>
    [[nodiscard]] T apply_nearbyint(const T& x)
    {
        using std::nearbyint;
        return nearbyint(x);
    }

    template<typename T>
    [[nodiscard]] T apply_rint(const T& x)
    {
        using std::rint;
        return rint(x);
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
    [[nodiscard]] T apply_cbrt(const T& x)
    {
        using std::cbrt;
        return cbrt(x);
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

    template<typename T, std::size_t ValueCount, typename Op>
    [[nodiscard]] bench_result benchmark_unary_bucket(
        const std::array<value_spec, ValueCount>& specs,
        std::int64_t total_iterations,
        Op&& op,
        bool scale_to_bucket = true)
    {
        std::array<T, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
            values[i] = make_value<T>(specs[i]);

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < ValueCount; ++i)
                    acc = blend_result(op(values[i]), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Scalar, std::size_t ValueCount, std::size_t ScalarCount, typename Op>
    [[nodiscard]] bench_result benchmark_scalar_overload_value_bucket(
        const std::array<value_spec, ValueCount>& specs,
        const std::array<Scalar, ScalarCount>& scalars,
        std::int64_t total_iterations,
        Op&& op,
        bool scale_to_bucket = true)
    {
        static_assert(ScalarCount > 0);

        std::array<T, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
            values[i] = make_value<T>(specs[i]);

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < ValueCount; ++i)
                {
                    const Scalar scalar = scalars[i % ScalarCount];
                    const bool scalar_left = ((static_cast<std::size_t>(outer) + i) & 1u) != 0u;
                    acc = blend_result(op(values[i], scalar, scalar_left), acc);
                }
            }

            return acc;
        });
    }

    template<typename T, std::size_t ValueCount, typename Op>
    [[nodiscard]] bench_result benchmark_binary_bucket(
        const std::array<binary_value_spec, ValueCount>& specs,
        std::int64_t total_iterations,
        Op&& op,
        bool scale_to_bucket = true)
    {
        std::array<std::pair<T, T>, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
            values[i] = { make_value<T>(specs[i].lhs), make_value<T>(specs[i].rhs) };

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            T acc = values.front().first;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < ValueCount; ++i)
                {
                    const T value = static_cast<T>(op(values[i].first, values[i].second));
                    acc = blend_result(value, acc);
                }
            }

            return acc;
        });
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_ldexp_bucket(
        const std::array<ldexp_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        struct ldexp_case
        {
            T value{};
            int exponent = 0;
        };

        std::array<ldexp_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].value = make_value<T>(specs[i].value);
            values[i].exponent = specs[i].exponent;
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<T>(iteration_count, [&]()
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

    template<mixed_recurrence_workload Workload, typename T, typename Spec, std::size_t ValueCount>
    [[nodiscard]] bench_result benchmark_mixed_recurrence_bucket(
        const std::array<Spec, ValueCount>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        struct scalar_recurrence_case
        {
            T x{};
            T y{};
            T a{};
            T b{};
            T c{};
            T d{};
        };

        std::array<scalar_recurrence_case, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
        {
            values[i].x = make_value<T>(specs[i].x);
            values[i].y = make_value<T>(specs[i].y);
            values[i].a = make_value<T>(specs[i].a);
            values[i].b = make_value<T>(specs[i].b);
            values[i].c = make_value<T>(specs[i].c);
            values[i].d = make_value<T>(specs[i].d);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            auto state = values;
            T acc_x = state.front().x;
            T acc_y = state.front().y;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (auto& item : state)
                {
                    const T x = item.x;
                    const T y = item.y;
                    const T a = item.a;
                    const T b = item.b;
                    const T c = item.c;
                    const T d = item.d;

                    if constexpr (Workload == mixed_recurrence_workload::one)
                    {
                        auto xx = x * x;
                        auto yy = y * y;
                        auto xy0 = x * y;
                        auto xy1 = x * y;

                        item.x = ((std::move(xx) - std::move(yy)) + a) / (c + 0.5);
                        item.y = ((std::move(xy0) + std::move(xy1)) + b) / (d + 1.5);
                    }
                    else if constexpr (Workload == mixed_recurrence_workload::two)
                    {
                        item.x = (((x * a) + (y * b)) + c) / (c + 2.0);
                        item.y = (((x * b) - (a * c)) - d) / (d + 2.5);
                    }
                    else if constexpr (Workload == mixed_recurrence_workload::three)
                    {
                        item.x = (((x * 1.125) + (y * -0.625)) + a) / (c + 1.75);
                        item.y = (b - (x * 0.375)) / (d + 2.25);
                    }
                    else if constexpr (Workload == mixed_recurrence_workload::four)
                    {
                        item.x = ((x + a) + b) / ((c + d) + 1.0);
                        item.y = ((y + c) - d) / ((a + b) + 2.0);
                    }
                    else
                    {
                        item.x = (((x * a) + (y * b)) + (c * d)) / ((c + d) + 2.0);
                        item.y = (((x * b) + (y * a)) + ((c * d) + (a * b))) / (c + 3.0);
                    }

                    acc_x = blend_result(item.x, acc_x);
                    acc_y = blend_result(item.y, acc_y);
                }
            }

            return std::pair<T, T>{ acc_x, acc_y };
        });
    }

    template<typename T, typename Spec, std::size_t ValueCount>
    [[nodiscard]] bench_result benchmark_affine_trig_transform_bucket(
        const std::array<Spec, ValueCount>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        struct transform_case
        {
            T x{};
            T y{};
            T angle{};
            T scale_x{};
            T scale_y{};
            T translate_x{};
            T translate_y{};
        };

        std::array<transform_case, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
        {
            values[i].x = make_value<T>(specs[i].x);
            values[i].y = make_value<T>(specs[i].y);
            values[i].angle = make_value<T>(specs[i].angle);
            values[i].scale_x = make_value<T>(specs[i].scale_x);
            values[i].scale_y = make_value<T>(specs[i].scale_y);
            values[i].translate_x = make_value<T>(specs[i].translate_x);
            values[i].translate_y = make_value<T>(specs[i].translate_y);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            auto state = values;
            T acc_x = state.front().x;
            T acc_y = state.front().y;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (auto& item : state)
                {
                    const T sin_angle = apply_sin(item.angle);
                    const T cos_angle = apply_cos(item.angle);
                    const T neg_sin_angle = -sin_angle;

                    const T m00 = cos_angle * item.scale_x;
                    const T m01 = neg_sin_angle * item.scale_y;
                    const T m10 = sin_angle * item.scale_x;
                    const T m11 = cos_angle * item.scale_y;

                    const T next_x = ((m00 * item.x) + (m01 * item.y) + item.translate_x) / (item.scale_x + 2.0);
                    const T next_y = ((m10 * item.x) + (m11 * item.y) + item.translate_y) / (item.scale_y + 2.0);

                    item.x = next_x;
                    item.y = next_y;

                    acc_x = blend_result(item.x, acc_x);
                    acc_y = blend_result(item.y, acc_y);
                }
            }

            return std::pair<T, T>{ acc_x, acc_y };
        });
    }

    template<typename T, typename Spec, std::size_t ValueCount>
    [[nodiscard]] bench_result benchmark_scalar_mixed_recurrence_bucket(
        const std::array<Spec, ValueCount>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        struct scalar_recurrence_case
        {
            T x{};
            T y{};
            T a{};
            T b{};
            T c{};
            T d{};
        };

        std::array<scalar_recurrence_case, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
        {
            values[i].x = make_value<T>(specs[i].x);
            values[i].y = make_value<T>(specs[i].y);
            values[i].a = make_value<T>(specs[i].a);
            values[i].b = make_value<T>(specs[i].b);
            values[i].c = make_value<T>(specs[i].c);
            values[i].d = make_value<T>(specs[i].d);
        }

        constexpr std::array<double, 8> add_rhs{ 0.125, -0.1875, 0.3125, -0.4375, 0.5625, -0.6875, 0.8125, -0.9375 };
        constexpr std::array<double, 8> add_lhs{ -0.03125, 0.09375, -0.15625, 0.21875, -0.28125, 0.34375, -0.40625, 0.46875 };
        constexpr std::array<double, 8> mul_rhs{ 0.875, -1.125, 1.375, -0.625, 0.5625, -0.8125, 1.0625, -1.3125 };
        constexpr std::array<double, 8> mul_lhs{ -1.0625, 0.6875, -0.9375, 1.1875, -0.75, 1.5, -1.25, 0.8125 };
        constexpr std::array<double, 8> div_rhs{ 1.125, -1.375, 1.625, -1.875, 2.125, -2.375, 2.625, -2.875 };

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            auto state = values;
            T acc_x = state.front().x;
            T acc_y = state.front().y;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < ValueCount; ++i)
                {
                    auto& item = state[i];
                    const std::size_t scalar_index = i % add_rhs.size();

                    const T scalar_rhs_add = item.x + add_rhs[scalar_index];
                    const T scalar_lhs_add = add_lhs[scalar_index] + item.y;
                    const T scalar_rhs_mul = scalar_rhs_add * mul_rhs[scalar_index];
                    const T scalar_lhs_mul = mul_lhs[scalar_index] * scalar_lhs_add;
                    const T scalar_div = scalar_rhs_mul / div_rhs[scalar_index];

                    const T qd_add = scalar_div + item.a;
                    const T qd_sub = scalar_lhs_mul - item.b;
                    const T qd_mul = qd_add * qd_sub;
                    const T c2 = item.c * item.c;
                    const T d2 = item.d * item.d;
                    const T denominator_base = c2 + d2;
                    const T denominator = denominator_base + 1.0;
                    const T qd_div = qd_mul / denominator;

                    const T x_denominator = denominator + 2.0;
                    const T c_half = item.c * 0.5;
                    const T y_denominator_base = denominator + c_half;
                    const T y_denominator = y_denominator_base + 2.5;

                    const T x_step = qd_div / x_denominator;
                    const T x_damping = 0.125 * item.a;
                    const T y_delta = qd_add - qd_sub;
                    const T y_step = y_delta / y_denominator;
                    const T y_damping = item.b * 0.0625;

                    item.x = x_step + x_damping;
                    item.y = y_step - y_damping;

                    acc_x = blend_result(item.x, acc_x);
                    acc_y = blend_result(item.y, acc_y);
                }
            }

            return std::pair<T, T>{ acc_x, acc_y };
        });
    }

    template<typename T, typename Spec, std::size_t ValueCount>
    [[nodiscard]] bench_result benchmark_fused_mixed_expression_bucket(
        const std::array<Spec, ValueCount>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        struct scalar_recurrence_case
        {
            T x{};
            T y{};
            T a{};
            T b{};
            T c{};
            T d{};
        };

        std::array<scalar_recurrence_case, ValueCount> values{};
        for (std::size_t i = 0; i < ValueCount; ++i)
        {
            values[i].x = make_value<T>(specs[i].x);
            values[i].y = make_value<T>(specs[i].y);
            values[i].a = make_value<T>(specs[i].a);
            values[i].b = make_value<T>(specs[i].b);
            values[i].c = make_value<T>(specs[i].c);
            values[i].d = make_value<T>(specs[i].d);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(static_cast<std::int64_t>(ValueCount), target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(ValueCount) - 1) / static_cast<std::int64_t>(ValueCount));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(ValueCount);

        return run_benchmark<T>(iteration_count, [&]()
        {
            auto state = values;
            T acc_x = state.front().x;
            T acc_y = state.front().y;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (auto& item : state)
                {
                    const T sum_plus_value = ((item.x * item.y) + (item.a * item.b)) + item.c;
                    const T diff_minus_value = ((item.x * item.a) - (item.b * item.c)) - item.d;
                    const T c2 = item.c * item.c;
                    const T d2 = item.d * item.d;
                    const T denominator_x = c2 + T(2.0);
                    const T denominator_y = d2 + T(2.5);

                    item.x = sum_plus_value / denominator_x;
                    item.y = diff_minus_value / denominator_y;

                    acc_x = blend_result(item.x, acc_x);
                    acc_y = blend_result(item.y, acc_y);
                }
            }

            return std::pair<T, T>{ acc_x, acc_y };
        });
    }

    template<typename T, typename Result, typename Op>
    [[nodiscard]] bench_result benchmark_value_bucket_result(
        const std::array<value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op,
        bool scale_to_bucket = true)
    {
        std::array<T, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = make_value<T>(specs[i]);

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<Result>(iteration_count, [&]()
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
        Op&& op,
        bool scale_to_bucket = true)
    {
        std::array<std::pair<T, T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].first = make_value<T>(specs[i].lhs);
            values[i].second = make_value<T>(specs[i].rhs);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<Result>(iteration_count, [&]()
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
        Op&& op,
        bool scale_to_bucket = true)
    {
        struct nexttoward_case
        {
            T from{};
            long double to = 0.0L;
        };

        std::array<nexttoward_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].from = make_value<T>(specs[i].lhs);
            const T target = make_value<T>(specs[i].rhs);
            values[i].to = static_cast<long double>(static_cast<double>(target));
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<Result>(iteration_count, [&]()
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
        Op&& op,
        bool scale_to_bucket = true)
    {
        struct exponent_case
        {
            T value{};
            Exponent exponent{};
        };

        std::array<exponent_case, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            values[i].value = make_value<T>(specs[i].value);
            values[i].exponent = static_cast<Exponent>(specs[i].exponent);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<Result>(iteration_count, [&]()
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
        Op&& op,
        bool scale_to_bucket = true)
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
            values[i].x = make_value<T>(specs[i].x);
            values[i].y = make_value<T>(specs[i].y);
            values[i].z = make_value<T>(specs[i].a);
        }

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<Result>(iteration_count, [&]()
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
        const bucket_array_set<value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_value_specs(specs);
        out.typical.f128 = benchmark_value_bucket_result<f128, FltxResult>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_integer_rounding_benchmark_result(
        const bucket_array_set<value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        static_assert(std::is_integral_v<FltxResult> && std::is_signed_v<FltxResult>);

        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_value_bucket_result<f128, FltxResult>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_integer_rounding_typical_specs<FltxResult>(specs);
        out.typical.f128 = benchmark_value_bucket_result<f128, FltxResult>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_value_bucket_result<mpfr_ref, MpfrResult>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_benchmark_result(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_binary_bucket_result<f128, FltxResult>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_binary_bucket_result<f128, FltxResult>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_binary_bucket_result<f128, FltxResult>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_binary_specs(specs);
        out.typical.f128 = benchmark_binary_bucket_result<f128, FltxResult>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_binary_bucket_result<mpfr_ref, MpfrResult>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_long_double_benchmark_result(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_binary_long_double_bucket_result<f128, FltxResult>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_binary_long_double_bucket_result<f128, FltxResult>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_binary_long_double_bucket_result<f128, FltxResult>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_binary_specs(specs);
        out.typical.f128 = benchmark_binary_long_double_bucket_result<f128, FltxResult>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_binary_long_double_bucket_result<mpfr_ref, MpfrResult>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Exponent, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_exponent_benchmark_result(
        const bucket_array_set<ldexp_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_exponent_bucket_result<f128, FltxResult, Exponent>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_exponent_bucket_result<f128, FltxResult, Exponent>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_exponent_bucket_result<f128, FltxResult, Exponent>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_ldexp_specs(specs);
        out.typical.f128 = benchmark_exponent_bucket_result<f128, FltxResult, Exponent>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_exponent_bucket_result<mpfr_ref, MpfrResult, Exponent>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename FltxResult, typename MpfrResult, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_ternary_benchmark_result(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_ternary_bucket_result<f128, FltxResult>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_ternary_bucket_result<f128, FltxResult>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_ternary_bucket_result<f128, FltxResult>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_recurrence_specs(specs);
        out.typical.f128 = benchmark_ternary_bucket_result<f128, FltxResult>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_ternary_bucket_result<mpfr_ref, MpfrResult>(typical_specs, total_iterations, op, false);

        return out;
    }

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_unary_benchmark(
        const bucket_array_set<value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_unary_bucket<f128>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_unary_bucket<f128>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_unary_bucket<f128>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_value_specs(specs);
        out.typical.f128 = benchmark_unary_bucket<f128>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_unary_bucket<mpfr_ref>(typical_specs, total_iterations, op, false);
        return out;
    }

    template<typename Scalar, std::size_t ScalarCount, typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_scalar_overload_value_benchmark(
        const bucket_array_set<value_spec>& specs,
        const std::array<Scalar, ScalarCount>& scalars,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_scalar_overload_value_bucket<f128>(specs.easy, scalars, total_iterations, op);
            out.easy.mpfr = benchmark_scalar_overload_value_bucket<mpfr_ref>(specs.easy, scalars, total_iterations, op);
            out.medium.f128 = benchmark_scalar_overload_value_bucket<f128>(specs.medium, scalars, total_iterations, op);
            out.medium.mpfr = benchmark_scalar_overload_value_bucket<mpfr_ref>(specs.medium, scalars, total_iterations, op);
            out.hard.f128 = benchmark_scalar_overload_value_bucket<f128>(specs.hard, scalars, total_iterations, op);
            out.hard.mpfr = benchmark_scalar_overload_value_bucket<mpfr_ref>(specs.hard, scalars, total_iterations, op);
        }
        const auto typical_specs = make_typical_value_specs(specs);
        out.typical.f128 = benchmark_scalar_overload_value_bucket<f128>(typical_specs, scalars, total_iterations, op, false);
        out.typical.mpfr = benchmark_scalar_overload_value_bucket<mpfr_ref>(typical_specs, scalars, total_iterations, op, false);
        return out;
    }

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_benchmark(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_binary_bucket<f128>(specs.easy, total_iterations, op);
            out.easy.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.easy, total_iterations, op);
            out.medium.f128 = benchmark_binary_bucket<f128>(specs.medium, total_iterations, op);
            out.medium.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.medium, total_iterations, op);
            out.hard.f128 = benchmark_binary_bucket<f128>(specs.hard, total_iterations, op);
            out.hard.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        }
        const auto typical_specs = make_typical_binary_specs(specs);
        out.typical.f128 = benchmark_binary_bucket<f128>(typical_specs, total_iterations, op, false);
        out.typical.mpfr = benchmark_binary_bucket<mpfr_ref>(typical_specs, total_iterations, op, false);
        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_ldexp_benchmark(
        const bucket_array_set<ldexp_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_ldexp_bucket<f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_ldexp_bucket<f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_ldexp_bucket<f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_ldexp_specs(specs);
        out.typical.f128 = benchmark_ldexp_bucket<f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_ldexp_bucket<mpfr_ref>(typical_specs, total_iterations, false);
        return out;
    }

    template<mixed_recurrence_workload Workload>
    [[nodiscard]] bucketed_comparison_result run_bucketed_mixed_recurrence_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_mixed_recurrence_bucket<Workload, f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_mixed_recurrence_bucket<Workload, mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_mixed_recurrence_bucket<Workload, f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_mixed_recurrence_bucket<Workload, mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_mixed_recurrence_bucket<Workload, f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_mixed_recurrence_bucket<Workload, mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_recurrence_specs(specs);
        out.typical.f128 = benchmark_mixed_recurrence_bucket<Workload, f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_mixed_recurrence_bucket<Workload, mpfr_ref>(typical_specs, total_iterations, false);
        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_affine_trig_transform_benchmark(
        const bucket_array_set<affine_transform_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_affine_trig_transform_bucket<f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_affine_trig_transform_bucket<mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_affine_trig_transform_bucket<f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_affine_trig_transform_bucket<mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_affine_trig_transform_bucket<f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_affine_trig_transform_bucket<mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_affine_transform_specs(specs);
        out.typical.f128 = benchmark_affine_trig_transform_bucket<f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_affine_trig_transform_bucket<mpfr_ref>(typical_specs, total_iterations, false);
        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_scalar_mixed_recurrence_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_scalar_mixed_recurrence_bucket<f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_scalar_mixed_recurrence_bucket<mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_scalar_mixed_recurrence_bucket<f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_scalar_mixed_recurrence_bucket<mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_scalar_mixed_recurrence_bucket<f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_scalar_mixed_recurrence_bucket<mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_recurrence_specs(specs);
        out.typical.f128 = benchmark_scalar_mixed_recurrence_bucket<f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_scalar_mixed_recurrence_bucket<mpfr_ref>(typical_specs, total_iterations, false);

        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_fused_mixed_expression_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_fused_mixed_expression_bucket<f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_fused_mixed_expression_bucket<mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_fused_mixed_expression_bucket<f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_fused_mixed_expression_bucket<mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_fused_mixed_expression_bucket<f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_fused_mixed_expression_bucket<mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_recurrence_specs(specs);
        out.typical.f128 = benchmark_fused_mixed_expression_bucket<f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_fused_mixed_expression_bucket<mpfr_ref>(typical_specs, total_iterations, false);

        return out;
    }

    template<typename T>
    struct mandelbrot_kernel_total
    {
        std::int64_t escape_iterations = 0;
        std::int64_t escaped_pixels = 0;
    };

    template<typename T>
    [[nodiscard]] T parse_mandelbrot_value(const char* text);

    template<>
    [[nodiscard]] f128 parse_mandelbrot_value<f128>(const char* text)
    {
        return f128{ to_f128(text) };
    }

    template<>
    [[nodiscard]] mpfr_ref parse_mandelbrot_value<mpfr_ref>(const char* text)
    {
        return mpfr_ref(text);
    }

    template<typename T>
    [[nodiscard]] mandelbrot_kernel_total<T> run_mandelbrot_kernel()
    {
        constexpr int max_iter = 20000;
        constexpr int width = mandelbrot_kernel_width;
        constexpr int height = mandelbrot_kernel_height;

        const T center_x = parse_mandelbrot_value<T>("-1.73200006480238126967529761198455");
        const T center_y = parse_mandelbrot_value<T>("0.00000019235376499049335337716270");
        const T zoom = parse_mandelbrot_value<T>("2.0e+28");

        const T width_value = T(width);
        const T height_value = T(height);
        const T scale_x = T(4.0) / T(zoom * width_value);
        const T scale_y = T(4.0) / T(zoom * height_value);
        const T half_w = T(width) * 0.5;
        const T half_h = T(height) * 0.5;

        mandelbrot_kernel_total<T> total{};

        for (int row = 0; row < height; ++row)
        {
            const int py = height - 1 - row;

            for (int px = 0; px < width; ++px)
            {
                const T cx = center_x + (T(px) - half_w) * scale_x;
                const T cy = center_y + (T(py) - half_h) * scale_y;
                T x = 0;
                T y = 0;

                int iter = 0;
                while (iter < max_iter)
                {
                    const T radius2 = x * x + y * y;
                    if (radius2 > 4.0)
                        break;

                    const T xx = x * x - y * y + cx;
                    y = 2.0 * x * y + cy;
                    x = xx;
                    ++iter;
                }

                total.escape_iterations += iter;
                if (iter != max_iter)
                    ++total.escaped_pixels;
            }
        }

        return total;
    }

    template<typename T>
    [[nodiscard]] bench_result benchmark_mandelbrot_kernel()
    {
        const auto start = clock_type::now();
        const mandelbrot_kernel_total<T> total = run_mandelbrot_kernel<T>();
        const auto end = clock_type::now();

        consume_result(total.escape_iterations);
        consume_result(total.escaped_pixels);

        const std::chrono::duration<double, std::milli> elapsed = end - start;

        bench_result result;
        result.total_ms = elapsed.count();
        result.ns_per_iter = (elapsed.count() * 1'000'000.0) / static_cast<double>(total.escape_iterations);
        result.iteration_count = total.escape_iterations;
        return result;
    }

    template<typename SignedInt>
    [[nodiscard]] bucket_array_set<value_spec> integer_rounding_specs()
    {
        static_assert(std::is_integral_v<SignedInt> && std::is_signed_v<SignedInt>);

        constexpr std::array<value_spec, 8> easy_values{{
            { -8.75, 0.0 },
            { -4.5, 0.0 },
            { -1.125, 0.0 },
            { -0.5, 0.0 },
            { 0.5, 0.0 },
            { 1.125, 0.0 },
            { 4.5, 0.0 },
            { 8.75, 0.0 }
        }};

        constexpr std::array<value_spec, 8> medium_values{{
            { -1048576.25, 0.0 },
            { -1024.75, 0.0 },
            { -16.25, 0.0 },
            { -0.75, 0.0 },
            { 0.75, 0.0 },
            { 16.25, 0.0 },
            { 1024.75, 0.0 },
            { 1048576.25, 0.0 }
        }};

        constexpr std::array<value_spec, 8> hard_values{{
            { -2000000000.25, 0.0 },
            { -1073741824.75, 0.0 },
            { -65536.25, 0.0 },
            { -1.75, 0.0 },
            { 1.75, 0.0 },
            { 65536.25, 0.0 },
            { 1073741824.25, 0.0 },
            { 2000000000.25, 0.0 }
        }};

        constexpr std::array<value_spec, 8> large_hard_values{{
            { -0x1.0p62, -0.75 },
            { -0x1.0p56, -0.5 },
            { -0x1.0p52, -0.25 },
            { -1.75, 0.0 },
            { 1.75, 0.0 },
            { 0x1.0p52, 0.25 },
            { 0x1.0p56, 0.5 },
            { 0x1.0p62, 0.75 }
        }};

        bucket_array_set<value_spec> out{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            out.easy[i] = easy_values[i % easy_values.size()];
            out.medium[i] = medium_values[i % medium_values.size()];
            if constexpr (std::numeric_limits<SignedInt>::digits > 52)
                out.hard[i] = large_hard_values[i % large_hard_values.size()];
            else
                out.hard[i] = hard_values[i % hard_values.size()];
        }

        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> generic_unary_specs()
    {
        static const bucket_array_set<value_spec> data = make_generic_unary_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> rounding_unary_specs()
    {
        static const bucket_array_set<value_spec> data = make_rounding_unary_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> positive_sqrt_specs()
    {
        static const bucket_array_set<value_spec> data = make_positive_sqrt_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> positive_log_specs()
    {
        static const bucket_array_set<value_spec> data = make_positive_log_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> exponent_specs()
    {
        static const bucket_array_set<value_spec> data = make_exponent_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> log1p_specs()
    {
        static const bucket_array_set<value_spec> data = make_log1p_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> hyperbolic_specs()
    {
        static const bucket_array_set<value_spec> data = make_hyperbolic_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> acosh_specs()
    {
        static const bucket_array_set<value_spec> data = make_acosh_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> atanh_specs()
    {
        static const bucket_array_set<value_spec> data = make_atanh_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> erf_specs()
    {
        static const bucket_array_set<value_spec> data = make_erf_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> gamma_specs()
    {
        static const bucket_array_set<value_spec> data = make_gamma_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> hypot_specs()
    {
        static const bucket_array_set<binary_value_spec> data = make_hypot_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<ldexp_value_spec> ldexp_specs()
    {
        static const bucket_array_set<ldexp_value_spec> data = make_ldexp_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> trig_specs()
    {
        static const bucket_array_set<value_spec> data = make_trig_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<value_spec> inverse_trig_specs()
    {
        static const bucket_array_set<value_spec> data = make_inverse_trig_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> generic_binary_specs()
    {
        static const bucket_array_set<binary_value_spec> data = make_generic_binary_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> pow_specs()
    {
        static const bucket_array_set<binary_value_spec> data = make_pow_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<binary_value_spec> fmod_specs()
    {
        static const bucket_array_set<binary_value_spec> data = make_fmod_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<recurrence_value_spec> mixed_recurrence_specs()
    {
        static const bucket_array_set<recurrence_value_spec> data = make_recurrence_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<recurrence_value_spec> scalar_mixed_recurrence_specs()
    {
        static const bucket_array_set<recurrence_value_spec> data = make_scalar_mixed_recurrence_specs();
        return data;
    }

    [[nodiscard]] bucket_array_set<affine_transform_value_spec> affine_transform_specs()
    {
        static const bucket_array_set<affine_transform_value_spec> data = make_affine_transform_specs();
        return data;
    }

    template<typename Op>
    void print_f128_scalar_overload_results(
        const char* label,
        std::int64_t total_iterations,
        Op&& op)
    {
        if constexpr (generate_compact_report)
            return;

        const auto specs = generic_unary_specs();
        print_bucketed_results("f128 <-> f64", label, run_bucketed_scalar_overload_value_benchmark(specs, arithmetic_f64_scalars, total_iterations, op));
        print_bucketed_results("f128 <-> f32", label, run_bucketed_scalar_overload_value_benchmark(specs, arithmetic_f32_scalars, total_iterations, op));
        print_bucketed_results("f128 <-> i64", label, run_bucketed_scalar_overload_value_benchmark(specs, arithmetic_i64_scalars, total_iterations, op));
        print_bucketed_results("f128 <-> i32", label, run_bucketed_scalar_overload_value_benchmark(specs, arithmetic_i32_scalars, total_iterations, op));
    }
}

TEST_CASE("f128 vs mpfr add performance", "[bench][fltx][f128][arithmetic][add]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a + b;
    });
    print_bucketed_results("f128 <-> f128", "add", results);
    print_f128_scalar_overload_results("add", total_iterations, [](const auto& value, auto scalar, bool scalar_left)
    {
        return apply_scalar_add(value, scalar, scalar_left);
    });
}

TEST_CASE("f128 vs mpfr subtract performance", "[bench][fltx][f128][arithmetic][subtract]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a - b;
    });
    print_bucketed_results("f128 <-> f128", "subtract", results);
    print_f128_scalar_overload_results("subtract", total_iterations, [](const auto& value, auto scalar, bool scalar_left)
    {
        return apply_scalar_subtract(value, scalar, scalar_left);
    });
}

TEST_CASE("f128 vs mpfr multiply performance", "[bench][fltx][f128][arithmetic][multiply]")
{
    const std::int64_t total_iterations = 20000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a * b;
    });
    print_bucketed_results("f128 <-> f128", "multiply", results);
    print_f128_scalar_overload_results("multiply", total_iterations, [](const auto& value, auto scalar, bool scalar_left)
    {
        return apply_scalar_multiply(value, scalar, scalar_left);
    });
}

TEST_CASE("f128 vs mpfr divide performance", "[bench][fltx][f128][arithmetic][divide]")
{
    const std::int64_t total_iterations = 12000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a / b;
    });
    print_bucketed_results("f128 <-> f128", "divide", results);
    print_f128_scalar_overload_results("divide", total_iterations, [](const auto& value, auto scalar, bool scalar_left)
    {
        return apply_scalar_divide(value, scalar, scalar_left);
    });
}

TEST_CASE("f128 vs mpfr square-difference mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload]")
{
    const std::int64_t total_iterations = 25000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark<mixed_recurrence_workload::one>(mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, mixed_recurrence_square_diff_label, results);
}

TEST_CASE("f128 vs mpfr product-sum mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark<mixed_recurrence_workload::two>(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, mixed_recurrence_product_sum_label, results);
}

TEST_CASE("f128 vs mpfr scaled-product-sum mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark<mixed_recurrence_workload::three>(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, mixed_recurrence_scaled_product_sum_label, results);
}

TEST_CASE("f128 vs mpfr shifted-sum mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark<mixed_recurrence_workload::four>(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, mixed_recurrence_shifted_sum_label, results);
}

TEST_CASE("f128 vs mpfr three-product-sum mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_mixed_recurrence_benchmark<mixed_recurrence_workload::five>(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, mixed_recurrence_three_product_sum_label, results);
}

TEST_CASE("f128 vs mpfr affine trig transform performance", "[bench][fltx][f128][mixed][mixed-workload][affine][trig]")
{
    const std::int64_t total_iterations = 800ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_affine_trig_transform_benchmark(affine_transform_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, affine_trig_transform_label, results);
}

TEST_CASE("f128 vs mpfr mandelbrot kernel performance", "[bench][fltx][f128][mixed][mixed-workload][mandelbrot][kernel]")
{
    comparison_result result{};
    result.f128 = benchmark_mandelbrot_kernel<f128>();
    result.mpfr = benchmark_mandelbrot_kernel<mpfr_ref>();

    chart_writer.record_result(bl::bench::mixed_workloads_group_name, mandelbrot_kernel_label, result.f128.ns_per_iter, result.mpfr.ns_per_iter);
    print_mandelbrot_result("mandelbrot kernel [32x32]", result);
}

TEST_CASE("f128 vs mpfr scalar mixed recurrence performance", "[bench][fltx][f128][mixed][mixed-workload][scalar]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_scalar_mixed_recurrence_benchmark(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, scalar_mixed_recurrence_label, results);
}

TEST_CASE("f128 vs mpfr fused mixed expression performance", "[bench][fltx][f128][mixed][mixed-workload][fused]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_fused_mixed_expression_benchmark(scalar_mixed_recurrence_specs(), total_iterations);
    print_bucketed_results(bl::bench::mixed_workloads_group_name, fused_mixed_expression_label, results);
}


TEST_CASE("f128 vs mpfr abs performance", "[bench][fltx][f128][arithmetic][abs]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f128, mpfr_ref>(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_abs(x); });
    print_bucketed_results("abs", results);
}

TEST_CASE("f128 vs mpfr fabs performance", "[bench][fltx][f128][arithmetic][fabs]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f128, mpfr_ref>(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_fabs(x); });
    print_bucketed_results("fabs", results);
}

TEST_CASE("f128 vs mpfr lround performance", "[bench][fltx][f128][rounding][lround]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_integer_rounding_benchmark_result<long, long>(integer_rounding_specs<long>(), total_iterations, [](const auto& x) { return apply_lround(x); });
    print_bucketed_results("lround", results);
}

TEST_CASE("f128 vs mpfr llround performance", "[bench][fltx][f128][rounding][llround]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_integer_rounding_benchmark_result<long long, long long>(integer_rounding_specs<long long>(), total_iterations, [](const auto& x) { return apply_llround(x); });
    print_bucketed_results("llround", results);
}

TEST_CASE("f128 vs mpfr lrint performance", "[bench][fltx][f128][rounding][lrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_integer_rounding_benchmark_result<long, long>(integer_rounding_specs<long>(), total_iterations, [](const auto& x) { return apply_lrint(x); });
    print_bucketed_results("lrint", results);
}

TEST_CASE("f128 vs mpfr llrint performance", "[bench][fltx][f128][rounding][llrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_integer_rounding_benchmark_result<long long, long long>(integer_rounding_specs<long long>(), total_iterations, [](const auto& x) { return apply_llrint(x); });
    print_bucketed_results("llrint", results);
}

TEST_CASE("f128 vs mpfr remquo performance", "[bench][fltx][f128][remquo]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<std::pair<f128, int>, std::pair<mpfr_ref, int>>(fmod_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_remquo(x, y); });
    print_bucketed_results("remquo", results);
}

TEST_CASE("f128 vs mpfr fma performance", "[bench][fltx][f128][fma]")
{
    const std::int64_t total_iterations = 20000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_ternary_benchmark_result<f128, mpfr_ref>(mixed_recurrence_specs(), total_iterations, [](const auto& x, const auto& y, const auto& z) { return apply_fma(x, y, z); });
    print_bucketed_results("fma", results);
}

TEST_CASE("f128 vs mpfr fmin performance", "[bench][fltx][f128][fmin]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_fmin(x, y); });
    print_bucketed_results("fmin", results);
}

TEST_CASE("f128 vs mpfr fmax performance", "[bench][fltx][f128][fmax]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_fmax(x, y); });
    print_bucketed_results("fmax", results);
}

TEST_CASE("f128 vs mpfr fdim performance", "[bench][fltx][f128][fdim]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_fdim(x, y); });
    print_bucketed_results("fdim", results);
}

TEST_CASE("f128 vs mpfr copysign performance", "[bench][fltx][f128][copysign]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_copysign(x, y); });
    print_bucketed_results("copysign", results);
}

TEST_CASE("f128 vs mpfr scalbn performance", "[bench][fltx][f128][scalbn]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_exponent_benchmark_result<f128, mpfr_ref, int>(ldexp_specs(), total_iterations, [](const auto& x, int e) { return apply_scalbn(x, e); });
    print_bucketed_results("scalbn", results);
}

TEST_CASE("f128 vs mpfr scalbln performance", "[bench][fltx][f128][scalbln]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_exponent_benchmark_result<f128, mpfr_ref, long>(ldexp_specs(), total_iterations, [](const auto& x, long e) { return apply_scalbln(x, e); });
    print_bucketed_results("scalbln", results);
}

TEST_CASE("f128 vs mpfr frexp performance", "[bench][fltx][f128][frexp]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<std::pair<f128, int>, std::pair<mpfr_ref, int>>(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_frexp(x); });
    print_bucketed_results("frexp", results);
}

TEST_CASE("f128 vs mpfr modf performance", "[bench][fltx][f128][modf]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<std::pair<f128, f128>, std::pair<mpfr_ref, mpfr_ref>>(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_modf(x); });
    print_bucketed_results("modf", results);
}

TEST_CASE("f128 vs mpfr ilogb performance", "[bench][fltx][f128][ilogb]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<int, int>(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_ilogb(x); });
    print_bucketed_results("ilogb", results);
}

TEST_CASE("f128 vs mpfr logb performance", "[bench][fltx][f128][logb]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<f128, mpfr_ref>(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_logb(x); });
    print_bucketed_results("logb", results);
}

TEST_CASE("f128 vs mpfr nextafter performance", "[bench][fltx][f128][nextafter]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& from, const auto& to) { return apply_nextafter(from, to); });
    print_bucketed_results("nextafter", results);
}

TEST_CASE("f128 vs mpfr nexttoward(type) performance", "[bench][fltx][f128][nexttoward]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& from, const auto& to) { return apply_nexttoward(from, to); });
    print_bucketed_results("nexttoward(type)", results);
}

TEST_CASE("f128 vs mpfr nexttoward(long double) performance", "[bench][fltx][f128][nexttoward]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_long_double_benchmark_result<f128, mpfr_ref>(generic_binary_specs(), total_iterations, [](const auto& from, long double to) { return apply_nexttoward_long_double(from, to); });
    print_bucketed_results("nexttoward(long double)", results);
}

TEST_CASE("f128 vs mpfr floor performance", "[bench][fltx][f128][rounding]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_floor(x); });
    print_bucketed_results("floor", results);
}

TEST_CASE("f128 vs mpfr ceil performance", "[bench][fltx][f128][rounding]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_ceil(x); });
    print_bucketed_results("ceil", results);
}

TEST_CASE("f128 vs mpfr trunc performance", "[bench][fltx][f128][rounding]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_trunc(x); });
    print_bucketed_results("trunc", results);
}

TEST_CASE("f128 vs mpfr round performance", "[bench][fltx][f128][rounding]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_round(x); });
    print_bucketed_results("round", results);
}

TEST_CASE("f128 vs mpfr sqrt performance", "[bench][fltx][f128][sqrt]")
{
    const std::int64_t total_iterations = 12000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(positive_sqrt_specs(), total_iterations, [](const auto& x) { return apply_sqrt(x); });
    print_bucketed_results("sqrt", results);
}

TEST_CASE("f128 vs mpfr exp performance", "[bench][fltx][f128][exp]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(exponent_specs(), total_iterations, [](const auto& x) { return apply_exp(x); });
    print_bucketed_results("exp", results);
}

TEST_CASE("f128 vs mpfr exp2 performance", "[bench][fltx][f128][exp2]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(exponent_specs(), total_iterations, [](const auto& x) { return apply_exp2(x); });
    print_bucketed_results("exp2", results);
}

TEST_CASE("f128 vs mpfr ldexp performance", "[bench][fltx][f128][ldexp]")
{
    const std::int64_t total_iterations = 16000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_ldexp_benchmark(ldexp_specs(), total_iterations);
    print_bucketed_results("ldexp", results);
}

//TEST_CASE("f128 vs mpfr log_as_double performance", "[bench][fltx][f128][log]")
//{
//    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
//    const auto results = run_bucketed_unary_benchmark(positive_log_specs(), total_iterations, [](const auto& x) { return apply_log_as_double(x); });
//    print_bucketed_results("log", results);
//}

TEST_CASE("f128 vs mpfr log performance", "[bench][fltx][f128][log]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(positive_log_specs(), total_iterations, [](const auto& x) { return apply_log(x); });
    print_bucketed_results("log", results);
}

TEST_CASE("f128 vs mpfr log2 performance", "[bench][fltx][f128][log2]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(positive_log_specs(), total_iterations, [](const auto& x) { return apply_log2(x); });
    print_bucketed_results("log2", results);
}

TEST_CASE("f128 vs mpfr log10 performance", "[bench][fltx][f128][log10]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(positive_log_specs(), total_iterations, [](const auto& x) { return apply_log10(x); });
    print_bucketed_results("log10", results);
}

TEST_CASE("f128 vs mpfr fmod performance", "[bench][fltx][f128][fmod]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(fmod_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_fmod(x, y); });
    print_bucketed_results("fmod", results);
}

TEST_CASE("f128 vs mpfr pow performance", "[bench][fltx][f128][pow]")
{
    const std::int64_t total_iterations = 3000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(pow_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_pow(x, y); });
    print_bucketed_results("pow", results);
}

TEST_CASE("f128 vs mpfr sin performance", "[bench][fltx][f128][transcendental][trig][sin]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(trig_specs(), total_iterations, [](const auto& x) { return apply_sin(x); });
    print_bucketed_results("sin", results);
}

TEST_CASE("f128 vs mpfr cos performance", "[bench][fltx][f128][transcendental][trig][cos]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(trig_specs(), total_iterations, [](const auto& x) { return apply_cos(x); });
    print_bucketed_results("cos", results);
}

TEST_CASE("f128 vs mpfr tan performance", "[bench][fltx][f128][transcendental][trig][tan]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(trig_specs(), total_iterations, [](const auto& x) { return apply_tan(x); });
    print_bucketed_results("tan", results);
}

TEST_CASE("f128 vs mpfr atan performance", "[bench][fltx][f128][transcendental][trig][atan]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto typical_specs = make_typical_atan_specs();
    bucketed_comparison_result results{};
    results.typical.f128 = benchmark_unary_bucket<f128>(typical_specs, total_iterations, [](const auto& x) { return apply_atan(x); }, false);
    results.typical.mpfr = benchmark_unary_bucket<mpfr_ref>(typical_specs, total_iterations, [](const auto& x) { return apply_atan(x); }, false);
    print_bucketed_results("atan", results);
}

TEST_CASE("f128 vs mpfr atan2 performance", "[bench][fltx][f128][transcendental][trig][atan2]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto typical_specs = make_typical_atan2_specs();
    bucketed_comparison_result results{};
    results.typical.f128 = benchmark_binary_bucket<f128>(typical_specs, total_iterations, [](const auto& y, const auto& x) { return apply_atan2(y, x); }, false);
    results.typical.mpfr = benchmark_binary_bucket<mpfr_ref>(typical_specs, total_iterations, [](const auto& y, const auto& x) { return apply_atan2(y, x); }, false);
    print_bucketed_results("atan2", results);
}

TEST_CASE("f128 vs mpfr asin performance", "[bench][fltx][f128][transcendental][trig][asin]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(inverse_trig_specs(), total_iterations, [](const auto& x) { return apply_asin(x); });
    print_bucketed_results("asin", results);
}

TEST_CASE("f128 vs mpfr acos performance", "[bench][fltx][f128][transcendental][trig][acos]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(inverse_trig_specs(), total_iterations, [](const auto& x) { return apply_acos(x); });
    print_bucketed_results("acos", results);
}

TEST_CASE("f128 vs mpfr nearbyint performance", "[bench][fltx][f128][rounding][nearbyint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_nearbyint(x); });
    print_bucketed_results("nearbyint", results);
}

TEST_CASE("f128 vs mpfr rint performance", "[bench][fltx][f128][rounding][rint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(rounding_unary_specs(), total_iterations, [](const auto& x) { return apply_rint(x); });
    print_bucketed_results("rint", results);
}

TEST_CASE("f128 vs mpfr cbrt performance", "[bench][fltx][f128][cbrt]")
{
    const std::int64_t total_iterations = 8000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_cbrt(x); });
    print_bucketed_results("cbrt", results);
}

TEST_CASE("f128 vs mpfr expm1 performance", "[bench][fltx][f128][expm1]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(exponent_specs(), total_iterations, [](const auto& x) { return apply_expm1(x); });
    print_bucketed_results("expm1", results);
}

TEST_CASE("f128 vs mpfr log1p performance", "[bench][fltx][f128][log1p]")
{
    const std::int64_t total_iterations = 6000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(log1p_specs(), total_iterations, [](const auto& x) { return apply_log1p(x); });
    print_bucketed_results("log1p", results);
}

TEST_CASE("f128 vs mpfr remainder performance", "[bench][fltx][f128][remainder]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(fmod_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_remainder(x, y); });
    print_bucketed_results("remainder", results);
}

TEST_CASE("f128 vs mpfr hypot performance", "[bench][fltx][f128][hypot]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(hypot_specs(), total_iterations, [](const auto& x, const auto& y) { return apply_hypot(x, y); });
    print_bucketed_results("hypot", results);
}

TEST_CASE("f128 vs mpfr sinh performance", "[bench][fltx][f128][transcendental][hyperbolic][sinh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(hyperbolic_specs(), total_iterations, [](const auto& x) { return apply_sinh(x); });
    print_bucketed_results("sinh", results);
}

TEST_CASE("f128 vs mpfr cosh performance", "[bench][fltx][f128][transcendental][hyperbolic][cosh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(hyperbolic_specs(), total_iterations, [](const auto& x) { return apply_cosh(x); });
    print_bucketed_results("cosh", results);
}

TEST_CASE("f128 vs mpfr tanh performance", "[bench][fltx][f128][transcendental][hyperbolic][tanh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(hyperbolic_specs(), total_iterations, [](const auto& x) { return apply_tanh(x); });
    print_bucketed_results("tanh", results);
}

TEST_CASE("f128 vs mpfr asinh performance", "[bench][fltx][f128][transcendental][hyperbolic][asinh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(generic_unary_specs(), total_iterations, [](const auto& x) { return apply_asinh(x); });
    print_bucketed_results("asinh", results);
}

TEST_CASE("f128 vs mpfr acosh performance", "[bench][fltx][f128][transcendental][hyperbolic][acosh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(acosh_specs(), total_iterations, [](const auto& x) { return apply_acosh(x); });
    print_bucketed_results("acosh", results);
}

TEST_CASE("f128 vs mpfr atanh performance", "[bench][fltx][f128][transcendental][hyperbolic][atanh]")
{
    const std::int64_t total_iterations = 5000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(atanh_specs(), total_iterations, [](const auto& x) { return apply_atanh(x); });
    print_bucketed_results("atanh", results);
}

TEST_CASE("f128 vs mpfr erf performance", "[bench][fltx][f128][special][erf]")
{
    const std::int64_t total_iterations = 600ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(erf_specs(), total_iterations, [](const auto& x) { return apply_erf(x); });
    print_bucketed_results("erf", results);
}

TEST_CASE("f128 vs mpfr erfc performance", "[bench][fltx][f128][special][erfc]")
{
    const std::int64_t total_iterations = 250ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(erf_specs(), total_iterations, [](const auto& x) { return apply_erfc(x); });
    print_bucketed_results("erfc", results);
}

TEST_CASE("f128 vs mpfr lgamma performance", "[bench][fltx][f128][special][gamma][lgamma]")
{
    const std::int64_t total_iterations = 1200ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(gamma_specs(), total_iterations, [](const auto& x) { return apply_lgamma(x); });
    print_bucketed_results("lgamma", results);
}

TEST_CASE("f128 vs mpfr tgamma performance", "[bench][fltx][f128][special][gamma][tgamma]")
{
    const std::int64_t total_iterations = 1200ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_unary_benchmark(gamma_specs(), total_iterations, [](const auto& x) { return apply_tgamma(x); });
    print_bucketed_results("tgamma", results);
}
