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
#include "benchmark_chart_writer.h"

using namespace bl;

namespace
{
    constexpr unsigned mpfr_digits10 = std::numeric_limits<f128>::digits10;
    using mpfr_ref = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<mpfr_digits10>>;
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 20;
    constexpr bool only_bench_typical = true;
    constexpr std::size_t bucket_value_count = 64;
    constexpr std::size_t bucket_count = 3;

    bl::bench::benchmark_chart_writer chart_writer{
        "f128",
        "mpfr",
        "f128 vs MPFR typical benchmark ratios",
        "benchmark_charts/f128_typical_ratios.csv",
        "benchmark_charts/f128_typical_ratios.svg"
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
        if (label == "add" || label == "subtract" || label == "multiply" || label == "divide" ||
            label == "mixed recurrence")
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

    void print_bucketed_results(const char* label, const bucketed_comparison_result& results)
    {
        if (std::string_view(label) != "fabs")
            chart_writer.record_result(benchmark_group_for_label(label), label, results.typical.f128.ns_per_iter, results.typical.mpfr.ns_per_iter);

        std::string easy_label = std::string(label) + " [easy]";
        std::string medium_label = std::string(label) + " [medium]";
        std::string hard_label = std::string(label) + " [hard]";
        std::string typical_label = std::string(label) + " [typical]";

        if constexpr (!only_bench_typical)
        {
            print_result(easy_label.c_str(), results.easy);
            print_result(medium_label.c_str(), results.medium);
            print_result(hard_label.c_str(), results.hard);
        }
        print_result(typical_label.c_str(), results.typical);
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

    template<typename T, typename Op>
    [[nodiscard]] bench_result benchmark_unary_bucket(
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

        return run_benchmark<T>(iteration_count, [&]()
        {
            T acc = values.front();

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
                    acc = blend_result(op(values[i]), acc);
            }

            return acc;
        });
    }

    template<typename T, typename Op>
    [[nodiscard]] bench_result benchmark_binary_bucket(
        const std::array<binary_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op,
        bool scale_to_bucket = true)
    {
        std::array<std::pair<T, T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = { make_value<T>(specs[i].lhs), make_value<T>(specs[i].rhs) };

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<T>(iteration_count, [&]()
        {
            T acc = values.front().first;

            for (std::int64_t outer = 0; outer < outer_loops; ++outer)
            {
                for (std::size_t i = 0; i < bucket_value_count; ++i)
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

    template<typename T>
    [[nodiscard]] bench_result benchmark_mixed_recurrence_bucket(
        const std::array<recurrence_value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        bool scale_to_bucket = true)
    {
        std::array<recurrence_case<T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = make_recurrence_case<T>(specs[i]);

        const std::int64_t target_iterations = scale_to_bucket ? total_iterations / static_cast<std::int64_t>(bucket_count) : total_iterations;
        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, target_iterations);
        const std::int64_t outer_loops = std::max<std::int64_t>(1, (bucket_iterations + static_cast<std::int64_t>(bucket_value_count) - 1) / static_cast<std::int64_t>(bucket_value_count));
        const std::int64_t iteration_count = outer_loops * static_cast<std::int64_t>(bucket_value_count);

        return run_benchmark<T>(iteration_count, [&]()
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

                    item.x = ((xx - yy) + item.a) / item.denom_x;
                    item.y = ((xy + xy) + item.b) / item.denom_y;

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

    [[nodiscard]] bucketed_comparison_result run_bucketed_mixed_recurrence_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        if constexpr (!only_bench_typical)
        {
            out.easy.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.easy, total_iterations);
            out.easy.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.easy, total_iterations);
            out.medium.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.medium, total_iterations);
            out.medium.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.medium, total_iterations);
            out.hard.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.hard, total_iterations);
            out.hard.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.hard, total_iterations);
        }
        const auto typical_specs = make_typical_recurrence_specs(specs);
        out.typical.f128 = benchmark_mixed_recurrence_bucket<f128>(typical_specs, total_iterations, false);
        out.typical.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(typical_specs, total_iterations, false);
        return out;
    }

    [[nodiscard]] bucket_array_set<value_spec> integer_rounding_specs()
    {
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

        bucket_array_set<value_spec> out{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
        {
            out.easy[i] = easy_values[i % easy_values.size()];
            out.medium[i] = medium_values[i % medium_values.size()];
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

    [[nodiscard]] bucket_array_set<binary_value_spec> atan2_specs()
    {
        static const bucket_array_set<binary_value_spec> data = make_atan2_specs();
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
}

TEST_CASE("f128 vs mpfr add performance", "[bench][fltx][f128][arithmetic][add]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a + b;
    });
    print_bucketed_results("add", results);
}

TEST_CASE("f128 vs mpfr subtract performance", "[bench][fltx][f128][arithmetic][subtract]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a - b;
    });
    print_bucketed_results("subtract", results);
}

TEST_CASE("f128 vs mpfr multiply performance", "[bench][fltx][f128][arithmetic][multiply]")
{
    const std::int64_t total_iterations = 20000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a * b;
    });
    print_bucketed_results("multiply", results);
}

TEST_CASE("f128 vs mpfr divide performance", "[bench][fltx][f128][arithmetic][divide]")
{
    const std::int64_t total_iterations = 12000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(generic_binary_specs(), total_iterations, [](const auto& a, const auto& b)
    {
        return a / b;
    });
    print_bucketed_results("divide", results);
}

TEST_CASE("f128 vs mpfr mixed recurrence performance", "[bench][fltx][f128][arithmetic]")
{
    const std::int64_t total_iterations = 40000ll * benchmark_scale;
    const auto results = run_bucketed_mixed_recurrence_benchmark(mixed_recurrence_specs(), total_iterations);
    print_bucketed_results("mixed recurrence", results);
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
    const auto results = run_bucketed_value_benchmark_result<long, long>(integer_rounding_specs(), total_iterations, [](const auto& x) { return apply_lround(x); });
    print_bucketed_results("lround", results);
}

TEST_CASE("f128 vs mpfr llround performance", "[bench][fltx][f128][rounding][llround]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long long, long long>(integer_rounding_specs(), total_iterations, [](const auto& x) { return apply_llround(x); });
    print_bucketed_results("llround", results);
}

TEST_CASE("f128 vs mpfr lrint performance", "[bench][fltx][f128][rounding][lrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long, long>(integer_rounding_specs(), total_iterations, [](const auto& x) { return apply_lrint(x); });
    print_bucketed_results("lrint", results);
}

TEST_CASE("f128 vs mpfr llrint performance", "[bench][fltx][f128][rounding][llrint]")
{
    const std::int64_t total_iterations = 50000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_value_benchmark_result<long long, long long>(integer_rounding_specs(), total_iterations, [](const auto& x) { return apply_llrint(x); });
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
    const auto results = run_bucketed_unary_benchmark(trig_specs(), total_iterations, [](const auto& x) { return apply_atan(x); });
    print_bucketed_results("atan", results);
}

TEST_CASE("f128 vs mpfr atan2 performance", "[bench][fltx][f128][transcendental][trig][atan2]")
{
    const std::int64_t total_iterations = 4000ll * benchmark_scale * 8ll;
    const auto results = run_bucketed_binary_benchmark(atan2_specs(), total_iterations, [](const auto& y, const auto& x) { return apply_atan2(y, x); });
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
