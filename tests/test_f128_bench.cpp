#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include <fltx/fltx_core.h>

using namespace bl;

namespace
{
    constexpr unsigned mpfr_digits10 = std::numeric_limits<f128>::digits10;
    using mpfr_ref = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<mpfr_digits10>>;
    using clock_type = std::chrono::steady_clock;

    constexpr int benchmark_scale = 50;
    constexpr std::size_t bucket_value_count = 64;
    constexpr std::size_t bucket_count = 3;

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
        comparison_result average{};
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
        T c{};
        T d{};
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
        const f128 r = bl::_f128_detail::renorm(hi, lo);
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
        out.c = make_value<T>(spec.c);
        out.d = make_value<T>(spec.d);
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

    void print_bucketed_results(const char* label, const bucketed_comparison_result& results)
    {
        std::string easy_label = std::string(label) + " [easy]";
        std::string medium_label = std::string(label) + " [medium]";
        std::string hard_label = std::string(label) + " [hard]";
        std::string average_label = std::string(label) + " [average]";

        print_result(easy_label.c_str(), results.easy);
        print_result(medium_label.c_str(), results.medium);
        print_result(hard_label.c_str(), results.hard);
        print_result(average_label.c_str(), results.average);
    }

    [[nodiscard]] comparison_result combine_results(
        const comparison_result& easy,
        const comparison_result& medium,
        const comparison_result& hard)
    {
        const auto combine = [](const bench_result& a, const bench_result& b, const bench_result& c)
        {
            bench_result out{};
            out.total_ms = a.total_ms + b.total_ms + c.total_ms;
            out.iteration_count = a.iteration_count + b.iteration_count + c.iteration_count;
            out.ns_per_iter = (out.total_ms * 1'000'000.0) / static_cast<double>(out.iteration_count);
            return out;
        };

        comparison_result out{};
        out.f128 = combine(easy.f128, medium.f128, hard.f128);
        out.mpfr = combine(easy.mpfr, medium.mpfr, hard.mpfr);
        return out;
    }

    template<typename T>
    [[nodiscard]] T blend_result(const T& value, const T& acc)
    {
        return value + acc * T(0.25);
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

    template<typename T, typename Op>
    [[nodiscard]] bench_result benchmark_unary_bucket(
        const std::array<value_spec, bucket_value_count>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        std::array<T, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = make_value<T>(specs[i]);

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
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
        Op&& op)
    {
        std::array<std::pair<T, T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = { make_value<T>(specs[i].lhs), make_value<T>(specs[i].rhs) };

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
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
            values[i].value = make_value<T>(specs[i].value);
            values[i].exponent = specs[i].exponent;
        }

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
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
        std::int64_t total_iterations)
    {
        std::array<recurrence_case<T>, bucket_value_count> values{};
        for (std::size_t i = 0; i < bucket_value_count; ++i)
            values[i] = make_recurrence_case<T>(specs[i]);

        const std::int64_t bucket_iterations = std::max<std::int64_t>(bucket_value_count, total_iterations / static_cast<std::int64_t>(bucket_count));
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

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_unary_benchmark(
        const bucket_array_set<value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f128 = benchmark_unary_bucket<f128>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.easy, total_iterations, op);
        out.medium.f128 = benchmark_unary_bucket<f128>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.medium, total_iterations, op);
        out.hard.f128 = benchmark_unary_bucket<f128>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_unary_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        out.average = combine_results(out.easy, out.medium, out.hard);
        return out;
    }

    template<typename Op>
    [[nodiscard]] bucketed_comparison_result run_bucketed_binary_benchmark(
        const bucket_array_set<binary_value_spec>& specs,
        std::int64_t total_iterations,
        Op&& op)
    {
        bucketed_comparison_result out{};
        out.easy.f128 = benchmark_binary_bucket<f128>(specs.easy, total_iterations, op);
        out.easy.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.easy, total_iterations, op);
        out.medium.f128 = benchmark_binary_bucket<f128>(specs.medium, total_iterations, op);
        out.medium.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.medium, total_iterations, op);
        out.hard.f128 = benchmark_binary_bucket<f128>(specs.hard, total_iterations, op);
        out.hard.mpfr = benchmark_binary_bucket<mpfr_ref>(specs.hard, total_iterations, op);
        out.average = combine_results(out.easy, out.medium, out.hard);
        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_ldexp_benchmark(
        const bucket_array_set<ldexp_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        out.easy.f128 = benchmark_ldexp_bucket<f128>(specs.easy, total_iterations);
        out.easy.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.easy, total_iterations);
        out.medium.f128 = benchmark_ldexp_bucket<f128>(specs.medium, total_iterations);
        out.medium.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.medium, total_iterations);
        out.hard.f128 = benchmark_ldexp_bucket<f128>(specs.hard, total_iterations);
        out.hard.mpfr = benchmark_ldexp_bucket<mpfr_ref>(specs.hard, total_iterations);
        out.average = combine_results(out.easy, out.medium, out.hard);
        return out;
    }

    [[nodiscard]] bucketed_comparison_result run_bucketed_mixed_recurrence_benchmark(
        const bucket_array_set<recurrence_value_spec>& specs,
        std::int64_t total_iterations)
    {
        bucketed_comparison_result out{};
        out.easy.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.easy, total_iterations);
        out.easy.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.easy, total_iterations);
        out.medium.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.medium, total_iterations);
        out.medium.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.medium, total_iterations);
        out.hard.f128 = benchmark_mixed_recurrence_bucket<f128>(specs.hard, total_iterations);
        out.hard.mpfr = benchmark_mixed_recurrence_bucket<mpfr_ref>(specs.hard, total_iterations);
        out.average = combine_results(out.easy, out.medium, out.hard);
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

