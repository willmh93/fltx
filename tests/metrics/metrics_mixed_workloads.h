#ifndef FLTX_TESTS_METRICS_MIXED_WORKLOADS_INCLUDED
#define FLTX_TESTS_METRICS_MIXED_WORKLOADS_INCLUDED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fltx/f128_string.h>
#include <fltx/f256_string.h>

#include "metrics_f128_primary.h"
#include "metrics_f256_primary.h"

namespace bl::test::metrics::mixed_workloads
{
    constexpr domain_id mixed_domain{ "mixed", domain_role::stress };
    constexpr int mandelbrot_width = 32;
    constexpr int mandelbrot_height = 32;
    constexpr int mandelbrot_max_iter = 20000;

    template<class T>
    [[nodiscard]] T constant(double value)
    {
        return T{ value };
    }

    template<class T>
    [[nodiscard]] T parse_constant(const char* text)
    {
        return T{ text };
    }

    template<>
    [[nodiscard]] inline bl::f128 parse_constant<bl::f128>(const char* text)
    {
        return bl::f128{ bl::to_f128(text) };
    }

    template<>
    [[nodiscard]] inline bl::f256 parse_constant<bl::f256>(const char* text)
    {
        return bl::f256{ bl::to_f256(text) };
    }

    template<>
    [[nodiscard]] inline dd_real parse_constant<dd_real>(const char* text)
    {
        return dd_real::read(text);
    }

    template<>
    [[nodiscard]] inline qd_real parse_constant<qd_real>(const char* text)
    {
        return qd_real::read(text);
    }

    struct mixed_value_sample
    {
        std::string_view label;
        std::string text;
        value_sample value;
    };

    struct recurrence_sample
    {
        std::string_view label;
        mixed_value_sample x;
        mixed_value_sample y;
        mixed_value_sample a;
        mixed_value_sample b;
        mixed_value_sample c;
        mixed_value_sample d;
    };

    struct affine_sample
    {
        std::string_view label;
        mixed_value_sample x;
        mixed_value_sample y;
        mixed_value_sample angle;
        mixed_value_sample scale_x;
        mixed_value_sample scale_y;
        mixed_value_sample translate_x;
        mixed_value_sample translate_y;
    };

    enum class arithmetic_kernel
    {
        square_difference,
        product_sum,
        scaled_product_sum,
        shifted_sum,
        three_product_sum,
        scalar_mix,
        fused_expression
    };

    template<class Float>
    struct mixed_profile;

    template<>
    struct mixed_profile<bl::f128>
    {
        using primary = f128_primary::references;
        using fltx_type = f128_primary::fltx_type;
        using perfect_ref = f128_primary::perfect_ref;
        using competitor_ref = f128_primary::competitor_ref;
        using extra_competitor_ref = f128_primary::extra_competitor_ref;

        static constexpr precision_type precision = precision_type::f128;
        static constexpr std::string_view competitor_name = f128_primary::references::competitor_name;
        static constexpr std::string_view extra_competitor_name = f128_primary::references::extra_competitor_name;
        static constexpr double domain_target_bits = f128_primary::domain_ideal_bits;

        [[nodiscard]] static mixed_value_sample parse_value(std::string_view label, std::string text)
        {
            const fltx_type value{ bl::to_f128(text.c_str()) };
            return { label, std::move(text), f128_primary::make_runtime_value(label, value) };
        }

        [[nodiscard]] static mixed_value_sample parse_value(std::string_view label, const char* text)
        {
            return parse_value(label, std::string{ text });
        }

        template<class T>
        [[nodiscard]] static T make_value(const mixed_value_sample& value)
        {
            if constexpr (std::is_same_v<T, fltx_type>)
                return f128_primary::make_fltx(value.value);
            else if constexpr (std::is_same_v<T, competitor_ref>)
                return parse_constant<T>(value.text.c_str());
            else if constexpr (std::is_same_v<T, extra_competitor_ref>)
                return parse_constant<T>(value.text.c_str());
            else
                return parse_constant<T>(value.text.c_str());
        }

        [[nodiscard]] static perfect_ref to_perfect(const fltx_type& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const competitor_ref& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const extra_competitor_ref& value) { return f128_primary::to_perfect(value); }
        [[nodiscard]] static const perfect_ref& to_perfect(const perfect_ref& value) { return value; }

        [[nodiscard]] static double matching_bits(const perfect_ref& actual, const perfect_ref& expected)
        {
            return f128_primary::matching_bits(actual, expected);
        }

        [[nodiscard]] static double finite_for_mean(double bits) noexcept { return f128_primary::finite_for_mean(bits); }
        [[nodiscard]] static std::size_t benchmark_repetitions(std::size_t sample_count) noexcept
        {
            return f128_primary::benchmark_repetitions(sample_count);
        }

        static void consume(const fltx_type& value) noexcept { f128_primary::consume_benchmark_value(value); }
        static void consume(const competitor_ref& value) { f128_primary::consume_benchmark_value(value); }
        static void consume(const extra_competitor_ref& value) { f128_primary::consume_benchmark_value(value); }
    };

    template<>
    struct mixed_profile<bl::f256>
    {
        using primary = f256_primary::references;
        using fltx_type = f256_primary::fltx_type;
        using perfect_ref = f256_primary::perfect_ref;
        using competitor_ref = f256_primary::competitor_ref;
        using extra_competitor_ref = f256_primary::extra_competitor_ref;

        static constexpr precision_type precision = precision_type::f256;
        static constexpr std::string_view competitor_name = f256_primary::references::competitor_name;
        static constexpr std::string_view extra_competitor_name = f256_primary::references::extra_competitor_name;
        static constexpr double domain_target_bits = f256_primary::domain_ideal_bits;

        [[nodiscard]] static mixed_value_sample parse_value(std::string_view label, std::string text)
        {
            const fltx_type value{ bl::to_f256(text.c_str()) };
            return { label, std::move(text), f256_primary::make_runtime_value(label, value) };
        }

        [[nodiscard]] static mixed_value_sample parse_value(std::string_view label, const char* text)
        {
            return parse_value(label, std::string{ text });
        }

        template<class T>
        [[nodiscard]] static T make_value(const mixed_value_sample& value)
        {
            if constexpr (std::is_same_v<T, fltx_type>)
                return f256_primary::make_fltx(value.value);
            else if constexpr (std::is_same_v<T, competitor_ref>)
                return parse_constant<T>(value.text.c_str());
            else if constexpr (std::is_same_v<T, extra_competitor_ref>)
                return parse_constant<T>(value.text.c_str());
            else
                return parse_constant<T>(value.text.c_str());
        }

        [[nodiscard]] static perfect_ref to_perfect(const fltx_type& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const competitor_ref& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static perfect_ref to_perfect(const extra_competitor_ref& value) { return f256_primary::to_perfect(value); }
        [[nodiscard]] static const perfect_ref& to_perfect(const perfect_ref& value) { return value; }

        [[nodiscard]] static double matching_bits(const perfect_ref& actual, const perfect_ref& expected)
        {
            return f256_primary::matching_bits(actual, expected);
        }

        [[nodiscard]] static double finite_for_mean(double bits) noexcept { return f256_primary::finite_for_mean(bits); }
        [[nodiscard]] static std::size_t benchmark_repetitions(std::size_t sample_count) noexcept
        {
            return f256_primary::benchmark_repetitions(sample_count);
        }

        static void consume(const fltx_type& value) noexcept { f256_primary::consume_benchmark_value(value); }
        static void consume(const competitor_ref& value) { f256_primary::consume_benchmark_value(value); }
        static void consume(const extra_competitor_ref& value) { f256_primary::consume_benchmark_value(value); }
    };

    template<class T>
    [[nodiscard]] T mixed_sin(const T& x)
    {
        using std::sin;
        return sin(x);
    }

    template<class T>
    [[nodiscard]] T mixed_cos(const T& x)
    {
        using std::cos;
        return cos(x);
    }

    template<class T>
    [[nodiscard]] std::pair<T, T> mixed_sincos(const T& x)
    {
        return { mixed_sin(x), mixed_cos(x) };
    }

    [[nodiscard]] inline std::pair<bl::f128, bl::f128> mixed_sincos(const bl::f128& x)
    {
        bl::f128 s{};
        bl::f128 c{};
        if (bl::sincos(x, s, c))
            return { s, c };
        return { bl::sin(x), bl::cos(x) };
    }

    [[nodiscard]] inline std::pair<bl::f256, bl::f256> mixed_sincos(const bl::f256& x)
    {
        bl::f256 s{};
        bl::f256 c{};
        if (bl::sincos(x, s, c))
            return { s, c };
        return { bl::sin(x), bl::cos(x) };
    }

    [[nodiscard]] inline std::pair<dd_real, dd_real> mixed_sincos(const dd_real& x)
    {
        dd_real s{};
        dd_real c{};
        ::sincos(x, s, c);
        return { s, c };
    }

    [[nodiscard]] inline std::pair<qd_real, qd_real> mixed_sincos(const qd_real& x)
    {
        qd_real s{};
        qd_real c{};
        ::sincos(x, s, c);
        return { s, c };
    }

    template<class T>
    [[nodiscard]] T blend_result(const T& value, const T& acc)
    {
        return value + acc * constant<T>(0.25);
    }

    template<class T>
    [[nodiscard]] T mixed_sqr(const T& value)
    {
        return value * value;
    }

    template<>
    [[nodiscard]] inline bl::f128 mixed_sqr<bl::f128>(const bl::f128& value)
    {
        return bl::detail::_f128::sqr_dd_inline(value);
    }

    template<>
    [[nodiscard]] inline dd_real mixed_sqr<dd_real>(const dd_real& value)
    {
        return sqr(value);
    }

    template<>
    [[nodiscard]] inline qd_real mixed_sqr<qd_real>(const qd_real& value)
    {
        return sqr(value);
    }

    template<class T>
    [[nodiscard]] T mixed_twice(const T& value)
    {
        return value + value;
    }

    template<>
    [[nodiscard]] inline bl::f128 mixed_twice<bl::f128>(const bl::f128& value)
    {
        return bl::detail::_f128::mul_pwr2_inline(value, 2.0);
    }

    template<>
    [[nodiscard]] inline dd_real mixed_twice<dd_real>(const dd_real& value)
    {
        return mul_pwr2(value, 2.0);
    }

    template<>
    [[nodiscard]] inline qd_real mixed_twice<qd_real>(const qd_real& value)
    {
        return mul_pwr2(value, 2.0);
    }

    template<class T>
    [[nodiscard]] bool mixed_mandelbrot_escaped(const T& x2, const T& y2)
    {
        return x2 + y2 > constant<T>(4.0);
    }

    template<>
    [[nodiscard]] inline bool mixed_mandelbrot_escaped<bl::f128>(const bl::f128& x2, const bl::f128& y2)
    {
        constexpr double bailout = 4.0;
        constexpr double bailout_margin = 0x1.0p-48;

        const double hi_sum = x2.hi + y2.hi;
        if (hi_sum > bailout + bailout_margin)
            return true;
        if (hi_sum < bailout - bailout_margin)
            return false;

        double s1{}, s2{};
        bl::detail::fp::two_sum_precise(x2.hi, y2.hi, s1, s2);

        double t1{}, t2{};
        bl::detail::fp::two_sum_precise(x2.lo, y2.lo, t1, t2);

        s2 += t1;
        bl::detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        s2 += t2;
        bl::detail::fp::quick_two_sum_precise(s1, s2, s1, s2);
        return s1 > bailout || (s1 == bailout && s2 > 0.0);
    }

    template<class T>
    struct recurrence_value
    {
        T x{};
        T y{};
        T a{};
        T b{};
        T c{};
        T d{};
    };

    template<class T>
    struct affine_value
    {
        T x{};
        T y{};
        T angle{};
        T scale_x{};
        T scale_y{};
        T translate_x{};
        T translate_y{};
    };

    class mixed_rng
    {
    public:
        constexpr explicit mixed_rng(std::uint64_t seed) noexcept
            : state(seed)
        {
        }

        [[nodiscard]] std::uint64_t next() noexcept
        {
            state += 0x9e3779b97f4a7c15ull;
            std::uint64_t value = state;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
            return value ^ (value >> 31);
        }

        [[nodiscard]] double unit() noexcept
        {
            return static_cast<double>(next() >> 11) * 0x1.0p-53;
        }

        [[nodiscard]] double uniform(double min, double max) noexcept
        {
            return min + (max - min) * unit();
        }

        [[nodiscard]] int integer(int min, int max) noexcept
        {
            return min + static_cast<int>(next() % static_cast<std::uint64_t>(max - min + 1));
        }

        [[nodiscard]] double sign() noexcept
        {
            return (next() & 1ull) == 0ull ? -1.0 : 1.0;
        }

    private:
        std::uint64_t state;
    };

    [[nodiscard]] inline std::string format_mixed_value(double value)
    {
        std::ostringstream out;
        out << std::scientific << std::setprecision(36) << value;
        return out.str();
    }

    template<class Profile>
    [[nodiscard]] mixed_value_sample make_generated_value(std::string_view label, double value)
    {
        return Profile::parse_value(label, format_mixed_value(value));
    }

    template<class Profile>
    [[nodiscard]] recurrence_sample make_recurrence_sample(
        std::string_view label,
        const std::array<const char*, 6>& values)
    {
        return {
            label,
            Profile::parse_value("x", values[0]),
            Profile::parse_value("y", values[1]),
            Profile::parse_value("a", values[2]),
            Profile::parse_value("b", values[3]),
            Profile::parse_value("c", values[4]),
            Profile::parse_value("d", values[5])
        };
    }

    template<class Profile>
    [[nodiscard]] affine_sample make_affine_sample(
        std::string_view label,
        const std::array<const char*, 7>& values)
    {
        return {
            label,
            Profile::parse_value("x", values[0]),
            Profile::parse_value("y", values[1]),
            Profile::parse_value("angle", values[2]),
            Profile::parse_value("scale_x", values[3]),
            Profile::parse_value("scale_y", values[4]),
            Profile::parse_value("translate_x", values[5]),
            Profile::parse_value("translate_y", values[6])
        };
    }

    template<class Profile>
    [[nodiscard]] std::vector<recurrence_sample> make_recurrence_samples()
    {
        static constexpr std::array<std::array<const char*, 6>, 16> source{{
            {{ "-1.1250000000000000000000000000000001", "0.3125000000000000000000000000000001", "0.8750000000000000000000000000000001", "-0.4062500000000000000000000000000001", "1.3125000000000000000000000000000001", "0.6875000000000000000000000000000001" }},
            {{ "0.6875000000000000000000000000000001", "-0.9375000000000000000000000000000001", "-0.5625000000000000000000000000000001", "0.7812500000000000000000000000000001", "0.8125000000000000000000000000000001", "1.1875000000000000000000000000000001" }},
            {{ "-0.3437500000000000000000000000000001", "0.8437500000000000000000000000000001", "0.4687500000000000000000000000000001", "0.6562500000000000000000000000000001", "1.5625000000000000000000000000000001", "0.9062500000000000000000000000000001" }},
            {{ "0.1562500000000000000000000000000001", "-0.7187500000000000000000000000000001", "-0.8437500000000000000000000000000001", "0.3437500000000000000000000000000001", "0.7187500000000000000000000000000001", "1.4062500000000000000000000000000001" }},
            {{ "1.0312500000000000000000000000000001", "0.4062500000000000000000000000000001", "0.5937500000000000000000000000000001", "-0.6875000000000000000000000000000001", "1.2187500000000000000000000000000001", "0.7812500000000000000000000000000001" }},
            {{ "-0.6562500000000000000000000000000001", "-0.1875000000000000000000000000000001", "-0.3125000000000000000000000000000001", "0.7187500000000000000000000000000001", "0.9062500000000000000000000000000001", "1.6250000000000000000000000000000001" }},
            {{ "0.7812500000000000000000000000000001", "-1.0312500000000000000000000000000001", "0.6875000000000000000000000000000001", "0.2187500000000000000000000000000001", "1.4375000000000000000000000000000001", "0.5937500000000000000000000000000001" }},
            {{ "-0.2187500000000000000000000000000001", "0.9375000000000000000000000000000001", "-0.4687500000000000000000000000000001", "-0.8125000000000000000000000000000001", "0.5937500000000000000000000000000001", "1.3125000000000000000000000000000001" }},
            {{ "-1.3333333333333333333333333333333333", "0.4142135623730950488016887242096981", "0.7071067811865475244008443621048490", "-0.5773502691896257645091487805019575", "1.6180339887498948482045868343656381", "0.6180339887498948482045868343656381" }},
            {{ "0.5773502691896257645091487805019575", "-1.2247448713915890490986420373529457", "-0.4142135623730950488016887242096981", "0.8660254037844386467637231707529362", "0.7071067811865475244008443621048490", "1.4142135623730950488016887242096981" }},
            {{ "-0.3819660112501051517954131656343619", "1.1180339887498948482045868343656381", "0.6180339887498948482045868343656381", "0.7320508075688772935274463415058724", "1.7320508075688772935274463415058724", "0.8164965809277260327324280249019638" }},
            {{ "0.2679491924311227064725536584941276", "-0.8164965809277260327324280249019638", "-0.8660254037844386467637231707529362", "0.3819660112501051517954131656343619", "0.7639320225002103035908263312687238", "1.6180339887498948482045868343656381" }},
            {{ "1.1547005383792515290182975610039150", "0.3660254037844386467637231707529362", "0.8164965809277260327324280249019638", "-0.7071067811865475244008443621048490", "1.3333333333333333333333333333333333", "0.6666666666666666666666666666666667" }},
            {{ "-0.7639320225002103035908263312687238", "-0.2360679774997896964091736687312762", "-0.3333333333333333333333333333333333", "0.7320508075688772935274463415058724", "0.8660254037844386467637231707529362", "1.7320508075688772935274463415058724" }},
            {{ "0.8660254037844386467637231707529362", "-1.1180339887498948482045868343656381", "0.7639320225002103035908263312687238", "0.2679491924311227064725536584941276", "1.5", "0.5773502691896257645091487805019575" }},
            {{ "-0.2679491924311227064725536584941276", "1.2247448713915890490986420373529457", "-0.6180339887498948482045868343656381", "-0.7639320225002103035908263312687238", "0.6666666666666666666666666666666667", "1.5" }}
        }};

        std::vector<recurrence_sample> samples;
        samples.reserve(source.size() + 48);
        for (std::size_t index = 0; index < source.size(); ++index)
            samples.push_back(make_recurrence_sample<Profile>("mixed", source[index]));

        mixed_rng rng{ 0x6d697865645f6175ull };
        for (std::size_t index = 0; index < 48; ++index)
        {
            const auto signed_value = [&]() {
                if ((rng.next() & 3ull) == 0ull)
                    return rng.sign() * std::ldexp(1.0 + rng.unit(), rng.integer(-8, 4));
                return rng.uniform(-2.5, 2.5);
            };

            const double c = (index % 3) == 0 ? rng.uniform(0.375, 3.0) : rng.uniform(0.5, 1.75);
            const double d = (index % 3) == 1 ? rng.uniform(0.375, 3.0) : rng.uniform(0.5, 1.75);
            samples.push_back({
                "generated",
                make_generated_value<Profile>("x", signed_value()),
                make_generated_value<Profile>("y", signed_value()),
                make_generated_value<Profile>("a", rng.uniform(-1.5, 1.5)),
                make_generated_value<Profile>("b", rng.uniform(-1.5, 1.5)),
                make_generated_value<Profile>("c", c),
                make_generated_value<Profile>("d", d)
            });
        }
        return samples;
    }

    template<class Profile>
    [[nodiscard]] std::vector<affine_sample> make_affine_samples()
    {
        static constexpr std::array<std::array<const char*, 7>, 16> source{{
            {{ "-1.1250000000000000000000000000000001", "0.3125000000000000000000000000000001", "-2.9137461023847500000000000000000000", "0.8750000000000000000000000000000001", "1.1250000000000000000000000000000001", "0.2500000000000000000000000000000001", "-0.1250000000000000000000000000000001" }},
            {{ "0.6875000000000000000000000000000001", "-0.9375000000000000000000000000000001", "-2.1845036821094100000000000000000000", "1.0312500000000000000000000000000001", "0.7812500000000000000000000000000001", "-0.3750000000000000000000000000000001", "0.5000000000000000000000000000000001" }},
            {{ "-0.3437500000000000000000000000000001", "0.8437500000000000000000000000000001", "-1.0362719548392000000000000000000000", "0.7187500000000000000000000000000001", "1.3125000000000000000000000000000001", "0.1250000000000000000000000000000001", "0.3750000000000000000000000000000001" }},
            {{ "0.1562500000000000000000000000000001", "-0.7187500000000000000000000000000001", "-0.4178293036751200000000000000000000", "1.4062500000000000000000000000000001", "0.9062500000000000000000000000000001", "-0.5000000000000000000000000000000001", "-0.2500000000000000000000000000000001" }},
            {{ "1.0312500000000000000000000000000001", "0.4062500000000000000000000000000001", "0.4637912285314400000000000000000000", "0.8125000000000000000000000000000001", "1.2187500000000000000000000000000001", "0.3125000000000000000000000000000001", "-0.4375000000000000000000000000000001" }},
            {{ "-0.6562500000000000000000000000000001", "-0.1875000000000000000000000000000001", "1.3379281740185600000000000000000000", "1.1875000000000000000000000000000001", "0.6875000000000000000000000000000001", "-0.1875000000000000000000000000000001", "0.2500000000000000000000000000000001" }},
            {{ "0.7812500000000000000000000000000001", "-1.0312500000000000000000000000000001", "2.1187636509482200000000000000000000", "0.9687500000000000000000000000000001", "1.3437500000000000000000000000000001", "0.0625000000000000000000000000000001", "-0.3125000000000000000000000000000001" }},
            {{ "-0.2187500000000000000000000000000001", "0.9375000000000000000000000000000001", "3.0824510067120300000000000000000000", "1.0937500000000000000000000000000001", "0.8437500000000000000000000000000001", "-0.2500000000000000000000000000000001", "0.1875000000000000000000000000000001" }},
            {{ "-4.1250000000000000000000000000000001", "2.3125000000000000000000000000000001", "-11.7849263152274000000000000000000000", "0.7812500000000000000000000000000001", "1.4375000000000000000000000000000001", "1.2500000000000000000000000000000001", "-1.1250000000000000000000000000000001" }},
            {{ "3.6875000000000000000000000000000001", "-3.9375000000000000000000000000000001", "-8.6027134903187000000000000000000000", "1.5312500000000000000000000000000001", "0.6562500000000000000000000000000001", "-1.3750000000000000000000000000000001", "1.5000000000000000000000000000000001" }},
            {{ "-2.3437500000000000000000000000000001", "3.8437500000000000000000000000000001", "-5.7314081860192000000000000000000000", "0.5937500000000000000000000000000001", "1.6250000000000000000000000000000001", "1.1250000000000000000000000000000001", "1.3750000000000000000000000000000001" }},
            {{ "2.1562500000000000000000000000000001", "-2.7187500000000000000000000000000001", "-4.2861034750118000000000000000000000", "1.6875000000000000000000000000000001", "0.5625000000000000000000000000000001", "-1.5000000000000000000000000000000001", "-1.2500000000000000000000000000000001" }},
            {{ "4.0312500000000000000000000000000001", "1.4062500000000000000000000000000001", "4.2349172307716000000000000000000000", "0.6875000000000000000000000000000001", "1.5625000000000000000000000000000001", "1.3125000000000000000000000000000001", "-1.4375000000000000000000000000000001" }},
            {{ "-3.6562500000000000000000000000000001", "-1.1875000000000000000000000000000001", "5.9472684042019000000000000000000000", "1.4375000000000000000000000000000001", "0.5937500000000000000000000000000001", "-1.1875000000000000000000000000000001", "1.2500000000000000000000000000000001" }},
            {{ "2.7812500000000000000000000000000001", "-4.0312500000000000000000000000000001", "8.5197046183375000000000000000000000", "0.8437500000000000000000000000000001", "1.6875000000000000000000000000000001", "1.0625000000000000000000000000000001", "-1.3125000000000000000000000000000001" }},
            {{ "-1.2187500000000000000000000000000001", "3.9375000000000000000000000000000001", "11.3582910427196000000000000000000000", "1.2187500000000000000000000000000001", "0.7187500000000000000000000000000001", "-1.2500000000000000000000000000000001", "1.1875000000000000000000000000000001" }}
        }};

        std::vector<affine_sample> samples;
        samples.reserve(source.size() + 48);
        for (std::size_t index = 0; index < source.size(); ++index)
            samples.push_back(make_affine_sample<Profile>("affine", source[index]));

        constexpr double pi = 3.14159265358979323846264338327950288;
        mixed_rng rng{ 0x616666696e655f6dull };
        for (std::size_t index = 0; index < 48; ++index)
        {
            const double coordinate_scale = (index % 4) == 0 ? 16.0 : 4.0;
            samples.push_back({
                "generated",
                make_generated_value<Profile>("x", rng.uniform(-coordinate_scale, coordinate_scale)),
                make_generated_value<Profile>("y", rng.uniform(-coordinate_scale, coordinate_scale)),
                make_generated_value<Profile>("angle", rng.uniform(-8.0 * pi, 8.0 * pi)),
                make_generated_value<Profile>("scale_x", rng.uniform(0.375, 1.875)),
                make_generated_value<Profile>("scale_y", rng.uniform(0.375, 1.875)),
                make_generated_value<Profile>("translate_x", rng.uniform(-2.0, 2.0)),
                make_generated_value<Profile>("translate_y", rng.uniform(-2.0, 2.0))
            });
        }
        return samples;
    }

    template<class Profile, class T>
    [[nodiscard]] std::vector<recurrence_value<T>> make_recurrence_inputs(const std::vector<recurrence_sample>& samples)
    {
        std::vector<recurrence_value<T>> values;
        values.reserve(samples.size());
        for (const recurrence_sample& sample : samples)
        {
            values.push_back({
                Profile::template make_value<T>(sample.x),
                Profile::template make_value<T>(sample.y),
                Profile::template make_value<T>(sample.a),
                Profile::template make_value<T>(sample.b),
                Profile::template make_value<T>(sample.c),
                Profile::template make_value<T>(sample.d)
            });
        }
        return values;
    }

    template<class Profile, class T>
    [[nodiscard]] std::vector<affine_value<T>> make_affine_inputs(const std::vector<affine_sample>& samples)
    {
        std::vector<affine_value<T>> values;
        values.reserve(samples.size());
        for (const affine_sample& sample : samples)
        {
            values.push_back({
                Profile::template make_value<T>(sample.x),
                Profile::template make_value<T>(sample.y),
                Profile::template make_value<T>(sample.angle),
                Profile::template make_value<T>(sample.scale_x),
                Profile::template make_value<T>(sample.scale_y),
                Profile::template make_value<T>(sample.translate_x),
                Profile::template make_value<T>(sample.translate_y)
            });
        }
        return values;
    }

    template<class Profile>
    struct recurrence_input_maker
    {
        template<class T>
        [[nodiscard]] auto operator()(const std::vector<recurrence_sample>& source) const
        {
            return make_recurrence_inputs<Profile, T>(source);
        }
    };

    template<class Profile>
    struct affine_input_maker
    {
        template<class T>
        [[nodiscard]] auto operator()(const std::vector<affine_sample>& source) const
        {
            return make_affine_inputs<Profile, T>(source);
        }
    };

    template<arithmetic_kernel Kernel, class T>
    [[nodiscard]] std::pair<T, T> run_arithmetic_workload(
        const std::vector<recurrence_value<T>>& values,
        std::size_t outer_loops)
    {
        auto state = values;
        T acc_x = state.front().x;
        T acc_y = state.front().y;

        constexpr std::array<double, 8> add_rhs{ 0.125, -0.1875, 0.3125, -0.4375, 0.5625, -0.6875, 0.8125, -0.9375 };
        constexpr std::array<double, 8> add_lhs{ -0.03125, 0.09375, -0.15625, 0.21875, -0.28125, 0.34375, -0.40625, 0.46875 };
        constexpr std::array<double, 8> mul_rhs{ 0.875, -1.125, 1.375, -0.625, 0.5625, -0.8125, 1.0625, -1.3125 };
        constexpr std::array<double, 8> mul_lhs{ -1.0625, 0.6875, -0.9375, 1.1875, -0.75, 1.5, -1.25, 0.8125 };
        constexpr std::array<double, 8> div_rhs{ 1.125, -1.375, 1.625, -1.875, 2.125, -2.375, 2.625, -2.875 };

        for (std::size_t outer = 0; outer < outer_loops; ++outer)
        {
            for (std::size_t i = 0; i < state.size(); ++i)
            {
                auto& item = state[i];
                const T x = item.x;
                const T y = item.y;
                const T a = item.a;
                const T b = item.b;
                const T c = item.c;
                const T d = item.d;

                if constexpr (Kernel == arithmetic_kernel::square_difference)
                {
                    item.x = ((x * x - y * y) + a) / (c + constant<T>(0.5));
                    item.y = ((x * y + x * y) + b) / (d + constant<T>(1.5));
                }
                else if constexpr (Kernel == arithmetic_kernel::product_sum)
                {
                    item.x = (((x * a) + (y * b)) + c) / (c + constant<T>(2.0));
                    item.y = (((x * b) - (a * c)) - d) / (d + constant<T>(2.5));
                }
                else if constexpr (Kernel == arithmetic_kernel::scaled_product_sum)
                {
                    item.x = (((x * constant<T>(1.125)) + (y * constant<T>(-0.625))) + a) / (c + constant<T>(1.75));
                    item.y = (b - (x * constant<T>(0.375))) / (d + constant<T>(2.25));
                }
                else if constexpr (Kernel == arithmetic_kernel::shifted_sum)
                {
                    item.x = ((x + a) + b) / ((c + d) + constant<T>(1.0));
                    item.y = ((y + c) - d) / ((a + b) + constant<T>(2.0));
                }
                else if constexpr (Kernel == arithmetic_kernel::three_product_sum)
                {
                    item.x = (((x * a) + (y * b)) + (c * d)) / ((c + d) + constant<T>(2.0));
                    item.y = (((x * b) + (y * a)) + ((c * d) + (a * b))) / (c + constant<T>(3.0));
                }
                else if constexpr (Kernel == arithmetic_kernel::scalar_mix)
                {
                    const std::size_t scalar_index = i % add_rhs.size();
                    const T scalar_rhs_add = item.x + constant<T>(add_rhs[scalar_index]);
                    const T scalar_lhs_add = constant<T>(add_lhs[scalar_index]) + item.y;
                    const T scalar_rhs_mul = scalar_rhs_add * constant<T>(mul_rhs[scalar_index]);
                    const T scalar_lhs_mul = constant<T>(mul_lhs[scalar_index]) * scalar_lhs_add;
                    const T scalar_div = scalar_rhs_mul / constant<T>(div_rhs[scalar_index]);

                    const T qd_add = scalar_div + item.a;
                    const T qd_sub = scalar_lhs_mul - item.b;
                    const T qd_mul = qd_add * qd_sub;
                    const T denominator_base = item.c * item.c + item.d * item.d;
                    const T denominator = denominator_base + constant<T>(1.0);

                    const T x_step = qd_mul / denominator / (denominator + constant<T>(2.0));
                    const T y_delta = qd_add - qd_sub;
                    const T y_denominator = denominator + item.c * constant<T>(0.5) + constant<T>(2.5);
                    item.x = x_step + constant<T>(0.125) * item.a;
                    item.y = y_delta / y_denominator - item.b * constant<T>(0.0625);
                }
                else
                {
                    const T sum_plus_value = ((item.x * item.y) + (item.a * item.b)) + item.c;
                    const T diff_minus_value = ((item.x * item.a) - (item.b * item.c)) - item.d;
                    item.x = sum_plus_value / (item.c * item.c + constant<T>(2.0));
                    item.y = diff_minus_value / (item.d * item.d + constant<T>(2.5));
                }

                acc_x = blend_result(item.x, acc_x);
                acc_y = blend_result(item.y, acc_y);
            }
        }

        return { acc_x, acc_y };
    }

    template<class T>
    [[nodiscard]] std::pair<T, T> run_affine_workload(
        const std::vector<affine_value<T>>& values,
        std::size_t outer_loops)
    {
        auto state = values;
        T acc_x = state.front().x;
        T acc_y = state.front().y;

        for (std::size_t outer = 0; outer < outer_loops; ++outer)
        {
            for (auto& item : state)
            {
                const auto [sin_angle, cos_angle] = mixed_sincos(item.angle);
                const T neg_sin_angle = -sin_angle;

                const T m00 = cos_angle * item.scale_x;
                const T m01 = neg_sin_angle * item.scale_y;
                const T m10 = sin_angle * item.scale_x;
                const T m11 = cos_angle * item.scale_y;

                const T next_x = ((m00 * item.x) + (m01 * item.y) + item.translate_x) / (item.scale_x + constant<T>(2.0));
                const T next_y = ((m10 * item.x) + (m11 * item.y) + item.translate_y) / (item.scale_y + constant<T>(2.0));

                item.x = next_x;
                item.y = next_y;

                acc_x = blend_result(item.x, acc_x);
                acc_y = blend_result(item.y, acc_y);
            }
        }

        return { acc_x, acc_y };
    }

    template<class Profile, class T>
    void consume_pair(const std::pair<T, T>& value)
    {
        Profile::consume(value.first);
        Profile::consume(value.second);
    }

    template<class Profile, class T, class WorkloadFn>
    [[nodiscard]] benchmark_result benchmark_pair_workload(
        const std::vector<T>& values,
        WorkloadFn workload)
    {
        const std::size_t repetitions = Profile::benchmark_repetitions(values.size());
        const auto start = std::chrono::steady_clock::now();
        const auto result = workload(values, repetitions);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        consume_pair<Profile>(result);

        const std::size_t iterations = repetitions * values.size();
        return {
            std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations),
            iterations
        };
    }

    template<class Profile, class Actual, class Expected>
    [[nodiscard]] accuracy_result pair_accuracy(
        const std::pair<Actual, Actual>& actual,
        const std::pair<Expected, Expected>& expected,
        std::size_t sample_count)
    {
        double x_bits = Profile::matching_bits(
            Profile::to_perfect(actual.first),
            Profile::to_perfect(expected.first));
        double y_bits = Profile::matching_bits(
            Profile::to_perfect(actual.second),
            Profile::to_perfect(expected.second));
        if (std::isnan(x_bits))
            x_bits = 0.0;
        if (std::isnan(y_bits))
            y_bits = 0.0;
        return {
            std::min(x_bits, y_bits),
            (Profile::finite_for_mean(x_bits) + Profile::finite_for_mean(y_bits)) * 0.5,
            sample_count,
            domain_score({
                domain_sample_score(x_bits, Profile::domain_target_bits),
                domain_sample_score(y_bits, Profile::domain_target_bits)
            })
        };
    }

    template<class Profile, class Sample, class MakeInputsFn, class WorkloadFn>
    [[nodiscard]] metrics_record run_pair_workload_case(
        std::string_view operation,
        const std::vector<Sample>& samples,
        MakeInputsFn make_inputs,
        WorkloadFn workload,
        bool include_benchmarks)
    {
        constexpr std::size_t accuracy_repetitions = 8;

        using fltx_type = typename Profile::fltx_type;
        using competitor_ref = typename Profile::competitor_ref;
        using extra_competitor_ref = typename Profile::extra_competitor_ref;
        using perfect_ref = typename Profile::perfect_ref;

        const auto fltx_inputs = make_inputs.template operator()<fltx_type>(samples);
        const auto competitor_inputs = make_inputs.template operator()<competitor_ref>(samples);
        const auto extra_inputs = make_inputs.template operator()<extra_competitor_ref>(samples);
        const auto perfect_inputs = make_inputs.template operator()<perfect_ref>(samples);

        const auto expected = workload(perfect_inputs, accuracy_repetitions);

        metrics_record record{};
        record.suite = { Profile::precision, operation_id{ operation, operation }, mixed_domain };
        record.competitor_name = Profile::competitor_name;
        record.fltx_accuracy =
            pair_accuracy<Profile>(workload(fltx_inputs, accuracy_repetitions), expected, samples.size());
        record.competitor_accuracy =
            pair_accuracy<Profile>(workload(competitor_inputs, accuracy_repetitions), expected, samples.size());
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = Profile::extra_competitor_name;
        extra_competitor.accuracy =
            pair_accuracy<Profile>(workload(extra_inputs, accuracy_repetitions), expected, samples.size());

        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_pair_workload<Profile>(fltx_inputs, workload);
            record.competitor_benchmark = benchmark_pair_workload<Profile>(competitor_inputs, workload);
            extra_competitor.benchmark = benchmark_pair_workload<Profile>(extra_inputs, workload);
        }

        return record;
    }

    [[nodiscard]] inline accuracy_result aggregate_accuracy(const std::vector<metrics_record>& records, auto selector)
    {
        double worst_bits = std::numeric_limits<double>::infinity();
        double total_mean = 0.0;
        double total_domain_score = 0.0;
        std::size_t sample_count = 0;

        for (const metrics_record& record : records)
        {
            const accuracy_result& accuracy = selector(record);
            worst_bits = std::min(worst_bits, accuracy.worst_bits);
            total_mean += accuracy.mean_bits * static_cast<double>(accuracy.sample_count);
            total_domain_score += accuracy.domain_score * static_cast<double>(accuracy.sample_count);
            sample_count += accuracy.sample_count;
        }

        return {
            worst_bits,
            sample_count == 0 ? 0.0 : total_mean / static_cast<double>(sample_count),
            sample_count,
            sample_count == 0 ? 0.0 : total_domain_score / static_cast<double>(sample_count)
        };
    }

    [[nodiscard]] inline benchmark_result aggregate_benchmark(const std::vector<metrics_record>& records, auto selector)
    {
        double total_ns = 0.0;
        std::size_t iteration_count = 0;
        for (const metrics_record& record : records)
        {
            const benchmark_result& benchmark = selector(record);
            total_ns += benchmark.ns_per_iter;
            iteration_count += benchmark.iteration_count;
        }

        return {
            records.empty() ? 0.0 : total_ns / static_cast<double>(records.size()),
            iteration_count
        };
    }

    template<class Profile>
    [[nodiscard]] metrics_record aggregate_mixed_arithmetic(const std::vector<metrics_record>& records)
    {
        metrics_record aggregate{};
        aggregate.suite = { Profile::precision, operation_id{ "mixed arithmetic", "mixed arithmetic" }, mixed_domain };
        aggregate.competitor_name = Profile::competitor_name;
        aggregate.fltx_accuracy = aggregate_accuracy(records, [](const metrics_record& record) -> const accuracy_result& { return record.fltx_accuracy; });
        aggregate.competitor_accuracy = aggregate_accuracy(records, [](const metrics_record& record) -> const accuracy_result& { return record.competitor_accuracy; });
        aggregate.fltx_benchmark = aggregate_benchmark(records, [](const metrics_record& record) -> const benchmark_result& { return record.fltx_benchmark; });
        aggregate.competitor_benchmark = aggregate_benchmark(records, [](const metrics_record& record) -> const benchmark_result& { return record.competitor_benchmark; });

        if (!records.empty() && !records.front().extra_competitors.empty())
        {
            auto& extra_competitor = aggregate.extra_competitors.emplace_back();
            extra_competitor.name = records.front().extra_competitors.front().name;
            extra_competitor.accuracy = aggregate_accuracy(
                records,
                [](const metrics_record& record) -> const accuracy_result& { return record.extra_competitors.front().accuracy; });
            extra_competitor.benchmark = aggregate_benchmark(
                records,
                [](const metrics_record& record) -> const benchmark_result& { return record.extra_competitors.front().benchmark; });
        }

        return aggregate;
    }

    template<arithmetic_kernel Kernel, class Profile>
    [[nodiscard]] metrics_record run_arithmetic_record(bool include_benchmarks)
    {
        const auto samples = make_recurrence_samples<Profile>();
        return run_pair_workload_case<Profile>(
            "mixed arithmetic detail",
            samples,
            recurrence_input_maker<Profile>{},
            [](const auto& values, std::size_t repetitions)
            {
                return run_arithmetic_workload<Kernel>(values, repetitions);
            },
            include_benchmarks);
    }

    template<class Profile>
    [[nodiscard]] metrics_record run_mixed_arithmetic_record(bool include_benchmarks)
    {
        std::vector<metrics_record> records;
        records.reserve(7);
        records.push_back(run_arithmetic_record<arithmetic_kernel::square_difference, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::product_sum, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::scaled_product_sum, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::shifted_sum, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::three_product_sum, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::scalar_mix, Profile>(include_benchmarks));
        records.push_back(run_arithmetic_record<arithmetic_kernel::fused_expression, Profile>(include_benchmarks));
        return aggregate_mixed_arithmetic<Profile>(records);
    }

    template<class Profile>
    [[nodiscard]] metrics_record run_affine_record(bool include_benchmarks)
    {
        const auto samples = make_affine_samples<Profile>();
        return run_pair_workload_case<Profile>(
            "affine transform",
            samples,
            affine_input_maker<Profile>{},
            [](const auto& values, std::size_t repetitions)
            {
                return run_affine_workload(values, repetitions);
            },
            include_benchmarks);
    }

    template<class T>
    struct mandelbrot_total
    {
        std::int64_t escape_iterations = 0;
        std::int64_t escaped_pixels = 0;
    };

    template<class T>
    struct mandelbrot_trace
    {
        mandelbrot_total<T> total{};
        std::array<int, mandelbrot_width * mandelbrot_height> iterations{};
    };

    template<class T>
    struct mandelbrot_constants
    {
        T center_x{};
        T center_y{};
        T zoom{};
    };

    struct mandelbrot_window
    {
        const char* center_x;
        const char* center_y;
        const char* zoom;
    };

    inline constexpr std::array<mandelbrot_window, 3> mandelbrot_windows{{
        { "-1.73200006480238126967529761198455", "0.00000019235376499049335337716270", "2.0e+28" },
        { "-0.74364388703715100000000000000000", "0.13182590420533000000000000000000", "1.0e+10" },
        { "-0.75000000000000000000000000000000", "0.10000000000000000000000000000000", "5.12e+2" }
    }};

    template<class Profile, class T>
    [[nodiscard]] mandelbrot_constants<T> make_mandelbrot_constants(const mandelbrot_window& window)
    {
        const mixed_value_sample center_x = Profile::parse_value("center_x", window.center_x);
        const mixed_value_sample center_y = Profile::parse_value("center_y", window.center_y);
        const mixed_value_sample zoom = Profile::parse_value("zoom", window.zoom);

        return {
            Profile::template make_value<T>(center_x),
            Profile::template make_value<T>(center_y),
            Profile::template make_value<T>(zoom)
        };
    }

    template<class Profile, class T>
    [[nodiscard]] std::vector<mandelbrot_constants<T>> make_mandelbrot_constant_set()
    {
        std::vector<mandelbrot_constants<T>> constants;
        constants.reserve(mandelbrot_windows.size());
        for (const mandelbrot_window& window : mandelbrot_windows)
            constants.push_back(make_mandelbrot_constants<Profile, T>(window));
        return constants;
    }

    struct ignore_mandelbrot_pixel
    {
        void operator()(int, int) const noexcept {}
    };

    template<class T, class RecordPixel>
    [[nodiscard]] mandelbrot_total<T> run_mandelbrot_kernel(
        const mandelbrot_constants<T>& constants,
        RecordPixel record_pixel)
    {
        const T& center_x = constants.center_x;
        const T& center_y = constants.center_y;
        const T& zoom = constants.zoom;

        const T width_value = constant<T>(static_cast<double>(mandelbrot_width));
        const T height_value = constant<T>(static_cast<double>(mandelbrot_height));
        const T scale_x = constant<T>(4.0) / (zoom * width_value);
        const T scale_y = constant<T>(4.0) / (zoom * height_value);
        const T half_w = width_value * constant<T>(0.5);
        const T half_h = height_value * constant<T>(0.5);
        mandelbrot_total<T> total{};
        for (int row = 0; row < mandelbrot_height; ++row)
        {
            const int py = mandelbrot_height - 1 - row;
            for (int px = 0; px < mandelbrot_width; ++px)
            {
                const T cx = center_x + (constant<T>(static_cast<double>(px)) - half_w) * scale_x;
                const T cy = center_y + (constant<T>(static_cast<double>(py)) - half_h) * scale_y;
                T x{};
                T y{};

                int iter = 0;
                while (iter < mandelbrot_max_iter)
                {
                    const T x2 = mixed_sqr(x);
                    const T y2 = mixed_sqr(y);
                    if (mixed_mandelbrot_escaped(x2, y2))
                        break;

                    const T xy = x * y;
                    const T xx = x2 - y2 + cx;
                    y = mixed_twice(xy) + cy;
                    x = xx;
                    ++iter;
                }

                total.escape_iterations += iter;
                if (iter != mandelbrot_max_iter)
                    ++total.escaped_pixels;
                record_pixel(row * mandelbrot_width + px, iter);
            }
        }
        return total;
    }

    template<class T>
    [[nodiscard]] mandelbrot_total<T> run_mandelbrot_kernel(const mandelbrot_constants<T>& constants)
    {
        return run_mandelbrot_kernel(constants, ignore_mandelbrot_pixel{});
    }

    template<class T>
    [[nodiscard]] mandelbrot_trace<T> run_mandelbrot_trace(const mandelbrot_constants<T>& constants)
    {
        mandelbrot_trace<T> trace{};
        trace.total = run_mandelbrot_kernel(
            constants,
            [&](int index, int iter) noexcept
            {
                trace.iterations[static_cast<std::size_t>(index)] = iter;
            });
        return trace;
    }

    template<class T>
    [[nodiscard]] std::vector<mandelbrot_trace<T>> run_mandelbrot_traces(
        const std::vector<mandelbrot_constants<T>>& constants)
    {
        std::vector<mandelbrot_trace<T>> traces;
        traces.reserve(constants.size());
        for (const mandelbrot_constants<T>& window : constants)
            traces.push_back(run_mandelbrot_trace(window));
        return traces;
    }

    inline volatile std::int64_t mandelbrot_sink = 0;

    template<class Profile, class T>
    [[nodiscard]] benchmark_result benchmark_mandelbrot()
    {
        const auto constants = make_mandelbrot_constant_set<Profile, T>();
        const auto start = std::chrono::steady_clock::now();
        mandelbrot_total<T> total{};
        for (const mandelbrot_constants<T>& window : constants)
        {
            const mandelbrot_total<T> window_total = run_mandelbrot_kernel<T>(window);
            total.escape_iterations += window_total.escape_iterations;
            total.escaped_pixels += window_total.escaped_pixels;
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        mandelbrot_sink += total.escape_iterations + total.escaped_pixels;

        return {
            std::chrono::duration<double, std::nano>(elapsed).count() /
                static_cast<double>(std::max<std::int64_t>(1, total.escape_iterations)),
            static_cast<std::size_t>(std::max<std::int64_t>(0, total.escape_iterations))
        };
    }

    template<class Profile, class ActualTrace, class ExpectedTrace>
    [[nodiscard]] accuracy_result mandelbrot_accuracy(
        const std::vector<ActualTrace>& actual,
        const std::vector<ExpectedTrace>& expected)
    {
        using perfect_ref = typename Profile::perfect_ref;

        double worst_bits = std::numeric_limits<double>::infinity();
        double total_bits = 0.0;
        std::vector<double> domain_scores;
        const std::size_t sample_count = expected.size() * mandelbrot_width * mandelbrot_height;
        domain_scores.reserve(sample_count);

        for (std::size_t window = 0; window < expected.size(); ++window)
        {
            for (std::size_t index = 0; index < mandelbrot_width * mandelbrot_height; ++index)
            {
                const int actual_iter = actual[window].iterations[index];
                const int expected_iter = expected[window].iterations[index];
                const double bits = actual_iter == expected_iter
                    ? std::numeric_limits<double>::infinity()
                    : Profile::matching_bits(perfect_ref{ actual_iter }, perfect_ref{ expected_iter });
                worst_bits = std::min(worst_bits, bits);
                total_bits += Profile::finite_for_mean(bits);
                domain_scores.push_back(domain_sample_score(bits, Profile::domain_target_bits));
            }
        }

        return {
            worst_bits,
            sample_count == 0 ? 0.0 : total_bits / static_cast<double>(sample_count),
            sample_count,
            domain_score(std::move(domain_scores))
        };
    }

    template<class Profile>
    [[nodiscard]] metrics_record run_mandelbrot_record(bool include_benchmarks)
    {
        using fltx_type = typename Profile::fltx_type;
        using competitor_ref = typename Profile::competitor_ref;
        using extra_competitor_ref = typename Profile::extra_competitor_ref;
        using perfect_ref = typename Profile::perfect_ref;

        const auto expected = run_mandelbrot_traces<perfect_ref>(
            make_mandelbrot_constant_set<Profile, perfect_ref>());
        const auto fltx = run_mandelbrot_traces<fltx_type>(
            make_mandelbrot_constant_set<Profile, fltx_type>());
        const auto competitor = run_mandelbrot_traces<competitor_ref>(
            make_mandelbrot_constant_set<Profile, competitor_ref>());
        const auto extra = run_mandelbrot_traces<extra_competitor_ref>(
            make_mandelbrot_constant_set<Profile, extra_competitor_ref>());

        metrics_record record{};
        record.suite = { Profile::precision, operation_id{ "mandelbrot", "mandelbrot" }, mixed_domain };
        record.competitor_name = Profile::competitor_name;
        record.fltx_accuracy = mandelbrot_accuracy<Profile>(fltx, expected);
        record.competitor_accuracy = mandelbrot_accuracy<Profile>(competitor, expected);
        auto& extra_competitor = record.extra_competitors.emplace_back();
        extra_competitor.name = Profile::extra_competitor_name;
        extra_competitor.accuracy = mandelbrot_accuracy<Profile>(extra, expected);

        if (include_benchmarks)
        {
            record.fltx_benchmark = benchmark_mandelbrot<Profile, fltx_type>();
            record.competitor_benchmark = benchmark_mandelbrot<Profile, competitor_ref>();
            extra_competitor.benchmark = benchmark_mandelbrot<Profile, extra_competitor_ref>();
        }

        return record;
    }

    template<class Profile>
    [[nodiscard]] std::vector<metrics_record> collect_records(bool include_benchmarks)
    {
        std::vector<metrics_record> records;
        records.reserve(3);
        records.push_back(run_mixed_arithmetic_record<Profile>(include_benchmarks));
        records.push_back(run_affine_record<Profile>(include_benchmarks));
        records.push_back(run_mandelbrot_record<Profile>(include_benchmarks));
        return records;
    }
}

#endif
