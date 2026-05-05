#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <fltx.h>
using namespace bl;
using namespace bl::literals;

template<typename T, std::size_t N>
struct derivative_rule
{
    std::array<T, N> offsets{};
    std::array<T, N> weights{};
};

template<typename T>
constexpr T abs_value(T x)
{
    return x < T{} ? -x : x;
}

template<typename T, std::size_t N>
consteval std::array<T, N> centered_offsets(T spacing)
{
    std::array<T, N> offsets{};
    constexpr int radius = static_cast<int>(N / 2);

    for (std::size_t i = 0; i < N; ++i)
        offsets[i] = T(static_cast<int>(i) - radius) * spacing;

    return offsets;
}

template<typename T, std::size_t N, int Derivative>
consteval derivative_rule<T, N> make_derivative_rule(std::array<T, N> offsets)
{
    std::array<std::array<T, Derivative + 1>, N> table{};

    T previous_product = T{ 1.0 };
    T previous_offset = offsets[0];
    table[0][0] = T{ 1.0 };

    for (std::size_t i = 1; i < N; ++i)
    {
        const int max_order = std::min<int>(static_cast<int>(i), Derivative);
        T current_product = T{ 1.0 };
        const T old_offset = previous_offset;
        previous_offset = offsets[i];

        for (std::size_t j = 0; j < i; ++j)
        {
            const T distance = offsets[i] - offsets[j];
            current_product *= distance;

            if (j == i - 1)
            {
                for (int order = max_order; order >= 1; --order)
                {
                    table[i][order] = previous_product
                        * (T(order) * table[i - 1][order - 1] - old_offset * table[i - 1][order])
                        / current_product;
                }

                table[i][0] = -previous_product * old_offset * table[i - 1][0] / current_product;
            }

            for (int order = max_order; order >= 1; --order)
                table[j][order] = (previous_offset * table[j][order] - T(order) * table[j][order - 1]) / distance;

            table[j][0] = previous_offset * table[j][0] / distance;
        }

        previous_product = current_product;
    }

    derivative_rule<T, N> rule{};
    rule.offsets = offsets;
    for (std::size_t i = 0; i < N; ++i)
        rule.weights[i] = table[i][Derivative];

    return rule;
}

template<typename Out, typename T, std::size_t N>
consteval derivative_rule<Out, N> cast_rule(const derivative_rule<T, N>& rule)
{
    derivative_rule<Out, N> out{};
    for (std::size_t i = 0; i < N; ++i)
    {
        out.offsets[i] = static_cast<Out>(rule.offsets[i]);
        out.weights[i] = static_cast<Out>(rule.weights[i]);
    }
    return out;
}

template<std::size_t N>
double estimate_exp_derivative_double(const derivative_rule<double, N>& rule)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        sum += rule.weights[i] * std::exp(rule.offsets[i]);
    return sum;
}

template<typename T, std::size_t N>
constexpr T estimate_exp_derivative(const derivative_rule<T, N>& rule)
{
    T sum{};
    for (std::size_t i = 0; i < N; ++i)
        sum += rule.weights[i] * bl::exp(rule.offsets[i]);
    return sum;
}

int main()
{
    constexpr std::size_t sample_count = 29;
    constexpr int derivative_order = 8;

    // High-order finite-difference weights are used in PDE solvers, simulation,
    // and signal processing. The coefficient solve is ill-conditioned, but the
    // stencil is fixed, so it is a good consteval precomputation target.
    constexpr auto offsets_f64 = centered_offsets<double, sample_count>(0.1);
    constexpr auto offsets_f256 = centered_offsets<f256, sample_count>(0.1_qd);

    constexpr auto rule_f64 = make_derivative_rule<double, sample_count, derivative_order>(offsets_f64);
    constexpr auto rule_f256 = make_derivative_rule<f256, sample_count, derivative_order>(offsets_f256);
    constexpr auto rule_f256_as_double = cast_rule<double>(rule_f256);

    const double f64_estimate = estimate_exp_derivative_double(rule_f64);
    const double f256_generated_double_estimate = estimate_exp_derivative_double(rule_f256_as_double);
    constexpr f256 f256_estimate = estimate_exp_derivative(rule_f256);

    // Every derivative of exp(x) at x=0 is exactly 1.
    constexpr double expected = 1.0;

    std::cout << "8th derivative of exp(x) at x=0\n";
    std::cout << "finite-difference stencil: " << sample_count << " samples, spacing 0.1\n\n";

    std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "double-generated double weights:\n";
    std::cout << "  estimate: " << f64_estimate << "\n";
    std::cout << "  error:    " << abs_value(f64_estimate - expected) << "\n\n";

    std::cout << "f256-generated double weights:\n";
    std::cout << "  estimate: " << f256_generated_double_estimate << "\n";
    std::cout << "  error:    " << abs_value(f256_generated_double_estimate - expected) << "\n\n";

    std::cout << std::setprecision(std::numeric_limits<f256>::digits10);
    std::cout << "f256-generated f256 weights:\n";
    std::cout << "  estimate: " << f256_estimate << "\n";
    std::cout << "  error:    " << abs_value(f256_estimate - f256{ 1.0 }) << "\n";
}
