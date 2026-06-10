#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fltx/f32_math.h>

using namespace bl;

namespace
{
    using mpfr_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<96>,
        boost::multiprecision::et_off>;
    using boost::multiprecision::cpp_int;

    constexpr int checked_digits  = std::numeric_limits<float>::digits10;
    constexpr int printed_digits  = std::numeric_limits<float>::max_digits10;
    constexpr float min_subnormal = std::numeric_limits<float>::denorm_min();

    constexpr std::uint64_t random_seed = 1ull;
    constexpr int random_sample_count   = 10;
    constexpr const char* type_label    = "f32";

    template<class T>
    struct sincos_pair
    {
        T c;
        T s;
    };

    struct accuracy_stats_entry
    {
        int samples = 0;
        int passed  = 0;
        std::vector<float> achieved_digits;
        std::uint32_t worst_ulp  = 0;
        float worst_scaled_error = 0.0f;
    };

    class accuracy_report_scope;
    thread_local accuracy_report_scope* current_accuracy_report_scope = nullptr;

    [[nodiscard]] std::string to_text(float value)
    {
        if (std::isnan(value))
            return "nan";
        if (std::isinf(value))
            return std::signbit(value) ? "-inf" : "inf";

        std::ostringstream out;
        out << std::setprecision(printed_digits) << std::defaultfloat << value;
        return out.str();
    }

    [[nodiscard]] std::string to_text_hex(float value)
    {
        std::ostringstream out;
        out << std::hexfloat << value;
        return out.str();
    }

    [[nodiscard]] std::string to_text(const mpfr_ref& value)
    {
        std::ostringstream out;
        out << std::setprecision(printed_digits + 24)
            << std::scientific
            << value;
        return out.str();
    }

    [[nodiscard]] bool op_is(const char* op_name, const char* expected)
    {
        return std::strcmp(op_name, expected) == 0;
    }

    [[nodiscard]] bool op_starts_with(const char* op_name, const char* prefix)
    {
        return std::strncmp(op_name, prefix, std::strlen(prefix)) == 0;
    }

    [[nodiscard]] float abs_ref(float value)
    {
        return value < 0.0 ? -value : value;
    }

    [[nodiscard]] mpfr_ref abs_ref(const mpfr_ref& value)
    {
        return value < 0 ? -value : value;
    }

    [[nodiscard]] mpfr_ref to_ref_exact(float value)
    {
        return mpfr_ref{ value };
    }

    [[nodiscard]] float round_ref_to_f32(const mpfr_ref& value)
    {
        return value.convert_to<float>();
    }

    [[nodiscard]] mpfr_ref ref_floor(const mpfr_ref& value)
    {
        return boost::multiprecision::floor(value);
    }

    [[nodiscard]] mpfr_ref ref_ceil(const mpfr_ref& value)
    {
        return boost::multiprecision::ceil(value);
    }

    [[nodiscard]] mpfr_ref ref_trunc(const mpfr_ref& value)
    {
        return value < 0 ? ref_ceil(value) : ref_floor(value);
    }

    [[nodiscard]] mpfr_ref ref_fmod(const mpfr_ref& x, const mpfr_ref& y)
    {
        return x - ref_trunc(x / y) * y;
    }

    [[nodiscard]] mpfr_ref ref_round_half_away_zero(const mpfr_ref& value)
    {
        return value < 0
            ? ref_ceil(value - mpfr_ref{ "0.5" })
            : ref_floor(value + mpfr_ref{ "0.5" });
    }

    [[nodiscard]] mpfr_ref ref_round_to_even(const mpfr_ref& value)
    {
        mpfr_ref rounded = ref_floor(value + mpfr_ref{ "0.5" });
        if ((rounded - value) == mpfr_ref{ "0.5" } && ref_fmod(rounded, mpfr_ref{ 2 }) != mpfr_ref{ 0 })
            rounded -= 1;
        return rounded;
    }

    [[nodiscard]] mpfr_ref ref_remainder(const mpfr_ref& x, const mpfr_ref& y)
    {
        return x - ref_round_to_even(x / y) * y;
    }

    struct exact_binary_float
    {
        cpp_int coeff;
        int exp2 = 0;
        bool neg = false;
    };

    struct exact_mod_reference
    {
        float fmod        = 0.0f;
        float remainder   = 0.0f;
        int quotient_bits = 0;
    };

    [[nodiscard]] exact_binary_float decompose_exact(float value)
    {
        const std::uint32_t bits     = std::bit_cast<std::uint32_t>(value);
        const std::uint32_t frac     = bits & ((std::uint32_t{ 1 } << 23) - 1u);
        const std::uint32_t exp_bits = (bits >> 23) & 0xffu;

        exact_binary_float out;
        out.neg = (bits >> 31) != 0;
        if (exp_bits == 0)
        {
            out.coeff = frac;
            out.exp2 = -149;
        }
        else
        {
            out.coeff = (std::uint32_t{ 1 } << 23) | frac;
            out.exp2 = static_cast<int>(exp_bits) - 127 - 23;
        }
        return out;
    }

    [[nodiscard]] int bit_length(const cpp_int& value)
    {
        return value == 0 ? 0 : static_cast<int>(boost::multiprecision::msb(value)) + 1;
    }

    [[nodiscard]] cpp_int rounded_shift_right(cpp_int value, int bits)
    {
        if (bits <= 0)
            return value;

        const bool round_bit = ((value >> (bits - 1)) & 1) != 0;
        const bool sticky    = bits > 1 && (value & ((cpp_int{ 1 } << (bits - 1)) - 1)) != 0;
        cpp_int out = value >> bits;
        if (round_bit && (sticky || (out & 1) != 0))
            ++out;
        return out;
    }

    [[nodiscard]] float round_exact_dyadic_to_f32(cpp_int coeff, int exp2, bool neg)
    {
        if (coeff == 0)
            return neg ? -0.0f : 0.0f;

        const int top_bit = bit_length(coeff) - 1;
        int unbiased_exp  = exp2 + top_bit;
        if (unbiased_exp > 127)
            return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

        if (unbiased_exp < -126)
        {
            const int scale = exp2 + 149;
            cpp_int units = scale >= 0 ? (coeff << scale) : rounded_shift_right(coeff, -scale);
            if (units == 0)
                return neg ? -0.0f : 0.0f;
            const float out = std::ldexp(units.convert_to<float>(), -149);
            return neg ? -out : out;
        }

        const int shift = top_bit - 23;
        cpp_int significand = shift > 0 ? rounded_shift_right(coeff, shift) : (coeff << -shift);
        int result_exp = exp2 + shift;

        if (bit_length(significand) > 24)
        {
            significand >>= 1;
            ++result_exp;
            ++unbiased_exp;
            if (unbiased_exp > 127)
                return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }

        const float out = std::ldexp(significand.convert_to<float>(), result_exp);
        return neg ? -out : out;
    }

    [[nodiscard]] int normalize_remquo_bits(const cpp_int& quotient, bool neg)
    {
        if (quotient == 0)
            return 0;

        int bits = static_cast<int>((quotient & 0x7).convert_to<unsigned>());
        if (bits == 0)
            bits = 8;
        return neg ? -bits : bits;
    }

    [[nodiscard]] exact_mod_reference exact_mod_reference_for(float lhs, float rhs)
    {
        exact_mod_reference out;
        if (std::isnan(lhs) || std::isnan(rhs) || rhs == 0.0f || std::isinf(lhs))
        {
            out.fmod = std::numeric_limits<float>::quiet_NaN();
            out.remainder = std::numeric_limits<float>::quiet_NaN();
            return out;
        }
        if (std::isinf(rhs) || lhs == 0.0f)
        {
            out.fmod = lhs;
            out.remainder = lhs;
            return out;
        }

        exact_binary_float x = decompose_exact(std::fabs(lhs));
        exact_binary_float y = decompose_exact(std::fabs(rhs));
        const int common_exp = std::min(x.exp2, y.exp2);

        cpp_int numerator = x.coeff << (x.exp2 - common_exp);
        cpp_int denominator = y.coeff << (y.exp2 - common_exp);
        cpp_int quotient = numerator / denominator;
        cpp_int remainder = numerator % denominator;

        const bool x_neg = std::signbit(lhs);
        out.fmod = round_exact_dyadic_to_f32(remainder, common_exp, x_neg);

        bool remainder_neg = x_neg;
        cpp_int nearest_quotient = quotient;
        cpp_int nearest_remainder = remainder;
        const cpp_int twice_remainder = remainder << 1;
        if (twice_remainder > denominator || (twice_remainder == denominator && (quotient & 1) != 0))
        {
            nearest_remainder = denominator - remainder;
            ++nearest_quotient;
            remainder_neg = !remainder_neg;
        }

        out.remainder = round_exact_dyadic_to_f32(nearest_remainder, common_exp, remainder_neg);
        if (out.remainder == 0.0f)
            out.remainder = std::copysign(0.0f, lhs);
        out.quotient_bits = normalize_remquo_bits(nearest_quotient, std::signbit(lhs) != std::signbit(rhs));
        return out;
    }

    [[nodiscard]] const mpfr_ref& ln2_ref()
    {
        static const mpfr_ref value = boost::multiprecision::log(mpfr_ref{ 2 });
        return value;
    }

    [[nodiscard]] const mpfr_ref& ln10_ref()
    {
        static const mpfr_ref value = boost::multiprecision::log(mpfr_ref{ 10 });
        return value;
    }

    [[nodiscard]] mpfr_ref ref_exp2(const mpfr_ref& value)
    {
        return boost::multiprecision::exp(value * ln2_ref());
    }

    [[nodiscard]] mpfr_ref ref_log2(const mpfr_ref& value)
    {
        return boost::multiprecision::log(value) / ln2_ref();
    }

    [[nodiscard]] mpfr_ref ref_log10(const mpfr_ref& value)
    {
        return boost::multiprecision::log(value) / ln10_ref();
    }

    [[nodiscard]] bool ref_is_integer(const mpfr_ref& value)
    {
        return boost::multiprecision::floor(value) == value;
    }

    [[nodiscard]] mpfr_ref ref_powi(mpfr_ref base, long long exponent)
    {
        if (exponent == 0)
            return mpfr_ref{ 1 };

        bool invert = exponent < 0;
        unsigned long long e = invert
            ? static_cast<unsigned long long>(-(exponent + 1)) + 1ull
            : static_cast<unsigned long long>(exponent);

        mpfr_ref result{ 1 };
        while (e != 0)
        {
            if ((e & 1ull) != 0)
                result *= base;
            e >>= 1ull;
            if (e != 0)
                base *= base;
        }

        return invert ? (mpfr_ref{ 1 } / result) : result;
    }

    [[nodiscard]] mpfr_ref ref_pow(const mpfr_ref& base, const mpfr_ref& exponent)
    {
        if (base < 0 && ref_is_integer(exponent))
        {
            const long long n  = exponent.convert_to<long long>();
            const mpfr_ref mag = ref_powi(-base, n);
            return (n & 1LL) ? -mag : mag;
        }

        return boost::multiprecision::exp(exponent * boost::multiprecision::log(base));
    }

    [[nodiscard]] mpfr_ref ref_cbrt(const mpfr_ref& value)
    {
        const mpfr_ref third = mpfr_ref{ 1 } / mpfr_ref{ 3 };
        return value < 0
            ? -boost::multiprecision::pow(-value, third)
            : boost::multiprecision::pow(value, third);
    }

    [[nodiscard]] mpfr_ref ref_hypot(const mpfr_ref& x, const mpfr_ref& y)
    {
        return boost::multiprecision::sqrt(x * x + y * y);
    }

    [[nodiscard]] mpfr_ref ref_ldexp(mpfr_ref value, int exponent)
    {
        if (exponent > 0)
        {
            for (int i = 0; i < exponent; ++i)
                value *= 2;
        }
        else
        {
            for (int i = 0; i < -exponent; ++i)
                value /= 2;
        }

        return value;
    }

    [[nodiscard]] bool try_mpfr_unary_reference(const char* op_name, float input, float& expected)
    {
        const mpfr_ref x = to_ref_exact(input);
        mpfr_ref ref{};

        if (op_is(op_name, "sin")) ref = boost::multiprecision::sin(x);
        else if (op_is(op_name, "cos")) ref = boost::multiprecision::cos(x);
        else if (op_is(op_name, "tan")) ref = boost::multiprecision::tan(x);
        else if (op_is(op_name, "atan")) ref = boost::multiprecision::atan(x);
        else if (op_is(op_name, "asin")) ref = boost::multiprecision::asin(x);
        else if (op_is(op_name, "acos")) ref = boost::multiprecision::acos(x);
        else if (op_is(op_name, "floor")) ref = ref_floor(x);
        else if (op_is(op_name, "ceil")) ref = ref_ceil(x);
        else if (op_is(op_name, "trunc")) ref = ref_trunc(x);
        else if (op_is(op_name, "round")) ref = ref_round_half_away_zero(x);
        else if (op_is(op_name, "nearbyint") || op_is(op_name, "rint")) ref = ref_round_to_even(x);
        else if (op_is(op_name, "exp")) ref = boost::multiprecision::exp(x);
        else if (op_is(op_name, "exp2")) ref = ref_exp2(x);
        else if (op_is(op_name, "expm1")) ref = boost::multiprecision::expm1(x);
        else if (op_is(op_name, "log")) ref = boost::multiprecision::log(x);
        else if (op_is(op_name, "log2")) ref = ref_log2(x);
        else if (op_is(op_name, "log10")) ref = ref_log10(x);
        else if (op_is(op_name, "log1p")) ref = boost::multiprecision::log1p(x);
        else if (op_is(op_name, "sqrt")) ref = boost::multiprecision::sqrt(x);
        else if (op_starts_with(op_name, "cbrt"))
        {
            if (input == 0.0f)
            {
                expected = input;
                return true;
            }
            ref = ref_cbrt(x);
        }
        else if (op_is(op_name, "sinh")) ref = boost::multiprecision::sinh(x);
        else if (op_is(op_name, "cosh")) ref = boost::multiprecision::cosh(x);
        else if (op_is(op_name, "tanh")) ref = boost::multiprecision::tanh(x);
        else if (op_is(op_name, "asinh")) ref = boost::multiprecision::asinh(x);
        else if (op_is(op_name, "acosh")) ref = boost::multiprecision::acosh(x);
        else if (op_is(op_name, "atanh")) ref = boost::multiprecision::atanh(x);
        else if (op_is(op_name, "erf")) ref = boost::multiprecision::erf(x);
        else if (op_is(op_name, "erfc")) ref = boost::multiprecision::erfc(x);
        else if (op_is(op_name, "lgamma")) ref = boost::multiprecision::lgamma(x);
        else if (op_is(op_name, "tgamma")) ref = boost::multiprecision::tgamma(x);
        else if (op_is(op_name, "fabs") || op_is(op_name, "abs"))
        {
            expected = std::fabs(input);
            return true;
        }
        else return false;

        expected = round_ref_to_f32(ref);
        if (expected == 0.0f &&
            (op_is(op_name, "sin") || op_is(op_name, "tan") || op_is(op_name, "atan") ||
             op_is(op_name, "asin") || op_is(op_name, "sinh") || op_is(op_name, "tanh") ||
             op_is(op_name, "asinh") || op_is(op_name, "erf") || op_is(op_name, "expm1") ||
             op_is(op_name, "log1p") || op_is(op_name, "floor") || op_is(op_name, "ceil") ||
             op_is(op_name, "trunc") || op_is(op_name, "round") || op_is(op_name, "nearbyint") ||
             op_is(op_name, "rint")))
        {
            expected = std::copysign(0.0f, input);
        }
        return true;
    }

    [[nodiscard]] bool try_mpfr_binary_reference(const char* op_name, float lhs, float rhs, float& expected)
    {
        const mpfr_ref a = to_ref_exact(lhs);
        const mpfr_ref b = to_ref_exact(rhs);
        mpfr_ref ref{};

        if (op_is(op_name, "add")) ref = a + b;
        else if (op_is(op_name, "subtract")) ref = a - b;
        else if (op_is(op_name, "multiply")) ref = a * b;
        else if (op_is(op_name, "divide")) ref = a / b;
        else if (op_is(op_name, "atan2")) ref = boost::multiprecision::atan2(a, b);
        else if (op_is(op_name, "hypot")) ref = ref_hypot(a, b);
        else if (op_is(op_name, "pow"))
        {
            if (lhs == 0.0f)
            {
                expected = std::pow(lhs, rhs);
                return true;
            }
            ref = ref_pow(a, b);
        }
        else if (op_is(op_name, "fmod"))
        {
            expected = exact_mod_reference_for(lhs, rhs).fmod;
            return true;
        }
        else if (op_is(op_name, "remainder"))
        {
            expected = exact_mod_reference_for(lhs, rhs).remainder;
            return true;
        }
        else if (op_is(op_name, "fdim")) ref = (a > b) ? (a - b) : mpfr_ref{ 0 };
        else return false;

        expected = round_ref_to_f32(ref);
        return true;
    }

    [[nodiscard]] float achieved_digits_from_error(float diff, float scale)
    {
        if (diff == 0.0)
            return static_cast<float>(checked_digits);

        const float scaled_error = diff / scale;
        if (!(scaled_error > 0.0))
            return static_cast<float>(checked_digits);

        if (scaled_error >= 1.0)
            return 0.0;

        const float digits = -std::log10(scaled_error);
        return digits < 0.0 ? 0.0 : digits;
    }

    [[nodiscard]] float normalized_accuracy_percent(float digits)
    {
        if (checked_digits <= 0)
            return 100.0;

        float ratio = digits / static_cast<float>(checked_digits);
        if (ratio < 0.0)
            ratio = 0.0;
        if (ratio > 1.0)
            ratio = 1.0;
        return ratio * 100.0;
    }

    [[nodiscard]] float median_digits(std::vector<float> values)
    {
        if (values.empty())
            return 0.0;

        std::sort(values.begin(), values.end());
        const std::size_t mid = values.size() / 2;
        if ((values.size() & 1u) != 0u)
            return values[mid];

        return (values[mid - 1] + values[mid]) * 0.5;
    }

    [[nodiscard]] std::uint32_t ordered_bits(float value)
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        if ((bits & 0x80000000u) != 0u)
            return ~bits;

        return bits | 0x80000000u;
    }

    [[nodiscard]] std::uint32_t ulp_distance(float a, float b)
    {
        const std::uint32_t oa = ordered_bits(a);
        const std::uint32_t ob = ordered_bits(b);
        return (oa >= ob) ? (oa - ob) : (ob - oa);
    }

    class accuracy_report_scope
    {
    public:
        explicit accuracy_report_scope(const char* test_name)
            : test_name(test_name), previous(current_accuracy_report_scope)
        {
            current_accuracy_report_scope = this;
        }

        ~accuracy_report_scope()
        {
            current_accuracy_report_scope = previous;

            if (stats.empty())
                return;

            const std::ios_base::fmtflags old_flags = std::cout.flags();
            const std::streamsize old_precision     = std::cout.precision();

            std::cout << "\naccuracy summary for " << test_name << ":\n";
            std::cout << std::fixed << std::setprecision(2);

            for (const auto& [op_name, entry] : stats)
            {
                const float median = median_digits(entry.achieved_digits);
                const float worst  = *std::min_element(entry.achieved_digits.begin(), entry.achieved_digits.end());

                std::cout << "  " << op_name
                          << ": pass " << entry.passed << "/" << entry.samples
                          << ", median " << median << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(median) << "%)"
                          << ", worst " << worst << "/" << checked_digits
                          << " digits (" << normalized_accuracy_percent(worst) << "%)"
                          << ", worst ulp " << entry.worst_ulp
                          << ", worst scaled error " << std::scientific << entry.worst_scaled_error << std::fixed
                          << "\n";
            }

            std::cout.flags(old_flags);
            std::cout.precision(old_precision);
        }

        void record(const char* op_name, float diff, float scale, std::uint32_t ulp_diff, bool passed)
        {
            auto& entry = stats[op_name];
            ++entry.samples;
            if (passed)
                ++entry.passed;
            entry.achieved_digits.push_back(achieved_digits_from_error(diff, scale));
            if (ulp_diff > entry.worst_ulp)
                entry.worst_ulp = ulp_diff;
            if (scale != 0.0f && std::isfinite(diff) && std::isfinite(scale))
                entry.worst_scaled_error = std::max(entry.worst_scaled_error, diff / scale);
        }

    private:
        std::string test_name;
        accuracy_report_scope* previous = nullptr;
        std::map<std::string, accuracy_stats_entry> stats;
    };

    void record_accuracy_sample(const char* op_name, float diff, float scale, std::uint32_t ulp_diff, bool passed)
    {
        if (current_accuracy_report_scope != nullptr)
            current_accuracy_report_scope->record(op_name, diff, scale, ulp_diff, passed);
    }

    struct tolerance_spec
    {
        float abs_tolerance    = 0.0;
        float rel_tolerance    = 0.0;
        std::uint32_t max_ulps = 0;
    };

    [[nodiscard]] tolerance_spec with_default_mpfr_tolerance(const char* op_name, tolerance_spec tolerance);

    [[nodiscard]] bool both_nan(float a, float b)
    {
        return std::isnan(a) && std::isnan(b);
    }

    [[nodiscard]] bool same_zero_with_same_sign(float a, float b)
    {
        return a == 0.0 && b == 0.0 && std::signbit(a) == std::signbit(b);
    }

    [[nodiscard]] bool same_infinity_with_same_sign(float a, float b)
    {
        return std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b);
    }

    [[nodiscard]] bool same_bits(float a, float b)
    {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }

    struct floating_compare_result
    {
        bool passed            = false;
        float signed_diff      = 0.0;
        float abs_diff         = 0.0;
        float rel_diff         = 0.0;
        float scale            = 1.0;
        float allowed_diff     = 0.0;
        float achieved_digits  = 0.0;
        std::uint32_t ulp_diff = 0;
        const char* reason     = "";
    };

    [[nodiscard]] floating_compare_result compare_floating_result(
        const char* op_name,
        float got,
        float expected,
        const tolerance_spec& tolerance)
    {
        floating_compare_result result{};

        if (both_nan(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "both NaN";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (same_infinity_with_same_sign(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "same signed infinity";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (same_zero_with_same_sign(got, expected))
        {
            result.passed = true;
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "same signed zero";
            record_accuracy_sample(op_name, 0.0, 1.0, 0, true);
            return result;
        }

        if (got == 0.0 && expected == 0.0)
        {
            result.passed = false;
            result.abs_diff = 0.0;
            result.scale = 1.0;
            result.allowed_diff = tolerance.abs_tolerance;
            result.ulp_diff = 1;
            result.reason = "zero sign mismatch";
            record_accuracy_sample(op_name, 0.0, 1.0, 1, false);
            return result;
        }

        if (same_bits(got, expected))
        {
            result.passed = true;
            result.scale = std::max(1.0f, abs_ref(expected));
            result.achieved_digits = static_cast<float>(checked_digits);
            result.reason = "bitwise equal";
            record_accuracy_sample(op_name, 0.0, result.scale, 0, true);
            return result;
        }

        if (!std::isfinite(got) || !std::isfinite(expected))
        {
            result.passed = false;
            result.abs_diff = std::numeric_limits<float>::infinity();
            result.rel_diff = std::numeric_limits<float>::infinity();
            result.scale = 1.0;
            result.allowed_diff = tolerance.abs_tolerance;
            result.ulp_diff = std::numeric_limits<std::uint32_t>::max();
            result.reason = "non-finite mismatch";
            record_accuracy_sample(op_name, std::numeric_limits<float>::infinity(), 1.0, std::numeric_limits<std::uint32_t>::max(), false);
            return result;
        }

        result.signed_diff = got - expected;
        result.abs_diff = abs_ref(result.signed_diff);
        result.scale = abs_ref(expected);
        if (result.scale < 1.0)
            result.scale = 1.0;

        const float rel_based_tolerance = tolerance.rel_tolerance * result.scale;
        result.allowed_diff = tolerance.abs_tolerance;
        if (rel_based_tolerance > result.allowed_diff)
            result.allowed_diff = rel_based_tolerance;

        if (expected == 0.0)
            result.rel_diff = (result.abs_diff == 0.0) ? 0.0 : std::numeric_limits<float>::infinity();
        else
            result.rel_diff = result.abs_diff / abs_ref(expected);

        result.ulp_diff = ulp_distance(got, expected);
        const bool within_diff = result.abs_diff <= result.allowed_diff;
        const bool within_ulps = result.ulp_diff <= tolerance.max_ulps;
        result.passed = within_diff || within_ulps;
        result.achieved_digits = achieved_digits_from_error(result.abs_diff, result.scale);
        result.reason = result.passed ? "within tolerance" : "outside tolerance";

        record_accuracy_sample(op_name, result.abs_diff, result.scale, result.ulp_diff, result.passed);
        return result;
    }

    [[nodiscard]] std::string build_comparison_message(
        const char* op_name,
        float got,
        float expected,
        const tolerance_spec& tolerance,
        const floating_compare_result& result)
    {
        std::ostringstream out;
        out << op_name << " mismatch"
            << "\n  got: " << to_text(got)
            << "\n  expected: " << to_text(expected)
            << "\n  got hex: " << to_text_hex(got)
            << "\n  expected hex: " << to_text_hex(expected)
            << "\n  signed diff (got - expected): " << to_text(result.signed_diff)
            << "\n  abs diff: " << to_text(result.abs_diff)
            << "\n  relative diff: " << to_text(result.rel_diff)
            << "\n  allowed abs diff: " << to_text(result.allowed_diff)
            << "\n  ulp diff: " << result.ulp_diff
            << " (limit " << tolerance.max_ulps << ")"
            << "\n  achieved digits: " << result.achieved_digits << "/" << checked_digits
            << "\n  reason: " << result.reason;
        return out.str();
    }

    template<typename F64Op, typename StdOp>
    void check_unary_op(
        const char* op_name,
        float input,
        const tolerance_spec& tolerance,
        F64Op&& f32_op,
        StdOp&& std_op)
    {
        const float got = f32_op(input);
        float expected{};
        if (!try_mpfr_unary_reference(op_name, input, expected))
            expected = std_op(input);
        const tolerance_spec effective_tolerance = with_default_mpfr_tolerance(op_name, tolerance);

        INFO(op_name
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input));

        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, effective_tolerance);
        INFO(build_comparison_message(op_name, got, expected, effective_tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    template<typename F64Op, typename StdOp>
    void check_binary_op(
        const char* op_name,
        float lhs,
        float rhs,
        const tolerance_spec& tolerance,
        F64Op&& f32_op,
        StdOp&& std_op)
    {
        const float got = f32_op(lhs, rhs);
        float expected{};
        if (!try_mpfr_binary_reference(op_name, lhs, rhs, expected))
            expected = std_op(lhs, rhs);
        const tolerance_spec effective_tolerance = with_default_mpfr_tolerance(op_name, tolerance);

        INFO(op_name
            << "\n  lhs: " << to_text(lhs)
            << "\n  rhs: " << to_text(rhs)
            << "\n  lhs hex: " << to_text_hex(lhs)
            << "\n  rhs hex: " << to_text_hex(rhs));

        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, effective_tolerance);
        INFO(build_comparison_message(op_name, got, expected, effective_tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    void check_fma_result(float x, float y, float z, const tolerance_spec& tolerance)
    {
        const char* op_name                      = "fma";
        const float got                          = bl::fma(x, y, z);
        const float expected                     = round_ref_to_f32(to_ref_exact(x) * to_ref_exact(y) + to_ref_exact(z));
        const tolerance_spec effective_tolerance = with_default_mpfr_tolerance(op_name, tolerance);

        INFO(op_name
            << "\n  x: " << to_text(x)
            << "\n  y: " << to_text(y)
            << "\n  z: " << to_text(z)
            << "\n  x hex: " << to_text_hex(x)
            << "\n  y hex: " << to_text_hex(y)
            << "\n  z hex: " << to_text_hex(z));

        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, effective_tolerance);
        INFO(build_comparison_message(op_name, got, expected, effective_tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    void check_scaled_result(const char* op_name, float value, int exponent, float got)
    {
        const float expected = value == 0.0f
            ? value
            : round_ref_to_f32(ref_ldexp(to_ref_exact(value), exponent));

        INFO(op_name
            << "\n  input: " << to_text(value)
            << "\n  exponent: " << exponent
            << "\n  input hex: " << to_text_hex(value));

        const tolerance_spec tolerance{};
        const floating_compare_result comparison = compare_floating_result(op_name, got, expected, tolerance);
        INFO(build_comparison_message(op_name, got, expected, tolerance, comparison));
        REQUIRE(comparison.passed);
    }

    template<typename Int>
    void check_exact_integer_result(const char* op_name, Int got, Int expected, float input)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(input));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    template<typename Int>
    void check_exact_integer_result(const char* op_name, Int got, Int expected, float lhs, float rhs)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(lhs));
        CAPTURE(to_text(rhs));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    void check_exact_bool_result(const char* op_name, bool got, bool expected, float lhs, float rhs)
    {
        CAPTURE(op_name);
        CAPTURE(to_text(lhs));
        CAPTURE(to_text(rhs));
        CAPTURE(got);
        CAPTURE(expected);
        REQUIRE(got == expected);
    }

    void check_frexp_result(float input)
    {
        int got_exp      = 0;
        int expected_exp = 0;

        const float got      = bl::frexp(input, &got_exp);
        const float expected = std::frexp(input, &expected_exp);

        INFO("frexp"
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input)
            << "\n  got exp: " << got_exp
            << "\n  expected exp: " << expected_exp);

        const floating_compare_result comparison = compare_floating_result("frexp", got, expected, tolerance_spec{});
        INFO(build_comparison_message("frexp", got, expected, tolerance_spec{}, comparison));
        REQUIRE(comparison.passed);
        REQUIRE(got_exp == expected_exp);
    }

    void check_modf_result(float input)
    {
        float got_int      = 0.0;
        float expected_int = 0.0;

        const float got      = bl::modf(input, &got_int);
        const float expected = std::modf(input, &expected_int);

        INFO("modf"
            << "\n  input: " << to_text(input)
            << "\n  input hex: " << to_text_hex(input)
            << "\n  got int: " << to_text(got_int)
            << "\n  expected int: " << to_text(expected_int));

        const floating_compare_result frac_comparison = compare_floating_result("modf.frac", got, expected, tolerance_spec{});
        INFO(build_comparison_message("modf.frac", got, expected, tolerance_spec{}, frac_comparison));
        REQUIRE(frac_comparison.passed);

        const floating_compare_result int_comparison = compare_floating_result("modf.int", got_int, expected_int, tolerance_spec{});
        INFO(build_comparison_message("modf.int", got_int, expected_int, tolerance_spec{}, int_comparison));
        REQUIRE(int_comparison.passed);
    }

    void check_remquo_result(float lhs, float rhs)
    {
        int got_quo = 0;

        const float got                        = bl::remquo(lhs, rhs, &got_quo);
        const exact_mod_reference expected_ref = exact_mod_reference_for(lhs, rhs);
        const float expected                   = expected_ref.remainder;
        const int expected_quo                 = expected_ref.quotient_bits;

        INFO("remquo"
            << "\n  lhs: " << to_text(lhs)
            << "\n  rhs: " << to_text(rhs)
            << "\n  lhs hex: " << to_text_hex(lhs)
            << "\n  rhs hex: " << to_text_hex(rhs)
            << "\n  got quo: " << got_quo
            << "\n  expected quo: " << expected_quo);

        const floating_compare_result comparison = compare_floating_result("remquo", got, expected, tolerance_spec{});
        INFO(build_comparison_message("remquo", got, expected, tolerance_spec{}, comparison));
        REQUIRE(comparison.passed);

        if (!std::isnan(expected))
        {
            const int got_sign          = (got_quo > 0) - (got_quo < 0);
            const int expected_sign     = (expected_quo > 0) - (expected_quo < 0);
            const int got_low_bits      = got_quo & 0x7;
            const int expected_low_bits = expected_quo & 0x7;

            CAPTURE(got_sign);
            CAPTURE(expected_sign);
            CAPTURE(got_low_bits);
            CAPTURE(expected_low_bits);

            REQUIRE(got_low_bits == expected_low_bits);
            if (expected_low_bits != 0)
                REQUIRE(got_sign == expected_sign);
        }
    }

    void print_random_run(const char* description, int count)
    {
        std::cout << type_label << " comparing: " << count << " " << description
                  << " (seed " << random_seed << ")...\n\n";
    }

    [[nodiscard]] float random_finite_for_f32(std::mt19937_64& rng)
    {
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> exponent_dist(-149, 127);
        std::uniform_real_distribution<float> mantissa_dist(0.5f, 1.0f);

        float value = std::ldexp(mantissa_dist(rng), exponent_dist(rng));
        if (!std::isfinite(value) || value == 0.0f)
            value = std::numeric_limits<float>::denorm_min();
        if (sign_dist(rng) != 0)
            value = -value;

        return value;
    }

    [[nodiscard]] float random_signed_interval(std::mt19937_64& rng, float limit)
    {
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(rng);
    }

    [[nodiscard]] float random_unit_interval(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        return dist(rng);
    }

    [[nodiscard]] float random_positive(std::mt19937_64& rng)
    {
        float value = std::fabs(random_finite_for_f32(rng));
        if (value < std::numeric_limits<float>::denorm_min())
            value = std::numeric_limits<float>::denorm_min();
        return value;
    }

    [[nodiscard]] float random_nonzero(std::mt19937_64& rng)
    {
        float value = random_finite_for_f32(rng);
        if (value == 0.0)
            value = 1.0;
        return value;
    }

    [[nodiscard]] float random_log1p_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(-0.95f, 16.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_acosh_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(1.0f, 64.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_atanh_argument(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(-0.95f, 0.95f);
        return dist(rng);
    }

    [[nodiscard]] float random_pow_base(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.125f, 8.0f);
        return dist(rng);
    }

    [[nodiscard]] float random_gamma_positive(std::mt19937_64& rng)
    {
        std::uniform_real_distribution<float> dist(0.125f, 15.0f);
        return dist(rng);
    }

    [[nodiscard]] constexpr tolerance_spec exact_tol()
    {
        return tolerance_spec{};
    }

    [[nodiscard]] constexpr tolerance_spec close_tol(std::uint32_t ulps, float abs_tol = 0.0, float rel_tol = 0.0)
    {
        return tolerance_spec{ abs_tol, rel_tol, ulps };
    }

    struct default_tolerance_entry
    {
        const char* op_name;
        std::uint32_t ulps;
        float tolerance;
    };

    [[nodiscard]] tolerance_spec default_mpfr_tolerance_for_op(const char* op_name)
    {
        constexpr std::array<default_tolerance_entry, 27> tolerances{{
            { "acos", 1, 2e-7f },
            { "acosh", 1, 2e-7f },
            { "asin", 1, 2e-7f },
            { "asinh", 1, 2e-7f },
            { "atan", 1, 2e-7f },
            { "atan2", 1, 1e-7f },
            { "atanh", 1, 2e-7f },
            { "cbrt", 1, 2e-7f },
            { "cbrt.neg", 1, 2e-7f },
            { "cos", 1, 3e-8f },
            { "cosh", 1, 7e-8f },
            { "erf", 3, 2e-7f },
            { "erfc", 3, 2e-7f },
            { "exp", 1, 2e-7f },
            { "exp2", 1, 1e-7f },
            { "expm1", 1, 2e-7f },
            { "hypot", 1, 1e-7f },
            { "lgamma", 2, 2e-7f },
            { "log", 1, 2e-7f },
            { "log10", 1, 2e-7f },
            { "log1p", 1, 9e-8f },
            { "pow", 1, 9e-8f },
            { "sin", 1, 6e-8f },
            { "sinh", 1, 2e-7f },
            { "tan", 1, 2e-7f },
            { "tanh", 1, 6e-8f },
            { "tgamma", 3, 4e-7f }
        }};

        for (const auto& entry : tolerances)
        {
            if (op_is(op_name, entry.op_name))
                return close_tol(entry.ulps, entry.tolerance, entry.tolerance);
        }

        return exact_tol();
    }

    [[nodiscard]] tolerance_spec with_default_mpfr_tolerance(const char* op_name, tolerance_spec tolerance)
    {
        if (tolerance.abs_tolerance != 0.0f || tolerance.rel_tolerance != 0.0f || tolerance.max_ulps != 0)
            return tolerance;

        return default_mpfr_tolerance_for_op(op_name);
    }

} // namespace

TEST_CASE("f32 matches MPFR for + - * /", "[fltx][f32][precision][arithmetic]")
{
    accuracy_report_scope report("f32 matches MPFR for + - * /");

    constexpr std::array<std::pair<float, float>, 10> cases{{
        { 0.0f, 0.0f },
        { 1.0f, 2.0f },
        { -1.0f, 2.0f },
        { 1.25f, -2.5f },
        { min_subnormal, min_subnormal },
        { -min_subnormal, min_subnormal },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::pi, (f32)std::numbers::sqrt2 },
        { 1e30f, -min_subnormal },
        { -1e30f, 3.0f }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("add", lhs, rhs, exact_tol(),
            [](float a, float b) { return a + b; },
            [](float a, float b) { return a + b; });

        check_binary_op("subtract", lhs, rhs, exact_tol(),
            [](float a, float b) { return a - b; },
            [](float a, float b) { return a - b; });

        check_binary_op("multiply", lhs, rhs, exact_tol(),
            [](float a, float b) { return a * b; },
            [](float a, float b) { return a * b; });

        if (rhs != 0.0)
        {
            check_binary_op("divide", lhs, rhs, exact_tol(),
                [](float a, float b) { return a / b; },
                [](float a, float b) { return a / b; });
        }
    }
}

TEST_CASE("f32 random arithmetic matches MPFR", "[fltx][f32][precision][arithmetic]")
{
    accuracy_report_scope report("f32 random arithmetic matches MPFR");
    print_random_run("random arithmetic pairs", random_sample_count);

    std::mt19937_64 rng(random_seed);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float lhs = random_finite_for_f32(rng);
        const float rhs = random_finite_for_f32(rng);

        check_binary_op("add", lhs, rhs, exact_tol(),
            [](float a, float b) { return a + b; },
            [](float a, float b) { return a + b; });

        check_binary_op("subtract", lhs, rhs, exact_tol(),
            [](float a, float b) { return a - b; },
            [](float a, float b) { return a - b; });

        check_binary_op("multiply", lhs, rhs, exact_tol(),
            [](float a, float b) { return a * b; },
            [](float a, float b) { return a * b; });

        if (rhs != 0.0)
        {
            check_binary_op("divide", lhs, rhs, exact_tol(),
                [](float a, float b) { return a / b; },
                [](float a, float b) { return a / b; });
        }
    }
}

TEST_CASE("f32 trig matches MPFR for fixed values", "[fltx][f32][precision][transcendental][trig]")
{
    accuracy_report_scope report("f32 trig matches MPFR for fixed values");

    constexpr std::array<float, 13> unary_cases{{
        -3.0,
        -1.5,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        1.5,
        3.0,
        std::numbers::pi / 6.0,
        std::numbers::pi / 4.0,
        std::numbers::pi / 3.0
    }};

    for (float input : unary_cases)
    {
        check_unary_op("sin", input, exact_tol(),
            [](float x) { return bl::sin(x); },
            [](float x) { return std::sin(x); });

        check_unary_op("cos", input, exact_tol(),
            [](float x) { return bl::cos(x); },
            [](float x) { return std::cos(x); });

        check_unary_op("tan", input, exact_tol(),
            [](float x) { return bl::tan(x); },
            [](float x) { return std::tan(x); });

        check_unary_op("atan", input, exact_tol(),
            [](float x) { return bl::atan(x); },
            [](float x) { return std::atan(x); });

        if (input >= -1.0 && input <= 1.0)
        {
            check_unary_op("asin", input, exact_tol(),
                [](float x) { return bl::asin(x); },
                [](float x) { return std::asin(x); });

            check_unary_op("acos", input, exact_tol(),
                [](float x) { return bl::acos(x); },
                [](float x) { return std::acos(x); });
        }
    }

    constexpr std::array<std::pair<float, float>, 8> atan2_cases{{
        { 0.0f, 1.0f },
        { 1.0f, 0.0f },
        { -1.0f, 0.0f },
        { 1.0f, 1.0f },
        { -1.0f, 1.0f },
        { 1.0f, -1.0f },
        { -1.0f, -1.0f },
        { (f32)std::numbers::pi, (f32)-std::numbers::e }
    }};

    for (const auto& [y, x] : atan2_cases)
    {
        check_binary_op("atan2", y, x, exact_tol(),
            [](float a, float b) { return bl::atan2(a, b); },
            [](float a, float b) { return std::atan2(a, b); });
    }
}

TEST_CASE("f32 sincos overloads match sin and cos", "[fltx][f32][precision][transcendental][trig][sincos]")
{
    const float input = 0.625f;

    float s_out{};
    float c_out{};
    REQUIRE(bl::sincos(input, s_out, c_out));
    REQUIRE(s_out == bl::sin(input));
    REQUIRE(c_out == bl::cos(input));

    sincos_pair<float> out{};
    REQUIRE(bl::sincos(input, out));
    REQUIRE(out.s == s_out);
    REQUIRE(out.c == c_out);

    const sincos_pair<float> returned = bl::sincos<float>(input);
    REQUIRE(returned.s == s_out);
    REQUIRE(returned.c == c_out);
}

TEST_CASE("f32 trig matches MPFR on random inputs", "[fltx][f32][precision][transcendental][trig]")
{
    accuracy_report_scope report("f32 trig matches MPFR on random inputs");
    print_random_run("random trig inputs", random_sample_count);

    std::mt19937_64 rng(random_seed);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float trig_input = random_signed_interval(rng, 16.0);
        const float tan_input  = random_signed_interval(rng, 1.2);
        const float unit_input = random_signed_interval(rng, 1.0);
        const float y          = random_signed_interval(rng, 128.0);
        const float x          = random_signed_interval(rng, 128.0);

        check_unary_op("sin", trig_input, exact_tol(),
            [](float v) { return bl::sin(v); },
            [](float v) { return std::sin(v); });

        check_unary_op("cos", trig_input, exact_tol(),
            [](float v) { return bl::cos(v); },
            [](float v) { return std::cos(v); });

        check_unary_op("tan", tan_input, exact_tol(),
            [](float v) { return bl::tan(v); },
            [](float v) { return std::tan(v); });

        check_unary_op("atan", random_signed_interval(rng, 128.0), exact_tol(),
            [](float v) { return bl::atan(v); },
            [](float v) { return std::atan(v); });

        check_unary_op("asin", unit_input, exact_tol(),
            [](float v) { return bl::asin(v); },
            [](float v) { return std::asin(v); });

        check_unary_op("acos", unit_input, exact_tol(),
            [](float v) { return bl::acos(v); },
            [](float v) { return std::acos(v); });

        if (x == 0.0 && y == 0.0)
            continue;

        check_binary_op("atan2", y, x, exact_tol(),
            [](float a, float b) { return bl::atan2(a, b); },
            [](float a, float b) { return std::atan2(a, b); });
    }
}

TEST_CASE("f32 rounding matches MPFR references", "[fltx][f32][precision][math][rounding]")
{
    accuracy_report_scope report("f32 rounding matches MPFR references");

    constexpr std::array<float, 16> fixed_inputs{{
        -3.75,
        -2.5,
        -1.5,
        -1.0,
        -0.5,
        -0.25,
        -0.0,
        0.0,
        0.25,
        0.5,
        1.0,
        1.5,
        2.5,
        3.75,
        123456.5,
        -123456.5
    }};

    for (float input : fixed_inputs)
    {
        check_unary_op("floor", input, exact_tol(),
            [](float x) { return bl::floor(x); },
            [](float x) { return std::floor(x); });

        check_unary_op("ceil", input, exact_tol(),
            [](float x) { return bl::ceil(x); },
            [](float x) { return std::ceil(x); });

        check_unary_op("trunc", input, exact_tol(),
            [](float x) { return bl::trunc(x); },
            [](float x) { return std::trunc(x); });

        check_unary_op("round", input, exact_tol(),
            [](float x) { return bl::round(x); },
            [](float x) { return std::round(x); });

        check_unary_op("nearbyint", input, exact_tol(),
            [](float x) { return bl::nearbyint(x); },
            [](float x) { return std::nearbyint(x); });

        check_unary_op("rint", input, exact_tol(),
            [](float x) { return bl::rint(x); },
            [](float x) { return std::rint(x); });

        check_exact_integer_result("lround", bl::lround(input), std::lround(input), input);
        check_exact_integer_result("llround", bl::llround(input), std::llround(input), input);
        check_exact_integer_result("lrint", bl::lrint(input), std::lrint(input), input);
        check_exact_integer_result("llrint", bl::llrint(input), std::llrint(input), input);
    }

    REQUIRE(bl::round_to_decimals(1.2345f, 2) == 1.23f);
    REQUIRE(bl::round_to_decimals(1.2345f, 3) == 1.235f);
    REQUIRE(bl::round_to_decimals(1.125f, 2) == 1.12f);
    REQUIRE(bl::round_to_decimals(1.375f, 2) == 1.38f);
    REQUIRE(bl::round_to_decimals(-1.375f, 2) == -1.38f);
    REQUIRE(bl::round_to_decimals(1.25f, 0) == 1.25f);

    constexpr float constexpr_rounded = bl::round_to_decimals(1.375f, 2);
    static_assert(constexpr_rounded == 1.38f);

    static_assert(bl::pow10<bl::f32>(0) == 1.0f);
    static_assert(bl::pow10<bl::f32>(3) == 1000.0f);
    static_assert(bl::pow10<bl::f32>(-3) == 1e-3f);

    REQUIRE(bl::pow10<bl::f32>(38) == 1e38f);
    REQUIRE(bl::isinf(bl::pow10<bl::f32>(39)));
    REQUIRE(bl::pow10<bl::f32>(-45) == 1e-45f);
    REQUIRE(bl::pow10<bl::f32>(-46) == 0.0f);

    REQUIRE(bl::log10(bl::pow10<bl::f32>(10)) == 10.0f);

    std::mt19937_64 rng(random_seed);
    print_random_run("random rounding inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float input = random_finite_for_f32(rng);

        check_unary_op("floor", input, exact_tol(),
            [](float x) { return bl::floor(x); },
            [](float x) { return std::floor(x); });

        check_unary_op("ceil", input, exact_tol(),
            [](float x) { return bl::ceil(x); },
            [](float x) { return std::ceil(x); });

        check_unary_op("trunc", input, exact_tol(),
            [](float x) { return bl::trunc(x); },
            [](float x) { return std::trunc(x); });

        check_unary_op("round", input, exact_tol(),
            [](float x) { return bl::round(x); },
            [](float x) { return std::round(x); });

        check_unary_op("nearbyint", input, exact_tol(),
            [](float x) { return bl::nearbyint(x); },
            [](float x) { return std::nearbyint(x); });

        check_unary_op("rint", input, exact_tol(),
            [](float x) { return bl::rint(x); },
            [](float x) { return std::rint(x); });

        if (std::fabs(input) < static_cast<float>(std::numeric_limits<long long>::max()) - 1.0)
        {
            check_exact_integer_result("lround", bl::lround(input), std::lround(input), input);
            check_exact_integer_result("llround", bl::llround(input), std::llround(input), input);
            check_exact_integer_result("lrint", bl::lrint(input), std::lrint(input), input);
            check_exact_integer_result("llrint", bl::llrint(input), std::llrint(input), input);
        }
    }
}

TEST_CASE("f32 exp and log families match MPFR", "[fltx][f32][precision][transcendental]")
{
    accuracy_report_scope report("f32 exp and log families match MPFR");

    constexpr std::array<float, 10> exp_inputs{{
        -20.0,
        -4.0,
        -1.0,
        -0.1,
        0.0,
        0.1,
        1.0,
        4.0,
        10.0,
        20.0
    }};

    for (float input : exp_inputs)
    {
        check_unary_op("exp", input, exact_tol(),
            [](float x) { return bl::exp(x); },
            [](float x) { return std::exp(x); });

        check_unary_op("exp2", input, exact_tol(),
            [](float x) { return bl::exp2(x); },
            [](float x) { return std::exp2(x); });

        check_unary_op("expm1", input, exact_tol(),
            [](float x) { return bl::expm1(x); },
            [](float x) { return std::expm1(x); });
    }

    constexpr std::array<float, 8> log_inputs{{
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.125,
        0.5,
        1.0,
        2.0,
        10.0,
        1e30f
    }};

    for (float input : log_inputs)
    {
        check_unary_op("log", input, exact_tol(),
            [](float x) { return bl::log(x); },
            [](float x) { return std::log(x); });

        check_unary_op("log2", input, exact_tol(),
            [](float x) { return bl::log2(x); },
            [](float x) { return std::log2(x); });

        check_unary_op("log10", input, exact_tol(),
            [](float x) { return bl::log10(x); },
            [](float x) { return std::log10(x); });
    }

    constexpr std::array<float, 8> log1p_inputs{{
        -0.95,
        -0.75,
        -0.5,
        -0.125,
        0.0,
        0.125,
        1.0,
        16.0
    }};

    for (float input : log1p_inputs)
    {
        check_unary_op("log1p", input, exact_tol(),
            [](float x) { return bl::log1p(x); },
            [](float x) { return std::log1p(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random exp/log inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float exp_input   = random_signed_interval(rng, 20.0);
        const float log_input   = random_positive(rng);
        const float log1p_input = random_log1p_argument(rng);

        check_unary_op("exp", exp_input, exact_tol(),
            [](float x) { return bl::exp(x); },
            [](float x) { return std::exp(x); });

        check_unary_op("exp2", exp_input, exact_tol(),
            [](float x) { return bl::exp2(x); },
            [](float x) { return std::exp2(x); });

        check_unary_op("expm1", exp_input, exact_tol(),
            [](float x) { return bl::expm1(x); },
            [](float x) { return std::expm1(x); });

        check_unary_op("log", log_input, exact_tol(),
            [](float x) { return bl::log(x); },
            [](float x) { return std::log(x); });

        check_unary_op("log2", log_input, exact_tol(),
            [](float x) { return bl::log2(x); },
            [](float x) { return std::log2(x); });

        check_unary_op("log10", log_input, exact_tol(),
            [](float x) { return bl::log10(x); },
            [](float x) { return std::log10(x); });

        check_unary_op("log1p", log1p_input, exact_tol(),
            [](float x) { return bl::log1p(x); },
            [](float x) { return std::log1p(x); });
    }
}

TEST_CASE("f32 root and power functions match MPFR", "[fltx][f32][precision][math]")
{
    accuracy_report_scope report("f32 root and power functions match MPFR");

    constexpr std::array<float, 8> root_inputs{{
        0.0,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.125,
        1.0,
        2.0,
        1000.0,
        1e30f
    }};

    for (float input : root_inputs)
    {
        check_unary_op("sqrt", input, exact_tol(),
            [](float x) { return bl::sqrt(x); },
            [](float x) { return std::sqrt(x); });

        check_unary_op("cbrt", input, exact_tol(),
            [](float x) { return bl::cbrt(x); },
            [](float x) { return std::cbrt(x); });

        check_unary_op("cbrt.neg", -input, exact_tol(),
            [](float x) { return bl::cbrt(x); },
            [](float x) { return std::cbrt(x); });
    }

    constexpr std::array<std::pair<float, float>, 8> hypot_cases{{
        { 0.0f, 0.0f },
        { 3.0f, 4.0f },
        { -3.0f, 4.0f },
        { min_subnormal, min_subnormal },
        { 1e30f, 1e30f },
        { 1e30f, 1.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::sqrt2, (f32)std::numbers::pi }
    }};

    for (const auto& [x, y] : hypot_cases)
    {
        check_binary_op("hypot", x, y, exact_tol(),
            [](float a, float b) { return bl::hypot(a, b); },
            [](float a, float b) { return std::hypot(a, b); });
    }

    constexpr std::array<std::pair<float, float>, 10> pow_cases{{
        { 2.0f, 3.0f },
        { 2.0f, -3.0f },
        { 0.5f, 0.5f },
        { 3.0f, 1.5f },
        { 10.0f, -2.0f },
        { -2.0f, 3.0f },
        { -2.0f, 4.0f },
        { -0.0f, 3.0f },
        { -0.0f, 4.0f },
        { 0.0f, 0.0f }
    }};

    for (const auto& [base, exponent] : pow_cases)
    {
        check_binary_op("pow", base, exponent, exact_tol(),
            [](float a, float b) { return bl::pow(a, b); },
            [](float a, float b) { return std::pow(a, b); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random root/pow inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float positive     = random_positive(rng);
        const float signed_input = random_signed_interval(rng, 1e12);
        const float x            = random_signed_interval(rng, 1e12);
        const float y            = random_signed_interval(rng, 1e12);
        const float base         = random_pow_base(rng);
        const float exponent     = random_signed_interval(rng, 8.0);

        check_unary_op("sqrt", positive, exact_tol(),
            [](float v) { return bl::sqrt(v); },
            [](float v) { return std::sqrt(v); });

        check_unary_op("cbrt", signed_input, exact_tol(),
            [](float v) { return bl::cbrt(v); },
            [](float v) { return std::cbrt(v); });

        check_binary_op("hypot", x, y, exact_tol(),
            [](float a, float b) { return bl::hypot(a, b); },
            [](float a, float b) { return std::hypot(a, b); });

        check_binary_op("pow", base, exponent, exact_tol(),
            [](float a, float b) { return bl::pow(a, b); },
            [](float a, float b) { return std::pow(a, b); });
    }
}

TEST_CASE("f32 fmod and remainder match exact references", "[fltx][f32][precision][math][fmod][remainder]")
{
    accuracy_report_scope report("f32 fmod and remainder match exact references");

    constexpr std::array<std::pair<float, float>, 10> cases{{
        { 5.25f, 2.0f },
        { -5.25f, 2.0f },
        { 5.25f, -2.0f },
        { -5.25f, -2.0f },
        { 1e30f, 3.0f },
        { min_subnormal, 3.0f },
        { (f32)std::numbers::pi, 0.5f },
        { (f32)-std::numbers::pi, 0.5f },
        { 17.0f, 0.25f },
        { -17.0f, 0.25f }
    }};

    for (const auto& [lhs, rhs] : cases)
    {
        check_binary_op("fmod", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmod(a, b); },
            [](float a, float b) { return std::fmod(a, b); });

        check_binary_op("remainder", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::remainder(a, b); },
            [](float a, float b) { return std::remainder(a, b); });

        check_remquo_result(lhs, rhs);
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random fmod/remainder inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float lhs = random_finite_for_f32(rng);
        const float rhs = random_nonzero(rng);

        check_binary_op("fmod", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmod(a, b); },
            [](float a, float b) { return std::fmod(a, b); });

        check_binary_op("remainder", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::remainder(a, b); },
            [](float a, float b) { return std::remainder(a, b); });

        check_remquo_result(lhs, rhs);
    }
}

TEST_CASE("f32 hyperbolic families match MPFR", "[fltx][f32][precision][transcendental][hyperbolic]")
{
    accuracy_report_scope report("f32 hyperbolic families match MPFR");

    constexpr std::array<float, 9> fixed_inputs{{
        -8.0,
        -2.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        2.0,
        4.0,
        8.0
    }};

    for (float input : fixed_inputs)
    {
        check_unary_op("sinh", input, exact_tol(),
            [](float x) { return bl::sinh(x); },
            [](float x) { return std::sinh(x); });

        check_unary_op("cosh", input, exact_tol(),
            [](float x) { return bl::cosh(x); },
            [](float x) { return std::cosh(x); });

        check_unary_op("tanh", input, exact_tol(),
            [](float x) { return bl::tanh(x); },
            [](float x) { return std::tanh(x); });

        check_unary_op("asinh", input, exact_tol(),
            [](float x) { return bl::asinh(x); },
            [](float x) { return std::asinh(x); });
    }

    constexpr std::array<float, 6> acosh_inputs{{ 1.0, 1.125, 1.5, 2.0, 10.0, 64.0 }};
    for (float input : acosh_inputs)
    {
        check_unary_op("acosh", input, exact_tol(),
            [](float x) { return bl::acosh(x); },
            [](float x) { return std::acosh(x); });
    }

    constexpr std::array<float, 8> atanh_inputs{{ -0.95, -0.5, -0.125, -0.0, 0.0, 0.125, 0.5, 0.95 }};
    for (float input : atanh_inputs)
    {
        check_unary_op("atanh", input, exact_tol(),
            [](float x) { return bl::atanh(x); },
            [](float x) { return std::atanh(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random hyperbolic inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float moderate    = random_signed_interval(rng, 8.0);
        const float acosh_input = random_acosh_argument(rng);
        const float atanh_input = random_atanh_argument(rng);

        check_unary_op("sinh", moderate, exact_tol(),
            [](float x) { return bl::sinh(x); },
            [](float x) { return std::sinh(x); });

        check_unary_op("cosh", moderate, exact_tol(),
            [](float x) { return bl::cosh(x); },
            [](float x) { return std::cosh(x); });

        check_unary_op("tanh", moderate, exact_tol(),
            [](float x) { return bl::tanh(x); },
            [](float x) { return std::tanh(x); });

        check_unary_op("asinh", moderate, exact_tol(),
            [](float x) { return bl::asinh(x); },
            [](float x) { return std::asinh(x); });

        check_unary_op("acosh", acosh_input, exact_tol(),
            [](float x) { return bl::acosh(x); },
            [](float x) { return std::acosh(x); });

        check_unary_op("atanh", atanh_input, exact_tol(),
            [](float x) { return bl::atanh(x); },
            [](float x) { return std::atanh(x); });
    }
}

TEST_CASE("f32 special functions match MPFR", "[fltx][f32][precision][transcendental][special]")
{
    accuracy_report_scope report("f32 special functions match MPFR");

    constexpr std::array<float, 9> erf_inputs{{
        -4.0,
        -2.0,
        -1.0,
        -0.125,
        0.0,
        0.125,
        1.0,
        2.0,
        4.0
    }};

    for (float input : erf_inputs)
    {
        check_unary_op("erf", input, exact_tol(),
            [](float x) { return bl::erf(x); },
            [](float x) { return std::erf(x); });

        check_unary_op("erfc", input, exact_tol(),
            [](float x) { return bl::erfc(x); },
            [](float x) { return std::erfc(x); });
    }

    constexpr std::array<float, 7> gamma_inputs{{ 0.125, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0 }};
    for (float input : gamma_inputs)
    {
        check_unary_op("lgamma", input, exact_tol(),
            [](float x) { return bl::lgamma(x); },
            [](float x) { return std::lgamma(x); });

        check_unary_op("tgamma", input, exact_tol(),
            [](float x) { return bl::tgamma(x); },
            [](float x) { return std::tgamma(x); });
    }

    std::mt19937_64 rng(random_seed);
    print_random_run("random special-function inputs", random_sample_count);

    for (int i = 0; i < random_sample_count; ++i)
    {
        const float erf_input   = random_signed_interval(rng, 4.0);
        const float gamma_input = random_gamma_positive(rng);

        check_unary_op("erf", erf_input, exact_tol(),
            [](float x) { return bl::erf(x); },
            [](float x) { return std::erf(x); });

        check_unary_op("erfc", erf_input, exact_tol(),
            [](float x) { return bl::erfc(x); },
            [](float x) { return std::erfc(x); });

        check_unary_op("lgamma", gamma_input, exact_tol(),
            [](float x) { return bl::lgamma(x); },
            [](float x) { return std::lgamma(x); });

        check_unary_op("tgamma", gamma_input, exact_tol(),
            [](float x) { return bl::tgamma(x); },
            [](float x) { return std::tgamma(x); });
    }
}

TEST_CASE("f32 decomposition and stepping functions match reference semantics", "[fltx][f32][precision][math][decomposition]")
{
    accuracy_report_scope report("f32 decomposition and stepping functions match reference semantics");

    constexpr std::array<float, 10> inputs{{
        -1e30f,
        -3.5,
        -0.5,
        -0.0,
        0.0,
        0.5,
        3.5,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        1e30f
    }};

    for (float input : inputs)
    {
        check_unary_op("fabs", input, exact_tol(),
            [](float x) { return bl::fabs(x); },
            [](float x) { return std::fabs(x); });

        check_unary_op("abs", input, exact_tol(),
            [](float x) { return bl::abs(x); },
            [](float x) { return std::fabs(x); });

        check_unary_op("logb", input, exact_tol(),
            [](float x) { return bl::logb(x); },
            [](float x) { return std::logb(x); });

        check_frexp_result(input);
        check_modf_result(input);

        check_exact_integer_result("ilogb", bl::ilogb(input), std::ilogb(input), input, 0.0);
        REQUIRE(bl::signbit(input) == std::signbit(input));
        REQUIRE(bl::isnan(input) == std::isnan(input));
        REQUIRE(bl::isinf(input) == std::isinf(input));
        REQUIRE(bl::isfinite(input) == std::isfinite(input));
        REQUIRE(bl::fpclassify(input) == std::fpclassify(input));
        REQUIRE(bl::isnormal(input) == std::isnormal(input));
    }

    constexpr std::array<std::pair<float, float>, 8> pairs{{
        { -0.0f, 0.0f },
        { 0.0f, -0.0f },
        { 1.0f, 2.0f },
        { 2.0f, 1.0f },
        { -1.0f, 1.0f },
        { std::numeric_limits<float>::denorm_min(), 0.0f },
        { 0.0f, std::numeric_limits<float>::denorm_min() },
        { (f32)std::numbers::pi, (f32)-std::numbers::e }
    }};

    for (const auto& [lhs, rhs] : pairs)
    {
        check_unary_op("nextafter", lhs, exact_tol(),
            [rhs](float x) { return bl::nextafter(x, rhs); },
            [rhs](float x) { return std::nextafter(x, rhs); });

        check_unary_op("nexttoward", lhs, exact_tol(),
            [rhs](float x) { return bl::nexttoward(x, static_cast<long double>(rhs)); },
            [rhs](float x) { return std::nexttoward(x, static_cast<long double>(rhs)); });

        check_exact_bool_result("isunordered", bl::isunordered(lhs, rhs), std::isunordered(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isgreater", bl::isgreater(lhs, rhs), std::isgreater(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isgreaterequal", bl::isgreaterequal(lhs, rhs), std::isgreaterequal(lhs, rhs), lhs, rhs);
        check_exact_bool_result("isless", bl::isless(lhs, rhs), std::isless(lhs, rhs), lhs, rhs);
        check_exact_bool_result("islessequal", bl::islessequal(lhs, rhs), std::islessequal(lhs, rhs), lhs, rhs);
        check_exact_bool_result("islessgreater", bl::islessgreater(lhs, rhs), std::islessgreater(lhs, rhs), lhs, rhs);
    }

    constexpr std::array<std::pair<float, int>, 8> ldexp_cases{{
        { 0.0f, 0 },
        { -0.0f, 3 },
        { 0.5f, 1 },
        { -0.5f, 1 },
        { 1.0f, -10 },
        { (f32)std::numbers::pi, 7 },
        { min_subnormal, 64 },
        { 1e30f, -64 }
    }};

    for (const auto& [value, exponent] : ldexp_cases)
    {
        check_scaled_result("ldexp", value, exponent, bl::ldexp(value, exponent));
        check_scaled_result("scalbn", value, exponent, bl::scalbn(value, exponent));
        check_scaled_result("scalbln", value, exponent, bl::scalbln(value, exponent));
    }
}

TEST_CASE("f32 utility helpers match reference semantics", "[fltx][f32][precision][math][utility]")
{
    accuracy_report_scope report("f32 utility helpers match reference semantics");

    constexpr std::array<std::pair<float, float>, 10> pairs{{
        { -0.0f, 0.0f },
        { 0.0f, -0.0f },
        { 1.0f, 2.0f },
        { 2.0f, 1.0f },
        { -2.0f, 3.0f },
        { 3.0f, -2.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e },
        { (f32)-std::numbers::pi, (f32)std::numbers::e },
        { 1e30f, min_subnormal },
        { min_subnormal, 1e30f }
    }};

    for (const auto& [lhs, rhs] : pairs)
    {
        check_binary_op("fmin", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmin(a, b); },
            [](float a, float b) { return std::fmin(a, b); });

        check_binary_op("fmax", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fmax(a, b); },
            [](float a, float b) { return std::fmax(a, b); });

        check_binary_op("fdim", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::fdim(a, b); },
            [](float a, float b) { return std::fdim(a, b); });

        check_binary_op("copysign", lhs, rhs, exact_tol(),
            [](float a, float b) { return bl::copysign(a, b); },
            [](float a, float b) { return std::copysign(a, b); });
    }

    constexpr std::array<std::tuple<float, float, float>, 6> fma_cases{{
        { 2.0f, 3.0f, 4.0f },
        { (f32)std::numbers::pi, (f32)std::numbers::e, (f32)-std::numbers::sqrt2 },
        { -3.0f, 4.0f, 5.0f },
        { 0.5f, -0.25f, 0.125f },
        { 6.0f, -7.0f, 8.0f }
    }};

    for (const auto& [x, y, z] : fma_cases)
    {
        check_fma_result(x, y, z, exact_tol());
    }
}

TEST_CASE("f32 IEEE special values match reference semantics", "[fltx][f32][precision][math][edge]")
{
    accuracy_report_scope report("f32 IEEE special values match reference semantics");

    const float nan     = std::numeric_limits<float>::quiet_NaN();
    const float inf     = std::numeric_limits<float>::infinity();
    const float neg_inf = -inf;

    check_unary_op("sqrt.nan", nan, exact_tol(),
        [](float x) { return bl::sqrt(x); },
        [](float x) { return std::sqrt(x); });

    check_unary_op("log.zero", 0.0f, exact_tol(),
        [](float x) { return bl::log(x); },
        [](float x) { return std::log(x); });

    check_unary_op("exp.neg_inf", neg_inf, exact_tol(),
        [](float x) { return bl::exp(x); },
        [](float x) { return std::exp(x); });

    check_binary_op("pow.neg_fractional", -1.0f, 0.5f, exact_tol(),
        [](float x, float y) { return bl::pow(x, y); },
        [](float x, float y) { return std::pow(x, y); });

    check_binary_op("fmod.inf", inf, 1.0f, exact_tol(),
        [](float x, float y) { return bl::fmod(x, y); },
        [](float x, float y) { return std::fmod(x, y); });

    check_binary_op("remainder.inf", inf, 1.0f, exact_tol(),
        [](float x, float y) { return bl::remainder(x, y); },
        [](float x, float y) { return std::remainder(x, y); });

    check_binary_op("fmin.nan_left", nan, 1.0f, exact_tol(),
        [](float x, float y) { return bl::fmin(x, y); },
        [](float x, float y) { return std::fmin(x, y); });

    check_binary_op("fmax.nan_right", 1.0f, nan, exact_tol(),
        [](float x, float y) { return bl::fmax(x, y); },
        [](float x, float y) { return std::fmax(x, y); });

    check_binary_op("copysign.neg_zero", 0.0f, -0.0f, exact_tol(),
        [](float x, float y) { return bl::copysign(x, y); },
        [](float x, float y) { return std::copysign(x, y); });

    check_unary_op("nextafter.inf_to_zero", inf, exact_tol(),
        [](float x) { return bl::nextafter(x, 0.0f); },
        [](float x) { return std::nextafter(x, 0.0f); });

    check_unary_op("nextafter.zero_to_negative", 0.0f, exact_tol(),
        [](float x) { return bl::nextafter(x, -1.0f); },
        [](float x) { return std::nextafter(x, -1.0f); });

    int got_quo                                     = 123;
    int expected_quo                                = 456;
    const float got_remquo                          = bl::remquo(inf, 1.0f, &got_quo);
    const float expected_remquo                     = std::remquo(inf, 1.0f, &expected_quo);
    const floating_compare_result remquo_comparison = compare_floating_result("remquo.inf", got_remquo, expected_remquo, exact_tol());
    INFO(build_comparison_message("remquo.inf", got_remquo, expected_remquo, exact_tol(), remquo_comparison));
    REQUIRE(remquo_comparison.passed);

    check_exact_bool_result("isunordered.nan", bl::isunordered(nan, 1.0f), std::isunordered(nan, 1.0f), nan, 1.0f);
    check_exact_bool_result("isless.nan", bl::isless(nan, 1.0f), std::isless(nan, 1.0f), nan, 1.0f);
    check_exact_bool_result("isgreater.inf", bl::isgreater(inf, 1.0f), std::isgreater(inf, 1.0f), inf, 1.0f);
    check_exact_bool_result("islessequal.neg_inf", bl::islessequal(neg_inf, 1.0f), std::islessequal(neg_inf, 1.0f), neg_inf, 1.0f);
}
