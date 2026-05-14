#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <f256_math.h>
#include <f256_io.h>

using namespace bl;

static_assert(std::is_aggregate_v<f256_s>);
static_assert(std::is_standard_layout_v<f256_s>);
static_assert(std::is_trivially_copyable_v<f256_s>);

static_assert(!detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256_s>() + std::declval<f256_s>())>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256>() + std::declval<f256>())>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256>() - std::declval<f256>())>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256>() * std::declval<f256>())>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256>() / std::declval<f256>())>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(std::declval<f256>() + 0.5)>>::value);
static_assert(detail::_f256_expr::is_expr<std::remove_cvref_t<decltype(0.5 + std::declval<f256>())>>::value);
static_assert(std::is_convertible_v<decltype(std::declval<f256>() * std::declval<f256>() + 0.5), f256_s>);
static_assert(std::is_convertible_v<decltype(std::declval<f256>() * std::declval<f256>() + 0.5), f256>);

namespace
{
    using mpfr_ref = boost::multiprecision::number<
        boost::multiprecision::mpfr_float_backend<320>,
        boost::multiprecision::et_off>;

    struct storage_point
    {
        f256_s x;
        f256_s y;
    };

    struct storage_rect
    {
        storage_point a;
        storage_point b;
    };

    volatile double expression_semantics_sink = 0.0;

    [[nodiscard]] f256 make_f256(const char* text)
    {
        return to_f256(text);
    }

    [[nodiscard]] mpfr_ref to_ref_exact(const f256_s& value)
    {
        mpfr_ref sum = 0;
        sum += mpfr_ref{ value.x0 };
        sum += mpfr_ref{ value.x1 };
        sum += mpfr_ref{ value.x2 };
        sum += mpfr_ref{ value.x3 };
        return sum;
    }

    [[nodiscard]] std::string to_text(const f256_s& value)
    {
        return bl::to_string(value, std::numeric_limits<f256>::max_digits10, false, true, false);
    }

    [[nodiscard]] std::string to_text(const mpfr_ref& value)
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<f256>::max_digits10 + 20)
            << std::scientific
            << value;
        return out.str();
    }

    [[nodiscard]] bool same_double_bits(double lhs, double rhs) noexcept
    {
        return std::bit_cast<std::uint64_t>(lhs) == std::bit_cast<std::uint64_t>(rhs);
    }

    [[nodiscard]] bool same_value_bits(const f256_s& lhs, const f256_s& rhs) noexcept
    {
        return same_double_bits(lhs.x0, rhs.x0) &&
               same_double_bits(lhs.x1, rhs.x1) &&
               same_double_bits(lhs.x2, rhs.x2) &&
               same_double_bits(lhs.x3, rhs.x3);
    }

    void clobber_stack_between_expression_creation_and_evaluation()
    {
        std::array<f256_s, 64> scratch{};
        f256_s accumulator = make_f256("0.0625");

        for (std::size_t i = 0; i < scratch.size(); ++i)
        {
            const f256_s scale = make_f256((i & 1u) == 0u ? "1.0000000000000000000000000000000001" : "-0.9999999999999999999999999999999999");
            const f256_s offset = make_f256((i % 3u) == 0u ? "0.3333333333333333333333333333333333" : "-0.1250000000000000000000000000000001");
            accumulator = accumulator * scale + offset;
            scratch[i] = accumulator;
        }

        expression_semantics_sink += scratch.back().x0;
    }

    [[nodiscard]] auto make_returned_basic_expression()
    {
        const f256 a = make_f256("1.0000000000000000000000000000000001");
        const f256 b = make_f256("-0.9999999999999999999999999999999997");
        const f256 c = make_f256("0.3333333333333333333333333333333333");
        const f256 d = make_f256("-0.2500000000000000000000000000000001");
        const f256 e = make_f256("0.1250000000000000000000000000000003");

        return ((a * b) + (c * d) - e) / (a + 2.0);
    }

    [[nodiscard]] f256 make_materialized_basic_expression()
    {
        const f256 a = make_f256("1.0000000000000000000000000000000001");
        const f256 b = make_f256("-0.9999999999999999999999999999999997");
        const f256 c = make_f256("0.3333333333333333333333333333333333");
        const f256 d = make_f256("-0.2500000000000000000000000000000001");
        const f256 e = make_f256("0.1250000000000000000000000000000003");

        const f256 result = ((a * b) + (c * d) - e) / (a + 2.0);
        return result;
    }

    [[nodiscard]] auto make_returned_log_minus_input_expression()
    {
        const f256 z = make_f256("1.3125000000000000000000000000000001");
        return log(z) - z;
    }

    [[nodiscard]] f256 make_materialized_log_minus_input_expression()
    {
        const f256 z = make_f256("1.3125000000000000000000000000000001");
        const f256 result = log(z) - z;
        return result;
    }

    [[nodiscard]] storage_rect make_storage_rect()
    {
        return {
            {
                make_f256("0.8750000000000000000000000000000001"),
                make_f256("-1.1250000000000000000000000000000001")
            },
            {
                make_f256("1.3750000000000000000000000000000001"),
                make_f256("-1.6250000000000000000000000000000001")
            }
        };
    }

    template<class Rect>
    [[nodiscard]] auto storage_member_expression(const Rect& rect)
    {
        return ((rect.a.x + rect.a.y * 0.5) - (rect.b.x / 3.0)) +
               ((2.0 * rect.b.y) - (rect.a.x * rect.b.y)) / (rect.a.y + 1.25);
    }

    [[nodiscard]] auto make_returned_storage_member_expression()
    {
        const storage_rect rect = make_storage_rect();
        return storage_member_expression(rect);
    }

    [[nodiscard]] f256 make_materialized_storage_member_expression()
    {
        const storage_rect rect = make_storage_rect();
        const f256 result = storage_member_expression(rect);
        return result;
    }

    [[nodiscard]] mpfr_ref storage_member_expression_reference()
    {
        const storage_rect rect = make_storage_rect();
        const mpfr_ref ax = to_ref_exact(rect.a.x);
        const mpfr_ref ay = to_ref_exact(rect.a.y);
        const mpfr_ref bx = to_ref_exact(rect.b.x);
        const mpfr_ref by = to_ref_exact(rect.b.y);

        return ((ax + ay * mpfr_ref{ 0.5 }) - (bx / mpfr_ref{ 3.0 })) +
               ((mpfr_ref{ 2.0 } * by) - (ax * by)) / (ay + mpfr_ref{ 1.25 });
    }

    void require_same_value_bits(const f256_s& delayed, const f256_s& immediate)
    {
        CAPTURE(to_text(delayed));
        CAPTURE(to_text(immediate));
        REQUIRE(same_value_bits(delayed, immediate));
    }

    void require_close_to_reference(const f256_s& got, const mpfr_ref& expected)
    {
        const mpfr_ref got_ref = to_ref_exact(got);
        const mpfr_ref diff = got_ref > expected ? got_ref - expected : expected - got_ref;
        mpfr_ref scale = expected;
        if (scale < 0)
            scale = -scale;
        if (scale < 1)
            scale = 1;

        const mpfr_ref tolerance = mpfr_ref{ "2e-62" } * scale;

        CAPTURE(to_text(got));
        CAPTURE(to_text(expected));
        CAPTURE(to_text(diff));
        CAPTURE(to_text(tolerance));
        REQUIRE(diff <= tolerance);
    }
}

TEST_CASE("f256 delayed fused expressions preserve normal value semantics", "[fltx][f256][precision][arithmetic][expressions][semantics]")
{
    const auto expression = make_returned_basic_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed = expression;
    const f256 immediate = make_materialized_basic_expression();

    require_same_value_bits(delayed, immediate);
}

TEST_CASE("f256 delayed expressions preserve math temporary semantics", "[fltx][f256][precision][arithmetic][expressions][semantics]")
{
    const auto expression = make_returned_log_minus_input_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed = expression;
    const f256 immediate = make_materialized_log_minus_input_expression();

    require_same_value_bits(delayed, immediate);
}

TEST_CASE("f256 delayed product expressions normalize simple multi-line forms", "[fltx][f256][precision][arithmetic][expressions][semantics][fusion]")
{
    const f256 x = make_f256("1.1250000000000000000000000000000001");
    const f256 y = make_f256("-0.8750000000000000000000000000000001");
    const f256 a = make_f256("0.3333333333333333333333333333333333");

    const auto xy = x * y;
    const auto xx = x * x;
    const auto yy = y * y;

    const f256 doubled = xy + xy;
    const f256 doubled_plus = a + (xy + xy);
    const f256 associated_plus = (xy + a) + xy;
    const f256 associated_minus = (xy - a) + xy;
    const f256 commuted_difference = a + (xx - yy);
    const f256 product_reassociated = xy + (xx + yy);

    require_same_value_bits(doubled, detail::_f256_runtime::mul_twice(x, y));
    require_same_value_bits(doubled_plus, detail::_f256_runtime::mul_twice_add(x, y, a));
    require_same_value_bits(associated_plus, detail::_f256_runtime::mul_twice_add(x, y, a));
    require_same_value_bits(associated_minus, detail::_f256_runtime::mul_twice_sub(x, y, a));
    require_same_value_bits(commuted_difference, detail::_f256_runtime::sqr_sub_sqr_add(x, y, a));
    require_same_value_bits(product_reassociated, detail::_f256_runtime::mul_add_mul_add_mul(x, x, y, y, x, y));
}

TEST_CASE("f256_s storage member arithmetic preserves normal value semantics", "[fltx][f256][precision][arithmetic][expressions][semantics][storage]")
{
    const auto value = make_returned_storage_member_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed_value = value;
    const f256_s delayed_storage = value;
    const f256 immediate = make_materialized_storage_member_expression();

    require_same_value_bits(delayed_value, immediate);
    require_same_value_bits(delayed_storage, immediate);
    require_close_to_reference(delayed_storage, storage_member_expression_reference());
}
