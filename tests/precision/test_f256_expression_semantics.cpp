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

using f256_prod_expr = std::remove_cvref_t<decltype(std::declval<f256>() * std::declval<f256>())>;
static_assert(std::is_convertible_v<f256_prod_expr, f256>);
static_assert(!std::is_convertible_v<f256_prod_expr&, f256>);
static_assert(!std::is_constructible_v<f256, f256_prod_expr&>);
static_assert(!std::is_constructible_v<f256, const f256_prod_expr&&>);
static_assert(std::is_constructible_v<f256, f256_prod_expr&&>);
using f256_prod_plus_value_sub_prod_expr = std::remove_cvref_t<decltype(
    (std::declval<f256>() * std::declval<f256>()) +
    (std::declval<f256>() - (std::declval<f256>() * std::declval<f256>())))>;
using f256_value_sub_prod_plus_prod_expr = std::remove_cvref_t<decltype(
    (std::declval<f256>() - (std::declval<f256>() * std::declval<f256>())) +
    (std::declval<f256>() * std::declval<f256>()))>;
using f256_prod_minus_value_sub_prod_expr = std::remove_cvref_t<decltype(
    (std::declval<f256>() * std::declval<f256>()) -
    (std::declval<f256>() - (std::declval<f256>() * std::declval<f256>())))>;
static_assert(detail::_f256_expr::is_prod_pair_value_v<f256_prod_plus_value_sub_prod_expr>);
static_assert(detail::_f256_expr::is_prod_pair_value_v<f256_value_sub_prod_plus_prod_expr>);
static_assert(detail::_f256_expr::is_prod_pair_value_v<f256_prod_minus_value_sub_prod_expr>);

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
    auto expression = make_returned_basic_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed = std::move(expression);
    const f256 immediate = make_materialized_basic_expression();

    require_same_value_bits(delayed, immediate);
}

TEST_CASE("f256 delayed expressions preserve math temporary semantics", "[fltx][f256][precision][arithmetic][expressions][semantics]")
{
    auto expression = make_returned_log_minus_input_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed = std::move(expression);
    const f256 immediate = make_materialized_log_minus_input_expression();

    require_same_value_bits(delayed, immediate);
}

TEST_CASE("f256 delayed product expressions preserve simple multi-line value semantics", "[fltx][f256][precision][arithmetic][expressions][semantics][fusion]")
{
    const f256 x = make_f256("1.1250000000000000000000000000000001");
    const f256 y = make_f256("-0.8750000000000000000000000000000001");
    const f256 a = make_f256("0.3333333333333333333333333333333333");

    const mpfr_ref x_ref = to_ref_exact(x);
    const mpfr_ref y_ref = to_ref_exact(y);
    const mpfr_ref a_ref = to_ref_exact(a);

    auto xy0 = x * y;
    auto xy1 = x * y;
    const f256 doubled = std::move(xy0) + std::move(xy1);

    auto xy2 = x * y;
    auto xy3 = x * y;
    const f256 doubled_plus = a + (std::move(xy2) + std::move(xy3));

    auto xy4 = x * y;
    auto xy5 = x * y;
    const f256 associated_plus = (std::move(xy4) + a) + std::move(xy5);

    auto xy6 = x * y;
    auto xy7 = x * y;
    const f256 associated_minus = (std::move(xy6) - a) + std::move(xy7);

    auto xx0 = x * x;
    auto yy0 = y * y;
    const f256 commuted_difference = a + (std::move(xx0) - std::move(yy0));

    auto xy8 = x * y;
    auto xx1 = x * x;
    auto yy1 = y * y;
    const f256 product_reassociated = std::move(xy8) + (std::move(xx1) + std::move(yy1));

    auto xx2 = x * x;
    auto yy2 = y * y;
    const f256 product_plus_value_sub_product = std::move(xx2) + (a - std::move(yy2));

    auto xx3 = x * x;
    auto yy3 = y * y;
    const f256 value_sub_product_plus_product = (a - std::move(xx3)) + std::move(yy3);

    auto xx4 = x * x;
    auto yy4 = y * y;
    const f256 product_minus_value_sub_product = std::move(xx4) - (a - std::move(yy4));

    const f256 scaled_right_associated = (x * 1.25) + (a + y * -0.75);
    const f256 scaled_value_sub_product = (x * 1.25) + (a - y * 0.75);
    const f256 scaled_sub_associated = (x * 1.25) - (a + y * 0.75);
    const f256 scaled_left_associated = (a + x * 1.25) - (y * 0.75);

    require_close_to_reference(doubled, x_ref * y_ref + x_ref * y_ref);
    require_close_to_reference(doubled_plus, a_ref + (x_ref * y_ref + x_ref * y_ref));
    require_close_to_reference(associated_plus, (x_ref * y_ref + a_ref) + x_ref * y_ref);
    require_close_to_reference(associated_minus, (x_ref * y_ref - a_ref) + x_ref * y_ref);
    require_close_to_reference(commuted_difference, a_ref + (x_ref * x_ref - y_ref * y_ref));
    require_close_to_reference(product_reassociated, x_ref * y_ref + (x_ref * x_ref + y_ref * y_ref));
    require_close_to_reference(product_plus_value_sub_product, x_ref * x_ref + (a_ref - y_ref * y_ref));
    require_close_to_reference(value_sub_product_plus_product, (a_ref - x_ref * x_ref) + y_ref * y_ref);
    require_close_to_reference(product_minus_value_sub_product, x_ref * x_ref - (a_ref - y_ref * y_ref));
    require_close_to_reference(scaled_right_associated, x_ref * mpfr_ref{ 1.25 } + (a_ref + y_ref * mpfr_ref{ -0.75 }));
    require_close_to_reference(scaled_value_sub_product, x_ref * mpfr_ref{ 1.25 } + (a_ref - y_ref * mpfr_ref{ 0.75 }));
    require_close_to_reference(scaled_sub_associated, x_ref * mpfr_ref{ 1.25 } - (a_ref + y_ref * mpfr_ref{ 0.75 }));
    require_close_to_reference(scaled_left_associated, (a_ref + x_ref * mpfr_ref{ 1.25 }) - y_ref * mpfr_ref{ 0.75 });
}

TEST_CASE("f256 named expressions require explicit consumption", "[fltx][f256][precision][arithmetic][expressions][semantics][fusion]")
{
    const f256 x = make_f256("1.1250000000000000000000000000000001");
    const f256 y = make_f256("-0.8750000000000000000000000000000001");
    const f256 cx = make_f256("0.3333333333333333333333333333333333");

    auto xx = x * x - y * y + cx;
    const f256 assigned = std::move(xx);
    const f256 direct = x * x - y * y + cx;

    require_same_value_bits(assigned, direct);
}

TEST_CASE("f256_s storage member arithmetic preserves normal value semantics", "[fltx][f256][precision][arithmetic][expressions][semantics][storage]")
{
    auto value_for_f256 = make_returned_storage_member_expression();
    auto value_for_storage = make_returned_storage_member_expression();
    clobber_stack_between_expression_creation_and_evaluation();

    const f256 delayed_value = std::move(value_for_f256);
    const f256 delayed_storage_value = std::move(value_for_storage);
    const f256_s delayed_storage = delayed_storage_value;
    const f256 immediate = make_materialized_storage_member_expression();

    require_same_value_bits(delayed_value, immediate);
    require_same_value_bits(delayed_storage, immediate);
    require_close_to_reference(delayed_storage, storage_member_expression_reference());
}
