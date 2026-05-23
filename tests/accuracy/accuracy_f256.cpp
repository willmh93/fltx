#include <fltx/f128.h>
#include <fltx/f256.h>

#include <catch2/catch_test_macros.hpp>

#include <limits>

namespace
{
using fltx_type = bl::f256;

template<class T>
[[nodiscard]] fltx_type materialize(T&& value)
{
    return fltx_type{ static_cast<T&&>(value) };
}

template<class T>
void require_exact_value(const char* label, T&& value, const fltx_type& expected)
{
    const fltx_type got = materialize(static_cast<T&&>(value));
    INFO(label);
    REQUIRE(got.x0 == expected.x0);
    REQUIRE(got.x1 == expected.x1);
    REQUIRE(got.x2 == expected.x2);
    REQUIRE(got.x3 == expected.x3);
}

template<class T>
void require_nan(T&& value)
{
    REQUIRE(bl::isnan(materialize(static_cast<T&&>(value))));
}

} // namespace

TEST_CASE("f256 arithmetic special values", "[accuracy][f256][arithmetic][special]")
{
    const fltx_type nan = std::numeric_limits<fltx_type>::quiet_NaN();
    const fltx_type inf = std::numeric_limits<fltx_type>::infinity();
    const fltx_type neg_inf = -inf;
    const fltx_type one{1.0};
    const fltx_type two{2.0};
    const fltx_type zero{0.0, 0.0, 0.0, 0.0};
    const fltx_type neg_zero{-0.0, 0.0, 0.0, 0.0};
    const fltx_type max = std::numeric_limits<fltx_type>::max();
    const fltx_type lowest = std::numeric_limits<fltx_type>::lowest();

    require_exact_value("arith.inf_add", inf + one, inf);
    require_exact_value("arith.neg_inf_add", neg_inf + one, neg_inf);
    require_exact_value("arith.inf_add_inf", inf + inf, inf);
    require_exact_value("arith.neg_inf_add_neg_inf", neg_inf + neg_inf, neg_inf);
    require_nan(nan + one);
    require_nan(one + nan);
    require_nan(nan + nan);
    require_nan(inf + neg_inf);

    require_nan(nan - one);
    require_nan(one - nan);
    require_nan(nan - nan);
    require_nan(inf - inf);
    require_exact_value("arith.inf_sub", inf - one, inf);
    require_exact_value("arith.sub_inf", one - inf, neg_inf);
    require_exact_value("arith.inf_sub_neg_inf", inf - neg_inf, inf);
    require_exact_value("arith.neg_inf_sub_inf", neg_inf - inf, neg_inf);

    require_exact_value("arith.inf_mul", inf * two, inf);
    require_exact_value("arith.inf_mul_neg", inf * -two, neg_inf);
    require_exact_value("arith.neg_inf_mul", neg_inf * two, neg_inf);
    require_exact_value("arith.neg_inf_mul_neg", neg_inf * -two, inf);
    require_nan(inf * zero);
    require_nan(zero * inf);
    require_nan(nan * one);
    require_nan(one * nan);
    require_nan(nan * nan);

    require_exact_value("arith.div_by_zero", one / zero, inf);
    require_exact_value("arith.neg_div_by_zero", -one / zero, neg_inf);
    require_exact_value("arith.neg_div_by_neg_zero", -one / neg_zero, inf);
    require_nan(nan / one);
    require_nan(one / nan);
    require_nan(nan / nan);
    require_nan(zero / zero);
    require_nan(inf / inf);
    require_exact_value("arith.inf_div", inf / two, inf);

    const fltx_type pos_zero = materialize(two / inf);
    REQUIRE(bl::iszero(pos_zero));
    REQUIRE(!bl::signbit(pos_zero));

    const fltx_type signed_zero = materialize(-two / inf);
    REQUIRE(bl::iszero(signed_zero));
    REQUIRE(bl::signbit(signed_zero));

    const fltx_type neg_zero_div_finite = materialize(neg_zero / two);
    REQUIRE(bl::iszero(neg_zero_div_finite));
    REQUIRE(bl::signbit(neg_zero_div_finite));

    const fltx_type zero_div_neg_finite = materialize(zero / -two);
    REQUIRE(bl::iszero(zero_div_neg_finite));
    REQUIRE(bl::signbit(zero_div_neg_finite));

    const fltx_type neg_zero_div_neg_finite = materialize(neg_zero / -two);
    REQUIRE(bl::iszero(neg_zero_div_neg_finite));
    REQUIRE(!bl::signbit(neg_zero_div_neg_finite));

    require_exact_value("arith.scalar_add_inf", one + std::numeric_limits<double>::infinity(), inf);
    require_exact_value("arith.scalar_sub_inf", one - std::numeric_limits<double>::infinity(), neg_inf);
    require_exact_value("arith.scalar_mul_inf", two * std::numeric_limits<double>::infinity(), inf);
    require_exact_value("arith.scalar_div_zero", one / 0.0, inf);
    require_nan(one + std::numeric_limits<double>::quiet_NaN());
    require_nan(one - std::numeric_limits<double>::quiet_NaN());
    require_nan(one * std::numeric_limits<double>::quiet_NaN());
    require_nan(one / std::numeric_limits<double>::quiet_NaN());

    require_exact_value("arith.add_overflow", max + max, inf);
    require_exact_value("arith.sub_overflow", lowest - max, neg_inf);
    require_exact_value("arith.mul_overflow", max * two, inf);
    require_exact_value("arith.neg_mul_overflow", lowest * two, neg_inf);
    require_exact_value("arith.div_overflow", max / fltx_type{0.5}, inf);
    require_exact_value("arith.neg_div_overflow", lowest / fltx_type{0.5}, neg_inf);

    const fltx_type first_limb_zero_with_tail{0.0, 1.0, 0.0, 0.0};
    require_nan(first_limb_zero_with_tail * inf);
    require_nan(first_limb_zero_with_tail / zero);

    const bl::f128 f128_nan = std::numeric_limits<bl::f128>::quiet_NaN();
    const bl::f128 f128_inf = std::numeric_limits<bl::f128>::infinity();
    const bl::f128 f128_zero{0.0};
    require_exact_value("interop.f128_add_inf", one + f128_inf, inf);
    require_exact_value("interop.f128_sub_inf", one - f128_inf, neg_inf);
    require_exact_value("interop.f128_mul_inf", two * f128_inf, inf);
    require_exact_value("interop.f128_div_zero", one / f128_zero, inf);
    require_nan(one + f128_nan);
    require_nan(one - f128_nan);
    require_nan(one * f128_nan);
    require_nan(one / f128_nan);
}
