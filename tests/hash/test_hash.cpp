#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fltx.h>

namespace
{
    template<class T>
    [[nodiscard]] std::size_t hash_value(const T& value) noexcept
    {
        return std::hash<T>{}(value);
    }

    template<class T>
    [[nodiscard]] constexpr bool hash_returns_size_t() noexcept
    {
        return std::is_same_v<decltype(std::hash<T>{}(std::declval<const T&>())), std::size_t>;
    }

    static_assert(hash_returns_size_t<bl::f128_s>());
    static_assert(hash_returns_size_t<bl::f128>());
    static_assert(hash_returns_size_t<bl::f256_s>());
    static_assert(hash_returns_size_t<bl::f256>());

    static_assert(noexcept(std::hash<bl::f128_s>{}(std::declval<const bl::f128_s&>())));
    static_assert(noexcept(std::hash<bl::f128>{}(std::declval<const bl::f128&>())));
    static_assert(noexcept(std::hash<bl::f256_s>{}(std::declval<const bl::f256_s&>())));
    static_assert(noexcept(std::hash<bl::f256>{}(std::declval<const bl::f256&>())));

} // namespace

TEST_CASE("fltx hash specializations normalize signed zero", "[fltx][hash]")
{
    const bl::f128_s f128_zero_a{ 0.0, -0.0 };
    const bl::f128_s f128_zero_b{ -0.0, 0.0 };
    REQUIRE(f128_zero_a == f128_zero_b);
    REQUIRE(hash_value(f128_zero_a) == hash_value(f128_zero_b));
    REQUIRE(hash_value(bl::f128{ f128_zero_a }) == hash_value(bl::f128{ f128_zero_b }));

    const bl::f256_s f256_zero_a{ 0.0, -0.0, 0.0, -0.0 };
    const bl::f256_s f256_zero_b{ -0.0, 0.0, -0.0, 0.0 };
    REQUIRE(f256_zero_a == f256_zero_b);
    REQUIRE(hash_value(f256_zero_a) == hash_value(f256_zero_b));
    REQUIRE(hash_value(bl::f256{ f256_zero_a }) == hash_value(bl::f256{ f256_zero_b }));

    std::unordered_set<bl::f128_s> f128_zeros;
    f128_zeros.insert(f128_zero_a);
    f128_zeros.insert(f128_zero_b);
    REQUIRE(f128_zeros.size() == 1u);

    std::unordered_set<bl::f256_s> f256_zeros;
    f256_zeros.insert(f256_zero_a);
    f256_zeros.insert(f256_zero_b);
    REQUIRE(f256_zeros.size() == 1u);
}

TEST_CASE("fltx hash specializations include all precision limbs", "[fltx][hash]")
{
    const bl::f128_s f128_base{ 1.0, 0.0 };
    const bl::f128_s f128_low_changed{ 1.0, 0x1p-80 };
    REQUIRE(f128_base != f128_low_changed);
    REQUIRE(hash_value(f128_base) != hash_value(f128_low_changed));
    REQUIRE(hash_value(bl::f128{ f128_base }) != hash_value(bl::f128{ f128_low_changed }));

    const bl::f256_s f256_base{ 1.0, 0.0, 0.0, 0.0 };
    const bl::f256_s f256_x1_changed{ 1.0, 0x1p-80, 0.0, 0.0 };
    const bl::f256_s f256_x2_changed{ 1.0, 0.0, 0x1p-160, 0.0 };
    const bl::f256_s f256_x3_changed{ 1.0, 0.0, 0.0, 0x1p-240 };

    REQUIRE(f256_base != f256_x1_changed);
    REQUIRE(f256_base != f256_x2_changed);
    REQUIRE(f256_base != f256_x3_changed);
    REQUIRE(hash_value(f256_base) != hash_value(f256_x1_changed));
    REQUIRE(hash_value(f256_base) != hash_value(f256_x2_changed));
    REQUIRE(hash_value(f256_base) != hash_value(f256_x3_changed));
    REQUIRE(hash_value(bl::f256{ f256_base }) != hash_value(bl::f256{ f256_x1_changed }));
    REQUIRE(hash_value(bl::f256{ f256_base }) != hash_value(bl::f256{ f256_x2_changed }));
    REQUIRE(hash_value(bl::f256{ f256_base }) != hash_value(bl::f256{ f256_x3_changed }));
}

TEST_CASE("fltx storage and value hashes match for the same representation", "[fltx][hash]")
{
    const bl::f128_s f128_storage{ -2.0, 0x1p-90 };
    const bl::f128 f128_value{ f128_storage };
    REQUIRE(hash_value(f128_storage) == hash_value(f128_value));

    const bl::f256_s f256_storage{ -2.0, 0x1p-90, -0x1p-170, 0x1p-240 };
    const bl::f256 f256_value{ f256_storage };
    REQUIRE(hash_value(f256_storage) == hash_value(f256_value));
}

TEST_CASE("fltx hash specializations support unordered maps and sets", "[fltx][hash]")
{
    const bl::f128_s f128_storage_key{ 3.0, 0x1p-82 };
    std::unordered_map<bl::f128_s, int> f128_storage_values;
    f128_storage_values.emplace(f128_storage_key, 128);
    REQUIRE(f128_storage_values.find(f128_storage_key) != f128_storage_values.end());
    REQUIRE(f128_storage_values.at(f128_storage_key) == 128);

    const bl::f128 f128_value_key{ -3.0, 0x1p-82 };
    std::unordered_map<bl::f128, std::string_view> f128_values;
    f128_values.emplace(f128_value_key, "f128");
    REQUIRE(f128_values.find(f128_value_key) != f128_values.end());
    REQUIRE(f128_values.at(f128_value_key) == "f128");

    const bl::f256_s f256_storage_key{ 5.0, 0x1p-80, -0x1p-160, 0x1p-240 };
    std::unordered_set<bl::f256_s> f256_storage_values;
    f256_storage_values.insert(f256_storage_key);
    REQUIRE(f256_storage_values.contains(f256_storage_key));
    REQUIRE_FALSE(f256_storage_values.contains(bl::f256_s{ 5.0, 0x1p-80, -0x1p-160, -0x1p-240 }));

    const bl::f256 f256_value_key{ -5.0, 0x1p-80, -0x1p-160, 0x1p-240 };
    std::unordered_set<bl::f256> f256_values;
    f256_values.insert(f256_value_key);
    REQUIRE(f256_values.contains(f256_value_key));
    REQUIRE_FALSE(f256_values.contains(bl::f256{ -5.0, 0x1p-80, -0x1p-160, -0x1p-240 }));
}

TEST_CASE("fltx hash specializations are deterministic for special values", "[fltx][hash]")
{
    const bl::f128 f128_inf = std::numeric_limits<bl::f128>::infinity();
    const bl::f128 f128_nan = std::numeric_limits<bl::f128>::quiet_NaN();
    REQUIRE(hash_value(f128_inf) == hash_value(f128_inf));
    REQUIRE(hash_value(f128_nan) == hash_value(f128_nan));

    const bl::f256 f256_inf = std::numeric_limits<bl::f256>::infinity();
    const bl::f256 f256_nan = std::numeric_limits<bl::f256>::quiet_NaN();
    REQUIRE(hash_value(f256_inf) == hash_value(f256_inf));
    REQUIRE(hash_value(f256_nan) == hash_value(f256_nan));
}
