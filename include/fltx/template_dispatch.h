/**
 * fltx/template_dispatch.h - Runtime-to-template dispatch helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_TEMPLATE_DISPATCH_INCLUDED
#define FLTX_TEMPLATE_DISPATCH_INCLUDED
#include <array>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <functional>

#ifndef BL_FORCE_INLINE
#if defined(_MSC_VER)
#define BL_FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define BL_FORCE_INLINE inline __attribute__((always_inline))
#else
#define BL_FORCE_INLINE inline
#endif
#endif

/// Usage: free function / local method
///
///    bl_table_invoke(
///        bl_dispatch_table(testFunc, runtime_arg),
///        bl_enum_type(FloatType::F32),  // maps the enum value to a template type
///        true                           // maps the bool to a template non-type
///    );
///
///    bl_table_invoke<f32>(
///        bl_dispatch_table(testFunc, runtime_arg),
///        bl_enum_type(FloatType::F128)
///    );
///
/// Raw bool/enums dispatch as template non-type constants.
/// Use bl_enum_type(value) when an enum value maps to a type via bl_enum_type_map.
/// Use bl_type<T> when a template type is already known at compile time.
///
/// Usage: targeted method
///
///    bl_table_invoke(
///        bl_dispatch_table_memfn(obj, MyClass::testMethod, arg1, arg2),
///        bl_enum_type(template_arg1),
///        template_arg2
///    );

///

// -----------------------------------------------------------------------------
// dispatch domain traits
//
// by default, enums are treated as a dense domain [0, E::COUNT]
// you may specialize bl_enum_domain_traits<E> to
//   - shrink the domain (e.g. a subset)
//   - remap arbitrary enum values to a dense index
//
// this keeps "COUNT bigger than last declared enumerator" tricks working
// (e.g. bitmask enums), while allowing explicit shrinking elsewhere
// -----------------------------------------------------------------------------

namespace bl::detail
{
    template<class T>
    struct is_dispatch_arg : std::bool_constant<std::is_enum_v<T> ||
                                                std::is_same_v<std::remove_cv_t<T>, bool>> {};

    template<class T>
    inline constexpr bool is_dispatch_arg_v = is_dispatch_arg<T>::value;

} // namespace bl::detail

template<class E>
struct bl_enum_domain_traits
{
    static_assert(std::is_enum_v<E>, "bl_enum_domain_traits<E> requires enum E");

    static constexpr std::size_t size = static_cast<std::size_t>(E::COUNT);

    static BL_FORCE_INLINE constexpr std::size_t index(E v)
    {
        const auto i = static_cast<std::size_t>(v);
        #if !defined(NDEBUG)
        if (i >= size)
        {
            #if defined(_MSC_VER)
            __debugbreak();
            #else
            __builtin_trap();
            #endif
        }
        #endif
        return i;
    }

    static BL_FORCE_INLINE constexpr E value(std::size_t i)
    {
        #if !defined(NDEBUG)
        if (i >= size)
        {
            #if defined(_MSC_VER)
            __debugbreak();
            #else
            __builtin_trap();
            #endif
        }
        #endif
        return static_cast<E>(i);
    }
};

template<>
struct bl_enum_domain_traits<bool>
{
    static constexpr std::size_t size = 2;

    static constexpr std::size_t index(bool v)
    {
        return v ? 1u : 0u;
    }

    static constexpr bool value(std::size_t i)
    {
        return i != 0;
    }
};

// helper for sparse/subset domains
template<class E, E... Vs>
struct bl_enum_domain_values
{
    static_assert(std::is_enum_v<E>, "bl_enum_domain_values requires enum E");
    static constexpr std::size_t size = sizeof...(Vs);
    static constexpr std::array<E, size> values = { Vs... };

    static constexpr std::size_t index(E v)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            if (values[i] == v)
            {
                return i;
            }
        }
        #if defined(_MSC_VER)
        __debugbreak();
        #else
        __builtin_trap();
        #endif
        return 0;
    }

    static constexpr E value(std::size_t i)
    {
        #if !defined(NDEBUG)
        if (i >= size)
        {
            #if defined(_MSC_VER)
            __debugbreak();
            #else
            __builtin_trap();
            #endif
        }
        #endif
        return values[i];
    }
};

namespace bl::detail
{
    template<class T>
    inline constexpr std::size_t dispatch_domain_size_v = bl_enum_domain_traits<std::remove_cv_t<T>>::size;

    template<class... Ts>
    struct type_list {};

    template<class List, class T>
    struct list_push_back;

    template<class... Ts, class T>
    struct list_push_back<type_list<Ts...>, T>
    {
        using type = type_list<Ts..., T>;
    };


    template<class E>
    using default_constant_t = std::conditional_t<
        std::is_same_v<std::remove_cv_t<E>, bool>,
        std::integral_constant<bool, false>,
        std::integral_constant<std::remove_cv_t<E>, bl_enum_domain_traits<std::remove_cv_t<E>>::value(0)>
    >;

} // namespace bl::detail

// -----------------------------------------------------------------------------
// enum => type mapping
//
// specialize bl_enum_type_map<E, V> to map a specific enum value V to a type
// used by bl_enum_type(...)
// -----------------------------------------------------------------------------

template<class E, E V>
struct bl_enum_type_map;

template<class E, E V>
using bl_enum_type_map_t = typename bl_enum_type_map<E, V>::type;

// -----------------------------------------------------------------------------
// type selector tags
// -----------------------------------------------------------------------------

namespace bl::detail
{
    template<class T>
    struct type_tag { using type = T; };

    template<class T>
    inline constexpr type_tag<T> type{};

    template<class E>
    struct mapped_enum
    {
        static_assert(std::is_enum_v<E>, "mapped_enum requires enum type");
        E value{};
    };

    template<class E>
    BL_FORCE_INLINE constexpr mapped_enum<E> enum_type(E v)
    {
        return mapped_enum<E>{ v };
    }

    template<class T>
    struct is_type_tag : std::false_type {};

    template<class T>
    struct is_type_tag<type_tag<T>> : std::true_type {};

    template<class T>
    inline constexpr bool is_type_tag_v = is_type_tag<std::remove_cv_t<T>>::value;

    template<class T>
    struct is_mapped_enum : std::false_type {};

    template<class E>
    struct is_mapped_enum<mapped_enum<E>> : std::true_type {};

    template<class T>
    inline constexpr bool is_mapped_enum_v = is_mapped_enum<std::remove_cv_t<T>>::value;

} // namespace bl::detail

// -----------------------------------------------------------------------------
// enum_expandN maps runtime enum values to types using bl_enum_type_map + bl_enum_domain_traits
// -----------------------------------------------------------------------------

namespace bl::detail
{
    template<class E, class Cont, class TsList, class... More>
    struct enum_expand_step;

    template<class E, class Cont, class... Ts>
    struct enum_expand_step<E, Cont, type_list<Ts...>>
    {
        static BL_FORCE_INLINE decltype(auto) invoke(Cont& cont)
        {
            return cont.template operator()<Ts...>();
        }
    };

    template<class E, class Cont, class... Ts, class... More>
    struct enum_expand_step<E, Cont, type_list<Ts...>, E, More...>
    {
        using Ret = decltype(std::declval<Cont&>().template operator()<Ts..., bl_enum_type_map_t<E, bl_enum_domain_traits<E>::value(0)>>());
        using Fn = Ret(*)(Cont&, More...);

        template<std::size_t I>
        static BL_FORCE_INLINE Ret entry(Cont& cont, More... rest)
        {
            using Mapped = bl_enum_type_map_t<E, bl_enum_domain_traits<E>::value(I)>;
            return enum_expand_step<E, Cont, type_list<Ts..., Mapped>, More...>::invoke(cont, rest...);
        }

        template<std::size_t... Is>
        static consteval auto build(std::index_sequence<Is...>)
        {
            return std::array<Fn, bl_enum_domain_traits<E>::size>{ &entry<Is>... };
        }

        static constexpr auto table = build(std::make_index_sequence<bl_enum_domain_traits<E>::size>{});

        static BL_FORCE_INLINE decltype(auto) invoke(Cont& cont, E v0, More... rest)
        {
            return table[bl_enum_domain_traits<E>::index(v0)](cont, rest...);
        }
    };

    template<class E, class Cont, class TsList, class... More>
    BL_FORCE_INLINE decltype(auto) enum_expandN_impl(Cont& cont, More... xs)
    {
        return enum_expand_step<E, Cont, TsList, More...>::invoke(cont, xs...);
    }

    template<class E, class Cont, class... Ts, class... More>
    BL_FORCE_INLINE decltype(auto) enum_expandN(Cont& cont, More... xs)
    {
        static_assert(std::is_enum_v<E>, "E must be an enum");
        return enum_expandN_impl<E, Cont, type_list<Ts...>>(cont, xs...);
    }

} // namespace bl::detail

// -----------------------------------------------------------------------------
// stepwise 1-D dispatch for value domains (bool + enums)
// -----------------------------------------------------------------------------

namespace bl::detail
{
    template<class E, std::size_t I>
    using domain_constant_t = std::conditional_t<
        std::is_same_v<std::remove_cv_t<E>, bool>,
        std::integral_constant<bool, (I != 0)>,
        std::integral_constant<std::remove_cv_t<E>, bl_enum_domain_traits<std::remove_cv_t<E>>::value(I)>
    >;

    template<class Ret, class Fun, class TsList, class CsList, class... Es>
    struct step_dispatch;

    template<class Ret, class Fun, class... Ts, class... Cs>
    struct step_dispatch<Ret, Fun, type_list<Ts...>, type_list<Cs...>>
    {
        static BL_FORCE_INLINE Ret invoke(Fun& func)
        {
            if constexpr (std::is_void_v<Ret>)
            {
                func.template operator()<Ts...>(Cs{}...);
                return;
            }
            else
            {
                return func.template operator()<Ts...>(Cs{}...);
            }
        }
    };

    template<class Ret, class Fun, class TsList, class CsList, class First, class... Rest>
    struct step_dispatch<Ret, Fun, TsList, CsList, First, Rest...>
    {
        using FirstT = std::remove_cv_t<First>;
        using Fn = Ret(*)(Fun&, Rest...);

        template<std::size_t I>
        static BL_FORCE_INLINE Ret entry(Fun& f, Rest... rest)
        {
            using C = domain_constant_t<FirstT, I>;
            using NextCs = typename list_push_back<CsList, C>::type;
            return step_dispatch<Ret, Fun, TsList, NextCs, Rest...>::invoke(f, rest...);
        }

        template<std::size_t... Is>
        static consteval auto build(std::index_sequence<Is...>)
        {
            return std::array<Fn, bl_enum_domain_traits<FirstT>::size>{ &entry<Is>... };
        }

        static constexpr auto table = build(std::make_index_sequence<bl_enum_domain_traits<FirstT>::size>{});

        static BL_FORCE_INLINE Ret invoke(Fun& func, First v0, Rest... rest)
        {
            return table[bl_enum_domain_traits<FirstT>::index(v0)](func, rest...);
        }
    };

    template<typename... Ts, typename F, typename... Es>
        requires ((is_dispatch_arg_v<std::remove_cvref_t<Es>> && ...))
    decltype(auto) table_invoke_values(F& func, Es... vs)
    {
        using Fun = std::decay_t<F>;
        using RetProbe = decltype(func.template operator()<Ts...>(default_constant_t<std::remove_cvref_t<Es>>{}...));
        return step_dispatch<RetProbe, Fun, type_list<Ts...>, type_list<>, std::remove_cvref_t<Es>...>::invoke(
            func,
            static_cast<std::remove_cvref_t<Es>>(vs)...);
    }

} // namespace bl::detail

// -----------------------------------------------------------------------------
// public table_invoke parses leading type selectors, then dispatch domains
//
// supported leading type selectors
//   - bl_type<T>
//   - bl_enum_type(v)
//
// after the first non-type-selector, all remaining args must be dispatch args
// -----------------------------------------------------------------------------

namespace bl::detail
{
    template<class Fun, class TsList, class... Args>
    struct table_invoke_parse;

    template<class Fun, class... Ts>
    struct table_invoke_parse<Fun, type_list<Ts...>>
    {
        static BL_FORCE_INLINE decltype(auto) invoke(Fun& func)
        {
            return table_invoke_values<Ts...>(func);
        }
    };

    template<class Fun, class... Ts, class T, class... Rest>
    struct table_invoke_parse<Fun, type_list<Ts...>, bl::detail::type_tag<T>, Rest...>
    {
        static BL_FORCE_INLINE decltype(auto) invoke(Fun& func, bl::detail::type_tag<T>, Rest... rest)
        {
            return table_invoke_parse<Fun, type_list<Ts..., T>, Rest...>::invoke(func, rest...);
        }
    };

    template<class Fun, class... Ts, class E, class... Rest>
    struct table_invoke_parse<Fun, type_list<Ts...>, bl::detail::mapped_enum<E>, Rest...>
    {
        static BL_FORCE_INLINE decltype(auto) invoke(Fun& func, bl::detail::mapped_enum<E> me, Rest... rest)
        {
            auto cont = [&]<class... AllTypes>() -> decltype(auto) {
                return table_invoke_parse<Fun, type_list<AllTypes...>, Rest...>::invoke(func, rest...);
            };
            return enum_expandN<E, decltype(cont), Ts...>(cont, me.value);
        }
    };

    template<class Fun, class... Ts, class First, class... Rest>
    struct table_invoke_parse<Fun, type_list<Ts...>, First, Rest...>
    {
        static BL_FORCE_INLINE decltype(auto) invoke(Fun& func, First first, Rest... rest)
        {
            static_assert(is_dispatch_arg_v<std::remove_cvref_t<First>>, "type selectors must come before dispatch args");
            static_assert(((is_dispatch_arg_v<std::remove_cvref_t<Rest>> && ...)), "all trailing args must be dispatch args");
            return table_invoke_values<Ts...>(
                func,
                static_cast<std::remove_cvref_t<First>>(first),
                static_cast<std::remove_cvref_t<Rest>>(rest)...);
        }
    };

} // namespace bl::detail

template<typename... Ts, typename F, typename... Args>
BL_FORCE_INLINE decltype(auto) bl_table_invoke(F&& f, Args... args)
{
    using Fun = std::decay_t<F>;
    Fun& func = const_cast<Fun&>(static_cast<const Fun&>(f));
    return bl::detail::table_invoke_parse<Fun, bl::detail::type_list<Ts...>, Args...>::invoke(func, args...);
}

template<class T>
inline constexpr bl::detail::type_tag<T> bl_type{};

template<class E>
BL_FORCE_INLINE constexpr bl::detail::mapped_enum<E> bl_enum_type(E v)
{
    return bl::detail::enum_type(v);
}


// -----------------------------------------------------------------------------
// macro helpers
// -----------------------------------------------------------------------------

#define bl_dispatch_table(func, ...)                                                  \
    ([&] <typename... Ts>(auto... Cs) -> decltype(auto) {                             \
        return [&]<typename... Us>(std::tuple<Us...>*) {                              \
            if constexpr (sizeof...(Ts) == 0)                                         \
                return func<Us::value...>(__VA_ARGS__);                               \
            else                                                                      \
                return func<Ts..., Us::value...>(__VA_ARGS__);                        \
        }((std::tuple<decltype(Cs)...>*)nullptr);                                     \
    })

#define bl_dispatch_table_memfn(obj, method, ...)                                     \
    ([&] <typename... Ts>(auto... Cs) -> decltype(auto) {                             \
        auto&& _obj = (obj);                                                          \
        return [&]<typename... Us>(std::tuple<Us...>*) {                              \
            if constexpr (sizeof...(Ts) == 0) {                                       \
                return std::invoke(&method<Us::value...>, _obj, __VA_ARGS__);         \
            } else {                                                                  \
                return std::invoke(&method<Ts..., Us::value...>, _obj, __VA_ARGS__);  \
            }                                                                         \
        }((std::tuple<decltype(Cs)...>*)nullptr);                                     \
    })

#define bl_dispatch_table_callable(func, ...)                                         \
    ([&] <typename... Ts>(auto... Cs) -> decltype(auto) {                             \
        return [&]<typename... Us>(std::tuple<Us...>*) {                              \
            if constexpr (sizeof...(Ts) == 0)                                         \
                return func.template operator()<Us::value...>(__VA_ARGS__);           \
            else                                                                      \
                return func.template operator()<Ts..., Us::value...>(__VA_ARGS__);    \
        }((std::tuple<decltype(Cs)...>*)nullptr);                                     \
    })

#define bl_map_enum_to_type(EnumValue, T) \
    template<> struct bl_enum_type_map<decltype(EnumValue), EnumValue> { using type = T; }


// dispatch table size helpers
namespace bl::detail
{
    template<class T>
    struct mapped_underlying;

    template<class E>
    struct mapped_underlying<bl::detail::mapped_enum<E>> { using type = E; };

    template<class T>
    using mapped_underlying_t = typename mapped_underlying<std::remove_cv_t<T>>::type;

    template<class Arg>
    consteval std::size_t dispatch_arg_domain_size_type()
    {
        using A = std::remove_cvref_t<Arg>;

        if constexpr (is_type_tag_v<A>)
        {
            return 1;
        }
        else if constexpr (is_mapped_enum_v<A>)
        {
            using E = mapped_underlying_t<A>;
            return dispatch_domain_size_v<E>;
        }
        else if constexpr (is_dispatch_arg_v<A>)
        {
            return dispatch_domain_size_v<A>;
        }
        else
        {
            return 1;
        }
    }

    template<class... Args>
    consteval std::size_t dispatch_table_variant_count_types()
    {
        return (std::size_t{ 1 } * ... * dispatch_arg_domain_size_type<Args>());
    }

    template<class... Args>
    consteval std::array<std::size_t, sizeof...(Args)> dispatch_table_domain_sizes_types()
    {
        return { dispatch_arg_domain_size_type<Args>()... };
    }

} // namespace bl::detail

template<class Arg>
[[nodiscard]] constexpr std::size_t bl_dispatch_arg_domain_size(const Arg&) noexcept
{
    return bl::detail::dispatch_arg_domain_size_type<Arg>();
}

template<class Arg>
[[nodiscard]] constexpr std::size_t bl_enum_type_domain_size(const Arg& arg) noexcept
{
    return bl_dispatch_arg_domain_size(arg);
}

template<class... Args>
[[nodiscard]] constexpr std::size_t bl_dispatch_table_variant_count(const Args&...) noexcept
{
    return bl::detail::dispatch_table_variant_count_types<Args...>();
}

template<class... Args>
[[nodiscard]] constexpr std::size_t bl_table_variants_count(const Args&... args) noexcept
{
    return bl_dispatch_table_variant_count(args...);
}

template<class... Args>
[[nodiscard]] constexpr std::array<std::size_t, sizeof...(Args)> bl_dispatch_table_domain_sizes(const Args&...) noexcept
{
    return bl::detail::dispatch_table_domain_sizes_types<Args...>();
}

template<class... Args>
[[nodiscard]] std::string bl_dispatch_table_report(const char* label, const Args&... args)
{
    const auto sizes = bl_dispatch_table_domain_sizes(args...);
    const auto total = bl_dispatch_table_variant_count(args...);

    std::string report;
    report.reserve(48 + sizes.size() * 24);
    report += label ? label : "dispatch";
    report += ": ";
    report += std::to_string(total);
    report += " variants\n";

    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        report += "  arg[";
        report += std::to_string(i);
        report += "] domain = ";
        report += std::to_string(sizes[i]);
        report += "\n";
    }

    return report;
}

#endif
