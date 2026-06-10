#include <iostream>
#include <iomanip>
#include <string_view>

#include <fltx/core.h>
#include <fltx/dispatch.h>
#include <fltx/math.h>
#include <fltx/io.h>

using namespace bl;

/* Notes:

    - Dispatch tables grow multiplicatively with each dispatched template argument.
    - Large domains can significantly increase executable size and compile time.
    - Use dispatch tables sparingly, mainly where runtime performance matters.
    - Prefer the smallest domain needed for the use case.

Example:

    If only f128/f256 are needed, create a smaller enum and map those types manually:

    enum struct WideFloatType { F128, F256, COUNT };
    bl_map_enum_to_type(WideFloatType::F128, bl::f128);
    bl_map_enum_to_type(WideFloatType::F256, bl::f256);

    bl_table_invoke(
        bl_dispatch_table(foo, runtime_arg0, runtime_arg1),
        bl_enum_type(WideFloatType::F128)
    );
*/

template<typename T1, typename T2, bool Test_Bool>
void print_math(std::string_view name)
{
    // T1 can be: f32 / f64 / f128 / f256
    constexpr T1 pi_v  = std::numbers::pi_v<T1>;
    constexpr T1 min_v = std::numeric_limits<T1>::min();
    constexpr T1 max_v = std::numeric_limits<T1>::max();

    constexpr T1 sin_pi_6 = bl::sin(pi_v / 6);
    constexpr T1 sin_pi_5 = bl::sin(pi_v / 5);

    std::cout << "print_math<" << name << ">():\n";

    std::cout << std::setprecision(std::numeric_limits<T1>::digits10);
    std::cout << "  " << name << " min\t\t= "  << min_v << "\n";
    std::cout << "  " << name << " max\t\t= "  << max_v << "\n";
    std::cout << "  " << name << " pi\t\t= "   << pi_v << "\n";
    std::cout << "  " << "bl::sin(pi / 6)\t= " << sin_pi_6 << "\n";
    std::cout << "  " << "bl::sin(pi / 5)\t= " << sin_pi_5 << "\n";

    // T2 is fixed to f32 by the dispatch argument below in this example
    T2 c = static_cast<T2>(sin_pi_5);

    std::cout << std::setprecision(std::numeric_limits<T2>::digits10);
    std::cout << "  c\t\t\t= " << c << "\n";

    // test_bool is a runtime value at the call site, but dispatch selects a specialization
    // where Test_Bool is a compile-time bool
    if constexpr (Test_Bool)
    {
        std::cout << "  Test_Bool\t\t= TRUE\n";
    }

    std::cout << "\n";
}

int main()
{
    // loop over each FloatType value
    for (int i = 0; i < (int)FloatType::COUNT; i++)
    {
        // set runtime args
        FloatType float_type = (FloatType)i;
        auto name       = bl::to_string(float_type);
        bool test_bool  = (i % 2) == 0;

        // invokes matching print_math<T1, T2, Test_Bool>(name) selected from runtime values
        bl_table_invoke(
            bl_dispatch_table(print_math, name),  // bind runtime args to print_math(...)
            bl_enum_type(float_type),             // map runtime enum value to T1
            bl_enum_type(FloatType::F32),         // map runtime enum value to T2
            test_bool                             // map runtime bool to Test_Bool
        );
    }

    // generate an automatic dispatch table report:
    {
        std::cout << bl_dispatch_table_report("print_math", // print_math: 32 variants
            bl_enum_type(FloatType::F64),                   //   arg[0] domain = 4
            bl_enum_type(FloatType::F32),                   //   arg[1] domain = 4
            true                                            //   arg[2] domain = 2
        );
    }

    /*
    // manually calculate domain size of a single mapped type
    {
        constexpr std::size_t template_arg_domain_size = bl_enum_type_domain_size(FloatType::F64);
        std::cout << "FloatType template arg domain: " << template_arg_domain_size << "\n\n";
    }


    // manually calculate and print dispatch table domain info
    {
        constexpr std::size_t table_variant_count = bl_table_variants_count(
            bl_enum_type(FloatType::F64),
            bl_enum_type(FloatType::F32),
            true
        );
        constexpr auto table_domain_sizes = bl_dispatch_table_domain_sizes(
            bl_enum_type(FloatType::F64),
            bl_enum_type(FloatType::F32),
            true
        );

        std::cout << "print_math: " << table_variant_count << " variants\n";
        for (std::size_t i = 0; i < table_domain_sizes.size(); i++)
            std::cout << "  arg[" << i << "] domain = " << table_domain_sizes[i] << "\n";

        std::cout << "\n";
    }
    */
}

/* Output:


print_math<f32>():
 f32 min        = 1.17549e-38
 f32 max        = 3.40282e+38
 f32 PI         = 3.14159
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785
   c: 0.587785
   Test_Bool is TRUE

print_math<f64>():
 f64 min        = 2.2250738585072e-308
 f64 max        = 1.79769313486232e+308
 f64 PI         = 3.14159265358979
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785252292473
   c: 0.587785

print_math<f128>():
 f128 min        = 2.225073858507201383090232717332e-308
 f128 max        = 1.797693134862315708145274237317e+308
 f128 PI         = 3.14159265358979323846264338328
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.5877852522924731291687059546391
   c: 0.587785
   Test_Bool is TRUE

print_math<f256>():
 f256 min        = 2.22507385850720138309023271733240406421921598046233183055332742e-308
 f256 max        = 1.79769313486231570814527423731704356798070567525844996598917477e+308
 f256 PI         = 3.14159265358979323846264338327950288419716939937510582097494459
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785252292473129168705954639072768597652437643145991072272481
   c: 0.587785

print_math: 32 variants
  arg[0] domain = 4
  arg[1] domain = 4
  arg[2] domain = 2

*/
