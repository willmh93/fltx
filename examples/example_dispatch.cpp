#include <iostream>
#include <iomanip>

#include <fltx.h>
using namespace bl;
using namespace bl::literals;

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

    bl::table_invoke(
        bl::dispatch_table(foo, runtime_arg0, runtime_arg1),
        bl::enum_type(WideFloatType::F128)
    );
*/

template<typename T1, typename T2, bool Test_Bool>
void print_math(const char* name)
{
    // T1 = f32/f64/f128/f256
    constexpr T1 pi_v  = std::numbers::pi_v<T1>;
    constexpr T1 min_v = std::numeric_limits<T1>::min();
    constexpr T1 max_v = std::numeric_limits<T1>::max();

    constexpr T1 sin_pi_6 = bl::sin(pi_v / T1{ 6.0 });
    constexpr T1 sin_pi_5 = bl::sin(pi_v / T1{ 5.0 });

    std::cout << "print_math<" << name << ">():\n";

    std::cout << std::setprecision(std::numeric_limits<T1>::digits10);
    std::cout << "   " << name << " min = " << min_v << "\n";
    std::cout << "   " << name << " max = " << max_v << "\n";
    std::cout << "   " << name << " PI = " << pi_v << "\n";
    std::cout << "   " << "bl::sin(pi_v / 6.0) = " << sin_pi_6 << "\n";
    std::cout << "   " << "bl::sin(pi_v / 5.0) = " << sin_pi_5 << "\n";

    // T2 is fixed to f32 by the dispatch argument below in this example
    T2 c = static_cast<T2>(sin_pi_5);

    std::cout << std::setprecision(std::numeric_limits<T2>::digits10);
    std::cout << "   c: " << c << "\n";

    // test_bool is a runtime value at the call site, but dispatch selects a specialization
    // where Test_Bool is a compile-time bool
    if constexpr (Test_Bool)
    {
        std::cout << "   Test_Bool is TRUE\n";
    }

    std::cout << "\n";
}

int main()
{
    // -- direct invocation example --
    // print_math<f32, f32, true>("F32");


    // -- dynamic dispatch table invocation --
    for (int i = 0; i < (int)FloatType::COUNT; i++)
    {
        const char* name = FloatTypeNames[i];
        FloatType float_type = (FloatType)i;
        bool test_bool = (i % 2) == 0;

        // invokes print_math<T1, T2, Test_Bool>(name), selected from runtime dispatch values
        bl::table_invoke(
            bl::dispatch_table(print_math, name), // bind runtime args passed to print_math(...)
            bl::enum_type(float_type),            // map runtime enum value to T1
            bl::enum_type(FloatType::F32),        // map runtime enum value to T2
            test_bool                             // map runtime bool to Test_Bool
        );
    }

    // print dispatch table info by providing example args (4 * 4 * 2 = 32 variants)
    bl::dispatch_table_info::print_from_args("print_math",
        bl::enum_type(FloatType::F64), // domain size = 4
        bl::enum_type(FloatType::F32), // domain size = 4
        true                           // domain size = 2
    );
}

/* Output:


print_math<F32>():
   F32 min = 1.17549e-38
   F32 max = 3.40282e+38
   F32 PI = 3.14159
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785
   c: 0.587785
   Test_Bool is TRUE

print_math<F64>():
   F64 min = 2.2250738585072e-308
   F64 max = 1.79769313486232e+308
   F64 PI = 3.14159265358979
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785252292473
   c: 0.587785

print_math<F128>():
   F128 min = 2.225073858507201383090232717332e-308
   F128 max = 1.797693134862315708145274237317e+308
   F128 PI = 3.14159265358979323846264338328
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.5877852522924731291687059546391
   c: 0.587785
   Test_Bool is TRUE

print_math<F256>():
   F256 min = 2.22507385850720138309023271733240406421921598046233183055332742e-308
   F256 max = 1.79769313486231570814527423731704356798070567525844996598917477e+308
   F256 PI = 3.14159265358979323846264338327950288419716939937510582097494459
   bl::sin(pi_v / 6.0) = 0.5
   bl::sin(pi_v / 5.0) = 0.587785252292473129168705954639072768597652437643145991072272481
   c: 0.587785

print_math: 32 variants
  arg[0] domain=4
  arg[1] domain=4
  arg[2] domain=2

*/