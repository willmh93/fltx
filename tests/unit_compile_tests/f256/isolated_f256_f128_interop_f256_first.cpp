#include <f256.h>
#include <f128.h>

namespace
{
    [[nodiscard]] constexpr bool check_f256_first_interop()
    {
        bl::f128_s narrow{ 1.0, 0x1p-60 };
        bl::f256 wide{ narrow };
        bl::f256_s difference = wide - bl::f128_s{ 0.25, 0.0 };
        bl::f256_s quotient = narrow / wide;
        wide *= narrow;

        const bl::f128_s roundtrip = static_cast<bl::f128_s>(wide);
        return difference.x0 != 0.0 &&
               quotient.x0 != 0.0 &&
               roundtrip.hi != 0.0;
    }

    static_assert(check_f256_first_interop());
}

void isolated_f256_f128_interop_f256_first()
{
}
