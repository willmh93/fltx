#include <f128.h>
#include <f256.h>

namespace
{
    [[nodiscard]] constexpr bool check_f128_first_interop()
    {
        bl::f128_s narrow{ 1.0, 0x1p-60 };
        bl::f256 wide{ narrow };
        bl::f256_s sum = narrow + wide;
        bl::f256_s product = wide * narrow;
        wide += narrow;

        const bl::f128_s roundtrip = static_cast<bl::f128_s>(wide);
        return sum.x0 != 0.0 &&
               product.x0 != 0.0 &&
               roundtrip.hi != 0.0;
    }

    static_assert(check_f128_first_interop());
}

void isolated_f256_f128_interop_f128_first()
{
}
