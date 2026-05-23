/**
 * fltx/f128_numbers.h - std::numbers specializations for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_F128_NUMBERS_INCLUDED
#define FLTX_F128_NUMBERS_INCLUDED
#include <numbers>

#include "fltx/f128_type.h"

namespace std::numbers
{
    template<> inline constexpr bl::f128_s e_v<bl::f128_s>          = { 0x1.5bf0a8b145769p+1,  0x1.4d57ee2b1013ap-53 };
    template<> inline constexpr bl::f128_s log2e_v<bl::f128_s>      = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56 };
    template<> inline constexpr bl::f128_s log10e_v<bl::f128_s>     = { 0x1.bcb7b1526e50ep-2,  0x1.95355baaafad3p-57 };
    template<> inline constexpr bl::f128_s pi_v<bl::f128_s>         = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53 };
    template<> inline constexpr bl::f128_s inv_pi_v<bl::f128_s>     = { 0x1.45f306dc9c883p-2, -0x1.6b01ec5417056p-56 };
    template<> inline constexpr bl::f128_s inv_sqrtpi_v<bl::f128_s> = { 0x1.20dd750429b6dp-1,  0x1.1ae3a914fed80p-57 };
    template<> inline constexpr bl::f128_s ln2_v<bl::f128_s>        = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56 };
    template<> inline constexpr bl::f128_s ln10_v<bl::f128_s>       = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53 };
    template<> inline constexpr bl::f128_s sqrt2_v<bl::f128_s>      = { 0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54 };
    template<> inline constexpr bl::f128_s sqrt3_v<bl::f128_s>      = { 0x1.bb67ae8584caap+0,  0x1.cec95d0b5c1e3p-54 };
    template<> inline constexpr bl::f128_s inv_sqrt3_v<bl::f128_s>  = { 0x1.279a74590331cp-1,  0x1.34863e0792bedp-55 };
    template<> inline constexpr bl::f128_s egamma_v<bl::f128_s>     = { 0x1.2788cfc6fb619p-1, -0x1.6cb90701fbfabp-58 };
    template<> inline constexpr bl::f128_s phi_v<bl::f128_s>        = { 0x1.9e3779b97f4a8p+0, -0x1.f506319fcfd19p-55 };

    template<> inline constexpr bl::f128 e_v<bl::f128>          = bl::f128{ e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 log2e_v<bl::f128>      = bl::f128{ log2e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 log10e_v<bl::f128>     = bl::f128{ log10e_v<bl::f128_s> };
    template<> inline constexpr bl::f128 pi_v<bl::f128>         = bl::f128{ pi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_pi_v<bl::f128>     = bl::f128{ inv_pi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_sqrtpi_v<bl::f128> = bl::f128{ inv_sqrtpi_v<bl::f128_s> };
    template<> inline constexpr bl::f128 ln2_v<bl::f128>        = bl::f128{ ln2_v<bl::f128_s> };
    template<> inline constexpr bl::f128 ln10_v<bl::f128>       = bl::f128{ ln10_v<bl::f128_s> };
    template<> inline constexpr bl::f128 sqrt2_v<bl::f128>      = bl::f128{ sqrt2_v<bl::f128_s> };
    template<> inline constexpr bl::f128 sqrt3_v<bl::f128>      = bl::f128{ sqrt3_v<bl::f128_s> };
    template<> inline constexpr bl::f128 inv_sqrt3_v<bl::f128>  = bl::f128{ inv_sqrt3_v<bl::f128_s> };
    template<> inline constexpr bl::f128 egamma_v<bl::f128>     = bl::f128{ egamma_v<bl::f128_s> };
    template<> inline constexpr bl::f128 phi_v<bl::f128>        = bl::f128{ phi_v<bl::f128_s> };

} // namespace std::numbers

#ifndef FLTX_NUMBERS_ALIAS_DEFINED
#define FLTX_NUMBERS_ALIAS_DEFINED
namespace bl
{
    namespace numbers = std::numbers;
} // namespace bl
#endif

#endif
