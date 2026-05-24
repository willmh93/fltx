/**
 * fltx/f256_numbers.h - std::numbers specializations for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_NUMBERS_INCLUDED
#define F256_NUMBERS_INCLUDED
#include <numbers>

#include "fltx/f256_type.h"

namespace std::numbers
{
    template<> inline constexpr bl::f256_s e_v<bl::f256_s>          = { 0x1.5bf0a8b145769p+1,  0x1.4d57ee2b1013ap-53, -0x1.618713a31d3e2p-109,  0x1.c5a6d2b53c26dp-163 };
    template<> inline constexpr bl::f256_s log2e_v<bl::f256_s>      = { 0x1.71547652b82fep+0,  0x1.777d0ffda0d24p-56, -0x1.60bb8a5442ab9p-110, -0x1.4b52d3ba6d74dp-166 };
    template<> inline constexpr bl::f256_s log10e_v<bl::f256_s>     = { 0x1.bcb7b1526e50ep-2,  0x1.95355baaafad3p-57,  0x1.ee191f71a3012p-112,  0x1.7268808e8fcb5p-167 };
    template<> inline constexpr bl::f256_s pi_v<bl::f256_s>         = { 0x1.921fb54442d18p+1,  0x1.1a62633145c07p-53, -0x1.f1976b7ed8fbcp-109,  0x1.4cf98e804177dp-163 };
    template<> inline constexpr bl::f256_s inv_pi_v<bl::f256_s>     = { 0x1.45f306dc9c883p-2, -0x1.6b01ec5417056p-56, -0x1.6447e493ad4cep-110,  0x1.e21c820ff28b2p-164 };
    template<> inline constexpr bl::f256_s inv_sqrtpi_v<bl::f256_s> = { 0x1.20dd750429b6dp-1,  0x1.1ae3a914fed80p-57, -0x1.3cbbebf65f145p-112, -0x1.e0c574632f53ep-167 };
    template<> inline constexpr bl::f256_s ln2_v<bl::f256_s>        = { 0x1.62e42fefa39efp-1,  0x1.abc9e3b39803fp-56,  0x1.7b57a079a1934p-111, -0x1.ace93a4ebe5d1p-165 };
    template<> inline constexpr bl::f256_s ln10_v<bl::f256_s>       = { 0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53, -0x1.9ebae3ae0260cp-107, -0x1.2d10378be1cf1p-161 };
    template<> inline constexpr bl::f256_s sqrt2_v<bl::f256_s>      = { 0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54,  0x1.57d3e3adec175p-108,  0x1.2775099da2f59p-164 };
    template<> inline constexpr bl::f256_s sqrt3_v<bl::f256_s>      = { 0x1.bb67ae8584caap+0,  0x1.cec95d0b5c1e3p-54, -0x1.f11db689f2ccfp-110,  0x1.3da4798c720a6p-164 };
    template<> inline constexpr bl::f256_s inv_sqrt3_v<bl::f256_s>  = { 0x1.279a74590331cp-1,  0x1.34863e0792bedp-55, -0x1.a82f9e6c53222p-109, -0x1.cb0f41134253ap-163 };
    template<> inline constexpr bl::f256_s egamma_v<bl::f256_s>     = { 0x1.2788cfc6fb619p-1, -0x1.6cb90701fbfabp-58, -0x1.34a95e3133c51p-112,  0x1.9730064300f7dp-166 };
    template<> inline constexpr bl::f256_s phi_v<bl::f256_s>        = { 0x1.9e3779b97f4a8p+0, -0x1.f506319fcfd19p-55,  0x1.b906821044ed8p-109, -0x1.8bb1b5c0f272cp-165 };

    template<> inline constexpr bl::f256 e_v<bl::f256>          = bl::f256{ e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 log2e_v<bl::f256>      = bl::f256{ log2e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 log10e_v<bl::f256>     = bl::f256{ log10e_v<bl::f256_s> };
    template<> inline constexpr bl::f256 pi_v<bl::f256>         = bl::f256{ pi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_pi_v<bl::f256>     = bl::f256{ inv_pi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_sqrtpi_v<bl::f256> = bl::f256{ inv_sqrtpi_v<bl::f256_s> };
    template<> inline constexpr bl::f256 ln2_v<bl::f256>        = bl::f256{ ln2_v<bl::f256_s> };
    template<> inline constexpr bl::f256 ln10_v<bl::f256>       = bl::f256{ ln10_v<bl::f256_s> };
    template<> inline constexpr bl::f256 sqrt2_v<bl::f256>      = bl::f256{ sqrt2_v<bl::f256_s> };
    template<> inline constexpr bl::f256 sqrt3_v<bl::f256>      = bl::f256{ sqrt3_v<bl::f256_s> };
    template<> inline constexpr bl::f256 inv_sqrt3_v<bl::f256>  = bl::f256{ inv_sqrt3_v<bl::f256_s> };
    template<> inline constexpr bl::f256 egamma_v<bl::f256>     = bl::f256{ egamma_v<bl::f256_s> };
    template<> inline constexpr bl::f256 phi_v<bl::f256>        = bl::f256{ phi_v<bl::f256_s> };

} // namespace std::numbers

#ifndef FLTX_NUMBERS_ALIAS_DEFINED
#define FLTX_NUMBERS_ALIAS_DEFINED
namespace bl
{
    namespace numbers = std::numbers;
} // namespace bl
#endif

#endif
