/**
 * fltx/detail/f64_math_basic.h - constexpr <cmath>-style basic math helpers for f64.
 *
 * f64 rounding, decomposition, remainder, min/max, and adjacent-value helpers.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F64_MATH_BASIC_INCLUDED
#define F64_MATH_BASIC_INCLUDED

#include "fltx/f64_classification.h"
#include "fltx/detail/common_decimal.h"
#include "fltx/detail/native_float_decimal.h"
#include "fltx/round_options.h"


namespace bl {

namespace detail::_f64_impl
{
    using detail::fp::isnan;
    using detail::fp::isinf;
    using detail::fp::isfinite;
    using detail::fp::signbit;
    using detail::fp::fabs;
    using detail::fp::floor;
    using detail::fp::ceil;
    using detail::fp::trunc;
    using detail::fp::nearbyint_ties_even;
    using detail::fp::nearbyint;
    using detail::fp::round_half_away_zero;
    using detail::fp::nextafter;
    using detail::fp::copysign;
    using detail::fp::to_signed_integer_or_zero;
    using detail::fp::frexp_exponent;
    using detail::fp::ldexp;
    using detail::fp::bit_length_u64;
    using detail::fp::log;
    using detail::fp::log1p;
    using detail::fp::sin;
    using detail::fp::cos;
    using detail::fp::tan;
    using detail::fp::atan;
    using detail::fp::atan2;
    using detail::fp::sqrt_seed;

    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double pi_2 = 1.570796326794896619231321691639751442;
    constexpr double pi_4 = 0.785398163397448309615660845819875721;
    constexpr double ln2 = 0.693147180559945309417232121458176568;
    constexpr double inv_ln2 = 1.442695040888963407359924681001892137;
    constexpr double inv_ln10 = 0.434294481903251827651128918916605082;

    BL_FORCE_INLINE constexpr int ilogb_finite(double x) noexcept
    {
        return frexp_exponent(x) - 1;
    }

    BL_FORCE_INLINE constexpr double powi_nonneg(double base, std::uint64_t exp) noexcept
    {
        double result = 1.0;
        while (exp != 0)
        {
            if ((exp & 1u) != 0)
                result *= base;
            exp >>= 1u;
            if (exp != 0)
                base *= base;
        }
        return result;
    }

    BL_FORCE_INLINE constexpr double powi(double base, long long exp) noexcept
    {
        if (exp == 0)
            return 1.0;

        if (exp < 0)
            return 1.0 / powi_nonneg(base, static_cast<std::uint64_t>(-(exp + 1))) / base;

        return powi_nonneg(base, static_cast<std::uint64_t>(exp));
    }


    struct dyadic_u64
    {
        std::uint64_t coeff = 0;
        int exp2 = 0;
    };
    struct exact_divmod_result
    {
        std::uint64_t remainder   = 0;
        int remainder_exp2   = 0;
        std::uint64_t denominator = 0;
        int denominator_exp2 = 0;
        unsigned quotient_low_bits = 0;
        bool quotient_nonzero = false;
    };

    [[nodiscard]] BL_FORCE_INLINE constexpr dyadic_u64 decompose_normalized_abs(double value) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value) & 0x7fffffffffffffffull;
        const std::uint64_t frac = bits & ((std::uint64_t{ 1 } << 52) - 1u);
        const std::uint32_t exp_bits = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);

        dyadic_u64 out;
        if (exp_bits == 0)
        {
            out.coeff = frac;
            out.exp2 = -1074;
        }
        else
        {
            out.coeff = (std::uint64_t{ 1 } << 52) | frac;
            out.exp2 = static_cast<int>(exp_bits) - 1023 - 52;
        }

        const int bits_used = bit_length_u64(out.coeff);
        const int shift     = 53 - bits_used;
        out.coeff <<= shift;
        out.exp2 -= shift;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr int compare_scaled_u64(std::uint64_t a, int a_exp2, std::uint64_t b, int b_exp2) noexcept
    {
        if (a == 0 || b == 0)
            return (a != 0) - (b != 0);

        const int a_bits = bit_length_u64(a);
        const int b_bits = bit_length_u64(b);
        const int a_top  = a_exp2 + a_bits - 1;
        const int b_top  = b_exp2 + b_bits - 1;
        if (a_top < b_top) return -1;
        if (a_top > b_top) return 1;

        if (a_exp2 >= b_exp2)
            a <<= (a_exp2 - b_exp2);
        else
            b <<= (b_exp2 - a_exp2);

        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }

    using detail::_native_float_decimal::rounded_shr_u64;
    using detail::_native_float_decimal::exact_dyadic_to_double;
    using detail::_native_float_decimal::f64_decimal_round_traits;
    using detail::_native_float_decimal::exact_decimal_to_native_value;

    inline constexpr int pow10_f64_min_exponent = -323;
    inline constexpr int pow10_f64_max_exponent = 308;
    inline constexpr std::uint64_t pow10_f64_bits[] =
    {
        0x0000000000000002ull, 0x0000000000000014ull, 0x00000000000000caull, 0x00000000000007e8ull,
        0x0000000000004f10ull, 0x00000000000316a2ull, 0x00000000001ee257ull, 0x000000000134d761ull,
        0x000000000c1069cdull, 0x0000000078a42205ull, 0x00000004b6695433ull, 0x0000002f201d49fbull,
        0x000001d74124e3d1ull, 0x000012688b70e62bull, 0x0000b8157268fdafull, 0x000730d67819e8d2ull,
        0x0031fa182c40c60dull, 0x0066789e3750f791ull, 0x009c16c5c5253575ull, 0x00d18e3b9b374169ull,
        0x0105f1ca820511c3ull, 0x013b6e3d22865634ull, 0x017124e63593f5e1ull, 0x01a56e1fc2f8f359ull,
        0x01dac9a7b3b7302full, 0x0210be08d0527e1dull, 0x0244ed8b04671da5ull, 0x027a28edc580e50eull,
        0x02b059949b708f29ull, 0x02e46ff9c24cb2f3ull, 0x03198bf832dfdfb0ull, 0x034feef63f97d79cull,
        0x0383f559e7bee6c1ull, 0x03b8f2b061aea072ull, 0x03ef2f5c7a1a488eull, 0x04237d99cc506d59ull,
        0x04585d003f6488afull, 0x048e74404f3daadbull, 0x04c308a831868ac9ull, 0x04f7cad23de82d7bull,
        0x052dbd86cd6238d9ull, 0x05629674405d6388ull, 0x05973c115074bc6aull, 0x05cd0b15a491eb84ull,
        0x060226ed86db3333ull, 0x0636b0a8e891ffffull, 0x066c5cd322b67fffull, 0x06a1ba03f5b21000ull,
        0x06d62884f31e93ffull, 0x070bb2a62fe638ffull, 0x07414fa7ddefe3a0ull, 0x0775a391d56bdc87ull,
        0x07ab0c764ac6d3a9ull, 0x07e0e7c9eebc444aull, 0x081521bc6a6b555cull, 0x084a6a2b85062ab3ull,
        0x0880825b3323dab0ull, 0x08b4a2f1ffecd15cull, 0x08e9cbae7fe805b3ull, 0x09201f4d0ff10390ull,
        0x0954272053ed4474ull, 0x098930e868e89591ull, 0x09bf7d228322baf5ull, 0x09f3ae3591f5b4d9ull,
        0x0a2899c2f6732210ull, 0x0a5ec033b40fea93ull, 0x0a9338205089f29cull, 0x0ac8062864ac6f43ull,
        0x0afe07b27dd78b14ull, 0x0b32c4cf8ea6b6ecull, 0x0b677603725064a8ull, 0x0b9d53844ee47dd1ull,
        0x0bd25432b14ecea3ull, 0x0c06e93f5da2824cull, 0x0c3ca38f350b22dfull, 0x0c71e6398126f5cbull,
        0x0ca65fc7e170b33eull, 0x0cdbf7b9d9cce00dull, 0x0d117ad428200c08ull, 0x0d45d98932280f0aull,
        0x0d7b4feb7eb212cdull, 0x0db111f32f2f4bc0ull, 0x0de5566ffafb1eb0ull, 0x0e1aac0bf9b9e65cull,
        0x0e50ab877c142ffaull, 0x0e84d6695b193bf8ull, 0x0eba0c03b1df8af6ull, 0x0ef047824f2bb6daull,
        0x0f245962e2f6a490ull, 0x0f596fbb9bb44db4ull, 0x0f8fcbaa82a16121ull, 0x0fc3df4a91a4dcb5ull,
        0x0ff8d71d360e13e2ull, 0x102f0ce4839198dbull, 0x1063680ed23aff89ull, 0x1098421286c9bf6bull,
        0x10ce5297287c2f45ull, 0x1102f39e794d9d8bull, 0x1137b08617a104eeull, 0x116d9ca79d89462aull,
        0x11a281e8c275cbdaull, 0x11d72262f3133ed1ull, 0x120ceafbafd80e85ull, 0x124212dd4de70913ull,
        0x12769794a160cb58ull, 0x12ac3d79c9b8fe2eull, 0x12e1a66c1e139eddull, 0x1316100725988694ull,
        0x134b9408eefea839ull, 0x13813c85955f2923ull, 0x13b58ba6fab6f36cull, 0x13eaee90b964b047ull,
        0x1420d51a73deee2dull, 0x14550a6110d6a9b8ull, 0x148a4cf9550c5426ull, 0x14c0701bd527b498ull,
        0x14f48c22ca71a1bdull, 0x1529af2b7d0e0a2dull, 0x15600d7b2e28c65cull, 0x159410d9f9b2f7f3ull,
        0x15c91510781fb5f0ull, 0x15ff5a549627a36cull, 0x16339874ddd8c623ull, 0x16687e92154ef7acull,
        0x169e9e369aa2b597ull, 0x16d322e220a5b17eull, 0x1707eb9aa8cf1ddeull, 0x173de6815302e556ull,
        0x1772b010d3e1cf56ull, 0x17a75c1508da432bull, 0x17dd331a4b10d3f6ull, 0x18123ff06eea847aull,
        0x1846cfec8aa52598ull, 0x187c83e7ad4e6efeull, 0x18b1d270cc51055full, 0x18e6470cff6546b6ull,
        0x191bd8d03f3e9864ull, 0x1951678227871f3eull, 0x1985c162b168e70eull, 0x19bb31bb5dc320d2ull,
        0x19f0ff151a99f483ull, 0x1a253eda614071a4ull, 0x1a5a8e90f9908e0dull, 0x1a90991a9bfa58c8ull,
        0x1ac4bf6142f8eefaull, 0x1af9ef3993b72ab8ull, 0x1b303583fc527ab3ull, 0x1b6442e4fb671960ull,
        0x1b99539e3a40dfb8ull, 0x1bcfa885c8d117a6ull, 0x1c03c9539d82aec8ull, 0x1c38bba884e35a7aull,
        0x1c6eea92a61c3118ull, 0x1ca3529ba7d19eafull, 0x1cd8274291c6065bull, 0x1d0e3113363787f2ull,
        0x1d42deac01e2b4f7ull, 0x1d779657025b6235ull, 0x1dad7becc2f23ac2ull, 0x1de26d73f9d764b9ull,
        0x1e1708d0f84d3de7ull, 0x1e4ccb0536608d61ull, 0x1e81fee341fc585dull, 0x1eb67e9c127b6e74ull,
        0x1eec1e43171a4a11ull, 0x1f2192e9ee706e4bull, 0x1f55f7a46a0c89ddull, 0x1f8b758d848fac55ull,
        0x1fc1297872d9cbb5ull, 0x1ff573d68f903ea2ull, 0x202ad0cc33744e4bull, 0x2060c27fa028b0efull,
        0x2094f31f8832dd2aull, 0x20ca2fe76a3f9475ull, 0x21005df0a267bcc9ull, 0x2134756ccb01abfbull,
        0x216992c7fdc216faull, 0x219ff779fd329cb9ull, 0x21d3faac3e3fa1f3ull, 0x2208f9574dcf8a70ull,
        0x223f37ad21436d0cull, 0x227382cc34ca2428ull, 0x22a8637f41fcad32ull, 0x22de7c5f127bd87eull,
        0x23130dbb6b8d674full, 0x2347d12a4670c123ull, 0x237dc574d80cf16bull, 0x23b29b69070816e3ull,
        0x23e7424348ca1c9cull, 0x241d12d41afca3c3ull, 0x24522bc490dde65aull, 0x2486b6b5b5155ff0ull,
        0x24bc6463225ab7ecull, 0x24f1bebdf578b2f4ull, 0x25262e6d72d6dfb0ull, 0x255bba08cf8c979dull,
        0x2591544581b7dec2ull, 0x25c5a956e225d672ull, 0x25fb13ac9aaf4c0full, 0x2630ec4be0ad8f89ull,
        0x2665275ed8d8f36cull, 0x269a71368f0f3047ull, 0x26d086c219697e2cull, 0x2704a8729fc3ddb7ull,
        0x2739d28f47b4d525ull, 0x277023998cd10537ull, 0x27a42c7ff0054685ull, 0x27d9379fec069826ull,
        0x280f8587e7083e30ull, 0x2843b374f06526deull, 0x2878a0522c7e7095ull, 0x28aec866b79e0cbaull,
        0x28e33d4032c2c7f5ull, 0x29180c903f7379f2ull, 0x294e0fb44f50586eull, 0x2982c9d0b1923745ull,
        0x29b77c44ddf6c516ull, 0x29ed5b561574765bull, 0x2a225915cd68c9f9ull, 0x2a56ef5b40c2fc77ull,
        0x2a8cab3210f3bb95ull, 0x2ac1eaff4a98553dull, 0x2af665bf1d3e6a8dull, 0x2b2bff2ee48e0530ull,
        0x2b617f7d4ed8c33eull, 0x2b95df5ca28ef40dull, 0x2bcb5733cb32b111ull, 0x2c0116805effaeaaull,
        0x2c355c2076bf9a55ull, 0x2c6ab328946f80eaull, 0x2ca0aff95cc5b092ull, 0x2cd4dbf7b3f71cb7ull,
        0x2d0a12f5a0f4e3e5ull, 0x2d404bd984990e6full, 0x2d745ecfe5bf520bull, 0x2da97683df2f268dull,
        0x2ddfd424d6faf031ull, 0x2e13e497065cd61full, 0x2e48ddbcc7f40ba6ull, 0x2e7f152bf9f10e90ull,
        0x2eb36d3b7c36a91aull, 0x2ee8488a5b445360ull, 0x2f1e5aacf2156838ull, 0x2f52f8ac174d6123ull,
        0x2f87b6d71d20b96cull, 0x2fbda48ce468e7c7ull, 0x2ff286d80ec190dcull, 0x3027288e1271f513ull,
        0x305cf2b1970e7258ull, 0x309217aefe690777ull, 0x30c69d9abe034955ull, 0x30fc45016d841baaull,
        0x3131ab20e472914aull, 0x316615e91d8f359dull, 0x319b9b6364f30304ull, 0x31d1411e1f17e1e3ull,
        0x32059165a6ddda5bull, 0x323af5bf109550f2ull, 0x3270d9976a5d5297ull, 0x32a50ffd44f4a73dull,
        0x32da53fc9631d10dull, 0x3310747ddddf22a8ull, 0x3344919d5556eb52ull, 0x3379b604aaaca626ull,
        0x33b011c2eaabe7d8ull, 0x33e41633a556e1ceull, 0x34191bc08eac9a41ull, 0x344f62b0b257c0d2ull,
        0x34839dae6f76d883ull, 0x34b8851a0b548ea4ull, 0x34eea6608e29b24dull, 0x352327fc58da0f70ull,
        0x3557f1fb6f10934cull, 0x358dee7a4ad4b81full, 0x35c2b50c6ec4f313ull, 0x35f7624f8a762fd8ull,
        0x362d3ae36d13bbceull, 0x366244ce242c5561ull, 0x3696d601ad376ab9ull, 0x36cc8b8218854567ull,
        0x3701d7314f534b61ull, 0x37364cfda3281e39ull, 0x376be03d0bf225c7ull, 0x37a16c262777579cull,
        0x37d5c72fb1552d83ull, 0x380b38fb9daa78e4ull, 0x3841039d428a8b8full, 0x38754484932d2e72ull,
        0x38aa95a5b7f87a0full, 0x38e09d8792fb4c49ull, 0x3914c4e977ba1f5cull, 0x3949f623d5a8a733ull,
        0x398039d665896880ull, 0x39b4484bfeebc2a0ull, 0x39e95a5efea6b347ull, 0x3a1fb0f6be506019ull,
        0x3a53ce9a36f23c10ull, 0x3a88c240c4aecb14ull, 0x3abef2d0f5da7dd9ull, 0x3af357c299a88ea7ull,
        0x3b282db34012b251ull, 0x3b5e392010175ee6ull, 0x3b92e3b40a0e9b4full, 0x3bc79ca10c924223ull,
        0x3bfd83c94fb6d2acull, 0x3c32725dd1d243acull, 0x3c670ef54646d497ull, 0x3c9cd2b297d889bcull,
        0x3cd203af9ee75616ull, 0x3d06849b86a12b9bull, 0x3d3c25c268497682ull, 0x3d719799812dea11ull,
        0x3da5fd7fe1796495ull, 0x3ddb7cdfd9d7bdbbull, 0x3e112e0be826d695ull, 0x3e45798ee2308c3aull,
        0x3e7ad7f29abcaf48ull, 0x3eb0c6f7a0b5ed8dull, 0x3ee4f8b588e368f1ull, 0x3f1a36e2eb1c432dull,
        0x3f50624dd2f1a9fcull, 0x3f847ae147ae147bull, 0x3fb999999999999aull, 0x3ff0000000000000ull,
        0x4024000000000000ull, 0x4059000000000000ull, 0x408f400000000000ull, 0x40c3880000000000ull,
        0x40f86a0000000000ull, 0x412e848000000000ull, 0x416312d000000000ull, 0x4197d78400000000ull,
        0x41cdcd6500000000ull, 0x4202a05f20000000ull, 0x42374876e8000000ull, 0x426d1a94a2000000ull,
        0x42a2309ce5400000ull, 0x42d6bcc41e900000ull, 0x430c6bf526340000ull, 0x4341c37937e08000ull,
        0x4376345785d8a000ull, 0x43abc16d674ec800ull, 0x43e158e460913d00ull, 0x4415af1d78b58c40ull,
        0x444b1ae4d6e2ef50ull, 0x4480f0cf064dd592ull, 0x44b52d02c7e14af6ull, 0x44ea784379d99db4ull,
        0x45208b2a2c280291ull, 0x4554adf4b7320335ull, 0x4589d971e4fe8402ull, 0x45c027e72f1f1281ull,
        0x45f431e0fae6d721ull, 0x46293e5939a08ceaull, 0x465f8def8808b024ull, 0x4693b8b5b5056e17ull,
        0x46c8a6e32246c99cull, 0x46fed09bead87c03ull, 0x4733426172c74d82ull, 0x476812f9cf7920e3ull,
        0x479e17b84357691bull, 0x47d2ced32a16a1b1ull, 0x48078287f49c4a1dull, 0x483d6329f1c35ca5ull,
        0x48725dfa371a19e7ull, 0x48a6f578c4e0a061ull, 0x48dcb2d6f618c879ull, 0x4911efc659cf7d4cull,
        0x49466bb7f0435c9eull, 0x497c06a5ec5433c6ull, 0x49b18427b3b4a05cull, 0x49e5e531a0a1c873ull,
        0x4a1b5e7e08ca3a8full, 0x4a511b0ec57e649aull, 0x4a8561d276ddfdc0ull, 0x4ababa4714957d30ull,
        0x4af0b46c6cdd6e3eull, 0x4b24e1878814c9ceull, 0x4b5a19e96a19fc41ull, 0x4b905031e2503da9ull,
        0x4bc4643e5ae44d13ull, 0x4bf97d4df19d6057ull, 0x4c2fdca16e04b86dull, 0x4c63e9e4e4c2f344ull,
        0x4c98e45e1df3b015ull, 0x4ccf1d75a5709c1bull, 0x4d03726987666191ull, 0x4d384f03e93ff9f5ull,
        0x4d6e62c4e38ff872ull, 0x4da2fdbb0e39fb47ull, 0x4dd7bd29d1c87a19ull, 0x4e0dac74463a989full,
        0x4e428bc8abe49f64ull, 0x4e772ebad6ddc73dull, 0x4eacfa698c95390cull, 0x4ee21c81f7dd43a7ull,
        0x4f16a3a275d49491ull, 0x4f4c4c8b1349b9b5ull, 0x4f81afd6ec0e1411ull, 0x4fb61bcca7119916ull,
        0x4feba2bfd0d5ff5bull, 0x502145b7e285bf99ull, 0x50559725db272f7full, 0x508afcef51f0fb5full,
        0x50c0de1593369d1bull, 0x50f5159af8044462ull, 0x512a5b01b605557bull, 0x516078e111c3556dull,
        0x5194971956342ac8ull, 0x51c9bcdfabc1357aull, 0x5200160bcb58c16cull, 0x52341b8ebe2ef1c7ull,
        0x526922726dbaae39ull, 0x529f6b0f092959c7ull, 0x52d3a2e965b9d81dull, 0x53088ba3bf284e24ull,
        0x533eae8caef261adull, 0x53732d17ed577d0cull, 0x53a7f85de8ad5c4full, 0x53ddf67562d8b363ull,
        0x5412ba095dc7701eull, 0x5447688bb5394c25ull, 0x547d42aea2879f2eull, 0x54b249ad2594c37dull,
        0x54e6dc186ef9f45cull, 0x551c931e8ab87173ull, 0x5551dbf316b346e8ull, 0x558652efdc6018a2ull,
        0x55bbe7abd3781ecaull, 0x55f170cb642b133full, 0x5625ccfe3d35d80eull, 0x565b403dcc834e12ull,
        0x569108269fd210cbull, 0x56c54a3047c694feull, 0x56fa9cbc59b83a3dull, 0x5730a1f5b8132466ull,
        0x5764ca732617ed80ull, 0x5799fd0fef9de8e0ull, 0x57d03e29f5c2b18cull, 0x58044db473335defull,
        0x583961219000356bull, 0x586fb969f40042c5ull, 0x58a3d3e2388029bbull, 0x58d8c8dac6a0342aull,
        0x590efb1178484135ull, 0x59435ceaeb2d28c1ull, 0x59783425a5f872f1ull, 0x59ae412f0f768fadull,
        0x59e2e8bd69aa19ccull, 0x5a17a2ecc414a03full, 0x5a4d8ba7f519c84full, 0x5a827748f9301d32ull,
        0x5ab7151b377c247eull, 0x5aecda62055b2d9eull, 0x5b22087d4358fc82ull, 0x5b568a9c942f3ba3ull,
        0x5b8c2d43b93b0a8cull, 0x5bc19c4a53c4e697ull, 0x5bf6035ce8b6203dull, 0x5c2b843422e3a84dull,
        0x5c6132a095ce4930ull, 0x5c957f48bb41db7cull, 0x5ccadf1aea12525bull, 0x5d00cb70d24b7379ull,
        0x5d34fe4d06de5057ull, 0x5d6a3de04895e46dull, 0x5da066ac2d5daec4ull, 0x5dd4805738b51a75ull,
        0x5e09a06d06e26112ull, 0x5e400444244d7cabull, 0x5e7405552d60dbd6ull, 0x5ea906aa78b912ccull,
        0x5edf485516e7577full, 0x5f138d352e5096afull, 0x5f48708279e4bc5bull, 0x5f7e8ca3185deb72ull,
        0x5fb317e5ef3ab327ull, 0x5fe7dddf6b095ff1ull, 0x601dd55745cbb7edull, 0x6052a5568b9f52f4ull,
        0x60874eac2e8727b1ull, 0x60bd22573a28f19dull, 0x60f2357684599702ull, 0x6126c2d4256ffcc3ull,
        0x615c73892ecbfbf4ull, 0x6191c835bd3f7d78ull, 0x61c63a432c8f5cd6ull, 0x61fbc8d3f7b3340cull,
        0x62315d847ad00087ull, 0x6265b4e5998400a9ull, 0x629b221effe500d4ull, 0x62d0f5535fef2084ull,
        0x630532a837eae8a5ull, 0x633a7f5245e5a2cfull, 0x63708f936baf85c1ull, 0x63a4b378469b6732ull,
        0x63d9e056584240feull, 0x64102c35f729689full, 0x6444374374f3c2c6ull, 0x647945145230b378ull,
        0x64af965966bce056ull, 0x64e3bdf7e0360c36ull, 0x6518ad75d8438f43ull, 0x654ed8d34e547314ull,
        0x6583478410f4c7ecull, 0x65b819651531f9e8ull, 0x65ee1fbe5a7e7861ull, 0x6622d3d6f88f0b3dull,
        0x665788ccb6b2ce0cull, 0x668d6affe45f818full, 0x66c262dfeebbb0f9ull, 0x66f6fb97ea6a9d38ull,
        0x672cba7de5054486ull, 0x6761f48eaf234ad4ull, 0x679671b25aec1d89ull, 0x67cc0e1ef1a724ebull,
        0x680188d357087713ull, 0x6835eb082cca94d7ull, 0x686b65ca37fd3a0dull, 0x68a11f9e62fe4448ull,
        0x68d56785fbbdd55aull, 0x690ac1677aad4ab1ull, 0x6940b8e0acac4eafull, 0x6974e718d7d7625aull,
        0x69aa20df0dcd3af1ull, 0x69e0548b68a044d6ull, 0x6a1469ae42c8560cull, 0x6a498419d37a6b8full,
        0x6a7fe52048590673ull, 0x6ab3ef342d37a408ull, 0x6ae8eb0138858d0aull, 0x6b1f25c186a6f04cull,
        0x6b537798f4285630ull, 0x6b88557f31326bbbull, 0x6bbe6adefd7f06aaull, 0x6bf302cb5e6f642aull,
        0x6c27c37e360b3d35ull, 0x6c5db45dc38e0c82ull, 0x6c9290ba9a38c7d1ull, 0x6cc734e940c6f9c6ull,
        0x6cfd022390f8b837ull, 0x6d3221563a9b7323ull, 0x6d66a9abc9424febull, 0x6d9c5416bb92e3e6ull,
        0x6dd1b48e353bce70ull, 0x6e0621b1c28ac20cull, 0x6e3baa1e332d728full, 0x6e714a52dffc6799ull,
        0x6ea59ce797fb817full, 0x6edb04217dfa61dfull, 0x6f10e294eebc7d2cull, 0x6f451b3a2a6b9c76ull,
        0x6f7a6208b5068394ull, 0x6fb07d457124123dull, 0x6fe49c96cd6d16ccull, 0x7019c3bc80c85c7full,
        0x70501a55d07d39cfull, 0x708420eb449c8843ull, 0x70b9292615c3aa54ull, 0x70ef736f9b3494e9ull,
        0x7123a825c100dd11ull, 0x7158922f31411456ull, 0x718eb6bafd91596bull, 0x71c33234de7ad7e3ull,
        0x71f7fec216198ddcull, 0x722dfe729b9ff153ull, 0x7262bf07a143f6d4ull, 0x72976ec98994f489ull,
        0x72cd4a7bebfa31abull, 0x73024e8d737c5f0bull, 0x7336e230d05b76cdull, 0x736c9abd04725481ull,
        0x73a1e0b622c774d0ull, 0x73d658e3ab795204ull, 0x740bef1c9657a686ull, 0x74417571ddf6c814ull,
        0x7475d2ce55747a18ull, 0x74ab4781ead1989eull, 0x74e10cb132c2ff63ull, 0x75154fdd7f73bf3cull,
        0x754aa3d4df50af0bull, 0x7580a6650b926d67ull, 0x75b4cffe4e7708c0ull, 0x75ea03fde214caf1ull,
        0x7620427ead4cfed6ull, 0x7654531e58a03e8cull, 0x768967e5eec84e2full, 0x76bfc1df6a7a61bbull,
        0x76f3d92ba28c7d15ull, 0x7728cf768b2f9c5aull, 0x775f03542dfb8370ull, 0x779362149cbd3226ull,
        0x77c83a99c3ec7eb0ull, 0x77fe494034e79e5cull, 0x7832edc82110c2f9ull, 0x7867a93a2954f3b8ull,
        0x789d9388b3aa30a5ull, 0x78d27c35704a5e67ull, 0x79071b42cc5cf601ull, 0x793ce2137f743382ull,
        0x79720d4c2fa8a031ull, 0x79a6909f3b92c83dull, 0x79dc34c70a777a4dull, 0x7a11a0fc668aac70ull,
        0x7a46093b802d578cull, 0x7a7b8b8a6038ad6full, 0x7ab137367c236c65ull, 0x7ae585041b2c477full,
        0x7b1ae64521f7595eull, 0x7b50cfeb353a97dbull, 0x7b8503e602893dd2ull, 0x7bba44df832b8d46ull,
        0x7bf06b0bb1fb384cull, 0x7c2485ce9e7a065full, 0x7c59a742461887f6ull, 0x7c9008896bcf54faull,
        0x7cc40aabc6c32a38ull, 0x7cf90d56b873f4c7ull, 0x7d2f50ac6690f1f8ull, 0x7d63926bc01a973bull,
        0x7d987706b0213d0aull, 0x7dce94c85c298c4cull, 0x7e031cfd3999f7b0ull, 0x7e37e43c8800759cull,
        0x7e6ddd4baa009303ull, 0x7ea2aa4f4a405be2ull, 0x7ed754e31cd072daull, 0x7f0d2a1be4048f90ull,
        0x7f423a516e82d9baull, 0x7f76c8e5ca239029ull, 0x7fac7b1f3cac7433ull, 0x7fe1ccf385ebc8a0ull
    };

    [[nodiscard]] BL_FORCE_INLINE constexpr double pow10(int exponent) noexcept
    {
        if (exponent > pow10_f64_max_exponent)
            return std::numeric_limits<double>::infinity();
        if (exponent < pow10_f64_min_exponent)
            return 0.0;

        return std::bit_cast<double>(pow10_f64_bits[exponent - pow10_f64_min_exponent]);
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr typename Traits::value_type round_to_decimals_native(
        typename Traits::value_type v,
        int prec) noexcept
    {
        constexpr int local_capacity = std::numeric_limits<typename Traits::value_type>::max_digits10;

        if (prec <= 0)
            return v;
        if (prec > local_capacity)
            prec = local_capacity;
        if (!Traits::isfinite(v) || v == Traits::zero(false))
            return v;

        using detail::exact_decimal::biguint;

        const bool neg = Traits::signbit(v);
        const double ax = static_cast<double>(Traits::abs(v));

        int exponent = 0;
        bool ignored_neg = false;
        const std::uint64_t mantissa = detail::exact_decimal::decompose_double_mantissa(ax, exponent, ignored_neg);
        if (mantissa == 0)
            return v;

        biguint num{ mantissa };
        biguint den{ 1 };
        if (exponent >= 0)
            num.shl_bits(exponent);
        else
            den.shl_bits(-exponent);

        for (int i = 0; i < prec; ++i)
            num.mul_small(10);

        biguint q;
        biguint r;
        detail::exact_decimal::divmod_bitwise(num, den, q, r);
        if (!r.is_zero())
        {
            biguint twice_r = r;
            twice_r.shl1();
            const int cmp = detail::exact_decimal::compare(twice_r, den);
            if (cmp > 0 || (cmp == 0 && q.is_odd()))
                q.add_small(1);
        }

        return exact_decimal_to_native_value<Traits>(q, -prec, neg);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double round_to_decimals(double v, int prec) noexcept
    {
        return round_to_decimals_native<f64_decimal_round_traits>(v, prec);
    }

    template<class Traits>
    [[nodiscard]] BL_FORCE_INLINE constexpr typename Traits::value_type round_to_significant_figures_native(
        typename Traits::value_type v,
        int figures) noexcept
    {
        if (figures <= 0 || !Traits::isfinite(v) || v == Traits::zero(false))
            return v;
        if (figures > std::numeric_limits<typename Traits::value_type>::max_digits10)
            figures = std::numeric_limits<typename Traits::value_type>::max_digits10;

        detail::exact_decimal::biguint coefficient;
        int exp10 = 0;
        const bool neg = Traits::signbit(v);
        if (!detail::exact_decimal::exact_significant_decimal<Traits>(Traits::abs(v), figures, coefficient, exp10))
            return v;

        return exact_decimal_to_native_value<Traits>(coefficient, exp10 - (figures - 1), neg);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double round_to_significant_figures(double v, int figures) noexcept
    {
        return round_to_significant_figures_native<f64_decimal_round_traits>(v, figures);
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr dyadic_u64 subtract_scaled_u64(std::uint64_t a, int a_exp2, std::uint64_t b, int b_exp2) noexcept
    {
        const int common_exp2 = a_exp2 < b_exp2 ? a_exp2 : b_exp2;
        const int a_shift     = a_exp2 - common_exp2;
        const int b_shift     = b_exp2 - common_exp2;
        return dyadic_u64{ (a << a_shift) - (b << b_shift), common_exp2 };
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr exact_divmod_result exact_abs_divmod(double ax, double ay) noexcept
    {
        const dyadic_u64 x = decompose_normalized_abs(ax);
        const dyadic_u64 y = decompose_normalized_abs(ay);

        exact_divmod_result out;
        out.denominator = y.coeff;
        out.denominator_exp2 = y.exp2;

        if (compare_scaled_u64(x.coeff, x.exp2, y.coeff, y.exp2) < 0)
        {
            out.remainder = x.coeff;
            out.remainder_exp2 = x.exp2;
            return out;
        }

        std::uint64_t rem = x.coeff;
        const int shift = x.exp2 - y.exp2;

        for (int i = shift; i >= 0; --i)
        {
            if (rem >= y.coeff)
            {
                rem -= y.coeff;
                out.quotient_nonzero = true;
                if (i < 3)
                    out.quotient_low_bits |= (1u << i);
            }

            if (i != 0)
                rem <<= 1;
        }

        out.remainder = rem;
        out.remainder_exp2 = y.exp2;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr int normalized_remquo_bits(unsigned quotient_low_bits, bool quotient_nonzero, bool quotient_negative) noexcept
    {
        int bits = detail::fp::remquo_low_quotient_bits(quotient_low_bits, false, 0x7u);
        if (bits == 0 && quotient_nonzero)
            bits = 8;
        return quotient_negative ? -bits : bits;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmod_exact(double x, double y) noexcept
    {
        if (detail::fp::isinf_or_nan(x) || detail::fp::iszero_or_nan(y))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y) || x == 0.0)
            return x;

        const exact_divmod_result divmod = exact_abs_divmod(abs(x), abs(y));
        double out = exact_dyadic_to_double(divmod.remainder, divmod.remainder_exp2, signbit(x));
        if (out == 0.0)
            out = signbit(x) ? -0.0 : 0.0;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double remainder_exact(double x, double y, int* quo = nullptr) noexcept
    {
        if (quo)
            *quo = 0;

        if (detail::fp::isinf_or_nan(x) || detail::fp::iszero_or_nan(y))
            return std::numeric_limits<double>::quiet_NaN();
        if (isinf(y) || x == 0.0)
            return x;

        const bool quotient_negative = signbit(x) != signbit(y);
        exact_divmod_result divmod = exact_abs_divmod(abs(x), abs(y));

        bool result_negative = signbit(x);
        std::uint64_t result_coeff = divmod.remainder;
        int result_exp2 = divmod.remainder_exp2;

        const int half_cmp = compare_scaled_u64(
            divmod.remainder,
            divmod.remainder_exp2 + 1,
            divmod.denominator,
            divmod.denominator_exp2);
        if (half_cmp > 0 || (half_cmp == 0 && (divmod.quotient_low_bits & 1u) != 0))
        {
            const dyadic_u64 adjusted = subtract_scaled_u64(
                divmod.denominator,
                divmod.denominator_exp2,
                divmod.remainder,
                divmod.remainder_exp2);

            result_coeff = adjusted.coeff;
            result_exp2 = adjusted.exp2;
            divmod.quotient_low_bits = (divmod.quotient_low_bits + 1u) & 0x7u;
            divmod.quotient_nonzero = true;
            result_negative = !result_negative;
        }

        if (quo)
            *quo = normalized_remquo_bits(divmod.quotient_low_bits, divmod.quotient_nonzero, quotient_negative);

        double out = exact_dyadic_to_double(result_coeff, result_exp2, result_negative);
        if (out == 0.0)
            out = signbit(x) ? -0.0 : 0.0;
        return out;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmin(double a, double b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a < b) return a;
        if (b < a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? a : b;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
    {
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        if (a > b) return a;
        if (b > a) return b;
        if (iszero(a) && iszero(b))
            return signbit(a) ? b : a;
        return a;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double frexp(double x, int* exp) noexcept
    {
        if (exp)
            *exp = 0;

        if (isnan(x) || isinf(x) || iszero(x))
            return x;

        int e = frexp_exponent(x);
        double m = ldexp(x, -e);
        const double am = abs(m);

        if (am < 0.5)
        {
            m *= 2.0;
            --e;
        }
        else if (am >= 1.0)
        {
            m *= 0.5;
            ++e;
        }

        if (exp)
            *exp = e;

        return m;
    }

    [[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
    {
        const double i = trunc(x);
        if (iptr)
            *iptr = i;

        double frac = x - i;
        if (iszero(frac))
            frac = signbit(x) ? -0.0 : 0.0;
        return frac;
    }

} // namespace detail::_f64_impl

[[nodiscard]] BL_FORCE_INLINE constexpr double floor(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::floor(x),
        std::floor(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ceil(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::ceil(x),
        std::ceil(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double trunc(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::trunc(x),
        std::trunc(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::round_half_away_zero(x),
        std::round(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round_to_decimals(double x, int prec) noexcept
{
    return detail::_f64_impl::round_to_decimals(x, prec);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round_to_precision(double x, int figures) noexcept
{
    return detail::_f64_impl::round_to_significant_figures(x, figures);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double round_to(double x, int precision, round_format format) noexcept
{
    return format == round_format::decimals
        ? round_to_decimals(x, precision)
        : round_to_precision(x, precision);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nearbyint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::nearbyint(x),
        std::nearbyint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double rint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::nearbyint(x),
        std::rint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long>(detail::_f64_impl::round_half_away_zero(x)),
        std::lround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llround(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long long>(detail::_f64_impl::round_half_away_zero(x)),
        std::llround(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long lrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long>(detail::_f64_impl::nearbyint(x)),
        std::lrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr long long llrint(double x) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::to_signed_integer_or_zero<long long>(detail::_f64_impl::nearbyint(x)),
        std::llrint(x)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmod(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::fmod_exact(x, y),
        std::fmod(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remainder(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::remainder_exact(x, y),
        std::remainder(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double remquo(double x, double y, int* quo) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::remainder_exact(x, y, quo),
        std::remquo(x, y, quo)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fma(double x, double y, double z) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        x * y + z,
        std::fma(x, y, z)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmin(double a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::fmin(a, b),
        std::fmin(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fmax(double a, double b) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::fmax(a, b),
        std::fmax(a, b)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double fdim(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        (x > y) ? (x - y) : 0.0,
        std::fdim(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double copysign(double x, double y) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::copysign(x, y),
        std::copysign(x, y)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double ldexp(double x, int e) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::ldexp(x, e),
        std::ldexp(x, e)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double scalbn(double x, int e) noexcept
{
    return ldexp(x, e);
}

[[nodiscard]] BL_FORCE_INLINE constexpr double scalbln(double x, long e) noexcept
{
    return ldexp(x, static_cast<int>(e));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double frexp(double x, int* exp) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::frexp(x, exp),
        std::frexp(x, exp)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double modf(double x, double* iptr) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::modf(x, iptr),
        std::modf(x, iptr)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr int ilogb(double x) noexcept
{
    if (isnan(x))
        return FP_ILOGBNAN;
    if (iszero(x))
        return FP_ILOGB0;
    if (isinf(x))
        return std::numeric_limits<int>::max();

    return detail::_f64_impl::ilogb_finite(abs(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double logb(double x) noexcept
{
    if (isnan(x))
        return x;
    if (iszero(x))
        return -std::numeric_limits<double>::infinity();
    if (isinf(x))
        return std::numeric_limits<double>::infinity();

    return static_cast<double>(ilogb(x));
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nextafter(double from, double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        detail::_f64_impl::nextafter(from, to),
        std::nextafter(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, long double to) noexcept
{
    BL_CONSTEXPR_RUNTIME_DISPATCH(
        nextafter(from, static_cast<double>(to)),
        std::nexttoward(from, to)
    );
}

[[nodiscard]] BL_FORCE_INLINE constexpr double nexttoward(double from, double to) noexcept
{
    return nextafter(from, to);
}

} // namespace bl

#endif
