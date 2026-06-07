#include <iostream>
#include <iomanip>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

using llong = long long;

consteval f32 f32_test()
{
    constexpr f32 abs_result = bl::abs(-0.123f);
    constexpr f32 fma_result = bl::fma(0.123f, 4.56f, -0.75f);
                                      
    constexpr f32 floor_result     = bl::floor(2.75f);
    constexpr f32 ceil_result      = bl::ceil(2.25f);
    constexpr f32 trunc_result     = bl::trunc(-2.75f);
    constexpr f32 round_result     = bl::round(2.5f);
    constexpr long lround_result   = bl::lround(2.5f);
    constexpr llong llround_result = bl::llround(2.5f);
    constexpr f32 nearbyint_result = bl::nearbyint(2.5f);
    constexpr f32 rint_result      = bl::rint(2.5f);
    constexpr long lrint_result    = bl::lrint(2.5f);
    constexpr llong llrint_result  = bl::llrint(2.5f);
                                      
    constexpr f32 fmod_result      = bl::fmod(5.5f, 2.0f);
    constexpr f32 remainder_result = bl::remainder(5.5f, 2.0f);
    constexpr f32 remquo_result1   = [] { int quo{}; return bl::remquo(5.5f, 2.0f, &quo); }();
    constexpr int remquo_result2   = [] { int quo{}; (void)bl::remquo(5.5f, 2.0f, &quo); return quo; }();
                                      
    constexpr f32 fmin_result     = bl::fmin(-0.25f, 0.5f);
    constexpr f32 fmax_result     = bl::fmax(-0.25f, 0.5f);
    constexpr f32 fdim_result     = bl::fdim(1.25f, 0.75f);
    constexpr f32 copysign_result = bl::copysign(0.125f, -1.0f);
                                      
    constexpr f32 sqrt_result  = bl::sqrt(0.123f);
    constexpr f32 cbrt_result  = bl::cbrt(0.123f);
    constexpr f32 hypot_result = bl::hypot(0.3f, 0.4f);
    constexpr f32 pow_result   = bl::pow(0.123f, 4.56f);
                                      
    constexpr f32 exp_result   = bl::exp(0.123f);
    constexpr f32 exp2_result  = bl::exp2(0.123f);
    constexpr f32 expm1_result = bl::expm1(0.123f);
    constexpr f32 log_result   = bl::log(1.123f);
    constexpr f32 log2_result  = bl::log2(1.123f);
    constexpr f32 log10_result = bl::log10(1.123f);
    constexpr f32 log1p_result = bl::log1p(0.123f);
    constexpr f32 logb_result  = bl::logb(12.5f);
    constexpr int ilogb_result = bl::ilogb(12.5f);
                                      
    constexpr f32 sin_result   = bl::sin(0.123f);
    constexpr f32 cos_result   = bl::cos(0.123f);
    constexpr f32 tan_result   = bl::tan(0.123f);
    constexpr f32 asin_result  = bl::asin(0.123f);
    constexpr f32 acos_result  = bl::acos(0.123f);
    constexpr f32 atan_result  = bl::atan(0.123f);
    constexpr f32 atan2_result = bl::atan2(0.123f, 0.456f);
                                      
    constexpr f32 sinh_result  = bl::sinh(0.123f);
    constexpr f32 cosh_result  = bl::cosh(0.123f);
    constexpr f32 tanh_result  = bl::tanh(0.123f);
    constexpr f32 asinh_result = bl::asinh(0.123f);
    constexpr f32 acosh_result = bl::acosh(1.123f);
    constexpr f32 atanh_result = bl::atanh(0.123f);
                                      
    constexpr f32 erf_result    = bl::erf(0.123f);
    constexpr f32 erfc_result   = bl::erfc(0.123f);
    constexpr f32 lgamma_result = bl::lgamma(1.123f);
    constexpr f32 tgamma_result = bl::tgamma(1.123f);
                                      
    constexpr f32 ldexp_result     = bl::ldexp(0.123f, 5);
    constexpr f32 scalbn_result    = bl::scalbn(0.123f, 5);
    constexpr f32 scalbln_result   = bl::scalbln(0.123f, 5L);
    constexpr f32 frexp_result     = [] { int exp{}; return bl::frexp(12.5f, &exp); }();
    constexpr int frexp_exp_result = [] { int exp{}; (void)bl::frexp(12.5f, &exp); return exp; }();
    constexpr f32 modf_result      = [] { f32 ip{}; return bl::modf(12.5f, &ip); }();
    constexpr f32 modf_int_result  = [] { f32 ip{}; (void)bl::modf(12.5f, &ip); return ip; }();

    constexpr f32 nextafter_result   = bl::nextafter(0.123f, 0.124f);
    constexpr f32 nexttoward_result1 = bl::nexttoward(0.123f, 0.124L);
    constexpr f32 nexttoward_result2 = bl::nexttoward(0.123f, 0.124f);

    constexpr f32 scalar_sum =
        abs_result + fma_result
        + floor_result + ceil_result + trunc_result + round_result
        + static_cast<f32>(lround_result) + static_cast<f32>(llround_result)
        + nearbyint_result + rint_result
        + static_cast<f32>(lrint_result) + static_cast<f32>(llrint_result)
        + fmod_result + remainder_result + remquo_result1 + static_cast<f32>(remquo_result2)
        + fmin_result + fmax_result + fdim_result + copysign_result
        + sqrt_result + cbrt_result + hypot_result + pow_result
        + exp_result + exp2_result + expm1_result
        + log_result + log2_result + log10_result + log1p_result + logb_result + static_cast<f32>(ilogb_result)
        + sin_result + cos_result + tan_result + asin_result + acos_result + atan_result + atan2_result
        + sinh_result + cosh_result + tanh_result + asinh_result + acosh_result + atanh_result
        + erf_result + erfc_result + lgamma_result + tgamma_result
        + ldexp_result + scalbn_result + scalbln_result
        + frexp_result + static_cast<f32>(frexp_exp_result) + modf_result + modf_int_result
        + nextafter_result + nexttoward_result1 + nexttoward_result2;

    return scalar_sum;
}

consteval f64 f64_test()
{
    constexpr f64 abs_result = bl::abs(-0.123);
    constexpr f64 fma_result = bl::fma(0.123, 4.56, -0.75);
                                      
    constexpr f64 floor_result     = bl::floor(2.75);
    constexpr f64 ceil_result      = bl::ceil(2.25);
    constexpr f64 trunc_result     = bl::trunc(-2.75);
    constexpr f64 round_result     = bl::round(2.5);
    constexpr long lround_result   = bl::lround(2.5);
    constexpr llong llround_result = bl::llround(2.5);
    constexpr f64 nearbyint_result = bl::nearbyint(2.5);
    constexpr f64 rint_result      = bl::rint(2.5);
    constexpr long lrint_result    = bl::lrint(2.5);
    constexpr llong llrint_result  = bl::llrint(2.5);
                                      
    constexpr f64 fmod_result      = bl::fmod(5.5, 2.0);
    constexpr f64 remainder_result = bl::remainder(5.5, 2.0);
    constexpr f64 remquo_result1   = [] { int quo{}; return bl::remquo(5.5, 2.0, &quo); }();
    constexpr int remquo_result2   = [] { int quo{}; (void)bl::remquo(5.5, 2.0, &quo); return quo; }();
                                      
    constexpr f64 fmin_result     = bl::fmin(-0.25, 0.5);
    constexpr f64 fmax_result     = bl::fmax(-0.25, 0.5);
    constexpr f64 fdim_result     = bl::fdim(1.25, 0.75);
    constexpr f64 copysign_result = bl::copysign(0.125, -1.0);
                                      
    constexpr f64 sqrt_result  = bl::sqrt(0.123);
    constexpr f64 cbrt_result  = bl::cbrt(0.123);
    constexpr f64 hypot_result = bl::hypot(0.3, 0.4);
    constexpr f64 pow_result   = bl::pow(0.123, 4.56);
                                      
    constexpr f64 exp_result   = bl::exp(0.123);
    constexpr f64 exp2_result  = bl::exp2(0.123);
    constexpr f64 expm1_result = bl::expm1(0.123);
    constexpr f64 log_result   = bl::log(1.123);
    constexpr f64 log2_result  = bl::log2(1.123);
    constexpr f64 log10_result = bl::log10(1.123);
    constexpr f64 log1p_result = bl::log1p(0.123);
    constexpr f64 logb_result  = bl::logb(12.5);
    constexpr int ilogb_result = bl::ilogb(12.5);
                                      
    constexpr f64 sin_result   = bl::sin(0.123);
    constexpr f64 cos_result   = bl::cos(0.123);
    constexpr f64 tan_result   = bl::tan(0.123);
    constexpr f64 asin_result  = bl::asin(0.123);
    constexpr f64 acos_result  = bl::acos(0.123);
    constexpr f64 atan_result  = bl::atan(0.123);
    constexpr f64 atan2_result = bl::atan2(0.123, 0.456);
                                      
    constexpr f64 sinh_result  = bl::sinh(0.123);
    constexpr f64 cosh_result  = bl::cosh(0.123);
    constexpr f64 tanh_result  = bl::tanh(0.123);
    constexpr f64 asinh_result = bl::asinh(0.123);
    constexpr f64 acosh_result = bl::acosh(1.123);
    constexpr f64 atanh_result = bl::atanh(0.123);
                                      
    constexpr f64 erf_result    = bl::erf(0.123);
    constexpr f64 erfc_result   = bl::erfc(0.123);
    constexpr f64 lgamma_result = bl::lgamma(1.123);
    constexpr f64 tgamma_result = bl::tgamma(1.123);
                                      
    constexpr f64 ldexp_result     = bl::ldexp(0.123, 5);
    constexpr f64 scalbn_result    = bl::scalbn(0.123, 5);
    constexpr f64 scalbln_result   = bl::scalbln(0.123, 5L);
    constexpr f64 frexp_result     = [] { int exp{}; return bl::frexp(12.5, &exp); }();
    constexpr int frexp_exp_result = [] { int exp{}; (void)bl::frexp(12.5, &exp); return exp; }();
    constexpr f64 modf_result      = [] { f64 ip{}; return bl::modf(12.5, &ip); }();
    constexpr f64 modf_int_result  = [] { f64 ip{}; (void)bl::modf(12.5, &ip); return ip; }();

    constexpr f64 nextafter_result   = bl::nextafter(0.123, 0.124);
    constexpr f64 nexttoward_result1 = bl::nexttoward(0.123, 0.124L);
    constexpr f64 nexttoward_result2 = bl::nexttoward(0.123, 0.124);

    constexpr f64 scalar_sum =
        abs_result + fma_result
        + floor_result + ceil_result + trunc_result + round_result
        + static_cast<f64>(lround_result) + static_cast<f64>(llround_result)
        + nearbyint_result + rint_result
        + static_cast<f64>(lrint_result) + static_cast<f64>(llrint_result)
        + fmod_result + remainder_result + remquo_result1 + static_cast<f64>(remquo_result2)
        + fmin_result + fmax_result + fdim_result + copysign_result
        + sqrt_result + cbrt_result + hypot_result + pow_result
        + exp_result + exp2_result + expm1_result
        + log_result + log2_result + log10_result + log1p_result + logb_result + static_cast<f64>(ilogb_result)
        + sin_result + cos_result + tan_result + asin_result + acos_result + atan_result + atan2_result
        + sinh_result + cosh_result + tanh_result + asinh_result + acosh_result + atanh_result
        + erf_result + erfc_result + lgamma_result + tgamma_result
        + ldexp_result + scalbn_result + scalbln_result
        + frexp_result + static_cast<f64>(frexp_exp_result) + modf_result + modf_int_result
        + nextafter_result + nexttoward_result1 + nexttoward_result2;

    return scalar_sum;
}

consteval f128 f128_test()
{
    constexpr f128 abs_result = bl::abs(-0.123_dd);
    constexpr f128 fma_result = bl::fma(0.123_dd, 4.56_dd, -0.75_dd);
                                      
    constexpr f128 floor_result     = bl::floor(2.75_dd);
    constexpr f128 ceil_result      = bl::ceil(2.25_dd);
    constexpr f128 trunc_result     = bl::trunc(-2.75_dd);
    constexpr f128 round_result     = bl::round(2.5_dd);
    constexpr long lround_result    = bl::lround(2.5_dd);
    constexpr llong llround_result  = bl::llround(2.5_dd);
    constexpr f128 nearbyint_result = bl::nearbyint(2.5_dd);
    constexpr f128 rint_result      = bl::rint(2.5_dd);
    constexpr long lrint_result     = bl::lrint(2.5_dd);
    constexpr llong llrint_result   = bl::llrint(2.5_dd);
                                      
    constexpr f128 fmod_result      = bl::fmod(5.5_dd, 2.0_dd);
    constexpr f128 remainder_result = bl::remainder(5.5_dd, 2.0_dd);
    constexpr f128 remquo_result1   = [] { int quo{}; return bl::remquo(5.5_dd, 2.0_dd, &quo); }();
    constexpr int remquo_result2    = [] { int quo{}; (void)bl::remquo(5.5_dd, 2.0_dd, &quo); return quo; }();
                                      
    constexpr f128 fmin_result     = bl::fmin(-0.25_dd, 0.5_dd);
    constexpr f128 fmax_result     = bl::fmax(-0.25_dd, 0.5_dd);
    constexpr f128 fdim_result     = bl::fdim(1.25_dd, 0.75_dd);
    constexpr f128 copysign_result = bl::copysign(0.125_dd, -1.0_dd);
                                      
    constexpr f128 sqrt_result       = bl::sqrt(0.123_dd);
    constexpr f128 cbrt_result       = bl::cbrt(0.123_dd);
    constexpr f128 hypot_result      = bl::hypot(0.3_dd, 0.4_dd);
    constexpr f128 pow_result        = bl::pow(0.123_dd, 4.56_dd);
    constexpr f128 pow_double_result = bl::pow(0.123_dd, 4.56);
                                      
    constexpr f128 exp_result   = bl::exp(0.123_dd);
    constexpr f128 exp2_result  = bl::exp2(0.123_dd);
    constexpr f128 expm1_result = bl::expm1(0.123_dd);
    constexpr f128 log_result   = bl::log(1.123_dd);
    constexpr f128 log2_result  = bl::log2(1.123_dd);
    constexpr f128 log10_result = bl::log10(1.123_dd);
    constexpr f128 log1p_result = bl::log1p(0.123_dd);
    constexpr f128 logb_result  = bl::logb(12.5_dd);
    constexpr int ilogb_result  = bl::ilogb(12.5_dd);
                                      
    constexpr f128 sin_result   = bl::sin(0.123_dd);
    constexpr f128 cos_result   = bl::cos(0.123_dd);
    constexpr f128 tan_result   = bl::tan(0.123_dd);
    constexpr f128 asin_result  = bl::asin(0.123_dd);
    constexpr f128 acos_result  = bl::acos(0.123_dd);
    constexpr f128 atan_result  = bl::atan(0.123_dd);
    constexpr f128 atan2_result = bl::atan2(0.123_dd, 0.456_dd);
                                      
    constexpr f128 sinh_result  = bl::sinh(0.123_dd);
    constexpr f128 cosh_result  = bl::cosh(0.123_dd);
    constexpr f128 tanh_result  = bl::tanh(0.123_dd);
    constexpr f128 asinh_result = bl::asinh(0.123_dd);
    constexpr f128 acosh_result = bl::acosh(1.123_dd);
    constexpr f128 atanh_result = bl::atanh(0.123_dd);
                                      
    constexpr f128 erf_result    = bl::erf(0.123_dd);
    constexpr f128 erfc_result   = bl::erfc(0.123_dd);
    constexpr f128 lgamma_result = bl::lgamma(1.123_dd);
    constexpr f128 tgamma_result = bl::tgamma(1.123_dd);
                                      
    constexpr f128 ldexp_result    = bl::ldexp(0.123_dd, 5);
    constexpr f128 scalbn_result   = bl::scalbn(0.123_dd, 5);
    constexpr f128 scalbln_result  = bl::scalbln(0.123_dd, 5L);
    constexpr f128 frexp_result    = [] { int exp{}; return bl::frexp(12.5_dd, &exp); }();
    constexpr int frexp_exp_result = [] { int exp{}; (void)bl::frexp(12.5_dd, &exp); return exp; }();
    constexpr f128 modf_result     = [] { f128 ip{}; return bl::modf(12.5_dd, &ip); }();
    constexpr f128 modf_int_result = [] { f128 ip{}; (void)bl::modf(12.5_dd, &ip); return ip; }();

    constexpr f128 nextafter_result   = bl::nextafter(0.123_dd, 0.124_dd);
    constexpr f128 nexttoward_result1 = bl::nexttoward(0.123_dd, 0.124L);
    constexpr f128 nexttoward_result2 = bl::nexttoward(0.123_dd, 0.124_dd);

    constexpr f128 scalar_sum =
        abs_result + fma_result
        + floor_result + ceil_result + trunc_result + round_result
        + f128{ static_cast<double>(lround_result) } + f128{ static_cast<double>(llround_result) }
        + nearbyint_result + rint_result
        + f128{ static_cast<double>(lrint_result) } + f128{ static_cast<double>(llrint_result) }
        + fmod_result + remainder_result + remquo_result1 + f128{ static_cast<double>(remquo_result2) }
        + fmin_result + fmax_result + fdim_result + copysign_result
        + sqrt_result + cbrt_result + hypot_result + pow_result + pow_double_result
        + exp_result + exp2_result + expm1_result
        + log_result + log2_result + log10_result + log1p_result + logb_result + f128{ static_cast<double>(ilogb_result) }
        + sin_result + cos_result + tan_result + asin_result + acos_result + atan_result + atan2_result
        + sinh_result + cosh_result + tanh_result + asinh_result + acosh_result + atanh_result
        + erf_result + erfc_result + lgamma_result + tgamma_result
        + ldexp_result + scalbn_result + scalbln_result
        + frexp_result + f128{ static_cast<double>(frexp_exp_result) } + modf_result + modf_int_result
        + nextafter_result + nexttoward_result1 + nexttoward_result2;

    return scalar_sum;
}

consteval f256 f256_test()
{
    constexpr f256 abs_result = bl::abs(-0.123_qd);
    constexpr f256 fma_result = bl::fma(0.123_qd, 4.56_qd, -0.75_qd);

    constexpr f256 floor_result     = bl::floor(2.75_qd);
    constexpr f256 ceil_result      = bl::ceil(2.25_qd);
    constexpr f256 trunc_result     = bl::trunc(-2.75_qd);
    constexpr f256 round_result     = bl::round(2.5_qd);
    constexpr long lround_result    = bl::lround(2.5_qd);
    constexpr llong llround_result  = bl::llround(2.5_qd);
    constexpr f256 nearbyint_result = bl::nearbyint(2.5_qd);
    constexpr f256 rint_result      = bl::rint(2.5_qd);
    constexpr long lrint_result     = bl::lrint(2.5_qd);
    constexpr llong llrint_result   = bl::llrint(2.5_qd);

    constexpr f256 fmod_result      = bl::fmod(5.5_qd, 2.0_qd);
    constexpr f256 remainder_result = bl::remainder(5.5_qd, 2.0_qd);
    constexpr f256 remquo_result1   = [] { int quo{}; return bl::remquo(5.5_qd, 2.0_qd, &quo); }();
    constexpr int remquo_result2    = [] { int quo{}; (void)bl::remquo(5.5_qd, 2.0_qd, &quo); return quo; }();

    constexpr f256 fmin_result     = bl::fmin(-0.25_qd, 0.5_qd);
    constexpr f256 fmax_result     = bl::fmax(-0.25_qd, 0.5_qd);
    constexpr f256 fdim_result     = bl::fdim(1.25_qd, 0.75_qd);
    constexpr f256 copysign_result = bl::copysign(0.125_qd, -1.0_qd);
                                      
    constexpr f256 sqrt_result       = bl::sqrt(0.123_qd);
    constexpr f256 cbrt_result       = bl::cbrt(0.123_qd);
    constexpr f256 hypot_result      = bl::hypot(0.3_qd, 0.4_qd);
    constexpr f256 pow_result        = bl::pow(0.123_qd, 4.56_qd);
    constexpr f256 pow_double_result = bl::pow(0.123_qd, 4.56);
                                      
    constexpr f256 exp_result   = bl::exp(0.123_qd);
    constexpr f256 exp2_result  = bl::exp2(0.123_qd);
    constexpr f256 expm1_result = bl::expm1(0.123_qd);
    constexpr f256 log_result   = bl::log(1.123_qd);
    constexpr f256 log2_result  = bl::log2(1.123_qd);
    constexpr f256 log10_result = bl::log10(1.123_qd);
    constexpr f256 log1p_result = bl::log1p(0.123_qd);
    constexpr f256 logb_result  = bl::logb(12.5_qd);
    constexpr int ilogb_result  = bl::ilogb(12.5_qd);
                                           
    constexpr f256 sin_result   = bl::sin(0.123_qd);
    constexpr f256 cos_result   = bl::cos(0.123_qd);
    constexpr f256 tan_result   = bl::tan(0.123_qd);
    constexpr f256 asin_result  = bl::asin(0.123_qd);
    constexpr f256 acos_result  = bl::acos(0.123_qd);
    constexpr f256 atan_result  = bl::atan(0.123_qd);
    constexpr f256 atan2_result = bl::atan2(0.123_qd, 0.456_qd);
                                           
    constexpr f256 sinh_result  = bl::sinh(0.123_qd);
    constexpr f256 cosh_result  = bl::cosh(0.123_qd);
    constexpr f256 tanh_result  = bl::tanh(0.123_qd);
    constexpr f256 asinh_result = bl::asinh(0.123_qd);
    constexpr f256 acosh_result = bl::acosh(1.123_qd);
    constexpr f256 atanh_result = bl::atanh(0.123_qd);
                                      
    constexpr f256 erf_result    = bl::erf(0.123_qd);
    constexpr f256 erfc_result   = bl::erfc(0.123_qd);
    constexpr f256 lgamma_result = bl::lgamma(1.123_qd);
    constexpr f256 tgamma_result = bl::tgamma(1.123_qd);
                                      
    constexpr f256 ldexp_result    = bl::ldexp(0.123_qd, 5);
    constexpr f256 scalbn_result   = bl::scalbn(0.123_qd, 5);
    constexpr f256 scalbln_result  = bl::scalbln(0.123_qd, 5L);
    constexpr f256 frexp_result    = [] { int exp{}; return bl::frexp(12.5_qd, &exp); }();
    constexpr int frexp_exp_result = [] { int exp{}; (void)bl::frexp(12.5_qd, &exp); return exp; }();
    constexpr f256 modf_result     = [] { f256 ip{}; return bl::modf(12.5_qd, &ip); }();
    constexpr f256 modf_int_result = [] { f256 ip{}; (void)bl::modf(12.5_qd, &ip); return ip; }();

    constexpr f256 nextafter_result   = bl::nextafter(0.123_qd, 0.124_qd);
    constexpr f256 nexttoward_result1 = bl::nexttoward(0.123_qd, 0.124L);
    constexpr f256 nexttoward_result2 = bl::nexttoward(0.123_qd, 0.124_qd);

    constexpr f256 scalar_sum =
        abs_result + fma_result
        + floor_result + ceil_result + trunc_result + round_result
        + f256{ static_cast<double>(lround_result) } + f256{ static_cast<double>(llround_result) }
        + nearbyint_result + rint_result
        + f256{ static_cast<double>(lrint_result) } + f256{ static_cast<double>(llrint_result) }
        + fmod_result + remainder_result + remquo_result1 + f256{ static_cast<double>(remquo_result2) }
        + fmin_result + fmax_result + fdim_result + copysign_result
        + sqrt_result + cbrt_result + hypot_result + pow_result + pow_double_result
        + exp_result + exp2_result + expm1_result
        + log_result + log2_result + log10_result + log1p_result + logb_result + f256{ static_cast<double>(ilogb_result) }
        + sin_result + cos_result + tan_result + asin_result + acos_result + atan_result + atan2_result
        + sinh_result + cosh_result + tanh_result + asinh_result + acosh_result + atanh_result
        + erf_result + erfc_result + lgamma_result + tgamma_result
        + ldexp_result + scalbn_result + scalbln_result
        + frexp_result + f256{ static_cast<double>(frexp_exp_result) } + modf_result + modf_int_result
        + nextafter_result + nexttoward_result1 + nexttoward_result2;

    return scalar_sum;
}

int main()
{
    constexpr f32 result_f32   = f32_test();
    constexpr f64 result_f64   = f64_test();
    constexpr f128 result_f128 = f128_test();
    constexpr f256 result_f256 = f256_test();

    std::cout
        << std::fixed
        << std::setprecision(std::numeric_limits<f256>::digits10)
        << result_f32 << "\n"
        << result_f64 << "\n"
        << result_f128 << "\n"
        << result_f256;

    return 0;
}
