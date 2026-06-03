#ifndef FLTX_TESTS_METRICS_F128_PRIMARY_CASES_INCLUDED
#define FLTX_TESTS_METRICS_F128_PRIMARY_CASES_INCLUDED

#define FLTX_PRIMARY_CASE_ORDER_add "001"
#define FLTX_PRIMARY_CASE_ORDER_subtract "002"
#define FLTX_PRIMARY_CASE_ORDER_multiply "003"
#define FLTX_PRIMARY_CASE_ORDER_divide "004"
#define FLTX_PRIMARY_CASE_ORDER_sqrt "005"
#define FLTX_PRIMARY_CASE_ORDER_cbrt "006"
#define FLTX_PRIMARY_CASE_ORDER_hypot "007"
#define FLTX_PRIMARY_CASE_ORDER_sin "008"
#define FLTX_PRIMARY_CASE_ORDER_cos "009"
#define FLTX_PRIMARY_CASE_ORDER_tan "010"
#define FLTX_PRIMARY_CASE_ORDER_atan "011"
#define FLTX_PRIMARY_CASE_ORDER_atan2 "012"
#define FLTX_PRIMARY_CASE_ORDER_asin "013"
#define FLTX_PRIMARY_CASE_ORDER_acos "014"
#define FLTX_PRIMARY_CASE_ORDER_exp "015"
#define FLTX_PRIMARY_CASE_ORDER_exp2 "016"
#define FLTX_PRIMARY_CASE_ORDER_expm1 "017"
#define FLTX_PRIMARY_CASE_ORDER_log "018"
#define FLTX_PRIMARY_CASE_ORDER_log2 "019"
#define FLTX_PRIMARY_CASE_ORDER_log10 "020"
#define FLTX_PRIMARY_CASE_ORDER_log1p "021"
#define FLTX_PRIMARY_CASE_ORDER_pow "022"
#define FLTX_PRIMARY_CASE_ORDER_sinh "023"
#define FLTX_PRIMARY_CASE_ORDER_cosh "024"
#define FLTX_PRIMARY_CASE_ORDER_tanh "025"
#define FLTX_PRIMARY_CASE_ORDER_asinh "026"
#define FLTX_PRIMARY_CASE_ORDER_acosh "027"
#define FLTX_PRIMARY_CASE_ORDER_atanh "028"
#define FLTX_PRIMARY_CASE_ORDER_fma "029"
#define FLTX_PRIMARY_CASE_ORDER_fabs "030"
#define FLTX_PRIMARY_CASE_ORDER_floor "031"
#define FLTX_PRIMARY_CASE_ORDER_ceil "032"
#define FLTX_PRIMARY_CASE_ORDER_trunc "033"
#define FLTX_PRIMARY_CASE_ORDER_round "034"
#define FLTX_PRIMARY_CASE_ORDER_nearbyint "035"
#define FLTX_PRIMARY_CASE_ORDER_rint "036"
#define FLTX_PRIMARY_CASE_ORDER_lround "037"
#define FLTX_PRIMARY_CASE_ORDER_llround "038"
#define FLTX_PRIMARY_CASE_ORDER_lrint "039"
#define FLTX_PRIMARY_CASE_ORDER_llrint "040"
#define FLTX_PRIMARY_CASE_ORDER_fmod "041"
#define FLTX_PRIMARY_CASE_ORDER_remainder "042"
#define FLTX_PRIMARY_CASE_ORDER_remquo "043"
#define FLTX_PRIMARY_CASE_ORDER_fmin "044"
#define FLTX_PRIMARY_CASE_ORDER_fmax "045"
#define FLTX_PRIMARY_CASE_ORDER_fdim "046"
#define FLTX_PRIMARY_CASE_ORDER_copysign "047"
#define FLTX_PRIMARY_CASE_ORDER_ldexp "048"
#define FLTX_PRIMARY_CASE_ORDER_scalbn "049"
#define FLTX_PRIMARY_CASE_ORDER_scalbln "050"
#define FLTX_PRIMARY_CASE_ORDER_nextafter "051"
#define FLTX_PRIMARY_CASE_ORDER_nexttoward "052"
#define FLTX_PRIMARY_CASE_ORDER_ilogb "053"
#define FLTX_PRIMARY_CASE_ORDER_logb "054"
#define FLTX_PRIMARY_CASE_ORDER_frexp "055"
#define FLTX_PRIMARY_CASE_ORDER_modf "056"
#define FLTX_PRIMARY_CASE_ORDER_erf "057"
#define FLTX_PRIMARY_CASE_ORDER_erfc "058"
#define FLTX_PRIMARY_CASE_ORDER_lgamma "059"
#define FLTX_PRIMARY_CASE_ORDER_tgamma "060"
#define FLTX_PRIMARY_CASE_ORDER_equal "061"
#define FLTX_PRIMARY_CASE_ORDER_not_equal "062"
#define FLTX_PRIMARY_CASE_ORDER_less "063"
#define FLTX_PRIMARY_CASE_ORDER_greater "064"
#define FLTX_PRIMARY_CASE_ORDER_less_equal "065"
#define FLTX_PRIMARY_CASE_ORDER_greater_equal "066"

#define FLTX_PRIMARY_CASE_ORDER(NAME) FLTX_PRIMARY_CASE_ORDER_##NAME

#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_add true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_subtract true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_multiply true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_divide true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_sqrt true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_cbrt true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_hypot false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_sin true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_cos true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_tan true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_atan true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_atan2 true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_asin true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_acos true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_exp true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_exp2 false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_expm1 true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_log true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_log2 false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_log10 true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_log1p true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_pow true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_sinh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_cosh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_tanh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_asinh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_acosh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_atanh true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fma false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fabs true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_floor true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_ceil true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_trunc true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_round true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_nearbyint true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_rint true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_lround false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_llround false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_lrint false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_llrint false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fmod true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_remainder true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_remquo false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fmin false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fmax false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_fdim false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_copysign false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_ldexp true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_scalbn true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_scalbln true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_nextafter false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_nexttoward false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_ilogb false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_logb false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_frexp false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_modf false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_erf false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_erfc false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_lgamma false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_tgamma false
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_equal true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_not_equal true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_less true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_greater true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_less_equal true
#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_greater_equal true

#define FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED(NAME) FLTX_PRIMARY_CASE_QDPP_EXTRA_SUPPORTED_##NAME

#define FLTX_F128_PRIMARY_CASES(UNARY, BINARY, TERNARY, UNARY_INT, UNARY_INTEGER, FREXP) \
    BINARY(add, "[arithmetic][add]", bits_90, make_arithmetic_samples, call_add, call_add) \
    BINARY(subtract, "[arithmetic][subtract]", bits_90, make_arithmetic_samples, call_subtract, call_subtract) \
    BINARY(multiply, "[arithmetic][multiply]", bits_90, make_arithmetic_samples, call_multiply, call_multiply) \
    BINARY(divide, "[arithmetic][divide]", bits_90, make_arithmetic_samples, call_divide, call_divide) \
    UNARY(sqrt, "[sqrt]", bits_90, make_positive_samples, call_sqrt) \
    UNARY(cbrt, "[cbrt]", bits_90, make_signed_samples, call_cbrt) \
    BINARY(hypot, "[hypot]", bits_90, make_hypot_samples, call_hypot, call_hypot) \
    UNARY(sin, "[transcendental][trig][sin]", bits_80, make_trig_samples, call_sin) \
    UNARY(cos, "[transcendental][trig][cos]", bits_80, make_trig_samples, call_cos) \
    UNARY(tan, "[transcendental][trig][tan]", bits_75, make_tan_samples, call_tan) \
    UNARY(atan, "[transcendental][trig][atan]", bits_80, make_signed_samples, call_atan) \
    BINARY(atan2, "[transcendental][trig][atan2]", bits_80, make_arithmetic_samples, call_atan2, call_atan2) \
    UNARY(asin, "[transcendental][trig][asin]", bits_80, make_unit_interval_samples, call_asin) \
    UNARY(acos, "[transcendental][trig][acos]", bits_80, make_unit_interval_samples, call_acos) \
    UNARY(exp, "[transcendental][exp]", bits_80, make_exp_samples, call_exp) \
    UNARY(exp2, "[transcendental][exp][exp2]", bits_80, make_exp2_samples, call_exp2) \
    UNARY(expm1, "[transcendental][exp][expm1]", bits_80, make_expm1_samples, call_expm1) \
    UNARY(log, "[transcendental][log]", bits_80, make_positive_samples, call_log) \
    UNARY(log2, "[transcendental][log][log2]", bits_80, make_positive_samples, call_log2) \
    UNARY(log10, "[transcendental][log][log10]", bits_80, make_positive_samples, call_log10) \
    UNARY(log1p, "[transcendental][log][log1p]", bits_80, make_log1p_samples, call_log1p) \
    BINARY(pow, "[transcendental][pow]", bits_80, make_pow_samples, call_pow, call_pow) \
    UNARY(sinh, "[transcendental][hyperbolic][sinh]", bits_80, make_hyperbolic_samples, call_sinh) \
    UNARY(cosh, "[transcendental][hyperbolic][cosh]", bits_80, make_hyperbolic_samples, call_cosh) \
    UNARY(tanh, "[transcendental][hyperbolic][tanh]", bits_80, make_hyperbolic_samples, call_tanh) \
    UNARY(asinh, "[transcendental][inverse_hyperbolic][asinh]", bits_80, make_asinh_samples, call_asinh) \
    UNARY(acosh, "[transcendental][inverse_hyperbolic][acosh]", bits_80, make_acosh_samples, call_acosh) \
    UNARY(atanh, "[transcendental][inverse_hyperbolic][atanh]", bits_80, make_atanh_samples, call_atanh) \
    TERNARY(fma, "[fma]", bits_90, make_fma_samples, call_fma, call_fma_reference) \
    UNARY(fabs, "[arithmetic][fabs][abs]", bits_90, make_signed_samples, call_fabs) \
    UNARY(floor, "[rounding][floor]", bits_90, make_rounding_samples, call_floor) \
    UNARY(ceil, "[rounding][ceil]", bits_90, make_rounding_samples, call_ceil) \
    UNARY(trunc, "[rounding][trunc]", bits_90, make_rounding_samples, call_trunc) \
    UNARY(round, "[rounding][round]", bits_90, make_rounding_samples, call_round) \
    UNARY(nearbyint, "[nearbyint]", bits_90, make_rounding_samples, call_nearbyint) \
    UNARY(rint, "[rint]", bits_90, make_rounding_samples, call_rint) \
    UNARY_INTEGER(lround, "[rounding][lround]", bits_90, make_rounding_samples, call_lround, call_lround) \
    UNARY_INTEGER(llround, "[rounding][llround]", bits_90, make_rounding_samples, call_llround, call_llround) \
    UNARY_INTEGER(lrint, "[rounding][lrint]", bits_90, make_rounding_samples, call_lrint, call_lrint_reference) \
    UNARY_INTEGER(llrint, "[rounding][llrint]", bits_90, make_rounding_samples, call_llrint, call_llrint_reference) \
    BINARY(fmod, "[fmod]", domain_ideal_bits, make_remainder_samples, call_fmod, call_fmod) \
    BINARY(remainder, "[remainder]", bits_80, make_remainder_samples, call_remainder, call_remainder) \
    BINARY(remquo, "[remquo]", bits_40, make_remainder_samples, call_remquo_value, call_remainder) \
    BINARY(fmin, "[fmin]", bits_90, make_arithmetic_samples, call_fmin, call_fmin) \
    BINARY(fmax, "[fmax]", bits_90, make_arithmetic_samples, call_fmax, call_fmax) \
    BINARY(fdim, "[fdim]", bits_90, make_arithmetic_samples, call_fdim, call_fdim) \
    BINARY(copysign, "[copysign]", bits_90, make_arithmetic_samples, call_copysign, call_copysign) \
    UNARY_INT(ldexp, "[scaling][ldexp]", bits_90, make_scaling_samples, call_ldexp) \
    UNARY_INT(scalbn, "[scaling][scalbn]", bits_90, make_scaling_samples, call_scalbn) \
    UNARY_INT(scalbln, "[scaling][scalbln]", bits_90, make_scaling_samples, call_scalbln) \
    BINARY(nextafter, "[adjacent][nextafter]", bits_90, make_nextafter_samples, call_nextafter, call_nextafter) \
    BINARY(nexttoward, "[adjacent][nexttoward]", bits_90, make_nextafter_samples, call_nexttoward, call_nextafter) \
    UNARY_INTEGER(ilogb, "[ilogb]", bits_90, make_positive_samples, call_ilogb, call_ilogb) \
    UNARY(logb, "[logb]", bits_90, make_positive_samples, call_logb) \
    FREXP(frexp, "[frexp]", bits_90, make_signed_samples, call_frexp) \
    UNARY(modf, "[modf]", bits_90, make_rounding_samples, call_modf_fraction) \
    UNARY(erf, "[transcendental][erf]", bits_80, make_erf_samples, call_erf) \
    UNARY(erfc, "[transcendental][erfc]", bits_80, make_erf_samples, call_erfc) \
    UNARY(lgamma, "[transcendental][gamma][lgamma]", bits_80, make_gamma_samples, call_lgamma) \
    UNARY(tgamma, "[transcendental][gamma][tgamma]", bits_80, make_gamma_samples, call_tgamma)

#define FLTX_F128_PRIMARY_COMPARISON_CASES(BINARY_BOOL) \
    BINARY_BOOL(equal, "operator==", "[comparison][equal]", bits_90, make_comparison_samples, call_equal, call_equal) \
    BINARY_BOOL(not_equal, "operator!=", "[comparison][not_equal]", bits_90, make_comparison_samples, call_not_equal, call_not_equal) \
    BINARY_BOOL(less, "operator<", "[comparison][less]", bits_90, make_comparison_samples, call_less, call_less) \
    BINARY_BOOL(greater, "operator>", "[comparison][greater]", bits_90, make_comparison_samples, call_greater, call_greater) \
    BINARY_BOOL(less_equal, "operator<=", "[comparison][less_equal]", bits_90, make_comparison_samples, call_less_equal, call_less_equal) \
    BINARY_BOOL(greater_equal, "operator>=", "[comparison][greater_equal]", bits_90, make_comparison_samples, call_greater_equal, call_greater_equal)

#endif
