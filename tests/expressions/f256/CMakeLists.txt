if(TARGET f256_expression_compile_bench)
    return()
endif()

if(NOT DEFINED FLTX_INCLUDE_DIR)
    get_filename_component(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
endif()

if(NOT COMMAND fltx_apply_msvc_timing_reports)
    include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/fltxMsvcTiming.cmake")
endif()

set(FLTX_F256_EXPRESSION_BENCH_REPEATS "3" CACHE STRING "Number of repeats for each f256 expression compile benchmark case.")
set(FLTX_F256_EXPRESSION_BENCH_COUNT "1000" CACHE STRING "Generated function count for the large f256 expression compile benchmark cases.")

set(FLTX_F256_EXPRESSION_TARGET_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/target_generated")
file(MAKE_DIRECTORY "${FLTX_F256_EXPRESSION_TARGET_SOURCE_DIR}")
set_property(GLOBAL PROPERTY FLTX_F256_EXPRESSION_TARGET_SOURCES "")

function(fltx_f256_expression_write_source _NAME _TEXT)
    set(_PATH "${FLTX_F256_EXPRESSION_TARGET_SOURCE_DIR}/${_NAME}")
    file(WRITE "${_PATH}" "${_TEXT}")
    set_property(GLOBAL APPEND PROPERTY FLTX_F256_EXPRESSION_TARGET_SOURCES "${_PATH}")
endfunction()

function(fltx_f256_expression_make_source _NAME _COUNT _KIND)
    set(_TEXT "#include \"f256.h\"\n\n")
    string(APPEND _TEXT "#if defined(_MSC_VER)\n")
    string(APPEND _TEXT "#define FLTX_EXPR_NOINLINE __declspec(noinline)\n")
    string(APPEND _TEXT "#elif defined(__GNUC__) || defined(__clang__)\n")
    string(APPEND _TEXT "#define FLTX_EXPR_NOINLINE __attribute__((noinline))\n")
    string(APPEND _TEXT "#else\n")
    string(APPEND _TEXT "#define FLTX_EXPR_NOINLINE\n")
    string(APPEND _TEXT "#endif\n\n")
    string(APPEND _TEXT "namespace bench {\n")

    math(EXPR _LAST_INDEX "${_COUNT} - 1")
    foreach(_I RANGE 0 ${_LAST_INDEX})
        if(_KIND STREQUAL "pass")
            set(_EXPR "return a;")
        elseif(_KIND STREQUAL "add")
            set(_EXPR "return a + b;")
        elseif(_KIND STREQUAL "muladdmul")
            set(_EXPR "return a * b + c * d;")
        elseif(_KIND STREQUAL "muladdmul_direct")
            set(_EXPR "return bl::f256{ bl::detail::_f256_runtime::mul_add_mul(static_cast<const bl::f256_s&>(a), static_cast<const bl::f256_s&>(b), static_cast<const bl::f256_s&>(c), static_cast<const bl::f256_s&>(d)) };")
        elseif(_KIND STREQUAL "eager_steps")
            set(_EXPR "bl::f256 ab = a * b; bl::f256 cd = c * d; return ab + cd;")
        elseif(_KIND STREQUAL "mixed")
            math(EXPR _VARIANT "${_I} % 8")
            if(_VARIANT EQUAL 0)
                set(_EXPR "return a * b + c * d;")
            elseif(_VARIANT EQUAL 1)
                set(_EXPR "return (a * b + c) / (d + 1.0);")
            elseif(_VARIANT EQUAL 2)
                set(_EXPR "return a * 2.0 + b * 3.0 + c;")
            elseif(_VARIANT EQUAL 3)
                set(_EXPR "return (a + b + c) / d;")
            elseif(_VARIANT EQUAL 4)
                set(_EXPR "return a - b * c;")
            elseif(_VARIANT EQUAL 5)
                set(_EXPR "return (a * b - c * d) / (a - 2.0);")
            elseif(_VARIANT EQUAL 6)
                set(_EXPR "return (a * 0.5 + b * 0.25) - c;")
            else()
                set(_EXPR "return (a * b + c * d + a * c + b * d);")
            endif()
        else()
            message(FATAL_ERROR "Unknown f256 expression compile-test kind '${_KIND}'.")
        endif()

        string(APPEND _TEXT "FLTX_EXPR_NOINLINE bl::f256 ${_KIND}_${_COUNT}_${_I}(bl::f256 a, bl::f256 b, bl::f256 c, bl::f256 d) { ${_EXPR} }\n")
    endforeach()

    string(APPEND _TEXT "}\n")
    fltx_f256_expression_write_source("${_NAME}" "${_TEXT}")
endfunction()

fltx_f256_expression_write_source("main.cpp" "int main() { return 0; }\n")
fltx_f256_expression_make_source("expr_pass_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "pass")
fltx_f256_expression_make_source("expr_add_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "add")
fltx_f256_expression_make_source("expr_muladdmul_100.cpp" 100 "muladdmul")
fltx_f256_expression_make_source("expr_muladdmul_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "muladdmul")
fltx_f256_expression_make_source("expr_muladdmul_direct_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "muladdmul_direct")
fltx_f256_expression_make_source("expr_eager_steps_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "eager_steps")
fltx_f256_expression_make_source("expr_mixed_${FLTX_F256_EXPRESSION_BENCH_COUNT}.cpp" "${FLTX_F256_EXPRESSION_BENCH_COUNT}" "mixed")

get_property(FLTX_F256_EXPRESSION_TARGET_SOURCES GLOBAL PROPERTY FLTX_F256_EXPRESSION_TARGET_SOURCES)
add_executable(f256_expression_compile_tests EXCLUDE_FROM_ALL
    ${FLTX_F256_EXPRESSION_TARGET_SOURCES}
)

target_include_directories(f256_expression_compile_tests PRIVATE "${FLTX_INCLUDE_DIR}")
target_compile_features(f256_expression_compile_tests PRIVATE cxx_std_23)

if(TARGET fltx::fltx)
    target_link_libraries(f256_expression_compile_tests PRIVATE fltx::fltx)
elseif(TARGET fltx)
    target_link_libraries(f256_expression_compile_tests PRIVATE fltx)
endif()

if(MSVC)
    target_compile_options(f256_expression_compile_tests PRIVATE /bigobj)
endif()

fltx_apply_msvc_timing_reports(f256_expression_compile_tests)

add_custom_target(f256_expression_compile_bench
    COMMAND ${CMAKE_COMMAND}
        -DFLTX_F256_EXPR_BENCH_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DFLTX_F256_EXPR_BENCH_INCLUDE_DIR=${FLTX_INCLUDE_DIR}
        -DFLTX_F256_EXPR_BENCH_COMPILER=${CMAKE_CXX_COMPILER}
        -DFLTX_F256_EXPR_BENCH_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
        -DFLTX_F256_EXPR_BENCH_REPEATS=${FLTX_F256_EXPRESSION_BENCH_REPEATS}
        -DFLTX_F256_EXPR_BENCH_COUNT=${FLTX_F256_EXPRESSION_BENCH_COUNT}
        -P ${CMAKE_CURRENT_LIST_DIR}/run_compile_bench.cmake
    USES_TERMINAL
    VERBATIM
    COMMENT "Timing f256 expression compile cases"
)
