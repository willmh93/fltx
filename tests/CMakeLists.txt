cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

if (NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxOptions.cmake")
endif()
if (NOT COMMAND fltx_apply_fast_math_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxFastMath.cmake")
endif()
if (NOT COMMAND fltx_apply_msvc_timing_reports)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxMsvcTiming.cmake")
endif()
if (NOT COMMAND fltx_apply_simd_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxSimd.cmake")
endif()

find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)
include(CTest)

if(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY AND FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    message(FATAL_ERROR "FLTX_PRECISION_TESTS_CONSTEXPR_PARITY and FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL are separate modes; enable only one.")
endif()

if(FLTX_BENCH_TESTS_CONSTEXPR_PARITY AND FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL)
    message(FATAL_ERROR "FLTX_BENCH_TESTS_CONSTEXPR_PARITY and FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL are separate modes; enable only one.")
endif()

string(TOUPPER "${FLTX_BENCH_FAST_MATH_MODE}" FLTX_BENCH_FAST_MATH_MODE)
if(NOT FLTX_BENCH_FAST_MATH_MODE MATCHES "^(DEFAULT|AUTO|ON|OFF)$")
    message(FATAL_ERROR "Invalid bench_tests fast-math mode '${FLTX_BENCH_FAST_MATH_MODE}'. Expected DEFAULT, AUTO, ON, or OFF.")
endif()

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.cpp")

file(GLOB FLTX_TESTS_PRECISION_SOURCE          CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/precision/*.cpp")
file(GLOB FLTX_TESTS_IO_SOURCE                 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB FLTX_TESTS_CONSTEXPR_PARITY_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/constexpr_parity/*.cpp")
file(GLOB FLTX_TESTS_BENCH_HEADERS             CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.h")
set(FLTX_TESTS_BENCH_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/test_f32_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/test_f64_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/test_f256_parse_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/test_f128_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/test_f256_bench.cpp"
)

set(FLTX_TEST_MACRO_STATUS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_macro_status.cpp")
set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
get_filename_component(FLTX_BENCH_OUTPUT_ROOT_NATIVE "${CMAKE_CURRENT_SOURCE_DIR}/../res/bench" REALPATH)
file(TO_CMAKE_PATH "${FLTX_BENCH_OUTPUT_ROOT_NATIVE}" FLTX_BENCH_OUTPUT_ROOT)

set(FLTX_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256_math_transcendental.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256_math_pow.cpp"
)

function (addFltxTestLibrary _TARGET)
    cmake_parse_arguments(FLTX_TEST_LIB "" "FAST_MATH_MODE" "DEFINITIONS" ${ARGN})

    add_library(${_TARGET} STATIC ${FLTX_LIBRARY_SOURCES})
    target_include_directories(${_TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_features(${_TARGET} PUBLIC cxx_std_23)
    fltx_apply_simd_options(${_TARGET} PUBLIC)
    target_compile_definitions(${_TARGET} PUBLIC ${FLTX_TEST_LIB_DEFINITIONS} ${FLTX_TEST_LIB_UNPARSED_ARGUMENTS})

    if(DEFINED FLTX_TEST_LIB_FAST_MATH_MODE)
        fltx_apply_fast_math_options(${_TARGET} MODE ${FLTX_TEST_LIB_FAST_MATH_MODE})
    else()
        fltx_apply_fast_math_options(${_TARGET})
    endif()

    fltx_apply_msvc_timing_reports(${_TARGET})
endfunction()

function (addFltxTestMacroStatus _TARGET)
    cmake_parse_arguments(FLTX_TEST_STATUS "" "FAST_MATH_MODE" "DEFINITIONS" ${ARGN})

    set(_STATUS_TARGET "${_TARGET}_macro_status")
    add_library(${_STATUS_TARGET} OBJECT "${FLTX_TEST_MACRO_STATUS_SOURCE}")
    target_include_directories(${_STATUS_TARGET} PRIVATE "${FLTX_INCLUDE_DIR}")
    target_compile_features(${_STATUS_TARGET} PRIVATE cxx_std_23)
    fltx_apply_simd_options(${_STATUS_TARGET})
    target_compile_definitions(${_STATUS_TARGET} PRIVATE
        FLTX_TEST_NAME=\"${_TARGET}\"
        ${FLTX_TEST_STATUS_DEFINITIONS}
        ${FLTX_TEST_STATUS_UNPARSED_ARGUMENTS}
    )

    if(DEFINED FLTX_TEST_STATUS_FAST_MATH_MODE)
        fltx_apply_fast_math_options(${_STATUS_TARGET} MODE ${FLTX_TEST_STATUS_FAST_MATH_MODE})
    else()
        fltx_apply_fast_math_options(${_STATUS_TARGET})
    endif()

    fltx_apply_msvc_timing_reports(${_STATUS_TARGET})
    target_sources(${_TARGET} PRIVATE $<TARGET_OBJECTS:${_STATUS_TARGET}>)
endfunction()

set(FLTX_PARITY_TEST_DEFINITIONS
    FLTX_CONSTEXPR_PARITY
    FLTX_SIMULATE_CONSTEVAL_MODE
)

addFltxTestLibrary(fltx_parity_test_lib ${FLTX_PARITY_TEST_DEFINITIONS})

set(FLTX_NEEDS_CONSTEXPR_PARITY_LIB "${FLTX_BENCH_TESTS_CONSTEXPR_PARITY}")
if(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    set(FLTX_NEEDS_CONSTEXPR_PARITY_LIB ON)
endif()

if(FLTX_NEEDS_CONSTEXPR_PARITY_LIB)
    addFltxTestLibrary(fltx_constexpr_parity_lib FLTX_CONSTEXPR_PARITY)
endif()

set(FLTX_NEEDS_SIMULATE_CONSTEVAL_LIB "${FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL}")
if(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    set(FLTX_NEEDS_SIMULATE_CONSTEVAL_LIB ON)
endif()

if(FLTX_NEEDS_SIMULATE_CONSTEVAL_LIB)
    addFltxTestLibrary(fltx_simulate_consteval_lib FLTX_SIMULATE_CONSTEVAL_MODE)
endif()

add_executable(precision_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}
    ${FLTX_TESTS_PRECISION_SOURCE}
)
add_executable(io_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_IO_SOURCE}
)
add_executable(parity_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_CONSTEXPR_PARITY_SOURCE}
)
add_executable(bench_tests
    ${FLTX_HEADERS} 
    ${FLTX_SOURCE}

    ${FLTX_TESTS_BENCH_HEADERS}
    ${FLTX_TESTS_BENCH_SOURCE}
)

if(EMSCRIPTEN)
    add_executable(bench_tests_browser
        ${FLTX_HEADERS}
        ${FLTX_SOURCE}

        ${FLTX_TESTS_BENCH_HEADERS}
        ${FLTX_TESTS_BENCH_SOURCE}
    )
endif()



find_path(MPFR_INCLUDE_DIR NAMES mpfr.h REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)

find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

function (prepareTarget _TARGET _FLTX_LIBRARY)
    if (NOT EMSCRIPTEN)
        if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	        target_compile_options(${_TARGET} PRIVATE -Wa,-mbig-obj)
        elseif (MSVC)
	        target_compile_options(${_TARGET} PRIVATE /bigobj)
        endif()
    endif()

    target_include_directories(${_TARGET} PRIVATE
        ${FLTX_INCLUDE_DIR}
    )

    target_link_libraries(${_TARGET} PRIVATE
        ${_FLTX_LIBRARY}
        Catch2::Catch2WithMain
    )

    target_include_directories(${_TARGET} PRIVATE
        ${MPFR_INCLUDE_DIR}
        ${GMP_INCLUDE_DIR}
    )

    target_link_libraries(${_TARGET} PRIVATE
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
    )

    target_compile_features(${_TARGET} PRIVATE cxx_std_23)
    fltx_apply_simd_options(${_TARGET})

    if(EMSCRIPTEN AND _TARGET STREQUAL "bench_tests")
        target_compile_definitions(${_TARGET} PRIVATE FLTX_BENCH_RUNTIME_NODE)
        target_compile_definitions(${_TARGET} PRIVATE "FLTX_BENCH_OUTPUT_ROOT=\"${FLTX_BENCH_OUTPUT_ROOT}\"")
        # f256 atan/atan2 benchmarks build large generated sample arrays; the
        # Emscripten default 64 KiB C++ stack is too small for those tests.
        target_link_options(${_TARGET} PRIVATE
            -sNODERAWFS=1
            -sSTACK_SIZE=1048576
        )
    elseif(EMSCRIPTEN AND _TARGET STREQUAL "bench_tests_browser")
        target_compile_definitions(${_TARGET} PRIVATE FLTX_BENCH_RUNTIME_CHROME)
        target_compile_definitions(${_TARGET} PRIVATE "FLTX_BENCH_OUTPUT_ROOT=\"/res/bench\"")
        target_link_options(${_TARGET} PRIVATE
            -sSTACK_SIZE=1048576
            -sENVIRONMENT=web
            -sEXPORTED_RUNTIME_METHODS=FS
        )
    elseif(_TARGET STREQUAL "bench_tests")
        target_compile_definitions(${_TARGET} PRIVATE "FLTX_BENCH_OUTPUT_ROOT=\"${FLTX_BENCH_OUTPUT_ROOT}\"")
    endif()

    fltx_apply_msvc_timing_reports(${_TARGET})
endfunction()

set(FLTX_PRECISION_TEST_LIBRARY fltx::fltx)
if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    set(FLTX_PRECISION_TEST_LIBRARY fltx_constexpr_parity_lib)
elseif(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    set(FLTX_PRECISION_TEST_LIBRARY fltx_simulate_consteval_lib)
endif()

set(FLTX_BENCH_TEST_LIBRARY fltx::fltx)
if (FLTX_BENCH_TESTS_CONSTEXPR_PARITY)
    if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
        set(FLTX_BENCH_TEST_LIBRARY fltx_constexpr_parity_lib)
    else()
        addFltxTestLibrary(fltx_bench_test_lib
            FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
            DEFINITIONS FLTX_CONSTEXPR_PARITY
        )
        set(FLTX_BENCH_TEST_LIBRARY fltx_bench_test_lib)
    endif()
elseif(FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL)
    if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
        set(FLTX_BENCH_TEST_LIBRARY fltx_simulate_consteval_lib)
    else()
        addFltxTestLibrary(fltx_bench_test_lib
            FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
            DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE
        )
        set(FLTX_BENCH_TEST_LIBRARY fltx_bench_test_lib)
    endif()
elseif(NOT FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
    addFltxTestLibrary(fltx_bench_test_lib
        FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
    )
    set(FLTX_BENCH_TEST_LIBRARY fltx_bench_test_lib)
endif()

prepareTarget(precision_tests ${FLTX_PRECISION_TEST_LIBRARY})
prepareTarget(io_tests        fltx::fltx)
prepareTarget(parity_tests    fltx_parity_test_lib)
prepareTarget(bench_tests     ${FLTX_BENCH_TEST_LIBRARY})
if(EMSCRIPTEN)
    prepareTarget(bench_tests_browser ${FLTX_BENCH_TEST_LIBRARY})
endif()

if(FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
    fltx_apply_fast_math_options(bench_tests)
    if(EMSCRIPTEN)
        fltx_apply_fast_math_options(bench_tests_browser)
    endif()
else()
    fltx_apply_fast_math_options(bench_tests MODE ${FLTX_BENCH_FAST_MATH_MODE})
    if(EMSCRIPTEN)
        fltx_apply_fast_math_options(bench_tests_browser MODE ${FLTX_BENCH_FAST_MATH_MODE})
    endif()
endif()

if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    addFltxTestMacroStatus(precision_tests DEFINITIONS FLTX_CONSTEXPR_PARITY)
elseif(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    addFltxTestMacroStatus(precision_tests DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
else()
    addFltxTestMacroStatus(precision_tests)
endif()

addFltxTestMacroStatus(io_tests)

if (FLTX_BENCH_TESTS_CONSTEXPR_PARITY)
    if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
        addFltxTestMacroStatus(bench_tests DEFINITIONS FLTX_CONSTEXPR_PARITY)
    else()
        addFltxTestMacroStatus(bench_tests
            FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
            DEFINITIONS FLTX_CONSTEXPR_PARITY
        )
    endif()
elseif(FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL)
    if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
        addFltxTestMacroStatus(bench_tests DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
    else()
        addFltxTestMacroStatus(bench_tests
            FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
            DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL
        )
    endif()
elseif(FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
    addFltxTestMacroStatus(bench_tests)
else()
    addFltxTestMacroStatus(bench_tests FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE})
endif()

if(EMSCRIPTEN)
    if (FLTX_BENCH_TESTS_CONSTEXPR_PARITY)
        if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
            addFltxTestMacroStatus(bench_tests_browser DEFINITIONS FLTX_CONSTEXPR_PARITY)
        else()
            addFltxTestMacroStatus(bench_tests_browser
                FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
                DEFINITIONS FLTX_CONSTEXPR_PARITY
            )
        endif()
    elseif(FLTX_BENCH_TESTS_SIMULATE_CONSTEVAL)
        if (FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
            addFltxTestMacroStatus(bench_tests_browser DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
        else()
            addFltxTestMacroStatus(bench_tests_browser
                FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
                DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL
            )
        endif()
    elseif(FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
        addFltxTestMacroStatus(bench_tests_browser)
    else()
        addFltxTestMacroStatus(bench_tests_browser FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE})
    endif()
endif()

addFltxTestMacroStatus(parity_tests ${FLTX_PARITY_TEST_DEFINITIONS})

if(NOT TARGET f256_expression_compile_bench)
    add_subdirectory(expressions/f256)
endif()
