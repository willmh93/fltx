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
if (NOT COMMAND fltx_enable_msvc_parallel_compile)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxMsvcParallel.cmake")
endif()
if (NOT COMMAND fltx_apply_simd_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxSimd.cmake")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxSources.cmake")

find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)
include(CTest)

option(FLTX_REGISTER_METRICS_CTESTS "Register fltx non-benchmark metrics cases with CTest." ON)

if(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY AND FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    message(FATAL_ERROR "FLTX_PRECISION_TESTS_CONSTEXPR_PARITY and FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL are separate modes; enable only one.")
endif()

file(GLOB_RECURSE FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB FLTX_ROOT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB_RECURSE FLTX_PUBLIC_TREE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/fltx/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.cpp")

file(GLOB FLTX_TESTS_PRECISION_SOURCE          CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/precision/*.cpp")
file(GLOB FLTX_TESTS_IO_SOURCE                 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB FLTX_TESTS_HASH_SOURCE               CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hash/*.cpp")
file(GLOB FLTX_TESTS_RANDOM_SOURCE             CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/random/*.cpp")
file(GLOB FLTX_TESTS_METRICS_HEADERS           CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/metrics/*.h")
file(GLOB FLTX_TESTS_METRICS_SOURCE            CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/metrics/*.cpp")
file(GLOB_RECURSE FLTX_TESTS_QDPP_HEADERS      CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/extern/qdpp/include/*.h")
file(GLOB FLTX_TESTS_CONSTEXPR_PARITY_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/constexpr_parity/*.cpp")

set(FLTX_TEST_MACRO_STATUS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_macro_status.cpp")
set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
set(FLTX_PUBLIC_TREE_INCLUDE_DIR "${FLTX_INCLUDE_DIR}/fltx")
get_filename_component(FLTX_METRICS_OUTPUT_ROOT_NATIVE "${CMAKE_CURRENT_SOURCE_DIR}/../res/metrics" ABSOLUTE)
file(TO_CMAKE_PATH "${FLTX_METRICS_OUTPUT_ROOT_NATIVE}" FLTX_METRICS_OUTPUT_ROOT)

function (organizeMetricsTestsSourceGroups)
    source_group(TREE "${FLTX_PUBLIC_TREE_INCLUDE_DIR}"
        PREFIX "fltx"
        FILES ${FLTX_PUBLIC_TREE_HEADERS}
    )
    source_group("fltx"
        FILES ${FLTX_ROOT_HEADERS}
    )
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/metrics"
        PREFIX "metrics"
        FILES
            ${FLTX_TESTS_METRICS_HEADERS}
            ${FLTX_TESTS_METRICS_SOURCE}
    )
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/extern/qdpp/include"
        PREFIX "extern\\qdpp"
        FILES ${FLTX_TESTS_QDPP_HEADERS}
    )
endfunction()

set(FLTX_LIBRARY_SOURCES "")
foreach(_FLTX_LIBRARY_SOURCE IN LISTS FLTX_LIBRARY_SOURCE_FILES)
    list(APPEND FLTX_LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../${_FLTX_LIBRARY_SOURCE}")
endforeach()

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
    fltx_enable_msvc_parallel_compile(${_TARGET})
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
    fltx_enable_msvc_parallel_compile(${_STATUS_TARGET})
    target_sources(${_TARGET} PRIVATE $<TARGET_OBJECTS:${_STATUS_TARGET}>)
endfunction()

function (registerFltxCatchTests _TARGET)
    cmake_parse_arguments(FLTX_CATCH "" "" "TEST_SPEC" ${ARGN})

    if(NOT BUILD_TESTING)
        return()
    endif()

    if(EMSCRIPTEN)
        add_test(NAME ${_TARGET} COMMAND ${_TARGET})
        set_tests_properties(${_TARGET} PROPERTIES LABELS "fltx;${_TARGET}")
        return()
    endif()

    set(_DISCOVERY_ARGS)
    if(FLTX_CATCH_TEST_SPEC)
        list(APPEND _DISCOVERY_ARGS TEST_SPEC ${FLTX_CATCH_TEST_SPEC})
    endif()

    catch_discover_tests(${_TARGET}
        ${_DISCOVERY_ARGS}
        TEST_PREFIX "${_TARGET}::"
        PROPERTIES LABELS "fltx;${_TARGET}"
        DISCOVERY_MODE POST_BUILD
    )
endfunction()

set(FLTX_PARITY_TEST_DEFINITIONS
    FLTX_CONSTEXPR_PARITY
    FLTX_SIMULATE_CONSTEVAL_MODE
)

addFltxTestLibrary(fltx_parity_test_lib ${FLTX_PARITY_TEST_DEFINITIONS})

if(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    addFltxTestLibrary(fltx_constexpr_parity_lib FLTX_CONSTEXPR_PARITY)
endif()

if(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
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
add_executable(hash_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_HASH_SOURCE}
)
add_executable(random_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_RANDOM_SOURCE}
)
add_executable(metrics_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_METRICS_HEADERS}
    ${FLTX_TESTS_METRICS_SOURCE}
    ${FLTX_TESTS_QDPP_HEADERS}
)
organizeMetricsTestsSourceGroups()
add_executable(parity_tests
    ${FLTX_HEADERS}
    ${FLTX_SOURCE}

    ${FLTX_TESTS_CONSTEXPR_PARITY_SOURCE}
)


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

    set(_CATCH_TARGET Catch2::Catch2WithMain)
    if(_TARGET STREQUAL "metrics_tests")
        set(_CATCH_TARGET Catch2::Catch2)
    endif()

    target_link_libraries(${_TARGET} PRIVATE
        ${_FLTX_LIBRARY}
        ${_CATCH_TARGET}
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

    if(_TARGET STREQUAL "metrics_tests")
        target_compile_definitions(${_TARGET} PRIVATE "FLTX_METRICS_OUTPUT_ROOT=\"${FLTX_METRICS_OUTPUT_ROOT}\"")
    endif()

    fltx_apply_msvc_timing_reports(${_TARGET})
    fltx_enable_msvc_parallel_compile(${_TARGET})
endfunction()

set(FLTX_PRECISION_TEST_LIBRARY fltx::fltx)
if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    set(FLTX_PRECISION_TEST_LIBRARY fltx_constexpr_parity_lib)
elseif(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    set(FLTX_PRECISION_TEST_LIBRARY fltx_simulate_consteval_lib)
endif()

prepareTarget(precision_tests ${FLTX_PRECISION_TEST_LIBRARY})
prepareTarget(io_tests        fltx::fltx)
prepareTarget(hash_tests      fltx::fltx)
prepareTarget(random_tests    fltx::fltx)
prepareTarget(metrics_tests   fltx::fltx)
prepareTarget(parity_tests    fltx_parity_test_lib)

if(VERBOSE_TESTS)
    target_compile_definitions(metrics_tests PRIVATE VERBOSE_TESTS=1)
else()
    target_compile_definitions(metrics_tests PRIVATE VERBOSE_TESTS=0)
endif()

target_include_directories(metrics_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/qdpp/include"
)

if(EMSCRIPTEN)
    target_link_options(metrics_tests PRIVATE
        -sALLOW_MEMORY_GROWTH=1
        -sINITIAL_MEMORY=268435456
        -sMAXIMUM_MEMORY=1073741824
    )
endif()

if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    addFltxTestMacroStatus(precision_tests DEFINITIONS FLTX_CONSTEXPR_PARITY)
elseif(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL)
    addFltxTestMacroStatus(precision_tests DEFINITIONS FLTX_SIMULATE_CONSTEVAL_MODE FLTX_TEST_FORCE_SIMULATED_CONSTEVAL)
else()
    addFltxTestMacroStatus(precision_tests)
endif()

addFltxTestMacroStatus(io_tests)
addFltxTestMacroStatus(hash_tests)
addFltxTestMacroStatus(random_tests)
addFltxTestMacroStatus(metrics_tests)
if(TARGET accuracy_tests)
    addFltxTestMacroStatus(accuracy_tests)
endif()

addFltxTestMacroStatus(parity_tests ${FLTX_PARITY_TEST_DEFINITIONS})

registerFltxCatchTests(precision_tests)
registerFltxCatchTests(io_tests)
registerFltxCatchTests(hash_tests)
registerFltxCatchTests(random_tests)
registerFltxCatchTests(parity_tests)
if(FLTX_REGISTER_METRICS_CTESTS)
    registerFltxCatchTests(metrics_tests TEST_SPEC "~[bench]")
endif()

if(NOT TARGET f256_expression_compile_bench)
    add_subdirectory(expressions/f256)
endif()
