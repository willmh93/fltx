cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

if (NOT COMMAND fltx_apply_fast_math_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fltxFastMath.cmake")
endif()

if (NOT EMSCRIPTEN)
    find_package(Catch2 3 CONFIG REQUIRED)
    include(Catch)
    include(CTest)
endif()

# These are project-local toggles, not persistent configure options. Clear
# legacy cache entries left by older option() based versions of this file.
unset(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY CACHE)
unset(FLTX_BENCH_TESTS_CONSTEXPR_PARITY CACHE)
unset(FLTX_BENCH_FAST_MATH_MODE CACHE)

if(NOT DEFINED FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    set(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY OFF)
endif()

if(NOT DEFINED FLTX_BENCH_TESTS_CONSTEXPR_PARITY)
    set(FLTX_BENCH_TESTS_CONSTEXPR_PARITY OFF)
endif()

if(NOT DEFINED FLTX_BENCH_FAST_MATH_MODE)
    set(FLTX_BENCH_FAST_MATH_MODE DEFAULT)
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
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_HEADERS   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.h")
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_SOURCE    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")

set(FLTX_TEST_MACRO_STATUS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_macro_status.cpp")
set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

set(FLTX_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256_math.cpp"
)

function (addFltxTestLibrary _TARGET)
    cmake_parse_arguments(FLTX_TEST_LIB "" "FAST_MATH_MODE" "DEFINITIONS" ${ARGN})

    add_library(${_TARGET} STATIC ${FLTX_LIBRARY_SOURCES})
    target_include_directories(${_TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_features(${_TARGET} PUBLIC cxx_std_20)
    target_compile_definitions(${_TARGET} PUBLIC ${FLTX_TEST_LIB_DEFINITIONS} ${FLTX_TEST_LIB_UNPARSED_ARGUMENTS})

    if(DEFINED FLTX_TEST_LIB_FAST_MATH_MODE)
        fltx_apply_fast_math_options(${_TARGET} MODE ${FLTX_TEST_LIB_FAST_MATH_MODE})
    else()
        fltx_apply_fast_math_options(${_TARGET})
    endif()
endfunction()

function (addFltxTestMacroStatus _TARGET)
    cmake_parse_arguments(FLTX_TEST_STATUS "" "FAST_MATH_MODE" "DEFINITIONS" ${ARGN})

    set(_STATUS_TARGET "${_TARGET}_macro_status")
    add_library(${_STATUS_TARGET} OBJECT "${FLTX_TEST_MACRO_STATUS_SOURCE}")
    target_include_directories(${_STATUS_TARGET} PRIVATE "${FLTX_INCLUDE_DIR}")
    target_compile_features(${_STATUS_TARGET} PRIVATE cxx_std_20)
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

    target_sources(${_TARGET} PRIVATE $<TARGET_OBJECTS:${_STATUS_TARGET}>)
endfunction()

set(FLTX_PARITY_TEST_DEFINITIONS
    FLTX_CONSTEXPR_PARITY
    FLTX_CONSTEXPR_PARITY_TEST_MODE
)

addFltxTestLibrary(fltx_parity_test_lib ${FLTX_PARITY_TEST_DEFINITIONS})

if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY OR FLTX_BENCH_TESTS_CONSTEXPR_PARITY)
    addFltxTestLibrary(fltx_constexpr_parity_lib FLTX_CONSTEXPR_PARITY)
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

    ${FLTX_TESTS_CONSTEXPR_BENCH_HEADERS}
    ${FLTX_TESTS_CONSTEXPR_BENCH_SOURCE}
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

    if (NOT EMSCRIPTEN)
        target_include_directories(${_TARGET} PRIVATE
            ${MPFR_INCLUDE_DIR}
            ${GMP_INCLUDE_DIR}
            ${FLTX_INCLUDE_DIR}
        )

        target_link_libraries(${_TARGET} PRIVATE
            ${_FLTX_LIBRARY}
            Catch2::Catch2WithMain
            ${MPFR_LIBRARY}
            ${GMP_LIBRARY}
        )
    endif()

    target_compile_features(${_TARGET} PRIVATE cxx_std_20)

    if (MSVC)
        target_compile_options(${_TARGET} PRIVATE /Bt+)
        target_link_options(${_TARGET} PRIVATE /TIME)
    endif()
endfunction()

set(FLTX_PRECISION_TEST_LIBRARY fltx::fltx)
if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    set(FLTX_PRECISION_TEST_LIBRARY fltx_constexpr_parity_lib)
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
elseif(NOT FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
    addFltxTestLibrary(fltx_bench_test_lib
        FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE}
    )
    set(FLTX_BENCH_TEST_LIBRARY fltx_bench_test_lib)
endif()

prepareTarget(precision_tests ${FLTX_PRECISION_TEST_LIBRARY})
prepareTarget(io_tests        fltx::fltx)
prepareTarget(bench_tests     ${FLTX_BENCH_TEST_LIBRARY})
prepareTarget(parity_tests    fltx_parity_test_lib)

if (FLTX_PRECISION_TESTS_CONSTEXPR_PARITY)
    addFltxTestMacroStatus(precision_tests DEFINITIONS FLTX_CONSTEXPR_PARITY)
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
elseif(FLTX_BENCH_FAST_MATH_MODE STREQUAL "DEFAULT")
    addFltxTestMacroStatus(bench_tests)
else()
    addFltxTestMacroStatus(bench_tests FAST_MATH_MODE ${FLTX_BENCH_FAST_MATH_MODE})
endif()

addFltxTestMacroStatus(parity_tests ${FLTX_PARITY_TEST_DEFINITIONS})
