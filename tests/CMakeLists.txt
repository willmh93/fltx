cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

if (NOT EMSCRIPTEN)
    find_package(Catch2 3 CONFIG REQUIRED)
    include(Catch)
    include(CTest)
endif()

option(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY "Build precision_tests with FLTX_CONSTEXPR_PARITY" OFF)
option(FLTX_BENCH_TESTS_CONSTEXPR_PARITY     "Build bench_tests with FLTX_CONSTEXPR_PARITY"     OFF)

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.cpp")

file(GLOB FLTX_TESTS_PRECISION_SOURCE          CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/precision/*.cpp")
file(GLOB FLTX_TESTS_IO_SOURCE                 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB FLTX_TESTS_CONSTEXPR_PARITY_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/constexpr_parity/*.cpp")
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_HEADERS   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.h")
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_SOURCE    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")

set(FLTX_LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f128_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/f256_math.cpp"
)

function (addFltxTestLibrary _TARGET)
    add_library(${_TARGET} STATIC ${FLTX_LIBRARY_SOURCES})
    target_include_directories(${_TARGET} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_features(${_TARGET} PUBLIC cxx_std_20)
    target_compile_definitions(${_TARGET} PUBLIC ${ARGN})
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



set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

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
    set(FLTX_BENCH_TEST_LIBRARY fltx_constexpr_parity_lib)
endif()

prepareTarget(precision_tests ${FLTX_PRECISION_TEST_LIBRARY})
prepareTarget(io_tests        fltx::fltx)
prepareTarget(bench_tests     ${FLTX_BENCH_TEST_LIBRARY})
prepareTarget(parity_tests    fltx_parity_test_lib)
