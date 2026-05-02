cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

if (NOT EMSCRIPTEN)
    find_package(Catch2 3 CONFIG REQUIRED)
    include(Catch)
    include(CTest)
endif()

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.cpp")

file(GLOB FLTX_TESTS_PRECISION_SOURCE          CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/precision/*.cpp")
file(GLOB FLTX_TESTS_IO_SOURCE                 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB FLTX_TESTS_CONSTEXPR_PARITY_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/constexpr_parity/*.cpp")
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_HEADERS   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.h")
file(GLOB FLTX_TESTS_CONSTEXPR_BENCH_SOURCE    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")

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

    ${FLTX_TESTS_CONSTEXPR_PARITY}
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

function (prepareTarget _TARGET)
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

prepareTarget(precision_tests)
prepareTarget(io_tests)
prepareTarget(parity_tests)
prepareTarget(bench_tests)
