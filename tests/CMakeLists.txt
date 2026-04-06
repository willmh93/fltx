cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

find_package(Catch2 3 CONFIG REQUIRED)

include(Catch)
include(CTest)

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/fltx/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/fltx/*.cpp")
file(GLOB FLTX_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(tests ${FLTX_SOURCE} ${FLTX_HEADERS} ${FLTX_TESTS})

if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_compile_options(tests PRIVATE -Wa,-mbig-obj)
elseif (MSVC)
	target_compile_options(tests PRIVATE /bigobj)
endif()

find_path(MPFR_INCLUDE_DIR NAMES mpfr.h REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)

find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

target_include_directories(tests PRIVATE
    ${MPFR_INCLUDE_DIR}
    ${GMP_INCLUDE_DIR}
    ${FLTX_INCLUDE_DIR}
)

target_link_libraries(tests PRIVATE
    Catch2::Catch2WithMain
    ${MPFR_LIBRARY}
    ${GMP_LIBRARY}
)

target_compile_features(tests PRIVATE cxx_std_20)

if (MSVC)
    target_compile_options(tests PRIVATE /Bt+)
    target_link_options(tests PRIVATE /TIME)
endif()

catch_discover_tests(tests REPORTER console EXTRA_ARGS -s)

set_property(TARGET tests PROPERTY VS_DEBUGGER_COMMAND_ARGUMENTS "--reporter console --success")

