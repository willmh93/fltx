cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

find_package(Catch2 3 CONFIG REQUIRED)

include(Catch)
include(CTest)

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/fltx/*.h")
file(GLOB FLTX_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
add_executable(fltx ${FLTX_HEADERS} ${FLTX_TESTS})

if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_compile_options(fltx PRIVATE -Wa,-mbig-obj)
elseif (MSVC)
	target_compile_options(fltx PRIVATE /bigobj)
endif()

find_path(MPFR_INCLUDE_DIR NAMES mpfr.h REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)

find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)

set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

target_include_directories(fltx PRIVATE
    ${MPFR_INCLUDE_DIR}
    ${GMP_INCLUDE_DIR}
    ${FLTX_INCLUDE_DIR}
)

target_link_libraries(fltx PRIVATE
    Catch2::Catch2WithMain
    ${MPFR_LIBRARY}
    ${GMP_LIBRARY}
)

target_compile_features(fltx PRIVATE cxx_std_23)

catch_discover_tests(fltx REPORTER console EXTRA_ARGS -s)

set_property(TARGET fltx PROPERTY VS_DEBUGGER_COMMAND_ARGUMENTS "--reporter console --success")