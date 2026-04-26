cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

if (NOT EMSCRIPTEN)
    find_package(Catch2 3 CONFIG REQUIRED)
    include(Catch)
    include(CTest)
endif()

file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.cpp")
file(GLOB FLTX_TESTS_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB FLTX_TESTS_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(tests 
    ${FLTX_HEADERS} 
    ${FLTX_SOURCE} 
    ${FLTX_TESTS_HEADERS}
    ${FLTX_TESTS_SOURCE}
)

if (NOT EMSCRIPTEN)
    if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	    target_compile_options(tests PRIVATE -Wa,-mbig-obj)
    elseif (MSVC)
	    target_compile_options(tests PRIVATE /bigobj)
    endif()

    find_path(MPFR_INCLUDE_DIR NAMES mpfr.h REQUIRED)
    find_library(MPFR_LIBRARY NAMES mpfr REQUIRED)

    find_path(GMP_INCLUDE_DIR NAMES gmp.h REQUIRED)
    find_library(GMP_LIBRARY NAMES gmp REQUIRED)
endif()

set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

if (NOT EMSCRIPTEN)
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
endif()

target_compile_features(tests PRIVATE cxx_std_20)

if (MSVC)
    target_compile_options(tests PRIVATE /Bt+)
    target_link_options(tests PRIVATE /TIME)
endif()