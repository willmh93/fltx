cmake_minimum_required(VERSION 3.21)
project(fltx VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/fltxOptions.cmake)
include(cmake/fltxFastMath.cmake)
include(cmake/fltxMsvcTiming.cmake)
include(cmake/fltxSimd.cmake)

set(FLTX_PUBLIC_SCOPE PUBLIC)
add_library(fltx
    src/f128.cpp
    src/f128_math.cpp
    src/f256.cpp
    src/f256_math.cpp
    src/f256_math_transcendental.cpp
    src/f256_math_pow.cpp
)

add_library(fltx::fltx ALIAS fltx)

target_include_directories(fltx
    ${FLTX_PUBLIC_SCOPE}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# source-tree / VS-project style use
target_sources(fltx
    ${FLTX_PUBLIC_SCOPE}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/fltx.natvis>
)

if(MSVC)
    target_link_options(fltx
        ${FLTX_PUBLIC_SCOPE}
            $<BUILD_INTERFACE:/NATVIS:${CMAKE_CURRENT_SOURCE_DIR}/include/fltx.natvis>
            $<INSTALL_INTERFACE:/NATVIS:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}/fltx.natvis>
    )
endif()

target_compile_features(fltx ${FLTX_PUBLIC_SCOPE} cxx_std_23)
fltx_apply_simd_options(fltx ${FLTX_PUBLIC_SCOPE})
fltx_apply_fast_math_options(fltx)
fltx_apply_msvc_timing_reports(fltx)

install(TARGETS fltx
    EXPORT fltxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT fltxTargets
    FILE fltxTargets.cmake
    NAMESPACE fltx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fltx
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fltxConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fltxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fltxConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fltx"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/fltxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/fltxConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fltx"
)

if(FLTX_DEVELOPER_BUILD)
    include(CTest)

    add_subdirectory(examples)
    add_subdirectory(tests)
    add_subdirectory(tests/unit_compile_tests)

    if(NOT TARGET f256_expression_compile_bench)
        add_subdirectory(tests/expressions/f256)
    endif()
endif()
