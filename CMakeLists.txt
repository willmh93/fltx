cmake_minimum_required(VERSION 3.21)
project(fltx VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(FLTX_BUILD_TESTS "Build fltx tests" ${PROJECT_IS_TOP_LEVEL})
option(FLTX_BUILD_EXAMPLES "Build fltx examples" ${PROJECT_IS_TOP_LEVEL})

add_library(fltx INTERFACE)
add_library(fltx::fltx ALIAS fltx)

target_include_directories(fltx
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(fltx INTERFACE cxx_std_20)

install(TARGETS fltx
    EXPORT fltxTargets
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

include(CTest)

if(FLTX_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(FLTX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

