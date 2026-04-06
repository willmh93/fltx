cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

set(FLTX_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
file(GLOB FLTX_HEADERS CONFIGURE_DEPENDS "${FLTX_INCLUDE_DIR}/fltx/*.h")
file(GLOB FLTX_SOURCE CONFIGURE_DEPENDS "${FLTX_INCLUDE_DIR}/fltx/*.cpp")

add_executable(examples
    ${FLTX_SOURCE}
    ${FLTX_HEADERS}
    "${CMAKE_CURRENT_SOURCE_DIR}/basic_usage.cpp"
)
target_compile_features(examples PRIVATE cxx_std_20)
target_include_directories(examples PRIVATE
    ${FLTX_INCLUDE_DIR}
)