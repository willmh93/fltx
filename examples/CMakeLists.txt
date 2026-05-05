cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

option(FLTX_EXAMPLE_VIEWER "Build examples with an SDL2 image viewer" ON)

if(FLTX_EXAMPLE_VIEWER)
    find_package(SDL2 CONFIG REQUIRED)

    add_library(fltx_example_viewer INTERFACE)
    target_compile_definitions(fltx_example_viewer INTERFACE
        FLTX_EXAMPLE_VIEWER_SDL2=1
        SDL_MAIN_HANDLED
    )
    target_link_libraries(fltx_example_viewer INTERFACE
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    )
endif()

function(fltx_add_example example_name)
    add_executable(${example_name} ${ARGN})
    target_link_libraries(${example_name} PRIVATE fltx::fltx)

    if(FLTX_EXAMPLE_VIEWER)
        target_link_libraries(${example_name} PRIVATE fltx_example_viewer)
    endif()

    target_compile_options(${example_name} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/constexpr:steps10000000>
        $<$<CXX_COMPILER_ID:MSVC>:/constexpr:backtrace100>
    )

    target_compile_features(${example_name} PRIVATE cxx_std_20)
    set_target_properties(${example_name} PROPERTIES FOLDER "fltx/examples")
endfunction()

fltx_add_example("1_example_basic"                  example_basic.cpp)
fltx_add_example("2_example_constexpr_io"           example_constexpr_io.cpp)
fltx_add_example("3_example_dispatch"               example_dispatch.cpp)
fltx_add_example("4_example_mandelbrot"             example_mandelbrot.cpp)
fltx_add_example("5_example_consteval_ellipse"      example_mandelbrot.cpp)
fltx_add_example("6_example_consteval_robust_geometry" example_consteval_robust_geometry.cpp)
fltx_add_example("7_example_consteval_derivative_weights" example_consteval_derivative_weights.cpp)
