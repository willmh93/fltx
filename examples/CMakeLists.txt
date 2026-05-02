cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

function(fltx_add_example example_name)
    add_executable(${example_name} ${ARGN})
    target_link_libraries(${example_name} PRIVATE fltx::fltx)

    target_compile_features(${example_name} PRIVATE cxx_std_20)
    set_target_properties(${example_name} PROPERTIES FOLDER "fltx/examples")
endfunction()

fltx_add_example("1_example_basic"        example_basic.cpp)
fltx_add_example("2_example_constexpr_io" example_constexpr_io.cpp)
fltx_add_example("3_example_dispatch"     example_dispatch.cpp)
fltx_add_example("4_example_mandelbrot"   example_mandelbrot.cpp)
