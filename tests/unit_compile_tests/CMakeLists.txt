cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

file(GLOB ISOLATED_F128_SOURCES CONFIGURE_DEPENDS
	"f128/isolated_f128_*.cpp"
)
file(GLOB ISOLATED_F256_SOURCES CONFIGURE_DEPENDS
	"f256/isolated_f256_*.cpp"
)

set(ISOLATED_F128_SOURCES
	"f128/main.cpp"
	"f128/isolated_runtime.h"
	"f128/isolated_runtime.cpp"
	${ISOLATED_F128_SOURCES}
)

set(ISOLATED_F256_SOURCES
	"f256/main.cpp"
	"f256/isolated_runtime.h"
	"f256/isolated_runtime.cpp"
	${ISOLATED_F256_SOURCES}
)

add_executable(isolated_f128_math ${ISOLATED_F128_SOURCES})
add_executable(isolated_f256_math ${ISOLATED_F256_SOURCES})

target_link_libraries(isolated_f128_math PRIVATE fltx::fltx)
target_link_libraries(isolated_f256_math PRIVATE fltx::fltx)

target_include_directories(isolated_f128_math PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/f128")
target_include_directories(isolated_f256_math PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/f256")

target_compile_features(isolated_f128_math PRIVATE cxx_std_20)
target_compile_features(isolated_f256_math PRIVATE cxx_std_20)

set_target_properties(isolated_f128_math PROPERTIES FOLDER "fltx/isolated/f128")
set_target_properties(isolated_f256_math PROPERTIES FOLDER "fltx/isolated/f256")

if (MSVC)
    target_compile_options(isolated_f128_math PRIVATE /Bt+)
    target_compile_options(isolated_f256_math PRIVATE /Bt+)
    target_link_options(isolated_f128_math PRIVATE /TIME)
    target_link_options(isolated_f256_math PRIVATE /TIME)
endif()
