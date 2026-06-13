cmake_minimum_required(VERSION 3.21)
project(fltx LANGUAGES CXX)

file(GLOB ISOLATED_F128_MATH_CASES CONFIGURE_DEPENDS
	"f128_math/isolated_f128_*.cpp"
)
file(GLOB ISOLATED_F256_MATH_CASES CONFIGURE_DEPENDS
	"f256_math/isolated_f256_*.cpp"
)
file(GLOB ISOLATED_F128_IO_CASES CONFIGURE_DEPENDS
	"f128_io/isolated_f128_*.cpp"
)
file(GLOB ISOLATED_F256_IO_CASES CONFIGURE_DEPENDS
	"f256_io/isolated_f256_*.cpp"
)

set(ISOLATED_F128_RUNTIME_SOURCES
	"f128_math/isolated_runtime.h"
	"f128_math/isolated_runtime.cpp"
)

set(ISOLATED_F256_RUNTIME_SOURCES
	"f256_math/isolated_runtime.h"
	"f256_math/isolated_runtime.cpp"
)

set(ISOLATED_F128_MATH_SOURCES
	"f128_math/main.cpp"
	${ISOLATED_F128_RUNTIME_SOURCES}
	${ISOLATED_F128_MATH_CASES}
)

set(ISOLATED_F256_MATH_SOURCES
	"f256_math/main.cpp"
	${ISOLATED_F256_RUNTIME_SOURCES}
	${ISOLATED_F256_MATH_CASES}
)

set(ISOLATED_F128_IO_SOURCES
	"f128_io/main.cpp"
	${ISOLATED_F128_RUNTIME_SOURCES}
	${ISOLATED_F128_IO_CASES}
)

set(ISOLATED_F256_IO_SOURCES
	"f256_io/main.cpp"
	${ISOLATED_F256_RUNTIME_SOURCES}
	${ISOLATED_F256_IO_CASES}
)

add_executable(isolated_f128_math ${ISOLATED_F128_MATH_SOURCES})
add_executable(isolated_f256_math ${ISOLATED_F256_MATH_SOURCES})
add_executable(isolated_f128_io ${ISOLATED_F128_IO_SOURCES})
add_executable(isolated_f256_io ${ISOLATED_F256_IO_SOURCES})

function(configure_isolated_target _TARGET _FOLDER)
	target_link_libraries(${_TARGET} PRIVATE fltx::fltx)
	add_dependencies(${_TARGET} fltx)
	target_compile_features(${_TARGET} PRIVATE cxx_std_23)
	set_target_properties(${_TARGET} PROPERTIES FOLDER "fltx/isolated/${_FOLDER}")

	if (MSVC)
		target_compile_options(${_TARGET} PRIVATE /Bt+)
		target_link_options(${_TARGET} PRIVATE /TIME)
	endif()
endfunction()

configure_isolated_target(isolated_f128_math "f128/math")
configure_isolated_target(isolated_f256_math "f256/math")
configure_isolated_target(isolated_f128_io "f128/io")
configure_isolated_target(isolated_f256_io "f256/io")

target_include_directories(isolated_f128_math PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/f128_math")
target_include_directories(isolated_f256_math PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/f256_math")
target_include_directories(isolated_f128_io PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/f128_io"
	"${CMAKE_CURRENT_SOURCE_DIR}/f128_math"
)
target_include_directories(isolated_f256_io PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/f256_io"
	"${CMAKE_CURRENT_SOURCE_DIR}/f256_math"
)
