cmake_minimum_required(VERSION 3.21)
project(fltx_binary_size LANGUAGES CXX)

if(NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/fltxOptions.cmake")
endif()
if(NOT COMMAND fltx_apply_fast_math_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/fltxFastMath.cmake")
endif()
if(NOT COMMAND fltx_apply_simd_options)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/fltxSimd.cmake")
endif()
if(NOT COMMAND fltx_enable_msvc_parallel_compile)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/fltxMsvcParallel.cmake")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

set(FLTX_BINARY_SIZE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${FLTX_BINARY_SIZE_GENERATED_DIR}")

set_property(GLOBAL PROPERTY FLTX_BINARY_SIZE_TARGETS "")
set_property(GLOBAL PROPERTY FLTX_BINARY_SIZE_MANIFEST_CONTENT "family,group,label,target,path\n")

function(fltx_binary_size_apply_size_link_options _TARGET)
    if(MSVC)
        target_compile_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:/Gy>
            $<$<CONFIG:Release>:/Gw>
        )
        target_link_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        target_compile_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:-ffunction-sections>
            $<$<CONFIG:Release>:-fdata-sections>
        )
        target_link_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:-Wl,--gc-sections>
        )
    endif()
endfunction()

function(fltx_binary_size_prepare_target _TARGET _FAMILY _GROUP _LABEL)
    target_link_libraries(${_TARGET} PRIVATE fltx::fltx)
    target_compile_features(${_TARGET} PRIVATE cxx_std_23)
    target_include_directories(${_TARGET} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/${_FAMILY}_math"
    )
    fltx_apply_fast_math_options(${_TARGET})
    fltx_apply_simd_options(${_TARGET})
    fltx_binary_size_apply_size_link_options(${_TARGET})
    fltx_enable_msvc_parallel_compile(${_TARGET})

    set_target_properties(${_TARGET} PROPERTIES FOLDER "fltx/binary-size/${_FAMILY}")

    set_property(GLOBAL APPEND PROPERTY FLTX_BINARY_SIZE_TARGETS "${_TARGET}")
    set_property(GLOBAL APPEND_STRING PROPERTY FLTX_BINARY_SIZE_MANIFEST_CONTENT
        "${_FAMILY},${_GROUP},${_LABEL},${_TARGET},$<TARGET_FILE:${_TARGET}>\n"
    )
endfunction()

function(fltx_binary_size_add_baseline _FAMILY)
    set(_TARGET "fltx_size_${_FAMILY}_baseline")
    set(_MAIN "${FLTX_BINARY_SIZE_GENERATED_DIR}/${_TARGET}.cpp")

    if(_FAMILY STREQUAL "f128")
        set(_HEADER "fltx/f128.h")
        set(_CALL "bl::isolated::runtime_f128(1.25)")
    elseif(_FAMILY STREQUAL "f256")
        set(_HEADER "fltx/f256.h")
        set(_CALL "bl::isolated::runtime_f256(1.25)")
    else()
        message(FATAL_ERROR "Unsupported fltx binary-size family '${_FAMILY}'.")
    endif()

    file(WRITE "${_MAIN}"
"#include <${_HEADER}>
#include \"isolated_runtime.h\"

int main()
{
    auto value = ${_CALL};
    bl::isolated::keep_value(value);
    return 0;
}
")

    add_executable(${_TARGET}
        "${_MAIN}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/${_FAMILY}_math/isolated_runtime.cpp"
    )
    fltx_binary_size_prepare_target(${_TARGET} "${_FAMILY}" "baseline" "baseline")
endfunction()

function(fltx_binary_size_add_case _FAMILY _SOURCE)
    get_filename_component(_STEM "${_SOURCE}" NAME_WE)
    string(REGEX REPLACE "^isolated_${_FAMILY}_" "" _LABEL "${_STEM}")
    set(_TARGET "fltx_size_${_FAMILY}_${_LABEL}")
    set(_MAIN "${FLTX_BINARY_SIZE_GENERATED_DIR}/${_TARGET}.cpp")

    file(WRITE "${_MAIN}"
"void ${_STEM}();

int main()
{
    ${_STEM}();
    return 0;
}
")

    add_executable(${_TARGET}
        "${_MAIN}"
        "${_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/${_FAMILY}_math/isolated_runtime.cpp"
    )
    fltx_binary_size_prepare_target(${_TARGET} "${_FAMILY}" "single" "${_LABEL}")
endfunction()

function(fltx_binary_size_add_callsite_case _FAMILY _LABEL _CASE_ID _REPEAT)
    set(_TARGET "fltx_size_${_FAMILY}_callsite_${_LABEL}_${_REPEAT}")

    add_executable(${_TARGET}
        "${CMAKE_CURRENT_SOURCE_DIR}/callsite_stress.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/${_FAMILY}_math/isolated_runtime.cpp"
    )

    if(_FAMILY STREQUAL "f128")
        target_compile_definitions(${_TARGET} PRIVATE FLTX_BINARY_SIZE_STRESS_F128=1)
    elseif(_FAMILY STREQUAL "f256")
        target_compile_definitions(${_TARGET} PRIVATE FLTX_BINARY_SIZE_STRESS_F256=1)
    else()
        message(FATAL_ERROR "Unsupported fltx binary-size family '${_FAMILY}'.")
    endif()

    target_compile_definitions(${_TARGET} PRIVATE
        FLTX_BINARY_SIZE_STRESS_CASE=${_CASE_ID}
        FLTX_BINARY_SIZE_STRESS_REPEAT=${_REPEAT}
    )

    fltx_binary_size_prepare_target(${_TARGET} "${_FAMILY}" "callsite" "${_LABEL}_${_REPEAT}")
endfunction()

fltx_binary_size_add_baseline(f128)
fltx_binary_size_add_baseline(f256)

file(GLOB FLTX_BINARY_SIZE_F128_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/f128_math/isolated_f128_*.cpp"
)
file(GLOB FLTX_BINARY_SIZE_F256_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../unit_build_tests/f256_math/isolated_f256_*.cpp"
)

foreach(_SOURCE IN LISTS FLTX_BINARY_SIZE_F128_SOURCES)
    fltx_binary_size_add_case(f128 "${_SOURCE}")
endforeach()

foreach(_SOURCE IN LISTS FLTX_BINARY_SIZE_F256_SOURCES)
    fltx_binary_size_add_case(f256 "${_SOURCE}")
endforeach()

foreach(_FAMILY f128 f256)
    fltx_binary_size_add_callsite_case(${_FAMILY} add 1 32)
    fltx_binary_size_add_callsite_case(${_FAMILY} mul 2 32)
    fltx_binary_size_add_callsite_case(${_FAMILY} div 3 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} mixed_arithmetic 4 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} sqrt 5 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} sin 6 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} cos 7 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} tan 8 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} exp 9 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} log 10 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} pow 11 16)
    fltx_binary_size_add_callsite_case(${_FAMILY} sincos 12 16)
endforeach()

set(FLTX_BINARY_SIZE_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/fltx_binary_size_manifest.csv")
set(FLTX_BINARY_SIZE_REPORT "${CMAKE_CURRENT_BINARY_DIR}/fltx_binary_size_report.csv")

get_property(FLTX_BINARY_SIZE_TARGETS GLOBAL PROPERTY FLTX_BINARY_SIZE_TARGETS)
get_property(FLTX_BINARY_SIZE_MANIFEST_CONTENT GLOBAL PROPERTY FLTX_BINARY_SIZE_MANIFEST_CONTENT)

file(GENERATE
    OUTPUT "${FLTX_BINARY_SIZE_MANIFEST}"
    CONTENT "${FLTX_BINARY_SIZE_MANIFEST_CONTENT}"
)

add_custom_target(fltx_binary_size
    DEPENDS ${FLTX_BINARY_SIZE_TARGETS}
)

if(Python3_Interpreter_FOUND)
    add_custom_target(fltx_binary_size_report
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/report_binary_size.py"
            "--manifest" "${FLTX_BINARY_SIZE_MANIFEST}"
            "--output" "${FLTX_BINARY_SIZE_REPORT}"
        DEPENDS fltx_binary_size
        USES_TERMINAL
    )
else()
    message(STATUS "Python3 interpreter not found; fltx_binary_size_report target will not be available.")
endif()
