if(NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_LIST_DIR}/fltxOptions.cmake")
endif()

function(fltx_msvc_target_link_options_if_supported _TARGET)
    get_target_property(_TARGET_TYPE ${_TARGET} TYPE)

    if(_TARGET_TYPE STREQUAL "EXECUTABLE" OR
       _TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR
       _TARGET_TYPE STREQUAL "MODULE_LIBRARY")
        target_link_options(${_TARGET} PRIVATE ${ARGN})
    endif()
endfunction()

function(fltx_apply_msvc_timing_reports _TARGET)
    cmake_parse_arguments(FLTX_MSVC_TIMING "DETAILED" "" "" ${ARGN})

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        return()
    endif()

    if(NOT FLTX_MSVC_TIMING_REPORTS)
        return()
    endif()

    set(_FLTX_MSVC_TIMING_COMPILE_OPTIONS /Bt+)

    if(FLTX_MSVC_TIMING_DETAILED OR FLTX_MSVC_DETAILED_TIMING_REPORTS)
        list(APPEND _FLTX_MSVC_TIMING_COMPILE_OPTIONS
            /d1reportTime
            /d2cgsummary
        )
    endif()

    target_compile_options(${_TARGET} PRIVATE ${_FLTX_MSVC_TIMING_COMPILE_OPTIONS})
    fltx_msvc_target_link_options_if_supported(${_TARGET} /TIME)
endfunction()
