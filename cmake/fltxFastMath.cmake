if(NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_LIST_DIR}/fltxOptions.cmake")
endif()

set(_FLTX_FAST_MATH_SUPPORTED OFF)
if(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(_FLTX_FAST_MATH_SUPPORTED ON)
endif()

set(_FLTX_FAST_MATH_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR};${CMAKE_VS_PLATFORM_NAME}")
string(TOLOWER "${_FLTX_FAST_MATH_PROCESSOR}" _FLTX_FAST_MATH_PROCESSOR)

set(_FLTX_FAST_MATH_ENABLE_AVX2 OFF)
if(_FLTX_FAST_MATH_PROCESSOR MATCHES "(^|;)(x86_64|amd64|x64|win32|i[3-6]86)(;|$)")
    set(_FLTX_FAST_MATH_ENABLE_AVX2 ON)
endif()

function(fltx_resolve_fast_math_mode _OUT _MODE)
    string(TOUPPER "${_MODE}" _FLTX_MODE)

    if(_FLTX_MODE STREQUAL "AUTO")
        set(${_OUT} ${_FLTX_FAST_MATH_SUPPORTED} PARENT_SCOPE)
    elseif(_FLTX_MODE STREQUAL "ON")
        if(_FLTX_FAST_MATH_SUPPORTED)
            set(${_OUT} ON PARENT_SCOPE)
        else()
            message(WARNING "fltx fast-math mode is ON, but no supported option set exists for ${CMAKE_CXX_COMPILER_ID}/${CMAKE_SYSTEM_NAME}; no fast-math options will be applied")
            set(${_OUT} OFF PARENT_SCOPE)
        endif()
    elseif(_FLTX_MODE STREQUAL "OFF")
        set(${_OUT} OFF PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Invalid fltx fast-math mode '${_MODE}'. Expected AUTO, ON, or OFF.")
    endif()
endfunction()

fltx_resolve_fast_math_mode(FLTX_ENABLE_FAST_MATH "${FLTX_FAST_MATH_MODE}")

function(fltx_target_link_options_if_supported _TARGET)
    get_target_property(_TARGET_TYPE ${_TARGET} TYPE)

    if(_TARGET_TYPE STREQUAL "EXECUTABLE" OR
       _TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR
       _TARGET_TYPE STREQUAL "MODULE_LIBRARY")
        target_link_options(${_TARGET} PRIVATE ${ARGN})
    endif()
endfunction()

function(fltx_apply_fast_math_options _TARGET)
    set(_FLTX_TARGET_FAST_MATH ${FLTX_ENABLE_FAST_MATH})

    if(ARGC GREATER 1)
        cmake_parse_arguments(FLTX_APPLY_FAST_MATH "" "MODE" "" ${ARGN})
        if(DEFINED FLTX_APPLY_FAST_MATH_MODE)
            fltx_resolve_fast_math_mode(_FLTX_TARGET_FAST_MATH "${FLTX_APPLY_FAST_MATH_MODE}")
        endif()
    endif()

    if(NOT _FLTX_TARGET_FAST_MATH)
        return()
    endif()

    target_compile_definitions(${_TARGET} PRIVATE
        $<$<CONFIG:Release>:BL_FAST_MATH>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${FLTX_FMA_AVAILABLE}>>:FMA_AVAILABLE>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${FLTX_FMA_AVAILABLE}>>>:FLTX_DISABLE_FMA_AVAILABLE=1>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${FLTX_SIMD_FMA_TWO_PROD}>>:BL_FLTX_SIMD_USE_FMA_TWO_PROD=1>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${FLTX_SIMD_FMA_TWO_PROD}>>>:BL_FLTX_SIMD_USE_FMA_TWO_PROD=0>
    )

    if(MSVC)
        target_compile_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:/fp:fast>
        )

        if(_FLTX_FAST_MATH_ENABLE_AVX2)
            target_compile_options(${_TARGET} PRIVATE
                $<$<CONFIG:Release>:/arch:AVX2>
            )
        endif()
    else()
        target_compile_options(${_TARGET} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-ffp-contract=fast>
            $<$<CONFIG:Release>:-fno-math-errno>
            $<$<CONFIG:Release>:-fno-trapping-math>
        )

        if(_FLTX_FAST_MATH_ENABLE_AVX2)
            target_compile_options(${_TARGET} PRIVATE
                $<$<CONFIG:Release>:-mavx2>
                $<$<CONFIG:Release>:-mfma>
            )
        endif()

        if(FLTX_FAST_MATH_NATIVE)
            target_compile_options(${_TARGET} PRIVATE
                $<$<CONFIG:Release>:-march=native>
                $<$<CONFIG:Release>:-mtune=native>
            )
        endif()

        fltx_target_link_options_if_supported(${_TARGET}
            $<$<CONFIG:Release>:-O3>
        )
    endif()
endfunction()
