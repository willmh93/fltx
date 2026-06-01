include_guard(GLOBAL)

set(FLTX_OPTIONS_INCLUDED TRUE)

function(fltx_define_cache_bool _NAME _DEFAULT _DOC)
    set(_DEFAULT_CACHE_NAME "_${_NAME}_DEFAULT_VALUE")

    get_property(_HAS_VALUE CACHE "${_NAME}" PROPERTY VALUE SET)
    get_property(_HAS_OLD_DEFAULT CACHE "${_DEFAULT_CACHE_NAME}" PROPERTY VALUE SET)

    if(_HAS_VALUE)
        get_property(_VALUE CACHE "${_NAME}" PROPERTY VALUE)
        if(_HAS_OLD_DEFAULT)
            get_property(_OLD_DEFAULT CACHE "${_DEFAULT_CACHE_NAME}" PROPERTY VALUE)
            if(_VALUE STREQUAL _OLD_DEFAULT)
                set(_VALUE "${_DEFAULT}")
            endif()
        endif()
    else()
        set(_VALUE "${_DEFAULT}")
    endif()

    set(${_NAME} "${_VALUE}" CACHE BOOL "${_DOC}" FORCE)
    set(${_DEFAULT_CACHE_NAME} "${_DEFAULT}" CACHE INTERNAL "Last default value for ${_NAME}" FORCE)
endfunction()

function(fltx_define_cache_string _NAME _DEFAULT _DOC)
    set(_DEFAULT_CACHE_NAME "_${_NAME}_DEFAULT_VALUE")

    get_property(_HAS_VALUE CACHE "${_NAME}" PROPERTY VALUE SET)
    get_property(_HAS_OLD_DEFAULT CACHE "${_DEFAULT_CACHE_NAME}" PROPERTY VALUE SET)

    if(_HAS_VALUE)
        get_property(_VALUE CACHE "${_NAME}" PROPERTY VALUE)
        if(_HAS_OLD_DEFAULT)
            get_property(_OLD_DEFAULT CACHE "${_DEFAULT_CACHE_NAME}" PROPERTY VALUE)
            if(_VALUE STREQUAL _OLD_DEFAULT)
                set(_VALUE "${_DEFAULT}")
            endif()
        endif()
    else()
        set(_VALUE "${_DEFAULT}")
    endif()

    set(${_NAME} "${_VALUE}" CACHE STRING "${_DOC}" FORCE)
    set(${_DEFAULT_CACHE_NAME} "${_DEFAULT}" CACHE INTERNAL "Last default value for ${_NAME}" FORCE)

    if(ARGC GREATER 3)
        set_property(CACHE ${_NAME} PROPERTY STRINGS ${ARGN})
    endif()
endfunction()

function(fltx_import_transient_cache_value _NAME _DEFAULT)
    if(DEFINED CACHE{${_NAME}})
        set(_VALUE "${${_NAME}}")
        unset(${_NAME} CACHE)
        set(${_NAME} "${_VALUE}" PARENT_SCOPE)
    elseif(NOT DEFINED ${_NAME})
        set(${_NAME} "${_DEFAULT}" PARENT_SCOPE)
    endif()
endfunction()

fltx_define_cache_bool(
    FLTX_DEVELOPER_BUILD
    "${PROJECT_IS_TOP_LEVEL}"
    "Build fltx developer targets: tests, examples, and isolated compile units."
)

fltx_define_cache_bool(
    FLTX_SIMD_ENABLED
    ON
    "Enable fltx SIMD paths when supported by the target compiler and architecture."
)

fltx_define_cache_string(
    FLTX_FAST_MATH_MODE
    ON
    "fltx fast-math mode for compiled runtime sources: ON, OFF, or AUTO."
    ON OFF AUTO
)

fltx_define_cache_bool(
    FLTX_FAST_MATH_NATIVE
    OFF
    "Tune fast-math builds for the local host CPU with -march=native/-mtune=native where supported."
)

set(_FLTX_DEFAULT_FMA_AVAILABLE ON)
if(EMSCRIPTEN)
    set(_FLTX_DEFAULT_FMA_AVAILABLE OFF)
endif()

fltx_define_cache_bool(
    FLTX_FMA_AVAILABLE
    "${_FLTX_DEFAULT_FMA_AVAILABLE}"
    "Use FMA-based error-free transforms where the source checks FMA_AVAILABLE."
)

set(_FLTX_SIMD_FMA_TWO_PROD_DEFAULT ON)
if(DEFINED CACHE{FLTX_F256_FMA_TWO_PROD} AND NOT DEFINED CACHE{FLTX_SIMD_FMA_TWO_PROD})
    get_property(_FLTX_SIMD_FMA_TWO_PROD_DEFAULT CACHE FLTX_F256_FMA_TWO_PROD PROPERTY VALUE)
endif()

fltx_define_cache_bool(
    FLTX_SIMD_FMA_TWO_PROD
    "${_FLTX_SIMD_FMA_TWO_PROD_DEFAULT}"
    "Use hardware FMA for SIMD two-product error terms when available."
)

if(DEFINED CACHE{FLTX_F256_FMA_TWO_PROD})
    unset(FLTX_F256_FMA_TWO_PROD CACHE)
endif()

fltx_define_cache_bool(
    FLTX_MSVC_TIMING_REPORTS
    "${FLTX_DEVELOPER_BUILD}"
    "Emit MSVC compiler and linker timing reports for fltx developer targets."
)

fltx_define_cache_bool(
    FLTX_MSVC_DETAILED_TIMING_REPORTS
    OFF
    "Emit detailed MSVC frontend and codegen timing reports for fltx developer targets."
)

fltx_define_cache_bool(
    VERBOSE_TESTS
    ON
    "Print quality/metrics console tables while running metrics_tests."
)

mark_as_advanced(
    FLTX_FAST_MATH_NATIVE
    FLTX_FMA_AVAILABLE
    FLTX_SIMD_FMA_TWO_PROD
    FLTX_MSVC_TIMING_REPORTS
    FLTX_MSVC_DETAILED_TIMING_REPORTS
)

fltx_import_transient_cache_value(FLTX_PRECISION_TESTS_CONSTEXPR_PARITY OFF)
fltx_import_transient_cache_value(FLTX_PRECISION_TESTS_SIMULATE_CONSTEVAL OFF)
