if(NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_LIST_DIR}/fltxOptions.cmake")
endif()

function(fltx_apply_simd_options _TARGET)
    set(_FLTX_SIMD_VISIBILITY PRIVATE)
    if(ARGC GREATER 1)
        set(_FLTX_SIMD_VISIBILITY "${ARGV1}")
    endif()

    if(EMSCRIPTEN)
        target_compile_options(${_TARGET} ${_FLTX_SIMD_VISIBILITY} -msimd128)

        get_target_property(_FLTX_SIMD_TARGET_TYPE ${_TARGET} TYPE)

        if(_FLTX_SIMD_TARGET_TYPE STREQUAL "STATIC_LIBRARY")
            if(_FLTX_SIMD_VISIBILITY STREQUAL "PUBLIC" OR
               _FLTX_SIMD_VISIBILITY STREQUAL "INTERFACE")
                target_link_options(${_TARGET} INTERFACE -msimd128)
            endif()
        elseif(_FLTX_SIMD_TARGET_TYPE STREQUAL "EXECUTABLE" OR
               _FLTX_SIMD_TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR
               _FLTX_SIMD_TARGET_TYPE STREQUAL "MODULE_LIBRARY")
            target_link_options(${_TARGET} ${_FLTX_SIMD_VISIBILITY} -msimd128)
        endif()
    endif()

    if(NOT FLTX_SIMD_ENABLED)
        target_compile_definitions(${_TARGET} ${_FLTX_SIMD_VISIBILITY}
            BL_F128_ENABLE_SIMD=0
            BL_F256_ENABLE_SIMD=0
            BL_F256_ENABLE_TRIG_SIMD=0
        )
        return()
    endif()

endfunction()
