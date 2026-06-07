if(NOT DEFINED FLTX_OPTIONS_INCLUDED)
    include("${CMAKE_CURRENT_LIST_DIR}/fltxOptions.cmake")
endif()

function(fltx_enable_msvc_parallel_compile _TARGET)
    if(NOT FLTX_MSVC_PARALLEL_COMPILE)
        return()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        return()
    endif()

    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        return()
    endif()

    target_compile_options(${_TARGET} PRIVATE /MP)
endfunction()
