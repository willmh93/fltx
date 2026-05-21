set(FLTX_LIBRARY_SOURCE_FILES
    src/f128.cpp
    src/f128_math.cpp
    src/f128_math_erf_gamma.cpp
    src/f128_math_exp_log_pow.cpp
    src/f128_math_trig.cpp
    src/f256.cpp
    src/f256_math.cpp
    src/f256_math_gamma.cpp
    src/f256_math_transcendental.cpp
    src/f256_math_pow.cpp
)

get_filename_component(FLTX_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(FLTX_LIBRARY_SOURCE_FILE_PATHS "")
foreach(_FLTX_LIBRARY_SOURCE IN LISTS FLTX_LIBRARY_SOURCE_FILES)
    get_filename_component(_FLTX_LIBRARY_SOURCE_PATH "${FLTX_PROJECT_ROOT}/${_FLTX_LIBRARY_SOURCE}" ABSOLUTE)
    list(APPEND FLTX_LIBRARY_SOURCE_FILE_PATHS "${_FLTX_LIBRARY_SOURCE_PATH}")
endforeach()

file(GLOB_RECURSE FLTX_LIBRARY_HEADER_FILES CONFIGURE_DEPENDS
    "${FLTX_PROJECT_ROOT}/include/*.h"
)
list(SORT FLTX_LIBRARY_HEADER_FILES)

set(FLTX_NATVIS_FILE "${FLTX_PROJECT_ROOT}/include/fltx.natvis")
