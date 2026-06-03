set(VCPKG_ENV_PASSTHROUGH_UNTRACKED EMSCRIPTEN_ROOT EMSDK PATH)

if(NOT DEFINED ENV{EMSCRIPTEN_ROOT})
   find_path(EMSCRIPTEN_ROOT "emcc")
else()
   set(EMSCRIPTEN_ROOT "$ENV{EMSCRIPTEN_ROOT}")
endif()

if(NOT EMSCRIPTEN_ROOT)
   if(NOT DEFINED ENV{EMSDK})
      message(FATAL_ERROR "The emcc compiler not found in PATH")
   endif()
   set(EMSCRIPTEN_ROOT "$ENV{EMSDK}/upstream/emscripten")
endif()

if(NOT EXISTS "${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake")
   message(FATAL_ERROR "Emscripten.cmake toolchain file not found")
endif()

set(VCPKG_TARGET_ARCHITECTURE wasm32)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Emscripten)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake")
set(VCPKG_MAKE_BUILD_TRIPLET "--host=wasm32-unknown-emscripten")

set(_FLTX_WASM_TARGET_FLAGS "-msimd128")
string(APPEND VCPKG_C_FLAGS " ${_FLTX_WASM_TARGET_FLAGS}")
string(APPEND VCPKG_CXX_FLAGS " ${_FLTX_WASM_TARGET_FLAGS}")
string(APPEND VCPKG_LINKER_FLAGS " ${_FLTX_WASM_TARGET_FLAGS}")

# GMP and MPFR use autotools in vcpkg. With the Emscripten chainload
# toolchain, the normal VCPKG_* flags do not always reach configure, so append
# the wasm target feature to the already-computed configure flags as well.
list(APPEND VCPKG_CONFIGURE_MAKE_OPTIONS
    "CFLAGS=$CFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
    "CXXFLAGS=$CXXFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
    "LDFLAGS=$LDFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
)
list(APPEND VCPKG_MAKE_CONFIGURE_OPTIONS
    "CFLAGS=$CFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
    "CXXFLAGS=$CXXFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
    "LDFLAGS=$LDFLAGS ${_FLTX_WASM_TARGET_FLAGS}"
)
