#ifndef FLTX_DISPATCH_INCLUDED
#define FLTX_DISPATCH_INCLUDED

#include "fltx_core.h"
#include "fltx_types.h"
#include "constexpr_dispatch.h"

bl_map_enum_to_type(bl::FloatType::F32,  bl::f32);
bl_map_enum_to_type(bl::FloatType::F64,  bl::f64);
bl_map_enum_to_type(bl::FloatType::F128, bl::f128);
bl_map_enum_to_type(bl::FloatType::F256, bl::f256);

#endif