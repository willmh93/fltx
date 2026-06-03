/**
 * fltx/chrono.h - std::chrono integration for fltx duration reps.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_CHRONO_INCLUDED
#define FLTX_CHRONO_INCLUDED

#include <chrono>
#include <type_traits>

#include "fltx/f128_type.h"
#include "fltx/f256_type.h"

template<>
struct std::chrono::treat_as_floating_point<bl::f128_s> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f128> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f256_s> : std::true_type
{
};

template<>
struct std::chrono::treat_as_floating_point<bl::f256> : std::true_type
{
};

#endif
