/**
 * fltx/round_options.h - Public rounding option tags.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_ROUND_OPTIONS_INCLUDED
#define FLTX_ROUND_OPTIONS_INCLUDED

namespace bl
{
    enum class round_format : unsigned char
    {
        decimals,
        significant_figures
    };

    inline constexpr round_format decimals            = round_format::decimals;
    inline constexpr round_format significant_figures = round_format::significant_figures;

} // namespace bl

#endif
