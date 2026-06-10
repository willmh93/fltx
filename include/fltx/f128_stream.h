/**
 * fltx/f128_stream.h - Stream formatting for f128.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F128_STREAM_INCLUDED
#define F128_STREAM_INCLUDED

#include "fltx/detail/common_stream.h"
#include "fltx/f128_string.h"

namespace bl
{
    inline std::ostream& operator<<(std::ostream& os, const f128_s& x)
    {
        return detail::write_to_stream<detail::_f128::f128_io_traits>(os, x);
    }

    inline std::istream& operator>>(std::istream& is, f128_s& x)
    {
        return detail::read_from_stream(is, x, detail::_f128::parse);
    }

    inline std::istream& operator>>(std::istream& is, f128& x)
    {
        return detail::read_from_stream(is, static_cast<f128_s&>(x), detail::_f128::parse);
    }

} // namespace bl

#endif
