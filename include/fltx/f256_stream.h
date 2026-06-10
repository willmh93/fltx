/**
 * fltx/f256_stream.h - Stream formatting for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_STREAM_INCLUDED
#define F256_STREAM_INCLUDED

#include "fltx/detail/common_stream.h"
#include "fltx/f256_string.h"

namespace bl
{
    inline std::ostream& operator<<(std::ostream& os, const f256_s& x)
    {
        return detail::write_to_stream<detail::_f256::f256_io_traits>(os, x);
    }

    inline std::istream& operator>>(std::istream& is, f256_s& x)
    {
        return detail::read_from_stream(is, x, detail::_f256::parse);
    }

    inline std::istream& operator>>(std::istream& is, f256& x)
    {
        return detail::read_from_stream(is, static_cast<f256_s&>(x), detail::_f256::parse);
    }

} // namespace bl

#endif
