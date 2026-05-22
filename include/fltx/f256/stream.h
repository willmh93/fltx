/**
 * fltx/f256/stream.h - Ostream formatting for f256.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef F256_STREAM_INCLUDED
#define F256_STREAM_INCLUDED

#include "fltx/detail/common/stream.h"
#include "fltx/f256/string.h"

namespace bl
{
    inline std::ostream& operator<<(std::ostream& os, const f256_s& x)
    {
        return detail::write_to_stream<detail::_f256::f256_io_traits>(os, x);
    }

} // namespace bl

#endif
