/**
 * fltx/detail/common_stream.h - Shared stream formatting support.
 *
 * Copyright (c) 2026 William Hemsworth
 *
 * This software is released under the MIT License.
 * See LICENSE for details.
 */

#ifndef FLTX_DETAIL_COMMON_STREAM_INCLUDED
#define FLTX_DETAIL_COMMON_STREAM_INCLUDED

#include <ios>
#include <istream>
#include <ostream>
#include <string>

#include "fltx/detail/common_io.h"

namespace bl
{
    template<std::size_t capacity>
    inline std::ostream& operator<<(std::ostream& os, const static_string<capacity>& text)
    {
        return os.write(text.data(), static_cast<std::streamsize>(text.size()));
    }

} // namespace bl

namespace bl::detail {

template<class Traits>
BL_NO_INLINE std::ostream& write_to_stream(std::ostream& os, const typename Traits::value_type& x)
{
    int prec = static_cast<int>(os.precision());
    if (prec < 0)
        prec = 6;

    std::string s;
    format_to_string<Traits>(s, x, prec, os.flags());
    os << s;
    return os;
}

template<class Value, class ParseFn>
std::istream& read_from_stream(std::istream& is, Value& x, ParseFn parse)
{
    std::string token;
    if (!(is >> token))
        return is;

    Value parsed{};
    const char* end = nullptr;
    if (!(parse(token.c_str(), parsed, &end) && end != nullptr && *end == '\0'))
    {
        is.setstate(std::ios_base::failbit);
        return is;
    }

    x = parsed;
    return is;
}

} // namespace bl::detail

#endif
