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
inline bool write_stream_special(std::ostream& os, const typename Traits::value_type& x, bool showpos, bool uppercase)
{
    const char* text = special_text<Traits>(x, uppercase);
    if (!text)
        return false;
    if (showpos && text[0] != '-')
        os << '+';
    os << text;
    return true;
}

template<class Traits>
BL_NO_INLINE std::ostream& write_to_stream(std::ostream& os, const typename Traits::value_type& x)
{
    int prec = static_cast<int>(os.precision());
    if (prec < 0)
        prec = 6;

    const auto flags = os.flags();
    const bool fixed = (flags & std::ios_base::fixed) != 0;
    const bool scientific = (flags & std::ios_base::scientific) != 0;
    const bool showpoint = (flags & std::ios_base::showpoint) != 0;
    const bool showpos = (flags & std::ios_base::showpos) != 0;
    const bool uppercase = (flags & std::ios_base::uppercase) != 0;

    if (write_stream_special<Traits>(os, x, showpos, uppercase))
        return os;

    std::string s;
    if (fixed && !scientific)
    {
        format_to_string<Traits>(s, x, prec, format_kind::fixed_frac, false);
    }
    else if (scientific && !fixed)
    {
        format_to_string<Traits>(s, x, prec, format_kind::scientific_frac, false);
    }
    else
    {
        const int sig = (prec == 0) ? 1 : prec;
        if (Traits::iszero(x))
        {
            if (showpoint)
            {
                format_to_string<Traits>(s, x, (sig > 1) ? (sig - 1) : 0, format_kind::fixed_frac, false);
                ensure_decimal_point(s);
            }
            else
            {
                s = "0";
            }
            apply_stream_decorations(s, showpos, uppercase);
            os << s;
            return os;
        }

        typename Traits::value_type m;
        int e10 = 0;
        Traits::normalize10(Traits::abs(x), m, e10);

        if (e10 >= -4 && e10 < sig)
        {
            const int frac = (sig > e10 + 1) ? (sig - (e10 + 1)) : 0;
            format_to_string<Traits>(s, x, frac, format_kind::fixed_frac, !showpoint);
        }
        else if (showpoint)
        {
            format_to_string<Traits>(s, x, (sig > 1) ? (sig - 1) : 0, format_kind::scientific_frac, false);
            ensure_decimal_point(s);
        }
        else
        {
            format_to_string<Traits>(s, x, sig, format_kind::scientific_sig, true);
        }
    }

    if (showpoint)
        ensure_decimal_point(s);
    apply_stream_decorations(s, showpos, uppercase);
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
