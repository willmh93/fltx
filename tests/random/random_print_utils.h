#ifndef FLTX_TEST_RANDOM_PRINT_UTILS_INCLUDED
#define FLTX_TEST_RANDOM_PRINT_UTILS_INCLUDED

#include <cstddef>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace bl::test::random_print
{
    class ostream_state_guard
    {
    public:
        explicit ostream_state_guard(std::ostream& stream)
            : stream_(stream), flags_(stream.flags()), precision_(stream.precision())
        {
        }

        ~ostream_state_guard()
        {
            stream_.flags(flags_);
            stream_.precision(precision_);
        }

    private:
        std::ostream& stream_;
        std::ios_base::fmtflags flags_;
        std::streamsize precision_;
    };

    template<class Real>
    [[nodiscard]] std::string to_max_digits_string(const Real& value)
    {
        std::ostringstream out;
        out << std::fixed
            << std::showpoint
            << std::setprecision(std::numeric_limits<Real>::max_digits10)
            << value;
        return out.str();
    }

    [[nodiscard]] inline std::size_t decimal_position(std::string_view value) noexcept
    {
        const std::size_t dot = value.find('.');
        return dot == std::string_view::npos ? value.size() : dot;
    }

    inline void print_aligned_samples(std::ostream& out, const std::vector<std::string>& samples)
    {
        std::size_t max_decimal_position = 0;
        for (const std::string& sample : samples)
        {
            const std::size_t position = decimal_position(sample);
            if (position > max_decimal_position)
                max_decimal_position = position;
        }

        const int index_width = samples.size() >= 100 ? 3 : 2;
        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const std::string& sample = samples[index];
            const std::size_t position = decimal_position(sample);
            out << "sample[" << std::setw(index_width) << index << "] = "
                << std::string(max_decimal_position - position, ' ')
                << sample << '\n';
        }
    }
}

#endif
