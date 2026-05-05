#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <fltx.h>
#include "example_viewer.h"
using namespace bl;
using namespace bl::literals;

template<typename T>
struct point2
{
    T x{};
    T y{};
};

struct image
{
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;
};

template<typename T>
constexpr T orient2d(point2<T> a, point2<T> b, point2<T> c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

template<typename T, std::size_t N>
consteval std::array<point2<double>, N> make_local_points(point2<T> origin, std::array<point2<T>, N> world_points)
{
    std::array<point2<double>, N> out{};
    for (std::size_t i = 0; i < N; ++i)
    {
        out[i] = {
            static_cast<double>(world_points[i].x - origin.x),
            static_cast<double>(world_points[i].y - origin.y)
        };
    }
    return out;
}

static bool contains_point(const std::array<point2<double>, 3>& triangle, point2<double> p)
{
    const double area = orient2d(triangle[0], triangle[1], triangle[2]);
    if (area == 0.0)
        return false;

    const double e0 = orient2d(triangle[0], triangle[1], p);
    const double e1 = orient2d(triangle[1], triangle[2], p);
    const double e2 = orient2d(triangle[2], triangle[0], p);

    return area > 0.0
        ? (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
        : (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
}

static void write_u16(std::ofstream& out, std::uint16_t value)
{
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8) & 0xffu));
}

static void write_u32(std::ofstream& out, std::uint32_t value)
{
    write_u16(out, static_cast<std::uint16_t>(value));
    write_u16(out, static_cast<std::uint16_t>(value >> 16));
}

static unsigned char to_byte(double x)
{
    if (x < 0.0)
        return 0;
    if (x > 255.0)
        return 255;
    return static_cast<unsigned char>(x + 0.5);
}

static image rasterize_triangle(const std::array<point2<double>, 3>& triangle, int width, int height)
{
    image out;
    out.width = width;
    out.height = height;
    out.rgb.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 42);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const point2<double> p{ static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5 };
            if (!contains_point(triangle, p))
                continue;

            const double shade = 90.0 + 120.0 * static_cast<double>(x) / static_cast<double>(width - 1);
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3u;
            out.rgb[i + 0] = to_byte(shade);
            out.rgb[i + 1] = to_byte(shade + 25.0);
            out.rgb[i + 2] = to_byte(255.0 - shade * 0.35);
        }
    }

    return out;
}

static image difference_image(const image& a, const image& b)
{
    image out;
    out.width = a.width;
    out.height = a.height;
    out.rgb.resize(a.rgb.size());

    for (std::size_t i = 0; i < a.rgb.size(); ++i)
        out.rgb[i] = static_cast<unsigned char>(std::abs(static_cast<int>(a.rgb[i]) - static_cast<int>(b.rgb[i])));

    return out;
}

static void write_bmp24(const std::filesystem::path& path, const image& img)
{
    const int row_stride = (img.width * 3 + 3) & ~3;
    const int pixel_bytes = row_stride * img.height;
    const int file_size = 54 + pixel_bytes;

    std::ofstream out(path, std::ios::binary);

    out.put('B');
    out.put('M');
    write_u32(out, static_cast<std::uint32_t>(file_size));
    write_u16(out, 0);
    write_u16(out, 0);
    write_u32(out, 54);

    write_u32(out, 40);
    write_u32(out, static_cast<std::uint32_t>(img.width));
    write_u32(out, static_cast<std::uint32_t>(img.height));
    write_u16(out, 1);
    write_u16(out, 24);
    write_u32(out, 0);
    write_u32(out, static_cast<std::uint32_t>(pixel_bytes));
    write_u32(out, 2835);
    write_u32(out, 2835);
    write_u32(out, 0);
    write_u32(out, 0);

    std::vector<unsigned char> row(static_cast<std::size_t>(row_stride));
    for (int y = img.height - 1; y >= 0; --y)
    {
        std::fill(row.begin(), row.end(), 0);
        for (int x = 0; x < img.width; ++x)
        {
            const std::size_t src_i = (static_cast<std::size_t>(y) * img.width + x) * 3u;
            const std::size_t dst_i = static_cast<std::size_t>(x) * 3u;
            row[dst_i + 0] = img.rgb[src_i + 2];
            row[dst_i + 1] = img.rgb[src_i + 1];
            row[dst_i + 2] = img.rgb[src_i + 0];
        }
        out.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
}

static void print_triangle(std::string_view name, const std::array<point2<double>, 3>& triangle)
{
    std::cout << name << "\n";
    std::cout << std::fixed << std::setprecision(1);
    for (const point2<double>& p : triangle)
        std::cout << "  (" << p.x << ", " << p.y << ")\n";
    std::cout << "  signed double-area: " << orient2d(triangle[0], triangle[1], triangle[2]) << "\n\n";
}

int main()
{
    constexpr point2<double> origin_f64{
          100000000000000000.0,
          100000000000000000.0
    };

    constexpr point2<f256> origin_f256{
          100000000000000000.0_qd,
          100000000000000000.0_qd
    };

    // These are ordinary-looking local offsets embedded in huge absolute
    // coordinates, a common shape in CAD, GIS, simulation, and meshing data.
    constexpr std::array<point2<double>, 3> world_triangle_f64{{
        { 100000000000000080.0,    100000000000000060.0 },
        { 100000000000000430.0,    100000000000000140.0 },
        { 100000000000000180.0,    100000000000000440.0 }
    }};

    constexpr std::array<point2<f256>, 3> world_triangle_f256{{
        { 100000000000000080.0_qd, 100000000000000060.0_qd },
        { 100000000000000430.0_qd, 100000000000000140.0_qd },
        { 100000000000000180.0_qd, 100000000000000440.0_qd }
    }};

    constexpr auto local_triangle_f64 = make_local_points(origin_f64, world_triangle_f64);
    constexpr auto local_triangle_f256 = make_local_points(origin_f256, world_triangle_f256);

    print_triangle("double after subtracting the world origin", local_triangle_f64);
    print_triangle("f256 after subtracting the world origin", local_triangle_f256);

    const image f64_img = rasterize_triangle(local_triangle_f64, 512, 512);
    const image f256_img = rasterize_triangle(local_triangle_f256, 512, 512);
    const image diff_img = difference_image(f64_img, f256_img);

    const std::filesystem::path f64_path = std::filesystem::absolute("geometry_f64.bmp");
    const std::filesystem::path f256_path = std::filesystem::absolute("geometry_f256.bmp");
    const std::filesystem::path diff_path = std::filesystem::absolute("geometry_difference.bmp");

    write_bmp24(f64_path, f64_img);
    write_bmp24(f256_path, f256_img);
    write_bmp24(diff_path, diff_img);

    std::cout << "wrote: " << f64_path.string() << "\n";
    std::cout << "wrote: " << f256_path.string() << "\n";
    std::cout << "wrote: " << diff_path.string() << "\n";

#ifdef FLTX_EXAMPLE_VIEWER_SDL2
    std::cout << "viewer: double | f256 | difference. Close the window to exit.\n";
    fltx_example::show_images("fltx consteval robust geometry", {
        { "double", f64_img.width, f64_img.height, f64_img.rgb.data() },
        { "f256", f256_img.width, f256_img.height, f256_img.rgb.data() },
        { "difference", diff_img.width, diff_img.height, diff_img.rgb.data() }
    });
#endif
}
