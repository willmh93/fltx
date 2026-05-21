#include <atomic>
#include <cmath>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <fltx.h>

using namespace bl;
using namespace bl::literals;

static void write_u16(std::ofstream& out, std::uint16_t value)
{
    out.put((unsigned char)(value & 0xff));
    out.put((unsigned char)((value >> 8) & 0xff));
}

static void write_u32(std::ofstream& out, std::uint32_t value)
{
    write_u16(out, (std::uint16_t)(value));
    write_u16(out, (std::uint16_t)(value >> 16));
}

static unsigned char to_byte(double x)
{
    if (x < 0.0) x = 0.0;
    if (x > 255.0) x = 255.0;
    return static_cast<unsigned char>(x);
}

int main()
{
    using flt = f256;

    constexpr int max_iter = 20000;
    constexpr flt center_x = flt("-1.73200006480238126967529761198455");
    constexpr flt center_y = flt("0.00000019235376499049335337716270");
    constexpr flt zoom     = flt("2.0e+28");

    constexpr int width  = 1024;
    constexpr int height = 1024;

    constexpr int row_stride  = (width * 3 + 3) & ~3;
    constexpr int pixel_bytes = row_stride * height;
    constexpr int file_size   = 54 + pixel_bytes;

    std::vector<unsigned char> pixels(pixel_bytes);
    std::atomic<int> rows_done = 0;

    const flt scale_x = 4.0 / (zoom * width);
    const flt scale_y = 4.0 / (zoom * height);
    const flt half_w  = width * 0.5;
    const flt half_h  = height * 0.5;

    std::size_t thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 1;
    if (thread_count > (std::size_t)height) thread_count = height;

    const int rows_per_thread = (height + (int)thread_count - 1) / (int)thread_count;

    auto render_rows = [&](int row_begin, int row_end)
    {
        for (int row = row_begin; row < row_end; ++row)
        {
            int py = height - 1 - row;
            unsigned char* out_row = pixels.data() + row * row_stride;

            for (int px = 0; px < width; ++px)
            {
                flt cx = center_x + (flt(px) - half_w) * scale_x;
                flt cy = center_y + (flt(py) - half_h) * scale_y;
                flt x = 0, y = 0;

                int iter = 0;
                while (iter < max_iter)
                {
                    flt xx = x * x - y * y + cx;
                    y = 2.0 * x * y + cy;
                    x = xx;
                    ++iter;

                    f64 _x = (f64)x;
                    f64 _y = (f64)y;
                    if (_x*_x + _y*_y > 4.0)
                        break;
                }

                unsigned char r, g, b;
                if (iter == max_iter)
                {
                    r = g = b = 0;
                }
                else
                {
                    constexpr f64 tau = 6.28318530717958647692;
                    f64 t             = iter * 0.005;

                    r = to_byte(127.5 + 127.5 * std::cos(tau * (t + 0.00)));
                    g = to_byte(127.5 + 127.5 * std::cos(tau * (t + 0.15)));
                    b = to_byte(127.5 + 127.5 * std::cos(tau * (t + 0.32)));
                }

                int i = px * 3;
                out_row[i + 0] = b;
                out_row[i + 1] = g;
                out_row[i + 2] = r;
            }

            rows_done.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    const auto render_start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < thread_count; ++i)
    {
        int row_begin = (int)i * rows_per_thread;
        int row_end   = row_begin + rows_per_thread;

        if (row_end > height)
            row_end = height;

        if (row_begin < row_end)
            threads.emplace_back(render_rows, row_begin, row_end);
    }

    int last_percent = -1;
    while (true)
    {
        int done    = rows_done.load(std::memory_order_relaxed);
        int percent = (done * 100) / height;

        if (percent != last_percent)
        {
            std::cout << "\r" << percent << "% complete" << std::flush;
            last_percent = percent;
        }

        if (done >= height)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (std::thread& thread : threads)
        thread.join();

    const auto render_end                           = std::chrono::steady_clock::now();
    const std::chrono::duration<double> render_time = render_end - render_start;

    std::cout << "\r100% complete\n";
    std::cout << "generation took " << render_time.count() << " seconds\n";

    const std::filesystem::path output_path = std::filesystem::absolute("mandelbrot.bmp");
    std::ofstream out(output_path, std::ios::binary);

    out.put('B');
    out.put('M');
    write_u32(out, file_size);
    write_u16(out, 0);
    write_u16(out, 0);
    write_u32(out, 54);

    write_u32(out, 40);
    write_u32(out, width);
    write_u32(out, height);
    write_u16(out, 1);
    write_u16(out, 24);
    write_u32(out, 0);
    write_u32(out, pixel_bytes);
    write_u32(out, 2835);
    write_u32(out, 2835);
    write_u32(out, 0);
    write_u32(out, 0);

    out.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    std::cout << "wrote: " << output_path.string() << "\n";
}
