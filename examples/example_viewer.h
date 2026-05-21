#pragma once
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef FLTX_EXAMPLE_VIEWER_SDL2
#ifndef SDL_MAIN_HANDLED
#define SDL_MAIN_HANDLED
#endif
#include <SDL.h>
#endif

namespace fltx_example
{
struct image_view
{
    std::string_view name;
    int width                = 0;
    int height               = 0;
    const unsigned char* rgb = nullptr;
};

inline bool show_images(std::string_view title, std::initializer_list<image_view> images)
{
#ifndef FLTX_EXAMPLE_VIEWER_SDL2
    (void)title;
    (void)images;
    return false;
#else
    if (images.size() == 0)
        return false;

    for (const image_view& img : images)
    {
        if (img.width <= 0 || img.height <= 0 || img.rgb == nullptr)
            return false;
    }

    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return false;
    }

    struct sdl_quit_guard
    {
        ~sdl_quit_guard()
        {
            SDL_Quit();
        }
    } guard;

    const std::vector<image_view> views(images);

    int max_image_w = 0;
    int max_image_h = 0;
    for (const image_view& img : views)
    {
        max_image_w = std::max(max_image_w, img.width);
        max_image_h = std::max(max_image_h, img.height);
    }

    SDL_Rect usable_bounds{ 0, 0, 1280, 720 };
    if (SDL_GetDisplayUsableBounds(0, &usable_bounds) != 0)
        usable_bounds = SDL_Rect{ 0, 0, 1280, 720 };

    const int max_window_w = std::max(320, usable_bounds.w - 80);
    const int max_window_h = std::max(240, usable_bounds.h - 120);

    int columns = std::min<int>(static_cast<int>(views.size()), 3);
    while (columns > 1 && columns * max_image_w > max_window_w)
        --columns;

    const int rows = (static_cast<int>(views.size()) + columns - 1) / columns;
    const double initial_scale = std::min(
        static_cast<double>(max_window_w) / static_cast<double>(columns * max_image_w),
        static_cast<double>(max_window_h) / static_cast<double>(rows * max_image_h));

    const int window_w = std::max(320, static_cast<int>(columns * max_image_w * initial_scale));
    const int window_h = std::max(240, static_cast<int>(rows * max_image_h * initial_scale));

    const std::string window_title(title);
    SDL_Window* window = SDL_CreateWindow(
        window_title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        window_w,
        window_h,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (window == nullptr)
    {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == nullptr)
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);

    if (renderer == nullptr)
    {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        return false;
    }

    std::vector<SDL_Texture*> textures;
    textures.reserve(views.size());
    for (const image_view& img : views)
    {
        SDL_Texture* texture = SDL_CreateTexture(
            renderer,
            SDL_PIXELFORMAT_RGB24,
            SDL_TEXTUREACCESS_STATIC,
            img.width,
            img.height);

        if (texture == nullptr)
        {
            std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << "\n";
            for (SDL_Texture* old_texture : textures)
                SDL_DestroyTexture(old_texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            return false;
        }

        SDL_UpdateTexture(texture, nullptr, img.rgb, img.width * 3);
        textures.push_back(texture);
    }

    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
                running = false;
            else if (event.type == SDL_KEYDOWN && (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q))
                running = false;
        }

        int current_w = 0;
        int current_h = 0;
        SDL_GetWindowSize(window, &current_w, &current_h);

        const double cell_w = static_cast<double>(current_w) / static_cast<double>(columns);
        const double cell_h = static_cast<double>(current_h) / static_cast<double>(rows);
        const double scale  = std::min(cell_w / static_cast<double>(max_image_w), cell_h / static_cast<double>(max_image_h));

        SDL_SetRenderDrawColor(renderer, 30, 31, 34, 255);
        SDL_RenderClear(renderer);

        for (std::size_t i = 0; i < views.size(); ++i)
        {
            const int column      = static_cast<int>(i) % columns;
            const int row         = static_cast<int>(i) / columns;
            const image_view& img = views[i];

            const int draw_w = std::max(1, static_cast<int>(static_cast<double>(img.width) * scale));
            const int draw_h = std::max(1, static_cast<int>(static_cast<double>(img.height) * scale));
            const int cell_x = static_cast<int>(static_cast<double>(column) * cell_w);
            const int cell_y = static_cast<int>(static_cast<double>(row) * cell_h);

            SDL_Rect dst{
                cell_x + static_cast<int>((cell_w - static_cast<double>(draw_w)) * 0.5),
                cell_y + static_cast<int>((cell_h - static_cast<double>(draw_h)) * 0.5),
                draw_w,
                draw_h
            };

            SDL_RenderCopy(renderer, textures[i], nullptr, &dst);
        }

        SDL_RenderPresent(renderer);
    }

    for (SDL_Texture* texture : textures)
        SDL_DestroyTexture(texture);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    return true;
#endif
}

} // namespace fltx_example
