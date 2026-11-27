// ************************************************************
// Code to compute the Mandelbrot figure
//
// This code is compile by the docker-compose into i .so lib
//
// Date : 2026-11-26
//
// ************************************************************

// Required librairies
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Persistent GPU VRAM pointer and allocated byte size to avoid reallocation overhead on every frame
static uint8_t* dev_img = nullptr;
static size_t allocated_size = 0;

// CUDA kernel: computes one pixel per thread using smooth logarithmic coloring to eliminate color banding
__global__ void mandelbrot_kernel(uint8_t* img, int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter) 
{
    // Calculate 2D pixel coordinates from block and thread indices
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Guard against out-of-bounds memory access when image dimensions are not multiples of block size
    if (px >= width || py >= height) return;

    // Map pixel coordinates to complex plane coordinates (c = x0 + i*y0)
    double x0 = x_min + (double)px / width * (x_max - x_min);
    double y0 = y_min + (double)py / height * (y_max - y_min);

    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    int iter = 0;

    // Escape-time loop using a bailout radius squared of 256.0 (instead of 4.0) for accurate smooth coloring
    while (x2 + y2 <= 256.0 && iter < max_iter) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }

    // Compute linear RGB offset (3 bytes per pixel)
    int idx = (py * width + px) * 3;
    if (iter == max_iter) {
        // Points inside the Mandelbrot set are rendered in pure black
        img[idx] = 0; img[idx+1] = 0; img[idx+2] = 0;
    } else {
        // Normalized continuous iteration count (fractional escape value)
        double log_zn = log(x2 + y2) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        double smooth_iter = iter + 1.0 - nu;

        // Cyclic trigonometric RGB palette producing seamless gradients at any zoom depth
        double freq = 0.05;
        img[idx]   = (uint8_t)(127.5 * (1.0 + sin(freq * smooth_iter + 0.0)));
        img[idx+1] = (uint8_t)(127.5 * (1.0 + sin(freq * smooth_iter + 2.094)));
        img[idx+2] = (uint8_t)(127.5 * (1.0 + sin(freq * smooth_iter + 4.188)));
    }
}

// Host entry point with C linkage callable directly from Python via ctypes
extern "C" void render_mandelbrot(uint8_t* host_img, int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter) 
{
    size_t size = (size_t)width * height * 3 * sizeof(uint8_t);

    // Reallocate GPU VRAM only when the client switches screen resolution
    if (size != allocated_size) {
        if (dev_img != nullptr) cudaFree(dev_img);
        cudaMalloc(&dev_img, size);
        allocated_size = size;
    }

    // Configure a 2D grid of 16x16 thread blocks covering the full image resolution
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    // Launch the CUDA kernel on the GPU
    mandelbrot_kernel<<<blocks, threads>>>(dev_img, width, height, x_min, x_max, y_min, y_max, max_iter);

    // Copy the rendered RGB buffer back from GPU VRAM to host RAM
    cudaMemcpy(host_img, dev_img, size, cudaMemcpyDeviceToHost);
}
