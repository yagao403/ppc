/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

// static inline int roundup(int a, int b) {
//     return divup(a, b) * b;
// }

#define CHECK(x) check(x, #x)

__global__ void kernel(int ny, int nx, const float *dGPU, float *rGPU)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= ny || col >= ny)
    {
        return;
    }

    if (row <= col)
    {
        // Calculate correlation coefficient
        float sum = 0.0f;
        for (int k = 0; k < nx; k++)
        {
            sum += dGPU[k + row * nx] * dGPU[k + col * nx];
        }
        rGPU[col + row * ny] = sum;
    }
    else
    {
        rGPU[col + row * ny] = 0.0f;
    }
}

void normalize(int ny, int nx, const float *data, std::vector<float> &matrix)
{
    for (int row = 0; row < ny; row++)
    {
        float sum = 0.0f;

        for (int col = 0; col < nx; col++)
        {
            sum += data[col + row * nx];
        }

        float mean = sum / nx;

        float square_sum = 0.0f;

        for (int col = 0; col < nx; col++)
        {
            float x = data[col + row * nx] - mean;
            matrix[col + row * nx] = x;
            square_sum += pow(x, 2);
        }

        square_sum = std::sqrt(square_sum);

        for (int col = 0; col < nx; col++)
        {
            matrix[col + row * nx] /= square_sum;
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<float> matrix(ny * nx);

    normalize(ny, nx, data, matrix);

    // Allocate memory & copy data to GPU
    float *dGPU = NULL;
    CHECK(cudaMalloc((void **)&dGPU, ny * nx * sizeof(float)));
    float *rGPU = NULL;
    CHECK(cudaMalloc((void **)&rGPU, ny * ny * sizeof(float)));

    // Transfer data to device
    CHECK(cudaMemcpy(dGPU, matrix.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel grid
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));

    // Run kernel
    kernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}