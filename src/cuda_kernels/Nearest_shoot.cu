#include <curand_kernel.h>
#include "Nearest_shoot.h"

__global__ void Nearest_shoot_Kernel(double* x, double* y, double* flux, int N, curandState* states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        curandState localState = states[i];
        x[i] = curand_uniform(&localState) - 0.5;
        y[i] = curand_uniform(&localState) - 0.5;
        flux[i] = 1.0 / double(N);
        states[i] = localState;
    }
}


__global__ void setup_kernel(curandState* states, int numElements, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        curand_init(seed, i, 0, &states[i]);
    }
}

extern "C" void setupCurandStates(curandState* states, int numElements, unsigned long seed)
{
    int threads = 256;
    int blocks = (numElements + threads - 1) / threads;

    setup_kernel<<<blocks, threads>>>(states, numElements, seed);
    cudaDeviceSynchronize();
}


namespace galsim {
    void Nearest_shoot_cuda(PhotonArray& photons, int N, unsigned long seed)
    {
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();
        curandState* d_states;

        cudaMalloc(&d_states, N * sizeof(curandState));
        setupCurandStates(d_states, N, seed);
        cudaDeviceSynchronize();

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        Nearest_shoot_Kernel<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, N, d_states);
        cudaDeviceSynchronize();

        cudaFree(d_states);
    }
}

