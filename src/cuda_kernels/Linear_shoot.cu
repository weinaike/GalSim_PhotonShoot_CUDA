#include <curand_kernel.h>
#include "Linear_shoot.h"
#include <cstdio>

// Kernel function to perform photon shooting
__global__ void Linear_shoot_kernel(double* d_x, double* d_y, double* d_flux, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        double ud1 = curand_uniform(&state);
        double ud2 = curand_uniform(&state);

        d_x[idx] = ud1 + ud2 - 1.0;
        d_y[idx] = ud1 + ud2 - 1.0;
        d_flux[idx] = 1.0 / N;
    }
}

namespace galsim {
    void Linear_shoot_cuda(PhotonArray& photons, UniformDeviate ud) {
        int N = photons.size();

        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();

        unsigned long long seed = ud.get_init_seed();

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        Linear_shoot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_flux, N, seed);

        cudaDeviceSynchronize();

        // Error check
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        }
    }
}