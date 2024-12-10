#include "SBGaussianImpl_shoot.h"
#include <curand_kernel.h>

__global__ void SBGaussianImpl_shoot_kernel(double* x, double* y, double* flux, size_t size, double sigma, curandState* states, double fluxPerPhoton)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double theta = 2.0 * M_PI * curand_uniform(&states[i]);
        double rsq = curand_uniform(&states[i]);
        double sint, cost;
        sincos(theta, &sint, &cost);
        double rFactor = sigma * sqrt(-2.0 * log(rsq));
        x[i] = rFactor * cost;
        y[i] = rFactor * sint;
        flux[i] = fluxPerPhoton;
    }
}

__global__ void initCurand(curandState* states, unsigned long seed, int N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        curand_init(seed, i, 0, &states[i]);
}

namespace galsim {

    void SBGaussianImpl_shoot_cuda(PhotonArray& photons, UniformDeviate ud, double sigma, double fluxPerPhoton)
    {
        const int N = photons.size();

        // Allocate curand states
        curandState* d_states;
        CUDA_CHECK_RETURN(cudaMalloc(&d_states, N * sizeof(curandState)));

        // GPU random number generator initialization
        initCurand<<<(N + 255) / 256, 256>>>(d_states, unsigned(time(NULL)), N);

        // Allocate GPU memory for result arrays
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();

        // Launch kernel
        SBGaussianImpl_shoot_kernel<<<(N + 255) / 256, 256>>>(d_x, d_y, d_flux, N, sigma, d_states, fluxPerPhoton);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Free the curand states memory
        CUDA_CHECK_RETURN(cudaFree(d_states));
    }

}
