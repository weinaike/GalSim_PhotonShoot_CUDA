// Filename: src/cuda_kernels/SBMoffatImpl_shoot.cu

#include "SBMoffatImpl_shoot.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_check.h"

__global__ void SBMoffatImpl_shoot_CUDA(
    double* x, double* y, double* flux, double fluxPerPhoton,
    int N, double fluxFactor, double beta, double rD, long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curandState state;
        curand_init(seed, i, 0, &state);

        double theta = 2.0 * M_PI * curand_uniform(&state);
        double rsq = curand_uniform(&state); // Cumulative dist function P(<r) = r^2 for unit circle
        double sint, cost;

        sincos(theta, &sint, &cost);

        double newRsq = pow(1.0 - rsq * fluxFactor, 1.0 / (1.0 - beta)) - 1.0;
        double rFactor = rD * sqrt(newRsq);
        x[i] = rFactor * cost;
        y[i] = rFactor * sint;
        flux[i] = fluxPerPhoton;
    }
}

namespace galsim {
    void SBMoffatImpl_shoot_cuda(PhotonArray& photons, UniformDeviate ud, double fluxFactor, double beta, double rD, double flux)
    {
        const int N = photons.size();

        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();

        double fluxPerPhoton = flux / N;

        unsigned long long seed = ud.get_init_seed(); 
        dim3 threads(256);
        dim3 blocks((N + threads.x - 1) / threads.x);

        SBMoffatImpl_shoot_CUDA<<<blocks, threads>>>(d_x, d_y, d_flux, fluxPerPhoton, N, fluxFactor, beta, rD, seed);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    }
}

#endif