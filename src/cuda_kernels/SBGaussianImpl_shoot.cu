#ifdef ENABLE_CUDA
#include "SBGaussianImpl_shoot.h"
#include <curand_kernel.h>

__global__ void SBGaussianImpl_shoot_kernel(double* x, double* y, double* flux, size_t size, double sigma, double fluxPerPhoton, long seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        curandState state;
        curand_init(seed, i, 0, &state);

        double theta = 2.0 * M_PI * curand_uniform(&state);
        double rsq = curand_uniform(&state);
        double sint, cost;
        sincos(theta, &sint, &cost);
        double rFactor = sigma * sqrt(-2.0 * log(rsq));
        x[i] = rFactor * cost;
        y[i] = rFactor * sint;
        flux[i] = fluxPerPhoton;
    }
}

namespace galsim {

    void SBGaussianImpl_shoot_cuda(PhotonArray& photons, UniformDeviate ud, double sigma, double fluxPerPhoton)
    {
        const int N = photons.size();


        // Allocate GPU memory for result arrays
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();
        unsigned long long seed = ud.get_init_seed(); 
        // Launch kernel
        SBGaussianImpl_shoot_kernel<<<(N + 255) / 256, 256>>>(d_x, d_y, d_flux, N, sigma, fluxPerPhoton, seed);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    }

}
#endif