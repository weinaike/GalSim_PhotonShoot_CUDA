
#ifdef ENABLE_CUDA

#include <curand_kernel.h>
#include "SBTopHatImpl_shoot.h"
#include <cuda_runtime.h>
#include "cuda_check.h"

// CUDA kernel function
__global__ void SBTopHatImpl_shoot_kernel(double* x, double* y, double* flux, const int N, double r0, double fluxPerPhoton, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        double xu, yu, rsq;
        do {
            xu = 2.0 * curand_uniform(&state) - 1.0;
            yu = 2.0 * curand_uniform(&state) - 1.0;
            rsq = xu * xu + yu * yu;
        } while (rsq >= 1.0);
        
        x[idx] = xu * r0;
        y[idx] = yu * r0;
        flux[idx] = fluxPerPhoton;
    }
}



namespace galsim {

    void SBTopHatImpl_shoot_cuda(PhotonArray& photons, double r0, double flux, UniformDeviate ud)
    {
        const int N = photons.size();
        const double fluxPerPhoton = flux / N;

        // Allocate device memory
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();


        unsigned long long seed = ud.get_init_seed(); // or a suitable method to generate a seed

        // Launch the kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        SBTopHatImpl_shoot_kernel<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, N, r0, fluxPerPhoton, seed);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    }

}

#endif //  ENABLE_CUDA