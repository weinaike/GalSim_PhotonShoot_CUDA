
#ifdef ENABLE_CUDA
#include <curand_kernel.h>
#include "Nearest_shoot.h"

__global__ void Nearest_shoot_Kernel(double* x, double* y, double* flux, int N, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        curandState localState;
        curand_init(seed, i, 0, &localState);
        x[i] = curand_uniform(&localState) - 0.5;
        y[i] = curand_uniform(&localState) - 0.5;
        flux[i] = 1.0 / double(N);
    }
}


namespace galsim {
    void Nearest_shoot_cuda(PhotonArray& photons, int N, unsigned long seed)
    {
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        Nearest_shoot_Kernel<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, N, seed);
        cudaDeviceSynchronize();
    }
}

#endif