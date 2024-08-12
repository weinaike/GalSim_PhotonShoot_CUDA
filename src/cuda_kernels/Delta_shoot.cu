#include <cuda_runtime.h>
#include <iostream>
#include "PhotonArray.h"
#include "Delta_shoot.h"

// CUDA kernel function
__global__ void delta_shoot_kernel(int N, double fluxPerPhoton, double* x, double* y, double* flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        x[i] = 0.;
        y[i] = 0.;
        flux[i] = fluxPerPhoton;
    }
}

namespace galsim {
    void Delta_shoot_cuda(PhotonArray& photons)
    {
        const int N = photons.size();
        double fluxPerPhoton = 1./N;

        double *d_x = photons.getXArrayGpu();
        double *d_y = photons.getYArrayGpu();
        double *d_flux = photons.getFluxArrayGpu();

        // Define block and grid sizes
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        // Launch kernel
        delta_shoot_kernel<<<numBlocks, blockSize>>>(N, fluxPerPhoton, d_x, d_y, d_flux);
        dbg<<"Delta Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}