
#include "SBDeltaFunctionImpl_shoot.h"
#include "PhotonArray.h"


#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cuda_check.h"


__global__ void deltaFunction_shoot_kernel(double * d_x, double * d_y, double * d_flux, double fluxPerPhoton, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_x[i] = 0.0;
        d_y[i] = 0.0;
        d_flux[i] = fluxPerPhoton;
    }
}


namespace galsim {
    void SBDeltaFunctionImpl_shoot_cuda(PhotonArray& photons, double fluxPerPhoton)
    {
        const int N = photons.size();

        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();
        
        dim3 blocks((N + 256 - 1) / 256);
        dim3 threads(256);

        deltaFunction_shoot_kernel<<<blocks, threads>>>(d_x, d_y, d_flux, fluxPerPhoton, N);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    }
}

#endif