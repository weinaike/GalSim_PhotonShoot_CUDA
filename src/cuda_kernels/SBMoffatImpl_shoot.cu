// Filename: src/cuda_kernels/SBMoffatImpl_shoot.cu

#include "SBMoffatImpl_shoot.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cuda_check.h"

__global__ void SBMoffatImpl_shoot_CUDA(
    double* x, double* y, double* flux, double fluxPerPhoton,
    int N, double fluxFactor, double beta, double rD, double* rands)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double theta = 2.0 * M_PI * rands[2*i];
        double rsq = rands[2*i+1]; // Cumulative dist function P(<r) = r^2 for unit circle
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

        // Generate random numbers
        std::vector<double> host_rands(2*N);
        for (int i = 0; i < 2*N; ++i) host_rands[i] = ud();
        double* d_rands;
        CUDA_CHECK_RETURN(cudaMalloc(&d_rands, 2*N*sizeof(double)));
        CUDA_CHECK_RETURN(cudaMemcpy(d_rands, host_rands.data(), 2*N*sizeof(double), cudaMemcpyHostToDevice));

        dim3 blocks((N + 256 - 1) / 256);
        dim3 threads(256);

        SBMoffatImpl_shoot_CUDA<<<blocks, threads>>>(d_x, d_y, d_flux, fluxPerPhoton, N, fluxFactor, beta, rD, d_rands);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaFree(d_rands));
    }
}

#endif