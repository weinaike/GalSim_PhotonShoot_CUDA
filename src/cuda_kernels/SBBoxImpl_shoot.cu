#include <curand_kernel.h>
#include "SBBoxImpl_shoot.h"
#include "galsim/PhotonArray.h"

#include "SBBoxImpl_shoot.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>

#ifdef ENABLE_CUDA

#include "cuda_check.h"

struct SBounds {
    double width, height;
};

template <typename T>
__global__ void SBBoxImpl_shoot_kernel(
    T* x, T* y, T* flux, const SBounds bounds, const double fluxPerPhoton, int N, curandState* state)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        curandState localState = state[id];
        double randX = curand_uniform(&localState) - 0.5;
        double randY = curand_uniform(&localState) - 0.5;
        x[id] = randX * bounds.width;
        y[id] = randY * bounds.height;
        flux[id] = fluxPerPhoton;
        state[id] = localState;
    }
}

// CUDA kernel for setting up the random states
__global__ void setup_kernel(curandState *state, int seed, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
        curand_init(seed, id, 0, &state[id]);
}


namespace galsim {
        
    void SBBoxImpl_shoot_cuda(PhotonArray& photons, double width, double height, double flux, UniformDeviate ud)
    {
        const int N = photons.size();
        double fluxPerPhoton = flux / N;

        // 在GPU上分配内存
        double* d_x = photons.getXArrayGpu();
        double* d_y = photons.getYArrayGpu();
        double* d_flux = photons.getFluxArrayGpu();
        curandState* d_state;

        cudaMalloc((void**)&d_state, N * sizeof(curandState));

        // 初始化curand状态
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        setup_kernel<<<gridSize, blockSize>>>(d_state, time(0), N);

        // 调用CUDA核函数
        SBounds bounds = {width, height};
        SBBoxImpl_shoot_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_flux, bounds, fluxPerPhoton, N, d_state);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        cudaFree(d_state);
    }

}

#endif