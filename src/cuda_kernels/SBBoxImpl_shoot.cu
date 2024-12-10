

#include "galsim/PhotonArray.h"

#ifdef ENABLE_CUDA
#include "SBBoxImpl_shoot.h"
#include "cuda_check.h"
#include <curand_kernel.h>
struct SBounds {
    double width, height;
};

template <typename T>
__global__ void SBBoxImpl_shoot_kernel(
    T* x, T* y, T* flux, const SBounds bounds, const double fluxPerPhoton, int N, long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        curandState localState;
        curand_init(seed, id, 0, &localState);
        double randX = curand_uniform(&localState) - 0.5;
        double randY = curand_uniform(&localState) - 0.5;
        x[id] = randX * bounds.width;
        y[id] = randY * bounds.height;
        flux[id] = fluxPerPhoton;
    }
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

        unsigned long long seed = ud.get_init_seed(); 
        // 初始化curand状态
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // 调用CUDA核函数
        SBounds bounds = {width, height};
        SBBoxImpl_shoot_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_flux, bounds, fluxPerPhoton, N, seed);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

}

#endif