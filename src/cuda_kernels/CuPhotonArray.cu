

#include "CuPhotonArray.h"
#include "time.h"


#ifdef ENABLE_CUDA

namespace galsim
{   
    
    struct cuBounds {
        int xmin, xmax, ymin, ymax;
        int step, stride;
    };


    template <typename T>
    __global__ void photonArray_addTo_Kernel_1(double* added_flux, double* x, double* y, double* flux, size_t size, T* target, cuBounds* cub)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            int ix = int(floor(x[i] + 0.5));
            int iy = int(floor(y[i] + 0.5));
            if (ix >= cub->xmin && ix <= cub->xmax && iy >= cub->ymin && iy <= cub->ymax) 
            {
                long int idx = (ix - cub->xmin) * cub->step +  (iy - cub->ymin) * cub->stride;
                atomicAdd(&(target[idx]), flux[i]);
                atomicAdd(added_flux,flux[i]); // 取消这一步 2.025000 ms ==》 0.255000 ms， 降低一个量级
            }
        }
    }


    template <typename T>
    __global__ void photonArray_addTo_Kernel(double* added_flux, double* x, double* y, double* flux, size_t size, T* target, cuBounds* cub)
    {
        extern __shared__ double shared_flux[];
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t tid = threadIdx.x;

        // Initialize shared memory
        shared_flux[tid] = 0.0;

        __syncthreads();

        if (i < size) {
            int ix = int(floor(x[i] + 0.5));
            int iy = int(floor(y[i] + 0.5));
            if (ix >= cub->xmin && ix <= cub->xmax && iy >= cub->ymin && iy <= cub->ymax) {
                long int idx = (ix - cub->xmin) * cub->step +  (iy - cub->ymin) * cub->stride;
                atomicAdd(&(target[idx]), flux[i]);
                shared_flux[tid] = flux[i];
            }
        }

        __syncthreads();

        // Reduce shared memory to a single value
        if (tid == 0) {
            double block_sum = 0.0;
            for (int j = 0; j < blockDim.x; ++j) {
                block_sum += shared_flux[j];
            }
            atomicAdd(added_flux, block_sum);
        }
    }

    __global__ void accumulateKernel(const double* flux, double* result, int N) {
        extern __shared__ double sharedData[];

        int tid = threadIdx.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // Load elements into shared memory
        if (index < N) {
            sharedData[tid] = flux[index];
        } else {
            sharedData[tid] = 0.0;
        }
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0) {
            atomicAdd(result, sharedData[0]);
        }
    }

    __global__ void scaleKernel(double* data, int N, double scale)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N)
        {
            data[idx] *= scale;
        }
    }


    void PhotonArray_scale(double * d_data, size_t _N, double scale)
        {
   
        time_t start, end;
        start = clock();
        

        int blockSize = 1024;
        int numBlocks = (_N + blockSize - 1) / blockSize;
        scaleKernel<<<numBlocks, blockSize>>>(d_data, _N, scale);
        CUDA_CHECK_RETURN(cudaGetLastError()); 


        end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("PhotonArray_scale time: %f ms,    %d\n", time, _N);

    }




    double PhotonArray_getTotalFlux(double * d_flux, size_t _N)
    {
        int blockSize = 256;
        int numBlocks = (_N + blockSize - 1) / blockSize;
        double* d_result;
        double result = 0.0;
        CUDA_CHECK_RETURN(cudaMalloc((void**) &d_result, sizeof(double)));
        CUDA_CHECK_RETURN(cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice));

        accumulateKernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_flux, d_result, _N);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        CUDA_CHECK_RETURN(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaFree(d_result));
        
        return result;
    }

    int PhotonArray_gpuToCpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N)
    {
        CUDA_CHECK_RETURN(cudaMemcpy(x, d_x, _N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(y, d_y, _N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(flux, d_flux, _N * sizeof(double), cudaMemcpyDeviceToHost));
        return 0;
    }

    int PhotonArray_cpuToGpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N)
    {
        CUDA_CHECK_RETURN(cudaMemcpy(d_x, x, _N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(d_y, y,  _N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(d_flux, flux, _N * sizeof(double), cudaMemcpyHostToDevice));
        return 0;
    }

    template <typename T>
    double PhotonArray_addTo_cuda(ImageView<T> &target, double* d_x, double* d_y, double* d_flux, size_t size)
    {
        clock_t start, end;
        start = clock();
        // this function should call CUDA kernel
        Bounds<int> b = target.getBounds();
        cuBounds cub = {0};
        cub.xmin = b.getXMin();
        cub.xmax = b.getXMax();
        cub.ymin = b.getYMin();
        cub.ymax = b.getYMax();
        cub.step = target.getStep();
        cub.stride = target.getStride();

        T * d_target = target.getGpuData();
        
        // (xmax - xmin + 1) * (ymax - ymin + 1);
        // initialize added flux
        double addedFlux = 0.;

        // allocate GPU memory
        double* d_added_flux;
        cuBounds* d_cub;

        CUDA_CHECK_RETURN(cudaMalloc((void**) &d_added_flux, sizeof(double)));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &d_cub, sizeof(cuBounds)));
        // copy the cpu memory to GPU       
        CUDA_CHECK_RETURN(cudaMemcpy(d_added_flux, &addedFlux, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(d_cub, &cub, sizeof(cuBounds), cudaMemcpyHostToDevice));



        #if 1
            int blockSize = 256; // Example block size
            int numBlocks = (size + blockSize - 1) / blockSize;
            photonArray_addTo_Kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_added_flux, d_x, d_y, d_flux, size, d_target, d_cub);
            cudaDeviceSynchronize();   
        #else
        
            dim3 blocks((size + 256 - 1) / 256);
            dim3 threads(256);    
            photonArray_addTo_Kernel_1<<<blocks, threads>>>(d_added_flux, d_x, d_y, d_flux, size, d_target, d_cub);  
            cudaDeviceSynchronize();   
        #endif

        // target.copyGpuDataToCpu();

        // copy memory back to CPU
        CUDA_CHECK_RETURN(cudaMemcpy(&addedFlux, d_added_flux, sizeof(double), cudaMemcpyDeviceToHost));

        // free the allocated GPU memory
        CUDA_CHECK_RETURN(cudaFree(d_added_flux));        
        CUDA_CHECK_RETURN(cudaFree(d_cub));
        end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("PhotonArray_addTo_cuda time: %f ms,    %d\n", time, size);
        return addedFlux;
    }

    template __global__ void photonArray_addTo_Kernel_1(double* added_flux, double* x, double* y, double* flux, size_t size, float* target,   cuBounds* cub);
    template __global__ void photonArray_addTo_Kernel_1(double* added_flux, double* x, double* y, double* flux, size_t size, double* target,   cuBounds* cub);
    template __global__ void photonArray_addTo_Kernel(double* added_flux, double* x, double* y, double* flux, size_t size, float* target,   cuBounds* cub);
    template __global__ void photonArray_addTo_Kernel(double* added_flux, double* x, double* y, double* flux, size_t size, double* target,   cuBounds* cub);
    template double PhotonArray_addTo_cuda(ImageView<float> & target, double* _x, double* _y, double* _flux, size_t size);
    template double PhotonArray_addTo_cuda(ImageView<double> & target, double* _x, double* _y, double* _flux, size_t size);
}
#endif