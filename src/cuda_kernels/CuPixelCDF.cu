#ifdef ENABLE_CUDA
#include "CuPixelCDF.h"
#include "cuda_check.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

namespace galsim {

    __global__ void computeCDFWithBlocksOptimized(
        Device_Pixel* d_pixels, 
        Device_Pixel* d_pixels_flux_cdf, 
        double* d_block_sums, 
        int num_pixels, 
        int block_size) 
    {
        extern __shared__ double shared_flux[];  // 动态共享内存分配
        int tid = threadIdx.x;                  // 当前线程在块内的索引
        int idx = blockIdx.x * blockDim.x + tid; // 全局索引

        // 初始化共享内存
        if (idx < num_pixels) {
            shared_flux[tid] = d_pixels[idx].flux;
        } else {
            shared_flux[tid] = 0.0;  // 超出范围的线程填充为0
        }
        __syncthreads();

        // 前缀和计算（块内）
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            double temp = 0.0;
            if (tid >= stride) {
                temp = shared_flux[tid - stride];
            }
            __syncthreads();
            shared_flux[tid] += temp;
            __syncthreads();
        }

        // 写回全局内存
        if (idx < num_pixels) {
            d_pixels_flux_cdf[idx].x = d_pixels[idx].x;
            d_pixels_flux_cdf[idx].y = d_pixels[idx].y;
            d_pixels_flux_cdf[idx].flux = shared_flux[tid];
            d_pixels_flux_cdf[idx].isPositive = d_pixels[idx].isPositive;
        }

        // 每个块的最后一个线程保存块的累积和
        if (tid == blockDim.x - 1) {
            d_block_sums[blockIdx.x] = shared_flux[tid];
        }
        __syncthreads();

        // 块间前缀和计算
        if (blockIdx.x == 0) {
            // 使用块0的线程来计算全局块的前缀和
            extern __shared__ double shared_block_sums[];
            if (tid < gridDim.x) {
                shared_block_sums[tid] = d_block_sums[tid];
            } else {
                shared_block_sums[tid] = 0.0;
            }
            __syncthreads();

            for (int stride = 1; stride < gridDim.x; stride *= 2) {
                double temp = 0.0;
                if (tid >= stride) {
                    temp = shared_block_sums[tid - stride];
                }
                __syncthreads();
                shared_block_sums[tid] += temp;
                __syncthreads();
            }

            // 写回全局块累积和
            if (tid < gridDim.x) {
                d_block_sums[tid] = shared_block_sums[tid];
            }
        }
        __syncthreads();

        // 应用块间增量修正
        if (blockIdx.x > 0 && idx < num_pixels) {
            d_pixels_flux_cdf[idx].flux += d_block_sums[blockIdx.x - 1];
        }
    }

    void computeCDFOptimized(Device_Pixel* d_pixels, Device_Pixel* d_pixels_flux_cdf, int num_pixels) {
        int block_size = 256;  // 每个块的线程数
        int num_blocks = (num_pixels + block_size - 1) / block_size;

        // 分配用于存储每个块累积值的数组
        double* d_block_sums;
        cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
        size_t shared_memory_size = max(block_size, num_blocks) * sizeof(double);
        // 1. 计算每个块内的前缀和
        computeCDFWithBlocksOptimized<<<num_blocks, block_size, shared_memory_size>>>(
            d_pixels, d_pixels_flux_cdf, d_block_sums, num_pixels, block_size);

        // 2. Check for errors
        cudaDeviceSynchronize();
        cudaGetLastError();

        // 3. 释放辅助数组
        cudaFree(d_block_sums);
    }

    __global__ void computeCDF(
        Device_Pixel* d_pixels, 
        Device_Pixel* d_pixels_flux_cdf, 
        double* block_sums, 
        int num_pixels) 
    {
        extern __shared__ double shared_flux[];  // 动态分配共享内存

        int tid = threadIdx.x;                  // 当前线程在块内的索引
        int idx = blockIdx.x * blockDim.x + tid; // 全局索引

        // 初始化共享内存
        if (idx < num_pixels) {
            shared_flux[tid] = d_pixels[idx].flux;
        } else {
            shared_flux[tid] = 0.0;  // 超出范围的线程填充为0
        }
        __syncthreads();

        // 前缀和计算（扫描算法）
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            double temp = 0.0;
            if (tid >= stride) {
                temp = shared_flux[tid - stride];
            }
            __syncthreads();
            shared_flux[tid] += temp;
            __syncthreads();
        }

        // 写回全局内存
        if (idx < num_pixels) {
            d_pixels_flux_cdf[idx].x = d_pixels[idx].x;
            d_pixels_flux_cdf[idx].y = d_pixels[idx].y;
            d_pixels_flux_cdf[idx].flux = shared_flux[tid];
            d_pixels_flux_cdf[idx].isPositive = d_pixels[idx].isPositive;
        }

        // 每个块的最后一个线程保存块的累积和
        if (tid == blockDim.x - 1) {
            block_sums[blockIdx.x] = shared_flux[tid];
        }
    }
    __global__ void computeBlockSums(double* block_sums, int num_blocks) {
        extern __shared__ double shared_sums[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;

        // 拷贝块的累积值到共享内存
        if (idx < num_blocks) {
            shared_sums[tid] = block_sums[idx];
        } else {
            shared_sums[tid] = 0.0;
        }
        __syncthreads();

        // 前缀和计算
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            double temp = 0.0;
            if (tid >= stride) {
                temp = shared_sums[tid - stride];
            }
            __syncthreads();
            shared_sums[tid] += temp;
            __syncthreads();
        }

        // 写回全局内存
        if (idx < num_blocks) {
            block_sums[idx] = shared_sums[tid];
        }
    }

    __global__ void applyBlockIncrements(
        Device_Pixel* d_pixels_flux_cdf, 
        double* block_sums, 
        int num_pixels, 
        int block_size) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_pixels && blockIdx.x > 0) {
            d_pixels_flux_cdf[idx].flux += block_sums[blockIdx.x - 1];
        }
    }


    void computeCDFWithBlocks(Device_Pixel* d_pixels, Device_Pixel* d_pixels_flux_cdf, int num_pixels) {
        int block_size = 256;  // 每个块的线程数
        int num_blocks = (num_pixels + block_size - 1) / block_size;

        // 分配用于存储每个块累积值的数组
        double* d_block_sums;
        cudaMalloc(&d_block_sums, num_blocks * sizeof(double));

        // 1. 计算每个块内的前缀和
        computeCDF<<<num_blocks, block_size, block_size * sizeof(double)>>>(
            d_pixels, d_pixels_flux_cdf, d_block_sums, num_pixels);

        // 2. 计算块间累积值
        computeBlockSums<<<1, num_blocks, num_blocks * sizeof(double)>>>(d_block_sums, num_blocks);

        // 3. 应用块间增量
        applyBlockIncrements<<<num_blocks, block_size>>>(
            d_pixels_flux_cdf, d_block_sums, num_pixels, block_size);

        // 释放辅助数组
        cudaFree(d_block_sums);
    }

    __global__ void OnceComputeCDF(Device_Pixel* d_pixels, Device_Pixel* d_pixels_flux_cdf, int num_pixels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_pixels) {
            double cumulative_flux = 0.0;
            for (int i = 0; i <= idx; ++i) {
                cumulative_flux += d_pixels[i].flux;
            }
            d_pixels_flux_cdf[idx].x = d_pixels[idx].x;
            d_pixels_flux_cdf[idx].y = d_pixels[idx].y;
            d_pixels_flux_cdf[idx].flux = cumulative_flux;
            d_pixels_flux_cdf[idx].isPositive = d_pixels[idx].isPositive;
        }
    }

    __global__ void binary_search_kernel(Device_Pixel* d_pixels_flux_cdf, int num, unsigned long seed, double* x, double* y, double* flux, int N, double fluxPerPhoton) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            // Generate random number
            curandState state;
            curand_init(seed, idx, 0, &state);
            double random_number = curand_uniform(&state);

            // Perform binary search on CDF
            double target = random_number * d_pixels_flux_cdf[num - 1].flux; // Ensure target is correctly calculated
            int left = 0;
            int right = num - 1;
            while (left < right) {
                int mid = (left + right) / 2;
                if (d_pixels_flux_cdf[mid].flux < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            x[idx] = d_pixels_flux_cdf[left].x;
            y[idx] = d_pixels_flux_cdf[left].y;
            flux[idx] = d_pixels_flux_cdf[left].isPositive ? fluxPerPhoton : -fluxPerPhoton;
        }
    }

    void computeCDFWithoutBlocks(Device_Pixel* d_pixels, Device_Pixel* d_pixels_flux_cdf, int num_pixels) 
    {
        // Define block and grid sizes
        int blockSize = 256;
        int numBlocks = (num_pixels + blockSize - 1) / blockSize;

        // Launch the kernel
        OnceComputeCDF<<<numBlocks, blockSize>>>(d_pixels, d_pixels_flux_cdf, num_pixels);
    }

    CuPixelCDF::~CuPixelCDF() {
        if (d_pixels != nullptr) {
            cudaFree(d_pixels);
            d_pixels = nullptr;
        }
        if (d_pixels_flux_cdf != nullptr) {
            cudaFree(d_pixels_flux_cdf);
            d_pixels_flux_cdf = nullptr;
        }

    }

    // 从 pixels 中构建 CDF
    void CuPixelCDF::buildCDF(double threshold)
    {

        num_pixels = pixels.size();
        // printf("num_pixels = %d\n", num_pixels);

        CUDA_CHECK_RETURN(cudaMalloc(&d_pixels, num_pixels * sizeof(Device_Pixel)));
        CUDA_CHECK_RETURN(cudaMalloc(&d_pixels_flux_cdf, num_pixels * sizeof(Device_Pixel)));

        CUDA_CHECK_RETURN(cudaMemcpy(d_pixels, pixels.data(), num_pixels * sizeof(Device_Pixel), cudaMemcpyHostToDevice));
        // 计算前缀和
        // computeCDFWithoutBlocks(d_pixels, d_pixels_flux_cdf, num_pixels);
        computeCDFWithBlocks(d_pixels, d_pixels_flux_cdf, num_pixels);
        // computeCDFOptimized(d_pixels, d_pixels_flux_cdf, num_pixels);

        // 4. Check for errors
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());

    }
    

    // 从 CDF 中采样 N 个像素
    // seed 为随机数种子
    // x, y, flux 为输出参数，分别保存采样到的像素的坐标与flux， x, y, flux 都是设备内容
    // x,y,flux 的长度为 N
    void CuPixelCDF::find(long seed, double * x, double* y, double* flux, int N, double fluxPerPhoton) const
    {
        // Launch kernel to perform binary search on CDF and find corresponding pixels
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        binary_search_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pixels_flux_cdf, num_pixels, seed, x, y, flux, N, fluxPerPhoton);

        // Check for errors
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

}
#endif