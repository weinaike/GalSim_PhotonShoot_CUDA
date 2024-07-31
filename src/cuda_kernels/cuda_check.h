#ifndef __CUDA_CHECK_H__
#define __CUDA_CHECK_H__

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        std::cerr << "Error " << cudaGetErrorString(_m_cudaStat)            \
                  << " at line " << __LINE__                                \
                  << " in file " << __FILE__ << std::endl;                  \
        exit(1);                                                            \
    }                                                                       \
}
#endif

#endif