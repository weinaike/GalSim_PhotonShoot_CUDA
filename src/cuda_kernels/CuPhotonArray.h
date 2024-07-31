
#ifndef __CU_PHOTONARRAY_h__
#define __CU_PHOTONARRAY_h__

#include <iostream>
#include "PhotonArray.h"


#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include "cuda_check.h"
namespace galsim
{
    template <typename T>
    double PhotonArray_addTo_cuda(ImageView<T> &target, double* _x, double* _y, double* _flux, size_t size);


    void PhotonArray_scale(double * d_data, size_t N ,  double scale);
    double PhotonArray_getTotalFlux(double * d_flux, size_t _N);

    int PhotonArray_cpuToGpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N);

    int PhotonArray_gpuToCpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N);
}
#endif
#endif // VIDEOPIPELINE_PREPROCESS_CUH