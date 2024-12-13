
#ifndef __CU_PHOTONARRAY_h__
#define __CU_PHOTONARRAY_h__

#include <iostream>
#include "PhotonArray.h"


#ifdef ENABLE_CUDA

#include "cuda_check.h"
namespace galsim
{
    template <typename T>
    double PhotonArray_addTo_cuda(ImageView<T> &target, double* _x, double* _y, double* _flux, size_t size);


    void PhotonArray_scale(double * d_data, size_t N ,  double scale);
    double PhotonArray_getTotalFlux(double * d_flux, size_t _N);
    void PhotonArray_fwdXY(double mA, double mB, double mC, double mD, double dx, double dy, double* _x_gpu, double* _y_gpu, int _n) ;
    void PhotonArray_convolveShuffle(double* d_x, double* d_y, double* d_flux, 
                    const double* d_rhs_x, const double* d_rhs_y, const double* d_rhs_flux, 
                    int N, long seed) ;
    void PhotonArray_convolve(double* d_x, double* d_y, double* d_flux, 
                    const double* d_rhs_x, const double* d_rhs_y, const double* d_rhs_flux,
                    double scale,  int N) ;
    void PhotonArray_assignAt(double* dest, int N_dest, int offset, const double* src, int N_src);
    int PhotonArray_cpuToGpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N);

    int PhotonArray_gpuToCpu(double * x, double * y, double * flux, double * d_x, double * d_y, double * d_flux, size_t _N);
}
#endif
#endif // VIDEOPIPELINE_PREPROCESS_CUH