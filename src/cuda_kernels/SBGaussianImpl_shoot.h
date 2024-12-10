#ifndef SBGaussianImpl_shoot_H
#define SBGaussianImpl_shoot_H

#include "PhotonArray.h"
#include "SBGaussianImpl.h"

#ifdef ENABLE_CUDA

#include <curand_kernel.h>
namespace galsim {
    void SBGaussianImpl_shoot_cuda(PhotonArray& photons, UniformDeviate ud, double sigma, double fluxPerPhoton);
}
#endif

#endif // SBGaussianImpl_shoot_H
