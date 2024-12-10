// Filename: src/cuda_kernels/SBMoffatImpl_shoot.h

#ifndef SBMoffatImpl_shoot_H
#define SBMoffatImpl_shoot_H
#ifdef ENABLE_CUDA
#include <cmath>
#include "PhotonArray.h"
#include "galsim/Random.h"

namespace galsim {
    void SBMoffatImpl_shoot_cuda(PhotonArray& photons, UniformDeviate ud, double fluxFactor, double beta, double rD, double flux);
}
#endif
#endif