#ifndef __SBTopHatImpl_shoot_h
#define __SBTopHatImpl_shoot_h
#include "Random.h"
#include "PhotonArray.h"

#ifdef ENABLE_CUDA

namespace galsim {
    void SBTopHatImpl_shoot_cuda(PhotonArray& photons, double r0, double flux, UniformDeviate ud);
}

#endif // ENABLE_CUDA
#endif // __SBTopHatImpl_shoot_h