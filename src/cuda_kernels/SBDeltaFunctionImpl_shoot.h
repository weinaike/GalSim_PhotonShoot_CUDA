#ifndef SBDELTAFUNCTIONIMPL_SHOOT_H
#define SBDELTAFUNCTIONIMPL_SHOOT_H

#include "PhotonArray.h"
#ifdef ENABLE_CUDA
namespace galsim {
    void SBDeltaFunctionImpl_shoot_cuda(PhotonArray& photons, double fluxPerPhoton);
}
#endif

#endif // SBDELTAFUNCTIONIMPL_SHOOT_H