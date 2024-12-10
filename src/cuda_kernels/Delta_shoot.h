#ifndef DELTA_SHOOT_H
#define DELTA_SHOOT_H

#include "PhotonArray.h"
#ifdef ENABLE_CUDA
namespace galsim {
    void Delta_shoot_cuda(PhotonArray& photons);
}
#endif
#endif // DELTA_SHOOT_H