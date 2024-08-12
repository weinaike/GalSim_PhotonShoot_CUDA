#ifndef DELTA_SHOOT_H
#define DELTA_SHOOT_H

#include "PhotonArray.h"

namespace galsim {
#ifdef ENABLE_CUDA
    void Delta_shoot_cuda(PhotonArray& photons);
#endif
}

#endif // DELTA_SHOOT_H