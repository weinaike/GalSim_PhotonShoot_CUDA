#ifndef SBBoxImpl_shoot_h
#define SBBoxImpl_shoot_h

#include "galsim/PhotonArray.h"
#ifdef ENABLE_CUDA
namespace galsim {
    void SBBoxImpl_shoot_cuda(PhotonArray& photons, double width, double height, double flux, UniformDeviate ud);
}
#endif
#endif