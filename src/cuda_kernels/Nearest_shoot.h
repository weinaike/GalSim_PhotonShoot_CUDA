#include "PhotonArray.h"
#ifdef ENABLE_CUDA
namespace galsim {
    void Nearest_shoot_cuda(PhotonArray& photons, int N, unsigned long seed);
}
#endif
