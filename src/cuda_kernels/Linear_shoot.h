#include <cuda_runtime.h>
#include "PhotonArray.h"
#include "galsim/Random.h"

#ifdef ENABLE_CUDA
namespace galsim {
    void Linear_shoot_cuda(PhotonArray& photons, UniformDeviate ud);
}
#endif