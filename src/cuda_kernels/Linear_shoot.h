
#include "PhotonArray.h"
#include "Random.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
namespace galsim {
    void Linear_shoot_cuda(PhotonArray& photons, UniformDeviate ud);
}
#endif