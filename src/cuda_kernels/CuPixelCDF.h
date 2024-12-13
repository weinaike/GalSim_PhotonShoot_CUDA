#ifndef __CuPixelCDF_h__
#define __CuPixelCDF_h__


#include "ProbabilityTree.h"
#include "SBInterpolatedImage.h"

#ifdef ENABLE_CUDA


namespace galsim {

    struct Device_Pixel
    {
        double x;
        double y;
        double flux;
        bool isPositive;
    };

    class CuPixelCDF 
    {
    public:
        CuPixelCDF() {d_pixels = nullptr; d_pixels_flux_cdf = nullptr;}

        ~CuPixelCDF();
          
        void buildCDF(double threshold=0.);

        void find(long seed, double * x, double* y, double* flux, int N, double fluxPerPhoton) const;
        void push_back(const shared_ptr<Pixel>& p) 
        {
            Device_Pixel dp;
            dp.x = p->x;
            dp.y = p->y;
            dp.flux = p->getFlux();
            dp.isPositive = p->isPositive;
            pixels.push_back(dp);
        }
        void clear() {pixels.clear();}
        bool empty() const {return pixels.empty();}
    private:       
        std::vector<Device_Pixel> pixels;
        Device_Pixel * d_pixels;
        Device_Pixel * d_pixels_flux_cdf;
        int num_pixels; // Add this line to store the number of pixels in d_pixels_flux_cdf

    };

}
#endif
#endif