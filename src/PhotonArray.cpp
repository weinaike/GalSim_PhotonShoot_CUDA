/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */
//
// PhotonArray Class members
//

// #define DEBUGLOGGING

#include <algorithm>
#include <numeric>
#include "PhotonArray.h"
#ifdef ENABLE_CUDA
#include "cuda_kernels/CuPhotonArray.h"
#endif
#include "time.h"

const double EPSILON = 1e-7;
bool isCloseToZero(double value) {
    return fabs(value) < EPSILON;
}


namespace galsim {

    template <typename T>
    struct ArrayDeleter {
        void operator()(T* p) const { delete [] p; }
    };

    PhotonArray::PhotonArray(int N) : 
        _N(N), _dxdz(0), _dydz(0), _wave(0), _is_correlated(false), _vx(N), _vy(N), _vflux(N)
    {
        _x = &_vx[0];
        _y = &_vy[0];
        _flux = &_vflux[0]; 
#ifdef ENABLE_CUDA
        // Allocate memory on the GPU
        if(N > 0 && _x_gpu == nullptr)
        {
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_x_gpu, N * sizeof(double)));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_y_gpu, N * sizeof(double)));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_flux_gpu, N * sizeof(double)));
        }
#endif           
    }


    PhotonArray::PhotonArray(size_t N, double* x, double* y, double* flux,
                double* dxdz, double* dydz, double* wave, bool is_corr) :
        _N(N), _x(x), _y(y), _flux(flux), _dxdz(dxdz), _dydz(dydz), _wave(wave),
        _is_correlated(is_corr) {
#ifdef ENABLE_CUDA
        // Allocate memory on the GPU
        if(N > 0 && _x_gpu == nullptr)
        {
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_x_gpu, N * sizeof(double)));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_y_gpu, N * sizeof(double)));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&_flux_gpu, N * sizeof(double)));
        }
#endif       
    }

    PhotonArray::~PhotonArray()
    {
#ifdef ENABLE_CUDA        
        // Allocate memory on the GPU
        CUDA_CHECK_RETURN(cudaFree(_x_gpu));
        CUDA_CHECK_RETURN(cudaFree(_y_gpu));
        CUDA_CHECK_RETURN(cudaFree(_flux_gpu));   
        _x_gpu = nullptr;
        _y_gpu = nullptr;
        _flux_gpu = nullptr;
#endif     
    }

    template <typename T>
    struct AddImagePhotons
    {
        AddImagePhotons(double* x, double* y, double* f,
                        double maxFlux, BaseDeviate rng) :
            _x(x), _y(y), _f(f), _maxFlux(maxFlux), _ud(rng), _count(0) {}

        void operator()(T flux, int i, int j)
        {
            int N = (std::abs(flux) <= _maxFlux) ? 1 : int(std::ceil(std::abs(flux) / _maxFlux));
            double fluxPer = double(flux) / N;
            for (int k=0; k<N; ++k) {
                double x = i + _ud() - 0.5;
                double y = j + _ud() - 0.5;
                _x[_count] = x;
                _y[_count] = y;
                _f[_count] = fluxPer;
                ++_count;
            }
        }

        int getCount() const { return _count; }

        double* _x;
        double* _y;
        double* _f;
        const double _maxFlux;
        UniformDeviate _ud;
        int _count;
    };

    template <class T>
    int PhotonArray::setFrom(const BaseImage<T>& image, double maxFlux, BaseDeviate rng)
    {
        dbg<<"bounds = "<<image.getBounds()<<std::endl;
        dbg<<"maxflux = "<<maxFlux<<std::endl;
        dbg<<"photon array size = "<<this->size()<<std::endl;
        AddImagePhotons<T> adder(_x, _y, _flux, maxFlux, rng);
        for_each_pixel_ij_ref(image, adder);
        dbg<<"Done: size = "<<adder.getCount()<<std::endl;
        assert(adder.getCount() <= _N);  // Else we've overrun the photon's arrays.
        _N = adder.getCount();
#ifdef ENABLE_CUDA
        PhotonArray_cpuToGpu(_x, _y, _flux, _x_gpu, _y_gpu, _flux_gpu, _N);
#endif
        return _N;
    }

    double PhotonArray::getTotalFlux() const
    {
#ifdef ENABLE_CUDA
        return PhotonArray_getTotalFlux(_flux_gpu, _N);
#else
        double total = 0.;
        return std::accumulate(_flux, _flux+_N, total);
#endif
    }

    void PhotonArray::setTotalFlux(double flux)
    {
        double oldFlux = getTotalFlux();
        if (oldFlux==0.) return; // Do nothing if the flux is zero to start with
        scaleFlux(flux / oldFlux);
    }

    struct Scaler
    {
        Scaler(double _scale): scale(_scale) {}
        double operator()(double x) { return x * scale; }
        double scale;
    };

    void PhotonArray::scaleFlux(double scale)
    {
        if(isCloseToZero(scale - 1.0 )) return;
#ifdef ENABLE_CUDA
        PhotonArray_scale(_flux_gpu, _N, scale);
        // CUDA_CHECK_RETURN(cudaMemcpy(_flux, _flux_gpu, _N * sizeof(double), cudaMemcpyDeviceToHost));
#else
        std::transform(_flux, _flux+_N, _flux, Scaler(scale));
#endif
    }

    void PhotonArray::scaleXY(double scale)
    {
        if(isCloseToZero(scale - 1.0 )) return;
#ifdef ENABLE_CUDA
        PhotonArray_scale(_x_gpu, _N, scale);
        PhotonArray_scale(_y_gpu, _N, scale);
        // CUDA_CHECK_RETURN(cudaMemcpy(_x, _x_gpu, _N * sizeof(double), cudaMemcpyDeviceToHost));
        // CUDA_CHECK_RETURN(cudaMemcpy(_y, _y_gpu, _N * sizeof(double), cudaMemcpyDeviceToHost));
#else
        std::transform(_x, _x+_N, _x, Scaler(scale));
        std::transform(_y, _y+_N, _y, Scaler(scale));
#endif
    }

    void PhotonArray::fwdXY(double mA, double mB, double mC,  double mD, double dx, double dy)
    {
#ifdef ENABLE_CUDA
        PhotonArray_fwdXY(mA, mB, mC, mD, dx, dy, _x_gpu, _y_gpu, _N);
#else

        if (isCloseToZero(mA - 1.0 ) && isCloseToZero(mB) && isCloseToZero(mC) && isCloseToZero(mD - 1.0)) {
            // Special case for no distortion
            for (int i=0; i<_N; i++) {
                _x[i] += dx;
                _y[i] += dy;
            }
        } else if( isCloseToZero(mB) && isCloseToZero(mC) ) {
            // Special case for just a scale and shift
            for (int i=0; i<_N; i++) {
                _x[i] = mA*_x[i] + dx;
                _y[i] = mD*_y[i] + dy;
            }
        } else {            // General case
            for (int i=0; i<_N; i++) {
                double x = _x[i];
                double y = _y[i];
                _x[i] = mA*x + mB*y + dx;
                _y[i] = mC*x + mD*y + dy;
            }
        }
#endif
    }


    void PhotonArray::assignAt(int istart, const PhotonArray& rhs)
    {
        if (istart + rhs.size() > size())
            throw std::runtime_error("Trying to assign past the end of PhotonArray");

        const int N2 = rhs.size();
#ifdef ENABLE_CUDA
        PhotonArray_assignAt(_x_gpu, _N, istart, rhs.getXArrayGpuConst(), rhs.size());
        PhotonArray_assignAt(_y_gpu, _N, istart, rhs.getYArrayGpuConst(), rhs.size());
        PhotonArray_assignAt(_flux_gpu, _N, istart, rhs.getFluxArrayGpuConst(), rhs.size());
#else
        std::copy(rhs._x, rhs._x+N2, _x+istart);
        std::copy(rhs._y, rhs._y+N2, _y+istart);
        std::copy(rhs._flux, rhs._flux+N2, _flux+istart);
#endif        
        // _dxdz, _dydz, _wave, gpu版本未做处理
        // todo: gpu版本未做处理
        if (hasAllocatedAngles() && rhs.hasAllocatedAngles()) {
            std::copy(rhs._dxdz, rhs._dxdz+N2, _dxdz+istart);
            std::copy(rhs._dydz, rhs._dydz+N2, _dydz+istart);
        }
        if (hasAllocatedWavelengths() && rhs.hasAllocatedWavelengths()) {
            std::copy(rhs._wave, rhs._wave+N2, _wave+istart);
        }
    }

    // Helper for multiplying x * y * N
    struct MultXYScale
    {
        MultXYScale(double scale) : _scale(scale) {}
        double operator()(double x, double y) { return x * y * _scale; }
        double _scale;
    };

    void PhotonArray::convolve(const PhotonArray& rhs, BaseDeviate rng)
    {
        // If both arrays have correlated photons, then we need to shuffle the photons
        // as we convolve them.
        if (_is_correlated && rhs._is_correlated) return convolveShuffle(rhs,rng);

        // If neither or only one is correlated, we are ok to just use them in order.
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
#ifdef ENABLE_CUDA
        const double * rhs_x =  rhs.getXArrayGpuConst();
        const double * rhs_y =  rhs.getYArrayGpuConst();
        const double * rhs_flux =  rhs.getFluxArrayGpuConst();
        PhotonArray_convolve(_x_gpu, _y_gpu, _flux_gpu, rhs_x, rhs_y, rhs_flux, (double)_N, _N) ;
#else
        // Add x coordinates:
        std::transform(_x, _x+_N, rhs._x, _x, std::plus<double>());
        // Add y coordinates:
        std::transform(_y, _y+_N, rhs._y, _y, std::plus<double>());
        // Multiply fluxes, with a factor of N needed:
        std::transform(_flux, _flux+_N, rhs._flux, _flux, MultXYScale(_N));
#endif
        // If rhs was correlated, then the output will be correlated.
        // This is ok, but we need to mark it as such.
        if (rhs._is_correlated) _is_correlated = true;
    }

    void PhotonArray::convolveShuffle(const PhotonArray& rhs, BaseDeviate rng)
    {        
        
        if (rhs.size() != size())
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        UniformDeviate ud(rng);
        long seed = ud.get_init_seed(); // 这个要生效， ud要改为引用 &ud
#ifdef ENABLE_CUDA
        
        const double * rhs_x =  rhs.getXArrayGpuConst();
        const double * rhs_y =  rhs.getYArrayGpuConst();
        const double * rhs_flux =  rhs.getFluxArrayGpuConst();
        PhotonArray_convolveShuffle(_x_gpu, _y_gpu, _flux_gpu, rhs_x, rhs_y, rhs_flux, _N, seed) ;

#else
        double xSave=0.;
        double ySave=0.;
        double fluxSave=0.;

        for (int iOut = _N-1; iOut>=0; iOut--) {
            // Randomly select an input photon to use at this output
            // NB: don't need floor, since rhs is positive, so floor is superfluous.
            int iIn = int((iOut+1)*ud());
            if (iIn > iOut) iIn=iOut;  // should not happen, but be safe
            if (iIn < iOut) {
                // Save input information
                xSave = _x[iOut];
                ySave = _y[iOut];
                fluxSave = _flux[iOut];
            }
            _x[iOut] = _x[iIn] + rhs._x[iOut];
            _y[iOut] = _y[iIn] + rhs._y[iOut];
            _flux[iOut] = _flux[iIn] * rhs._flux[iOut] * _N;
            if (iIn < iOut) {
                // Move saved info to new location in array
                _x[iIn] = xSave;
                _y[iIn] = ySave ;
                _flux[iIn] = fluxSave;
            }
        }
#endif
    }

    template <class T>
    double PhotonArray::addTo(ImageView<T> target) const
    {
        dbg<<"Start addTo\n";
        Bounds<int> b = target.getBounds();
        dbg<<"bounds = "<<b<<std::endl;
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");
        #ifdef ENABLE_CUDA
            // cuda version
            
            dbg<<"==========cuda version\n";
            time_t start, end;
            start = clock();
            // PhotonArray_cpuToGpu(_x, _y, _flux, _x_gpu, _y_gpu, _flux_gpu, _N);
            double addedFlux = PhotonArray_addTo_cuda(target, _x_gpu, _y_gpu, _flux_gpu, _N);           
            // printf("addedFlux = %f\n", addedFlux);
            dbg<<"==========addedFlux = "<<addedFlux<<std::endl;
            end = clock();
            double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
            // printf("addto time: %f ms\n", time);
            dbg<<"==========addto time: "<<time<<" ms\n";
            return addedFlux;
        #else    
            dbg<<"==========c++ version\n";
            time_t start, end;
            start = clock();
            double addedFlux = 0.;
            for (size_t i=0; i<size(); i++) {
                int ix = int(floor(_x[i] + 0.5));
                int iy = int(floor(_y[i] + 0.5));
                if (b.includes(ix,iy)) {
                    target(ix,iy) += _flux[i];
                    addedFlux += _flux[i];
                }
            }
            // printf("addedFlux = %f\n", addedFlux);
            dbg<<"==========addedFlux = "<<addedFlux<<std::endl;
            end = clock();
            double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
            // printf("addto time: %f ms\n", time);
            dbg<<"==========addto time: "<<time<<" ms\n";
            
            return addedFlux;
        #endif

    }

    // instantiate template functions for expected image types
    template double PhotonArray::addTo(ImageView<float> image) const;
    template double PhotonArray::addTo(ImageView<double> image) const;
    template int PhotonArray::setFrom(const BaseImage<float>& image, double maxFlux,
                                      BaseDeviate rng);
    template int PhotonArray::setFrom(const BaseImage<double>& image, double maxFlux,
                                      BaseDeviate rng);
}
