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

#ifndef GalSim_OneDimensionalDeviate_H
#define GalSim_OneDimensionalDeviate_H

#include <list>
#include <vector>
#include <functional>
#include "Random.h"
#include "PhotonArray.h"
#include "ProbabilityTree.h"
#include "SBProfile.h"
#include "Std.h"
#include "Interval.h"
#include "cuda_kernels/CuProbabilityTree.h"
namespace galsim {

    /**
     * @brief Class which implements random sampling of an arbitrary one-dimensional distribution,
     * for photon shooting.
     *
     * The point of this class is to take any function that is derived from `FluxDensity` and be
     * able to sample it with photons such that the expectation value of the flux density matches
     * the input function exactly.  This class is for functions which do not have convenient
     * analytic means of inverting their cumulative flux distribution.
     *
     * As explained in SBProfile::shoot(), both positive and negative-flux photons can exist, but we
     * aim that the absolute value of flux be nearly constant so that statistical errors are
     * predictable.  This code does this by first dividing the domain of the function into
     * `Interval` objects, with known integrated (absolute) flux in each.  To shoot a photon, a
     * UniformDeviate is selected and scaled to represent the cumulative flux that should exist
     * within the position of the photon.  The class first uses the binary-search feature built into
     * the Standard Library `set` container to locate the `Interval` that will contain the photon.
     * Then it asks the `Interval` to decide where within the `Interval` to place the photon.  As
     * noted in the `Interval` docstring, this can be done either by rejection sampling, or - if the
     * range of FluxDensity values within an interval is small - by simply adjusting the flux to
     * account for deviations from uniform flux density within the interval.
     *
     * On construction, the class must be provided with some information about the nature of the
     * function being sampled.  The length scale and flux scale of the function should be of order
     * unity.  The elements of the `range` array should be ordered, span the desired domain of the
     * function, and split the domain into intervals such that:
     * - There are no sign changes within an interval
     * - There is at most one extremum within the interval
     * - Any extremum can be localized by sampling the interval at `RANGE_DIVISION_FOR_EXTREMA`
         equidistant points.
     * - The function is smooth enough to be integrated over the interval with standard basic
     *   methods.
     */
    class PUBLIC_API OneDimensionalDeviate
    {
    public:
        /**
         * @brief constructor
         * @param[in] fluxDensity  The FluxDensity being sampled.  No copy is made, original must
         *                         stay in existence.
         * @param[in] range        Ordered argument vector specifying the domain for sampling as
         *                         described in class docstring.
         * @param[in] isRadial     Set true for an axisymmetric function on the plane; false
         *                         for linear domain.
         * @param[in] nominal_flux The expected true integral of the input fluxDensity function.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         operations, if different from the default.
         */
        OneDimensionalDeviate(
            const FluxDensity& fluxDensity, std::vector<double>& range, bool isRadial,
            double nominal_flux, const GSParams& gsparams);

        /// @brief Return total flux in positive regions of FluxDensity
        double getPositiveFlux() const {return _positiveFlux;}

        /// @brief Return absolute value of total flux in regions of negative FluxDensity
        double getNegativeFlux() const {return _negativeFlux;}

        /**
         * @brief Draw photons from the distribution.
         *
         * If `_isRadial=true`, photons will populate the plane.  Otherwise only the x coordinate
         * of photons will be generated, for 1d distribution.
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @param[in] xandy Whether to populate both x and y values (true) or just x (false)
         */
        void shoot(PhotonArray& photons, UniformDeviate ud, bool xandy=false) const;

    private:

        const FluxDensity& _fluxDensity; // Function being sampled
#ifdef ENABLE_CUDA
        CuIntervalProbabilityTree _pt;
#else
        ProbabilityTree<Interval> _pt; // Binary tree of intervals for photon shooting
#endif
        double _positiveFlux; // Stored total positive flux
        double _negativeFlux; // Stored total negative flux
        const bool _isRadial; // True for 2d axisymmetric function, false for 1d function
        GSParams _gsparams;
    };

} // namespace galsim

#endif
