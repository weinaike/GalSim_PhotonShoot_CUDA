

#ifndef GALSIM_INTERVAL_H
#define GALSIM_INTERVAL_H

#include <list>
#include <vector>
#include <functional>
#include "Random.h"
#include "Std.h"
#include "SBProfile.h"

namespace galsim {


    struct Device_Interval {
        double _xLower;
        double _xUpper;
        double _xRange;        
        double _flux;        
        double _a;
        double _b;
        double _c;
        double _d;
        double shoot_accuracy;
        bool _isRadial;
        bool _fluxIsReady;
    };

    /**
     * @brief An interface class for functions giving differential flux vs x or r.
     *
     * Functions derived from this interface can be integrated by `Int.h` and
     * can be sampled by `OneDimensionalDeviate`.
     */
    class PUBLIC_API FluxDensity
    {
    public:
        /// @brief virtual destructor for base class
        virtual ~FluxDensity() {}
        /**
         * @brief Interface requires implementation of operator()
         * @param[in] x The linear position or radius
         * @returns Flux or probability density at location of argument.
         */
        virtual double operator()(double x) const=0;
    };

    /**
     * @brief Class used to represent a linear interval or an annulus of probability function
     *
     * An `Interval` is a contiguous domain over which a `FluxDensity` function is well-behaved,
     * having no sign changes or extrema, which will make it easier to sample the FluxDensity
     * function over its domain using either rejection sampling or by weighting uniformly
     * distributed photons.
     *
     * This class could be made a subclass of `OneDimensionalDeviate` as it should only be used by
     * methods of that class.
     *
     * The `Interval` represents flux (or unnormalized probability) density in a continguous
     * interval on on the line, or, for `_isRadial=true`, represents axisymmetric density in an
     * annulus on the plane.
     *
     * The object keeps track of the integrated flux (or unnormalized probability) in its
     * interval/annulus, and the cumulative flux of all intervals up to and including this one.
     *
     * The `drawWithin()` method will select one photon (and flux) drawn from within this interval
     * or annulus, such that the expected flux distribution matches the FluxDensity function.
     *
     * See the `OneDimensionalDeviate` docstrings for more information.
     */


    class PUBLIC_API Interval
    {
    public:
        /**
         * @brief Constructor
         *
         * Note that no copy of the function is saved.  The function whose reference is passed must
         * remain in existence through useful lifetime of the `Interval`
         * @param[in] fluxDensity  The function giving flux (= unnormalized probability) density.
         * @param[in] xLower       Lower bound in x (or radius) of this interval.
         * @param[in] xUpper       Upper bound in x (or radius) of this interval.
         * @param[in] isRadial     Set true if this is an annulus on a plane, false for linear
         *                         interval.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         operations.
         */
        Interval(const FluxDensity& fluxDensity,
                 double xLower,
                 double xUpper,
                 bool isRadial,
                 const GSParams& gsparams) :
            _fluxDensityPtr(&fluxDensity),
            _xLower(xLower),
            _xUpper(xUpper),
            _xRange(_xUpper - _xLower),
            _isRadial(isRadial),
            _gsparams(gsparams),
            _fluxIsReady(false)
        {}

        Interval(const Interval& rhs) :
            _fluxDensityPtr(rhs._fluxDensityPtr),
            _xLower(rhs._xLower),
            _xUpper(rhs._xUpper),
            _xRange(rhs._xRange),
            _isRadial(rhs._isRadial),
            _gsparams(rhs._gsparams),
            _fluxIsReady(false),
            _a(rhs._a), _b(rhs._b), _c(rhs._c), _d(rhs._d)
        {}

        Interval& operator=(const Interval& rhs)
        {
            // Everything else is constant, so no need to copy.
            _xLower = rhs._xLower;
            _xUpper = rhs._xUpper;
            _xRange = rhs._xRange;
            _isRadial = rhs._isRadial;
            _fluxIsReady = false;
            _a = rhs._a;
            _b = rhs._b;
            _c = rhs._c;
            _d = rhs._d;
            return *this;
        }

        /**
         * @brief Draw one photon position and flux from within this interval
         * @param[in] unitRandom An initial uniform deviate to select photon
         * @param[out] x (or radial) coordinate of the selected photon.
         * @param[out] flux flux of the selected photon = +-1
         */
        void drawWithin(double unitRandom, double& x, double& flux) const;

        /**
         * @brief Get integrated flux over this interval or annulus.
         *
         * Performs integral if not already cached.
         *
         * @returns Integrated flux in interval.
         */
        double getFlux() const { checkFlux(); return _flux; }

        /**
         * @brief Report interval bounds
         * @param[out] xLower Interval lower bound
         * @param[out] xUpper Interval upper bound
         */
        void getRange(double& xLower, double& xUpper) const
        {
            xLower = _xLower;
            xUpper = _xUpper;
        }

        /**
         * @brief Return a list of intervals that divide this one into acceptably small ones.
         *
         * This routine works by recursive bisection.  Intervals that are returned have all had
         * their fluxes integrated.  Intervals are split until the error from a linear
         * approximation to f(x) is less than toler;
         * @param[in] toler Tolerance on the flux error below which a sub-interval is not split.
         * @returns List of contiguous Intervals whose union is this one.
         */
        std::list<shared_ptr<Interval> > split(double toler);
#ifdef ENABLE_CUDA
        void get_interval_data(Device_Interval & data);
#endif
    private:

        const FluxDensity* _fluxDensityPtr;  // Pointer to the parent FluxDensity function.
        double _xLower; // Interval lower bound
        double _xUpper; // Interval upper bound
        double _xRange; // _xUpper - _xLower  (used a lot)
        bool _isRadial; // True if domain is an annulus, otherwise domain is a linear interval.
        const GSParams& _gsparams;

        mutable bool _fluxIsReady; // True if flux has been integrated
        void checkFlux() const; // Calculate flux if it has not already been done.
        mutable double _flux; // Integrated flux in this interval (can be negative)

        // Finds the x or radius coord that would enclose fraction of this interval's flux
        // if flux were constant.
        double interpolateFlux(double fraction) const;

        double _a, _b, _c, _d;  // Coefficients used for solving for dx in the interval.

    };

}
#endif