/* -*- c++ -*-
 * Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

#ifndef GalSim_Math_Angle_H
#define GalSim_Math_Angle_H

#include "Std.h"

namespace galsim {
namespace math {

    // A function to compute sin and cos in one statement, possibly more efficiently than
    // doing the two trig functions separately.
    PUBLIC_API void sincos(double theta, double& sint, double& cost);

}}

#endif
