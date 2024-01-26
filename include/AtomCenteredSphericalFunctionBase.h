// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H

#include <vector>
#include <boost/math/quadrature/gauss_kronrod.hpp>
namespace dftfe
{
  class AtomCenteredSphericalFunctionBase
  {
  public:
    virtual double
    getRadialValue(double r) const = 0;

    // The following functions need not be re-defined in the
    // derived classes. So it is being defined in this class
    unsigned int
    getQuantumNumberl() const;

    double
    getIntegralValue() const;

    double
    getRadialCutOff() const;

    bool
    isDataPresent() const;

    virtual std::vector<double>
    getDerivativeValue(double r) const = 0;

  protected:
    double       d_cutOff;
    unsigned int d_lQuantumNumber;
    bool         d_DataPresent;


  }; // end of class AtomCenteredSphericalFunctionBase
} // end of namespace dftfe
#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONBASE_H
