// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri

#include <linearSolverProblemDevice.h>

namespace dftfe
{
  // Constructor.
  linearSolverProblemDevice::linearSolverProblemDevice()
  {
    return;
  }

  // Default: no custom preconditioner
  bool
  linearSolverProblemDevice::usesCustomPreconditioner() const
  {
    return false;
  }

  void
  linearSolverProblemDevice::resetMatVecCount()
  {
    return;
  }

  dftfe::uInt
  linearSolverProblemDevice::getMatVecCount() const
  {
    return 0;
  }

  // Default preconditioner: assert (should not be called unless overridden)
  void
  linearSolverProblemDevice::applyPreconditioner(
    distributedDeviceVec<double> &dst,
    distributedDeviceVec<double> &src)
  {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "DFT-FE Error: applyPreconditioner not implemented for this problem class."));
  }

} // namespace dftfe
