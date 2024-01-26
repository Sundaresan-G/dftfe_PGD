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

#include "AtomCenteredSphericalFunctionSpline.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionSpline::AtomCenteredSphericalFunctionSpline(
     std::string  filename,
     unsigned int l,
     double trunctationTol,
     bool         consider0thEntry)
  {
    d_lQuantumNumber = l;
    std::vector<std::vector<double>> radialFunctionData(0);
    unsigned int                     fileReadFlag =
      dftUtils::readPsiFile(2, radialFunctionData, filename);
    d_DataPresent = fileReadFlag;
    d_cutOff  = 0.0;
    if (fileReadFlag)
      {

        unsigned int        numRows = radialFunctionData.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        unsigned int maxRowId = 0;
        for (unsigned int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = radialFunctionData[irow][0];
            yData[irow] = radialFunctionData[irow][1];

            if (std::fabs(yData[irow]) > trunctationTol)
              maxRowId = irow;
          }

        if (!consider0thEntry)
          yData[0] = yData[1];

        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type_L = 1;
        alglib::ae_int_t natural_bound_type_R = 1;
        spline1dbuildcubic(x,
                           y,
                           numRows,
                           natural_bound_type_L,
                           0.0,
                           natural_bound_type_R,
                           0.0,
                           d_radialSplineObject);
        d_cutOff = xData[maxRowId];
        d_rMin = xData[0];
      }
  }

  AtomCenteredSphericalFunctionSpline::AtomCenteredSphericalFunctionSpline(
    std::string  filename,
    unsigned int l,
    int          radialPower,
    int          colIndex,
    int          totalColSize,
    int          maxRowIndex)
  {
    d_lQuantumNumber = l;
    std::ifstream       pspFile(filename);
    double              orbValue      = 0.0;
    double              radialValue   = 0.0;
    double              truncationTol = 1E-12;
    std::vector<double> radVec;
    radVec.reserve(5000);
    std::vector<double> orbVec;
    orbVec.reserve(5000);
    std::vector<std::vector<double>> radialFunctionData(0);
    unsigned int                     numMeshSize = 0;
    dftUtils::readFile(totalColSize, radialFunctionData, filename);
    numMeshSize           = radialFunctionData.size();
    unsigned int maxRowId = 0;
    for (int iRow = 0; iRow < numMeshSize; iRow++)
      {
        radialValue = radialFunctionData[iRow][0];
        radVec.push_back(radialValue);
        orbValue = radialFunctionData[iRow][colIndex];
        if (radialPower != 0)
          orbValue *= pow(radialValue, radialPower);
        orbVec.push_back(orbValue);
        if (std::fabs(orbValue) > truncationTol)
          maxRowId = iRow;
      }
    d_rMin = radVec[0];

    // orbVec[0] = orbVec[1];
    alglib::real_1d_array x;
    x.setcontent(numMeshSize, &radVec[0]);
    alglib::real_1d_array y;
    y.setcontent(numMeshSize, &orbVec[0]);
    alglib::ae_int_t natural_bound_typeL = 1;
    alglib::ae_int_t natural_bound_typeR = 1;
    alglib::spline1dbuildcubic(x,
                               y,
                               numMeshSize,
                               natural_bound_typeL,
                               0.0,
                               natural_bound_typeR,
                               0.0,
                               d_radialSplineObject);
    unsigned int maxIndex = std::min(maxRowId + 10, numMeshSize - 1);
    d_cutOff      = maxRowIndex == -1 ? radVec[maxIndex] : radVec[maxRowIndex];
    d_DataPresent = true;
    pspFile.close();
  }
  AtomCenteredSphericalFunctionSpline::AtomCenteredSphericalFunctionSpline(
    std::string filename,
    int         typeBoundaryConditionL,
    bool        valueBoundaryConditionL,
    int         typeBoundaryConditionR,
    bool        valueBoundaryConditionR,
    bool        tailCheck,
    double      maxAllowedTail,
    double      truncationTol,
    double      alpha,
    int         radialPower)
  {
    d_lQuantumNumber = 0;
    std::ifstream       pspFile(filename);
    double              orbValue    = 0.0;
    double              radialValue = 0.0;
    std::vector<double> radVec;
    radVec.reserve(5000);
    std::vector<double> orbVec;
    orbVec.reserve(5000);
    std::vector<std::vector<double>> radialFunctionData(0);
    unsigned int                     numMeshSize = 0;
    dftUtils::readFile(2, radialFunctionData, filename);
    numMeshSize           = radialFunctionData.size();
    unsigned int maxRowId = 0;
    for (int iRow = 0; iRow < numMeshSize; iRow++)
      {
        radialValue = radialFunctionData[iRow][0];
        radVec.push_back(radialValue);
        orbValue = radialFunctionData[iRow][1];
        orbVec.push_back(orbValue);
        if (iRow > 0 && radialValue < maxAllowedTail)
          {
            if (std::abs(orbValue - (-(alpha) / radialValue)) > truncationTol &&
                tailCheck)
              maxRowId = iRow;
            else if (std::fabs(orbValue) > truncationTol)
              maxRowId = iRow;
          }
      }
    d_rMin = radVec[0];

    // orbVec[0] = orbVec[1];
    alglib::real_1d_array x;
    x.setcontent(numMeshSize, &radVec[0]);
    alglib::real_1d_array y;
    y.setcontent(numMeshSize, &orbVec[0]);
    alglib::ae_int_t natural_bound_typeL = typeBoundaryConditionL;
    alglib::ae_int_t natural_bound_typeR = typeBoundaryConditionR;
    double           valueBC_L           = 0.0;
    double           valueBC_R           = 0.0;
    if (valueBoundaryConditionL)
      valueBC_L = y[0];
    else
      valueBC_L = (y[1] - y[0]) / (x[1] - x[0]);
    if (valueBoundaryConditionR)
      valueBC_R = y[numMeshSize];
    else
      valueBC_R = (-y[numMeshSize - 1]) / (x[numMeshSize - 1]);
    alglib::spline1dbuildcubic(x,
                               y,
                               numMeshSize,
                               natural_bound_typeL,
                               valueBC_L,
                               natural_bound_typeR,
                               valueBC_R,
                               d_radialSplineObject);
    unsigned int maxIndex = std::min(maxRowId + 10, numMeshSize - 1);
    d_cutOff              = radVec[maxIndex];
    pspFile.close();
    d_DataPresent = true;
  }
  double
  AtomCenteredSphericalFunctionSpline::getRadialValue(double r) const
  {
    if (r >= d_cutOff)
      return 0.0;

    if (r <= d_rMin)
      r = d_rMin;

    double v = alglib::spline1dcalc(d_radialSplineObject, r);
    return v;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionSpline::getDerivativeValue(double r) const
  {
    std::vector<double> Value(3, 0.0);
    if (r >= d_cutOff)
      return Value;

    if (r <= d_rMin)
      r = d_rMin;
    alglib::spline1ddiff(d_radialSplineObject, r, Value[0], Value[1], Value[2]);

    return Value;
  }

  double
  AtomCenteredSphericalFunctionSpline::getrMinVal() const
  {
    return d_rMin;
  }
} // end of namespace dftfe
