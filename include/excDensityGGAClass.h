// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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
#ifndef DFTFE_EXCDENSITYGGACLASS_H
#define DFTFE_EXCDENSITYGGACLASS_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  class NNGGA;
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    excDensityGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr);


    excDensityGGAClass(std::shared_ptr<xc_func_type> funcXPtr,
                       std::shared_ptr<xc_func_type> funcCPtr,
                       std::string                   modelXCInputFile);


    ~excDensityGGAClass();

    void
    computeOutputXCData(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &xDataOut,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &cDataout) const override;

    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const override;

    void
    applyWaveFunctionDependentFuncDer(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) override;
    void
    updateWaveFunctionDependentFuncDer(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) override;
    double
    computeWaveFunctionDependentExcEnergy(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) override;

  private:
    NNGGA *                       d_NNGGAPtr;
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
    std::vector<double>           d_spacingFDStencil;
    unsigned int                  d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYGGACLASS_H
