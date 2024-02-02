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
// @author Sambit Das
//

// skip1 and skip2 are flags used by chebyshevFilter function to perform overlap
// of computation and communication. When either skip1 or skip2 flags are set to
// true all communication calls are skipped as they are directly called in
// chebyshevFilter Only one of the skip flags is set to true in a call. When
// skip1 is set to true extraction and C^{T}*X computation are skipped and
// computations directly start from V*C^{T}*X. When skip2 is set to true only
// extraction and C^{T}*X computations are performed.
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeNonLocalHamiltonianTimesX(
    const dataTypes::number *                src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
    const unsigned int                       numberWaveFunctions,
    dataTypes::number *                      dst,
    const bool                               skip1,
    const bool                               skip2)
{
  if (d_totalNonlocalElemsPseudo > 0 && !skip1)
    {
      // dftfe::utils::deviceSynchronize();
      // std::cout<<"Starting CTX: "<<std::endl;
      d_ONCVnonLocalOperator->applyCconjtrans_onX(
        d_cellWaveFunctionMatrix,
        std::pair<unsigned int, unsigned int>(0, d_totalNonlocalElemsPseudo));
    }

  // this routine was interfering with overlapping communication and compute. So
  // called separately inside chebyshevFilter. So skip this if either skip1 or
  // skip2 are set to true
  if (!skip1 && !skip2)
    {
      projectorKetTimesVector.setValue(0);
    }

  d_ONCVnonLocalOperator->applyAllReduceonCTX(projectorKetTimesVector,
                                              skip1,
                                              skip2);

  // Operations related to skip2 (extraction and C^{T}*X) are over. So return
  // control back to chebyshevFilter
  if (skip2)
    return;



  if (d_totalNonlocalElemsPseudo > 0)
    {
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        couplingMatrix = d_oncvClassPtr->getCouplingMatrix();
      d_ONCVnonLocalOperator->applyV_onCconjtransX(CouplingStructure::diagonal,
                                                   couplingMatrix,
                                                   projectorKetTimesVector,
                                                   true);
      d_ONCVnonLocalOperator->applyC_VCconjtransX(
        d_cellHamMatrixTimesWaveMatrix,
        std::pair<unsigned int, unsigned int>(0, d_totalNonlocalElemsPseudo));
    }

  if (std::is_same<dataTypes::number, std::complex<double>>::value)
    {
      utils::deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (d_tempRealVec.size()),
        dst,
        d_tempRealVec.begin(),
        d_tempImagVec.begin());

      dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
        numberWaveFunctions,
        d_numLocallyOwnedCells * d_numberNodesPerElement,
        d_cellHamMatrixTimesWaveMatrix.begin(),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());


      utils::deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        (d_tempRealVec.size()),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        dst);
    }
  else
    dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
      numberWaveFunctions,
      d_numLocallyOwnedCells * d_numberNodesPerElement,
      d_cellHamMatrixTimesWaveMatrix.begin(),
      dst,
      d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    const dataTypes::number *                src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
    const unsigned int                       numberWaveFunctions)
{
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();
  const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                          scalarCoeffBeta  = dataTypes::number(0.0);

  //
  // compute C^{\dagger}*X
  //


  if (d_totalNonlocalElemsPseudo > 0)
    {
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
        numberWaveFunctions,
        totalLocallyOwnedCells * d_numberNodesPerElement,
        src,
        d_cellWaveFunctionMatrix.begin(),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
      d_ONCVnonLocalOperator->applyCconjtrans_onX(
        d_cellWaveFunctionMatrix,
        std::pair<unsigned int, unsigned int>(0, d_totalNonlocalElemsPseudo));
    }

  projectorKetTimesVector.setValue(0);
  d_ONCVnonLocalOperator->applyAllReduceonCTX(projectorKetTimesVector);
  // d_oncvClassPtr->applynonLocalHamiltonianMatrix(projectorKetTimesVector,
  //                                                false);
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
    couplingMatrix = d_oncvClassPtr->getCouplingMatrix();
  d_ONCVnonLocalOperator->applyV_onCconjtransX(CouplingStructure::diagonal,
                                               couplingMatrix,
                                               projectorKetTimesVector,
                                               false);
}
