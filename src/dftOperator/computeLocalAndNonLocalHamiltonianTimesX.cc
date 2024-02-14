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
// @author Phani Motamarri, Department of Computational and Data Sciences, IISc Bangalore
//


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */


template <unsigned int              FEOrder,
          unsigned int              FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro, memorySpace>::
  computeHamiltonianTimesXInternal(
    const distributedCPUMultiVec<dataTypes::number> &src,
    distributedCPUMultiVec<dataTypes::number> &      dst,
    const double                                     scalarHX,
    const double                                     scalarY,
    const double                                     scalarX,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
{
  if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
    {
      const unsigned int kpointSpinIndex =
        (1 + dftPtr->d_dftParamsPtr->spinPolarized) * d_kPointIndex +
        d_spinIndex;
      const dataTypes::number zero(0.0), one(1.0);
      //
      // element level matrix-vector multiplications
      //
      if (dftPtr->d_dftParamsPtr->isPseudopotential &&
          !dftPtr->d_dftParamsPtr->pawPseudoPotential)
        {
          d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
          d_SphericalFunctionKetTimesVectorParFlattened.setValue(0.0);
        }

      const unsigned int inc = 1;

      const unsigned int nRelaventDofs       = src.localSize();
      const unsigned int nLocalDofs          = src.locallyOwnedSize();
      const unsigned int numberWaveFunctions = src.numVectors();

      const unsigned int totalLocallyOwnedCells =
        dftPtr->matrix_free_data.n_physical_cells();
      std::vector<bool> dofEncountered(nRelaventDofs, false);
      if (dftPtr->d_dftParamsPtr->isPseudopotential &&
          !dftPtr->d_dftParamsPtr->pawPseudoPotential &&
          d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
            0 &&
          !onlyHPrimePartForFirstOrderDensityMatResponse)
        {
          std::vector<unsigned int> cellsOfInterest =
            d_ONCVnonLocalOperator->getNonlocalElementToCellIdVector();
          for (unsigned int iElem = 0; iElem < cellsOfInterest.size(); ++iElem)
            {
              unsigned int iCell = cellsOfInterest[iElem];
              for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index localNodeId =
                    d_flattenedArrayCellLocalProcIndexIdMap[iCell][iNode];
                  const double scalarCoeffAlpha =
                    d_invSqrtElementalMassVector[iCell *
                                                   d_numberNodesPerElement +
                                                 iNode];
                  std::transform(src.begin() + localNodeId,
                                 src.begin() + localNodeId +
                                   numberWaveFunctions,
                                 d_cellWaveFunctionMatrix.begin() +
                                   numberWaveFunctions * iNode,
                                 [&scalarCoeffAlpha](auto &a) {
                                   return scalarCoeffAlpha * a;
                                 });
                } // scaling

              d_ONCVnonLocalOperator->applyCconjtransOnX(
                d_cellWaveFunctionMatrix,
                std::pair<unsigned int, unsigned int>(iElem, iElem + 1));


            } // Cell Loop
          d_ONCVnonLocalOperator->applyAllReduceOnCconjtransX(
            d_SphericalFunctionKetTimesVectorParFlattened);
          d_ONCVnonLocalOperator->applyVOnCconjtransX(
            CouplingStructure::diagonal,
            d_oncvClassPtr->getCouplingMatrix(),
            d_SphericalFunctionKetTimesVectorParFlattened);



        } // nonlocal

      for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
        {
          for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
            {
              dealii::types::global_dof_index localNodeId =
                d_flattenedArrayCellLocalProcIndexIdMap[iCell][iNode];
              const double scalarCoeffAlpha =
                d_invSqrtElementalMassVector[iCell * d_numberNodesPerElement +
                                             iNode];
              std::transform(src.begin() + localNodeId,
                             src.begin() + localNodeId + numberWaveFunctions,
                             d_cellWaveFunctionMatrix.begin() +
                               numberWaveFunctions * iNode,
                             [&scalarCoeffAlpha](auto &a) {
                               return scalarCoeffAlpha * a;
                             });
            }
          d_BLASWrapperPtrHost->xgemm(
            'N',
            std::is_same<dataTypes::number, std::complex<double>>::value ? 'T' :
                                                                           'N',
            numberWaveFunctions,
            d_numberNodesPerElement,
            d_numberNodesPerElement,
            &one,
            &d_cellWaveFunctionMatrix[0],
            numberWaveFunctions,
            &d_cellHamiltonianMatrix[kpointSpinIndex][iCell][0],
            d_numberNodesPerElement,
            &zero,
            &d_cellHamMatrixTimesWaveMatrix[0],
            numberWaveFunctions);

          if (dftPtr->d_dftParamsPtr->isPseudopotential &&
              !dftPtr->d_dftParamsPtr->pawPseudoPotential &&
              !onlyHPrimePartForFirstOrderDensityMatResponse)
            {
              d_ONCVnonLocalOperator->applyCOnVCconjtransX(
                d_cellHamMatrixTimesWaveMatrix,
                std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
            }


          for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
            {
              dealii::types::global_dof_index localNodeId =
                d_flattenedArrayCellLocalProcIndexIdMap[iCell][iNode];
              const double scalarCoeffAlpha =
                scalarHX *
                d_invSqrtElementalMassVector[iCell * d_numberNodesPerElement +
                                             iNode];
              if (dofEncountered[localNodeId / numberWaveFunctions])
                std::transform(dst.begin() + localNodeId,
                               dst.begin() + localNodeId + numberWaveFunctions,
                               d_cellHamMatrixTimesWaveMatrix.begin() +
                                 numberWaveFunctions * iNode,
                               dst.begin() + localNodeId,
                               [&scalarCoeffAlpha](auto &a, auto &b) {
                                 return a + scalarCoeffAlpha * b;
                               });
              else
                {
                  dofEncountered[localNodeId / numberWaveFunctions] = true;
                  if (d_isConstrained[localNodeId / numberWaveFunctions] ||
                      localNodeId / numberWaveFunctions >= nLocalDofs)
                    std::transform(d_cellHamMatrixTimesWaveMatrix.begin() +
                                     numberWaveFunctions * iNode,
                                   d_cellHamMatrixTimesWaveMatrix.begin() +
                                     numberWaveFunctions * iNode +
                                     numberWaveFunctions,
                                   dst.begin() + localNodeId,
                                   [&scalarCoeffAlpha](auto &a) {
                                     return scalarCoeffAlpha * a;
                                   });
                  else
                    for (auto i = 0; i < numberWaveFunctions; ++i)
                      dst.data()[localNodeId + i] =
                        scalarCoeffAlpha * d_cellHamMatrixTimesWaveMatrix
                                             [numberWaveFunctions * iNode + i] +
                        scalarY * dst.data()[localNodeId + i] +
                        scalarX * src.data()[localNodeId + i];
                }
            }

        } // cell loop
    }
}
