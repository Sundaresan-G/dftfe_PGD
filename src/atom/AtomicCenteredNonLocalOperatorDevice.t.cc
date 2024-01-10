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
// @author Kartick Ramakrishnan
//


namespace dftfe
{
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst)
  {}


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    transferCMatrixEntriesfromHostObject(
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::HOST>>
        nonLocalOperatorHost,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr        )
  {

    const unsigned int numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWaveFunctions = 0;  
    d_cellHamMatrixTimesWaveMatrixNonLocalDevice.resize(d_totalNonlocalElems*numberWaveFunctions*numberNodesPerElement);
        d_cellHamiltonianMatrixNonLocalFlattenedConjugate.clear();
        d_cellHamiltonianMatrixNonLocalFlattenedConjugate.resize(
          d_kPointWeights.size() * d_totalNonlocalElems *
            numberNodesPerElement * d_maxSingleAtomContribution,
          dataTypes::number(0.0));
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.clear();
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
          d_kPointWeights.size() * d_totalNonlocalElems *
            numberNodesPerElement * d_maxSingleAtomContribution,
          dataTypes::number(0.0));
        
        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.clear();
        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.resize(
          d_totalNonlocalElems * numberNodesPerElement, 0);
        d_projectorKetTimesVectorAllCellsDevice.resize(
          d_totalNonlocalElems * numberWaveFunctions * d_maxSingleAtomContribution,
          dataTypes::number(0.0));
        d_projectorIdsParallelNumberingMap.clear();
        d_projectorIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);
        d_projectorKetTimesVectorParFlattenedDevice.resize(
          numberWaveFunctions * d_totalNonLocalEntries, 0.0);
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution, -1);
        d_nonlocalElemIdToLocalElemIdMap.clear();
        d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);
        d_projectorKetTimesVectorAllCellsReduction.clear();
        d_projectorKetTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution *
            d_totalNonLocalEntries,
          dataTypes::number(0.0));

        d_cellNodeIdMapNonLocalToLocal.clear();
        d_cellNodeIdMapNonLocalToLocal.resize(d_totalNonlocalElems *
                                              numberNodesPerElement);
      std::vector<unsigned int> atomIdsInCurrentProcess = d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();


        unsigned int countElemNode   = 0;
        unsigned int countElem       = 0;
        unsigned int countAlpha = 0;
      unsigned int numShapeFnsAccum = 0;
      for(int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
          const unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];

          const unsigned int Zno = atomicNumber[atomId];
          const unsigned int numberSphericalFunctions = d_atomCenteredSphericalFunctionContainer->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
          
          for(unsigned int alpha = 0; alpha < numberSphericalFunctions; alpha ++)
          {
                const unsigned int id = 0;
                //  id =
                //   dftPtr->d_projectorKetTimesVectorPar[0]
                //     .get_partitioner()
                //     ->global_to_local(
                //       dftPtr->d_projectorIdsNumberingMapCurrentProcess
                //         [std::make_pair(atomId, ipseudowfc)]);

                // d_projectorIdsParallelNumberingMap[countAlpha] = id;

              for(unsigned int iElemComp = 0; iElemComp < d_totalNonlocalElems; iElemComp++)
              {
                  d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                    [d_numberCellsAccumNonLocalAtoms[iAtom] *
                       d_maxSingleAtomContribution +
                     iElemComp * d_maxSingleAtomContribution + alpha] =
                      id; 

                countAlpha++;                  
              }              
          }
            for (unsigned int iElemComp = 0;
                 iElemComp <
                 d_totalNonlocalElems;
                 ++iElemComp)
              {
                const unsigned int elementId =
                  elementIndexesInAtomCompactSupport[iElemComp];
                for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                     ++iNode)
                  {
                    dftfe::global_size_type localNodeId = numberWaveFunctions*basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap[elementId*numberNodesPerElement+iNode];
                    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal
                      [countElemNode] = localNodeId;
                    d_cellNodeIdMapNonLocalToLocal[countElemNode] =
                      elementId * numberNodesPerElement + iNode;
                    countElemNode++;
                  }
              }

            for (unsigned int iElemComp = 0;
                 iElemComp <
                 d_totalNonlocalElems;
                 ++iElemComp)
              {
                const unsigned int elementId =
                  elementIndexesInAtomCompactSupport[iElemComp];
                d_nonlocalElemIdToLocalElemIdMap[countElem] = elementId;

                for (unsigned int ikpoint = 0;
                     ikpoint < d_kPointWeights.size();
                     ikpoint++)
                  for (unsigned int iNode = 0; iNode < numberNodesPerElement;
                       ++iNode)
                    {
                      for (unsigned int alpha = 0;
                           alpha < numberSphericalFunctions;
                           ++alpha)
                        {
                          d_cellHamiltonianMatrixNonLocalFlattenedConjugate
                            [ikpoint * d_totalNonlocalElems *
                               numberNodesPerElement *
                               d_maxSingleAtomContribution +
                             countElem * d_maxSingleAtomContribution *
                               numberNodesPerElement +
                             numberNodesPerElement * alpha + iNode] =
                              nonLocalOperatorHost ->d_CMatrixEntriesConjugate
                                  [atomId][iElemComp]
                                  [ikpoint * numberNodesPerElement *
                                     numberSphericalFunctions +
                                   numberNodesPerElement * alpha +
                                   iNode];

                          d_cellHamiltonianMatrixNonLocalFlattenedTranspose
                            [ikpoint * d_totalNonlocalElems *
                               numberNodesPerElement *
                               d_maxSingleAtomContribution +
                             countElem * numberNodesPerElement *
                               d_maxSingleAtomContribution +
                             d_maxSingleAtomContribution * iNode + alpha] =
                              nonLocalOperatorHost ->d_CMatrixEntriesTranspose
                                  [atomId][iElemComp]
                                  [ikpoint * numberNodesPerElement *
                                     numberSphericalFunctions +
                                   numberSphericalFunctions * iNode +
                                   alpha];
                        }
                    }


                for (unsigned int alpha = 0;
                     alpha < numberSphericalFunctions;
                     ++alpha)
                  {
                    const unsigned int columnStartId =
                      (numShapeFnsAccum + alpha) *
                      d_totalNonlocalElems * d_maxSingleAtomContribution;
                    const unsigned int columnRowId =
                      countElem * d_maxSingleAtomContribution + alpha;
                    d_projectorKetTimesVectorAllCellsReduction[columnStartId +
                                                               columnRowId] =
                      dataTypes::number(1.0);
                  }

                countElem++;
              }

          numShapeFnsAccum += numberSphericalFunctions;
      }



        //Do the copying from HostPtr to temp structures
        d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.resize(
          d_cellHamiltonianMatrixNonLocalFlattenedConjugate.size());
        d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.copyFrom(
          d_cellHamiltonianMatrixNonLocalFlattenedConjugate);

        d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.resize(
          d_cellHamiltonianMatrixNonLocalFlattenedTranspose.size());
        d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.copyFrom(
          d_cellHamiltonianMatrixNonLocalFlattenedTranspose);

        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice.resize(
          d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.size());
        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice.copyFrom(
          d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal);

        d_projectorIdsParallelNumberingMapDevice.resize(
          d_projectorIdsParallelNumberingMap.size());
        d_projectorIdsParallelNumberingMapDevice.copyFrom(
          d_projectorIdsParallelNumberingMap);

        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.resize(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size());
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.copyFrom(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);

        d_projectorKetTimesVectorAllCellsReductionDevice.resize(
          d_projectorKetTimesVectorAllCellsReduction.size());
        d_projectorKetTimesVectorAllCellsReductionDevice.copyFrom(
          d_projectorKetTimesVectorAllCellsReduction);

        d_nonLocalPseudoPotentialConstantsDevice.resize(
          d_nonLocalPseudoPotentialConstants.size());
        d_nonLocalPseudoPotentialConstantsDevice.copyFrom(
          d_nonLocalPseudoPotentialConstants);

        d_cellNodeIdMapNonLocalToLocalDevice.resize(
          d_cellNodeIdMapNonLocalToLocal.size());
        d_cellNodeIdMapNonLocalToLocalDevice.copyFrom(
          d_cellNodeIdMapNonLocalToLocal);





  }
} // namespace dftfe
