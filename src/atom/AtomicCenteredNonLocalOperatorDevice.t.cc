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
#include <AtomicCenteredNonLocalOperatorKernelsDevice.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
namespace dftfe
{
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyVCconjtransOnX(
      const ValueType *       src,
      const unsigned int      kPointIndex,
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &couplingMatrix,
      distributedDeviceVec<ValueType>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    d_basisOperatorPtr->reinit(d_numberWaveFunctions, 0, 0, false);

    initialiseOperatorActionOnX(kPointIndex);
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      cellWaveFunctionMatrix;
    cellWaveFunctionMatrix.resize(d_locallyOwnedCells *
                                    d_numberNodesPerElement *
                                    d_numberWaveFunctions,
                                  0.0);
    initialiseCellWaveFunctionPointers(cellWaveFunctionMatrix);
    if (d_totalNonlocalElems > 0)
      {
        d_BLASWrapperPtr->stridedCopyToBlock(
          d_numberWaveFunctions,
          d_locallyOwnedCells * d_numberNodesPerElement,
          src,
          cellWaveFunctionMatrix.begin(),
          d_basisOperatorPtr->d_flattenedCellDofIndexToProcessDofIndexMap
            .begin());
        applyCconjtrans_onX(
          cellWaveFunctionMatrix,
          std::pair<unsigned int, unsigned int>(0, d_totalNonlocalElems));
      }

    sphericalFunctionKetTimesVectorParFlattened.setValue(0);
    applyAllReduceonCTX(sphericalFunctionKetTimesVectorParFlattened);

    applyV_onCconjtransX(couplingtype,
                         couplingMatrix,
                         sphericalFunctionKetTimesVectorParFlattened,
                         false);
  }



  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyCconjtrans_onX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &X,
      std::pair<unsigned int, unsigned int> cellRange)
  {
    const ValueType scalarCoeffAlpha = ValueType(1.0),
                    scalarCoeffBeta  = ValueType(0.0);

    d_BLASWrapperPtr->xgemmBatched(
      'N',
      'N',
      d_numberWaveFunctions,
      d_maxSingleAtomContribution,
      d_numberNodesPerElement,
      &scalarCoeffAlpha,
      //(X.data() + cellRange.first),
      (const ValueType **)deviceWfcPointers,
      d_numberWaveFunctions,
      //(devicePointerCDagger.data() + cellRange.first),
      (const ValueType **)devicePointerCDagger,
      d_numberNodesPerElement,
      &scalarCoeffBeta,
      devicePointerCDaggerOutTemp,
      // devicePointerCDaggerOutTemp.data() + cellRange.first,
      d_numberWaveFunctions,
      cellRange.second - cellRange.first);


    if (cellRange.second == d_totalNonlocalElems)
      d_BLASWrapperPtr->xgemm(
        'N',
        'N',
        d_numberWaveFunctions,
        d_totalNonLocalEntries,
        d_totalNonlocalElems * d_maxSingleAtomContribution,
        &scalarCoeffAlpha,
        d_sphericalFnTimesVectorAllCellsDevice.begin(),
        d_numberWaveFunctions,
        d_sphericalFnTimesVectorAllCellsReductionDevice.begin(),
        d_totalNonlocalElems * d_maxSingleAtomContribution,
        &scalarCoeffBeta,
        d_sphericalFnTimesWavefunctionMatrix.begin(),
        d_numberWaveFunctions);
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyC_VCconjtransX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange)
  {
    long long int strideA = d_numberWaveFunctions * d_maxSingleAtomContribution;
    long long int strideB =
      d_maxSingleAtomContribution * d_numberNodesPerElement;
    long long int   strideC = d_numberWaveFunctions * d_numberNodesPerElement;
    const ValueType scalarCoeffAlpha = ValueType(1.0),
                    scalarCoeffBeta  = ValueType(0.0);


    d_BLASWrapperPtr->xgemmStridedBatched(
      'N',
      'N',
      d_numberWaveFunctions,
      d_numberNodesPerElement,
      d_maxSingleAtomContribution,
      &scalarCoeffAlpha,
      d_sphericalFnTimesVectorAllCellsDevice.begin(),
      d_numberWaveFunctions,
      strideA,
      d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.begin() +
        d_kPointIndex * d_totalNonlocalElems * d_maxSingleAtomContribution *
          d_numberNodesPerElement +
        cellRange.first * d_maxSingleAtomContribution * d_numberNodesPerElement,
      d_maxSingleAtomContribution,
      strideB,
      &scalarCoeffBeta,
      d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin() +
        cellRange.first * d_numberNodesPerElement,
      d_numberWaveFunctions,
      strideC,
      cellRange.second - cellRange.first);

    if (cellRange.second == d_totalNonlocalElems)
      {
        for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            const unsigned int accum  = d_numberCellsAccumNonLocalAtoms[iAtom];
            const unsigned int Ncells = d_numberCellsForEachAtom[iAtom];

            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              addNonLocalContribution(
                Ncells,
                d_numberNodesPerElement,
                d_numberWaveFunctions,
                accum,
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice,
                Xout,
                d_cellNodeIdMapNonLocalToLocalDevice);
          }
      }
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyAllReduceonCTX(distributedDeviceVec<ValueType>
                          &        sphericalFunctionKetTimesVectorParFlattened,
                        const bool skip1,
                        const bool skip2)
  {
    if (!skip1)
      dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
        copyToDealiiParallelNonLocalVec(
          d_numberWaveFunctions,
          d_totalNonLocalEntries,
          d_sphericalFnTimesWavefunctionMatrix.begin(),
          sphericalFunctionKetTimesVectorParFlattened.begin(),
          d_sphericalFnIdsParallelNumberingMapDevice.begin());

    if (!skip1 && !skip2)
      {
        sphericalFunctionKetTimesVectorParFlattened.accumulateAddLocallyOwned(
          1);
        sphericalFunctionKetTimesVectorParFlattened.updateGhostValues(1);
      }
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyV_onCconjtransX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &couplingMatrix,
      distributedDeviceVec<ValueType>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagCopyToCellVector)
  {
    if (couplingtype == CouplingStructure::diagonal)
      {
        d_BLASWrapperPtr->stridedBlockScale(
          d_numberWaveFunctions,
          d_totalNonLocalEntries,
          1.0,
          couplingMatrix.begin(),
          sphericalFunctionKetTimesVectorParFlattened.begin());
      }

    if (flagCopyToCellVector)
      dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
        copyFromParallelNonLocalVecToAllCellsVec(
          d_numberWaveFunctions,
          d_totalNonlocalElems,
          d_maxSingleAtomContribution,
          sphericalFunctionKetTimesVectorParFlattened.begin(),
          d_sphericalFnTimesVectorAllCellsDevice.begin(),
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.begin());
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    transferCMatrixEntriesfromHostObject(
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::HOST>>
        nonLocalOperatorHost)
  {
    d_numberNodesPerElement = d_basisOperatorPtr->nDofsPerCell();
    d_locallyOwnedCells     = d_basisOperatorPtr->nCells();
    d_cellHamiltonianMatrixNonLocalFlattenedConjugate.clear();
    d_cellHamiltonianMatrixNonLocalFlattenedConjugate.resize(
      d_kPointWeights.size() * d_totalNonlocalElems * d_numberNodesPerElement *
        d_maxSingleAtomContribution,
      ValueType(0.0));
    d_cellHamiltonianMatrixNonLocalFlattenedTranspose.clear();
    d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
      d_kPointWeights.size() * d_totalNonlocalElems * d_numberNodesPerElement *
        d_maxSingleAtomContribution,
      ValueType(0.0));
    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.clear();
    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.resize(
      d_totalNonlocalElems * d_numberNodesPerElement, 0);

    d_sphericalFnIdsParallelNumberingMap.clear();
    d_sphericalFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);

    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
      d_totalNonlocalElems * d_maxSingleAtomContribution, -1);

    d_nonlocalElemIdToLocalElemIdMap.clear();
    d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);

    d_sphericalFnTimesVectorAllCellsReduction.resize(
      d_totalNonlocalElems * d_maxSingleAtomContribution *
        d_totalNonLocalEntries,
      ValueType(0.0));

    d_cellNodeIdMapNonLocalToLocal.resize(d_totalNonlocalElems *
                                          d_numberNodesPerElement);



    std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();


    unsigned int countElemNode    = 0;
    unsigned int countElem        = 0;
    unsigned int countAlpha       = 0;
    unsigned int numShapeFnsAccum = 0;

    int totalElements = 0;
    for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
        const unsigned int        atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        unsigned int totalAtomIdElementIterators =
          elementIndexesInAtomCompactSupport.size();
        totalElements += totalAtomIdElementIterators;
        const unsigned int Zno = atomicNumber[atomId];
        const unsigned int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);

        for (unsigned int alpha = 0; alpha < numberSphericalFunctions; alpha++)
          {
            unsigned int globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, alpha)];

            const unsigned int id = d_SphericalFunctionKetTimesVectorPar[0]
                                      .get_partitioner()
                                      ->global_to_local(globalId);

            d_sphericalFnIdsParallelNumberingMap[countAlpha] = id;

            for (unsigned int iElemComp = 0;
                 iElemComp < totalAtomIdElementIterators;
                 iElemComp++)
              {
                d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                  [d_numberCellsAccumNonLocalAtoms[iAtom] *
                     d_maxSingleAtomContribution +
                   iElemComp * d_maxSingleAtomContribution + alpha] = id;
              }
            countAlpha++;
          }
        for (unsigned int iElemComp = 0;
             iElemComp < totalAtomIdElementIterators;
             ++iElemComp)
          {
            const unsigned int elementId =
              elementIndexesInAtomCompactSupport[iElemComp];

            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                dftfe::global_size_type localNodeId =
                  d_basisOperatorPtr->d_cellDofIndexToProcessDofIndexMap
                    [elementId * d_numberNodesPerElement + iNode];
                d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal
                  [countElemNode] = localNodeId;
                d_cellNodeIdMapNonLocalToLocal[countElemNode] =
                  elementId * d_numberNodesPerElement + iNode;
                countElemNode++;
              }
          }

        for (unsigned int iElemComp = 0;
             iElemComp < totalAtomIdElementIterators;
             ++iElemComp)
          {
            const unsigned int elementId =
              elementIndexesInAtomCompactSupport[iElemComp];
            d_nonlocalElemIdToLocalElemIdMap[countElem] = elementId;

            for (unsigned int ikpoint = 0; ikpoint < d_kPointWeights.size();
                 ikpoint++)
              for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                   ++iNode)
                {
                  for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                       ++alpha)
                    {
                      d_cellHamiltonianMatrixNonLocalFlattenedConjugate
                        [ikpoint * d_totalNonlocalElems *
                           d_numberNodesPerElement *
                           d_maxSingleAtomContribution +
                         countElem * d_maxSingleAtomContribution *
                           d_numberNodesPerElement +
                         d_numberNodesPerElement * alpha + iNode] =
                          nonLocalOperatorHost->d_CMatrixEntriesConjugate
                            [atomId][iElemComp]
                            [ikpoint * d_numberNodesPerElement *
                               numberSphericalFunctions +
                             d_numberNodesPerElement * alpha + iNode];

                      d_cellHamiltonianMatrixNonLocalFlattenedTranspose
                        [ikpoint * d_totalNonlocalElems *
                           d_numberNodesPerElement *
                           d_maxSingleAtomContribution +
                         countElem * d_numberNodesPerElement *
                           d_maxSingleAtomContribution +
                         d_maxSingleAtomContribution * iNode + alpha] =
                          nonLocalOperatorHost->d_CMatrixEntriesTranspose
                            [atomId][iElemComp]
                            [ikpoint * d_numberNodesPerElement *
                               numberSphericalFunctions +
                             numberSphericalFunctions * iNode + alpha];
                    }
                }


            for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                 ++alpha)
              {
                const unsigned int columnStartId = (numShapeFnsAccum + alpha) *
                                                   d_totalNonlocalElems *
                                                   d_maxSingleAtomContribution;
                const unsigned int columnRowId =
                  countElem * d_maxSingleAtomContribution + alpha;
                d_sphericalFnTimesVectorAllCellsReduction[columnStartId +
                                                          columnRowId] =
                  ValueType(1.0);
              }

            countElem++;
          }

        numShapeFnsAccum += numberSphericalFunctions;
      }


    // Do the copying from HostPtr to temp
    // structures
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

    d_sphericalFnIdsParallelNumberingMapDevice.resize(
      d_sphericalFnIdsParallelNumberingMap.size());
    d_sphericalFnIdsParallelNumberingMapDevice.copyFrom(
      d_sphericalFnIdsParallelNumberingMap);

    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.resize(
      d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size());
    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.copyFrom(
      d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);

    d_sphericalFnTimesVectorAllCellsReductionDevice.resize(
      d_sphericalFnTimesVectorAllCellsReduction.size());
    d_sphericalFnTimesVectorAllCellsReductionDevice.copyFrom(
      d_sphericalFnTimesVectorAllCellsReduction);

    // d_nonLocalPseudoPotentialConstantsDevice.resize(
    //   d_nonLocalPseudoPotentialConstants.size());
    // d_nonLocalPseudoPotentialConstantsDevice.copyFrom(
    //   d_nonLocalPseudoPotentialConstants);

    d_cellNodeIdMapNonLocalToLocalDevice.resize(
      d_cellNodeIdMapNonLocalToLocal.size());
    d_cellNodeIdMapNonLocalToLocalDevice.copyFrom(
      d_cellNodeIdMapNonLocalToLocal);
  }


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseOperatorActionOnX(unsigned int kPointIndex)
  {
    d_kPointIndex = kPointIndex;

    for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
      {
        hostPointerCDagger[i] =
          d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.begin() +
          d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomContribution +
          i * d_numberNodesPerElement * d_maxSingleAtomContribution;
      }

    dftfe::utils::deviceMemcpyH2D(devicePointerCDagger,
                                  hostPointerCDagger,
                                  d_totalNonlocalElems * sizeof(ValueType *));
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    d_numberWaveFunctions = numberWaveFunctions;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
      numberWaveFunctions,
      sphericalFunctionKetTimesVectorParFlattened);
    d_sphericalFnTimesVectorAllCellsDevice.resize(d_totalNonlocalElems *
                                                    d_numberWaveFunctions *
                                                    d_maxSingleAtomContribution,
                                                  ValueType(0.0));


    d_cellHamMatrixTimesWaveMatrixNonLocalDevice.resize(
      d_numberWaveFunctions * d_totalNonlocalElems * d_numberNodesPerElement,
      ValueType(0.0));
    if (d_isMallocCalled)
      {
        free(hostWfcPointers);
        dftfe::utils::deviceFree(deviceWfcPointers);
        free(hostPointerCDagger);
        free(hostPointerCDaggeOutTemp);
        dftfe::utils::deviceFree(devicePointerCDagger);
        dftfe::utils::deviceFree(devicePointerCDaggerOutTemp);
      }
    hostWfcPointers =
      (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));
    hostPointerCDagger =
      (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));
    hostPointerCDaggeOutTemp =
      (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));

    for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
      {
        hostPointerCDaggeOutTemp[i] =
          d_sphericalFnTimesVectorAllCellsDevice.begin() +
          i * d_numberWaveFunctions * d_maxSingleAtomContribution;
      }

    dftfe::utils::deviceMalloc((void **)&deviceWfcPointers,
                               d_totalNonlocalElems * sizeof(ValueType *));


    dftfe::utils::deviceMalloc((void **)&devicePointerCDagger,
                               d_totalNonlocalElems * sizeof(ValueType *));

    dftfe::utils::deviceMalloc((void **)&devicePointerCDaggerOutTemp,
                               d_totalNonlocalElems * sizeof(ValueType *));

    dftfe::utils::deviceMemcpyH2D(devicePointerCDaggerOutTemp,
                                  hostPointerCDaggeOutTemp,
                                  d_totalNonlocalElems * sizeof(ValueType *));

    d_isMallocCalled = true;


    d_sphericalFnTimesWavefunctionMatrix.clear();
    d_sphericalFnTimesWavefunctionMatrix.resize(d_numberWaveFunctions *
                                                d_totalNonLocalEntries);
  }
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix)
  {
    for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
      {
        hostWfcPointers[i] = cellWaveFunctionMatrix.begin() +
                             d_nonlocalElemIdToLocalElemIdMap[i] *
                               d_numberWaveFunctions * d_numberNodesPerElement;
      }
    dftfe::utils::deviceMemcpyH2D(deviceWfcPointers,
                                  hostWfcPointers,
                                  d_totalNonlocalElems * sizeof(ValueType *));
  }



  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseDeviceVectors()
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    freeDeviceVectors()
  {
    if (d_isMallocCalled)
      {
        free(hostWfcPointers);
        dftfe::utils::deviceFree(deviceWfcPointers);
        free(hostPointerCDagger);
        free(hostPointerCDaggeOutTemp);
        dftfe::utils::deviceFree(devicePointerCDagger);
        dftfe::utils::deviceFree(devicePointerCDaggerOutTemp);
      }
  }



} // namespace dftfe
