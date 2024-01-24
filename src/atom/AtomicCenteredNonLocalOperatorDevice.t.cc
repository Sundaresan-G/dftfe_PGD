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
    applyCTonX(
      ValueType **X,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &                                   shapeFnTimesWavefunctionMatrix,
      std::pair<unsigned int, unsigned int> cellRange)
  {
    const ValueType scalarCoeffAlpha = ValueType(1.0),
                    scalarCoeffBeta  = ValueType(0.0);

    d_BLASWrapperPtr->xgemmBatched('N',
                                   'N',
                                   d_numberWaveFunctions,
                                   d_maxSingleAtomContribution,
                                   d_numberNodesPerElement,
                                   &scalarCoeffAlpha,
                                   (const ValueType **)X,
                                   d_numberWaveFunctions,
                                   (const ValueType **)devicePointerCDagger,
                                   d_numberNodesPerElement,
                                   &scalarCoeffBeta,
                                   devicePointerCDaggerOutTemp,
                                   d_numberWaveFunctions,
                                   d_totalNonlocalElems);



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
      shapeFnTimesWavefunctionMatrix.begin(),
      d_numberWaveFunctions);
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyConVCTX(
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
          d_numberNodesPerElement,
      d_maxSingleAtomContribution,
      strideB,
      &scalarCoeffBeta,
      Xout.begin(),
      d_numberWaveFunctions,
      strideC,
      d_totalNonlocalElems);
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyAllReduceonCTX(
      distributedDeviceVec<ValueType>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &shapeFnTimesWavefunctionMatrix)
  {
    dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
      copyToDealiiParallelNonLocalVec(
        d_numberWaveFunctions,
        d_totalNonLocalEntries,
        shapeFnTimesWavefunctionMatrix.begin(),
        sphericalFunctionKetTimesVectorParFlattened.begin(),
        d_shapeFnIdsParallelNumberingMapDevice.begin());


    sphericalFunctionKetTimesVectorParFlattened.accumulateAddLocallyOwned(1);
    sphericalFunctionKetTimesVectorParFlattened.updateGhostValues(1);
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyV_onCTX(
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
      std::shared_ptr<AtomicCenteredNonLocalOperator<
        ValueType,
        dftfe::utils::MemorySpace::HOST>> nonLocalOperatorHost,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr)
  {
    d_numberNodesPerElement = basisOperationsPtr->nDofsPerCell();

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
    MPI_Barrier(d_mpi_communicator);

    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.clear();
    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.resize(
      d_totalNonlocalElems * d_numberNodesPerElement, 0);

    d_shapeFnIdsParallelNumberingMap.clear();
    d_shapeFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);

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

    MPI_Barrier(d_mpi_communicator);

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

            d_shapeFnIdsParallelNumberingMap[countAlpha] = id;

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
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
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

    d_shapeFnIdsParallelNumberingMapDevice.resize(
      d_shapeFnIdsParallelNumberingMap.size());
    d_shapeFnIdsParallelNumberingMapDevice.copyFrom(
      d_shapeFnIdsParallelNumberingMap);

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
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &shapeFnTimesWavefunctionMatrix,
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



    if (d_isMallocCalled)
      {
        free(hostPointerCDagger);
        free(hostPointerCDaggeOutTemp);
        dftfe::utils::deviceFree(devicePointerCDagger);
        dftfe::utils::deviceFree(devicePointerCDaggerOutTemp);
      }
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


    dftfe::utils::deviceMalloc((void **)&devicePointerCDagger,
                               d_totalNonlocalElems * sizeof(ValueType *));

    dftfe::utils::deviceMalloc((void **)&devicePointerCDaggerOutTemp,
                               d_totalNonlocalElems * sizeof(ValueType *));

    dftfe::utils::deviceMemcpyH2D(devicePointerCDaggerOutTemp,
                                  hostPointerCDaggeOutTemp,
                                  d_totalNonlocalElems * sizeof(ValueType *));

    d_isMallocCalled = true;


    shapeFnTimesWavefunctionMatrix.clear();
    shapeFnTimesWavefunctionMatrix.resize(d_numberWaveFunctions *
                                          d_totalNonLocalEntries);
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
        free(hostPointerCDagger);
        free(hostPointerCDaggeOutTemp);
        dftfe::utils::deviceFree(devicePointerCDagger);
        dftfe::utils::deviceFree(devicePointerCDaggerOutTemp);
      }
  }

  template <typename ValueType>
  std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getCellHamiltonianMatrixNonLocalFlattenedConjugate()
  {
    return (d_cellHamiltonianMatrixNonLocalFlattenedConjugate);
  }
  template <typename ValueType>
  std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getCellHamiltonianMatrixNonLocalFlattenedTranspose()
  {
    return (d_cellHamiltonianMatrixNonLocalFlattenedTranspose);
  }
  template <typename ValueType>
  std::vector<dftfe::global_size_type> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getFlattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal()
  {
    return (d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal);
  }

  template <typename ValueType>
  std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getShapeFnIdsParallelNumberingMap()
  {
    return (d_shapeFnIdsParallelNumberingMap);
  }

  template <typename ValueType>
  std::vector<int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getIndexMapFromPaddedNonLocalVecToParallelNonLocalVec()
  {
    return (d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);
  }  

  template <typename ValueType>
  std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getNonLocalElemIdToLocalElemIdMap()
  {
    return (d_nonlocalElemIdToLocalElemIdMap);
  }    
  template <typename ValueType>
  std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    getSphericalFnTimesVectorAllCellsReduction()
  {
    return (d_sphericalFnTimesVectorAllCellsReduction);
  }  

  template <typename ValueType>
  std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
  getCellNodeIdMapNonLocalToLocal()
  {
    return (d_cellNodeIdMapNonLocalToLocal);
  } 

  template <typename ValueType>
  std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
  getAtomWiseNumberCellsInCompactSupport()
  {
    d_numberCellsForEachAtom.clear();
    d_numberCellsForEachAtom.resize(d_totalAtomsInCurrentProc,0);
    std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    for(unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
    {

        const unsigned int        atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
      d_numberCellsForEachAtom[iAtom] = elementIndexesInAtomCompactSupport.size();
    }
    return (d_numberCellsForEachAtom);
  } 

  template <typename ValueType>
  std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
  getAtomWiseNumberCellsAccumulated()
  {
    return (d_numberCellsAccumNonLocalAtoms);
  }


} // namespace dftfe
