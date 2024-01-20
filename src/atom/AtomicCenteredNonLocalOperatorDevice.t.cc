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
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: Number of wfc: " << d_numberWaveFunctions <<
    // std::endl; std::cout << "DEBUG: Number of Nodes per element: "
    //           << d_numberNodesPerElement << std::endl;
    // std::cout << "DEBUG: MAX atom Contribution: " <<
    // d_maxSingleAtomContribution
    //           << std::endl;
    // std::cout << "DEBUG: Total Nonlocal Elements: " << d_totalNonlocalElems
    //           << std::endl;
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: CTX Line 38" << std::endl;
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

    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: CTX Line 53" << std::endl;


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
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: CTX Line 70" << std::endl;
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


    sphericalFunctionKetTimesVectorParFlattened.accumulateAddLocallyOwned();
    sphericalFunctionKetTimesVectorParFlattened.updateGhostValues();
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
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    //   dftfe::utils::MemoryStorage<double,
    //                           dftfe::utils::MemorySpace::HOST>
    //                           nonLocalPseudoPotentialConstantsHost;
    //     nonLocalPseudoPotentialConstantsHost.resize(couplingMatrix.size());
    //     nonLocalPseudoPotentialConstantsHost.copyFrom(couplingMatrix) ;

    // // for(int i = 0; i < nonLocalPseudoPotentialConstantsHost.size(); i++)
    // // {
    // //   std::cout<<"nonlocal COnstatns:
    // "<<nonLocalPseudoPotentialConstantsHost[i]<<std::endl;
    // // }
    // // std::cout<<std::endl;


    if (couplingtype == CouplingStructure::diagonal)
      {
        d_BLASWrapperPtr->stridedBlockScale(
          d_numberWaveFunctions,
          d_totalNonLocalEntries,
          1.0,
          couplingMatrix.begin(),
          sphericalFunctionKetTimesVectorParFlattened.begin());
      }

    dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
      copyFromParallelNonLocalVecToAllCellsVec(
        d_numberWaveFunctions,
        d_totalNonlocalElems,
        d_maxSingleAtomContribution,
        sphericalFunctionKetTimesVectorParFlattened.begin(),
        d_sphericalFnTimesVectorAllCellsDevice.begin(),
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.begin());

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      tempprojectorKetReduceHost;
    tempprojectorKetReduceHost.resize(
      d_sphericalFnTimesVectorAllCellsDevice.size());
    tempprojectorKetReduceHost.copyFrom(d_sphericalFnTimesVectorAllCellsDevice);

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
    // std::cout << "DEBUG: TOtal number of nodes: " << d_numberNodesPerElement
    //           << std::endl;
    // std::cout << "DEBUG: TOtal nonLocal ENtries: " << d_totalNonLocalEntries
    //           << std::endl;
    // std::cout << "DEBUG: TOtal nonlocal elements: " << d_totalNonlocalElems
    //           << std::endl;


    d_cellHamiltonianMatrixNonLocalFlattenedConjugate.clear();
    d_cellHamiltonianMatrixNonLocalFlattenedConjugate.resize(
      d_kPointWeights.size() * d_totalNonlocalElems * d_numberNodesPerElement *
        d_maxSingleAtomContribution,
      ValueType(0.0));
    // std::cout
    //   << "SIZES DEBUG:
    //   d_cellHamiltonianMatrixNonLocalFlattenedConjugate.size(): "
    //   << d_cellHamiltonianMatrixNonLocalFlattenedConjugate.size() <<
    //   std::endl;
    d_cellHamiltonianMatrixNonLocalFlattenedTranspose.clear();
    d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
      d_kPointWeights.size() * d_totalNonlocalElems * d_numberNodesPerElement *
        d_maxSingleAtomContribution,
      ValueType(0.0));
    // std::cout
    //   << "SIZES DEBUG:
    //   d_cellHamiltonianMatrixNonLocalFlattenedTranspose.size(): "
    //   << d_cellHamiltonianMatrixNonLocalFlattenedTranspose.size() <<
    //   std::endl;
    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.clear();
    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.resize(
      d_totalNonlocalElems * d_numberNodesPerElement, 0);

    d_shapeFnIdsParallelNumberingMap.clear();
    d_shapeFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);
    // std::cout << "SIZES DEBUG: d_shapeFnIdsParallelNumberingMap.size(): "
    //           << d_shapeFnIdsParallelNumberingMap.size() << std::endl;
    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
      d_totalNonlocalElems * d_maxSingleAtomContribution, -1);
    // std::cout
    //   << "SIZES DEBUG:
    //   d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size(): "
    //   << d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size()
    //   << std::endl;
    d_nonlocalElemIdToLocalElemIdMap.clear();
    d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);
    // std::cout << "SIZES DEBUG: d_nonlocalElemIdToLocalElemIdMap.size(): "
    //           << d_nonlocalElemIdToLocalElemIdMap.size() << std::endl;
    // d_sphericalFnTimesVectorAllCellsReduction.clear();
    d_sphericalFnTimesVectorAllCellsReduction.resize(
      d_totalNonlocalElems * d_maxSingleAtomContribution *
        d_totalNonLocalEntries,
      ValueType(0.0));
    // std::cout
    //   << "SIZES DEBUG: d_sphericalFnTimesVectorAllCellsReduction.size(): "
    //   << d_sphericalFnTimesVectorAllCellsReduction.size() << std::endl;
    // d_cellNodeIdMapNonLocalToLocal.clear();
    d_cellNodeIdMapNonLocalToLocal.resize(d_totalNonlocalElems *
                                          d_numberNodesPerElement);
    // std::cout << "SIZES DEBUG: d_cellNodeIdMapNonLocalToLocal.size(): "
    //           << d_cellNodeIdMapNonLocalToLocal.size() << std::endl;



    std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();


    unsigned int countElemNode    = 0;
    unsigned int countElem        = 0;
    unsigned int countAlpha       = 0;
    unsigned int numShapeFnsAccum = 0;


    for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
        // std::cout<<"DEBUG: Line 225"<<std::endl;
        const unsigned int        atomId = atomIdsInCurrentProcess[iAtom];
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        unsigned int totalAtomIdElementIterators =
          elementIndexesInAtomCompactSupport.size();

        const unsigned int Zno = atomicNumber[atomId];
        const unsigned int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);

        for (unsigned int alpha = 0; alpha < numberSphericalFunctions; alpha++)
          {
            // std::cout<<"DEBUG: Line 238"<<std::endl;
            unsigned int globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, alpha)];
            // std::cout<<"DEBUG: global ID: "<<globalId<<std::endl;
            const unsigned int id = d_SphericalFunctionKetTimesVectorPar[0]
                                      .get_partitioner()
                                      ->global_to_local(globalId);
            // std::cout<<"DEBUG: id "<<id<<std::endl;
            d_shapeFnIdsParallelNumberingMap[countAlpha] = id;

            for (unsigned int iElemComp = 0;
                 iElemComp < totalAtomIdElementIterators;
                 iElemComp++)
              {
                d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                  [d_numberCellsAccumNonLocalAtoms[iAtom] *
                     d_maxSingleAtomContribution +
                   iElemComp * d_maxSingleAtomContribution + alpha] = id;

                // std::cout << "DEBUGIMP MAP: " << id << " " << countAlpha << " "
                //           << (d_numberCellsAccumNonLocalAtoms[iAtom] *
                //                 d_maxSingleAtomContribution +
                //               iElemComp * d_maxSingleAtomContribution + alpha)
                //           << std::endl;
              }
            countAlpha++;
          }
        for (unsigned int iElemComp = 0;
             iElemComp < totalAtomIdElementIterators;
             ++iElemComp)
          {
            const unsigned int elementId =
              elementIndexesInAtomCompactSupport[iElemComp];
            // std::cout<<"DEBUG: Line 275 and elementId"<<"
            // "<<elementId<<std::endl;
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                dftfe::global_size_type localNodeId =
                  d_numberWaveFunctions *
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    [elementId * d_numberNodesPerElement + iNode];
                // std::cout<<"LocalNode ID: "<<  localNodeId<<std::endl;
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

    // std::cout<<"DEBUG: Line 344"<<std::endl;

    // Do the copying from HostPtr to temp structures
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

    d_nonLocalPseudoPotentialConstantsDevice.resize(
      d_nonLocalPseudoPotentialConstants.size());
    d_nonLocalPseudoPotentialConstantsDevice.copyFrom(
      d_nonLocalPseudoPotentialConstants);

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
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: Filling CT on GPU" << std::endl;
    for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
      {
        hostPointerCDagger[i] =
          d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.begin() +
          d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomContribution +
          i * d_numberNodesPerElement * d_maxSingleAtomContribution;
      }
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: starting H2D copy Line 460" << std::endl;
    dftfe::utils::deviceMemcpyH2D(devicePointerCDagger,
                                  hostPointerCDagger,
                                  d_totalNonlocalElems * sizeof(ValueType *));
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: Finished H2D copy Line 460" << std::endl;
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
    // std::cout << "SIZES DEBUG: d_sphericalFnTimesVectorAllCellsDevice.size():
    // "
    //           << d_sphericalFnTimesVectorAllCellsDevice.size() << std::endl;


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

    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: starting cudaMalloc Line 504" << std::endl;
    dftfe::utils::deviceMalloc((void **)&devicePointerCDagger,
                               d_totalNonlocalElems * sizeof(ValueType *));
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: starting cudaMalloc Line 508" << std::endl;
    dftfe::utils::deviceMalloc((void **)&devicePointerCDaggerOutTemp,
                               d_totalNonlocalElems * sizeof(ValueType *));
    // dftfe::utils::deviceSynchronize();
    // std::cout << "DEBUG: starting H2D copy Line 512" << std::endl;
    dftfe::utils::deviceMemcpyH2D(devicePointerCDaggerOutTemp,
                                  hostPointerCDaggeOutTemp,
                                  d_totalNonlocalElems * sizeof(ValueType *));

    d_isMallocCalled = true;


    shapeFnTimesWavefunctionMatrix.clear();
    shapeFnTimesWavefunctionMatrix.resize(d_numberWaveFunctions *
                                          d_totalNonLocalEntries);

    // std::cout << "SIZES DEBUG: shapeFnTimesWavefunctionMatrix.size(): "
    //           << shapeFnTimesWavefunctionMatrix.size() << std::endl;
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

} // namespace dftfe
