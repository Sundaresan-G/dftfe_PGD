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
// @author  Kartick Ramakrishnan, Vishal Subramanian, Sambit Das
//

#ifndef DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
#define DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
#include <MultiVector.h>
#include <headers.h>
#include <AtomCenteredSphericalFunctionContainer.h>
#include <sphericalHarmonicUtils.h>
#include <BLASWrapper.h>
#include <memory>
#include <MemorySpaceType.h>
#include "FEBasisOperations.h"
#include <headers.h>
#include <dftUtils.h>
#include <pseudoUtils.h>
#include <vectorUtilities.h>
#include <MPIPatternP2P.h>
#include <MultiVector.h>
#include <DeviceTypeConfig.h>

namespace dftfe
{
  /**
   * @brief Enum class that lists
   * used in the non-local Operator
   *
   */
  enum class CouplingStructure
  {
    diagonal,
    dense,
    blockDiagonal
  };



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class AtomicCenteredNonLocalOperatorBase
  {
  public:
    AtomicCenteredNonLocalOperatorBase(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<AtomCenteredSphericalFunctionContainer>
                         atomCenteredSphericalFunctionContainer,
      const unsigned int numVectors,
      const MPI_Comm &   mpi_comm_parent);

    void
    initKpoints(const std::vector<double> &kPointWeights,
                const std::vector<double> &kPointCoordinates);


    void
    InitalisePartitioner(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr);



    // virtual void
    // initialiseFlattenedDataStructure(
    //   unsigned int numberWaveFunctions,
    //   std::map<unsigned int,
    //            dftfe::utils::MemoryStorage<ValueType, memorySpace>>
    //     &sphericalFnTimesWavefunctionMatri,
    //   dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
    //     &sphericalFunctionKetTimesVectorParFlattened) = 0;

    unsigned int
    getTotalAtomInCurrentProcessor();

    unsigned int
    getTotalNonLocalElementsInCurrentProcessor();

    unsigned int
    getTotalNonLocalEntriesCurrentProcessor();

    unsigned int
    getMaxSingleAtomEntries();

    bool
    atomSupportInElement(unsigned int iElem);

    unsigned int
    getGlobalIdofAtomIdSphericalFnPair(const unsigned int atomId,
                                       const unsigned int alpha);

    unsigned int
    getLocalIdOfDistributedVec(const unsigned int globalId);


  protected:
    unsigned int        d_numberOfVectors;
    bool                d_AllReduceCompleted;
    std::vector<double> d_kPointWeights;
    std::vector<double> d_kPointCoordinates;
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomCenteredSphericalFunctionContainer;
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                              d_mpiPatternP2P;
    std::vector<unsigned int> d_numberCellsForEachAtom;

#ifdef USE_COMPLEX
    std::vector<distributedCPUVec<std::complex<double>>>
      d_SphericalFunctionKetTimesVectorPar;

#else
    std::vector<distributedCPUVec<double>> d_SphericalFunctionKetTimesVectorPar;
#endif

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_sphericalFunctionIdsNumberingMapCurrentProcess;

    std::map<unsigned int, std::vector<unsigned int>>
      d_cellIdToAtomIdsLocalCompactSupportMap;

    dealii::IndexSet d_locallyOwnedAtomCenteredFnIdsCurrentProcess;
    dealii::IndexSet d_ghostAtomCenteredFnIdsCurrentProcess;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_AtomCenteredFnIdsNumberingMapCurrentProcess;
    std::vector<std::vector<
      std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>>>>
                               d_CMatrixEntries;
    dealii::ConditionalOStream pcout;
    const MPI_Comm             d_mpi_communicator;
    const unsigned int         d_this_mpi_process;
    const unsigned int         d_n_mpi_processes;
    dealii::IndexSet           d_locallyOwnedSphericalFunctionIdsCurrentProcess;
    dealii::IndexSet           d_ghostSphericalFunctionIdsCurrentProcess;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      d_SphericalFunctionKetTimesVectorParFlattened;


    unsigned int d_totalAtomsInCurrentProc; // number of atoms of interst with
                                            // compact in current processor
    unsigned int
      d_totalNonlocalElems; // number of nonlocal FE celss having nonlocal
                            // contribution in current processor
    unsigned int d_totalNonLocalEntries; // Total number of nonlocal components
    unsigned int
      d_maxSingleAtomContribution; // maximum number of nonlocal indexes across
                                   // all atoms of interset
    std::vector<unsigned int> d_numberCellsAccumNonLocalAtoms;
    unsigned int d_numberNodesPerElement; // Access from BasisOperator WHile
                                          // filling CMatrixEntries
    unsigned int d_locallyOwnedCells;
    unsigned int d_numberWaveFunctions;
    unsigned int d_kPointIndex;
    bool         d_isMallocCalled = false;
  };

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class AtomicCenteredNonLocalOperator
    : AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>
  {};

  template <typename ValueType>
  class AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::HOST>
    : public AtomicCenteredNonLocalOperatorBase<ValueType,
                                                dftfe::utils::MemorySpace::HOST>
  {
  public:
    void
    initialiseOperatorActionOnX(
      unsigned int kPointIndex,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix);

    void
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened);

    void
    computeCMatrixEntries(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr,
      const unsigned int quadratureIndex,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost);

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_numberNodesPerElement;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_this_mpi_process;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_AllReduceCompleted;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_numberWaveFunctions;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_kPointIndex;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_kPointWeights;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_kPointCoordinates;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_numberOfVectors;

    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_sphericalFunctionIdsNumberingMapCurrentProcess;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::pcout;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_mpi_communicator;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_SphericalFunctionKetTimesVectorPar;
    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_SphericalFunctionKetTimesVectorParFlattened;
    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_totalAtomsInCurrentProc; // number of atoms of interst with
                                 // compact in current processor
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_locallyOwnedCells;

    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_totalNonlocalElems; // number of nonlocal FE celss having nonlocal
                            // contribution in current processor
    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_totalNonLocalEntries; // Total number of nonlocal components
    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_maxSingleAtomContribution; // maximum number of nonlocal indexes across
                                   // all atoms of interset
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_numberCellsAccumNonLocalAtoms;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::AtomicCenteredNonLocalOperatorBase;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::HOST>::d_BLASWrapperPtr;


    // using AtomicCenteredNonLocalOperatorBase<
    //   ValueType,
    //   dftfe::utils::MemorySpace::HOST>::d_CouplingMatrixFlattened;
    using AtomicCenteredNonLocalOperatorBase<ValueType,
                                             dftfe::utils::MemorySpace::HOST>::
      d_atomCenteredSphericalFunctionContainer;



    std::vector<std::vector<std::vector<ValueType>>> d_CMatrixEntriesConjugate,
      d_CMatrixEntriesTranspose;


    void
    applyV_onCTX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix);

    void
    applyAllReduceonCTX(
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix);


    void
    applyCTonX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix,
      const std::pair<unsigned int, unsigned int> cellRange);



    void
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &Xout,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &sphericalFnTimesWavefunctionMatrix,
      const std::pair<unsigned int, unsigned int> cellRange);



  private:
    std::map<unsigned int, std::vector<ValueType>> d_ShapeFnTimesWavefunction;
  };

  template <typename ValueType>
  class AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::DEVICE>
    : public AtomicCenteredNonLocalOperatorBase<
        ValueType,
        dftfe::utils::MemorySpace::DEVICE>
  {
  public:
    void
    initialiseOperatorActionOnX(unsigned int kPointIndex);

    void
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesWavefunctionMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix);

    void
    initialiseDeviceVectors();

    void
    freeDeviceVectors();
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_BLASWrapperPtr;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_mpi_communicator;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_this_mpi_process;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_atomCenteredSphericalFunctionContainer;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::AtomicCenteredNonLocalOperatorBase;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_SphericalFunctionKetTimesVectorParFlattened;

    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_SphericalFunctionKetTimesVectorPar;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_numberCellsForEachAtom;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_numberNodesPerElement;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_numberWaveFunctions;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_kPointIndex;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_AllReduceCompleted;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_kPointWeights;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_kPointCoordinates;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_numberOfVectors;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_isMallocCalled;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_sphericalFunctionIdsNumberingMapCurrentProcess;
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_totalAtomsInCurrentProc; // number of atoms of interst with
                                 // compact in current processor
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_totalNonlocalElems; // number of nonlocal FE celss having nonlocal
                            // contribution in current processor
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_totalNonLocalEntries; // Total number of nonlocal components
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::
      d_maxSingleAtomContribution; // maximum number of nonlocal indexes across
                                   // all atoms of interset
    using AtomicCenteredNonLocalOperatorBase<
      ValueType,
      dftfe::utils::MemorySpace::DEVICE>::d_numberCellsAccumNonLocalAtoms;
    void
    transferCMatrixEntriesfromHostObject(
      std::shared_ptr<AtomicCenteredNonLocalOperator<
        ValueType,
        dftfe::utils::MemorySpace::HOST>> nonLocalOperatorHost,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr);



    void
    applyCTonX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &                                   sphericalFnTimesWavefunctionMatrix,
      std::pair<unsigned int, unsigned int> cellRange);

    void
    applyAllReduceonCTX(
      distributedDeviceVec<ValueType>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesWavefunctionMatrix);

    void
    applyV_onCTX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &couplingMatrix,
      distributedDeviceVec<ValueType>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagCopyToCellVector);


    void
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange);



    std::vector<unsigned int> &
    getNonLocalElemIdToLocalElemIdMap();

    std::vector<unsigned int> &
    getAtomWiseNumberCellsInCompactSupport();

    std::vector<unsigned int> &
    getAtomWiseNumberCellsAccumulated();

  private:
    // Pointer of pointers for BatchedGEMM call in applyCTonX()
    ValueType **hostPointerCDagger, **hostPointerCDaggeOutTemp,
      **hostWfcPointers;
    ValueType **devicePointerCDagger, **devicePointerCDaggerOutTemp,
      **deviceWfcPointers;
    std::vector<unsigned int> d_nonlocalElemIdToLocalElemIdMap;

    // Data structures moved from KSOperatorDevice
    std::vector<ValueType> d_cellHamiltonianMatrixNonLocalFlattenedConjugate;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                           d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice;
    std::vector<ValueType> d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_cellHamMatrixTimesWaveMatrixNonLocalDevice;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_sphericalFnTimesVectorAllCellsDevice;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                        d_sphericalFnTimesVectorDevice;
    std::vector<double> d_nonLocalPseudoPotentialConstants;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_nonLocalPseudoPotentialConstantsDevice;

    std::vector<ValueType> d_sphericalFnTimesVectorAllCellsReduction;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_sphericalFnTimesVectorAllCellsReductionDevice;
    std::vector<dftfe::global_size_type>
      d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal;
    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::DEVICE>
                              d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice;
    std::vector<unsigned int> d_sphericalFnIdsParallelNumberingMap;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
                     d_sphericalFnIdsParallelNumberingMapDevice;
    std::vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE>
                              d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice;
    std::vector<unsigned int> d_cellNodeIdMapNonLocalToLocal;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_cellNodeIdMapNonLocalToLocalDevice;
  };



} // namespace dftfe
#include "../src/atom/AtomicCenteredNonLocalOperator.t.cc"
#include "../src/atom/AtomicCenteredNonLocalOperatorHost.t.cc"
#if defined(DFTFE_WITH_DEVICE)
#  include "../src/atom/AtomicCenteredNonLocalOperatorDevice.t.cc"
#endif

#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
