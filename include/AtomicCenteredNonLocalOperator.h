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
#include <cmath>

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
  class AtomicCenteredNonLocalOperator
  {
  public:
    AtomicCenteredNonLocalOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        basisOperatorPtr,
      std::shared_ptr<AtomCenteredSphericalFunctionContainer>
                      atomCenteredSphericalFunctionContainer,
      const MPI_Comm &mpi_comm_parent);



    void
    initialiseOperatorActionOnX(unsigned int kPointIndex);

    void
    initialiseFlattenedDataStructure(
      unsigned int waveFunctionBlockSize,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened);

    void
    intitialisePartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr,
      const unsigned int quadratureIndex);
#if defined(DFTFE_WITH_DEVICE)
    // for device specific initialise
    void
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix);

    void
    freeDeviceVectors();
#endif
    // Getter functions

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
    getGlobalDofAtomIdSphericalFnPair(const unsigned int atomId,
                                      const unsigned int alpha);

    unsigned int
    getLocalIdOfDistributedVec(const unsigned int globalId);

    std::vector<unsigned int> &
    getNonLocalElemIdToLocalElemIdMap();

    std::vector<unsigned int> &
    getAtomWiseNumberCellsInCompactSupport();

    std::vector<unsigned int> &
    getAtomWiseNumberCellsAccumulated();

    const std::vector<ValueType> &
    getAtomCenteredKpointIndexedSphericalFnQuadValues();

    const std::vector<ValueType> &
    getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues();

    const std::map<unsigned int, std::vector<unsigned int>> &
    getCellIdToAtomIdsLocalCompactSupportMap();

    const std::vector<unsigned int> &
    getNonTrivialSphericalFnsPerCell();

    const std::vector<unsigned int> &
    getNonTrivialSphericalFnsCellStartIndex();

    const unsigned int
    getTotalNonTrivialSphericalFnsOverAllCells();

    const std::vector<unsigned int> &
    getNonTrivialAllCellsSphericalFnAlphaToElemIdMap();

    const std::map<unsigned int, std::vector<unsigned int>> &
    getAtomIdToNonTrivialSphericalFnCellStartIndex();

    const std::vector<unsigned int> &
    getSphericalFnTimesVectorFlattenedVectorLocalIds();


    // Calls for both device and host
    void
    applyV_onCconjtransX(
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagCopyResultsToMatrix = true);

    void
    applyAllReduceonCTX(
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool skip1 = false,
      const bool skip2 = false);


    void
    applyCconjtrans_onX(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &X,
      const std::pair<unsigned int, unsigned int>                cellRange);


    void
    applyVCconjtransOnX(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      const unsigned int                                         kPointIndex,
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened);


    void
    applyC_VCconjtransX(
      dftfe::utils::MemoryStorage<ValueType, memorySpace> &Xout,
      const std::pair<unsigned int, unsigned int>          cellRange);


  protected:
    bool                d_AllReduceCompleted;
    std::vector<double> d_kPointWeights;
    std::vector<double> d_kPointCoordinates;
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomCenteredSphericalFunctionContainer;
    std::shared_ptr<
      const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
                              d_mpiPatternP2P;
    std::vector<unsigned int> d_numberCellsForEachAtom;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
      d_basisOperatorPtr;


    // Required by force.cc
    std::vector<ValueType> d_atomCenteredKpointIndexedSphericalFnQuadValues;
    // Required for stress compute
    std::vector<ValueType>
      d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues;

    /// map from cell number to set of non local atom ids (local numbering)
    std::map<unsigned int, std::vector<unsigned int>>
      d_cellIdToAtomIdsLocalCompactSupportMap;

    /// vector of size num physical cells
    std::vector<unsigned int> d_nonTrivialSphericalFnPerCell;

    /// vector of size num physical cell with starting index for each cell for
    /// the above array
    std::vector<unsigned int> d_nonTrivialSphericalFnsCellStartIndex;

    std::vector<unsigned int> d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap;

    /// map from local nonlocal atomid to vector over cells
    std::map<unsigned int, std::vector<unsigned int>>
      d_atomIdToNonTrivialSphericalFnCellStartIndex;

    unsigned int d_sumNonTrivialSphericalFnOverAllCells;

    std::vector<unsigned int> d_sphericalFnTimesVectorFlattenedVectorLocalIds;

    // The above set of variables are needed in force class

#ifdef USE_COMPLEX
    std::vector<distributedCPUVec<std::complex<double>>>
      d_SphericalFunctionKetTimesVectorPar;

#else
    std::vector<distributedCPUVec<double>> d_SphericalFunctionKetTimesVectorPar;
#endif

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_sphericalFunctionIdsNumberingMapCurrentProcess;


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
    // Host CMatrix Entries are stored here
    std::vector<std::vector<std::vector<ValueType>>> d_CMatrixEntriesConjugate,
      d_CMatrixEntriesTranspose;

  private:
    void
    initKpoints(const std::vector<double> &kPointWeights,
                const std::vector<double> &kPointCoordinates);
    void
    initalisePartitioner(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr);

    void
    computeCMatrixEntries(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr,
      const unsigned int quadratureIndex);

    std::map<
      unsigned int,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
      d_sphericalFnTimesWavefunMatrix;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_sphericalFnTimesWavefunctionMatrix;
#if defined(DFTFE_WITH_DEVICE)
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
#endif
  };



} // namespace dftfe
#include "../src/atom/AtomicCenteredNonLocalOperator.t.cc"

#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
