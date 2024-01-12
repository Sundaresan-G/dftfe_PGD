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


    virtual void
    initialiseOperatorActionOnX(unsigned int kPointIndex) = 0;

    virtual void
    initialiseFlattenedDataStructure(unsigned int numberWaveFunctions) = 0;

    unsigned int
    getTotalAtomInCurrentProcessor();

    unsigned int
    getTotalNonLocalElementsInCurrentProcessor();

    unsigned int
    getTotalNonLocalEntries();

    unsigned int
    getMaxSingleAtomEntries();
#ifdef USE_COMPLEX
    std::vector<distributedCPUVec<std::complex<double>>>
      d_SphericalFunctionKetTimesVectorPar;

#else
    std::vector<distributedCPUVec<double>> d_SphericalFunctionKetTimesVectorPar;
#endif

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_sphericalFunctionIdsNumberingMapCurrentProcess;

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
    initialiseOperatorActionOnX(unsigned int kPointIndex) override;

    void
    initialiseFlattenedDataStructure(unsigned int numberWaveFunctions) override;

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
      CouplingStructure couplingtype,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix);

    void
    applyCTonX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X,
      const std::pair<unsigned int, unsigned int> cellRange);

    void
    applyCTonX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X);


    void
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange);

    void
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &Xout);



    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class dst
    // and src have the same size
    void
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class This is
    // same as the previous function but does it for cell wise data The
    // cellRange is to determine what cells the operation is carried out on dst
    // and src have the same size
    void
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class src and
    // dst will have different sizes
    void
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class This is
    // same as the previous function but does it for cell wise data The
    // cellRange is to determine what cells the operation is carried out on src
    // and dst will have different sizes
    void
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange);

    const std::map<unsigned int, std::vector<ValueType>> &
    getScaledShapeFnTimesWaveFunction();

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
    initialiseOperatorActionOnX(unsigned int kPointIndex) override;

    void
    initialiseFlattenedDataStructure(unsigned int numberWaveFunctions) override;

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
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &X,
      std::pair<unsigned int, unsigned int> &cellRange);


    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class dst
    // and src have the same size
    void
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &dst);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class This is
    // same as the previous function but does it for cell wise data The
    // cellRange is to determine what cells the operation is carried out on dst
    // and src have the same size
    void
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class src and
    // dst will have different sizes
    void
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &dst);

    // Assumes that constraints.distribute is called on src and
    // update_ghost_Values() on src src is not changed inside this class This is
    // same as the previous function but does it for cell wise data The
    // cellRange is to determine what cells the operation is carried out on src
    // and dst will have different sizes
    void
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::DEVICE>
        &                                                                   src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange);

  private:
    // Data structures moved from KSOperatorDevice
    std::vector<dataTypes::number>
      d_cellHamiltonianMatrixNonLocalFlattenedConjugate;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice;
    std::vector<dataTypes::number>
      d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamMatrixTimesWaveMatrixNonLocalDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_projectorKetTimesVectorParFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_projectorKetTimesVectorAllCellsDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
                        d_projectorKetTimesVectorDevice;
    std::vector<double> d_nonLocalPseudoPotentialConstants;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_nonLocalPseudoPotentialConstantsDevice;

    std::vector<dataTypes::number> d_projectorKetTimesVectorAllCellsReduction;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_projectorKetTimesVectorAllCellsReductionDevice;
    std::vector<dftfe::global_size_type>
      d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal;
    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::DEVICE>
                              d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice;
    std::vector<unsigned int> d_shapeFnIdsParallelNumberingMap;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
                     d_shapeFnIdsParallelNumberingMapDevice;
    std::vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE>
                              d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice;
    std::vector<unsigned int> d_cellNodeIdMapNonLocalToLocal;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_cellNodeIdMapNonLocalToLocalDevice;

    std::vector<unsigned int> d_nonlocalElemIdToLocalElemIdMap;
  };



} // namespace dftfe
#include "../src/atom/AtomicCenteredNonLocalOperator.t.cc"
#include "../src/atom/AtomicCenteredNonLocalOperatorHost.t.cc"
#if defined(DFTFE_WITH_DEVICE)
#  include "../src/atom/AtomicCenteredNonLocalOperatorDevice.t.cc"
#endif

#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
