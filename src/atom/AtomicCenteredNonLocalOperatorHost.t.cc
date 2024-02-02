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
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    computeCMatrixEntries(const unsigned int quadratureIndex)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyAllReduceonCTX(
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyCconjtrans_onX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X,
      const std::pair<unsigned int, unsigned int> cellRange)
  {}


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyV_onCconjtransX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagCopyResultsToMatrix)
  {}


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyC_VCconjtransX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange)
  {}
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    initialiseOperatorActionOnX(unsigned int kPointIndex)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyVCconjtransOnX(
      const distributedCPUMultiVec<ValueType> &src,
      const unsigned int                       kPointIndex,
      const CouplingStructure                  couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {}

  template <typename ValueType>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomCenteredKpointIndexedSphericalFnQuadValues()
  {}
  template <typename ValueType>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues()
  {}

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getSphericalFnTimesVectorFlattenedVectorLocalIds()

  {}

  template <typename ValueType>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomIdToNonTrivialSphericalFnCellStartIndex()
  {}

  template <typename ValueType>
  const unsigned int
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getTotalNonTrivialSphericalFnsOverAllCells()
  {}

  template <typename ValueType>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getCellIdToAtomIdsLocalCompactSupportMap()
  {}

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialSphericalFnsPerCell()
  {}

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialSphericalFnsCellStartIndex()
  {}

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialAllCellsSphericalFnAlphaToElemIdMap()
  {}



} // namespace dftfe
