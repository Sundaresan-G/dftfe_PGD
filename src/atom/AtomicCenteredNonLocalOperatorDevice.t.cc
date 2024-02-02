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
    applyVCconjtransOnX(
      const ValueType *       src,
      const unsigned int      kPointIndex,
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &couplingMatrix,
      distributedDeviceVec<ValueType>
        &sphericalFunctionKetTimesVectorParFlattened)
  {}



  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyCconjtrans_onX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &X,
      std::pair<unsigned int, unsigned int> cellRange)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyC_VCconjtransX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    applyAllReduceonCTX(distributedDeviceVec<ValueType>
                          &        sphericalFunctionKetTimesVectorParFlattened,
                        const bool skip1,
                        const bool skip2)
  {}

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
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    transferCMatrixEntriesfromHostObject(
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::HOST>>
        nonLocalOperatorHost)
  {}


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseOperatorActionOnX(unsigned int kPointIndex)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFunctionKetTimesVectorParFlattened)
  {}
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix)
  {}



  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    initialiseDeviceVectors()
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::DEVICE>::
    freeDeviceVectors()
  {}



} // namespace dftfe
