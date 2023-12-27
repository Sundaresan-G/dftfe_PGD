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
} // namespace dftfe
