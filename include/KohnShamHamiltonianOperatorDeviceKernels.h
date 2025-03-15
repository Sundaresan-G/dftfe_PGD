// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#ifndef kohnShamHamiltonianOperatorDeviceKernels_H_
#define kohnShamHamiltonianOperatorDeviceKernels_H_

#include <BLASWrapper.h>
#include <DataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryStorage.h>
namespace dftfe
{
  namespace internal
  {
    template <dftfe::utils::MemorySpace memorySpace>
    void
    computeCellHamiltonianMatrixNonCollinearFromBlocks(
      const std::pair<unsigned int, unsigned int> cellRange,
      const unsigned int                          nDofsPerCell,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &tempHamMatrixRealBlock,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &tempHamMatrixImagBlock,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &tempHamMatrixBZBlockNonCollin,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &tempHamMatrixBYBlockNonCollin,
      const dftfe::utils::MemoryStorage<double, memorySpace>
        &tempHamMatrixBXBlockNonCollin,
      dftfe::utils::MemoryStorage<std::complex<double>, memorySpace>
        &cellHamiltonianMatrix);
  };
} // namespace dftfe
#endif
