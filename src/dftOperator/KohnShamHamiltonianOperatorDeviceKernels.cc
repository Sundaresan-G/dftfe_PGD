#include <KohnShamHamiltonianOperatorDeviceKernels.h>
namespace dftfe
{
  namespace
  {
    __global__ void
    computeCellHamiltonianMatrixNonCollinearFromBlocksDeviceKernel(
      const unsigned int                 numCells,
      const unsigned int                 nDofsPerCell,
      const unsigned int                 cellStartIndex,
      const double *                     tempHamMatrixRealBlock,
      const double *                     tempHamMatrixImagBlock,
      const double *                     tempHamMatrixBZBlockNonCollin,
      const double *                     tempHamMatrixBYBlockNonCollin,
      const double *                     tempHamMatrixBXBlockNonCollin,
      dftfe::utils::deviceDoubleComplex *cellHamiltonianMatrix)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numCells * nDofsPerCell * nDofsPerCell;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int jDoF   = index % nDofsPerCell;
          const unsigned int iBlock = index / nDofsPerCell;
          const unsigned int iDoF   = iBlock % nDofsPerCell;
          const unsigned int iCell  = cellStartIndex + iBlock / nDofsPerCell;
          const unsigned int iCellBlock = iBlock / nDofsPerCell;
          const double       H_realIJ =
            tempHamMatrixRealBlock[jDoF + nDofsPerCell * iDoF +
                                   iCellBlock * nDofsPerCell * nDofsPerCell];
          const double H_imagIJ =
            tempHamMatrixImagBlock[jDoF + nDofsPerCell * iDoF +
                                   iCellBlock * nDofsPerCell * nDofsPerCell];
          const double H_bzIJ =
            tempHamMatrixBZBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                          iCellBlock * nDofsPerCell *
                                            nDofsPerCell];
          const double H_byIJ =
            tempHamMatrixBYBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                          iCellBlock * nDofsPerCell *
                                            nDofsPerCell];
          const double H_bxIJ =
            tempHamMatrixBXBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                          iCellBlock * nDofsPerCell *
                                            nDofsPerCell];
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF + 1) + 2 * jDoF +
                                1]
            .x = H_realIJ - H_bzIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF + 1) + 2 * jDoF +
                                1]
            .y = H_imagIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF]
            .x = H_realIJ + H_bzIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF]
            .y = H_imagIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF + 1) + 2 * jDoF]
            .x = H_bxIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF + 1) + 2 * jDoF]
            .y = H_byIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF + 1]
            .x = H_bxIJ;
          cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF + 1]
            .y = -H_byIJ;
        }
    }
  } // namespace
  namespace internal
  {
    template <>
    void
    computeCellHamiltonianMatrixNonCollinearFromBlocks(
      const std::pair<unsigned int, unsigned int> cellRange,
      const unsigned int                          nDofsPerCell,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixRealBlock,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixImagBlock,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBZBlockNonCollin,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBYBlockNonCollin,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBXBlockNonCollin,
      dftfe::utils::MemoryStorage<std::complex<double>,
                                  dftfe::utils::MemorySpace::DEVICE>
        &cellHamiltonianMatrix)
    {
      const unsigned int nCells = cellRange.second - cellRange.first;
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeCellHamiltonianMatrixNonCollinearFromBlocksDeviceKernel<<<
        (nCells * nDofsPerCell * nDofsPerCell) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        nCells,
        nDofsPerCell,
        cellRange.first,
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixRealBlock.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixImagBlock.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBZBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBYBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBXBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          cellHamiltonianMatrix.data()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        computeCellHamiltonianMatrixNonCollinearFromBlocksDeviceKernel,
        (nCells * nDofsPerCell * nDofsPerCell) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        nCells,
        nDofsPerCell,
        cellRange.first,
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixRealBlock.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixImagBlock.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBZBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBYBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBXBlockNonCollin.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          cellHamiltonianMatrix.data()));
#endif
    }
  }; // namespace internal
} // namespace dftfe
