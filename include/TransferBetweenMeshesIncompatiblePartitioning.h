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

#ifndef DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H
#define DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H


#include "InterpolateCellWiseDataToPoints.h"
//#include "TransferBetweenMeshesBase.h"
#include "headers.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "vectorUtilities.h"


namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class TransferDataBetweenMeshesIncompatiblePartitioning // : public TransferDataBetweenMeshesBase<memorySpace>
  {
  public:
    TransferDataBetweenMeshesIncompatiblePartitioning(const dealii::MatrixFree<3, double> &matrixFreeMesh1,
                                                      const unsigned int                   matrixFreeMesh1VectorComponent,
                                                      const unsigned int matrixFreeMesh1QuadratureComponent,
                                                      const dealii::MatrixFree<3, double> &matrixFreeMesh2,
                                                      const unsigned int                   matrixFreeMesh2VectorComponent,
                                                      const unsigned int matrixFreeMesh2QuadratureComponent,
                                                      const MPI_Comm & mpiComm);

    template <typename T>
    void
    interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<T, memorySpace> &outputQuadData,
      bool resizeOutputVec); // override;

    template <typename T>
      void
      interpolateMesh2DataToMesh1QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
        const dftfe::linearAlgebra::MultiVector<T,
                                                memorySpace> &inputVec,
        const unsigned int                    numberOfVectors,
        const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<T, memorySpace> &                 outputQuadData,
        bool resizeOutputVec); // override;

    template <typename T>
    void
    interpolateMesh1DataToMesh2QuadPoints(
      const distributedCPUVec<T> &inputVec,
      const unsigned int numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                        dftfe::utils::MemorySpace::HOST> &fullFlattenedArrayCellLocalProcIndexIdMapParent,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &outputQuadData,
      bool resizeOutputVec) ; //override;

    template <typename T>
    void
    interpolateMesh2DataToMesh1QuadPoints(
      const distributedCPUVec<T> &inputVec,
      const unsigned int               numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &            outputQuadData,
      bool resizeOutputVec) ; //override;

  private:

    const dealii::MatrixFree<3, double> * d_matrixFreeMesh1Ptr;
    const dealii::MatrixFree<3, double> * d_matrixFreeMesh2Ptr;

    size_type d_matrixFreeMesh1VectorComponent,d_matrixFreeMesh1QuadratureComponent;

    size_type d_matrixFreeMesh2VectorComponent, d_matrixFreeMesh2QuadratureComponent;

    std::shared_ptr<InterpolateCellWiseDataToPoints<memorySpace>> d_mesh1toMesh2;
    std::shared_ptr<InterpolateCellWiseDataToPoints<memorySpace>> d_mesh2toMesh1;

    const MPI_Comm d_mpiComm;


  };
}
#endif // DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H
