//
// Created by VISHAL SUBRAMANIAN on 4/30/24.
//

#include "MultiVectorAdjointLinearSolverProblem.h"

#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceBlasWrapper.h>

namespace dftfe
{

  namespace
  {

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType1, typename ValueType2>
    __global__ void
    rMatrixDeviceKernel(const dftfe::size_type numLocalCells,
                        const dftfe::size_type numDofsPerElem,
                        const dftfe::size_type numQuadPoints,
                        const ValueType1      *shapeFunc,
                        const ValueType1      *shapeFuncTranspose,
                        const ValueType2      *inputJxW,
                        ValueType2            *rMatrix)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numLocalCells * numDofsPerElem * numDofsPerElem;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type iElem = index / (numDofsPerElem * numDofsPerElem);
          dftfe::size_type nodeIndex =
            index % (numDofsPerElem * numDofsPerElem);
          dftfe::size_type iNode = nodeIndex / (numDofsPerElem);
          dftfe::size_type jNode = nodeIndex % (numDofsPerElem);

          dftfe::size_type elemRIndex = iElem * numDofsPerElem * numDofsPerElem;
          dftfe::size_type nodeRIndex = iNode * numDofsPerElem + jNode;

          dftfe::size_type iNodeQuadIndex = iNode * numQuadPoints;
          dftfe::size_type jNodeQuadIndex = jNode * numQuadPoints;

          dftfe::size_type elemQuadIndex = iElem * numQuadPoints;
          for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++)
            {
              dftfe::utils::copyValue(
                rMatrix + elemRIndex + nodeRIndex,
                dftfe::utils::add(
                  rMatrix[elemRIndex + nodeRIndex],
                  dftfe::utils::mult(
                    shapeFuncTranspose[iNodeQuadIndex + iQuad],
                    dftfe::utils::mult(
                      shapeFuncTranspose[jNodeQuadIndex + iQuad],
                      inputJxW[elemQuadIndex + iQuad]))));
            }
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    rMatrixMemSpaceKernel(
      const dftfe::size_type numLocalCells,
      const dftfe::size_type numDofsPerElem,
      const dftfe::size_type numQuadPoints,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      const dftfe::utils::MemoryStorage < ValueType1,
                                        dftfe::utils::MemorySpace::DEVICE> &shapeFunc,
      const dftfe::utils::MemoryStorage<ValueType2,
                                        dftfe::utils::MemorySpace::DEVICE>
        &shapeFuncTranspose,
      const dftfe::utils::MemoryStorage<ValueType1,
                                        dftfe::utils::MemorySpace::DEVICE>
                                                                     &inputJxW,
      dftfe::utils::MemoryStorage<ValueType1,
                                  dftfe::utils::MemorySpace::DEVICE> &rMatrix)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      rMatrixDeviceKernel<<<(numLocalCells * numDofsPerElem * numDofsPerElem) /
                                dftfe::utils::DEVICE_BLOCK_SIZE +
                              1,
                            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numLocalCells,
        numDofsPerElem,
        numQuadPoints,
        dftfe::utils::makeDataTypeDeviceCompatible(shapeFunc.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(shapeFuncTranspose.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(rMatrix.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        rMatrixDeviceKernel,
        (numLocalCells * numDofsPerElem * numDofsPerElem) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numLocalCells,
        numDofsPerElem,
        numQuadPoints,
        dftfe::utils::makeDataTypeDeviceCompatible(shapeFunc.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(shapeFuncTranspose.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(rMatrix.begin()));
#endif
    }

    template <typename ValueType>
    __global__ void
    muMatrixDeviceKernel(const dftfe::size_type numLocalCells,
                         const dftfe::size_type numVec,
                         const dftfe::size_type numQuadPoints,
                         const dftfe::size_type blockSize,
                         const ValueType *     orbitalOccupancy,
                         const unsigned int *     vecList,
                         const ValueType *     cellLevelQuadValues,
                         const ValueType *     inputJxW,
                         ValueType *           muMatrixCellWise)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries = numLocalCells * numVec;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type iElem        = index / (numVec);
          dftfe::size_type vecIndex     = index % numVec;
          dftfe::size_type vecId        = vecList[2 * vecIndex];
          dftfe::size_type degenerateId = vecList[2 * vecIndex + 1];

          dftfe::size_type elemQuadIndex = iElem * numQuadPoints * blockSize;
          dftfe::size_type elemInputQuadIndex = iElem * numQuadPoints;
          for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++)
            {
              dftfe::utils::copyValue(
                muMatrixCellWise + iElem * numVec + vecIndex,
                dftfe::utils::add(
                  muMatrixCellWise[iElem * numVec + vecIndex],
                  dftfe::utils::mult(
                    dftfe::utils::mult(2.0, orbitalOccupancy[vecId]),
                    dftfe::utils::mult(
                      cellLevelQuadValues[elemQuadIndex + vecId +
                                          iQuad * blockSize],
                      dftfe::utils::mult(
                        cellLevelQuadValues[elemQuadIndex + degenerateId +
                                            iQuad * blockSize],
                        inputJxW[elemInputQuadIndex + iQuad])))));
            }
        }
    }

    template <typename ValueType>
    void
    muMatrixMemSpaceKernel(
      const dftfe::size_type numLocalCells,
      const dftfe::size_type numVec,
      const dftfe::size_type numQuadPoints,
      const dftfe::size_type blockSize,
       const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      const dftfe::utils::MemoryStorage < ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &orbitalOccupancy,
      const dftfe::utils::MemoryStorage < unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE> & vecList,
      const dftfe::utils::MemoryStorage < ValueType,
                                        dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValues,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
                                                                     &inputJxW,
      dftfe::utils::MemoryStorage<ValueType,
                                  dftfe::utils::MemorySpace::DEVICE> &muMatrixCellWise)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      muMatrixDeviceKernel<<<
        (numLocalCells * numVec) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numLocalCells,
        numVec,
        numQuadPoints,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancy.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(vecList.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(muMatrixCellWise.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        muMatrixDeviceKernel,
        (numLocalCells * numVec) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numLocalCells,
        numVec,
        numQuadPoints,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancy.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(vecList.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(muMatrixCellWise.begin()));
#endif
    }
#endif


    template <typename ValueType1, typename ValueType2>
    void
    rMatrixMemSpaceKernel(
      const dftfe::size_type numLocalCells,
      const dftfe::size_type numDofsPerElem,
      const dftfe::size_type numQuadPoints,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      const dftfe::utils::MemoryStorage < ValueType1,
                                        dftfe::utils::MemorySpace::HOST> &shapeFunc,
      const dftfe::utils::MemoryStorage < ValueType1,
      dftfe::utils::MemorySpace::HOST> &shapeFuncTranspose,
      const dftfe::utils::MemoryStorage<ValueType2,
                                        dftfe::utils::MemorySpace::HOST>
                                                                   &inputJxW,
      dftfe::utils::MemoryStorage<ValueType2,
                                  dftfe::utils::MemorySpace::HOST> &rMatrix)
    {
      AssertThrow(inputJxW.size() == (numQuadPoints * numLocalCells),
                  dealii::ExcMessage(
                    "In inputJxW the inputJxW should have only one component"
                    "u(r) = w(r)*(rho_target(r) - rho_KS(r))"));

      std::fill(rMatrix.begin(), rMatrix.end(), 0.0);
      const unsigned int inc = 1;

      std::vector<double> cellLevelJxW, cellLevelShapeFunction,
        cellLevelRhsInput;
      cellLevelJxW.resize(numQuadPoints);

      std::vector<double> shapeFuncIJ(numDofsPerElem * numQuadPoints, 0.0);
      std::vector<double> cellLevelR(numDofsPerElem * numLocalCells,
                                     0.0);


      double beta = 0.0, alpha = 1.0;
      char   transA = 'N', transB = 'N';
      for (unsigned int iNode = 0; iNode < numDofsPerElem; iNode++)
        {
          for (unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++)
            {
              for (unsigned int jNode = 0; jNode < numDofsPerElem; jNode++)
                {
                  shapeFuncIJ[iQuad * numDofsPerElem + jNode] =
                    shapeFuncTranspose[iNode * numQuadPoints +
                                                   iQuad] *
                    shapeFunc[jNode + iQuad * numDofsPerElem];
                }
            }
	  //std::cout<<" shapeFuncTranspose = "<<shapeFuncTranspose.size()<<"\n";
	  //std::cout<<" shapeFunc = "<<shapeFunc.size()<<"\n";

	  if (numLocalCells ==0)
	  {
		   std::cout<<" Error in numLocalCells is zero !!!!!!\n";
	  }
	  if(inputJxW.size() != numQuadPoints*numLocalCells)
	  {
		  std::cout<<" inputJxW error in compute r mat\n"; 
	  }
          BLASWrapperPtr->xgemm(transA,
                                transB,
                                numDofsPerElem,
                                numLocalCells,
                                numQuadPoints,
                                &alpha,
                                &shapeFuncIJ[0],
                                numDofsPerElem,
                                &inputJxW[0],
                                numQuadPoints,
                                &beta,
                                &cellLevelR[0],
                                numDofsPerElem);
          for (unsigned int elemId = 0; elemId < numLocalCells; elemId++)
            {
              dcopy_(&numDofsPerElem,
                     &cellLevelR[elemId * numDofsPerElem],
                     &inc,
                     &rMatrix[elemId * numDofsPerElem * numDofsPerElem +
                              iNode * numDofsPerElem],
                     &inc);
            }
        }
    }

    template <typename ValueType>
    void
    muMatrixMemSpaceKernel(
      const dftfe::size_type numLocalCells,
      const dftfe::size_type numVec,
      const dftfe::size_type numQuadPoints,
      const dftfe::size_type blockSize,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      const dftfe::utils::MemoryStorage < ValueType,
                                        dftfe::utils::MemorySpace::HOST> &orbitalOccupancy,
      const dftfe::utils::MemoryStorage < unsigned int,
                                        dftfe::utils::MemorySpace::HOST> & vecList,
      const dftfe::utils::MemoryStorage < ValueType,
      dftfe::utils::MemorySpace::HOST> &cellLevelQuadValues,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
                                                                   &inputJxW,
      dftfe::utils::MemoryStorage<ValueType,
                                  dftfe::utils::MemorySpace::HOST> &muMatrixCellWise)
    {
      for( unsigned int index = 0 ; index < numLocalCells * numVec; index++ )
        {
          dftfe::size_type iElem        = index / (numVec);
          dftfe::size_type vecIndex     = index % numVec;
          dftfe::size_type vecId        = vecList[2 * vecIndex];
          dftfe::size_type degenerateId = vecList[2 * vecIndex + 1];

          dftfe::size_type elemQuadIndex = iElem * numQuadPoints * blockSize;
          dftfe::size_type elemInputQuadIndex = iElem * numQuadPoints;
          for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++)
            {
              muMatrixCellWise[iElem * numVec + vecIndex] +=   2.0*orbitalOccupancy[vecId]*
                                                                cellLevelQuadValues[elemQuadIndex + vecId +
                                                                                    iQuad * blockSize]*
							        cellLevelQuadValues[elemQuadIndex + degenerateId +
                                                                                    iQuad * blockSize]*
                                                                inputJxW[elemInputQuadIndex + iQuad];

            }
        }
    }
  }


  // constructor
  template <dftfe::utils::MemorySpace memorySpace>
  MultiVectorAdjointLinearSolverProblem<memorySpace>::MultiVectorAdjointLinearSolverProblem(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr                      = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
    d_cellBlockSize = 100; // TODO set this based on rum time.
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::reinit(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      basisOperationsPtr,
    KohnShamHamiltonianOperator<
      memorySpace> & ksHamiltonianObj,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhs,
    const bool              isComputeDiagonalA)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);

    d_BLASWrapperPtr = BLASWrapperPtr;
    d_basisOperationsPtr        = basisOperationsPtr;
    d_matrixFreeDataPtr         = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPtr       = &constraintMatrix;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentRhs =
      matrixFreeQuadratureComponentRhs;

    d_numCells       = d_basisOperationsPtr->nCells();

    d_cellBlockSize = std::min(d_cellBlockSize ,d_numCells);

    d_basisOperationsPtr->reinit(1,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 true); // TODO should this be set to true

    d_locallyOwnedSize = d_basisOperationsPtr->nOwnedDofs();
    d_numberDofsPerElement = d_basisOperationsPtr->nDofsPerCell();

    d_numQuadsPerCell = d_basisOperationsPtr->nQuadsPerCell();

    d_ksOperatorPtr = &ksHamiltonianObj;



    //std::cout<<" local size in adjoint = "<<d_locallyOwnedSize<<"\n";

    if (isComputeDiagonalA)
      {
        computeDiagonalA();
        d_isComputeDiagonalA = true;
      }

    d_constraintsInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(
        matrixFreeVectorComponent),
      constraintMatrix);


    d_onesMemSpace.resize(d_locallyOwnedSize);
    d_onesMemSpace.setValue(1.0);

    d_onesQuadMemSpace.resize(d_numCells*d_numQuadsPerCell);
    d_onesQuadMemSpace.setValue(1.0);

  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
    MultiVectorAdjointLinearSolverProblem<memorySpace>::computeDiagonalA()
  {
    d_basisOperationsPtr->computeStiffnessVector(true, true);
    d_basisOperationsPtr->computeInverseSqrtMassVector();

    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
      nodeIds;
    nodeIds.resize(d_locallyOwnedSize);
    for(size_type i = 0 ; i < d_locallyOwnedSize;i++)
      {
        nodeIds.data()[i] = i;
      }

    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> mapNodeIdToProcId;
    mapNodeIdToProcId.resize(d_locallyOwnedSize);
    mapNodeIdToProcId.copyFrom(nodeIds);

    auto sqrtMassMat = d_basisOperationsPtr->sqrtMassVectorBasisData();
    auto inverseStiffVec = d_basisOperationsPtr->inverseStiffnessVectorBasisData();
    auto inverseSqrtStiffVec = d_basisOperationsPtr->inverseSqrtStiffnessVectorBasisData();

    d_basisOperationsPtr->createMultiVector(1,d_diagonalA);
    d_diagonalA.setValue(1.0);
    d_BLASWrapperPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0/0.5,
      inverseStiffVec.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_basisOperationsPtr->createMultiVector(1,d_diagonalSqrtA);
    d_diagonalSqrtA.setValue(1.0);
    d_BLASWrapperPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      std::sqrt(1.0/0.5),
      inverseSqrtStiffVec.data(),
      d_diagonalSqrtA.data(),
      d_diagonalSqrtA.data(),
      mapNodeIdToProcId.data());

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalSqrtA.data(),
      d_diagonalSqrtA.data(),
      mapNodeIdToProcId.data());

    std::vector<double> d_diagonalANorm, d_diagonalSqrtANorm;
    d_diagonalANorm.resize(1);
    d_diagonalSqrtANorm.resize(1);
    //d_diagonalA.l2Norm(&d_diagonalANorm[0]);
    //d_diagonalSqrtA.l2Norm(&d_diagonalSqrtANorm[0]);
    pcout<<" Norm of d_diagonalA = "<<d_diagonalANorm[0]<<"\n";
    pcout<<" Norm of d_diagonalSqrtA = "<<d_diagonalSqrtANorm[0]<<"\n";
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
    MultiVectorAdjointLinearSolverProblem<memorySpace>::precondition_Jacobi(
    dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                      memorySpace> &      dst,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                            memorySpace> &src,
    const double                     omega) const
  {/*
	          d_BLASWrapperPtr->axpby(d_locallyOwnedSize*d_blockSize,
                          1.0,
                          src.begin(),
                          0.0,
                          dst.begin());
			  */
		  
    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      d_diagonalA.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
    
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::
    precondition_JacobiSqrt(dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                             memorySpace> &      dst,
                          const dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                                  memorySpace> &src,
                          const double omega) const
  {
	  /*
	   d_BLASWrapperPtr->axpby(d_locallyOwnedSize*d_blockSize,
                          1.0,
                          src.begin(),
                          0.0,
                          dst.begin());
			  */
	   
    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      d_diagonalSqrtA.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
      
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                    memorySpace> &
    MultiVectorAdjointLinearSolverProblem<memorySpace>::computeRhs(
    dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                      memorySpace> &       NDBCVec,
    dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                      memorySpace> &       outputVec,
    unsigned int                      blockSizeInput)
  {
    dealii::TimerOutput computing_timer(mpi_communicator,
                                        pcout,
                                        dealii::TimerOutput::summary,
                                        dealii::TimerOutput::wall_times);

    d_basisOperationsPtr->reinit(blockSizeInput,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 true); // TODO should this be set to true

    if(d_blockSize != blockSizeInput)
      {
        d_blockSize = blockSizeInput;
        dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                    dftfe::utils::MemorySpace::HOST>
          nodeIds, quadIds;
        nodeIds.resize(d_locallyOwnedSize);
        for(size_type i = 0 ; i < d_locallyOwnedSize;i++)
          {
            nodeIds.data()[i] = i*d_blockSize;
          }
        d_mapNodeIdToProcId.resize(d_locallyOwnedSize);
        d_mapNodeIdToProcId.copyFrom(nodeIds);

        quadIds.resize(d_numCells*d_numQuadsPerCell);
        for(size_type i = 0 ; i < d_numCells*d_numQuadsPerCell;i++)
          {
            quadIds.data()[i] = i*d_blockSize;
          }
        d_mapQuadIdToProcId.resize(d_numCells*d_numQuadsPerCell);
        d_mapQuadIdToProcId.copyFrom(quadIds);

        d_basisOperationsPtr->createMultiVector(d_blockSize,d_rhsMemSpace);

        vec1QuadValues.resize(d_blockSize*d_numCells*d_numQuadsPerCell);
        vec2QuadValues.resize(d_blockSize*d_numCells*d_numQuadsPerCell);
        vecOutputQuadValues.resize(d_blockSize*d_numCells*d_numQuadsPerCell);

        tempOutputDotProdMemSpace.resize(d_blockSize);
        oneBlockSizeMemSpace.resize(d_blockSize);
        oneBlockSizeMemSpace.setValue(1.0);


        d_basisOperationsPtr->computeCellStiffnessMatrix(
          d_matrixFreeQuadratureComponentRhs, 1, true, false);
        d_basisOperationsPtr->computeCellMassMatrix(
          d_matrixFreeQuadratureComponentRhs, 1, true, false);

        d_basisOperationsPtr->computeInverseSqrtMassVector(true, false);

      }
    d_blockedXPtr = &outputVec;


    // psiTemp = M^{1/2} psi
    // TODO check if this works
    //    computing_timer.leave_subsection("Rhs init  MPI");
    //
    computing_timer.enter_subsection("Rhs init MemSpace MPI");
    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> psiTempMemSpace;
    psiTempMemSpace.reinit(*d_psiMemSpace);
    //psiTempMemSpace = *d_psiMemSpace;
    
        d_BLASWrapperPtr->axpby(d_locallyOwnedSize*d_blockSize,
                          1.0,
                          d_psiMemSpace->begin(),
                          0.0,
                          psiTempMemSpace.begin());
    psiTempMemSpace.updateGhostValues();
    d_constraintsInfo.distribute(psiTempMemSpace);


    std::vector<double> l2NormVec(d_blockSize,0.0);

    //psiTempMemSpace.l2Norm(&l2NormVec[0]);

    /*
    pcout<<" psiTempMemSpace = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
	    pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */

    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> psiTempMemSpace2;
    psiTempMemSpace2.reinit(*d_psiMemSpace);

    psiTempMemSpace2.setValue(0.0);


    computing_timer.leave_subsection("Rhs init MemSpace MPI");

    computing_timer.enter_subsection("M^(-1/2) MemSpace MPI");

    auto sqrtMassMat = d_basisOperationsPtr->sqrtMassVectorBasisData();

    //pcout<<" sqrtMassMat = "<<sqrtMassMat.size()<<"\n";
    //pcout<<" psiTempMemSpace = "<<psiTempMemSpace.locallyOwnedSize()<<"\n";
    //pcout<<" d_mapNodeIdToProcId = "<<d_mapNodeIdToProcId.size()<<"\n";

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      psiTempMemSpace.data(),
      psiTempMemSpace.data(),
      d_mapNodeIdToProcId.data());

    //psiTempMemSpace.l2Norm(&l2NormVec[0]);
/*
    pcout<<" psiTempMemSpace  sqrt mass mat = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    psiTempMemSpace.updateGhostValues();
    computing_timer.leave_subsection("M^(-1/2) MemSpace MPI");

    
    computing_timer.enter_subsection("computeR MemSpace MPI");
    computeRMatrix(d_inputJxWMemSpace);
    computing_timer.leave_subsection("computeR MemSpace MPI");
    computing_timer.enter_subsection("computeMu MemSpace MPI");
    computeMuMatrix(d_inputJxWMemSpace, *d_psiMemSpace);



    computing_timer.leave_subsection("computeMu MemSpace MPI");


    computing_timer.enter_subsection("Mu*Psi MemSpace MPI");
    //    rhs = 0.0;
    d_rhsMemSpace.setValue(0.0);
    // Calculating the rhs from the quad points
    // multiVectorInput is stored on the quad points

    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0, alpha_minus_two = -2.0,
                 alpha_minus_one = -1.0;

    // rhs = Psi*Mu. Since blas/lapack assume a column-major format whereas the
    // Psi is stored in a row major format, we do Mu^T*\Psi^T = Mu*\Psi^T
    // (because Mu is symmetric)

    d_BLASWrapperPtr->xgemm(
      'N',
      'N',
      d_blockSize,
      d_locallyOwnedSize,
      d_blockSize,
      &alpha_minus_two,
      d_MuMatrixMemSpace.data(),
      d_blockSize,
      psiTempMemSpace.data(),
      d_blockSize,
      &beta,
      d_rhsMemSpace.data(),
      d_blockSize);


        //d_rhsMemSpace.l2Norm(&l2NormVec[0]);

    /*
	    pcout<<" d_rhsMemSpace = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    computing_timer.leave_subsection("Mu*Psi MemSpace MPI");


    //
    // y = M^{-1/2} * R * M^{-1/2} * PsiTemp
    // 1. Do PsiTemp = M^{-1/2}*PsiTemp
    // 2. Do PsiTemp2 = R*PsiTemp
    // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
    //

    computing_timer.enter_subsection("psi*M(-1/2) MemSpace MPI");

    auto invSqrtMassMat = d_basisOperationsPtr->inverseSqrtMassVectorBasisData();
    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      invSqrtMassMat.data(),
      psiTempMemSpace.data(),
      psiTempMemSpace.data(),
      d_mapNodeIdToProcId.data());


            //psiTempMemSpace.l2Norm(&l2NormVec[0]);

    /*
    pcout<<" psiTempMemSpace invSqrt = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    psiTempMemSpace.updateGhostValues();
    d_constraintsInfo.distribute(psiTempMemSpace);
    computing_timer.leave_subsection("psi*M(-1/2) MemSpace MPI");

    computing_timer.enter_subsection("R times psi MemSpace MPI");

    // 2. Do PsiTemp2 = R*PsiTemp
    d_cellWaveFunctionMatrixMemSpace.setValue(0.0);
    std::pair<unsigned int, unsigned int> cellRange = std::make_pair(0,d_numCells);
    
        d_basisOperationsPtr->reinit(d_blockSize,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 false); // TODO should this be set to true
    d_basisOperationsPtr->extractToCellNodalDataKernel(psiTempMemSpace,
                                                       d_cellWaveFunctionMatrixMemSpace.data(),
                                                       cellRange);

    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
    const unsigned int strideB =
      d_numberDofsPerElement * d_numberDofsPerElement;
    const unsigned int strideC = d_numberDofsPerElement * d_blockSize;


    d_BLASWrapperPtr->xgemmStridedBatched(
      'N',
      'N',
      d_blockSize,
      d_numberDofsPerElement,
      d_numberDofsPerElement,
      &scalarCoeffAlpha,
      d_cellWaveFunctionMatrixMemSpace.begin(),
      d_blockSize,
      strideA,
      d_RMatrixMemSpace.begin(),
      d_numberDofsPerElement,
      strideB,
      &scalarCoeffBeta,
      d_cellRMatrixTimesWaveMatrixMemSpace.begin(),
      d_blockSize,
      strideC,
      d_numCells);

        d_basisOperationsPtr->reinit(d_blockSize,
                                 d_cellBlockSize,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 false); // TODO should this be set to true
					//
      d_basisOperationsPtr->accumulateFromCellNodalData(
      d_cellRMatrixTimesWaveMatrixMemSpace.begin(),
      psiTempMemSpace2);
    d_constraintsInfo.distribute_slave_to_master(psiTempMemSpace2);
    psiTempMemSpace2.accumulateAddLocallyOwned();


    //psiTempMemSpace2.l2Norm(&l2NormVec[0]);

    /*
    pcout<<" psiTempMemSpace2 r mat = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
    computing_timer.leave_subsection("R times psi MemSpace MPI");

    computing_timer.enter_subsection("psiTemp M^(-1/2) MemSpace MPI");


    psiTempMemSpace2.updateGhostValues();
    d_constraintsInfo.distribute(psiTempMemSpace2);

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      invSqrtMassMat.data(),
      psiTempMemSpace2.data(),
      psiTempMemSpace2.data(),
      d_mapNodeIdToProcId.data());

    psiTempMemSpace2.updateGhostValues();
    d_constraintsInfo.distribute(psiTempMemSpace2);

        //psiTempMemSpace2.l2Norm(&l2NormVec[0]);
/*
    pcout<<" psiTempMemSpace2 inv mat = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(d_blockSize,
                                                      d_locallyOwnedSize,
						      psiTempMemSpace2.data(),
                                                      d_4xeffectiveOrbitalOccupancyMemSpace.data(),
                                                      d_rhsMemSpace.data());
/*
    pcout<<" d_4xeffectiveOrbitalOccupancyMemSpace size = "<<d_4xeffectiveOrbitalOccupancyMemSpace.size()<<"\n";
    for( unsigned int iBlock = 0 ; iBlock < d_blockSize; iBlock++)
    {
pcout<<" iB = "<<iBlock<<" occ = "<<d_4xeffectiveOrbitalOccupancyMemSpace.data()[iBlock]<<"\n";
    }
    */
    //d_rhsMemSpace.l2Norm(&l2NormVec[0]);
    /*    
    pcout<<" d_rhsMemSpace before set = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    d_constraintsInfo.set_zero(d_rhsMemSpace);

    //d_rhsMemSpace.l2Norm(&l2NormVec[0]);

    /*
    pcout<<" d_rhsMemSpace final = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    computing_timer.leave_subsection("psiTemp M^(-1/2) MemSpace MPI");

    return d_rhsMemSpace;
  }


  // TODO PLease call d_kohnShamClassPtr->reinitkPointSpinIndex() before
  // calling this functions.

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::updateInputPsi(
    dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                      memorySpace> &psiInputVecMemSpace, // need to call distribute
    std::vector<double>
                                           &effectiveOrbitalOccupancy, // incorporates spin information
    dftfe::utils::MemoryStorage<double , memorySpace> &differenceInDensity,
    std::vector<std::vector<unsigned int>> &degeneracy,
    std::vector<double> &                   eigenValues,
    unsigned int                            blockSize)
  {
    pcout << " updating psi inside adjoint\n";

    d_psiMemSpace = &psiInputVecMemSpace;
    d_psiMemSpace->updateGhostValues();
    d_constraintsInfo.distribute(*d_psiMemSpace);

            std::vector<double> l2NormVec(blockSize,0.0);

    //d_psiMemSpace->l2Norm(&l2NormVec[0]);
/*
    pcout<<" d_psiMemSpace = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    d_RMatrixMemSpace.resize(d_numCells * d_numberDofsPerElement *
                               d_numberDofsPerElement);
    d_RMatrixMemSpace.setValue(0.0);

    d_MuMatrixMemSpace.resize(blockSize * blockSize);
    d_MuMatrixMemSpace.setValue(0.0);

    std::vector<double> effectiveOrbitalOccupancyHost;
    effectiveOrbitalOccupancyHost = effectiveOrbitalOccupancy;
    d_effectiveOrbitalOccupancyMemSpace.resize(blockSize);
    d_effectiveOrbitalOccupancyMemSpace.copyFrom(effectiveOrbitalOccupancyHost);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      effectiveOrbitalOccupancyHost_4x;
    effectiveOrbitalOccupancyHost_4x.resize(blockSize);

    for( unsigned int i = 0 ; i <blockSize; i++)
      {
        effectiveOrbitalOccupancyHost_4x[i] = 4.0*effectiveOrbitalOccupancy[i];
      }
    d_4xeffectiveOrbitalOccupancyMemSpace.resize(blockSize);
    d_4xeffectiveOrbitalOccupancyMemSpace.copyFrom(effectiveOrbitalOccupancyHost_4x);


    d_degenerateState = degeneracy;
    d_eigenValues     = eigenValues;

    d_vectorList.resize(0);
    for (unsigned int iVec = 0; iVec < blockSize; iVec++)
      {
        unsigned int totalNumDegenerateStates = d_degenerateState[iVec].size();
        for (unsigned int jVec = 0; jVec < totalNumDegenerateStates; jVec++)
          {
            d_vectorList.push_back(iVec);
            d_vectorList.push_back(d_degenerateState[iVec][jVec]);
          }
      }

    d_MuMatrixMemSpaceCellWise.resize((d_vectorList.size() / 2) *
                                      d_numCells,
                                    0.0);
    d_MuMatrixHostCellWise.resize((d_vectorList.size() / 2) *
                                    d_numCells,
                              0.0);

    d_MuMatrixHost.resize(blockSize*blockSize);
    std::fill(d_MuMatrixHost.begin(),d_MuMatrixHost.end(),0.0);
    d_vectorListMemSpace.resize(d_vectorList.size());
    d_vectorListMemSpace.copyFrom(d_vectorList);
    if (blockSize != d_blockSize)
      {
        // If the number of vectors in the size is different, then the Map has
        // to be re-initialised. The d_blockSize is set to -1 in the
        // constructor, so that this if condition is satisfied the first time
        // the code is called.

        d_cellWaveFunctionMatrixMemSpace.resize(
          d_numCells * d_numberDofsPerElement * blockSize, 0.0);

        d_cellRMatrixTimesWaveMatrixMemSpace.resize(
          d_numCells * d_numberDofsPerElement * blockSize, 0.0);
        d_cellWaveFunctionQuadMatrixMemSpace.resize(d_numCells*
                                                      d_numQuadsPerCell *
                                                    blockSize);
        d_cellWaveFunctionQuadMatrixMemSpace.setValue(0.0);
      }

    d_negEigenValuesMemSpace.resize(blockSize);

    for(signed int iBlock = 0 ; iBlock < blockSize; iBlock++)
      {
        eigenValues[iBlock] = -1.0*eigenValues[iBlock];
      }
    d_negEigenValuesMemSpace.copyFrom(eigenValues);


    auto cellJxW = d_basisOperationsPtr->JxW();
    d_inputJxWMemSpace.resize(d_numQuadsPerCell*d_numCells);
    d_BLASWrapperPtr->hadamardProduct(d_numCells*d_numQuadsPerCell,
                                      differenceInDensity.data(),
                                      cellJxW.data(),
                                      d_inputJxWMemSpace.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::computeMuMatrix(
    dftfe::utils::MemoryStorage<double, memorySpace>
                                 &                           inputJxwMemSpace,
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      memorySpace> &psiVecMemSpace)
  {

    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0;
    char               transposeMat      = 'T';
    char               doNotTransposeMat = 'N';

    d_MuMatrixMemSpace.setValue(0.0);
    d_MuMatrixHost.setValue(0.0);

    d_cellWaveFunctionMatrixMemSpace.setValue(0.0);


 //       std::vector<dealii::types::global_dof_index> fullFlattenedMap;
 //   vectorTools::computeCellLocalIndexSetMap(
 //     psiVecMemSpace.getMPIPatternP2P(),
 //     *d_matrixFreeDataPtr,
 //     d_matrixFreeVectorComponent,
 //     d_blockSize,
 //     fullFlattenedMap);




     d_basisOperationsPtr->reinit(d_blockSize,
                                 d_numCells,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 false); // TODO should this be set to true
                                        //

    std::pair<unsigned int, unsigned int> cellRange = std::make_pair(0,d_numCells);
    d_basisOperationsPtr->extractToCellNodalDataKernel(psiVecMemSpace,
                                                       d_cellWaveFunctionMatrixMemSpace.data(),
                                                       cellRange);

    //for( unsigned int iCell = 0; iCell < d_numCells; iCell++)
    //{
    //	    for(unsigned int dofId = 0; dofId < d_numberDofsPerElement; dofId++)
    //
    //{
//		    dealii::types::global_dof_index localNodeIdInput =
     //          fullFlattenedMap[iCell*d_numberDofsPerElement + dofId];
//
//              dcopy_(&d_blockSize,
//                     psiVecMemSpace.data() + localNodeIdInput,
//                     &inc,
//                     &d_cellWaveFunctionMatrixMemSpace.data()[iCell*d_numberDofsPerElement*d_blockSize + d_blockSize * dofId],
//                     &inc);
//
//	    }
//    }

    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
    const unsigned int strideB = 0;
    const unsigned int strideC = d_numQuadsPerCell * d_blockSize;

    auto shapeFunctionData = d_basisOperationsPtr->shapeFunctionData(false);
    d_BLASWrapperPtr->xgemmStridedBatched(
      'N',
      'N',
      d_blockSize,
      d_numQuadsPerCell,
      d_numberDofsPerElement,
      &alpha,
      d_cellWaveFunctionMatrixMemSpace.begin(),
      d_blockSize,
      strideA,
      shapeFunctionData.begin(),
      d_numberDofsPerElement,
      strideB,
      &beta,
      d_cellWaveFunctionQuadMatrixMemSpace.begin(),
      d_blockSize,
      strideC,
      d_numCells);
/*
                    double normcellWaveFunction = 0.0;
     d_BLASWrapperPtr->xnrm2(
      d_cellWaveFunctionMatrixMemSpace.size(),
      d_cellWaveFunctionMatrixMemSpace.data(),
      inc,
      mpi_communicator,
      &normcellWaveFunction);
     pcout<<" normcellWaveFunction = "<<normcellWaveFunction<<"\n";

                double normshapeFunctionData= 0.0;
     d_BLASWrapperPtr->xnrm2(
      shapeFunctionData.size(),
      shapeFunctionData.data(),
      inc,
      mpi_communicator,
      &normshapeFunctionData);
     pcout<<" normshapeFunctionData = "<<normshapeFunctionData<<"\n";

            double normcellWaveFunctionQuadMatrix = 0.0;
     d_BLASWrapperPtr->xnrm2(
      d_cellWaveFunctionQuadMatrixMemSpace.size(),
      d_cellWaveFunctionQuadMatrixMemSpace.data(),
      inc,
      mpi_communicator,
      &normcellWaveFunctionQuadMatrix);
     pcout<<" normcellWaveFunctionQuadMatrix = "<<normcellWaveFunctionQuadMatrix<<"\n";
  */
     unsigned int numVec = d_vectorList.size() / 2;
    d_MuMatrixMemSpaceCellWise.setValue(0.0);
/*
    pcout<<" d_effectiveOrbitalOccupancyMemSpace = "<<d_effectiveOrbitalOccupancyMemSpace.size()<<" d_vectorListMemSpace = "<<
	    d_vectorListMemSpace.size()<<" d_cellWaveFunctionQuadMatrixMemSpace = "<<d_cellWaveFunctionQuadMatrixMemSpace.size()<<
	    " inputJxwMemSpace = "<<inputJxwMemSpace.size()<<" d_MuMatrixMemSpaceCellWise = "<<d_MuMatrixMemSpaceCellWise.size()<<"\n";
  */  
    
    muMatrixMemSpaceKernel(d_numCells,
                   numVec,
                   d_numQuadsPerCell,
                   d_blockSize,
                   d_BLASWrapperPtr,
                   d_effectiveOrbitalOccupancyMemSpace,
                   d_vectorListMemSpace,
                   d_cellWaveFunctionQuadMatrixMemSpace,
                   inputJxwMemSpace,
                   d_MuMatrixMemSpaceCellWise);


   // pcout<<" d_numCells = "<< d_numCells <<" numVec = "<<numVec<<" d_numQuadsPerCell = "<<d_numQuadsPerCell<<"  d_blockSize = "<<d_blockSize<<"\n";
/*
    pcout<<" d_effectiveOrbitalOccupancyMemSpace = \n";
    for( unsigned int iBlock = 0 ; iBlock < d_effectiveOrbitalOccupancyMemSpace.size(); iBlock++)
    {
	    pcout<<d_effectiveOrbitalOccupancyMemSpace.data()[iBlock]<<" ";
    }
    pcout<<"\n";

    pcout<<" d_vectorListMemSpace = \n";
        for( unsigned int iBlock = 0 ; iBlock < d_vectorListMemSpace.size(); iBlock++)
    {
            pcout<<d_vectorListMemSpace.data()[iBlock]<<" ";
    }
	pcout<<"\n";
	*/
    /*
    double norminputJxwMemSpace = 0.0;
     d_BLASWrapperPtr->xnrm2(
      inputJxwMemSpace.size(),
      inputJxwMemSpace.data(),
      inc,
      mpi_communicator,
      &norminputJxwMemSpace);
     pcout<<" norminputJxwMemSpace = "<<norminputJxwMemSpace<<"\n";
   */
     d_MuMatrixHostCellWise.copyFrom(d_MuMatrixMemSpaceCellWise);

    //pcout<<"  = numVec = "<<numVec<<"\n";

    for (unsigned int iVecList = 0; iVecList < numVec; iVecList++)
      {
        unsigned int iVec            = d_vectorList[2 * iVecList];
        unsigned int degenerateVecId = d_vectorList[2 * iVecList + 1];
        for (unsigned int iCell = 0; iCell < d_numCells; iCell++)
          {
            d_MuMatrixHost[iVec * d_blockSize + degenerateVecId] +=
              d_MuMatrixHostCellWise[iVecList + iCell * numVec];
          }
      }
/*
    double normMuMatrixHostCellWise = 0.0;
     d_BLASWrapperPtr->xnrm2(
      d_MuMatrixHostCellWise.size(),
      d_MuMatrixHostCellWise.data(),
      inc,
      mpi_communicator,
      &normMuMatrixHostCellWise);
    pcout<<" norm of d_MuMatrixHostCellWise = "<<normMuMatrixHostCellWise;
  */
    MPI_Allreduce(MPI_IN_PLACE,
                  &d_MuMatrixHost[0],
                  d_blockSize * d_blockSize,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    d_MuMatrixMemSpace.copyFrom(d_MuMatrixHost);

   /* 
   pcout<<" d_MuMatrixMemSpace = \n"; 
   for( unsigned int iBlock = 0 ; iBlock < d_blockSize; iBlock++)
    {
	    for ( unsigned int jBlock = 0; jBlock < d_blockSize; jBlock++)
	    {
		    pcout<< d_MuMatrixHost.data()[jBlock + iBlock*d_blockSize] <<" "; 
	    }
	    pcout<<"\n";
    
    }
    */
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::computeRMatrix(
    dftfe::utils::MemoryStorage<double, memorySpace>
      &inputJxwMemSpace)
  {

    auto shapeFunctionDataTranspose = d_basisOperationsPtr->shapeFunctionData(true);
    auto shapeFunctionData = d_basisOperationsPtr->shapeFunctionData(false);

    rMatrixMemSpaceKernel(d_numCells,
                  d_numberDofsPerElement,
                  d_numQuadsPerCell,
                  d_BLASWrapperPtr,
                  shapeFunctionData,
                  shapeFunctionDataTranspose,
                  d_inputJxWMemSpace,
                  d_RMatrixMemSpace);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::distributeX()
  {

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dotProductHost(d_blockSize, 0.0);

    auto invSqrtMassMat = d_basisOperationsPtr->inverseSqrtMassVectorBasisData();

    std::vector<double> l2NormVec(d_blockSize,0.0);

    //d_blockedXPtr->l2Norm(&l2NormVec[0]);

    /*
    pcout<<" d_blockedXPtr before inv = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */

    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      invSqrtMassMat.data(),
      d_blockedXPtr->data(),
      d_blockedXPtr->data(),
      d_mapNodeIdToProcId.data());


    //d_blockedXPtr->l2Norm(&l2NormVec[0]);
/*
    pcout<<" d_blockedXPtr after inv = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    d_blockedXPtr->updateGhostValues();


    d_constraintsInfo.distribute(*d_blockedXPtr);


    //d_blockedXPtr->l2Norm(&l2NormVec[0]);
/*
    pcout<<" d_blockedXPtr after dist = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    multiVectorDotProdQuadWise(*d_blockedXPtr,
                               *d_psiMemSpace,
                               dotProductHost);
   /*
    pcout<<" dotProdVals = \n";
   for(unsigned int iB = 0 ; iB  <d_blockSize; iB++)
   {
	   pcout<<" iB = "<<iB<<" dot prod = "<<dotProductHost[iB] <<"\n";
   }
   */

    dftfe::utils::MemoryStorage<dataTypes::number , memorySpace>
      dotProductMemSpace(d_blockSize, 0.0);

    for( unsigned int i = 0 ; i <d_blockSize; i++)
      {
        dotProductHost[i] = -1.0*dotProductHost[i];
      }

    dotProductMemSpace.copyFrom(dotProductHost);


    d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(d_blockSize,
                                                      d_locallyOwnedSize,
                                                      d_psiMemSpace->data(),
						      dotProductMemSpace.data(),
                                                      d_blockedXPtr->data());

    //d_blockedXPtr->l2Norm(&l2NormVec[0]);

    /*
    pcout<<" d_blockedXPtr final = \n";
    for(unsigned int iB = 0; iB  < d_blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */

    d_blockedXPtr->updateGhostValues();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::multiVectorDotProdQuadWise(dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                                                                                         memorySpace> &      vec1,
                                                                                 dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                                                                                         memorySpace> &vec2,
                                                                                 dftfe::utils::MemoryStorage<dataTypes::number , dftfe::utils::MemorySpace::HOST>&
                                                                                   dotProductOutputHost)
  {
	   d_basisOperationsPtr->reinit(d_blockSize,
                                // d_cellBlockSize,
				    d_numCells,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 false); // TODO should this be set to true
                                        //


    d_basisOperationsPtr->interpolate(vec1,
                vec1QuadValues.data());

    d_basisOperationsPtr->interpolate(vec2,
                                      vec2QuadValues.data());

    auto jxwVec = d_basisOperationsPtr->JxW();


    d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_numCells*d_numQuadsPerCell,
      1.0,
      jxwVec.data(),
      vec1QuadValues.data(),
      vec1QuadValues.data(),
      d_mapQuadIdToProcId.data());

    d_BLASWrapperPtr->hadamardProduct(d_blockSize*d_numCells*d_numQuadsPerCell,
                                      vec1QuadValues.data(),
                                      vec2QuadValues.data(),
                                      vecOutputQuadValues.data());

    unsigned int one = 1;
    double oneDouble = 1.0;
    double zeroDouble = 0.0;
    d_BLASWrapperPtr->xgemm(
      'N',
      'T',
      one,
      d_blockSize,
      d_numCells*d_numQuadsPerCell,
      &oneDouble,
      d_onesQuadMemSpace.data(),
      one,
      vecOutputQuadValues.data(),
      d_blockSize,
      &zeroDouble,
      tempOutputDotProdMemSpace.data(),
      one);

    dotProductOutputHost.resize(d_blockSize);
    dotProductOutputHost.copyFrom(tempOutputDotProdMemSpace);

    MPI_Allreduce(MPI_IN_PLACE,
                  &dotProductOutputHost[0],
                  d_blockSize,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::vmult(dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                                           memorySpace> &Ax,
                                                            dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                                                                              memorySpace> &x,
                                         unsigned int blockSize)
  {
    Ax.setValue(0.0);
    
                std::vector<double> l2NormVec(blockSize,0.0);

    //x.l2Norm(&l2NormVec[0]);
/*
    pcout<<" x in vmult = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
*/
    d_ksOperatorPtr->HX(x,
                        1.0, //scalarHX,
                        0.0, //scalarY,
                        0.0, //scalarX
                        Ax,
                        false); // onlyHPrimePartForFirstOrderDensityMatResponse

    d_constraintsInfo.set_zero(x);
    d_constraintsInfo.set_zero(Ax);

        //Ax.l2Norm(&l2NormVec[0]);

    /*
    pcout<<" Ax in vmult = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
    d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
                               d_blockSize,
      d_locallyOwnedSize,
      x.data(),
      d_negEigenValuesMemSpace.data(),
      Ax.data());

     //x.l2Norm(&l2NormVec[0]);
/*
    pcout<<" x after HX in vmult = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */

    //Ax.l2Norm(&l2NormVec[0]);
/*
    pcout<<" Ax at end in vmult = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
  }

  template <dftfe::utils::MemorySpace memorySpace>
  MultiVectorAdjointLinearSolverProblem<memorySpace>::~MultiVectorAdjointLinearSolverProblem()
  {

  }


  template class MultiVectorAdjointLinearSolverProblem<dftfe::utils::MemorySpace::HOST>;
  
#ifdef DFTFE_WITH_DEVICE
  template class MultiVectorAdjointLinearSolverProblem<dftfe::utils::MemorySpace::DEVICE>;
#endif 

}
