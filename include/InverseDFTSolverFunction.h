// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
//
// @authors Bikash Kanungo, Vishal Subramanian
//

#ifndef DFTFE_INVERSEDFTSOLVERFUNCTION_H
#define DFTFE_INVERSEDFTSOLVERFUNCTION_H

#include <TransferBetweenMeshesIncompatiblePartitioning.h>
#include <headers.h>
#include <MultiVectorAdjointLinearSolverProblem.h>
#include <MultiVectorMinResSolver.h>
#include <nonlinearSolverFunction.h>
#include <constraintMatrixInfo.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <dft.h>
#include "inverseDFTParameters.h"
namespace dftfe
{
  /**
   * @brief Class implementing the inverse DFT problem
   *
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro, dftfe::utils::MemorySpace memorySpace>
  class InverseDFTSolverFunction
  {
  public:
    /**
     * @brief Constructor
     */
    InverseDFTSolverFunction(const MPI_Comm &mpi_comm_parent,
                             const MPI_Comm &mpi_comm_domain,
                             const MPI_Comm &mpi_comm_interpool,
                             const MPI_Comm &mpi_comm_interband);

    //
    // reinit
    //
    void
    reinit(
      const std::vector<dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &rhoTargetQuadDataHost,
      const std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> &weightQuadDataHost,
      const std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> &potBaseQuadDataHost,
      dftfe::dftClass<FEOrder, FEOrder, memorySpace> &dftClass,
      const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousPsi, // assumes that the constraint matrix
                                         // has homogenous BC
      const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousAdjoint, // assumes that the constraint
                                             // matrix has homogenous BC
      const dealii::AffineConstraints<double> &constraintMatrixPot,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
      std::vector<std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>>
        &basisOperationsParentPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        & basisOperationsChildPtr,
      dftfe::KohnShamHamiltonianOperator<
        memorySpace> & kohnShamClass,
      const std::shared_ptr<TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>>
                                &  inverseDFTDoFManagerObjPtr,
      const std::vector<double> &kpointWeights,
      const unsigned int         numSpins,
      const unsigned int         numEigenValues,
      const unsigned int         matrixFreePsiVectorComponent,
      const unsigned int         matrixFreeAdjointVectorComponent,
      const unsigned int         matrixFreePotVectorComponent,
      const unsigned int         matrixFreeQuadratureComponentAdjointRhs,
      const unsigned int         matrixFreeQuadratureComponentPot,
      const bool                 isComputeDiagonalA,
      const bool                 isComputeShapeFunction,
      const dftParameters &      dftParams,
      const inverseDFTParameters & inverseDFTParams);


    void
    writeVxcDataToFile(std::vector<distributedCPUVec<double>> &pot,
                       unsigned int                            counter);

    void
    solveEigen(const std::vector<distributedCPUVec<double>> &pot);

    void
    dotProduct(const distributedCPUVec<double> &vec1,
               const distributedCPUVec<double> &vec2,
               unsigned int                     blockSize,
               std::vector<double> &            outputDot);

    void
    setInitialGuess(const std::vector<distributedCPUVec<double>> &pot,
                    const std::vector<std::vector<std::vector<double>>>
                      &targetPotValuesParentQuadData);

    std::vector<distributedCPUVec<double>>
    getInitialGuess() const;

    void
    getForceVector(std::vector<distributedCPUVec<double>> &pot,
                   std::vector<distributedCPUVec<double>> &force,
                   std::vector<double> &                   loss);



    void
    setSolution(const std::vector<distributedCPUVec<double>> &pot);

    void
    integrateWithShapeFunctionsForChildData(
      distributedCPUVec<double> &outputVec,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &quadInputData);
  private:

    void preComputeChildShapeFunction();
    void preComputeParentJxW();
      const dealii::MatrixFree<3, double> *    d_matrixFreeDataParent;
    const dealii::MatrixFree<3, double> *    d_matrixFreeDataChild;
    std::vector<distributedCPUVec<double>> d_pot;
    std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                  dftfe::utils::MemorySpace::HOST>>
      d_solutionPotVecForWritingInParentNodes;
//    std::vector<dftfe::linearAlgebra::MultiVector<double,
//                                                  dftfe::utils::MemorySpace::HOST>>
//                                                  d_solutionPotVecForWritingInParentNodesMFVec;

    std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> d_rhoTargetQuadDataHost;
    std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> d_rhoKSQuadDataHost;
    std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> d_weightQuadDataHost;
    std::vector<dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST>> d_potBaseQuadDataHost;
    dftfe::linearAlgebra::MultiVector<double,
      dftfe::utils::MemorySpace::HOST> d_adjointBlock;
    MultiVectorAdjointLinearSolverProblem<memorySpace> d_multiVectorAdjointProblem;
    MultiVectorMinResSolver d_multiVectorLinearMINRESSolver;

    dftfe::utils::MemoryStorage<dealii::types::global_dof_index, memorySpace>
      fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace,
      fullFlattenedArrayCellLocalProcIndexIdMapAdjointMemSpace,
      d_fullFlattenedMapChild;

    std::vector<dealii::types::global_dof_index> fullFlattenedArrayCellLocalProcIndexIdMapPsiHost,fullFlattenedArrayCellLocalProcIndexIdMapAdjointHost;

    dftfe::utils::MemoryStorage<double,
                                memorySpace>
    psiChildQuadDataMemorySpace, adjointChildQuadDataMemorySpace;

    // TODO remove this from gerForceVectorCPU
    dftfe::dftUtils::constraintMatrixInfo<memorySpace> constraintsMatrixPsiDataInfo,
      constraintsMatrixAdjointDataInfo;

    dftfe::linearAlgebra::MultiVector<double, memorySpace> psiBlockVecMemSpace,
      multiVectorAdjointOutputWithPsiConstraintsMemSpace,
      adjointInhomogenousDirichletValuesMemSpace, multiVectorAdjointOutputWithAdjointConstraintsMemSpace;
    const dealii::AffineConstraints<double> *d_constraintMatrixHomogeneousPsi;
    const dealii::AffineConstraints<double>
                                            *d_constraintMatrixHomogeneousAdjoint;
    const dealii::AffineConstraints<double> *d_constraintMatrixPot;
    dftfe::dftUtils::constraintMatrixInfo<memorySpace>           d_constraintsMatrixDataInfoPot;
    KohnShamHamiltonianOperator<
      memorySpace> *                       d_kohnShamClass;

    std::shared_ptr<TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>> d_transferDataPtr;
    std::vector<double>   d_kpointWeights;

    std::vector<double> d_childCellJxW, d_childCellShapeFunctionValue;
    std::vector<double> d_parentCellJxW, d_shapeFunctionValueParent;
    unsigned int        d_numSpins;
    unsigned int        d_numKPoints;
    unsigned int        d_matrixFreePsiVectorComponent;
    unsigned int        d_matrixFreeAdjointVectorComponent;
    unsigned int        d_matrixFreePotVectorComponent;
    unsigned int        d_matrixFreeQuadratureComponentAdjointRhs;
    unsigned int        d_matrixFreeQuadratureComponentPot;
    bool                d_isComputeDiagonalA;
    bool                d_isComputeShapeFunction;
    double              d_degeneracyTol;
    double              d_adjointTol;
    int                 d_adjointMaxIterations;
    MPI_Comm            d_mpi_comm_parent;
    MPI_Comm            d_mpi_comm_domain;
    MPI_Comm            d_mpi_comm_interband;
    MPI_Comm            d_mpi_comm_interpool;

    unsigned int                 d_numLocallyOwnedCellsParent;
    unsigned int                 d_numLocallyOwnedCellsChild;
    const dealii::DoFHandler<3> *d_dofHandlerParent;
    const dealii::DoFHandler<3> *d_dofHandlerChild;
    const dftParameters *        d_dftParams;
    const inverseDFTParameters * d_inverseDFTParams;

    distributedCPUVec<double>        d_MInvSqrt;
    distributedCPUVec<double>        d_MSqrt;
    unsigned int                     d_numEigenValues;
    std::vector<std::vector<double>> d_fractionalOccupancy;

    std::vector<double> d_wantedLower;
    std::vector<double> d_unwantedUpper;
    std::vector<double> d_unwantedLower;
    unsigned int        d_getForceCounter;
    double              d_fractionalOccupancyTol;
    dftBase *           d_dft;

    unsigned int                                    d_maxChebyPasses;
    double                                          d_lossPreviousIteration;
    double                                          d_tolForChebFiltering;
    elpaScalaManager *                              d_elpaScala;
#ifdef DFTFE_WITH_DEVICE

    chebyshevOrthogonalizedSubspaceIterationSolverDevice *d_subspaceIterationSolver;

#else
    chebyshevOrthogonalizedSubspaceIterationSolver *d_subspaceIterationSolver;
#endif


    std::vector<std::vector<double>> d_residualNormWaveFunctions;
    std::vector<std::vector<double>> d_eigenValues;
    unsigned int                     d_numElectrons;

    dealii::ConditionalOStream pcout;

    // TODO implemented for debugging purpose
    std::vector< dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>>
      d_targetPotValuesParentQuadData;

    bool d_resizeMemSpaceVecDuringInterpolation;

    dftfe::dftClass<FEOrder, FEOrder, memorySpace> *d_dftClassPtr;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> d_BLASWrapperPtr;

    std::vector<std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>>
      d_basisOperationsParentPtr;

      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperationsChildPtr;

      unsigned int d_numCellsParent, d_numCellsChild;

      unsigned int cellBlockSizeLimit;
  };
} // end of namespace dftfe
#endif // DFTFE_INVERSEDFTSOLVERFUNCTION_H
