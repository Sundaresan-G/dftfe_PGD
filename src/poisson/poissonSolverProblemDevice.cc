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

/**
 * @author Gourab Panigrahi
 *
 */

#include <poissonSolverProblemDevice.h>
#include <MemoryTransfer.h>
#include <feevaluationWrapper.h>
#include <random>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::uInt FEOrderElectro>
  poissonSolverProblemDevice<FEOrderElectro>::poissonSolverProblemDevice(
    const MPI_Comm &mpi_comm)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
  {
    d_isMeanValueConstraintComputed      = false;
    d_isGradSmearedChargeRhs             = false;
    d_isStoreSmearedChargeRhs            = false;
    d_isReuseSmearedChargeRhs            = false;
    d_isFastConstraintsInitialized       = false;
    d_isHomogenousConstraintsInitialized = false;
    d_isSpectrumComputed                 = false;
    d_useChebyshevPreconditioner         = true;
    d_chebyDegree                        = 5;
    d_chebyDegreeConfigured              = 5;
    d_matVecCount                        = 0;
    d_chebyLambdaMax                     = 0.0;
    d_chebyLambdaMin                     = 0.0;
    d_areChebyWorkVecsInitialized        = false;
    d_rhoValuesPtr                       = NULL;
    d_atomsPtr                           = NULL;
    d_smearedChargeValuesPtr             = NULL;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::clear()
  {
    d_diagonalA.reinit(0);
    d_rhsSmearedCharge.reinit(0);
    d_meanValueConstraintVec.reinit(0);
    d_isMeanValueConstraintComputed      = false;
    d_isGradSmearedChargeRhs             = false;
    d_isStoreSmearedChargeRhs            = false;
    d_isReuseSmearedChargeRhs            = false;
    d_isFastConstraintsInitialized       = false;
    d_isHomogenousConstraintsInitialized = false;
    d_isSpectrumComputed                 = false;
    d_chebyDegree                        = d_chebyDegreeConfigured;
    d_matVecCount                        = 0;
    d_chebyLambdaMax                     = 0.0;
    d_chebyLambdaMin                     = 0.0;
    d_areChebyWorkVecsInitialized        = false;
    d_rhoValuesPtr                       = NULL;
    d_atomsPtr                           = NULL;
    d_smearedChargeValuesPtr             = NULL;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::reinit(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                            &basisOperationsPtr,
    distributedCPUVec<double>               &x,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const dftfe::uInt                        matrixFreeVectorComponent,
    const dftfe::uInt matrixFreeQuadratureComponentRhsDensity,
    const dftfe::uInt matrixFreeQuadratureComponentAX,
    const std::map<dealii::types::global_dof_index, double> &atoms,
    const std::map<dealii::CellId, std::vector<double>> &smearedChargeValues,
    const dftfe::uInt smearedChargeQuadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoValues,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                      BLASWrapperPtr,
    const bool        isComputeDiagonalA,
    const bool        isComputeMeanValueConstraint,
    const bool        smearedNuclearCharges,
    const bool        isRhoValues,
    const bool        isGradSmearedChargeRhs,
    const dftfe::uInt smearedChargeGradientComponentId,
    const bool        storeSmearedChargeRhs,
    const bool        reuseSmearedChargeRhs,
    const bool        reinitializeFastConstraints)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double time = MPI_Wtime();

    d_basisOperationsPtr = basisOperationsPtr;
    d_matrixFreeDataPtr  = &(basisOperationsPtr->matrixFreeData());
    d_xPtr               = &x;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_xPtr->get_partitioner(), 1, d_xDevice);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xDevice.locallyOwnedSize() *
                                               d_xDevice.numVectors(),
                                             d_xDevice.begin(),
                                             d_xPtr->begin());

    d_constraintMatrixPtr       = &constraintMatrix;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentRhsDensity =
      matrixFreeQuadratureComponentRhsDensity;
    d_matrixFreeQuadratureComponentAX = matrixFreeQuadratureComponentAX;
    d_rhoValuesPtr                    = isRhoValues ? &rhoValues : NULL;
    d_atomsPtr                        = smearedNuclearCharges ? NULL : &atoms;
    d_smearedChargeValuesPtr =
      smearedNuclearCharges ? &smearedChargeValues : NULL;
    d_smearedChargeQuadratureId        = smearedChargeQuadratureId;
    d_isGradSmearedChargeRhs           = isGradSmearedChargeRhs;
    d_smearedChargeGradientComponentId = smearedChargeGradientComponentId;
    d_isStoreSmearedChargeRhs          = storeSmearedChargeRhs;
    d_isReuseSmearedChargeRhs          = reuseSmearedChargeRhs;
    d_BLASWrapperPtr                   = BLASWrapperPtr;
    d_nLocalCells                      = d_matrixFreeDataPtr->n_cell_batches();
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();
    d_areChebyWorkVecsInitialized = false;

    AssertThrow(
      storeSmearedChargeRhs == false || reuseSmearedChargeRhs == false,
      dealii::ExcMessage(
        "DFT-FE Error: both store and reuse smeared charge rhs cannot be true at the same time."));

    if (isComputeMeanValueConstraint)
      {
        computeMeanValueConstraint();
        d_isMeanValueConstraintComputed = true;
      }

    if (isComputeDiagonalA)
      computeDiagonalA();

    if (!d_isFastConstraintsInitialized || reinitializeFastConstraints)
      {
        d_constraintsInfo.initialize(
          d_matrixFreeDataPtr->get_vector_partitioner(
            matrixFreeVectorComponent),
          constraintMatrix);

        setupConstraints();

        // Setup MatrixFree
        unsigned int nVectors = 1;

        // Create matrixFreeWrapperDevice
        d_matrixFreeWrapperDevice = std::make_unique<
          dftfe::MatrixFreeWrapperClass<double,
                                        dftfe::operatorList::Laplace,
                                        dftfe::utils::MemorySpace::DEVICE,
                                        false>>(
          FEOrderElectro + 1,
          mpi_communicator,
          d_matrixFreeDataPtr,
          constraintMatrix,
          d_BLASWrapperPtr,
          d_matrixFreeVectorComponent,
          d_matrixFreeQuadratureComponentAX,
          nVectors);

        // Init MatrixFree
        d_matrixFreeWrapperDevice->init();

        d_isFastConstraintsInitialized       = true;
        d_isHomogenousConstraintsInitialized = true;
      }
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::distributeX()
  {
    if (d_isMeanValueConstraintComputed)
      {
        // The raw-K CG solution is determined up to an additive constant.
        // Subtract the integral mean to enforce zero-integral-mean gauge:
        // mu = (sum_i a_i x_i) / Omega, where a_i = int N_i dx.
        // No distribute needed first: a_i = 0 on slave DOFs (assembled via
        // distribute_local_to_global), so slaves don't affect the dot product.
        double integralPhiH = 0.0;
        d_BLASWrapperPtr->xdot(d_xLocalDof,
                               d_xDevice.begin(),
                               1,
                               d_meanValueWeightsDevice.begin(),
                               1,
                               mpi_communicator,
                               &integralPhiH);
        double negMu = -integralPhiH / d_domainVolume;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &negMu, d_onesDevice.begin(), 1, d_xDevice.begin(), 1);
      }

    // Distribute to fill slave DOFs from masters
    d_inhomogenousConstraintsTotalPotentialInfo.distribute(d_xDevice);
  }

  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  poissonSolverProblemDevice<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    rhs.reinit(*d_xPtr);
    rhs = 0;

    if (d_isStoreSmearedChargeRhs)
      {
        d_rhsSmearedCharge.reinit(*d_xPtr);
        d_rhsSmearedCharge = 0;
      }

    distributedCPUVec<double> tempvec;
    tempvec.reinit(rhs);
    tempvec = 0.0;
    tempvec.update_ghost_values();
    d_constraintsInfo.distribute(tempvec);

    dealii::FEEvaluation<3, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      *d_matrixFreeDataPtr,
      d_matrixFreeVectorComponent,
      d_matrixFreeQuadratureComponentAX);

    dftfe::Int isPerformStaticCondensation =
      (tempvec.linfty_norm() > 1e-10) ? 1 : 0;

    MPI_Bcast(&isPerformStaticCondensation,
              1,
              dftfe::dataTypes::mpi_type_id(&isPerformStaticCondensation),
              0,
              mpi_communicator);

    if (isPerformStaticCondensation == 1)
      {
        dealii::VectorizedArray<double> quarter =
          dealii::make_vectorized_array(1.0 / (4.0 * M_PI));
        for (dftfe::uInt macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            fe_eval.reinit(macrocell);
            fe_eval.read_dof_values_plain(tempvec);
            fe_eval.evaluate(dealii::EvaluationFlags::gradients);
            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              {
                fe_eval.submit_gradient(-quarter * fe_eval.get_gradient(q), q);
              }
            fe_eval.integrate(dealii::EvaluationFlags::gradients);
            fe_eval.distribute_local_to_global(rhs);
          }
      }

    // rhs contribution from electronic charge
    if (d_rhoValuesPtr)
      {
        FEEvaluationWrapperClass<1> fe_eval_density(
          *d_matrixFreeDataPtr,
          d_matrixFreeVectorComponent,
          d_matrixFreeQuadratureComponentRhsDensity);

        dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuads(
          fe_eval_density.n_q_points, dealii::make_vectorized_array(0.0));
        for (dftfe::uInt macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            fe_eval_density.reinit(macrocell);

            std::fill(rhoQuads.begin(),
                      rhoQuads.end(),
                      dealii::make_vectorized_array(0.0));
            const dftfe::uInt numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId subCellId = subCellPtr->id();
                dftfe::uInt    cellIndex =
                  d_basisOperationsPtr->cellIndex(subCellId);
                const double *tempVec = d_rhoValuesPtr->data() +
                                        cellIndex * fe_eval_density.n_q_points;

                for (dftfe::uInt q = 0; q < fe_eval_density.n_q_points; ++q)
                  rhoQuads[q][iSubCell] = tempVec[q];
              }


            for (dftfe::uInt q = 0; q < fe_eval_density.n_q_points; ++q)
              {
                fe_eval_density.submit_value(rhoQuads[q], q);
              }
            fe_eval_density.integrate(dealii::EvaluationFlags::values);
            fe_eval_density.distribute_local_to_global(rhs);
          }
      }

    // rhs contribution from atomic charge at fem nodes
    if (d_atomsPtr != NULL)
      for (std::map<dealii::types::global_dof_index, double>::const_iterator
             it = (*d_atomsPtr).begin();
           it != (*d_atomsPtr).end();
           ++it)
        {
          std::vector<dealii::AffineConstraints<double>::size_type>
            local_dof_indices_origin(1, it->first); // atomic node
          dealii::Vector<double> cell_rhs_origin(1);
          cell_rhs_origin(0) = -(it->second); // atomic charge

          d_constraintMatrixPtr->distribute_local_to_global(
            cell_rhs_origin, local_dof_indices_origin, rhs);
        }
    else if (d_smearedChargeValuesPtr != NULL && !d_isGradSmearedChargeRhs &&
             !d_isReuseSmearedChargeRhs)
      {
        // const dftfe::uInt   num_quad_points_sc =
        // d_matrixFreeDataPtr->get_quadrature(d_smearedChargeQuadratureId).size();

        dealii::FEEvaluation<3, -1> fe_eval_sc(*d_matrixFreeDataPtr,
                                               d_matrixFreeVectorComponent,
                                               d_smearedChargeQuadratureId);

        const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc.n_q_points;

        dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
          numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
        for (dftfe::uInt macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool              isMacroCellTrivial = true;
            const dftfe::uInt numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId             subCellId = subCellPtr->id();
                const std::vector<double> &tempVec =
                  d_smearedChargeValuesPtr->find(subCellId)->second;
                if (tempVec.size() == 0)
                  continue;

                for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                  smearedbQuads[q][iSubCell] = tempVec[q];

                isMacroCellTrivial = false;
              }

            if (!isMacroCellTrivial)
              {
                fe_eval_sc.reinit(macrocell);
                for (dftfe::uInt q = 0; q < fe_eval_sc.n_q_points; ++q)
                  {
                    fe_eval_sc.submit_value(smearedbQuads[q], q);
                  }
                fe_eval_sc.integrate(dealii::EvaluationFlags::values);

                fe_eval_sc.distribute_local_to_global(rhs);

                if (d_isStoreSmearedChargeRhs)
                  {
                    fe_eval_sc.reinit(macrocell);
                    for (dftfe::uInt q = 0; q < fe_eval_sc.n_q_points; ++q)
                      {
                        fe_eval_sc.submit_value(smearedbQuads[q], q);
                      }
                    fe_eval_sc.integrate(dealii::EvaluationFlags::values);

                    fe_eval_sc.distribute_local_to_global(d_rhsSmearedCharge);
                  }
              }
          }
      }
    else if (d_smearedChargeValuesPtr != NULL && d_isGradSmearedChargeRhs)
      {
        dealii::FEEvaluation<3, -1> fe_eval_sc2(*d_matrixFreeDataPtr,
                                                d_matrixFreeVectorComponent,
                                                d_smearedChargeQuadratureId);

        const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc2.n_q_points;

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
        for (dftfe::uInt i = 0; i < 3; i++)
          zeroTensor[i] = dealii::make_vectorized_array(0.0);

        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          smearedbQuads(numQuadPointsSmearedb, zeroTensor);
        for (dftfe::uInt macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool              isMacroCellTrivial = true;
            const dftfe::uInt numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId             subCellId = subCellPtr->id();
                const std::vector<double> &tempVec =
                  d_smearedChargeValuesPtr->find(subCellId)->second;
                if (tempVec.size() == 0)
                  continue;

                for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                  smearedbQuads[q][d_smearedChargeGradientComponentId]
                               [iSubCell] = tempVec[q];

                isMacroCellTrivial = false;
              }

            if (!isMacroCellTrivial)
              {
                fe_eval_sc2.reinit(macrocell);
                for (dftfe::uInt q = 0; q < fe_eval_sc2.n_q_points; ++q)
                  {
                    fe_eval_sc2.submit_gradient(smearedbQuads[q], q);
                  }
                fe_eval_sc2.integrate(dealii::EvaluationFlags::gradients);
                fe_eval_sc2.distribute_local_to_global(rhs);
              }
          }
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    if (d_isReuseSmearedChargeRhs)
      rhs += d_rhsSmearedCharge;

    if (d_isStoreSmearedChargeRhs)
      d_rhsSmearedCharge.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPtr->set_zero(rhs);

    // For fully periodic systems, enforce solvability: 1^T b = 0.
    // K is singular (K1=0), so Kx=b requires b perpendicular to null(K)=span(1).
    // Subtract the nodal mean (L2 projection) after set_zero so that
    // constrained rows (which are zero) do not pollute the sum.
    if (d_isMeanValueConstraintComputed)
      {
        const dftfe::uInt localSize = rhs.locally_owned_size();
        double            localSum  = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          localSum += rhs.local_element(i);
        double globalData[2] = {localSum,
                                static_cast<double>(localSize)};
        MPI_Allreduce(
          MPI_IN_PLACE, globalData, 2, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        const double mu = globalData[0] / globalData[1];
        for (dftfe::uInt i = 0; i < localSize; ++i)
          rhs.local_element(i) -= mu;
        d_constraintMatrixPtr->set_zero(rhs);
      }
  }


  //
  // Compute mean value constraint which is required in case of fully periodic
  // boundary conditions
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::computeMeanValueConstraint()
  {
    // allocate parallel distibuted vector to store mean value constraint
    d_meanValueConstraintVec.reinit(*d_xPtr);
    d_meanValueConstraintVec = 0;

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const dftfe::uInt      dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt      num_quad_points = quadrature.size();
    dealii::Vector<double> elementalValues(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          elementalValues = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; i++)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              elementalValues(i) +=
                fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);

          d_constraintMatrixPtr->distribute_local_to_global(
            elementalValues, local_dof_indices, d_meanValueConstraintVec);
        }

    // MPI operation to sync data
    d_meanValueConstraintVec.compress(dealii::VectorOperation::add);

    // Save un-normalized mass-lumped weights a_i = int N_i dx
    // and compute domain volume before the vector is normalized.
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_meanValueConstraintVec.get_partitioner(),
      1,
      d_meanValueWeightsDevice);
    {
      const dftfe::uInt localSize =
        d_meanValueConstraintVec.locally_owned_size();
      double localVol = 0.0;
      for (dftfe::uInt i = 0; i < localSize; ++i)
        localVol += d_meanValueConstraintVec.local_element(i);
      MPI_Allreduce(
        &localVol, &d_domainVolume, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                   dftfe::utils::MemorySpace::HOST>::
        copy(localSize,
             d_meanValueWeightsDevice.begin(),
             d_meanValueConstraintVec.begin());
    }

    dealii::IndexSet locallyOwnedElements =
      d_meanValueConstraintVec.locally_owned_elements();

    dealii::IndexSet locallyRelevantElements =
      d_constraintMatrixPtr->get_local_lines();

    // pick mean value constrained node such that it is not part
    // of periodic and hanging node constraint equations (both slave and master
    // node). This is done for simplicity of implementation.
    dealii::IndexSet allIndicesTouchedByConstraints(
      d_meanValueConstraintVec.size());
    std::vector<dealii::types::global_dof_index> tempSet;
    for (dealii::IndexSet::ElementIterator it = locallyRelevantElements.begin();
         it < locallyRelevantElements.end();
         it++)
      if (d_constraintMatrixPtr->is_constrained(*it))
        {
          const dealii::types::global_dof_index lineDof = *it;
          const std::vector<std::pair<dealii::types::global_dof_index, double>>
            *rowData = d_constraintMatrixPtr->get_constraint_entries(lineDof);
          tempSet.push_back(lineDof);
          for (dftfe::uInt j = 0; j < rowData->size(); ++j)
            tempSet.push_back((*rowData)[j].first);
        }

    if (d_atomsPtr)
      for (std::map<dealii::types::global_dof_index, double>::const_iterator
             it = (*d_atomsPtr).begin();
           it != (*d_atomsPtr).end();
           ++it)
        tempSet.push_back(it->first);

    allIndicesTouchedByConstraints.add_indices(tempSet.begin(), tempSet.end());
    locallyOwnedElements.subtract_set(allIndicesTouchedByConstraints);


    const dftfe::uInt localSizeOfPotentialChoices =
      locallyOwnedElements.n_elements();
    const dftfe::uInt totalProcs =
      dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    const dftfe::uInt this_mpi_process =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::vector<dftfe::uInt> localSizesOfPotentialChoices(totalProcs, 0);
    MPI_Allgather(&localSizeOfPotentialChoices,
                  1,
                  dftfe::dataTypes::mpi_type_id(&localSizeOfPotentialChoices),
                  &localSizesOfPotentialChoices[0],
                  1,
                  dftfe::dataTypes::mpi_type_id(
                    localSizesOfPotentialChoices.data()),
                  mpi_communicator);

    d_meanValueConstraintProcId = 0;
    for (dftfe::uInt iproc = 0; iproc < totalProcs; iproc++)
      {
        if (localSizesOfPotentialChoices[iproc] > 0)
          {
            d_meanValueConstraintProcId = iproc;
            break;
          }
      }

    double valueAtConstraintNode = 0;
    if (this_mpi_process == d_meanValueConstraintProcId)
      {
        AssertThrow(locallyOwnedElements.size() != 0,
                    dealii::ExcMessage(
                      "DFT-FE Error: please contact developers."));
        d_meanValueConstraintNodeId = *locallyOwnedElements.begin();
        AssertThrow(!d_constraintMatrixPtr->is_constrained(
                      d_meanValueConstraintNodeId),
                    dealii::ExcMessage(
                      "DFT-FE Error: Mean value constraint creation bug."));
        valueAtConstraintNode =
          d_meanValueConstraintVec[d_meanValueConstraintNodeId];
      }

    MPI_Bcast(&valueAtConstraintNode,
              1,
              MPI_DOUBLE,
              d_meanValueConstraintProcId,
              mpi_communicator);

    d_meanValueConstraintVec /= -valueAtConstraintNode;

    if (this_mpi_process == d_meanValueConstraintProcId)
      d_meanValueConstraintVec[d_meanValueConstraintNodeId] = 0;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::computeDiagonalA()
  {
    d_diagonalA.reinit(*d_xPtr);
    d_diagonalA = 0;

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
    const dftfe::uInt      dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt      num_quad_points = quadrature.size();
    dealii::Vector<double> elementalDiagonalA(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          elementalDiagonalA = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; i++)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) += (1.0 / (4.0 * M_PI)) *
                                       (fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_grad(i, q_point)) *
                                       fe_values.JxW(q_point);

          d_constraintMatrixPtr->distribute_local_to_global(elementalDiagonalA,
                                                            local_dof_indices,
                                                            d_diagonalA);
        }

    // MPI operation to sync data
    d_diagonalA.compress(dealii::VectorOperation::add);

    // Guard against pathologically small unconstrained diagonal entries that
    // can over-amplify D^{-1}A and destabilize Lanczos spectral bounds.
    double localMaxDiag = 0.0;
    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); i++)
      if (d_diagonalA.in_local_range(i) &&
          !d_constraintMatrixPtr->is_constrained(i))
        localMaxDiag = std::max(localMaxDiag, std::abs(d_diagonalA(i)));

    double globalMaxDiag = 0.0;
    MPI_Allreduce(
      &localMaxDiag, &globalMaxDiag, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
    const double diagFloor = std::max(globalMaxDiag * 1.0e-12, 1.0e-20);

    // Store un-inverted diagonal for Lanczos D-inner product
    {
      d_diagonalARawHost.reinit(d_diagonalA);
      d_diagonalARawHost = d_diagonalA;
      // Set constrained DOFs and mean-value constrained DOF to 0
      for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); i++)
        if (d_diagonalA.in_local_range(i))
          if (d_constraintMatrixPtr->is_constrained(i) ||
              (d_isMeanValueConstraintComputed &&
               i == d_meanValueConstraintNodeId &&
               dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                 d_meanValueConstraintProcId))
            d_diagonalARawHost(i) = 0.0;
          else
            d_diagonalARawHost(i) =
              std::max(std::abs(d_diagonalARawHost(i)), diagFloor);

      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        d_diagonalARawHost.get_partitioner(), 1, d_diagonalARawDevice);
      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                               d_diagonalARawDevice.begin(),
                                               d_diagonalARawHost.begin());
    }

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); i++)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPtr->is_constrained(i) &&
            !(d_isMeanValueConstraintComputed &&
              i == d_meanValueConstraintNodeId &&
              dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                d_meanValueConstraintProcId))
          d_diagonalA(i) = 1.0 / std::max(std::abs(d_diagonalA(i)), diagFloor);
        else
          d_diagonalA(i) = 0.0;

    d_diagonalA.compress(dealii::VectorOperation::insert);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_diagonalA.get_partitioner(), 1, d_diagonalAdevice);


    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_diagonalAdevice.begin(),
                                             d_diagonalA.begin());

    // Reset spectrum cache since diagonal changed
    d_isSpectrumComputed = false;

    // Compute projection weight for constant-mode deflation (fully periodic).
    // d_projWeight = 1 / (1^T D 1), where D = d_diagonalARaw (host copy).
    d_projWeight = 0.0;
    if (d_isMeanValueConstraintComputed)
      {
        double localDsum = 0.0;
        for (dftfe::uInt i = 0; i < d_diagonalARawHost.locally_owned_size();
             ++i)
          localDsum += d_diagonalARawHost.local_element(i);
        double globalDsum = 0.0;
        MPI_Allreduce(
          &localDsum, &globalDsum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        d_projWeight = (globalDsum > 1e-30) ? (1.0 / globalDsum) : 0.0;

        // Build ones-vector on device for axpy-based constant subtraction
        distributedCPUVec<double> onesHost;
        onesHost.reinit(d_diagonalA);
        for (dftfe::uInt i = 0; i < onesHost.locally_owned_size(); ++i)
          onesHost.local_element(i) = 1.0;
        d_onesDevice.reinit(d_xDevice);
        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                                 d_onesDevice.begin(),
                                                 onesHost.begin());
      }
  }


  //
  // Project out the constant mode in the D-inner product (fully periodic).
  // P_D v = v - (v^T D 1)/(1^T D 1) * 1
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::projectOutConstantMode(
    distributedDeviceVec<double> &vec)
  {
    if (!d_isMeanValueConstraintComputed || d_projWeight == 0.0)
      return;

    // vDot = v^T D 1 = sum_i v_i * D_i (via dot with d_diagonalARawDevice)
    double localDotVD = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           vec.begin(),
                           1,
                           d_diagonalARawDevice.begin(),
                           1,
                           mpi_communicator,
                           &localDotVD);

    // mu = (v^T D 1) / (1^T D 1)
    double negMu = -localDotVD * d_projWeight;

    // v -= mu * 1
    d_BLASWrapperPtr->xaxpy(
      d_xLocalDof, &negMu, d_onesDevice.begin(), 1, vec.begin(), 1);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  poissonSolverProblemDevice<FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::setupConstraints()
  {
    d_inhomogenousConstraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      *d_constraintMatrixPtr);
  }


  // computeAX
  // Raw Laplacian (with periodic constraint handling, without mean-value
  // Schur complement).  K is symmetric and maps the zero-mean subspace to
  // itself (1^T K v = 0), so CG stays in the correct subspace without any
  // projection here.  Constant-mode removal is handled in the RHS and the
  // preconditioner output instead.
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    ++d_matVecCount;

    dftfe::utils::deviceMemset(Ax.begin(), 0, d_xLen * sizeof(double));

    x.updateGhostValues();

    d_matrixFreeWrapperDevice->constraintsDistribute(x.data());

    d_matrixFreeWrapperDevice->computeAX(Ax.data(), x.data());

    d_matrixFreeWrapperDevice->constraintsDistributeTranspose(Ax.data(),
                                                              x.data());

    Ax.accumulateAddLocallyOwned();

    x.zeroOutGhosts();
    Ax.zeroOutGhosts();
    d_inhomogenousConstraintsTotalPotentialInfo.set_zero(x);
  }


  //
  // usesCustomPreconditioner
  //
  template <dftfe::uInt FEOrderElectro>
  bool
  poissonSolverProblemDevice<FEOrderElectro>::usesCustomPreconditioner() const
  {
    return d_useChebyshevPreconditioner;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::setPreconditionerOptions(
    const bool        useChebyshev,
    const dftfe::uInt chebyDegree)
  {
    d_useChebyshevPreconditioner = useChebyshev;
    d_chebyDegreeConfigured      = std::max<dftfe::uInt>(1, chebyDegree);
    d_chebyDegree                = d_chebyDegreeConfigured;

    // Re-tune immediately if spectral bounds are already available.
    if (d_isSpectrumComputed)
      tuneChebyshevDegreeFromSpectrum();
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::resetMatVecCount()
  {
    d_matVecCount = 0;
  }


  template <dftfe::uInt FEOrderElectro>
  dftfe::uInt
  poissonSolverProblemDevice<FEOrderElectro>::getMatVecCount() const
  {
    return d_matVecCount;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::tuneChebyshevDegreeFromSpectrum()
  {
    d_chebyDegree = d_chebyDegreeConfigured;

    if (d_chebyLambdaMin <= 0.0 || d_chebyLambdaMax <= d_chebyLambdaMin)
      return;

    const double kappa = d_chebyLambdaMax / d_chebyLambdaMin;
    if (kappa <= 1.0 + 1e-12)
      {
        d_chebyDegree = 1;
        return;
      }

    const double sqrtKappa = std::sqrt(kappa);
    const double rho       = (sqrtKappa - 1.0) / (sqrtKappa + 1.0);
    if (rho <= 1e-12)
      {
        d_chebyDegree = 1;
        return;
      }

    // Select d in [1, d_chebyDegreeConfigured] that minimises estimated total
    // matvec count: d * ceil(T/d), where T is the estimated Jacobi-CG
    // iteration count for a representative tolerance of 1e-7.
    // On ties prefer the larger d (fewer global synchronisations).
    const double logRhoInv = -std::log(rho); // > 0
    const double T_nominal =
      std::log(2.0 / 1e-7) / (2.0 * logRhoInv);

    dftfe::uInt bestD     = 1;
    double      bestTotal = static_cast<double>(d_chebyDegreeConfigured) *
                       std::ceil(T_nominal /
                                 static_cast<double>(d_chebyDegreeConfigured));
    for (dftfe::uInt d = 1; d <= d_chebyDegreeConfigured; ++d)
      {
        const double kEst  = std::ceil(T_nominal / static_cast<double>(d));
        const double total = static_cast<double>(d) * kEst;
        if (total <= bestTotal) // <= so ties go to larger d
          {
            bestTotal = total;
            bestD     = d;
          }
      }
    d_chebyDegree = bestD;
  }


  //
  // Lanczos-based spectral bound estimation for D^{-1}A
  // Adapted from generalisedLanczosLowerUpperBoundEigenSpectrum
  // in linearAlgebraOperationsOpt.cc.
  // Uses D-inner product: <u,v>_D = u^T D v, making D^{-1}A self-adjoint.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::computeSpectralBounds()
  {
    if (d_isSpectrumComputed)
      return;

    const dftfe::uInt lanczosIterations = 20;

    // Allocate temporary device vectors
    distributedDeviceVec<double> vVec, wVec, zVec, tempAx;
    vVec.reinit(d_xDevice);
    wVec.reinit(d_xDevice);
    zVec.reinit(d_xDevice);
    tempAx.reinit(d_xDevice);

    // Generate random vector on host, copy to device
    {
      distributedCPUVec<double> vHost;
      vHost.reinit(*d_xPtr);
      vHost = 0.0;
      std::mt19937                           rng(this_mpi_process);
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      for (dftfe::uInt i = 0; i < vHost.locally_owned_size(); ++i)
        vHost.local_element(i) = dist(rng);
      // Set constrained DOFs to zero
      d_constraintMatrixPtr->set_zero(vHost);

      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                               vVec.begin(),
                                               vHost.begin());
    }
    vVec.zeroOutGhosts();
    projectOutConstantMode(vVec);

    // Normalize in D-inner product: ||v||_D = sqrt(v^T D v)
    // tempAx = D * v (element-wise multiply by raw diagonal)
    d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                      d_diagonalARawDevice.begin(),
                                      vVec.begin(),
                                      tempAx.begin());
    double Dnormsq = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           vVec.begin(),
                           1,
                           tempAx.begin(),
                           1,
                           mpi_communicator,
                           &Dnormsq);
    double invDnorm = 1.0 / std::sqrt(std::abs(Dnormsq));
    d_BLASWrapperPtr->xscal(vVec.begin(), invDnorm, d_xLocalDof);

    // First Lanczos step: w = D^{-1} A v
    computeAX(tempAx, vVec);

    //   w = D^{-1} * tempAx = Jacobi .* tempAx
    d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                      d_diagonalAdevice.begin(),
                                      tempAx.begin(),
                                      wVec.begin());
    projectOutConstantMode(wVec);

    // alpha_0 = <v, w>_D = v^T D w = v^T (A v) = v^T tempAx
    double alpha = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           vVec.begin(),
                           1,
                           tempAx.begin(),
                           1,
                           mpi_communicator,
                           &alpha);
    // w = w - alpha * v
    double negAlpha = -alpha;
    d_BLASWrapperPtr->xaxpy(
      d_xLocalDof, &negAlpha, vVec.begin(), 1, wVec.begin(), 1);

    // Build tridiagonal matrix T (lower triangular storage)
    std::vector<double> Tlanczos(lanczosIterations * lanczosIterations, 0.0);
    Tlanczos[0] = alpha;

    dftfe::uInt index = 0;
    double      beta  = 0.0;

    for (dftfe::uInt j = 1; j < lanczosIterations; ++j)
      {
        // beta = D-norm of w = sqrt(w^T D w)
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalARawDevice.begin(),
                                          wVec.begin(),
                                          tempAx.begin());
        double betaSq = 0.0;
        d_BLASWrapperPtr->xdot(d_xLocalDof,
                               wVec.begin(),
                               1,
                               tempAx.begin(),
                               1,
                               mpi_communicator,
                               &betaSq);
        beta = std::sqrt(std::abs(betaSq));

        // z = v (save old v)
        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                                   zVec.begin(),
                                                   vVec.begin());

        // v = w / beta
        double invBeta = 1.0 / beta;
        d_BLASWrapperPtr->axpby(
          d_xLocalDof, invBeta, wVec.begin(), 0.0, vVec.begin());

        // w = D^{-1} A v
        computeAX(tempAx, vVec);
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalAdevice.begin(),
                                          tempAx.begin(),
                                          wVec.begin());
        projectOutConstantMode(wVec);

        // w -= beta * z
        double negBeta = -beta;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &negBeta, zVec.begin(), 1, wVec.begin(), 1);

        // alpha = <v, w>_D = v^T D w = v^T (A v) = v^T tempAx
        d_BLASWrapperPtr->xdot(d_xLocalDof,
                               vVec.begin(),
                               1,
                               tempAx.begin(),
                               1,
                               mpi_communicator,
                               &alpha);

        // w -= alpha * v
        negAlpha = -alpha;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &negAlpha, vVec.begin(), 1, wVec.begin(), 1);

        // Fill tridiagonal matrix (lower triangular)
        index += 1;
        Tlanczos[index] = beta; // sub-diagonal
        index += lanczosIterations;
        Tlanczos[index] = alpha; // diagonal
      }

    // Final beta for error bound
    d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                      d_diagonalARawDevice.begin(),
                                      wVec.begin(),
                                      tempAx.begin());
    double betaSqFinal = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           wVec.begin(),
                           1,
                           tempAx.begin(),
                           1,
                           mpi_communicator,
                           &betaSqFinal);
    beta = std::sqrt(std::abs(betaSqFinal));

    // Eigendecomposition of tridiagonal matrix T
    std::vector<double> eigenValuesT(lanczosIterations);
    char                jobz = 'N', uplo = 'L';
    const unsigned int  n = lanczosIterations, lda = lanczosIterations;
    int                 info;
    const unsigned int  lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
    std::vector<int>    iwork(liwork, 0);
    std::vector<double> work(lwork, 0.0);
    dsyevd_(&jobz,
            &uplo,
            &n,
            &Tlanczos[0],
            &lda,
            &eigenValuesT[0],
            &work[0],
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    std::sort(eigenValuesT.begin(), eigenValuesT.end());

    d_chebyLambdaMin = eigenValuesT[0];
    d_chebyLambdaMax = eigenValuesT[lanczosIterations - 1] + beta / 10.0;

    // Safety margin
    if (d_chebyLambdaMin < 1e-10)
      d_chebyLambdaMin = d_chebyLambdaMax / 30.0;

    tuneChebyshevDegreeFromSpectrum();

    pcout << "Device Poisson Chebyshev preconditioner spectrum: lambdaMin = "
          << d_chebyLambdaMin << ", lambdaMax = " << d_chebyLambdaMax
          << ", kappa = " << d_chebyLambdaMax / d_chebyLambdaMin
          << ", degree = " << d_chebyDegree << std::endl;

    d_isSpectrumComputed = true;
  }


  //
  // Chebyshev-Jacobi preconditioner: dst ≈ A^{-1} src
  //
  // 3-vector recurrence for D^{-1}A (Varga, "Matrix Iterative Analysis"):
  //   z_0 = (1/θ) D^{-1} r
  //   z_{k+1} = ρ_k z_k + (ρ_k/θ) w_k + (1-ρ_k) z_{k-1}
  // where w_k = D^{-1}(r - A z_k), θ = (λ_max+λ_min)/2, δ = (λ_max-λ_min)/2,
  //       β = δ/θ, ρ_1 = 1/(1-β²/2), ρ_k = 1/(1-β²ρ_{k-1}/4).
  //
  // Vectors: dst = z_curr, d_chebyWorkVec1 = z_prev, d_chebyWorkVec2 = temp.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrderElectro>::applyPreconditioner(
    distributedDeviceVec<double> &dst,
    distributedDeviceVec<double> &src)
  {
    if (!d_useChebyshevPreconditioner)
      {
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalAdevice.begin(),
                                          src.begin(),
                                          dst.begin());
        return;
      }

    if (!d_isSpectrumComputed)
      computeSpectralBounds();

    const double theta = (d_chebyLambdaMax + d_chebyLambdaMin) / 2.0;
    const double delta = (d_chebyLambdaMax - d_chebyLambdaMin) / 2.0;

    // Fallback: if spectral bounds are unreasonable, use plain Jacobi
    if (theta > 1e6 || d_chebyLambdaMax <= 0.0)
      {
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalAdevice.begin(),
                                          src.begin(),
                                          dst.begin());
        return;
      }

    const double betaSq   = (delta / theta) * (delta / theta);
    const double invTheta = 1.0 / theta;

    if (!d_areChebyWorkVecsInitialized)
      {
        d_chebyWorkVec1.reinit(d_xDevice);
        d_chebyWorkVec2.reinit(d_xDevice);
        d_chebyWorkVec3.reinit(d_xDevice);
        d_areChebyWorkVecsInitialized = true;
      }

    // z_0 = (1/θ) D^{-1} r
    d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                      d_diagonalAdevice.begin(),
                                      src.begin(),
                                      dst.begin());
    d_BLASWrapperPtr->xscal(dst.begin(), invTheta, d_xLocalDof);
    projectOutConstantMode(dst);

    if (d_chebyDegree <= 1)
      return;

    // z_prev = 0
    dftfe::utils::deviceMemset(d_chebyWorkVec1.begin(),
                               0,
                               d_xLocalDof * sizeof(double));

    double rho = 0.0;
    for (dftfe::uInt k = 1; k < d_chebyDegree; ++k)
      {
        if (k == 1)
          rho = 1.0 / (1.0 - betaSq / 2.0);
        else
          rho = 1.0 / (1.0 - betaSq * rho / 4.0);

        const double coeffCurr = rho;
        const double coeffPrev = 1.0 - rho;
        const double coeffW    = rho * invTheta;

        // temp = A * z_curr
        computeAX(d_chebyWorkVec2, dst);

        // temp = src - temp  (residual)
        d_BLASWrapperPtr->xscal(d_chebyWorkVec2.begin(), -1.0, d_xLocalDof);
        double one = 1.0;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &one, src.begin(), 1, d_chebyWorkVec2.begin(), 1);

        // temp = D^{-1} * residual  (w_k, in-place hadamard is safe)
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalAdevice.begin(),
                                          d_chebyWorkVec2.begin(),
                                          d_chebyWorkVec2.begin());

        // z_next = ρ z_curr + (ρ/θ) w + (1-ρ) z_prev
        // Write into d_chebyWorkVec3 (AX copy no longer needed here).
        d_BLASWrapperPtr->axpby(
          d_xLocalDof, coeffCurr, dst.begin(), 0.0, d_chebyWorkVec3.begin());
        d_BLASWrapperPtr->xaxpy(d_xLocalDof,
                                &coeffPrev,
                                d_chebyWorkVec1.begin(),
                                1,
                                d_chebyWorkVec3.begin(),
                                1);
        d_BLASWrapperPtr->xaxpy(d_xLocalDof,
                                &coeffW,
                                d_chebyWorkVec2.begin(),
                                1,
                                d_chebyWorkVec3.begin(),
                                1);

        // Advance: z_prev <- old z_curr, z_curr <- z_next.
        // Explicit copies keep dst's buffer stable (no swap).
        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                                   d_chebyWorkVec1.begin(),
                                                   dst.begin());
        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                                   dst.begin(),
                                                   d_chebyWorkVec3.begin());
        projectOutConstantMode(dst);
      }
  }


#include "poissonSolverProblemDevice.inst.cc"
} // namespace dftfe
