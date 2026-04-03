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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

#include <constants.h>
#include <poissonSolverProblem.h>
#include <vectorUtilities.h>
#include <feevaluationWrapper.h>
#include <linearAlgebraOperations.h>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::uInt FEOrderElectro>
  poissonSolverProblem<FEOrderElectro>::poissonSolverProblem(
    const MPI_Comm &mpi_comm)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
  {
    d_isMeanValueConstraintComputed = false;
    d_isGradSmearedChargeRhs        = false;
    d_isStoreSmearedChargeRhs       = false;
    d_isReuseSmearedChargeRhs       = false;
    d_isFastConstraintsInitialized  = false;
    d_rhoValuesPtr                  = NULL;
    d_atomsPtr                      = NULL;
    d_smearedChargeValuesPtr        = NULL;
    d_isSpectrumComputed            = false;
    d_useChebyshevPreconditioner    = true;
    d_chebyDegree                   = 5;
    d_chebyDegreeConfigured         = 5;
    d_matVecCount                   = 0;
    d_chebyLambdaMax                = 0.0;
    d_chebyLambdaMin                = 0.0;
    d_arePrimitiveTimesCached       = false;
    d_cachedMatvecTime              = 0.0;
    d_cachedAllreduceTime           = 0.0;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::clear()
  {
    d_diagonalA.reinit(0);
    d_rhsSmearedCharge.reinit(0);
    d_meanValueConstraintVec.reinit(0);
    d_isMeanValueConstraintComputed = false;
    d_isGradSmearedChargeRhs        = false;
    d_isStoreSmearedChargeRhs       = false;
    d_isReuseSmearedChargeRhs       = false;
    d_isFastConstraintsInitialized  = false;
    d_rhoValuesPtr                  = NULL;
    d_atomsPtr                      = NULL;
    d_smearedChargeValuesPtr        = NULL;
    d_isSpectrumComputed            = false;
    d_chebyDegree                   = d_chebyDegreeConfigured;
    d_matVecCount                   = 0;
    d_arePrimitiveTimesCached       = false;
    d_cachedMatvecTime              = 0.0;
    d_cachedAllreduceTime           = 0.0;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::reinit(
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

    d_basisOperationsPtr        = basisOperationsPtr;
    d_matrixFreeDataPtr         = &(basisOperationsPtr->matrixFreeData());
    d_xPtr                      = &x;
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

        d_isFastConstraintsInitialized = true;
      }
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::distributeX()
  {
    if (d_isMeanValueConstraintComputed)
      {
        // The raw-K CG solution is determined up to an additive constant.
        // Subtract the integral mean to enforce zero-integral-mean gauge:
        // mu = (sum_i a_i x_i) / Omega, where a_i = int N_i dx.
        // No distribute needed first: a_i = 0 on slave DOFs (assembled via
        // distribute_local_to_global), so slaves don't affect the dot product.
        const dftfe::uInt localSize = d_xPtr->locally_owned_size();
        double            localDot  = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          localDot +=
            d_meanValueWeights.local_element(i) * d_xPtr->local_element(i);
        double integralPhiH = 0.0;
        MPI_Allreduce(
          &localDot, &integralPhiH, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        const double mu = integralPhiH / d_domainVolume;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          d_xPtr->local_element(i) -= mu;
      }

    // Distribute to fill slave DOFs from masters
    d_constraintsInfo.distribute(*d_xPtr);
  }

  template <dftfe::uInt FEOrderElectro>
  distributedCPUVec<double> &
  poissonSolverProblem<FEOrderElectro>::getX()
  {
    return *d_xPtr;
  }

  template <dftfe::uInt FEOrderElectro>
  const distributedCPUVec<double> &
  poissonSolverProblem<FEOrderElectro>::getX() const
  {
    return *d_xPtr;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::computeRhs(
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

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

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
    // K is singular (K1=0), so Kx=b requires b perpendicular to
    // null(K)=span(1). Subtract the nodal mean (L2 projection) after set_zero
    // so that constrained rows (which are zero) do not pollute the sum.
    if (d_isMeanValueConstraintComputed)
      {
        const dftfe::uInt localSize = rhs.locally_owned_size();
        double            localSum  = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          localSum += rhs.local_element(i);
        double globalData[2] = {localSum, static_cast<double>(localSize)};
        MPI_Allreduce(
          MPI_IN_PLACE, globalData, 2, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        const double mu = globalData[0] / globalData[1];
        for (dftfe::uInt i = 0; i < localSize; ++i)
          rhs.local_element(i) -= mu;
        d_constraintMatrixPtr->set_zero(rhs);
      }
  }

  // Matrix-Free Jacobi preconditioner application
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::precondition_Jacobi(
    distributedCPUVec<double>       &dst,
    const distributedCPUVec<double> &src,
    const double                     omega) const
  {
    // dst = src;
    // dst.scale(d_diagonalA);

    for (dftfe::uInt i = 0; i < dst.locally_owned_size(); i++)
      dst.local_element(i) =
        d_diagonalA.local_element(i) * src.local_element(i);
  }

  //
  // Compute mean value constraint which is required in case of fully periodic
  // boundary conditions
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::computeMeanValueConstraint()
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
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
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
    d_meanValueWeights.reinit(d_meanValueConstraintVec);
    {
      const dftfe::uInt localSize =
        d_meanValueConstraintVec.locally_owned_size();
      double localVol = 0.0;
      for (dftfe::uInt i = 0; i < localSize; ++i)
        localVol += d_meanValueConstraintVec.local_element(i);
      MPI_Allreduce(
        &localVol, &d_domainVolume, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
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
  poissonSolverProblem<FEOrderElectro>::computeDiagonalA()
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
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
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
    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i) &&
          !d_constraintMatrixPtr->is_constrained(i))
        localMaxDiag = std::max(localMaxDiag, std::abs(d_diagonalA(i)));

    double globalMaxDiag = 0.0;
    MPI_Allreduce(
      &localMaxDiag, &globalMaxDiag, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
    const double diagFloor = std::max(globalMaxDiag * 1.0e-12, 1.0e-20);

    // Store un-inverted diagonal for Lanczos D-inner product.
    // Zero both diagonals at constrained DOFs and at the mean-value
    // constrained DOF so that they are invisible to D^{-1} scaling
    // and to the Lanczos D-inner product.
    d_diagonalARaw = d_diagonalA;
    for (dealii::types::global_dof_index i = 0; i < d_diagonalARaw.size(); ++i)
      if (d_diagonalARaw.in_local_range(i) &&
          d_constraintMatrixPtr->is_constrained(i))
        d_diagonalARaw(i) = 0.0;

    // Also zero the mean-value constrained DOF
    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        d_diagonalARaw(d_meanValueConstraintNodeId) = 0.0;

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPtr->is_constrained(i) &&
            !(d_isMeanValueConstraintComputed &&
              i == d_meanValueConstraintNodeId &&
              dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                d_meanValueConstraintProcId))
          {
            d_diagonalARaw(i) =
              std::max(std::abs(d_diagonalARaw(i)), diagFloor);
            d_diagonalA(i) = 1.0 / d_diagonalARaw(i);
          }
        else
          d_diagonalA(i) = 0.0;

    d_diagonalA.compress(dealii::VectorOperation::insert);

    // Compute projection weight for constant-mode deflation (fully periodic).
    // d_projWeight = 1 / (1^T D 1), where D = d_diagonalARaw.
    d_projWeight = 0.0;
    if (d_isMeanValueConstraintComputed)
      {
        double localDsum = 0.0;
        for (dftfe::uInt i = 0; i < d_diagonalARaw.locally_owned_size(); ++i)
          localDsum += d_diagonalARaw.local_element(i);
        double globalDsum = 0.0;
        MPI_Allreduce(
          &localDsum, &globalDsum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        d_projWeight = (globalDsum > 1e-30) ? (1.0 / globalDsum) : 0.0;
      }

    // Diagonal changed, invalidate cached spectral bounds.
    d_isSpectrumComputed = false;
    d_arePrimitiveTimesCached = false;
  }


  //
  // Project out the constant mode in the D-inner product (fully periodic).
  // P_D v = v - (v^T D 1)/(1^T D 1) * 1
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::projectOutConstantMode(
    distributedCPUVec<double> &vec) const
  {
    if (!d_isMeanValueConstraintComputed || d_projWeight == 0.0)
      return;

    const dftfe::uInt localSize  = vec.locally_owned_size();
    double            localDotVD = 0.0;
    for (dftfe::uInt i = 0; i < localSize; ++i)
      localDotVD += vec.local_element(i) * d_diagonalARaw.local_element(i);

    double mu = 0.0;
    MPI_Allreduce(&localDotVD, &mu, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    mu *= d_projWeight;

    for (dftfe::uInt i = 0; i < localSize; ++i)
      vec.local_element(i) -= mu;
  }


  // Ax
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::AX(
    const dealii::MatrixFree<3, double>       &matrixFreeData,
    distributedCPUVec<double>                 &dst,
    const distributedCPUVec<double>           &src,
    const std::pair<dftfe::uInt, dftfe::uInt> &cell_range) const
  {
    dealii::VectorizedArray<double> quarter =
      dealii::make_vectorized_array(1.0 / (4.0 * M_PI));

    dealii::FEEvaluation<3, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      matrixFreeData,
      d_matrixFreeVectorComponent,
      d_matrixFreeQuadratureComponentAX);

    for (dftfe::uInt cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        // fe_eval.gather_evaluate(src,dealii::EvaluationFlags::gradients);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(dealii::EvaluationFlags::gradients);
        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          {
            fe_eval.submit_gradient(fe_eval.get_gradient(q) * quarter, q);
          }
        fe_eval.integrate(dealii::EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
        // fe_eval.integrate_scatter(dealii::EvaluationFlags::gradients,dst);
      }
  }


  // Raw Laplacian (with periodic constraint handling, without mean-value
  // Schur complement).  K is symmetric and maps the zero-mean subspace to
  // itself (1^T K v = 0), so CG stays in the correct subspace without any
  // projection here.  Constant-mode removal is handled in the RHS and the
  // preconditioner output instead.
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::vmult(distributedCPUVec<double> &Ax,
                                              distributedCPUVec<double> &x)
  {
    ++d_matVecCount;

    Ax = 0.0;
    x.update_ghost_values();
    AX(*d_matrixFreeDataPtr,
       Ax,
       x,
       std::make_pair(0, d_matrixFreeDataPtr->n_cell_batches()));
    Ax.compress(dealii::VectorOperation::add);
    x.zero_out_ghost_values();
    d_constraintsInfo.set_zero(x, 1);
  }


  //
  // usesCustomPreconditioner
  //
  template <dftfe::uInt FEOrderElectro>
  bool
  poissonSolverProblem<FEOrderElectro>::usesCustomPreconditioner() const
  {
    return d_useChebyshevPreconditioner;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::setPreconditionerOptions(
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
  poissonSolverProblem<FEOrderElectro>::resetMatVecCount()
  {
    d_matVecCount = 0;
  }


  template <dftfe::uInt FEOrderElectro>
  dftfe::uInt
  poissonSolverProblem<FEOrderElectro>::getMatVecCount() const
  {
    return d_matVecCount;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::tuneChebyshevDegreeFromSpectrum()
  {
    tunePreconditionerForSolve(1.0, 1e-7);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::tunePreconditionerForSolve(
    const double initialResidual,
    const double absTolerance)
  {
    d_chebyDegree = d_chebyDegreeConfigured;

    if (!d_useChebyshevPreconditioner || d_xPtr == NULL)
      return;

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

    const double logRhoInv = -std::log(rho); // > 0

    if (!d_arePrimitiveTimesCached)
      {
        distributedCPUVec<double> sampleSrc, sampleAx;
        sampleSrc.reinit(*d_xPtr);
        sampleAx.reinit(*d_xPtr);
        sampleSrc = 1.0;

        const dftfe::Int repeats = 12;

        MPI_Barrier(mpi_communicator);
        double tMatvecStart = MPI_Wtime();
        for (dftfe::Int r = 0; r < repeats; ++r)
          vmult(sampleAx, sampleSrc);
        double tMatvecLocal = (MPI_Wtime() - tMatvecStart) / repeats;
        MPI_Allreduce(&tMatvecLocal,
                      &d_cachedMatvecTime,
                      1,
                      MPI_DOUBLE,
                      MPI_MAX,
                      mpi_communicator);

        double tReduceLocal = 0.0;
        {
          double reduceBuf2[2] = {1.0, 2.0};
          double reduceBuf1[1] = {1.0};
          MPI_Barrier(mpi_communicator);
          double tReduceStart = MPI_Wtime();
          for (dftfe::Int r = 0; r < repeats; ++r)
            {
              MPI_Allreduce(MPI_IN_PLACE,
                            reduceBuf2,
                            2,
                            MPI_DOUBLE,
                            MPI_SUM,
                            mpi_communicator);
              MPI_Allreduce(MPI_IN_PLACE,
                            reduceBuf1,
                            1,
                            MPI_DOUBLE,
                            MPI_SUM,
                            mpi_communicator);
            }
          tReduceLocal = (MPI_Wtime() - tReduceStart) / repeats;
        }
        MPI_Allreduce(&tReduceLocal,
                      &d_cachedAllreduceTime,
                      1,
                      MPI_DOUBLE,
                      MPI_MAX,
                      mpi_communicator);

        d_arePrimitiveTimesCached = true;
      }

    const double safeInitial = std::max(initialResidual, 1.0e-30);
    const double safeAbsTol  = std::max(absTolerance, 1.0e-30);
    const double relNeed     = std::max(safeInitial / safeAbsTol, 1.0 + 1e-12);
    const double T_nominal   = std::log(2.0 * relNeed) / (2.0 * logRhoInv);

    dftfe::uInt bestD        = 1;
    double      bestPredTime = 1.0e300;

    for (dftfe::uInt d = 1; d <= d_chebyDegreeConfigured; ++d)
      {
        const double kEst = std::ceil(T_nominal / static_cast<double>(d));
        const double tPrec = static_cast<double>(d) * d_cachedMatvecTime;
        const double pred =
          kEst * (d_cachedMatvecTime + tPrec + d_cachedAllreduceTime);

        if (pred <= bestPredTime)
          {
            bestPredTime = pred;
            bestD        = d;
          }
      }
    d_chebyDegree = bestD;

        pcout << "Poisson Chebyshev tune: r0=" << initialResidual
          << ", tMatvec=" << d_cachedMatvecTime
          << ", tAllreduce=" << d_cachedAllreduceTime
          << ", degree=" << d_chebyDegree << std::endl;
  }


  //
  // Lanczos-based spectral bound estimation for D^{-1}A on CPU.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::computeSpectralBounds()
  {
    if (d_isSpectrumComputed)
      return;

    const unsigned int lanczosIterations = 20;
    const dftfe::uInt  localSize         = d_xPtr->locally_owned_size();

    distributedCPUVec<double> vVec, wVec, zVec, tempAx;
    vVec.reinit(*d_xPtr);
    wVec.reinit(*d_xPtr);
    zVec.reinit(*d_xPtr);
    tempAx.reinit(*d_xPtr);

    std::vector<double> Tlanczos(lanczosIterations * lanczosIterations, 0.0);

    // Random initial vector
    for (dftfe::uInt i = 0; i < localSize; ++i)
      vVec.local_element(i) =
        static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);

    // Project out constant mode for fully periodic systems
    projectOutConstantMode(vVec);

    // Normalize in D-inner product: <v,v>_D = v^T D v
    double vDv = 0.0;
    for (dftfe::uInt i = 0; i < localSize; ++i)
      vDv += vVec.local_element(i) * d_diagonalARaw.local_element(i) *
             vVec.local_element(i);
    MPI_Allreduce(MPI_IN_PLACE, &vDv, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    double invNorm = 1.0 / std::sqrt(vDv);
    vVec *= invNorm;

    // w = D^{-1} A v
    vmult(tempAx, vVec);
    for (dftfe::uInt i = 0; i < localSize; ++i)
      wVec.local_element(i) =
        d_diagonalA.local_element(i) * tempAx.local_element(i);
    projectOutConstantMode(wVec);

    // alpha = <v, w>_D = v^T A v = v^T tempAx
    double alpha = 0.0;
    for (dftfe::uInt i = 0; i < localSize; ++i)
      alpha += vVec.local_element(i) * tempAx.local_element(i);
    MPI_Allreduce(
      MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    // w -= alpha * v
    wVec.add(-alpha, vVec);

    // T[0,0] = alpha
    Tlanczos[0] = alpha;

    double beta  = 0.0;
    int    index = 0;

    for (unsigned int j = 1; j < lanczosIterations; ++j)
      {
        // beta = D-norm of w = sqrt(w^T D w)
        double betaSq = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          betaSq += wVec.local_element(i) * d_diagonalARaw.local_element(i) *
                    wVec.local_element(i);
        MPI_Allreduce(
          MPI_IN_PLACE, &betaSq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        beta = std::sqrt(std::abs(betaSq));

        if (beta < 1e-30)
          {
            break;
          }

        // z = v
        zVec = vVec;

        // v = w / beta
        for (dftfe::uInt i = 0; i < localSize; ++i)
          vVec.local_element(i) = wVec.local_element(i) / beta;

        // w = D^{-1} A v
        vmult(tempAx, vVec);
        for (dftfe::uInt i = 0; i < localSize; ++i)
          wVec.local_element(i) =
            d_diagonalA.local_element(i) * tempAx.local_element(i);
        projectOutConstantMode(wVec);

        // w -= beta * z
        wVec.add(-beta, zVec);

        // alpha = v^T tempAx
        alpha = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          alpha += vVec.local_element(i) * tempAx.local_element(i);
        MPI_Allreduce(
          MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        // w -= alpha * v
        wVec.add(-alpha, vVec);

        // Fill tridiagonal matrix
        index += 1;
        Tlanczos[index] = beta;
        index += lanczosIterations;
        Tlanczos[index] = alpha;
      }

    // Final beta for error bound
    double betaSqFinal = 0.0;
    for (dftfe::uInt i = 0; i < localSize; ++i)
      betaSqFinal += wVec.local_element(i) * d_diagonalARaw.local_element(i) *
                     wVec.local_element(i);
    MPI_Allreduce(
      MPI_IN_PLACE, &betaSqFinal, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    beta = std::sqrt(std::abs(betaSqFinal));

    // Eigendecomposition of tridiagonal T
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

    if (d_chebyLambdaMin < 1e-10)
      d_chebyLambdaMin = d_chebyLambdaMax / 30.0;

    d_isSpectrumComputed = true;
    tuneChebyshevDegreeFromSpectrum();

    pcout << "Poisson Chebyshev preconditioner spectrum: lambdaMin = "
          << d_chebyLambdaMin << ", lambdaMax = " << d_chebyLambdaMax
          << ", kappa = " << d_chebyLambdaMax / d_chebyLambdaMin
          << ", degree = " << d_chebyDegree << std::endl;

  }


  //
  // Chebyshev-Jacobi preconditioner on CPU: incremental form (deal.II
  // convention)
  //   x_0 = (1/θ) D^{-1} r,  d_0 = x_0
  //   d_{k+1} = ρ_new·ρ_old · d_k + (2ρ_new/δ) D^{-1}(r - A x_k)
  //   x_{k+1} = x_k + d_{k+1}
  //
  template <dftfe::uInt FEOrderElectro>
  void
  poissonSolverProblem<FEOrderElectro>::applyPreconditioner(
    distributedCPUVec<double>       &dst,
    const distributedCPUVec<double> &src)
  {
    if (!d_useChebyshevPreconditioner)
      {
        precondition_Jacobi(dst, src, 0.3);
        projectOutConstantMode(dst);
        return;
      }

    if (!d_isSpectrumComputed)
      computeSpectralBounds();

    const double theta    = (d_chebyLambdaMax + d_chebyLambdaMin) / 2.0;
    const double delta    = (d_chebyLambdaMax - d_chebyLambdaMin) / 2.0;
    const double sigma    = theta / delta;
    const double invTheta = 1.0 / theta;

    const dftfe::uInt localSize = d_xPtr->locally_owned_size();

    if (d_chebyWorkVec1.size() == 0)
      {
        d_chebyWorkVec1.reinit(*d_xPtr);
        d_chebyWorkVec2.reinit(*d_xPtr);
      }

    // Step 0: dst = (1/θ) D^{-1} src
    for (dftfe::uInt i = 0; i < localSize; ++i)
      dst.local_element(i) =
        invTheta * d_diagonalA.local_element(i) * src.local_element(i);
    projectOutConstantMode(dst);

    if (d_chebyDegree <= 1)
      return;

    // update = dst (first approximation is the first "update")
    for (dftfe::uInt i = 0; i < localSize; ++i)
      d_chebyWorkVec1.local_element(i) = dst.local_element(i);

    double rhoOld = 1.0 / sigma;
    for (dftfe::uInt k = 1; k < d_chebyDegree; ++k)
      {
        const double rhoNew  = 1.0 / (2.0 * sigma - rhoOld);
        const double factor1 = rhoNew * rhoOld;
        const double factor2 = 2.0 * rhoNew / delta;

        // temp = A * dst
        vmult(d_chebyWorkVec2, dst);

        // update = factor1*update + factor2*D^{-1}(src - A·dst)
        // dst += update
        for (dftfe::uInt i = 0; i < localSize; ++i)
          {
            const double w_i =
              d_diagonalA.local_element(i) *
              (src.local_element(i) - d_chebyWorkVec2.local_element(i));
            d_chebyWorkVec1.local_element(i) =
              factor1 * d_chebyWorkVec1.local_element(i) + factor2 * w_i;
            dst.local_element(i) += d_chebyWorkVec1.local_element(i);
          }

        projectOutConstantMode(dst);
        rhoOld = rhoNew;
      }
  }


#include "poissonSolverProblem.inst.cc"
} // namespace dftfe
