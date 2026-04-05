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
// @author Phani Motamarri
//

#include <constants.h>
#include <kerkerSolverProblem.h>
#include <feevaluationWrapper.h>
#include <linearAlgebraOperations.h>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::uInt FEOrderElectro>
  kerkerSolverProblem<FEOrderElectro>::kerkerSolverProblem(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_matVecCount                = 0;
    d_isSpectrumComputed         = false;
    d_useChebyshevPreconditioner = true;
    d_chebyDegree                = 5;
    d_chebyDegreeConfigured      = 5;
    d_chebyLambdaMax             = 0.0;
    d_chebyLambdaMin             = 0.0;
    d_arePrimitiveTimesCached    = false;
    d_cachedMatvecTime           = 0.0;
    d_cachedAllreduceTime        = 0.0;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::init(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                      &basisOperationsPtr,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double>         &x,
    double                             kerkerMixingParameter,
    const dftfe::uInt                  matrixFreeVectorComponent,
    const dftfe::uInt                  matrixFreeQuadratureComponent,
    const dftfe::uInt                  matrixFreeAxQuadratureComponent)
  {
    d_basisOperationsPtr              = basisOperationsPtr;
    d_matrixFreeDataPRefinedPtr       = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPRefinedPtr     = &constraintMatrixPRefined;
    d_gamma                           = kerkerMixingParameter;
    d_matrixFreeVectorComponent       = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent   = matrixFreeQuadratureComponent;
    d_matrixFreeAxQuadratureComponent = matrixFreeAxQuadratureComponent;
    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
    computeDiagonalA();
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::reinit(
    distributedCPUVec<double> &x,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPointValues)
  {
    d_xPtr                  = &x;
    d_residualQuadValuesPtr = &quadPointValues;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::distributeX()
  {
    d_constraintMatrixPRefinedPtr->distribute(*d_xPtr);
  }

  template <dftfe::uInt FEOrderElectro>
  distributedCPUVec<double> &
  kerkerSolverProblem<FEOrderElectro>::getX()
  {
    return *d_xPtr;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::resetMatVecCount()
  {
    d_matVecCount = 0;
  }

  template <dftfe::uInt FEOrderElectro>
  dftfe::uInt
  kerkerSolverProblem<FEOrderElectro>::getMatVecCount() const
  {
    return d_matVecCount;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    FEEvaluationWrapperClass<1> fe_eval(*d_matrixFreeDataPRefinedPtr,
                                        d_matrixFreeVectorComponent,
                                        d_matrixFreeQuadratureComponent);

    dealii::VectorizedArray<double> zeroVec = 0.0;

    dealii::AlignedVector<dealii::VectorizedArray<double>> residualQuads(
      fe_eval.n_q_points, zeroVec);
    for (dftfe::uInt macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_cell_batches();
         ++macrocell)
      {
        std::fill(residualQuads.begin(), residualQuads.end(), zeroVec);
        const dftfe::uInt numSubCells =
          d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
            macrocell);
        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            dealii::CellId    subCellId = subCellPtr->id();
            const dftfe::uInt cellIndex =
              d_basisOperationsPtr->cellIndex(subCellId);
            const double *tempVec =
              d_residualQuadValuesPtr->data() + fe_eval.n_q_points * cellIndex;

            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              residualQuads[q][iSubCell] = -tempVec[q];
          }

        fe_eval.reinit(macrocell);
        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_value(residualQuads[q], q);

        fe_eval.integrate(dealii::EvaluationFlags::values);

        fe_eval.distribute_local_to_global(rhs);
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }

  // Matrix-Free Jacobi preconditioner application
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::precondition_Jacobi(
    distributedCPUVec<double>       &dst,
    const distributedCPUVec<double> &src,
    const double                     omega) const
  {
    dst = src;
    dst.scale(d_diagonalA);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    dealii::QGauss<3>      quadrature(C_num1DQuad(FEOrderElectro));
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
              elementalDiagonalA(i) +=
                (fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(i, q_point) +
                 4 * M_PI * d_gamma * fe_values.shape_value(i, q_point) *
                   fe_values.shape_value(i, q_point)) *
                fe_values.JxW(q_point);

          d_constraintMatrixPRefinedPtr->distribute_local_to_global(
            elementalDiagonalA, local_dof_indices, d_diagonalA);
        }

    // MPI operation to sync data
    d_diagonalA.compress(dealii::VectorOperation::add);

    // Store un-inverted diagonal for Lanczos D-inner product
    d_diagonalARaw.reinit(d_diagonalA);
    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        {
          if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
            d_diagonalARaw(i) = std::abs(d_diagonalA(i));
          else
            d_diagonalARaw(i) = 0.0;
        }
    d_diagonalARaw.compress(dealii::VectorOperation::insert);

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          d_diagonalA(i) = 1.0 / d_diagonalA(i);

    d_diagonalA.compress(dealii::VectorOperation::insert);

    d_isSpectrumComputed      = false;
    d_arePrimitiveTimesCached = false;
  }

  // Ax
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::AX(
    const dealii::MatrixFree<3, double>       &matrixFreeData,
    distributedCPUVec<double>                 &dst,
    const distributedCPUVec<double>           &src,
    const std::pair<dftfe::uInt, dftfe::uInt> &cell_range) const
  {
    FEEvaluationWrapperClass<1> fe_eval(matrixFreeData,
                                        d_matrixFreeVectorComponent,
                                        d_matrixFreeAxQuadratureComponent);

    dealii::VectorizedArray<double> kerkerConst =
      dealii::make_vectorized_array(4 * M_PI * d_gamma);


    for (dftfe::uInt cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        // fe_eval.gather_evaluate(src,dealii::EvaluationFlags::values|dealii::EvaluationFlags::gradients);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(dealii::EvaluationFlags::values |
                         dealii::EvaluationFlags::gradients);
        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          {
            fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
            fe_eval.submit_value(fe_eval.get_value(q) * kerkerConst, q);
          }
        // fe_eval.integrate_scatter(dealii::EvaluationFlags::values|dealii::EvaluationFlags::gradients,dst);
        fe_eval.integrate(dealii::EvaluationFlags::values |
                          dealii::EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::vmult(distributedCPUVec<double> &Ax,
                                             distributedCPUVec<double> &x)
  {
    ++d_matVecCount;

    Ax = 0.0;
    x.update_ghost_values();
    AX(*d_matrixFreeDataPRefinedPtr,
       Ax,
       x,
       std::make_pair(0, d_matrixFreeDataPRefinedPtr->n_cell_batches()));
    Ax.compress(dealii::VectorOperation::add);
    // d_matrixFreeDataPRefinedPtr->cell_loop(
    //  &kerkerSolverProblem<FEOrderElectro>::AX, this, Ax, x);
  }


  //
  // usesCustomPreconditioner
  //
  template <dftfe::uInt FEOrderElectro>
  bool
  kerkerSolverProblem<FEOrderElectro>::usesCustomPreconditioner() const
  {
    return d_useChebyshevPreconditioner;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::setPreconditionerOptions(
    const bool        useChebyshev,
    const dftfe::uInt chebyDegree)
  {
    d_useChebyshevPreconditioner = useChebyshev;
    d_chebyDegreeConfigured      = std::max<dftfe::uInt>(1, chebyDegree);
    d_chebyDegree                = d_chebyDegreeConfigured;

    if (d_isSpectrumComputed)
      tuneChebyshevDegreeFromSpectrum();
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::tuneChebyshevDegreeFromSpectrum()
  {
    tunePreconditionerForSolve(1.0, 1e-7);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::tunePreconditionerForSolve(
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

    const double logRhoInv = -std::log(rho);

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
        const double kEst  = std::ceil(T_nominal / static_cast<double>(d));
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

    pcout << "Kerker Chebyshev tune: r0=" << initialResidual
          << ", tMatvec=" << d_cachedMatvecTime
          << ", tAllreduce=" << d_cachedAllreduceTime
          << ", degree=" << d_chebyDegree << std::endl;
  }


  //
  // Lanczos-based spectral bound estimation for D^{-1}A on CPU.
  // Helmholtz operator is positive definite — no null-space projection needed.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::computeSpectralBounds()
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
    d_constraintMatrixPRefinedPtr->set_zero(vVec);

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

    // alpha = <v, w>_D = v^T A v = v^T tempAx
    double alpha = 0.0;
    for (dftfe::uInt i = 0; i < localSize; ++i)
      alpha += vVec.local_element(i) * tempAx.local_element(i);
    MPI_Allreduce(
      MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    // w -= alpha * v
    wVec.add(-alpha, vVec);

    Tlanczos[0] = alpha;

    double beta  = 0.0;
    int    index = 0;

    for (unsigned int j = 1; j < lanczosIterations; ++j)
      {
        // beta = D-norm of w
        double betaSq = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          betaSq += wVec.local_element(i) * d_diagonalARaw.local_element(i) *
                    wVec.local_element(i);
        MPI_Allreduce(
          MPI_IN_PLACE, &betaSq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        beta = std::sqrt(std::abs(betaSq));

        if (beta < 1e-30)
          break;

        zVec = vVec;

        // v = w / beta
        for (dftfe::uInt i = 0; i < localSize; ++i)
          vVec.local_element(i) = wVec.local_element(i) / beta;

        // w = D^{-1} A v
        vmult(tempAx, vVec);
        for (dftfe::uInt i = 0; i < localSize; ++i)
          wVec.local_element(i) =
            d_diagonalA.local_element(i) * tempAx.local_element(i);

        wVec.add(-beta, zVec);

        alpha = 0.0;
        for (dftfe::uInt i = 0; i < localSize; ++i)
          alpha += vVec.local_element(i) * tempAx.local_element(i);
        MPI_Allreduce(
          MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        wVec.add(-alpha, vVec);

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

    pcout << "Kerker Chebyshev preconditioner spectrum: lambdaMin = "
          << d_chebyLambdaMin << ", lambdaMax = " << d_chebyLambdaMax
          << ", kappa = " << d_chebyLambdaMax / d_chebyLambdaMin
          << ", degree = " << d_chebyDegree << std::endl;
  }


  //
  // Chebyshev-Jacobi preconditioner on CPU: incremental form (deal.II
  // convention).
  // Helmholtz operator is positive definite — no null-space projection needed.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::applyPreconditioner(
    distributedCPUVec<double>       &dst,
    const distributedCPUVec<double> &src)
  {
    if (!d_useChebyshevPreconditioner)
      {
        precondition_Jacobi(dst, src, 0.3);
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

    if (d_chebyDegree <= 1)
      return;

    // update = dst
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

        rhoOld = rhoNew;
      }
  }


  template class kerkerSolverProblem<1>;
  template class kerkerSolverProblem<2>;
  template class kerkerSolverProblem<3>;
  template class kerkerSolverProblem<4>;
  template class kerkerSolverProblem<5>;
  template class kerkerSolverProblem<6>;
  template class kerkerSolverProblem<7>;
  template class kerkerSolverProblem<8>;
  template class kerkerSolverProblem<9>;
  template class kerkerSolverProblem<10>;
  template class kerkerSolverProblem<11>;
  template class kerkerSolverProblem<12>;
  template class kerkerSolverProblem<13>;
  template class kerkerSolverProblem<14>;
  template class kerkerSolverProblem<15>;
  template class kerkerSolverProblem<16>;
} // namespace dftfe
