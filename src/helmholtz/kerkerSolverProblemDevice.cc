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
// @author Gourab Panigrahi
//

#include <constants.h>
#include <kerkerSolverProblemDevice.h>
#include <MemoryTransfer.h>
#include <feevaluationWrapper.h>
#include <linearAlgebraOperations.h>
#include <chebyshevPreconditionerDeviceKernels.h>
#include <random>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::uInt FEOrderElectro>
  kerkerSolverProblemDevice<FEOrderElectro>::kerkerSolverProblemDevice(
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
    d_matVecCount                 = 0;
    d_isSpectrumComputed          = false;
    d_useChebyshevPreconditioner  = true;
    d_chebyDegree                 = 5;
    d_chebyDegreeConfigured       = 5;
    d_chebyLambdaMax              = 0.0;
    d_chebyLambdaMin              = 0.0;
    d_arePrimitiveTimesCached     = false;
    d_cachedMatvecTime            = 0.0;
    d_cachedAllreduceTime         = 0.0;
    d_areChebyWorkVecsInitialized = false;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::init(
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
    d_nLocalCells = d_matrixFreeDataPRefinedPtr->n_cell_batches();

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      x.get_partitioner(), 1, d_xDevice);

    d_xPtr      = &x;
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();

    computeDiagonalA();
    setupConstraints();

    // Create BLASWrapper
    d_BLASWrapperPtr = std::make_shared<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>();

    // Setup MatrixFree
    unsigned int nVectors = 1;

    // Create matrixFreeWrapperDevice
    d_matrixFreeWrapperDevice = std::make_unique<
      dftfe::MatrixFreeWrapperClass<double,
                                    dftfe::operatorList::Helmholtz,
                                    dftfe::utils::MemorySpace::DEVICE,
                                    false>>(FEOrderElectro + 1,
                                            mpi_communicator,
                                            d_matrixFreeDataPRefinedPtr,
                                            constraintMatrixPRefined,
                                            d_BLASWrapperPtr,
                                            d_matrixFreeVectorComponent,
                                            d_matrixFreeAxQuadratureComponent,
                                            nVectors);

    // Init MatrixFree
    d_matrixFreeWrapperDevice->init();

    // Set Helmholtz coefficient
    d_matrixFreeWrapperDevice->initOperatorCoeffs(4 * M_PI * d_gamma);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::reinit(
    distributedCPUVec<double> &x,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPointValues)
  {
    d_xPtr                  = &x;
    d_residualQuadValuesPtr = &quadPointValues;

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_xDevice.begin(),
                                             d_xPtr->begin());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setupConstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      *d_constraintMatrixPRefinedPtr);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice);
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::resetMatVecCount()
  {
    d_matVecCount = 0;
  }

  template <dftfe::uInt FEOrderElectro>
  dftfe::uInt
  kerkerSolverProblemDevice<FEOrderElectro>::getMatVecCount() const
  {
    return d_matVecCount;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dftfe::Int feOrder1 =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent)
        .get_fe()
        .tensor_degree();
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


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeDiagonalA()
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
    {
      distributedCPUVec<double> diagonalARawHost;
      diagonalARawHost.reinit(d_diagonalA);
      for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
        if (d_diagonalA.in_local_range(i))
          {
            if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
              diagonalARawHost(i) = std::abs(d_diagonalA(i));
            else
              diagonalARawHost(i) = 0.0;
          }
      diagonalARawHost.compress(dealii::VectorOperation::insert);
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        diagonalARawHost.get_partitioner(), 1, d_diagonalARawDevice);
      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                               d_diagonalARawDevice.begin(),
                                               diagonalARawHost.begin());
    }

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          d_diagonalA(i) = 1.0 / d_diagonalA(i);

    d_diagonalA.compress(dealii::VectorOperation::insert);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_diagonalA.get_partitioner(), 1, d_diagonalAdevice);


    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_diagonalAdevice.begin(),
                                             d_diagonalA.begin());

    d_isSpectrumComputed      = false;
    d_arePrimitiveTimesCached = false;
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeAX(
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
  }


  template <dftfe::uInt FEOrderElectro>
  bool
  kerkerSolverProblemDevice<FEOrderElectro>::usesCustomPreconditioner() const
  {
    return d_useChebyshevPreconditioner;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setPreconditionerOptions(
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
  kerkerSolverProblemDevice<FEOrderElectro>::tuneChebyshevDegreeFromSpectrum()
  {
    tunePreconditionerForSolve(1.0, 1e-7);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::tunePreconditionerForSolve(
    const double initialResidual,
    const double absTolerance)
  {
    d_chebyDegree = d_chebyDegreeConfigured;

    if (!d_useChebyshevPreconditioner || d_xLocalDof <= 0)
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
        distributedDeviceVec<double> sampleSrc, sampleAx;
        sampleSrc.reinit(d_xDevice);
        sampleAx.reinit(d_xDevice);
        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                                   sampleSrc.begin(),
                                                   d_xDevice.begin());

        const dftfe::Int repeats = 12;

        MPI_Barrier(mpi_communicator);
        double tMatvecStart = MPI_Wtime();
        for (dftfe::Int r = 0; r < repeats; ++r)
          computeAX(sampleAx, sampleSrc);
        dftfe::utils::deviceSynchronize();
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

    pcout << "Device Kerker Chebyshev tune: r0=" << initialResidual
          << ", tMatvec=" << d_cachedMatvecTime
          << ", tAllreduce=" << d_cachedAllreduceTime
          << ", degree=" << d_chebyDegree << std::endl;
  }


  //
  // Lanczos-based spectral bound estimation for D^{-1}A (Helmholtz operator).
  // Uses D-inner product: <u,v>_D = u^T D v.
  //
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeSpectralBounds()
  {
    if (d_isSpectrumComputed)
      return;

    const dftfe::uInt lanczosIterations = 20;

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
      d_constraintMatrixPRefinedPtr->set_zero(vHost);

      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                               vVec.begin(),
                                               vHost.begin());
    }
    vVec.zeroOutGhosts();

    // Normalize in D-inner product
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
    d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                      d_diagonalAdevice.begin(),
                                      tempAx.begin(),
                                      wVec.begin());

    // alpha = v^T tempAx  (= <v, Av> = <v, w>_D)
    double alpha = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           vVec.begin(),
                           1,
                           tempAx.begin(),
                           1,
                           mpi_communicator,
                           &alpha);
    double negAlpha = -alpha;
    d_BLASWrapperPtr->xaxpy(
      d_xLocalDof, &negAlpha, vVec.begin(), 1, wVec.begin(), 1);

    std::vector<double> Tlanczos(lanczosIterations * lanczosIterations, 0.0);
    Tlanczos[0] = alpha;

    dftfe::uInt index = 0;
    double      beta  = 0.0;

    for (dftfe::uInt j = 1; j < lanczosIterations; ++j)
      {
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

        dftfe::utils::MemoryTransfer<
          dftfe::utils::MemorySpace::DEVICE,
          dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                                   zVec.begin(),
                                                   vVec.begin());

        double invBeta = 1.0 / beta;
        d_BLASWrapperPtr->axpby(
          d_xLocalDof, invBeta, wVec.begin(), 0.0, vVec.begin());

        computeAX(tempAx, vVec);
        d_BLASWrapperPtr->hadamardProduct(d_xLocalDof,
                                          d_diagonalAdevice.begin(),
                                          tempAx.begin(),
                                          wVec.begin());

        double negBeta = -beta;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &negBeta, zVec.begin(), 1, wVec.begin(), 1);

        d_BLASWrapperPtr->xdot(d_xLocalDof,
                               vVec.begin(),
                               1,
                               tempAx.begin(),
                               1,
                               mpi_communicator,
                               &alpha);

        negAlpha = -alpha;
        d_BLASWrapperPtr->xaxpy(
          d_xLocalDof, &negAlpha, vVec.begin(), 1, wVec.begin(), 1);

        index += 1;
        Tlanczos[index] = beta;
        index += lanczosIterations;
        Tlanczos[index] = alpha;
      }

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

    pcout << "Device Kerker Chebyshev preconditioner spectrum: lambdaMin = "
          << d_chebyLambdaMin << ", lambdaMax = " << d_chebyLambdaMax
          << ", kappa = " << d_chebyLambdaMax / d_chebyLambdaMin
          << ", degree = " << d_chebyDegree << std::endl;
  }


  //
  // Chebyshev-Jacobi preconditioner: dst ≈ A^{-1} src
  // Incremental form (deal.II convention).
  //
  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::applyPreconditioner(
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

    const double theta    = (d_chebyLambdaMax + d_chebyLambdaMin) / 2.0;
    const double delta    = (d_chebyLambdaMax - d_chebyLambdaMin) / 2.0;
    const double sigma    = theta / delta;
    const double invTheta = 1.0 / theta;

    if (!d_areChebyWorkVecsInitialized)
      {
        d_chebyWorkVec1.reinit(d_xDevice);
        d_chebyWorkVec2.reinit(d_xDevice);
        d_areChebyWorkVecsInitialized = true;
      }

    // Step 0: dst = (1/θ) D^{-1} src  [fused kernel]
    chebyshevPrecondStep0Device(dst.begin(),
                                src.begin(),
                                d_diagonalAdevice.begin(),
                                invTheta,
                                d_xLocalDof);

    if (d_chebyDegree <= 1)
      return;

    // update = dst
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLocalDof,
                                               d_chebyWorkVec1.begin(),
                                               dst.begin());

    double rhoOld = 1.0 / sigma;
    for (dftfe::uInt k = 1; k < d_chebyDegree; ++k)
      {
        const double rhoNew  = 1.0 / (2.0 * sigma - rhoOld);
        const double factor1 = rhoNew * rhoOld;
        const double factor2 = 2.0 * rhoNew / delta;

        computeAX(d_chebyWorkVec2, dst);

        // Fused: w = D^{-1}(src - Ax), update = f1*update + f2*w, dst += update
        chebyshevPrecondStepDevice(dst.begin(),
                                   d_chebyWorkVec1.begin(),
                                   src.begin(),
                                   d_chebyWorkVec2.begin(),
                                   d_diagonalAdevice.begin(),
                                   factor1,
                                   factor2,
                                   d_xLocalDof);
        rhoOld = rhoNew;
      }
  }


#include "kerkerSolverProblemDevice.inst.cc"
} // namespace dftfe
