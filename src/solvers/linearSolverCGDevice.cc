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

#include <linearSolverCGDevice.h>
#include <MemoryTransfer.h>
#include <MemoryStorage.h>
#include "linearSolverCGDeviceKernels.h"
#include <cmath>

namespace dftfe
{
  // constructor
  linearSolverCGDevice::linearSolverCGDevice(
    const MPI_Comm  &mpi_comm_parent,
    const MPI_Comm  &mpi_comm_domain,
    const solverType type,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , d_type(type)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_BLASWrapperPtr(BLASWrapperPtr)
  {}


  // solve
  void
  linearSolverCGDevice::solve(linearSolverProblemDevice &problem,
                              const double               absTolerance,
                              const dftfe::uInt          maxNumberIterations,
                              const dftfe::Int           debugLevel,
                              bool                       distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double start_time = MPI_Wtime();
    double time;

    // compute RHS
    distributedCPUVec<double> rhsHost;
    problem.computeRhs(rhsHost);

    distributedDeviceVec<double> &x = problem.getX();
    distributedDeviceVec<double>  rhsDevice;
    rhsDevice.reinit(x);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(rhsDevice.locallyOwnedSize() *
                                               rhsDevice.numVectors(),
                                             rhsDevice.begin(),
                                             rhsHost.begin());


    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (debugLevel >= 4)
      pcout << "Time for compute rhsHost and copy to Device: "
            << time - start_time << std::endl;


    problem.resetMatVecCount();

    d_xLocalDof = x.locallyOwnedSize() * x.numVectors();
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_localDotSums(2);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_localRR(1);

    double     res = 0.0, initial_res = 0.0;
    bool       conv = false;
    dftfe::Int it   = 0;

    try
      {
        x.updateGhostValues();

        if (d_type == CG)
          {
            // -------------------------------------------------
            // Standard preconditioned CG (2 allreduces/iter)
            // Convention: r = Ax - b  (positive residual)
            //   z = M^{-1} r, p = -z
            //   w = Ap, alpha = delta/(p·w)
            //   x += alpha*p, r += alpha*w
            //   z = M^{-1} r, delta_new = r·z, beta = delta_new/delta_old
            // Convergence check uses ||r||_2 from r·r.
            // r·z and r·r are reduced together in one MPI_Allreduce.
            //   p = -z + beta*p
            // -------------------------------------------------

            d_rvec.reinit(x);
            d_uvec.reinit(x);
            d_pvec.reinit(x);
            d_wvec.reinit(x);

            d_rvec.zeroOutGhosts();
            d_uvec.zeroOutGhosts();
            d_pvec.zeroOutGhosts();
            d_wvec.zeroOutGhosts();

            // r = Ax - rhs
            problem.computeAX(d_rvec, x);
            double mOne = -1.0;
            d_BLASWrapperPtr->xaxpy(
              d_xLocalDof, &mOne, rhsDevice.begin(), 1, d_rvec.begin(), 1);

            d_BLASWrapperPtr->xnrm2(
              d_xLocalDof, d_rvec.begin(), 1, mpi_communicator, &res);
            initial_res = res;

            if (res < absTolerance)
              conv = true;

            if (!conv)
              problem.tunePreconditionerForSolve(initial_res, absTolerance);

            if (!conv)
              {
                // z = M^{-1} r  (using d_uvec as z)
                problem.applyPreconditioner(d_uvec, d_rvec);

                // Batch local r·z and r·r in one device pass
                double localDotData[2] = {0.0, 0.0};
                dftfe::utils::deviceMemset(
                  d_localDotSums.data(), 0, 2 * sizeof(double));
                computeLocalDotRZAndRRDevice(d_rvec.begin(),
                                             d_uvec.begin(),
                                             d_localDotSums.data(),
                                             d_xLocalDof);
                dftfe::utils::MemoryTransfer<
                  dftfe::utils::MemorySpace::HOST,
                  dftfe::utils::MemorySpace::DEVICE>::copy(
                  2, localDotData, d_localDotSums.data());
                MPI_Allreduce(MPI_IN_PLACE,
                              localDotData,
                              2,
                              MPI_DOUBLE,
                              MPI_SUM,
                              mpi_communicator);
                double delta = localDotData[0];
                res          = std::sqrt(std::abs(localDotData[1]));

                // p = -z
                d_BLASWrapperPtr->axpby(d_xLocalDof,
                                        -1.0,
                                        d_uvec.begin(),
                                        0.0,
                                        d_pvec.begin());

                while ((!conv) && (it < maxNumberIterations))
                  {
                    it++;

                    // w = A p
                    problem.computeAX(d_wvec, d_pvec);

                    // pAp = p · w
                    double pAp = 0.0;
                    d_BLASWrapperPtr->xdot(d_xLocalDof,
                                           d_pvec.begin(),
                                           1,
                                           d_wvec.begin(),
                                           1,
                                           mpi_communicator,
                                           &pAp);

                    double alpha = delta / pAp;

                    // Fused: x += alpha*p, r += alpha*w, local r·r
                    dftfe::utils::deviceMemset(d_localRR.data(),
                                               0,
                                               sizeof(double));
                    updateXRandComputeLocalRRDevice(x.begin(),
                                                    d_rvec.begin(),
                                                    d_pvec.begin(),
                                                    d_wvec.begin(),
                                                    alpha,
                                                    d_localRR.data(),
                                                    d_xLocalDof);

                    // z = M^{-1} r
                    problem.applyPreconditioner(d_uvec, d_rvec);

                    // Batch delta_new = r·z and ||r||_2^2 = r·r into one allreduce
                    localDotData[0] = 0.0;
                    localDotData[1] = 0.0;
                    d_BLASWrapperPtr->xdot(d_xLocalDof,
                                           d_rvec.begin(),
                                           1,
                                           d_uvec.begin(),
                                           1,
                                           &localDotData[0]);
                    dftfe::utils::MemoryTransfer<
                      dftfe::utils::MemorySpace::HOST,
                      dftfe::utils::MemorySpace::DEVICE>::copy(
                      1, &localDotData[1], d_localRR.data());
                    MPI_Allreduce(MPI_IN_PLACE,
                                  localDotData,
                                  2,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  mpi_communicator);
                    double deltaNew = localDotData[0];
                    res             = std::sqrt(std::abs(localDotData[1]));
                    if (res < absTolerance)
                      {
                        conv = true;
                        break;
                      }

                    double beta = deltaNew / delta;
                    delta       = deltaNew;

                    // p = -z + beta * p
                    d_BLASWrapperPtr->axpby(d_xLocalDof,
                                            -1.0,
                                            d_uvec.begin(),
                                            beta,
                                            d_pvec.begin());
                  }
              }

            if (!conv)
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "DFT-FE Error: Solver did not converge\n"));
              }
          }
        else if (d_type == GMRES)
          {
            AssertThrow(false,
                        dealii::ExcMessage("DFT-FE Error: Not implemented"));
          }

        x.updateGhostValues();

        if (distributeFlag)
          problem.distributeX();

        problem.copyXfromDeviceToHost();
      }

    catch (...)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Poisson solver did not converge as per set tolerances. consider increasing MAXIMUM ITERATIONS in Poisson problem parameters. In rare cases for all-electron problems this can also occur due to a known parallel constraints issue in dealii library. Try using set CONSTRAINTS FROM SERIAL DOFHANDLER=true under the Boundary conditions subsection."));
        pcout
          << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
        pcout << "Current abs. residual in Device: " << res << std::endl;
      }

    if (debugLevel >= 2)
      {
        pcout << std::endl;
        pcout << "initial abs. residual in Device: " << initial_res
              << " , current abs. residual in Device: " << res
              << " , nsteps: " << it
              << " , abs. tolerance criterion in Device:  " << absTolerance
              << "\n\n";
        pcout << "total Device Poisson/Helmholtz operator matvecs: "
              << problem.getMatVecCount() << std::endl;
      }

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime() - time;

    if (debugLevel >= 4)
      pcout << "Time for Device Poisson/Helmholtz problem CG iterations: "
            << time << std::endl;
  }


} // namespace dftfe
