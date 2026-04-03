// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef poissonSolverProblemDevice_H_
#    define poissonSolverProblemDevice_H_

#    include <linearSolverProblemDevice.h>
#    include <constraintMatrixInfo.h>
#    include <constants.h>
#    include <dftUtils.h>
#    include <headers.h>
#    include "FEBasisOperations.h"
#    include "BLASWrapper.h"
#    include "MatrixFreeWrapper.h"
#    include <DeviceAPICalls.h>

namespace dftfe
{
  /**
   * @brief poisson solver problem device class template. template parameter FEOrderElectro
   * is the finite element polynomial order. The class should not be used with
   * FLOATING NUCLEAR CHARGES = false or POINT WISE DIRICHLET CONSTRAINT = true
   *
   * @author Gourab Panigrahi
   */
  template <dftfe::uInt FEOrderElectro>
  class poissonSolverProblemDevice : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    poissonSolverProblemDevice(const MPI_Comm &mpi_comm);

    /**
     * @brief clears all datamembers and reset to original state.
     *
     *
     */
    void
    clear();

    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(
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
      const bool        isComputeDiagonalA               = true,
      const bool        isComputeMeanValueConstraints    = false,
      const bool        smearedNuclearCharges            = false,
      const bool        isRhoValues                      = true,
      const bool        isGradSmearedChargeRhs           = false,
      const dftfe::uInt smearedChargeGradientComponentId = 0,
      const bool        storeSmearedChargeRhs            = false,
      const bool        reuseSmearedChargeRhs            = false,
      const bool        reinitializeFastConstraints      = false);

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x);

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedDeviceVec<double> &
    getX();

    /**
     * @brief get the reference to Preconditioner
     *
     * @return reference to Preconditioner
     */
    distributedDeviceVec<double> &
    getPreconditioner();

    /**
     * @brief Copies x from Device to Host
     *
     */
    void
    copyXfromDeviceToHost();

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    void
    distributeX();


    void
    setX();

    /**
     * @brief Apply Chebyshev-Jacobi preconditioner: dst ≈ A^{-1} src.
     * Overrides the base class default Jacobi with a Chebyshev polynomial
     * accelerated iteration. Spectral bounds and work vectors are cached.
     */
    void
    applyPreconditioner(distributedDeviceVec<double> &dst,
                        distributedDeviceVec<double> &src) override;

    /**
     * @brief Returns true since this class uses Chebyshev-Jacobi.
     */
    bool
    usesCustomPreconditioner() const override;

    /**
     * @brief Configure Poisson preconditioner style and Chebyshev degree.
     *
     * @param useChebyshev If true, use Chebyshev-Jacobi path where applicable.
     * @param chebyDegree Polynomial degree for Chebyshev-Jacobi preconditioner.
     */
    void
    setPreconditionerOptions(const bool        useChebyshev,
                             const dftfe::uInt chebyDegree);

    void
    resetMatVecCount() override;

    dftfe::uInt
    getMatVecCount() const override;

    void
    tunePreconditionerForSolve(const double initialResidual,
                               const double absTolerance) override;


  private:
    /**
     * @brief Sets up the constraints matrix
     *
     */
    void
    setupConstraints();

    /**
     * @brief Compute the diagonal of A.
     *
     */
    void
    computeDiagonalA();

    /**
     * @brief Estimate spectral bounds of D^{-1}A using Lanczos iteration.
     * Results are cached in d_chebyLambdaMin and d_chebyLambdaMax.
     */
    void
    computeSpectralBounds();

    /**
     * @brief Project out constant mode in D-inner product (fully periodic).
     */
    void
    projectOutConstantMode(distributedDeviceVec<double> &vec);

    /**
     * @brief Tune active Chebyshev degree from estimated condition number.
     *
     * Uses Lanczos-estimated spectral bounds of D^{-1}A to choose an effective
     * polynomial degree for the current solve while keeping the configured
     * degree as the baseline.
     */
    void
    tuneChebyshevDegreeFromSpectrum();

    /**
     * @brief Compute mean value constraint which is required in case of fully periodic
     * boundary conditions.
     *
     */
    void
    computeMeanValueConstraint();

    /// storage for diagonal of the A matrix
    distributedCPUVec<double>    d_diagonalA;
    distributedDeviceVec<double> d_diagonalAdevice;

    /// storage for smeared charge rhs in case of total potential solve (doesn't
    /// change every scf)
    distributedCPUVec<double> d_rhsSmearedCharge;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to the x vector being solved for
    distributedCPUVec<double>   *d_xPtr;
    distributedDeviceVec<double> d_xDevice;

    // number of cells local to each mpi task, number of degrees of freedom
    // locally owned and total degrees of freedom including ghost
    dftfe::Int d_nLocalCells, d_xLocalDof, d_xLen;

    // Matrix free wrapper object
    std::unique_ptr<
      dftfe::MatrixFreeWrapperClass<double,
                                    dftfe::operatorList::Laplace,
                                    dftfe::utils::MemorySpace::DEVICE,
                                    false>>
      d_matrixFreeWrapperDevice;

    // constraints
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
      d_inhomogenousConstraintsTotalPotentialInfo;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    /// matrix free index required to access the DofHandler and
    /// dealii::AffineConstraints<double> objects corresponding to the problem
    dftfe::uInt d_matrixFreeVectorComponent;

    /// matrix free quadrature index
    dftfe::uInt d_matrixFreeQuadratureComponentRhsDensity;

    /// matrix free quadrature index
    dftfe::uInt d_matrixFreeQuadratureComponentAX;

    /// pointer to electron density cell quadrature data
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      *d_rhoValuesPtr;

    /// pointer to smeared charge cell quadrature data
    const std::map<dealii::CellId, std::vector<double>>
      *d_smearedChargeValuesPtr;

    ///
    dftfe::uInt d_smearedChargeQuadratureId;

    /// pointer to map between global dof index in current processor and the
    /// atomic charge on that dof
    const std::map<dealii::types::global_dof_index, double> *d_atomsPtr;

    /// storage for mean value constraint vector
    distributedCPUVec<double> d_meanValueConstraintVec;

    /// boolean flag to query if mean value constraint datastructures are
    /// precomputed
    bool d_isMeanValueConstraintComputed;

    ///
    bool d_isGradSmearedChargeRhs;

    ///
    bool d_isStoreSmearedChargeRhs;

    ///
    bool d_isReuseSmearedChargeRhs;

    ///
    dftfe::uInt d_smearedChargeGradientComponentId;

    /// mean value constraints: mean value constrained node
    dealii::types::global_dof_index d_meanValueConstraintNodeId;

    /// mean value constraints: constrained proc id containing the mean value
    /// constrained node
    dftfe::uInt d_meanValueConstraintProcId;

    /// duplicate constraints object with flattened maps for faster access
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      d_constraintsInfo;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtr;
    ///
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
         d_BLASWrapperPtr;
    bool d_isFastConstraintsInitialized;
    bool d_isHomogenousConstraintsInitialized;

    /// Chebyshev-Jacobi preconditioner: spectral bounds of D^{-1}A
    double      d_chebyLambdaMax;
    double      d_chebyLambdaMin;
    bool        d_isSpectrumComputed;
    bool        d_useChebyshevPreconditioner;
    dftfe::uInt d_chebyDegree;
    dftfe::uInt d_chebyDegreeConfigured;
    dftfe::uInt d_matVecCount;

    /// Cached primitive timings for degree selection model.
    bool   d_arePrimitiveTimesCached;
    double d_cachedMatvecTime;
    double d_cachedAllreduceTime;

    /// Chebyshev preconditioner work vectors (allocated once, reused)
    distributedDeviceVec<double> d_chebyWorkVec1;
    distributedDeviceVec<double> d_chebyWorkVec2;
    bool                         d_areChebyWorkVecsInitialized;

    /// Un-inverted diagonal of A (needed for Lanczos D-inner product)
    distributedDeviceVec<double> d_diagonalARawDevice;

    /// Host copy of un-inverted diagonal (needed for RHS projection)
    distributedCPUVec<double> d_diagonalARawHost;

    /// Projection weight for constant-mode deflation: 1/(1^T D 1)
    double d_projWeight;

    /// Ones vector on device for constant-mode projection
    distributedDeviceVec<double> d_onesDevice;

    /// FE mass-lumped weights a_i = int N_i dx (for integral-mean gauge)
    distributedDeviceVec<double> d_meanValueWeightsDevice;
    double                       d_domainVolume;

    const MPI_Comm             mpi_communicator;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#  endif // poissonSolverProblemDevice_H_
#endif
