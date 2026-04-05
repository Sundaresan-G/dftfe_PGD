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


#ifndef poissonSolverProblem_H_
#define poissonSolverProblem_H_

#include <dealiiLinearSolverProblem.h>
#include <constraintMatrixInfo.h>
#include "FEBasisOperations.h"

namespace dftfe
{
  /**
   * @brief poisson solver problem class template. template parameter FEOrderElectro
   * is the finite element polynomial order.
   *
   * @author Shiva Rudraraju, Phani Motamarri, Sambit Das
   */
  template <dftfe::uInt FEOrderElectro>
  class poissonSolverProblem : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    poissonSolverProblem(const MPI_Comm &mpi_comm);


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
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedCPUVec<double> &
    getX();

    /**
     * @brief const overload to access x field in debug/const contexts.
     */
    const distributedCPUVec<double> &
    getX() const;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x);

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief Jacobi preconditioning.
     *
     */
    void
    precondition_Jacobi(distributedCPUVec<double>       &dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const;

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    void
    distributeX();

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    void
    subscribe(std::atomic<bool> *const validity,
              const std::string       &identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    void
    unsubscribe(std::atomic<bool> *const validity,
                const std::string       &identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix
    bool
    operator!=(double val) const
    {
      return true;
    };

    /**
     * @brief Apply Chebyshev-Jacobi preconditioner: dst ≈ A^{-1} src.
     * Overrides the base class default Jacobi with Chebyshev polynomial
     * acceleration.
     */
    void
    applyPreconditioner(distributedCPUVec<double>       &dst,
                        const distributedCPUVec<double> &src) override;

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
     * @brief required for the cell_loop operation in dealii's MatrixFree class
     *
     */
    void
    AX(const dealii::MatrixFree<3, double>       &matrixFreeData,
       distributedCPUVec<double>                 &dst,
       const distributedCPUVec<double>           &src,
       const std::pair<dftfe::uInt, dftfe::uInt> &cell_range) const;


    /**
     * @brief Compute the diagonal of A.
     *
     */
    void
    computeDiagonalA();

    /**
     * @brief Estimate spectral bounds of D^{-1}A using Lanczos iteration.
     */
    void
    computeSpectralBounds();

    /**
     * @brief Project out constant mode in D-inner product (fully periodic).
     */
    void
    projectOutConstantMode(distributedCPUVec<double> &vec) const;

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
    distributedCPUVec<double> d_diagonalA;

    /// storage for smeared charge rhs in case of total potential solve (doesn't
    /// change every scf)
    distributedCPUVec<double> d_rhsSmearedCharge;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to the x vector being solved for
    distributedCPUVec<double> *d_xPtr;

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
    dftfe::uInt d_meanValueConstraintNodeId;

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
    bool d_isFastConstraintsInitialized;

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
    distributedCPUVec<double> d_chebyWorkVec1;
    distributedCPUVec<double> d_chebyWorkVec2;

    /// Un-inverted diagonal of A (needed for Lanczos D-inner product)
    distributedCPUVec<double> d_diagonalARaw;

    /// Projection weight for constant-mode deflation: 1/(1^T D 1)
    double d_projWeight;

    /// FE mass-lumped weights a_i = int N_i dx (for integral-mean gauge)
    distributedCPUVec<double> d_meanValueWeights;
    double                    d_domainVolume;

    const MPI_Comm             mpi_communicator;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // poissonSolverProblem_H_
