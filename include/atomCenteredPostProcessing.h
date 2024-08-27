#ifndef ATOMCENTEREDOORBTIALPOSTPROCESING_H
#define ATOMCENTEREDOORBTIALPOSTPROCESING_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <excManager.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class atomCenteredOrbitalsPostProcessing
  {
  public:
    atomCenteredOrbitalsPostProcessing(
      const MPI_Comm &                            mpi_comm_parent,
      const std::string &                         scratchFolderName,
      const std::set<unsigned int> &              atomTypes,
      const bool                                  floatingNuclearCharges,
      const unsigned int                          nOMPThreads,
      const std::map<unsigned int, unsigned int> &atomAttributes,
      const bool                                  reproducibleOutput,
      const int                                   verbosity,
      const bool                                  useDevice);
    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */

    double
    smearFunction(double x, const dftParameters *dftParamsPtr);

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        basisOperationsDevicePtr,
#endif
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      unsigned int                             densityQuadratureId,
      unsigned int                             localContributionQuadratureId,
      unsigned int                             sparsityPatternQuadratureId,
      unsigned int                             nlpspQuadratureId,
      unsigned int                             densityQuadratureIdElectro,
      std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
      const std::vector<std::vector<double>> & atomLocations,
      unsigned int                             numEigenValues,
      const bool                               singlePrecNonLocalOperator);

    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] bQuadValuesAllAtoms address of nuclear charge field
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */
    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double> &             kPointWeights,
      const std::vector<double> &             kPointCoordinates,
      const bool                              updateNonlocalSparsity);

    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    void
    computeAtomCenteredEntries(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
      const unsigned int                      totalNumWaveFunctions,
      const std::vector<std::vector<double>> &eigenValues,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        &basisOperationsPtr,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                 BLASWrapperPtr,
      const unsigned int         quadratureIndex,
      const std::vector<double> &kPointWeights,
      const MPI_Comm &           interBandGroupComm,
      const MPI_Comm &           interpoolComm,
      const dftParameters *      dftParamsPtr,
      double                     fermiEnergy,
      unsigned int               highestStateNscfSolve);

  private:
    const MPI_Comm                       d_mpiCommParent;
    const unsigned int                   d_this_mpi_process;
    std::string                          d_dftfeScratchFolderName;
    std::set<unsigned int>               d_atomTypes;
    bool                                 d_floatingNuclearCharges;
    unsigned int                         d_nOMPThreads;
    bool                                 d_reproducible_output;
    unsigned int                         d_verbosity;
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;
    bool                                 d_useDevice;
    dealii::ConditionalOStream           pcout;
    bool                                 d_singlePrecNonLocalOperator;
    unsigned int                         d_sparsityPatternQuadratureId;
    unsigned int                         d_nlpspQuadratureId;
    unsigned int                         d_densityQuadratureId;
    unsigned int                         d_localContributionQuadratureId;
    unsigned int                         d_nuclearChargeQuadratureIdElectro;
    unsigned int                         d_densityQuadratureIdElectro;
    unsigned int                         d_numEigenValues;

    void
    createAtomCenteredSphericalFunctionsForOrbitals();


    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicOrbitalFnsContainer;

    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicOrbitalFnsMap;

    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;

    std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_nonLocalOperatorSinglePrec;

    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;


    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperDevicePtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorDevicePtr;
#endif
  };


} // namespace dftfe

#endif
