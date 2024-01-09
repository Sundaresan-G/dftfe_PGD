// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author  Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ONCVCLASS_H
#define DFTFE_ONCVCLASS_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionBase.h"
#include "AtomCenteredSphericalFunctionBessel.h"
#include "AtomCenteredSphericalFunctionSinc.h"
#include "AtomCenteredSphericalFunctionGaussian.h"
#include "AtomCenteredSphericalFunctionSpline.h"
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
  template <typename ValueType>
  class oncvClass
  {
  public:
    oncvClass(const MPI_Comm &              mpi_comm_parent,
              const std::string &           scratchFolderName,
              const std::set<unsigned int> &atomTypes,
              const bool                    floatingNuclearCharges,
              const unsigned int            nOMPThreads,
              const std::map<unsigned int, unsigned int> &atomAttributes,
              const bool                                  reproducibleOutput,
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

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      unsigned int                            densityQuadratureId,
      unsigned int                            localContributionQuadratureId,
      unsigned int                            sparsityPatternQuadratureId,
      unsigned int                            nlpspQuadratureId,
      unsigned int                            densityQuadratureIdElectro,
      excManager *                            excFunctionalPtr,
      const std::vector<std::vector<double>> &atomLocations,
      unsigned int                            numEigenValues);

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


    /**
     * @brief Initialises local potential
     */
    void
    initLocalPotential();

    void
    getRadialValenceDensity(unsigned int         Zno,
                            double               rad,
                            std::vector<double> &Val);

    double
    getRadialValenceDensity(unsigned int Zno, double rad);

    double
    getRmaxValenceDensity(unsigned int Zno);

    void
    getRadialCoreDensity(unsigned int         Zno,
                         double               rad,
                         std::vector<double> &Val);

    double
    getRadialCoreDensity(unsigned int Zno, double rad);

    double
    getRmaxCoreDensity(unsigned int Zno);

    double
    getRadialLocalPseudo(unsigned int Zno, double rad);

    double
    getRmaxLocalPot(unsigned int Zno);

    bool
    coreNuclearDensityPresent(unsigned int Zno);

    // Creating Object for Atom Centerd Nonlocal Operator
    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number,
                                     dftfe::utils::MemorySpace::HOST>>
      d_nonLocalOperatorHost;
    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number,
                                     dftfe::utils::MemorySpace::DEVICE>>
      d_nonLocalOperatorDevice;

  private:
    /**
     * @brief Converts the periodic image data structure to relevant form for the container class
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     * @param[out] imageIdsTemp image IDs of periodic cell
     * @param[out] imageCoordsTemp coordinates of image atoms
     */
    void
    setImageCoordinates(const std::vector<std::vector<double>> &atomLocations,
                        const std::vector<int> &                imageIds,
                        const std::vector<std::vector<double>> &periodicCoords,
                        std::vector<unsigned int> &             imageIdsTemp,
                        std::vector<double> &imageCoordsTemp);
    /**
     * @brief Creating Density splines for all atomTypes
     */
    void
    createAtomCenteredSphericalFunctionsForDensities();

    void
    computeNonlocalPseudoPotentialConstants();
    void
    createAtomCenteredSphericalFunctionsForProjectors();
    void
    createAtomCenteredSphericalFunctionsForLocalPotential();

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                     d_BLASWrapperDevicePtr;
    std::vector<std::vector<double>> d_nonLocalPseudoPotentialConstants;
    std::map<unsigned int, std::vector<double>>
      d_atomicNonLocalPseudoPotentialConstants;
    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicWaveFnsVector;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    // parallel communication objects
    const MPI_Comm     d_mpiCommParent;
    const unsigned int d_this_mpi_process;

    // conditional stream object
    dealii::ConditionalOStream pcout;
    bool                       d_useDevice;
    unsigned int               d_densityQuadratureId;
    unsigned int               d_localContributionQuadratureId;
    unsigned int               d_nuclearChargeQuadratureIdElectro;
    unsigned int               d_densityQuadratureIdElectro;
    unsigned int               d_sparsityPatternQuadratureId;
    unsigned int               d_nlpspQuadratureId;
    excManager *               d_excManagerPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;



    std::map<unsigned int, bool>                      d_atomTypeCoreFlagMap;
    bool                                              d_floatingNuclearCharges;
    int                                               d_verbosity;
    std::vector<std::vector<double>>                  d_atomLocations;
    std::set<unsigned int>                            d_atomTypes;
    std::map<unsigned int, std::vector<unsigned int>> d_atomTypesList;
    std::string                                       d_dftfeScratchFolderName;
    std::vector<int>                                  d_imageIds;
    std::vector<std::vector<double>>                  d_imagePositions;
    unsigned int                                      d_numEigenValues;
    unsigned int                                      d_nOMPThreads;

    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsVector;
    std::vector<std::map<unsigned int, AtomCenteredSphericalFunctionBase *>>
      d_atomicLocalPotVector;
    std::vector<std::map<unsigned int, AtomCenteredSphericalFunctionBase *>>
      d_atomicValenceDensityVector;
    std::map<unsigned int, AtomCenteredSphericalFunctionBase *>
      d_atomicCoreDensityMap;
    std::map<unsigned int, AtomCenteredSphericalFunctionBase *>
         atomicValenceDensityMap;
    bool d_reproducible_output;
    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;



  }; // end of class

} // end of namespace dftfe
#include "../src/pseudo/oncvClass.t.cc"
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
