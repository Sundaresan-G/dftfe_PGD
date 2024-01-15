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
// @author Kartick Ramakrishnan
//

namespace dftfe
{
  template <typename ValueType>
  oncvClass<ValueType>::oncvClass(
    const MPI_Comm &                            mpi_comm_parent,
    const std::string &                         scratchFolderName,
    const std::set<unsigned int> &              atomTypes,
    const bool                                  floatingNuclearCharges,
    const unsigned int                          nOMPThreads,
    const std::map<unsigned int, unsigned int> &atomAttributes,
    const bool                                  reproducibleOutput,
    const int                                   verbosity,
    const bool                                  useDevice)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_dftfeScratchFolderName = scratchFolderName;
    d_atomTypes              = atomTypes;
    d_floatingNuclearCharges = floatingNuclearCharges;
    d_nOMPThreads            = nOMPThreads;
    d_reproducible_output    = reproducibleOutput;
    d_verbosity              = verbosity;
    d_atomTypeAtributes      = atomAttributes;
    d_useDevice              = useDevice;
  }

  template <typename ValueType>
  void
  oncvClass<ValueType>::createAtomCenteredSphericalFunctionsForDensities()
  {
    d_atomicCoreDensityMap.clear();
    d_atomicValenceDensityVector.clear();
    atomicValenceDensityMap.clear();
    d_atomicValenceDensityVector.resize(d_nOMPThreads);
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;
        char         valenceDataFile[256];
        strcpy(valenceDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/density.inp")
                 .c_str());
        char coreDataFile[256];
        strcpy(coreDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/coreDensity.inp")
                 .c_str());

        d_atomicCoreDensityMap[atomicNumber] =
          new AtomCenteredSphericalFunctionSpline(coreDataFile, 0, 0, 1, 2);
        atomicValenceDensityMap[atomicNumber] =
          new AtomCenteredSphericalFunctionSpline(valenceDataFile, 0, 0, 1, 2);
        double IntegralRho =
          atomicValenceDensityMap[atomicNumber]->getIntegralValue();
        double IntegralCoreRho =
          d_atomicCoreDensityMap[atomicNumber]->getIntegralValue();
        for (unsigned int i = 0; i < d_nOMPThreads; i++)
          d_atomicValenceDensityVector[i][*it] =
            new AtomCenteredSphericalFunctionSpline(
              valenceDataFile, 0, 0, 1, 2);
        if (IntegralCoreRho > 1E-8)
          d_atomTypeCoreFlagMap[atomicNumber] = true;
        else
          d_atomTypeCoreFlagMap[atomicNumber] = false;
      } //*it loop
  }



  template <typename ValueType>
  void
  oncvClass<ValueType>::initialise(
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
    unsigned int                            numEigenValues)
  {
    MPI_Barrier(d_mpiCommParent);
    d_BasisOperatorHostPtr = basisOperationsPtr;
    d_BLASWrapperHostPtr   = BLASWrapperPtrHost;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr = BLASWrapperPtrDevice;
#endif


    std::vector<unsigned int> atomicNumbers;
    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
      }

    d_densityQuadratureId           = densityQuadratureId;
    d_localContributionQuadratureId = localContributionQuadratureId;
    d_densityQuadratureIdElectro    = densityQuadratureIdElectro;
    d_sparsityPatternQuadratureId   = sparsityPatternQuadratureId;
    d_nlpspQuadratureId             = nlpspQuadratureId;
    d_excManagerPtr                 = excFunctionalPtr;
    d_numEigenValues                = numEigenValues;

    createAtomCenteredSphericalFunctionsForDensities();
    createAtomCenteredSphericalFunctionsForProjectors();
    createAtomCenteredSphericalFunctionsForLocalPotential();

    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(atomicNumbers, d_atomicProjectorFnsMap);

    d_nonLocalOperatorHost = std::make_shared<
      AtomicCenteredNonLocalOperator<ValueType,
                                     dftfe::utils::MemorySpace::HOST>>(
      d_BLASWrapperHostPtr,
      d_atomicProjectorFnsContainer,
      d_numEigenValues,
      d_mpiCommParent);
#if defined(DFTFE_WITH_DEVICE)
    if (d_useDevice)
      d_nonLocalOperatorDevice = std::make_shared<
        AtomicCenteredNonLocalOperator<ValueType,
                                       dftfe::utils::MemorySpace::DEVICE>>(
        d_BLASWrapperDevicePtr,
        d_atomicProjectorFnsContainer,
        d_numEigenValues,
        d_mpiCommParent);
#endif
    computeNonlocalPseudoPotentialConstants();
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double> &             kPointWeights,
    const std::vector<double> &             kPointCoordinates,
    const bool                              updateNonlocalSparsity)
  {
    std::vector<unsigned int> atomicNumbers;
    std::vector<double>       atomCoords;


    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }


    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);


    if (updateNonlocalSparsity)
      {
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        d_nonLocalOperatorHost->InitalisePartitioner(d_BasisOperatorHostPtr);
        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "ONCVclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    d_nonLocalOperatorHost->initKpoints(kPointWeights, kPointCoordinates);
    d_nonLocalOperatorHost->computeCMatrixEntries(d_BasisOperatorHostPtr,
                                                  d_nlpspQuadratureId,
                                                  d_BLASWrapperHostPtr);
#if defined(DFTFE_WITH_DEVICE)
    if (d_useDevice)
      {
        d_nonLocalOperatorDevice->initKpoints(kPointWeights, kPointCoordinates);
        d_nonLocalOperatorDevice->transferCMatrixEntriesfromHostObject(
          d_nonLocalOperatorHost, d_BasisOperatorHostPtr);
      }
#endif
    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "ONCVclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
  }

  template <typename ValueType>
  void
  oncvClass<ValueType>::computeNonlocalPseudoPotentialConstants()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        const unsigned int Zno = *it;
        const std::map<std::pair<unsigned int, unsigned int>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        unsigned int numRadProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Zno);
        unsigned int numTotalProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
        char denominatorDataFileName[256];
        strcpy(denominatorDataFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(Zno) + "/" +
                "denom.dat")
                 .c_str());
        std::vector<std::vector<double>> denominator(0);
        dftUtils::readFile(numRadProjectors,
                           denominator,
                           denominatorDataFileName);
        std::vector<double> pseudoPotentialConstants(numTotalProjectors, 0.0);
        unsigned int        ProjId = 0;
        for (unsigned int iProj = 0; iProj < numRadProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(Zno, iProj))->second;
            unsigned int lQuantumNumber = sphFn->getQuantumNumberl();
            for (int l = 0; l < 2 * lQuantumNumber + 1; l++)
              {
                pseudoPotentialConstants[ProjId] = denominator[iProj][iProj];
                ProjId++;
              }
          }
        d_atomicNonLocalPseudoPotentialConstants[Zno] =
          pseudoPotentialConstants;
        d_nonlocalHamiltonianEntriesUpdated = false;
      } //*it
  }



  template <typename ValueType>
  void
  oncvClass<ValueType>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    d_atomicProjectorFnsVector.clear();
    std::vector<std::vector<int>> projectorIdDetails;
    std::vector<std::vector<int>> atomicFunctionIdDetails;
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char         pseudoAtomDataFile[256];
        unsigned int cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        unsigned int  Zno = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        unsigned int  numberOfProjectors;
        readPseudoDataFileNames >> numberOfProjectors;
        readPseudoDataFileNames.ignore();
        projectorIdDetails.resize(numberOfProjectors);
        std::string   readLine;
        std::set<int> radFunctionIds;
        atomicFunctionIdDetails.resize(numberOfProjectors);
        for (unsigned int i = 0; i < numberOfProjectors; ++i)
          {
            std::vector<int> &radAndAngularFunctionId =
              atomicFunctionIdDetails[i];
            radAndAngularFunctionId.resize(3, 0);
            std::getline(readPseudoDataFileNames, readLine);

            std::istringstream lineString(readLine);
            unsigned int       count = 0;
            int                Id;
            double             mollifierRadius;
            std::string        dummyString;
            while (lineString >> dummyString)
              {
                if (count < 3)
                  {
                    Id = atoi(dummyString.c_str());

                    if (count == 1)
                      radFunctionIds.insert(Id);
                    radAndAngularFunctionId[count] = Id;
                  }

                if (count > 3)
                  {
                    std::cerr << "Invalid argument in the SingleAtomData file"
                              << std::endl;
                    exit(-1);
                  }

                count++;
              }
          }
        std::string  tempProjRadialFunctionFileName;
        unsigned int numProj;
        unsigned int alpha = 0;
        for (std::set<int>::iterator i = radFunctionIds.begin();
             i != radFunctionIds.end();
             ++i)
          {
            char         projRadialFunctionFileName[512];
            unsigned int lQuantumNo = *i;
            readPseudoDataFileNames >> tempProjRadialFunctionFileName;
            readPseudoDataFileNames >> numProj;
            strcpy(projRadialFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/" + tempProjRadialFunctionFileName)
                     .c_str());

            //
            // 2D vector to store the radial coordinate and its
            // corresponding function value
            std::vector<std::vector<double>> radialFunctionData(0);

            //
            // read the radial function file
            //
            dftUtils::readFile(numProj + 1,
                               radialFunctionData,
                               projRadialFunctionFileName);

            for (int j = 1; j < numProj + 1; j++)
              {
                d_atomicProjectorFnsMap[std::make_pair(Zno, alpha)] =
                  std::make_shared<AtomCenteredSphericalFunctionSpline>(
                    projRadialFunctionFileName, lQuantumNo, 0, j, numProj + 1);
                alpha++;
              }
          } // i loop

      } // for loop *it
  }



  template <typename ValueType>
  void
  oncvClass<ValueType>::createAtomCenteredSphericalFunctionsForLocalPotential()
  {
    d_atomicLocalPotVector.clear();
    d_atomicLocalPotVector.resize(d_nOMPThreads);

    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        unsigned int atomicNumber = *it;
        char         LocalDataFile[256];
        strcpy(LocalDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/locPot.dat")
                 .c_str());
        for (unsigned int i = 0; i < d_nOMPThreads; i++)
          d_atomicLocalPotVector[i][*it] =
            new AtomCenteredSphericalFunctionSpline(
              LocalDataFile,
              0,
              false,
              1,
              false,
              true,
              d_reproducible_output ? 8.0001 : 10.0001,
              d_reproducible_output ? 1.0e-8 : 1.0e-7,
              d_atomTypeAtributes[*it],
              -1);

      } //*it loop
  }
  template <typename ValueType>
  double
  oncvClass<ValueType>::getRadialValenceDensity(unsigned int Zno, double rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value =
      d_atomicValenceDensityVector[threadId][Zno]->getRadialValue(rad);

    return (Value);
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::getRadialValenceDensity(unsigned int         Zno,
                                                double               rad,
                                                std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicValenceDensityVector[threadId][Zno]->getDerivativeValue(rad);
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRmaxValenceDensity(unsigned int Zno)
  {
    return (d_atomicValenceDensityVector[0][Zno]->getRadialCutOff());
  }
  template <typename ValueType>
  double
  oncvClass<ValueType>::getRmaxCoreDensity(unsigned int Zno)
  {
    return (d_atomicCoreDensityMap[Zno]->getRadialCutOff());
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRadialCoreDensity(unsigned int Zno, double rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value    = d_atomicCoreDensityMap[Zno]->getRadialValue(rad);
    return (Value);
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::getRadialCoreDensity(unsigned int         Zno,
                                             double               rad,
                                             std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicCoreDensityMap[Zno]->getDerivativeValue(rad);
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRadialLocalPseudo(unsigned int Zno, double rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double Value = d_atomicLocalPotVector[threadId][Zno]->getRadialValue(rad);
    return (Value);
  }
  template <typename ValueType>
  double
  oncvClass<ValueType>::getRmaxLocalPot(unsigned int Zno)
  {
    return (d_atomicLocalPotVector[0][Zno]->getRadialCutOff());
  }
  template <typename ValueType>
  bool
  oncvClass<ValueType>::coreNuclearDensityPresent(unsigned int Zno)
  {
    return (d_atomTypeCoreFlagMap[Zno]);
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::setImageCoordinates(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    std::vector<unsigned int> &             imageIdsTemp,
    std::vector<double> &                   imageCoordsTemp)
  {
    imageIdsTemp.clear();
    imageCoordsTemp.clear();
    imageCoordsTemp.resize(imageIds.size() * 3, 0.0);
    std::vector<unsigned int> imageLoc(int(atomLocations.size()), 0.0);
    for (int jImage = 0; jImage < imageIds.size(); jImage++)
      {
        unsigned int atomId = (imageIds[jImage]);
        imageIdsTemp.push_back(atomId);
        int startLoc = imageLoc[atomId];
        imageCoordsTemp[3 * jImage + 0] =
          periodicCoords[atomId][3 * startLoc + 0];
        imageCoordsTemp[3 * jImage + 1] =
          periodicCoords[atomId][3 * startLoc + 1];
        imageCoordsTemp[3 * jImage + 2] =
          periodicCoords[atomId][3 * startLoc + 2];
        imageLoc[atomId] += 1;
      }
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::applynonLocalHamiltonianMatrix(
    const dftfe::linearAlgebra::MultiVector<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
      &sphericalFunctionKetTimesVectorParFlattened,
    std::map<
      unsigned int,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
      &shapeFnTimesWavefunctionMatrix)
  {
    if (!d_nonlocalHamiltonianEntriesUpdated)
      {
        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<unsigned int> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        d_nonLocalHamiltonianEntriesHost.clear();
        std::vector<double> Entries;
        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            unsigned int atomId = atomIdsInProcessor[iAtom];
            unsigned int Zno    = atomicNumber[atomId];
            unsigned int numberSphericalFunctions =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
            for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                double V = d_atomicNonLocalPseudoPotentialConstants[Zno][alpha];
                Entries.push_back(V);
              }
          }
        d_nonLocalHamiltonianEntriesHost.resize(Entries.size());
        d_nonLocalHamiltonianEntriesHost.copyFrom(Entries);
      }


    if (!d_useDevice)
      {
        d_nonLocalOperatorHost->applyV_onCTX(
          CouplingStructure::diagonal,
          d_nonLocalHamiltonianEntriesHost,
          sphericalFunctionKetTimesVectorParFlattened,
          shapeFnTimesWavefunctionMatrix);
      }
    else
      {
        // d_nonLocalOperatorDevice->applyV_onCTX(CouplingStructure::diagonal,d_nonLocalHamiltonianEntriesDevice);
      }
  }

} // namespace dftfe
