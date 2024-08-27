#include <atomCenteredPostProcessing.h>
#include <deviceKernelsGeneric.h>
#include <constants.h>
#include <cassert>

namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>

  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    atomCenteredOrbitalsPostProcessing(
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

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::smearFunction(
    double               x,
    const dftParameters *dftParamsPtr)
  {
    double sigma = C_kb * dftParamsPtr->smearTval;
    double arg = std::min((x / sigma) * (x / sigma), 200.0);
    return (std::exp(-arg) / sqrt(M_PI)) / (sigma * C_haToeV);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::initialise(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
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
    unsigned int                            densityQuadratureId,
    unsigned int                            localContributionQuadratureId,
    unsigned int                            sparsityPatternQuadratureId,
    unsigned int                            nlpspQuadratureId,
    unsigned int                            densityQuadratureIdElectro,
    std::shared_ptr<excManager>             excFunctionalPtr,
    const std::vector<std::vector<double>> &atomLocations,
    unsigned int                            numEigenValues,
    const bool                              singlePrecNonLocalOperator)
  {
    MPI_Barrier(d_mpiCommParent);
    d_BasisOperatorHostPtr = basisOperationsHostPtr;
    d_BLASWrapperHostPtr   = BLASWrapperPtrHost;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr   = BLASWrapperPtrDevice;
    d_BasisOperatorDevicePtr = basisOperationsDevicePtr;
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
    // d_excManagerPtr                  = excFunctionalPtr;
    d_numEigenValues             = numEigenValues;
    d_singlePrecNonLocalOperator = singlePrecNonLocalOperator;

    createAtomCenteredSphericalFunctionsForOrbitals();

    d_atomicOrbitalFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicOrbitalFnsContainer->init(atomicNumbers, d_atomicOrbitalFnsMap);

    // understand the below part of d_useDevice

    if (!d_useDevice)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperHostPtr,
            d_BasisOperatorHostPtr,
            d_atomicOrbitalFnsContainer,
            d_mpiCommParent);
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperHostPtr,
                              d_BasisOperatorHostPtr,
                              d_atomicOrbitalFnsContainer,
                              d_mpiCommParent);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperDevicePtr,
            d_BasisOperatorDevicePtr,
            d_atomicOrbitalFnsContainer,
            d_mpiCommParent);
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperDevicePtr,
                              d_BasisOperatorDevicePtr,
                              d_atomicOrbitalFnsContainer,
                              d_mpiCommParent);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    initialiseNonLocalContribution(
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

    d_atomicOrbitalFnsContainer->initaliseCoordinates(atomCoords,
                                                      periodicCoords,
                                                      imageIds);

    if (updateNonlocalSparsity) // what is updateNonlocalSparsity?
      {
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicOrbitalFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "atomCenteredorbitalPostProcessing: Time taken for computeSparseStructure: "
            << TotalTime << std::endl;
      }
    // What does the other functions listed inside the
    // below function do?

    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_nlpspQuadratureId);

    if (d_singlePrecNonLocalOperator)
      d_nonLocalOperatorSinglePrec
        ->intitialisePartitionerKPointsAndComputeCMatrixEntries(
          updateNonlocalSparsity,
          kPointWeights,
          kPointCoordinates,
          d_BasisOperatorHostPtr,
          d_nlpspQuadratureId);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  atomCenteredOrbitalsPostProcessing<ValueType,
                                     memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    createAtomCenteredSphericalFunctionsForOrbitals()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         it++)
      {
        char pseudoAtomDataFile[256];
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());
        unsigned int  Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        std::set<std::pair<int, int>> radFunctionIds;
        std::string                   readLine;
        char                          waveFunctionFileName[512];

        if (!readPseudoDataFileNames.is_open())
          {
            std::cerr << "Failed to open file: " << pseudoAtomDataFile
                      << std::endl;
            exit(-1);
          }

        while (std::getline(readPseudoDataFileNames, readLine))
          {
            if (readLine.find("psi") != std::string::npos)
              {
                // This line contains "psi"

                std::string nlPart =
                  readLine.substr(3,
                                  readLine.size() - 7); // "20" from "psi20.inp"
                // Assuming the format is always psiNL.inp, where N and L are
                // single digits
                int nQuantumNumber = std::atoi(
                  nlPart.substr(0, 1).c_str()); // Extracts '2' from "20"
                int lQuantumNumber = std::atoi(
                  nlPart.substr(1, 1).c_str()); // Extracts '0' from "20"
                radFunctionIds.insert(
                  std::make_pair(nQuantumNumber, lQuantumNumber));
              }
          }
        readPseudoDataFileNames.close();
        unsigned int alpha = 0;
        for (std::set<std::pair<int, int>>::iterator i = radFunctionIds.begin();
             i != radFunctionIds.end();
             ++i)
          {
            int nQuantumNumber = i->first;
            int lQuantumNumber = i->second;
            strcpy(waveFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/psi" + std::to_string(nQuantumNumber) +
                    std::to_string(lQuantumNumber) + ".inp")
                     .c_str());
            d_atomicOrbitalFnsMap[std::make_pair(Znum, alpha)] =
              std::make_shared<AtomCenteredSphericalFunctionProjectorSpline>(
                waveFunctionFileName,
                lQuantumNumber,
                -1, // radial power
                1,
                2,
                1E-12); // we should pass the radial power as a parameter to the
                        // user?
            alpha++;
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
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
      unsigned int               highestStateNscfSolve)
  {
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    unsigned int       numSpinComponents;
    numSpinComponents               = dftParamsPtr->spinPolarized + 1;
    const unsigned int numLocalDofs = basisOperationsPtr->nOwnedDofs();

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    const unsigned int numNodesPerElement = basisOperationsPtr->nDofsPerCell();

    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    const unsigned int        bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    unsigned int BVec = std::min(dftParamsPtr->chebyWfcBlockSize,
                                 bandGroupLowHighPlusOneIndices[1]);

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlock;

    unsigned int previousSize = 0;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> resVec;

    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;

    std::vector<double> eigenValuesBlock;
    std::vector<double> smearedComponentBlock;

    int indexFermiEnergy = -1.0;
    for (int i = 0; i < totalNumWaveFunctions; ++i)
      if (eigenValues[0][i] >= fermiEnergy)
        {
          if (i > indexFermiEnergy)
            {
              indexFermiEnergy = i;
              break;
            }
        }
    std::vector<double> eigenValuesAllkPoints;

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter <= highestStateNscfSolve;
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(eigenValues[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    const double totalEigenValues = eigenValuesAllkPoints.size();
    const double intervalSize =
      dftParamsPtr->intervalSize / C_haToeV; // eV to Ha

    double lowerBoundEpsilon = eigenValuesAllkPoints[0];
    double upperBoundEpsilon = eigenValuesAllkPoints[totalEigenValues - 1];

    MPI_Allreduce(MPI_IN_PLACE,
                  &lowerBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&lowerBoundEpsilon),
                  MPI_MIN,
                  interpoolComm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &upperBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&upperBoundEpsilon),
                  MPI_MAX,
                  interpoolComm);

    lowerBoundEpsilon =
      lowerBoundEpsilon - 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);
    upperBoundEpsilon =
      upperBoundEpsilon + 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);

    const unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);

    // maps atomId to number of sphericalfunctions
    std::map<unsigned int, unsigned int> numberSphericalFunctionsMap;

    std::vector<unsigned int> atomicNumbers =
          d_atomicOrbitalFnsContainer->getAtomicNumbers();

    // spin,kpoint,atomId, beta x numWfc vector
    std::vector<std::vector<std::map<unsigned int, std::vector<double>>>>
      pdosKernelWithoutSmearFunction;
    pdosKernelWithoutSmearFunction.resize(numSpinComponents);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        pdosKernelWithoutSmearFunction[spinIndex].resize(kPointWeights.size());
      }

    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); kPoint++)
          {
            // what does this do?
            d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);

            std::map<unsigned int, std::vector<double>>
              &pdosKernelWithoutSmearFunctionSpinKpoint =
                pdosKernelWithoutSmearFunction[spinIndex][kPoint];

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);

                flattenedArrayBlock =
                  &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

                // this opertaion sets d_numberWaveFunctions =
                // currentBlockSize
                // size of resVec =
                // currentBlockSize*(number of atomic Orbitals for each
                // atom) summed over all atoms that have compact support on
                // the elements in the processor
                d_nonLocalOperator->initialiseFlattenedDataStructure(
                  currentBlockSize, resVec);

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * totalNumWaveFunctions *
                                        (numSpinComponents * kPoint +
                                         spinIndex) +
                                      iNode * totalNumWaveFunctions + jvec,
                                    currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyToBlockConstantStride(
                          currentBlockSize,
                          totalNumWaveFunctions,
                          numLocalDofs,
                          jvec,
                          X->data() +
                            numLocalDofs * totalNumWaveFunctions *
                              (numSpinComponents * kPoint + spinIndex),
                          flattenedArrayBlock->data());
#endif
                    basisOperationsPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                               quadratureIndex,
                                               false);

                    flattenedArrayBlock->updateGhostValues();
                    // distribute applies the constraints
                    basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;

                            if (currentCellsBlockSize * currentBlockSize !=
                                previousSize)
                              {
                                tempCellNodalData.resize(currentCellsBlockSize *
                                                         currentBlockSize *
                                                         numNodesPerElement);
                                previousSize =
                                  currentCellsBlockSize * currentBlockSize;
                              }

                            basisOperationsPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));
                            d_nonLocalOperator->applyCconjtransOnX(
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));
                          }
                      } // iblock
                    // This is not actually an allreduce. It simply does
                    // boundary communication. After this, the values for one
                    // atom is shared across the boundary
                    d_nonLocalOperator->applyAllReduceOnCconjtransX(resVec);

                    // contains for every atomId a matrix of size beta_a x
                    // currentblocksize
                    std::map<unsigned int, std::vector<ValueType>>
                      extractedAtomicMap =
                        d_nonLocalOperator
                          ->extractLocallyOwnedAtomFromDistributedVector(
                            resVec);

                    std::map<unsigned int, std::vector<double>>
                      extractedAtomicMapSquaredBlock;

                    for (const auto &pair : extractedAtomicMap)
                      {
                        unsigned int atomId = pair.first; // globa Id
                        // size beta x currentblocksize (row major)
                        std::vector<ValueType> tempVec = pair.second;

                        // needed to be filled only once per spin
                        if (kPoint == 0 && jvec == 0)
                          {
                            const unsigned int numSphericalFunctions =
                              tempVec.size() / currentBlockSize;
                            numberSphericalFunctionsMap[atomId] =
                              numSphericalFunctions;
                          }
                        if (jvec == 0)
                          {
                            pdosKernelWithoutSmearFunctionSpinKpoint[atomId]
                              .resize(numberSphericalFunctionsMap[atomId] *
                                      totalNumWaveFunctions);
                          }


                        // absolute value squared. Block refers to wavefunction
                        // block
                        extractedAtomicMapSquaredBlock[atomId].clear();
                        extractedAtomicMapSquaredBlock[atomId].resize(
                          tempVec.size());
                        std::transform(
                          tempVec.begin(),
                          tempVec.end(),
                          extractedAtomicMapSquaredBlock[atomId].begin(),
                          [](const auto &a) {
                            return std::abs(a) * std::abs(a);
                          });

                        std::memcpy(
                          &pdosKernelWithoutSmearFunctionSpinKpoint
                            [atomId]
                            [jvec * numberSphericalFunctionsMap[atomId]],
                          &(extractedAtomicMapSquaredBlock[atomId][0]),
                          extractedAtomicMapSquaredBlock[atomId].size() *
                            sizeof(double));
                      }
                  } // current bandgroup check
              }     // jVec
          }         // kpoint
      }             // spinIndex

    // Till now all the data has been collected
    // In the above process be attentive about the band parallelization and the
    // device implementations

    std::vector<double> smearedValues;
    // spin, (atomId, beta x numEnergies)
    // summed over wfc blocks and kpoints
    std::vector<std::map<unsigned int, std::vector<double>>> summedOverBlocks;
    summedOverBlocks.resize(numSpinComponents);

    std::vector<double> numTotalAtomicOrbitals;
    std::vector<double> cumulativeNumAtomicOrbitals;

    numTotalAtomicOrbitals.resize(dftParamsPtr->natoms, 0.0);
    cumulativeNumAtomicOrbitals.resize(dftParamsPtr->natoms, 0.0);

    for (auto pair : pdosKernelWithoutSmearFunction[0][0])
      {
        unsigned atomId = pair.first;

        // 2 different prpocesses can have the same atomIDs.
        numTotalAtomicOrbitals[atomId] = numberSphericalFunctionsMap[atomId];
      }

    // Do an MPI_allreduce to get the atomIDs in one vector;
    // MAX operation ensures that only one of the identical non-zero element is
    // taken
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTotalAtomicOrbitals[0],
                  dftParamsPtr->natoms,
                  dataTypes::mpi_type_id(&numTotalAtomicOrbitals[0]),
                  MPI_MAX,
                  d_mpiCommParent);

    for (unsigned int i = 0; i < dftParamsPtr->natoms; i++)
      {
        if (i == 0)
          {
            cumulativeNumAtomicOrbitals[i] = numTotalAtomicOrbitals[i];
          }
        else
          {
            cumulativeNumAtomicOrbitals[i] =
              numTotalAtomicOrbitals[i] + cumulativeNumAtomicOrbitals[i - 1];
          }
      }

    std::vector<double> mpiVectorSummedOverBlocks;

    mpiVectorSummedOverBlocks.resize(
      numSpinComponents *
        cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] * numberIntervals,
      0.0);

    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); kPoint++)
          {
            const std::map<unsigned int, std::vector<double>>
              &spinKpointKernel =
                pdosKernelWithoutSmearFunction[spinIndex][kPoint];

            for (unsigned int epsInt = 0; epsInt < numberIntervals; epsInt++)
              {
                smearedValues.clear();
                double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                for (unsigned int iEigenVec = 0;
                     iEigenVec < totalNumWaveFunctions;
                     iEigenVec++)
                  {
                    double eigenValue =
                      eigenValues[kPoint][totalNumWaveFunctions * spinIndex +
                                          iEigenVec];
                    if (numSpinComponents == 2)
                      {
                        if (iEigenVec > highestStateNscfSolve)
                          {
                            smearedValues.push_back(0.0);
                          }
                        else
                          {
                            smearedValues.push_back(
                              (smearFunction(epsValue - eigenValue,
                                             dftParamsPtr)));
                          }
                      }
                    else
                      {
                        if (iEigenVec > highestStateNscfSolve)
                          {
                            smearedValues.push_back(0.0);
                          }
                        else
                          {
                            smearedValues.push_back(
                              (2 * smearFunction(epsValue - eigenValue,
                                                 dftParamsPtr)));
                          }
                      }
                  }
                for (auto &pair : spinKpointKernel)
                  {
                    unsigned int        atomId      = pair.first;
                    std::vector<double> kernelValue = pair.second;

                    const unsigned int numSphericalFunctions =
                      numberSphericalFunctionsMap[atomId];
                    if (epsInt == 0)
                      {
                        summedOverBlocks[spinIndex][atomId].resize(
                          numberIntervals * numSphericalFunctions);
                      }

                    const double zero(0.0), one(1.0);

                    for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                         jvec += BVec)
                      {
                        const unsigned int currentBlockSize =
                          std::min(BVec, totalNumWaveFunctions - jvec);

                        // In summedOverBlocks, summation of kpoints is done
                        BLASWrapperPtr->xgemm(
                          'N',
                          'N',
                          1,
                          numSphericalFunctions,
                          currentBlockSize,
                          &kPointWeights[kPoint],
                          &smearedValues[jvec],
                          1,
                          &kernelValue[jvec * numSphericalFunctions],
                          currentBlockSize,
                          &one,
                          &(summedOverBlocks[spinIndex][atomId]
                                            [epsInt * numSphericalFunctions]),
                          1);
                      }
                  } // atomID
              }     // epsInt

          } // kpoint

        // for (std::vector<double>::iterator it =
        //        (summedOverBlocks[spinIndex][0]).begin();
        //      it < (summedOverBlocks[spinIndex][0]).end();
        //      it++)
        //   {
        //     pcout << "Summed over blocks val:  " << *it << std::endl;
        //   }

        for (auto pair : pdosKernelWithoutSmearFunction[spinIndex][0])
          {
            unsigned int atomId         = pair.first;
            double numSphericalFunction = numberSphericalFunctionsMap[atomId];

            if (atomId != 0)
              {
                std::memcpy(
                  &mpiVectorSummedOverBlocks
                    [spinIndex *
                       cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                       numberIntervals +
                     cumulativeNumAtomicOrbitals[atomId - 1] * numberIntervals],
                  &summedOverBlocks[spinIndex][atomId][0],
                  summedOverBlocks[spinIndex][atomId].size() * sizeof(double));
              }
            else
              {
                std::memcpy(
                  &mpiVectorSummedOverBlocks
                    [spinIndex *
                     cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                     numberIntervals],
                  &summedOverBlocks[spinIndex][atomId][0],
                  summedOverBlocks[spinIndex][atomId].size() * sizeof(double));
              }
          }

      } // spinIndex

    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // for (int i = 0; i < size; ++i) {
    //     if (rank == i) {
    //         std::cout << "Processor " << rank << " performing operation:" << std::endl;
    //         for (auto i : mpiVectorSummedOverBlocks)
    //         // for (auto i : summedOverBlocks[0][0])
    //           if (i > 1e-05)
    //             {
    //               std::cout <<"iVal :" << i << std::endl;
    //             } 
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD); // Synchronize processors
    // }

    // //This allreduce won't work because there would be non zero entries for an atom in diferent tasks 
    // MPI_Allreduce(MPI_IN_PLACE,
    //               &mpiVectorSummedOverBlocks[0],
    //               cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
    //                 numberIntervals,
    //               dataTypes::mpi_type_id(&mpiVectorSummedOverBlocks[0]),
    //               MPI_SUM,
    //               interpoolComm);
    
    MPI_Allreduce(MPI_IN_PLACE,
                  &mpiVectorSummedOverBlocks[0],
                  cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                    numberIntervals,
                  dataTypes::mpi_type_id(&mpiVectorSummedOverBlocks[0]),
                  MPI_MAX,
                  d_mpiCommParent);


    pcout << "Writing PDOS File... (No shift in Energy)" << std::endl;

    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::string   pdosFileName = "PDOS.out";
        std::ofstream outFile(pdosFileName.c_str());
        outFile.setf(std::ios_base::fixed);
        outFile << std::setprecision(18);
        std::set<unsigned int> lNums;
        for (auto &pair : d_atomicOrbitalFnsContainer->getSphericalFunctions())
          {
            lNums.insert(pair.second->getQuantumNumberl());
          }

        std::vector<unsigned int> atomicNumbers =
          d_atomicOrbitalFnsContainer->getAtomicNumbers();

        if (outFile.is_open())
          {
            if (dftParamsPtr->spinPolarized == 1)
              {
              }
            else
              {
                outFile
                  << "E (eV)          s  py  pz  px  dxy  dyz  dz2  dxz  dx2-y2  "
                  << std::endl;
                for (auto atomId = 0; atomId < dftParamsPtr->natoms; atomId++)
                  {
                    outFile << "Atom ID: " << atomId << std::endl;

                    unsigned int atomicNum = atomicNumbers[atomId];

                    for (unsigned int epsInt = 0; epsInt < numberIntervals;
                         ++epsInt)
                      {
                        double epsValue =
                          lowerBoundEpsilon + epsInt * intervalSize;

                        outFile << epsValue * C_haToeV;
                        std::vector<double>::iterator startIt;
                        if (atomId != 0)
                          startIt = mpiVectorSummedOverBlocks.begin() +
                                    cumulativeNumAtomicOrbitals[atomId - 1] *
                                      numberIntervals +
                                    epsInt * numTotalAtomicOrbitals[atomId];

                        else
                          startIt = mpiVectorSummedOverBlocks.begin() +
                                    epsInt * numTotalAtomicOrbitals[atomId];

                        auto endIt = startIt + numTotalAtomicOrbitals[atomId];
                        for (auto it = startIt; it < endIt; it++)
                          {
                            outFile << "  " << *it;
                          }
                        outFile << std::endl;
                      }
                  }
              }
          }
      }
    pcout << "PDOS computation completed" << std::endl;
  }

  template class atomCenteredOrbitalsPostProcessing<
    dataTypes::number,
    dftfe::utils::MemorySpace::HOST>;

  // *********apply for the device as well***********

  // #if defined(DFTFE_WITH_DEVICE)
  //   template class atomCenteredOrbitalsPostProcessing<
  //     dataTypes::number,
  //     dftfe::utils::MemorySpace::DEVICE>;
  // #endif


} // namespace dftfe
