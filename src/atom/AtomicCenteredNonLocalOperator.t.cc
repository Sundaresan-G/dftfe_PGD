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
// @author Kartick Ramakrishnan, Sambit Das, Phani Motamarri, Vishal Subramanian
//


namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    AtomicCenteredNonLocalOperatorBase(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<AtomCenteredSphericalFunctionContainer>
                         atomCenteredSphericalFunctionContainer,
      const unsigned int numVectors,
      const MPI_Comm &   mpi_comm_parent)
    : d_mpi_communicator(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , d_n_mpi_processes(
        dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_BLASWrapperPtr = BLASWrapperPtr;
    d_atomCenteredSphericalFunctionContainer =
      atomCenteredSphericalFunctionContainer;
    d_numberOfVectors = numVectors;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::init(
    std::vector<double> &kPointWeights,
    std::vector<double> &kPointCoordinates,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                       basisOperationsPtr,
    const unsigned int quadratureIndex,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost)
  {
    d_kPointWeights     = kPointWeights;
    d_kPointCoordinates = kPointCoordinates;
    computeCMatrixEntries(basisOperationsPtr,
                          quadratureIndex,
                          BLASWrapperPtrHost);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    initCouplingMatrix(
      const CouplingEntries,
      const std::map<unsigned int, std::vector<double>> *matrixVec)
  {}

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    InitalisePartitioner(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr)
  {
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();
    const unsigned int numCells = totalLocallyOwnedCells;
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                              quadraturePointsVector = basisOperationsPtr->quadPoints();
    std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const unsigned int numberAtoms =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();



    // //
    // // data structures for memory optimization of projectorKetTimesVector
    // //
    std::vector<unsigned int> atomIdsAllProcessFlattened;
    pseudoUtils::exchangeLocalList(atomIdsInCurrentProcess,
                                   atomIdsAllProcessFlattened,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<unsigned int> atomIdsSizeCurrentProcess(1);
    atomIdsSizeCurrentProcess[0] = atomIdsInCurrentProcess.size();
    std::vector<unsigned int> atomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(atomIdsSizeCurrentProcess,
                                   atomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<std::vector<unsigned int>> atomIdsInAllProcess(
      d_n_mpi_processes);
    unsigned int count = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        for (unsigned int j = 0; j < atomIdsSizesAllProcess[iProc]; j++)
          {
            atomIdsInAllProcess[iProc].push_back(
              atomIdsAllProcessFlattened[count]);
            count++;
          }
      }
    atomIdsAllProcessFlattened.clear();

    dealii::IndexSet ownedAtomIdsInCurrentProcess;
    ownedAtomIdsInCurrentProcess.set_size(numberAtoms);
    ownedAtomIdsInCurrentProcess.add_indices(atomIdsInCurrentProcess.begin(),
                                             atomIdsInCurrentProcess.end());
    dealii::IndexSet ghostAtomIdsInCurrentProcess(ownedAtomIdsInCurrentProcess);
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            dealii::IndexSet temp;
            temp.set_size(numberAtoms);
            temp.add_indices(atomIdsInAllProcess[iProc].begin(),
                             atomIdsInAllProcess[iProc].end());
            ownedAtomIdsInCurrentProcess.subtract_set(temp);
          }
      }

    ghostAtomIdsInCurrentProcess.subtract_set(ownedAtomIdsInCurrentProcess);

    std::vector<unsigned int> ownedAtomIdsSizeCurrentProcess(1);
    ownedAtomIdsSizeCurrentProcess[0] =
      ownedAtomIdsInCurrentProcess.n_elements();
    std::vector<unsigned int> ownedAtomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(ownedAtomIdsSizeCurrentProcess,
                                   ownedAtomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);
    // // renumbering to make contiguous set of nonLocal atomIds
    std::map<int, int> oldToNewAtomIds;
    std::map<int, int> newToOldAtomIds;
    unsigned int       startingCount = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            startingCount += ownedAtomIdsSizesAllProcess[iProc];
          }
      }

    dealii::IndexSet ownedAtomIdsInCurrentProcessRenum,
      ghostAtomIdsInCurrentProcessRenum;
    ownedAtomIdsInCurrentProcessRenum.set_size(numberAtoms);
    ghostAtomIdsInCurrentProcessRenum.set_size(numberAtoms);
    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcess.begin();
         it != ownedAtomIdsInCurrentProcess.end();
         it++)
      {
        oldToNewAtomIds[*it]           = startingCount;
        newToOldAtomIds[startingCount] = *it;
        ownedAtomIdsInCurrentProcessRenum.add_index(startingCount);
        startingCount++;
      }

    pseudoUtils::exchangeNumberingMap(oldToNewAtomIds,
                                      d_n_mpi_processes,
                                      d_mpi_communicator);
    pseudoUtils::exchangeNumberingMap(newToOldAtomIds,
                                      d_n_mpi_processes,
                                      d_mpi_communicator);

    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcess.begin();
         it != ghostAtomIdsInCurrentProcess.end();
         it++)
      {
        unsigned int newAtomId = oldToNewAtomIds[*it];
        ghostAtomIdsInCurrentProcess.add_index(newAtomId);
      }

    if (d_this_mpi_process == 0 && false)
      {
        for (std::map<int, int>::const_iterator it = oldToNewAtomIds.begin();
             it != oldToNewAtomIds.end();
             it++)
          std::cout << " old nonlocal atom id: " << it->first
                    << " new nonlocal atomid: " << it->second << std::endl;

        std::cout
          << "number of local owned non local atom ids in all processors"
          << '\n';
        for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
          std::cout << ownedAtomIdsSizesAllProcess[iProc] << ",";
        std::cout << std::endl;
      }
    if (false)
      {
        std::stringstream ss1;
        ownedAtomIdsInCurrentProcess.print(ss1);
        std::stringstream ss2;
        ghostAtomIdsInCurrentProcess.print(ss2);
        std::string s1(ss1.str());
        s1.pop_back();
        std::string s2(ss2.str());
        s2.pop_back();
        std::cout << "procId: " << d_this_mpi_process << " old owned: " << s1
                  << " old ghost: " << s2 << std::endl;
        std::stringstream ss3;
        ownedAtomIdsInCurrentProcessRenum.print(ss3);
        std::stringstream ss4;
        ghostAtomIdsInCurrentProcessRenum.print(ss4);
        std::string s3(ss3.str());
        s3.pop_back();
        std::string s4(ss4.str());
        s4.pop_back();
        std::cout << "procId: " << d_this_mpi_process << " new owned: " << s3
                  << " new ghost: " << s4 << std::endl;
      }
    AssertThrow(
      ownedAtomIdsInCurrentProcessRenum.is_ascending_and_one_to_one(
        d_mpi_communicator),
      dealii::ExcMessage(
        "Incorrect renumbering and/or partitioning of non local atom ids"));

    int                       numberLocallyOwnedSphericalFunctions = 0;
    int                       numberGhostSphericalFunctions        = 0;
    std::vector<unsigned int> coarseNodeIdsCurrentProcess;
    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcessRenum.begin();
         it != ownedAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        coarseNodeIdsCurrentProcess.push_back(
          numberLocallyOwnedSphericalFunctions);
        numberLocallyOwnedSphericalFunctions +=
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(
              atomicNumber[newToOldAtomIds[*it]]);
      }

    std::vector<unsigned int> ghostAtomIdNumberSphericalFunctions;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        const unsigned temp = d_atomCenteredSphericalFunctionContainer
                                ->getTotalNumberOfSphericalFunctionsPerAtom(
                                  atomicNumber[newToOldAtomIds[*it]]);
        numberGhostSphericalFunctions += temp;
        ghostAtomIdNumberSphericalFunctions.push_back(temp);
      }

    std::vector<unsigned int>
      numberLocallyOwnedSphericalFunctionsCurrentProcess(1);
    numberLocallyOwnedSphericalFunctionsCurrentProcess[0] =
      numberLocallyOwnedSphericalFunctions;
    std::vector<unsigned int> numberLocallyOwnedSphericalFunctionsAllProcess;
    pseudoUtils::exchangeLocalList(
      numberLocallyOwnedSphericalFunctionsCurrentProcess,
      numberLocallyOwnedSphericalFunctionsAllProcess,
      d_n_mpi_processes,
      d_mpi_communicator);

    startingCount = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            startingCount +=
              numberLocallyOwnedSphericalFunctionsAllProcess[iProc];
          }
      }

    d_locallyOwnedSphericalFunctionIdsCurrentProcess.clear();
    d_locallyOwnedSphericalFunctionIdsCurrentProcess.set_size(
      std::accumulate(numberLocallyOwnedSphericalFunctionsAllProcess.begin(),
                      numberLocallyOwnedSphericalFunctionsAllProcess.end(),
                      0));
    std::vector<unsigned int> v(numberLocallyOwnedSphericalFunctions);
    std::iota(std::begin(v), std::end(v), startingCount);
    d_locallyOwnedSphericalFunctionIdsCurrentProcess.add_indices(v.begin(),
                                                                 v.end());

    std::vector<unsigned int> coarseNodeIdsAllProcess;
    for (unsigned int i = 0; i < coarseNodeIdsCurrentProcess.size(); ++i)
      coarseNodeIdsCurrentProcess[i] += startingCount;
    pseudoUtils::exchangeLocalList(coarseNodeIdsCurrentProcess,
                                   coarseNodeIdsAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    d_ghostSphericalFunctionIdsCurrentProcess.clear();
    d_ghostSphericalFunctionIdsCurrentProcess.set_size(
      std::accumulate(numberLocallyOwnedSphericalFunctionsAllProcess.begin(),
                      numberLocallyOwnedSphericalFunctionsAllProcess.end(),
                      0));
    unsigned int localGhostCount = 0;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        std::vector<unsigned int> g(
          ghostAtomIdNumberSphericalFunctions[localGhostCount]);
        std::iota(std::begin(g), std::end(g), coarseNodeIdsAllProcess[*it]);
        d_ghostSphericalFunctionIdsCurrentProcess.add_indices(g.begin(),
                                                              g.end());
        localGhostCount++;
      }
    if (false)
      {
        std::stringstream ss1;
        d_locallyOwnedSphericalFunctionIdsCurrentProcess.print(ss1);
        std::stringstream ss2;
        d_ghostSphericalFunctionIdsCurrentProcess.print(ss2);
        std::string s1(ss1.str());
        s1.pop_back();
        std::string s2(ss2.str());
        s2.pop_back();
        std::cout << "procId: " << d_this_mpi_process
                  << " projectors owned: " << s1 << " projectors ghost: " << s2
                  << std::endl;
      }
    AssertThrow(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess
        .is_ascending_and_one_to_one(d_mpi_communicator),
      dealii::ExcMessage(
        "Incorrect numbering and/or partitioning of non local projectors"));

    d_sphericalFunctionIdsNumberingMapCurrentProcess.clear();

    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcess.begin();
         it != ownedAtomIdsInCurrentProcess.end();
         it++)
      {
        const int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);

        for (unsigned int i = 0; i < numberSphericalFunctions; ++i)
          {
            d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
              *it, i)] = coarseNodeIdsAllProcess[oldToNewAtomIds[*it]] + i;
          }
      }

    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcess.begin();
         it != ghostAtomIdsInCurrentProcess.end();
         it++)
      {
        const int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);

        for (unsigned int i = 0; i < numberSphericalFunctions; ++i)
          {
            d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
              *it, i)] = coarseNodeIdsAllProcess[oldToNewAtomIds[*it]] + i;
          }
      }

    if (false)
      {
        for (std::map<std::pair<unsigned int, unsigned int>,
                      unsigned int>::const_iterator it =
               d_sphericalFunctionIdsNumberingMapCurrentProcess.begin();
             it != d_sphericalFunctionIdsNumberingMapCurrentProcess.end();
             ++it)
          {
            std::cout << "procId: " << d_this_mpi_process << " ["
                      << it->first.first << "," << it->first.second << "] "
                      << it->second << std::endl;
          }
      }
    d_mpiPatternP2P =
      std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
        d_locallyOwnedSphericalFunctionIdsCurrentProcess,
        d_ghostSphericalFunctionIdsCurrentProcess,
        d_mpi_communicator);
    ValueType zero = 0.0;
    d_SphericalFunctionKetTimesVectorFlattened =
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>(
        d_mpiPatternP2P, d_numberOfVectors, zero);
    // d_SphericalFunctionKetTimesVectorPar.resize(1);
    // d_SphericalFunctionKetTimesVectorPar[0].reinit(vec);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    computeCMatrixEntries(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr,
      const unsigned int quadratureIndex,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost)
  {
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    const unsigned int numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();
    const unsigned int numCells = totalLocallyOwnedCells;
    const dftfe::utils::MemoryStorage<ValueType,
                                      dftfe::utils::MemorySpace::HOST>
      &shapeValQuads = basisOperationsPtr->shapeFunctionBasisData();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = basisOperationsPtr->quadPoints();
    const dftfe::utils::MemoryStorage<ValueType,
                                      dftfe::utils::MemorySpace::HOST>
                                     JxwVector = basisOperationsPtr->JxW();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomCenteredSphericalFunctionContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomCenteredSphericalFunctionContainer
        ->getPeriodicImageCoordinatesList();
    const unsigned int maxkPoints = d_kPointWeights.size();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomCenteredSphericalFunctionContainer->getSphericalFunctions();
    std::vector<std::complex<double>> complexProjectorComponent(
      maxkPoints * numberQuadraturePoints, 0.0);

    // std::vector<ValueType> sphericalFunctionBasis(maxkPoints *
    //                                                  numberQuadraturePoints,
    //                                                0.0);

    for (unsigned int ChargeId = 0; ChargeId < numberAtomsOfInterest;
         ++ChargeId)
      {
        dealii::Point<3>   nuclearCoordinates(atomCoordinates[3 * ChargeId + 0],
                                            atomCoordinates[3 * ChargeId + 1],
                                            atomCoordinates[3 * ChargeId + 2]);
        const unsigned int atomId = ChargeId;
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;

        const unsigned int Zno = atomicNumber[ChargeId];
        const unsigned int NumRadialSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Zno);
        const unsigned int NumTotalSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[ChargeId];
        const unsigned int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();

        unsigned int imageIdsSize = imageCoordinates.size() / 3;


        //
        // allocate element Matrices
        //
        if (numberElementsInAtomCompactSupport > 0)
          {
            d_CMatrixEntriesConjugate[ChargeId].resize(
              numberElementsInAtomCompactSupport);
            d_CMatrixEntriesTranspose[ChargeId].resize(
              numberElementsInAtomCompactSupport);
          }
#ifdef USE_COMPLEX
        std::vector<double> sphericalFunctionBasisRealTimesJxW(
          numberElementsInAtomCompactSupport * maxkPoints *
            NumTotalSphericalFunctions * numberQuadraturePoints,
          0.0);
        std::vector<double> sphericalFunctionBasisImagTimesJxW(
          numberElementsInAtomCompactSupport * maxkPoints *
            NumTotalSphericalFunctions * numberQuadraturePoints,
          0.0);
#else
        std::vector<double> ZetalmDeltaVlTimesJxW(
          numberElementsInAtomCompactSupport * NumTotalSphericalFunctions *
            numberQuadraturePoints,
          0.0);
#endif
        // std::vector<ValueType> sphericalFunctionBasisTimesJxW(
        //   numberElementsInAtomCompactSupport * maxkPoints *
        //     NumTotalSphericalFunctions * numberQuadraturePoints,
        //   0.0);



        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            unsigned int totalSphFns = 0;
            for (unsigned int alpha = 0; alpha < NumRadialSphericalFunctions;
                 ++alpha)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                  sphericalFunction.find(std::make_pair(Zno, alpha))->second;

                unsigned int       lQuantumNumber = sphFn->getQuantumNumberl();
                const unsigned int startIndex =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalSphericalFunctionIndexStart(Zno, alpha);
                unsigned int endIndex = startIndex + 2 * lQuantumNumber + 1;
                std::vector<double> sphericalFunctionBasisReal(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  0.0);
                std::vector<double> sphericalFunctionBasisImag(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  0.0);
                std::vector<dataTypes::number> ZetalmDeltaVl(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  dataTypes::number(0.0));
                // std::fill(sphericalFunctionBasis.begin(),
                //           sphericalFunctionBasis.end(),
                //           ValueType(0.0));
                for (int iImageAtomCount = -1; iImageAtomCount < imageIdsSize;
                     ++imageIdsSize)
                  {
                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == -1)
                      {
                        chargePoint = nuclearCoordinates;
                      }
                    else
                      {
                        chargePoint[0] =
                          imageCoordinates[3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          imageCoordinates[3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          imageCoordinates[3 * iImageAtomCount + 2];
                      }
                    double x[3], pointMinusLatticeVector[3];
                    double sphericalHarmonicVal, radialVal,
                      sphericalFunctionValue;
                    double r, theta, phi, angle;

                    for (int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        x[0] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint] -
                          chargePoint[0];
                        x[1] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint + 1] -
                          chargePoint[1];
                        x[2] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint + 2] -
                          chargePoint[2];
                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        radialVal              = sphFn->getRadialValue(r);
                        unsigned int tempIndex = 0;
                        for (int mQuantumNumber = -lQuantumNumber;
                             mQuantumNumber <= lQuantumNumber;
                             mQuantumNumber++)
                          {
                            sphericalHarmonicUtils::getSphericalHarmonicVal(
                              theta,
                              phi,
                              lQuantumNumber,
                              mQuantumNumber,
                              sphericalHarmonicVal);

                            sphericalFunctionValue =
                              radialVal * sphericalHarmonicVal;



                            //
                            // kpoint loop
                            //
#ifdef USE_COMPLEX
                            pointMinusLatticeVector[0] =
                              x[0] + nuclearCoordinates[0];
                            pointMinusLatticeVector[1] =
                              x[1] + nuclearCoordinates[1];
                            pointMinusLatticeVector[2] =
                              x[2] + nuclearCoordinates[2];
                            for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                              {
                                angle = d_kPointCoordinates[3 * kPoint + 0] *
                                          pointMinusLatticeVector[0] +
                                        d_kPointCoordinates[3 * kPoint + 1] *
                                          pointMinusLatticeVector[1] +
                                        d_kPointCoordinates[3 * kPoint + 2] *
                                          pointMinusLatticeVector[2];

                                sphericalFunctionBasisReal
                                  [kPoint * numberQuadraturePoints *
                                     (2 * lQuantumNumber + 1) +
                                   tempIndex * numberQuadraturePoints +
                                   iQuadPoint] +=
                                  cos(angle) * sphericalFunctionValue;
                                sphericalFunctionBasisImag
                                  [kPoint * numberQuadraturePoints *
                                     (2 * lQuantumNumber + 1) +
                                   tempIndex * numberQuadraturePoints +
                                   iQuadPoint] +=
                                  -sin(angle) * sphericalFunctionValue;

                                // sphericalFunctionBasis[kPoint *
                                // numberQuadraturePoints +
                                //  iQuadPoint] +=
                                // exp(-angle) * sphericalFunctionValue;


                              } // k-Point Loop
#else
                            ZetalmDeltaVl[tempIndex * numberQuadraturePoints +
                                          iQuadPoint] += sphericalFunctionValue;
                            // sphericalFunctionBasis[iQuadPoint] +=
                            // sphericalFunctionValue;
#endif
                            tempIndex++;
                          } // Angular momentum m loop

                      } // quad loop

                  } // image atom loop

#ifdef USE_COMPLEX
                for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  for (int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints;
                       ++iQuadPoint)
                    {
                      for (unsigned int beta = startIndex; beta < endIndex;
                           beta++)
                        {
                          sphericalFunctionBasisRealTimesJxW
                            [iElemComp * maxkPoints *
                               NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             kPoint * NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             beta * numberQuadraturePoints + iQuadPoint] =
                              sphericalFunctionBasisReal
                                [kPoint * numberQuadraturePoints *
                                   (2 * lQuantumNumber + 1) +
                                 (beta - startIndex) * numberQuadraturePoints +
                                 iQuadPoint] *
                              real(JxwVector[elementIndex *
                                               numberQuadraturePoints +
                                             iQuadPoint]);
                          sphericalFunctionBasisImagTimesJxW
                            [iElemComp * maxkPoints *
                               NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             kPoint * NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             beta * numberQuadraturePoints + iQuadPoint] =
                              sphericalFunctionBasisImag
                                [kPoint * numberQuadraturePoints *
                                   (2 * lQuantumNumber + 1) +
                                 (beta - startIndex) * numberQuadraturePoints +
                                 iQuadPoint] *
                              real(JxwVector[elementIndex *
                                               numberQuadraturePoints +
                                             iQuadPoint]);
                        }

                      // sphericalFunctionBasisTimesJxW
                      //   [iElemComp * maxkPoints * NumTotalSphericalFunctions
                      //   *
                      //      numberQuadraturePoints +
                      //    kPoint * NumTotalSphericalFunctions *
                      //      numberQuadraturePoints +
                      //    beta * numberQuadraturePoints + iQuadPoint] =
                      //     sphericalFunctionBasis[kPoint *
                      //                                  numberQuadraturePoints
                      //                                  +
                      //                                iQuadPoint] *
                      //     JxwVector[elementIndex*numberQuadraturePoints +
                      //     iQuadPoint];
                    }
#else
                for (int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints;
                     ++iQuadPoint)
                  {
                    for (unsigned int beta = startIndex; beta < endIndex;
                         beta++)
                      {
                        ZetalmDeltaVlTimesJxW[iElemComp *
                                                NumTotalSphericalFunctions *
                                                numberQuadraturePoints +
                                              beta * numberQuadraturePoints +
                                              iQuadPoint] =
                          ZetalmDeltaVl[(beta - startIndex) *
                                          numberQuadraturePoints +
                                        iQuadPoint] *
                          JxwVector[elementIndex * numberQuadraturePoints +
                                    iQuadPoint];
                        // sphericalFunctionBasisTimesJxW[iElemComp *
                        // NumTotalSphericalFunctions *
                        //                         numberQuadraturePoints +
                        //                       beta * numberQuadraturePoints +
                        //                       iQuadPoint] =
                        //   sphericalFunctionBasis[iQuadPoint] *
                        //   JxwVector[elementIndex*numberQuadraturePoints +
                        //   iQuadPoint];
                      }
                  }
#endif
              } // alpha loop
                // if(totalSphFns != NumTotalSphericalFunctions)
                // {
            //   pcout<<"Error!! Not all spherical FUnctions considered in the
            //   function"<<std::endl; std::exit(0);
            // }

          } // element loop



        const char         transA = 'N', transB = 'N';
        const double       scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
        const unsigned int inc = 1;
        const unsigned int n = numberElementsInAtomCompactSupport * maxkPoints *
                               NumTotalSphericalFunctions;
        const unsigned int  m = numberNodesPerElement;
        const unsigned int  k = numberQuadraturePoints;
        std::vector<double> projectorMatricesReal(m * n, 0.0);
        std::vector<double> projectorMatricesImag(m * n, 0.0);
        // std::vector<ValueType> projectorMatricesReal(m * n, 0.0);
        if (numberElementsInAtomCompactSupport > 0)
          {
#ifdef USE_COMPLEX
            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &sphericalFunctionBasisRealTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesReal[0],
                   &m);

            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &sphericalFunctionBasisImagTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesImag[0],
                   &m);
#else
            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &ZetalmDeltaVlTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesReal[0],
                   &m);
#endif
            // d_BLASWrapperPtrHost->xgemm(&transA,
            //        &transB,
            //        &m,
            //        &n,
            //        &k,
            //        &scalarCoeffAlpha,
            //        &shapeValQuads[0],
            //        &m,
            //        &sphericalFunctionBasisTimesJxW[0],
            //        &k,
            //        &scalarCoeffBeta,
            //        &projectorMatrices[0],
            //        &m);
          }

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
#ifdef USE_COMPLEX
            d_CMatrixEntriesConjugate[ChargeId][iElemComp].resize(
              maxkPoints * numberNodesPerElement * NumTotalSphericalFunctions,
              std::complex<double>(0.0));
            d_CMatrixEntriesTranspose[ChargeId][iElemComp].resize(
              maxkPoints * numberNodesPerElement * NumTotalSphericalFunctions,
              std::complex<double>(0.0));

            std::vector<std::complex<double>> &CMatrixEntriesConjugateAtomElem =
              d_CMatrixEntriesConjugate[ChargeId][iElemComp];

            std::vector<std::complex<double>> &CMatrixEntriesTransposeAtomElem =
              d_CMatrixEntriesTranspose[ChargeId][iElemComp];

#else
            d_CMatrixEntriesConjugate[ChargeId][iElemComp].resize(
              numberNodesPerElement * NumTotalSphericalFunctions, 0.0);
            d_CMatrixEntriesTranspose[ChargeId][iElemComp].resize(
              numberNodesPerElement * NumTotalSphericalFunctions, 0.0);

            std::vector<double> &CMatrixEntriesConjugateAtomElem =
              d_CMatrixEntriesConjugate[ChargeId][iElemComp];


            std::vector<double> &CMatrixEntriesTransposeAtomElem =
              d_CMatrixEntriesTranspose[ChargeId][iElemComp];


#endif
            for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
              {
                for (int iPseudoWave = 0;
                     iPseudoWave < NumTotalSphericalFunctions;
                     ++iPseudoWave)
                  for (int iNode = 0; iNode < numberNodesPerElement; ++iNode)
                    {
                      const unsigned int flattenedIndex =
                        iElemComp * maxkPoints * NumTotalSphericalFunctions *
                          numberNodesPerElement +
                        kPoint * NumTotalSphericalFunctions *
                          numberNodesPerElement +
                        iPseudoWave * numberNodesPerElement + iNode;
                      const double tempReal =
                        projectorMatricesReal[flattenedIndex];
                      const double tempImag =
                        projectorMatricesImag[flattenedIndex];
                      // const ValueType temp =
                      // projectorMatrices[flattenedIndex];
#ifdef USE_COMPLEX
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         numberNodesPerElement * iPseudoWave + iNode]
                          .real(tempReal);
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         numberNodesPerElement * iPseudoWave + iNode]
                          .imag(-tempImag);

                      CMatrixEntriesTransposeAtomElem
                        [kPoint * numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + iPseudoWave]
                          .real(tempReal);
                      CMatrixEntriesTransposeAtomElem
                        [kPoint * numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + iPseudoWave]
                          .imag(tempImag);
                      // CMatrixEntriesConjugateAtomElem
                      //   [kPoint * numberNodesPerElement *
                      //      NumTotalSphericalFunctions +
                      //    numberNodesPerElement * iPseudoWave + iNode]
                      //     =temp;


                      // CMatrixEntriesTransposeAtomElem
                      //   [kPoint * numberNodesPerElement *
                      //      NumTotalSphericalFunctions +
                      //    NumTotalSphericalFunctions * iNode + iPseudoWave]
                      //      = std::conj(temp);

#else
                      CMatrixEntriesConjugateAtomElem[numberNodesPerElement *
                                                        iPseudoWave +
                                                      iNode] = tempReal;

                      CMatrixEntriesTransposeAtomElem
                        [NumTotalSphericalFunctions * iNode + iPseudoWave] =
                          tempReal;
#endif
                    } // node loop
              }       // k point loop
          }           // non-trivial element loop



      } // ChargeId loop

    //   //     //
    //   //     // Add mpi accumulation
    //   //     //
    // Copy from Temp to HOST/DEVICE
  }



} // namespace dftfe
