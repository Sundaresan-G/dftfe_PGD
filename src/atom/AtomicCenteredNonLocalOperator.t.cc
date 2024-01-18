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
    d_numberOfVectors           = numVectors;
    d_maxSingleAtomContribution = d_atomCenteredSphericalFunctionContainer
                                    ->getMaximumNumberOfSphericalFunctions();
    d_totalNonLocalEntries = d_atomCenteredSphericalFunctionContainer
                               ->getTotalNumberOfSphericalFunctions();
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::initKpoints(
    const std::vector<double> &kPointWeights,
    const std::vector<double> &kPointCoordinates)
  {
    d_kPointWeights     = kPointWeights;
    d_kPointCoordinates = kPointCoordinates;
    d_atomCenteredSphericalFunctionContainer
      ->getTotalAtomsAndNonLocalElementsInCurrentProcessor(
        d_totalAtomsInCurrentProc,
        d_totalNonlocalElems,
        d_numberCellsAccumNonLocalAtoms);
  }



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
    // for (int i = 0; i < atomIdsInCurrentProcess.size(); i++)
    //   std::cout << "Nonlocal Atom in proc: " << d_this_mpi_process << " "
    //             << atomIdsInCurrentProcess[i] << std::endl;


    // //
    // // data structures for memory optimization of projectorKetTimesVector
    // //
    std::vector<unsigned int> atomIdsAllProcessFlattened;
    MPI_Barrier(d_mpi_communicator);
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
    ownedAtomIdsInCurrentProcess.set_size(numberAtoms); // Check this
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
        ghostAtomIdsInCurrentProcessRenum.add_index(newAtomId);
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


#ifdef USE_COMPLEX
    distributedCPUVec<std::complex<double>> vec(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess,
      d_ghostSphericalFunctionIdsCurrentProcess,
      d_mpi_communicator);
#else
    distributedCPUVec<double> vec(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess,
      d_ghostSphericalFunctionIdsCurrentProcess,
      d_mpi_communicator);
#endif


    vec.update_ghost_values();
    d_SphericalFunctionKetTimesVectorPar.resize(1);
    d_SphericalFunctionKetTimesVectorPar[0].reinit(vec);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    getTotalAtomInCurrentProcessor()
  {
    return (d_totalAtomsInCurrentProc);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    getTotalNonLocalElementsInCurrentProcessor()
  {
    return (d_totalNonlocalElems);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperatorBase<ValueType,
                                     memorySpace>::getTotalNonLocalEntries()
  {
    return (d_totalNonLocalEntries);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperatorBase<ValueType,
                                     memorySpace>::getMaxSingleAtomEntries()
  {
    return (d_maxSingleAtomContribution);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperatorBase<ValueType, memorySpace>::
    atomSupportInElement(unsigned int iElem)
  {
    return (
      d_atomCenteredSphericalFunctionContainer->atomSupportInElement(iElem));
  }

} // namespace dftfe
