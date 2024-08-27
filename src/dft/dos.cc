// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020x The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//
#include <dft.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <sys/stat.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

namespace dftfe
{
  void
  loadSingleAtomPSIFiles(
    unsigned int  Z,
    unsigned int  n,
    unsigned int  l,
    unsigned int &fileReadFlag,
    double &      wfcInitTruncation,
    std::map<unsigned int,
             std::map<unsigned int,
                      std::map<unsigned int, alglib::spline1dinterpolant>>>
      &                  radValues,
    const MPI_Comm &     mpiCommParent,
    const dftParameters &dftParams)
  {
    if (radValues[Z][n].count(l) > 0)
      {
        fileReadFlag = 1;
        return;
      }

    //
    // set the paths for the Single-Atom wavefunction data
    //
    char psiFile[256];

    if (dftParams.isPseudopotential)
      {
        if (dftParams.readWfcForPdosPspFile && Z == 78)
          {
            sprintf(
              psiFile,
              "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomDataKB/psi%u%u.inp",
              DFTFE_PATH,
              Z,
              n,
              l);
          }
        else
          {
            sprintf(
              psiFile,
              "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp",
              DFTFE_PATH,
              Z,
              n,
              l);
          }
      }
    else
      sprintf(
        psiFile,
        "%s/data/electronicStructure/allElectron/z%u/singleAtomData/psi%u%u.inp",
        DFTFE_PATH,
        Z,
        n,
        l);

    std::vector<std::vector<double>> values;

    const double truncationTol = 1e-8;
    fileReadFlag               = dftUtils::readPsiFile(2, values, psiFile);

    //
    // spline fitting for single-atom wavefunctions
    //
    if (fileReadFlag > 0)
      {
        double       maxTruncationRadius = 0.0;
        unsigned int truncRowId          = 0;
        if (!dftParams.reproducible_output)
          {
            if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
              std::cout << "reading data from file: " << psiFile << std::endl;
          }

        int                 numRows = values.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        // x
        for (int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = values[irow][0];
          }
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);

        // y
        for (int irow = 0; irow < numRows; ++irow)
          {
            yData[irow] = values[irow][1];

            if (std::fabs(yData[irow]) > truncationTol)
              truncRowId = irow;
          }
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type = 0;
        alglib::spline1dbuildcubic(x,
                                   y,
                                   numRows,
                                   natural_bound_type,
                                   0.0,
                                   natural_bound_type,
                                   0.0,
                                   radValues[Z][n][l]);

        maxTruncationRadius = xData[truncRowId];
        if (maxTruncationRadius > wfcInitTruncation)
          wfcInitTruncation = maxTruncationRadius;
      }
  }



  // compute tdos
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::compute_tdos(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const unsigned int                      highestStateOfInterest,
    const std::string &                     dosFileName)
  {
    computing_timer.enter_subsection("DOS computation");

    // from 0th kpoint and 0th spin index
    int indexFermiEnergy = -1.0;
    for (int i = 0; i < d_numEigenValues; ++i)
      if (eigenValuesInput[0][i] >= fermiEnergy)
        {
          if (i > indexFermiEnergy)
            {
              indexFermiEnergy = i;
              break;
            }
        }

    // from 0th spin as this is only to get a printing range
    std::vector<double> eigenValuesAllkPoints;
    for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      for (int statesIter = 0; statesIter <= d_highestStateForNscfCalculation;
           ++statesIter)
        eigenValuesAllkPoints.push_back(eigenValuesInput[kPoint][statesIter]);

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    const double totalEigenValues  = eigenValuesAllkPoints.size();
    const double intervalSize =
      d_dftParamsPtr->intervalSize / C_haToeV;
    const double sigma             = C_kb * d_dftParamsPtr->TVal;
    double       lowerBoundEpsilon = eigenValuesAllkPoints[0];
    double upperBoundEpsilon = eigenValuesAllkPoints[totalEigenValues - 1];

    MPI_Allreduce(MPI_IN_PLACE,
                  &lowerBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&lowerBoundEpsilon),
                  MPI_MIN,
                  interpoolcomm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &upperBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&upperBoundEpsilon),
                  MPI_MAX,
                  interpoolcomm);

    lowerBoundEpsilon =
      lowerBoundEpsilon - 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);
    upperBoundEpsilon =
      upperBoundEpsilon + 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);

    const unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);

    std::vector<double> densityOfStates, densityOfStatesUp, densityOfStatesDown;


    if (d_dftParamsPtr->spinPolarized == 1)
      {
        densityOfStatesUp.resize(numberIntervals, 0.0);
        densityOfStatesDown.resize(numberIntervals, 0.0);
        for (int epsInt = 0; epsInt < numberIntervals; ++epsInt)
          {
            double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
            for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
              {
                for (unsigned int spinType = 0;
                     spinType < 1 + d_dftParamsPtr->spinPolarized;
                     ++spinType)
                  {
                    for (unsigned int statesIter = 0;
                         statesIter <= d_highestStateForNscfCalculation;
                         ++statesIter)
                      {
                        double term1 =
                          (epsValue -
                           eigenValuesInput[kPoint]
                                           [spinType * d_numEigenValues +
                                            statesIter]);
                        double denom = term1 * term1 + sigma * sigma;
                        if (spinType == 0)
                          densityOfStatesUp[epsInt] += d_kPointWeights[kPoint] *
                                                       (sigma / M_PI) *
                                                       (1.0 / denom);
                        else
                          densityOfStatesDown[epsInt] +=
                            d_kPointWeights[kPoint] * (sigma / M_PI) *
                            (1.0 / denom);
                      }
                  }
              }
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &densityOfStatesUp[0],
                      densityOfStatesUp.size(),
                      dataTypes::mpi_type_id(&densityOfStatesUp[0]),
                      MPI_SUM,
                      interpoolcomm);

        MPI_Allreduce(MPI_IN_PLACE,
                      &densityOfStatesDown[0],
                      densityOfStatesDown.size(),
                      dataTypes::mpi_type_id(&densityOfStatesDown[0]),
                      MPI_SUM,
                      interpoolcomm);
      }
    else
      {
        densityOfStates.resize(numberIntervals, 0.0);
        for (int epsInt = 0; epsInt < numberIntervals; ++epsInt)
          {
            double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
            for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
              {
                for (unsigned int statesIter = 0;
                     statesIter <= d_highestStateForNscfCalculation;
                     ++statesIter)
                  {
                    double term1 =
                      (epsValue - eigenValuesInput[kPoint][statesIter]);
                    densityOfStates[epsInt] +=
                      2.0 * d_kPointWeights[kPoint] *
                      d_atomCenteredOrbitalsPostProcessingPtr->smearFunction(
                        term1, d_dftParamsPtr);
                  }
              }
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &densityOfStates[0],
                      densityOfStates.size(),
                      dataTypes::mpi_type_id(&densityOfStates[0]),
                      MPI_SUM,
                      interpoolcomm);
      }


    if (d_dftParamsPtr->reproducible_output && d_dftParamsPtr->verbosity == 0)
      {
        pcout << "Writing tdos File..." << std::endl;
        if (d_dftParamsPtr->spinPolarized)
          pcout << "E(eV)          SpinUpDos SpinDownDos" << std::endl;
        else
          pcout << "E(eV)          Dos" << std::endl;
      }

    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::ofstream outFile(dosFileName.c_str());
        outFile.setf(std::ios_base::fixed);

        if (outFile.is_open())
          {
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << "  " << densityOfStatesUp[epsInt] << " "
                            << densityOfStatesDown[epsInt] << std::endl;
                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double epsValueTrunc =
                          std::floor(
                            1000000000 *
                            (lowerBoundEpsilon + epsInt * intervalSize) *
                            27.21138602) /
                          1000000000.0;
                        double dosSpinUpTrunc =
                          std::floor(1000000000 * densityOfStatesUp[epsInt]) /
                          1000000000.0;

                        double dosSpinDownTrunc =
                          std::floor(1000000000 * densityOfStatesDown[epsInt]) /
                          1000000000.0;


                        pcout << std::fixed << std::setprecision(8)
                              << epsValueTrunc << "  " << dosSpinUpTrunc << " "
                              << dosSpinDownTrunc << std::endl;
                      }
                  }
              }
            else
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << "  " << densityOfStates[epsInt] << std::endl;

                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double epsValueTrunc =
                          std::floor(
                            1000000000 *
                            (lowerBoundEpsilon + epsInt * intervalSize) *
                            27.21138602) /
                          1000000000.0;
                        double dosTrunc =
                          std::floor(1000000000 * densityOfStates[epsInt]) /
                          1000000000.0;
                        pcout << std::fixed << std::setprecision(8)
                              << epsValueTrunc << "  " << dosTrunc << std::endl;
                      }
                  }
              }
          }
      }
    computing_timer.leave_subsection("DOS computation");
  }


  // compute local density of states
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::compute_ldos(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const std::string &                     ldosFileName)
  {
    computing_timer.enter_subsection("LDOS computation");
    //
    // create a map of cellId and atomId
    //

    // loop over elements
    std::vector<double> eigenValuesAllkPoints;
    for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    double totalEigenValues  = eigenValuesAllkPoints.size();
    double intervalSize      = 0.001;
    double sigma             = C_kb * d_dftParamsPtr->TVal;
    double lowerBoundEpsilon = 1.5 * eigenValuesAllkPoints[0];
    double upperBoundEpsilon =
      eigenValuesAllkPoints[totalEigenValues - 1] * 1.5;

    MPI_Allreduce(MPI_IN_PLACE,
                  &lowerBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&lowerBoundEpsilon),
                  MPI_MIN,
                  interpoolcomm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &upperBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&upperBoundEpsilon),
                  MPI_MAX,
                  interpoolcomm);

    unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);
    unsigned int numberGlobalAtoms = atomLocations.size();

    // map each cell to an atom based on closest atom to the centroid of each
    // cell
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    std::map<dealii::CellId, unsigned int> cellToAtomIdMap;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            const dealii::Point<3> center(cell->center());

            // loop over all atoms
            double           distanceToClosestAtom = 1e8;
            dealii::Point<3> closestAtom;
            unsigned int     closestAtomId;
            for (unsigned int n = 0; n < atomLocations.size(); n++)
              {
                dealii::Point<3> atom(atomLocations[n][2],
                                      atomLocations[n][3],
                                      atomLocations[n][4]);
                if (center.distance(atom) < distanceToClosestAtom)
                  {
                    distanceToClosestAtom = center.distance(atom);
                    closestAtom           = atom;
                    closestAtomId         = n;
                  }
              }
            cellToAtomIdMap[cell->id()] = closestAtomId;
          }
      }

    std::vector<double> localDensityOfStates, localDensityOfStatesUp,
      localDensityOfStatesDown;
    localDensityOfStates.resize(numberGlobalAtoms * numberIntervals, 0.0);
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        localDensityOfStatesUp.resize(numberGlobalAtoms * numberIntervals, 0.0);
        localDensityOfStatesDown.resize(numberGlobalAtoms * numberIntervals,
                                        0.0);
      }

    // access finite-element data
    dealii::QGauss<3>   quadrature_formula(C_num1DQuad<FEOrder>());
    dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                  quadrature_formula,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int  dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
    const unsigned int  n_q_points    = quadrature_formula.size();


    const unsigned int blockSize =
      std::min(d_dftParamsPtr->wfcBlockSize, d_numEigenValues);

    std::vector<double> tempContribution(blockSize, 0.0);
    std::vector<double> tempQuadPointValues(n_q_points);

    const unsigned int localVectorSize =
      matrix_free_data.get_vector_partitioner()->locally_owned_size();
    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());
    std::vector<distributedCPUVec<dataTypes::number>>
      eigenVectorsFlattenedBlock((1 + d_dftParamsPtr->spinPolarized) *
                                 d_kPointWeights.size());

    for (unsigned int ivec = 0; ivec < d_numEigenValues; ivec += blockSize)
      {
        const unsigned int currentBlockSize =
          std::min(blockSize, d_numEigenValues - ivec);

        if (currentBlockSize != blockSize || ivec == 0)
          {
            for (unsigned int kPoint = 0;
                 kPoint <
                 (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
                 ++kPoint)
              {
                eigenVectors[kPoint].resize(currentBlockSize);
                for (unsigned int i = 0; i < currentBlockSize; ++i)
                  eigenVectors[kPoint][i].reinit(d_tempEigenVec);


                vectorTools::createDealiiVector<dataTypes::number>(
                  matrix_free_data.get_vector_partitioner(),
                  currentBlockSize,
                  eigenVectorsFlattenedBlock[kPoint]);
                eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
              }
          }


        std::vector<std::vector<double>> blockedEigenValues(
          d_kPointWeights.size(),
          std::vector<double>((1 + d_dftParamsPtr->spinPolarized) *
                                currentBlockSize,
                              0.0));
        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
            {
              blockedEigenValues[kPoint][iWave] =
                eigenValues[kPoint][ivec + iWave];
              if (d_dftParamsPtr->spinPolarized == 1)
                blockedEigenValues[kPoint][currentBlockSize + iWave] =
                  eigenValues[kPoint][d_numEigenValues + ivec + iWave];
            }

        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          {
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
                eigenVectorsFlattenedBlock[kPoint].local_element(
                  iNode * currentBlockSize + iWave) =
                  d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                                localVectorSize +
                                              iNode * d_numEigenValues + ivec +
                                              iWave];

            eigenVectorsFlattenedBlock[kPoint].update_ghost_values();
            constraintsNoneDataInfo.distribute(
              eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
            eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              eigenVectors[kPoint],
              false);

            // FIXME: The underlying call to update_ghost_values
            // is required because currently localProc_dof_indicesReal
            // and localProc_dof_indicesImag are only available for
            // locally owned nodes. Once they are also made available
            // for ghost nodes- use true for the last argument in
            // copyFlattenedDealiiVecToSingleCompVec(..) above and supress
            // underlying call.
            for (unsigned int i = 0; i < currentBlockSize; ++i)
              eigenVectors[kPoint][i].update_ghost_values();
#else
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              eigenVectors[kPoint],
              true);

#endif
          }

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            for (unsigned int spinType = 0; spinType < 2; ++spinType)
              {
                typename dealii::DoFHandler<3>::active_cell_iterator
                  cellN = dofHandler.begin_active(),
                  endcN = dofHandler.end();

                for (; cellN != endcN; ++cellN)
                  {
                    if (cellN->is_locally_owned())
                      {
                        fe_values.reinit(cellN);
                        unsigned int globalAtomId =
                          cellToAtomIdMap[cellN->id()];

                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            fe_values.get_function_values(
                              eigenVectors[spinType][iEigenVec],
                              tempQuadPointValues);

                            tempContribution[iEigenVec] = 0.0;
                            for (unsigned int q_point = 0; q_point < n_q_points;
                                 ++q_point)
                              {
                                tempContribution[iEigenVec] +=
                                  tempQuadPointValues[q_point] *
                                  tempQuadPointValues[q_point] *
                                  fe_values.JxW(q_point);
                              }
                          }

                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          for (unsigned int epsInt = 0;
                               epsInt < numberIntervals;
                               ++epsInt)
                            {
                              double epsValue =
                                lowerBoundEpsilon + epsInt * intervalSize;
                              double term1 =
                                (epsValue -
                                 blockedEigenValues[0][spinType *
                                                         currentBlockSize +
                                                       iEigenVec]);
                              double smearedEnergyLevel =
                                (sigma / M_PI) *
                                (1.0 / (term1 * term1 + sigma * sigma));

                              if (spinType == 0)
                                localDensityOfStatesUp[numberIntervals *
                                                         globalAtomId +
                                                       epsInt] +=
                                  tempContribution[iEigenVec] *
                                  smearedEnergyLevel;
                              else
                                localDensityOfStatesDown[numberIntervals *
                                                           globalAtomId +
                                                         epsInt] +=
                                  tempContribution[iEigenVec] *
                                  smearedEnergyLevel;
                            }
                      }
                  }
              }
          }
        else
          {
            typename dealii::DoFHandler<3>::active_cell_iterator
              cellN = dofHandler.begin_active(),
              endcN = dofHandler.end();

            for (; cellN != endcN; ++cellN)
              {
                if (cellN->is_locally_owned())
                  {
                    fe_values.reinit(cellN);
                    unsigned int globalAtomId = cellToAtomIdMap[cellN->id()];

                    for (unsigned int iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      {
                        fe_values.get_function_values(
                          eigenVectors[0][iEigenVec], tempQuadPointValues);

                        tempContribution[iEigenVec] = 0.0;
                        for (unsigned int q_point = 0; q_point < n_q_points;
                             ++q_point)
                          {
                            tempContribution[iEigenVec] +=
                              tempQuadPointValues[q_point] *
                              tempQuadPointValues[q_point] *
                              fe_values.JxW(q_point);
                          }
                      }

                    for (unsigned int iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      for (unsigned int epsInt = 0; epsInt < numberIntervals;
                           ++epsInt)
                        {
                          double epsValue =
                            lowerBoundEpsilon + epsInt * intervalSize;
                          double term1 =
                            (epsValue - blockedEigenValues[0][iEigenVec]);
                          double smearedEnergyLevel =
                            (sigma / M_PI) *
                            (1.0 / (term1 * term1 + sigma * sigma));
                          localDensityOfStates[numberIntervals * globalAtomId +
                                               epsInt] +=
                            2.0 * tempContribution[iEigenVec] *
                            smearedEnergyLevel;
                        }
                  }
              }
          }
      } // ivec loop

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dealii::Utilities::MPI::sum(localDensityOfStatesUp,
                                    mpi_communicator,
                                    localDensityOfStatesUp);

        dealii::Utilities::MPI::sum(localDensityOfStatesDown,
                                    mpi_communicator,
                                    localDensityOfStatesDown);
      }
    else
      {
        dealii::Utilities::MPI::sum(localDensityOfStates,
                                    mpi_communicator,
                                    localDensityOfStates);
      }

    double checkSum = 0;
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::ofstream outFile(ldosFileName.c_str());
        outFile.setf(std::ios_base::fixed);

        if (outFile.is_open())
          {
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << " ";
                    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms;
                         ++iAtom)
                      {
                        outFile
                          << std::setprecision(18)
                          << localDensityOfStatesUp[numberIntervals * iAtom +
                                                    epsInt]
                          << " "
                          << localDensityOfStatesDown[numberIntervals * iAtom +
                                                      epsInt]
                          << " ";
                        ;
                        checkSum +=
                          std::fabs(
                            localDensityOfStatesUp[numberIntervals * iAtom +
                                                   epsInt]) +
                          std::fabs(
                            localDensityOfStatesDown[numberIntervals * iAtom +
                                                     epsInt]);
                      }
                    outFile << std::endl;
                  }
              }
            else
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << " ";
                    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms;
                         ++iAtom)
                      {
                        outFile
                          << std::setprecision(18)
                          << localDensityOfStates[numberIntervals * iAtom +
                                                  epsInt]
                          << " ";
                        checkSum += std::fabs(
                          localDensityOfStates[numberIntervals * iAtom +
                                               epsInt]);
                      }
                    outFile << std::endl;
                  }
              }
          }
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Absolute sum of all ldos values: " << checkSum << std::endl;

    computing_timer.leave_subsection("LDOS computation");
  }

#include "dft.inst.cc"



} // namespace dftfe
