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

#include "AtomCenteredSphericalFunctionContainer.h"

namespace dftfe
{
  void
  AtomCenteredSphericalFunctionContainer::init(
    const std::vector<unsigned int> &atomicNumbers,
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      &listOfSphericalFunctions)
  {
    std::cout << "Initialising Container Class: " << std::endl;
    d_atomicNumbers               = atomicNumbers;
    d_sphericalFunctionsContainer = listOfSphericalFunctions;
    for (const auto &[key, value] : listOfSphericalFunctions)
      {
        unsigned int atomicNumber = key.first;
        unsigned int alpha        = key.second;
        unsigned int lIndex       = value->getQuantumNumberl();
        if (auto atomNumSize = d_numRadialSphericalFunctions.find(atomicNumber);
            atomNumSize != d_numRadialSphericalFunctions.end())
          {
            d_numRadialSphericalFunctions[atomicNumber] =
              d_numRadialSphericalFunctions[atomicNumber] + 1;
            d_numSphericalFunctions[atomicNumber] =
              d_numSphericalFunctions[atomicNumber] + (2 * lIndex + 1);
            for (unsigned int i = 0; i < 2 * lIndex + 1; i++)
              d_totalSphericalFunctionIndexStart[atomicNumber].push_back(
                d_totalSphericalFunctionIndexStart[atomicNumber].back() + 1);
          }
        else
          {
            d_numRadialSphericalFunctions[atomicNumber] = 1;
            d_numSphericalFunctions[atomicNumber]       = (2 * lIndex + 1);
            d_totalSphericalFunctionIndexStart[atomicNumber].push_back(0);
            for (unsigned int i = 0; i < 2 * lIndex; i++)
              d_totalSphericalFunctionIndexStart[atomicNumber].push_back(
                d_totalSphericalFunctionIndexStart[atomicNumber].back() + 1);
          }
      }
  }
  void
  AtomCenteredSphericalFunctionContainer::initaliseCoordinates(
    const std::vector<double> &      atomCoords,
    const std::vector<double> &      periodicCoords,
    const std::vector<unsigned int> &imageIds)
  {
    d_atomCoords = atomCoords;
    std::cout << "Setting Image coordinates " << std::endl;
    setImageCoordinates(imageIds, periodicCoords);
    std::cout << "Finished Setting Image coordinates " << std::endl;
    // AssertChecks
    AssertThrow(
      d_atomicNumbers.size() == d_atomCoords.size() / 3,
      dealii::ExcMessage(
        "DFT-FE Error: Number of atom Coordinates if interest is differnt from number of atomic Numbers"));
  }

  void
  AtomCenteredSphericalFunctionContainer::setImageCoordinates(
    const std::vector<unsigned int> &imageIds,
    const std::vector<double> &      periodicCoords)
  {
    d_periodicImageCoord.clear();
    for (unsigned int iAtom = 0; iAtom < d_atomicNumbers.size(); iAtom++)
      {
        d_periodicImageCoord[iAtom] = std::vector<double>(0);
      }

    for (unsigned int jImageAtom = 0; jImageAtom < imageIds.size();
         jImageAtom++)
      {
        // assumes dim =3;
        d_periodicImageCoord[imageIds[jImageAtom]].push_back(
          periodicCoords[3 * jImageAtom + 0]);
        d_periodicImageCoord[imageIds[jImageAtom]].push_back(
          periodicCoords[3 * jImageAtom + 1]);
        d_periodicImageCoord[imageIds[jImageAtom]].push_back(
          periodicCoords[3 * jImageAtom + 2]);
      }
  }

  unsigned int
  AtomCenteredSphericalFunctionContainer::getNumAtomCentersSize()
  {
    return d_atomicNumbers.size();
  }

  const std::vector<double> &
  AtomCenteredSphericalFunctionContainer::getAtomCoordinates() const
  {
    return d_atomCoords;
  }

  const std::map<unsigned int, std::vector<double>> &
  AtomCenteredSphericalFunctionContainer::getPeriodicImageCoordinatesList()
    const
  {
    return d_periodicImageCoord;
  }

  unsigned int
  AtomCenteredSphericalFunctionContainer::
    getTotalNumberOfRadialSphericalFunctionsPerAtom(unsigned int atomicNumber)
  {
    if (auto atomNumSize = d_numRadialSphericalFunctions.find(atomicNumber);
        atomNumSize != d_numRadialSphericalFunctions.end())
      {
        return atomNumSize->second;
      }
    else
      return 0;
  }

  unsigned int
  AtomCenteredSphericalFunctionContainer::
    getTotalNumberOfSphericalFunctionsPerAtom(unsigned int atomicNumber)
  {
    if (auto atomNumSize = d_numSphericalFunctions.find(atomicNumber);
        atomNumSize != d_numSphericalFunctions.end())
      {
        return atomNumSize->second;
      }
    else
      return 0;
  }

  const unsigned int
  AtomCenteredSphericalFunctionContainer::getTotalSphericalFunctionIndexStart(
    unsigned int Zno,
    unsigned int alpha)
  {
    std::vector<unsigned int> beta = d_totalSphericalFunctionIndexStart[Zno];
    if (alpha < getTotalNumberOfRadialSphericalFunctionsPerAtom(Zno))
      return beta[alpha];
    else
      {
        std::cout
          << "Warning Illegal Access in Line 132 of AtomCenteredContainerClass"
          << std::endl;
        std::exit(0);
      }
  }


  const std::vector<unsigned int> &
  AtomCenteredSphericalFunctionContainer::getAtomicNumbers() const
  {
    return (d_atomicNumbers);
  }
  const std::vector<unsigned int> &
  AtomCenteredSphericalFunctionContainer::getAtomIdsInCurrentProcess() const
  {
    return (d_AtomIdsInCurrentProcess);
  }


  const std::map<std::pair<unsigned int, unsigned int>,
                 std::shared_ptr<AtomCenteredSphericalFunctionBase>> &
  AtomCenteredSphericalFunctionContainer::getSphericalFunctions() const
  {
    return d_sphericalFunctionsContainer;
  }
  template <typename NumberType>
  void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int quadratureIndex,
    const double       cutOffVal,
    const unsigned int cutOffType)
  {
    //
    // get the number of non-local atoms
    //
    int numberAtomsOfInterest = d_atomicNumbers.size(); //

    std::cout << "Debug: Line 201" << std::endl;

    //     //
    //     // pre-allocate data structures that stores the sparsity of deltaVl
    //     //
    d_sparsityPattern.clear();
    d_elementIdsInAtomCompactSupport.clear();
    d_elementIndexesInAtomCompactSupport.clear();

    // d_sparsityPattern.resize(numberAtomsOfInterest);
    d_elementIdsInAtomCompactSupport.resize(numberAtomsOfInterest);
    d_elementIndexesInAtomCompactSupport.resize(numberAtomsOfInterest);
    d_AtomIdsInCurrentProcess.clear();

    //
    // loop over nonlocal atoms
    //
    unsigned int sparseFlag         = 0;
    int          cumulativeSplineId = 0;
    int          waveFunctionId;
    std::cout << "Debug: Line 221" << std::endl;
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = basisOperationsPtr->quadPoints();
    //
    // get number of global charges
    //
    unsigned int       numberGlobalCharges = d_atomicNumbers.size();
    const unsigned int numberElements      = totalLocallyOwnedCells;
    std::vector<int>   sparsityPattern(numberElements, -1);
    std::cout << "Debug: Line 234" << std::endl;
    for (int iAtom = 0; iAtom < numberAtomsOfInterest; ++iAtom)
      {
        //
        // temp variables
        //
        int          matCount            = 0;
        bool         isAtomIdInProcessor = false;
        unsigned int Zno                 = d_atomicNumbers[iAtom];
        //
        //
        int numberSphericalFunctions = d_numSphericalFunctions[Zno];

        //
        // get the global charge Id of the current nonlocal atom
        //


        std::cout << "Debug: Line 251" << std::endl;
        unsigned int imageIdsSize = d_periodicImageCoord[iAtom].size() / 3;

        //
        // resize the data structure corresponding to sparsity pattern
        //
        // std::vector<int> sparsityPattern;(numberElements,-1);
        d_sparsityPattern[iAtom].resize(numberElements, -1);

        if (imageIdsSize != 0)
          {
            std::fill(sparsityPattern.begin(), sparsityPattern.end(), -1);
            //
            // parallel loop over all elements
            //

            for (int iCell = 0; iCell < totalLocallyOwnedCells; iCell++)
              {
                std::vector<double> quadPoints(numberQuadraturePoints * 3, 0.0);
                for (int iQuad = 0; iQuad < numberQuadraturePoints; iQuad++)
                  {
                    quadPoints[iQuad * 3 + 0] =
                      quadraturePointsVector[iCell *
                                               (numberQuadraturePoints * 3) +
                                             iQuad * 3 + 0];
                    quadPoints[iQuad * 3 + 1] =
                      quadraturePointsVector[iCell *
                                               (numberQuadraturePoints * 3) +
                                             iQuad * 3 + 1];
                    quadPoints[iQuad * 3 + 2] =
                      quadraturePointsVector[iCell *
                                               (numberQuadraturePoints * 3) +
                                             iQuad * 3 + 2];
                  }
                sparseFlag = 0;

                for (int iImageAtomCount = -1; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    std::vector<double> x(3, 0.0);
                    dealii::Point<3>    chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == -1)
                      {
                        chargePoint[0] = d_atomCoords[3 * iAtom + 0];
                        chargePoint[1] = d_atomCoords[3 * iAtom + 1];
                        chargePoint[2] = d_atomCoords[3 * iAtom + 2];
                      }
                    else
                      {
                        chargePoint[0] =
                          d_periodicImageCoord[iAtom][3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          d_periodicImageCoord[iAtom][3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          d_periodicImageCoord[iAtom][3 * iImageAtomCount + 2];
                      }

                    // Any equivalent step available?
                    // if (chargePoint.distance(cell->center()) >
                    // d_nlPSPCutOff)
                    //   continue;

                    for (int iPsp = 0; iPsp < numberSphericalFunctions; ++iPsp)
                      {
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>
                          SphericalFunction =
                            d_sphericalFunctionsContainer[std::make_pair(Zno,
                                                                         iPsp)];
                        double radialProjVal;
                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          {
                            x[0] = quadPoints[3 * iQuadPoint] - chargePoint[0];
                            x[1] =
                              quadPoints[3 * iQuadPoint + 1] - chargePoint[1];
                            x[2] =
                              quadPoints[3 * iQuadPoint + 2] - chargePoint[2];
                            const double r = std::sqrt(
                              x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
                            if (cutOffType == 0)
                              {
                                double RadVal =
                                  SphericalFunction->getRadialValue(r);

                                if (RadVal >= cutOffVal)
                                  {
                                    sparseFlag = 1;
                                    break;
                                  }
                              }
                            else if (cutOffType == 1 && r < cutOffVal)
                              {
                                sparseFlag = 1;
                                break;
                              }
                          } // quadrature loop
                        if (sparseFlag == 1)
                          break;

                      } // iPsp loop ("l" loop)

                    if (sparseFlag == 1)
                      break;


                  } // image atom loop

                if (sparseFlag == 1)
                  {
                    dealii::CellId cell    = basisOperationsPtr->cellID(iCell);
                    sparsityPattern[iCell] = matCount;
                    d_elementIdsInAtomCompactSupport[iAtom].push_back(cell);
                    d_elementIndexesInAtomCompactSupport[iAtom].push_back(
                      iCell);
                    matCount += 1;
                    isAtomIdInProcessor = true;
                  }
              } // iCell
          }

#ifdef DEBUG
        if (d_dftParamsPtr->verbosity >= 4)
          std::cout << "No.of non zero elements in the compact support of
            atom "
                    << iAtom << " is "
                    << d_elementIteratorsInAtomCompactSupport[iAtom].size()
                    << std::endl;
#endif

        if (isAtomIdInProcessor)
          {
            d_AtomIdsInCurrentProcess.push_back(iAtom);
            d_sparsityPattern[iAtom] = sparsityPattern;
          }

      } // atom loop

    d_AtomIdsInElement.clear();
    d_AtomIdsInElement.resize(numberElements);

    for (int iCell = 0; iCell < numberElements; ++iCell)
      {
        for (int iAtom = 0; iAtom < d_AtomIdsInCurrentProcess.size(); iAtom++)
          {
            if (d_sparsityPattern[d_AtomIdsInCurrentProcess[iAtom]][iCell] >= 0)
              {
                d_AtomIdsInElement[iCell].push_back(
                  d_AtomIdsInCurrentProcess[iAtom]);
              }
          }
      }
  }
  template void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int quadratureIndex,
    const double       cutOffVal,
    const unsigned int cutOffType);

  template void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<std::complex<double>,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int quadratureIndex,
    const double       cutOffVal,
    const unsigned int cutOffType);

} // end of namespace dftfe
