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

#include <cmath>

namespace dftfe
{
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    computeCMatrixEntries(const unsigned int quadratureIndex)
  {
    d_locallyOwnedCells = d_basisOperatorPtr->nCells();
    d_basisOperatorPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      d_basisOperatorPtr->nQuadsPerCell();
    d_numberNodesPerElement     = d_basisOperatorPtr->nDofsPerCell();
    const unsigned int numCells = d_locallyOwnedCells;
    const dftfe::utils::MemoryStorage<double, // ValueType for complex
                                      dftfe::utils::MemorySpace::HOST>
      &shapeValQuads =
        d_basisOperatorPtr
          ->shapeFunctionBasisData(); // shapeFunctionData() for complex
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = d_basisOperatorPtr->quadPoints();
    const dftfe::utils::MemoryStorage<ValueType,
                                      dftfe::utils::MemorySpace::HOST>
                                     JxwVector = d_basisOperatorPtr->JxW();
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

    // std::vector<ValueType> sphericalFunctionBasis(maxkPoints *
    //                                                  numberQuadraturePoints,
    //                                                0.0);
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);
    d_atomCenteredKpointIndexedSphericalFnQuadValues.clear();
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.clear();
    d_cellIdToAtomIdsLocalCompactSupportMap.clear();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    d_nonTrivialSphericalFnPerCell.clear();
    d_nonTrivialSphericalFnPerCell.resize(numCells, 0);

    d_nonTrivialSphericalFnsCellStartIndex.clear();
    d_nonTrivialSphericalFnsCellStartIndex.resize(numCells, 0);

    d_atomIdToNonTrivialSphericalFnCellStartIndex.clear();
    std::map<unsigned int, std::vector<unsigned int>>
                              globalAtomIdToNonTrivialSphericalFnsCellStartIndex;
    std::vector<unsigned int> accumTemp(numCells, 0);
    // Loop over atoms to determine sizes of various vectors for forces
    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        unsigned int       atomId = atomIdsInProc[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const unsigned int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom] =
          std::vector<unsigned int>(numCells, 0);
        globalAtomIdToNonTrivialSphericalFnsCellStartIndex[atomId] =
          std::vector<unsigned int>(numCells, 0);
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const unsigned int elementId =
              elementIndexesInAtomCompactSupport[iElemComp];

            d_cellIdToAtomIdsLocalCompactSupportMap[elementId].push_back(iAtom);

            d_nonTrivialSphericalFnPerCell[elementId] += numSphericalFunctions;
            d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom][elementId] =
              accumTemp[elementId];
            globalAtomIdToNonTrivialSphericalFnsCellStartIndex
              [atomId][elementId] = accumTemp[elementId];
            accumTemp[elementId] += numSphericalFunctions;
          }
      }

    d_sumNonTrivialSphericalFnOverAllCells =
      std::accumulate(d_nonTrivialSphericalFnPerCell.begin(),
                      d_nonTrivialSphericalFnPerCell.end(),
                      0);

    unsigned int accumNonTrivialSphericalFnCells = 0;
    for (int iElem = 0; iElem < numCells; ++iElem)
      {
        d_nonTrivialSphericalFnsCellStartIndex[iElem] =
          accumNonTrivialSphericalFnCells;
        accumNonTrivialSphericalFnCells +=
          d_nonTrivialSphericalFnPerCell[iElem];
      }
    d_atomCenteredKpointIndexedSphericalFnQuadValues.resize(
      maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
        numberQuadraturePoints,
      ValueType(0));
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.resize(
      maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
        numberQuadraturePoints * 3,
      ValueType(0));

    std::vector<std::vector<unsigned int>> sphericalFnKetTimesVectorLocalIds;
    sphericalFnKetTimesVectorLocalIds.clear();
    sphericalFnKetTimesVectorLocalIds.resize(d_totalAtomsInCurrentProc);
    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


        for (unsigned int alpha = 0; alpha < numSphericalFunctions; ++alpha)
          {
            unsigned int globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;

            unsigned int localId = d_SphericalFunctionKetTimesVectorPar[0]
                                     .get_partitioner()
                                     ->global_to_local(globalId);
            sphericalFnKetTimesVectorLocalIds[iAtom].push_back(localId);
          }
      }

    d_sphericalFnTimesVectorFlattenedVectorLocalIds.clear();
    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.clear();
    for (unsigned int ielem = 0; ielem < numCells; ++ielem)
      {
        for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            bool isNonTrivial = false;
            for (unsigned int i = 0;
                 i < d_cellIdToAtomIdsLocalCompactSupportMap[ielem].size();
                 i++)
              if (d_cellIdToAtomIdsLocalCompactSupportMap[ielem][i] == iAtom)
                {
                  isNonTrivial = true;
                  break;
                }
            if (isNonTrivial)
              {
                unsigned int       atomId = atomIdsInProc[iAtom];
                const unsigned int Znum   = atomicNumber[atomId];
                const unsigned int numSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (unsigned int iAlpha = 0; iAlpha < numSphericalFunctions;
                     ++iAlpha)
                  {
                    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.push_back(
                      ielem);
                    d_sphericalFnTimesVectorFlattenedVectorLocalIds.push_back(
                      sphericalFnKetTimesVectorLocalIds[iAtom][iAlpha]);
                  }
              }
          }
      }


    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        unsigned int       ChargeId = atomIdsInProc[iAtom];
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
        std::vector<double> sphericalFunctionBasisTimesJxW(
          numberElementsInAtomCompactSupport * NumTotalSphericalFunctions *
            numberQuadraturePoints,
          0.0);
#endif
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
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
                std::vector<dataTypes::number> sphericalFunctionBasis(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  ValueType(0.0));
                std::vector<dataTypes::number>
                  sphericalFunctionBasisTimesImageDist(
                    maxkPoints * numberQuadraturePoints *
                      (2 * lQuantumNumber + 1) * 3,
                    ValueType(0.0));
                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == 0)
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
                        if (r <= sphFn->getRadialCutOff())
                          {
                            radialVal = sphFn->getRadialValue(r);

                            unsigned int tempIndex = 0;
                            for (int mQuantumNumber = int(-lQuantumNumber);
                                 mQuantumNumber <= int(lQuantumNumber);
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
                                for (int kPoint = 0; kPoint < maxkPoints;
                                     ++kPoint)
                                  {
                                    angle =
                                      d_kPointCoordinates[3 * kPoint + 0] *
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

                                    sphericalFunctionBasis
                                      [kPoint * numberQuadraturePoints *
                                         (2 * lQuantumNumber + 1) +
                                       tempIndex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      ValueType(cos(angle) *
                                                  sphericalFunctionValue,
                                                -sin(angle) *
                                                  sphericalFunctionValue);

                                    for (unsigned int iDim = 0; iDim < 3;
                                         ++iDim)
                                      sphericalFunctionBasisTimesImageDist
                                        [kPoint * numberQuadraturePoints *
                                           (2 * lQuantumNumber + 1) * 3 +
                                         tempIndex * numberQuadraturePoints *
                                           3 +
                                         iQuadPoint * 3 + iDim] +=
                                        dataTypes::number(
                                          cos(angle) * sphericalFunctionValue *
                                            x[iDim],
                                          -sin(angle) * sphericalFunctionValue *
                                            x[iDim]);
                                  } // k-Point Loop
#else
                                sphericalFunctionBasis
                                  [tempIndex * numberQuadraturePoints +
                                   iQuadPoint] += sphericalFunctionValue;
                                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                                  sphericalFunctionBasisTimesImageDist
                                    [tempIndex * numberQuadraturePoints * 3 +
                                     iQuadPoint * 3 + iDim] +=
                                    sphericalFunctionValue * x[iDim];
                                  // sphericalFunctionBasis[iQuadPoint] +=
                                  // sphericalFunctionValue;
#endif
                                tempIndex++;
                              } // Angular momentum m loop
                          }     // inside r <= Rmax

                      } // quad loop

                  } // image atom loop
                const unsigned int startIndex1 =
                  d_nonTrivialSphericalFnsCellStartIndex[elementIndex];
                const unsigned int startIndex2 =
                  globalAtomIdToNonTrivialSphericalFnsCellStartIndex
                    [ChargeId][elementIndex];
                for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  {
                    for (unsigned int tempIndex = startIndex;
                         tempIndex < endIndex;
                         tempIndex++)
                      {
                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          d_atomCenteredKpointIndexedSphericalFnQuadValues
                            [kPoint * d_sumNonTrivialSphericalFnOverAllCells *
                               numberQuadraturePoints +
                             startIndex1 * numberQuadraturePoints +
                             (startIndex2 + tempIndex) *
                               numberQuadraturePoints +
                             iQuadPoint] = sphericalFunctionBasis
                              [kPoint * numberQuadraturePoints *
                                 (2 * lQuantumNumber + 1) +
                               (tempIndex - startIndex) *
                                 numberQuadraturePoints +
                               iQuadPoint];

                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          for (unsigned int iDim = 0; iDim < 3; ++iDim)
                            d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                              [kPoint * d_sumNonTrivialSphericalFnOverAllCells *
                                 numberQuadraturePoints * 3 +
                               startIndex1 * numberQuadraturePoints * 3 +
                               (startIndex2 + tempIndex) *
                                 numberQuadraturePoints * 3 +
                               iQuadPoint * 3 + iDim] =
                                sphericalFunctionBasisTimesImageDist
                                  [kPoint * numberQuadraturePoints *
                                     (2 * lQuantumNumber + 1) * 3 +
                                   (tempIndex - startIndex) *
                                     numberQuadraturePoints * 3 +
                                   iQuadPoint * 3 + iDim];
                      } // tempIndex
                  }



#ifdef USE_COMPLEX
                for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  for (unsigned int beta = startIndex; beta < endIndex; beta++)
                    {
                      for (int iQuadPoint = 0;
                           iQuadPoint < numberQuadraturePoints;
                           ++iQuadPoint)
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
                        } // quadPoint

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
                    } // beta
#else
                for (unsigned int beta = startIndex; beta < endIndex; beta++)
                  {
                    for (int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        sphericalFunctionBasisTimesJxW
                          [iElemComp * NumTotalSphericalFunctions *
                             numberQuadraturePoints +
                           beta * numberQuadraturePoints + iQuadPoint] =
                            sphericalFunctionBasis[(beta - startIndex) *
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
                      } // quadPoint
                  }     // beta
#endif
              } // alpha loop


          } // element loop



        const char         transA = 'N', transB = 'N';
        const double       scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
        const unsigned int inc = 1;
        const unsigned int n = numberElementsInAtomCompactSupport * maxkPoints *
                               NumTotalSphericalFunctions;
        const unsigned int  m = d_numberNodesPerElement;
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
                   &sphericalFunctionBasisTimesJxW[0],
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
            d_CMatrixEntriesConjugate[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));
            d_CMatrixEntriesTranspose[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));

            std::vector<ValueType> &CMatrixEntriesConjugateAtomElem =
              d_CMatrixEntriesConjugate[ChargeId][iElemComp];


            std::vector<ValueType> &CMatrixEntriesTransposeAtomElem =
              d_CMatrixEntriesTranspose[ChargeId][iElemComp];



            for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
              {
                for (int beta = 0; beta < NumTotalSphericalFunctions; ++beta)
                  for (int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
                    {
                      const unsigned int flattenedIndex =
                        iElemComp * maxkPoints * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        kPoint * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        beta * d_numberNodesPerElement + iNode;
                      const double tempReal =
                        projectorMatricesReal[flattenedIndex];
                      const double tempImag =
                        projectorMatricesImag[flattenedIndex];
                      if (isnan(tempReal))
                        std::cout
                          << "Real->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << beta << " " << iNode << std::endl;
                      if (isnan(tempImag))
                        std::cout
                          << "Imag->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << beta << " " << iNode << std::endl;
                        // const ValueType temp =
                        // projectorMatrices[flattenedIndex];
#ifdef USE_COMPLEX
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * beta + iNode]
                          .real(tempReal);
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * beta + iNode]
                          .imag(-tempImag);

                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + beta]
                          .real(tempReal);
                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + beta]
                          .imag(tempImag);



#else
                      CMatrixEntriesConjugateAtomElem[d_numberNodesPerElement *
                                                        beta +
                                                      iNode] = tempReal;

                      CMatrixEntriesTransposeAtomElem
                        [NumTotalSphericalFunctions * iNode + beta] = tempReal;
#endif
                    } // node loop
              }       // k point loop
          }           // non-trivial element loop



      } // ChargeId loop
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyAllReduceonCTX(
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        unsigned int       Zno    = atomicNumber[atomId];
        const unsigned int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
        for (unsigned int alpha = 0; alpha < numberSphericalFunctions; alpha++)
          {
            const unsigned int id =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;
            std::memcpy(sphericalFunctionKetTimesVectorParFlattened.data() +
                          sphericalFunctionKetTimesVectorParFlattened
                              .getMPIPatternP2P()
                              ->globalToLocal(id) *
                            d_numberWaveFunctions,
                        d_sphericalFnTimesWavefunMatrix[atomId].begin() +
                          d_numberWaveFunctions * alpha,
                        d_numberWaveFunctions * sizeof(ValueType));


            // d_BLASWrapperPtr->xcopy(
            //   d_numberWaveFunctions,
            //   &d_sphericalFnTimesWavefunMatrix[atomId]
            //                                  [d_numberWaveFunctions * alpha],
            //   inc,
            //   sphericalFunctionKetTimesVectorParFlattened.data() +
            //     sphericalFunctionKetTimesVectorParFlattened.getMPIPatternP2P()
            //     ->globalToLocal(id) *d_numberWaveFunctions,
            //   inc);
          }
      }
    sphericalFunctionKetTimesVectorParFlattened.accumulateAddLocallyOwned();
    sphericalFunctionKetTimesVectorParFlattened.updateGhostValues();
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyCconjtrans_onX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X,
      const std::pair<unsigned int, unsigned int> cellRange)
  {
    const ValueType    zero(0.0), one(1.0);
    const unsigned int inc                            = 1;
    d_AllReduceCompleted                              = false;
    int                              numberOfElements = d_locallyOwnedCells;
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::map<unsigned int, std::vector<int>> sparsityPattern =
      d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    for (int iElem = cellRange.first; iElem < cellRange.second; iElem++)
      {
        const std::vector<int> atomIdsInElement =
          d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(iElem);
        int numOfAtomsInElement = atomIdsInElement.size();
        for (int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
          {
            unsigned int       atomId = atomIdsInElement[iAtom];
            unsigned int       Zno    = atomicNumber[atomId];
            const unsigned int numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
            const int nonZeroElementMatrixId =
              sparsityPattern.find(atomId)->second[iElem];

            d_BLASWrapperPtr->xgemm(
              'N',
              'N',
              d_numberWaveFunctions,
              numberSphericalFunctions,
              d_numberNodesPerElement,
              &one,
              &X[0],
              d_numberWaveFunctions,
              &d_CMatrixEntriesConjugate[atomId][nonZeroElementMatrixId]
                                        [d_kPointIndex *
                                         d_numberNodesPerElement *
                                         numberSphericalFunctions],
              d_numberNodesPerElement,
              &one,
              &d_sphericalFnTimesWavefunMatrix[atomId][0],
              d_numberWaveFunctions);

          } // iAtom

      } // iElem
  }


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyV_onCconjtransX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagCopyResultsToMatrix)
  {
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    if (couplingtype == CouplingStructure::diagonal)
      {
        unsigned int       startIndex = 0;
        const unsigned int inc        = 1;
        for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            const unsigned int atomId = atomIdsInProc[iAtom];
            const unsigned int Zno    = atomicNumber[atomId];
            const unsigned int numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);


            for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                double nonlocalConstantV = couplingMatrix[startIndex++];
                const unsigned int localId =
                  sphericalFunctionKetTimesVectorParFlattened
                    .getMPIPatternP2P()
                    ->globalToLocal(
                      d_sphericalFunctionIdsNumberingMapCurrentProcess
                        .find(std::make_pair(atomId, alpha))
                        ->second);
                if (flagCopyResultsToMatrix)
                  {
                    std::transform(
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions,
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions + d_numberWaveFunctions,
                      d_sphericalFnTimesWavefunMatrix[atomId].begin() +
                        d_numberWaveFunctions * alpha,
                      [&nonlocalConstantV](auto &a) {
                        return nonlocalConstantV * a;
                      });
                  }
                else
                  {
                    d_BLASWrapperPtr->xscal(
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions,
                      nonlocalConstantV,
                      d_numberWaveFunctions);
                  }
              }
          }
      }
  }


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyC_VCconjtransX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &                                         Xout,
      const std::pair<unsigned int, unsigned int> cellRange)
  {
    const ValueType                                zero(0.0), one(1.0);
    const unsigned int                             inc = 1;
    const std::map<unsigned int, std::vector<int>> sparsityPattern =
      d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    for (int iElem = cellRange.first; iElem < cellRange.second; iElem++)
      {
        const std::vector<int> atomIdsInElement =
          d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(iElem);


        int numOfAtomsInElement = atomIdsInElement.size();
        for (int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
          {
            unsigned int atomId = atomIdsInElement[iAtom];

            unsigned int       Zno = atomicNumber[atomId];
            const unsigned int numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
            const int nonZeroElementMatrixId =
              sparsityPattern.find(atomId)->second[iElem];
            d_BLASWrapperPtr->xgemm(
              'N',
              'N',
              d_numberWaveFunctions,
              d_numberNodesPerElement,
              numberSphericalFunctions,
              &one,
              &d_sphericalFnTimesWavefunMatrix[atomId][0],
              d_numberWaveFunctions,
              &d_CMatrixEntriesTranspose[atomId][nonZeroElementMatrixId]
                                        [d_kPointIndex *
                                         d_numberNodesPerElement *
                                         numberSphericalFunctions],
              numberSphericalFunctions,
              &one,
              &Xout[(iElem - cellRange.first) * d_numberNodesPerElement *
                    d_numberNodesPerElement],
              d_numberWaveFunctions);

          } // iAtom


      } // iElem
  }
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    initialiseOperatorActionOnX(unsigned int kPointIndex)
  {
    d_kPointIndex = kPointIndex;

    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
        unsigned int atomId = atomIdsInProcessor[iAtom];

        d_sphericalFnTimesWavefunMatrix[atomId].setValue(0.0);
      }
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    d_numberWaveFunctions = numberWaveFunctions;

    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
      numberWaveFunctions,
      sphericalFunctionKetTimesVectorParFlattened);
    d_sphericalFnTimesWavefunMatrix.clear();
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
      {
        unsigned int atomId = atomIdsInProcessor[iAtom];
        unsigned int Zno    = atomicNumber[atomId];
        unsigned int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
        d_sphericalFnTimesWavefunMatrix[atomId].resize(
          numberSphericalFunctions * d_numberWaveFunctions, ValueType(0.0));
      }
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyVCconjtransOnX(
      const distributedCPUMultiVec<ValueType> &src,
      const unsigned int                       kPointIndex,
      const CouplingStructure                  couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    initialiseOperatorActionOnX(kPointIndex);
    sphericalFunctionKetTimesVectorParFlattened.setValue(0.0);

    const unsigned int inc = 1;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
      cellWaveFunctionMatrix;

    cellWaveFunctionMatrix.resize(d_numberNodesPerElement *
                                    d_numberWaveFunctions,
                                  0.0);


    if (d_totalNonlocalElems)
      {
        for (unsigned int iCell = 0; iCell < d_locallyOwnedCells; ++iCell)
          {
            if (AtomicCenteredNonLocalOperatorBase<
                  ValueType,
                  dftfe::utils::MemorySpace::HOST>::atomSupportInElement(iCell))
              {
                for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    dealii::types::global_dof_index localNodeId =
                      (d_basisOperatorPtr->d_cellDofIndexToProcessDofIndexMap
                         [iCell * d_numberNodesPerElement + iNode]) *
                      d_numberWaveFunctions;
                    d_BLASWrapperPtr->xcopy(
                      d_numberWaveFunctions,
                      src.data() + localNodeId,
                      inc,
                      &cellWaveFunctionMatrix[d_numberWaveFunctions * iNode],
                      inc);

                  } // Cell Extraction

                applyCconjtrans_onX(
                  cellWaveFunctionMatrix,
                  std::pair<unsigned int, unsigned int>(iCell, iCell + 1));

              } // if nonlocalAtomPResent
          }     // Cell Loop
        applyAllReduceonCTX(sphericalFunctionKetTimesVectorParFlattened);
        applyV_onCconjtransX(couplingtype,
                             couplingMatrix,
                             sphericalFunctionKetTimesVectorParFlattened,
                             false);



      } // nonlocal
  }

  template <typename ValueType>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomCenteredKpointIndexedSphericalFnQuadValues()
  {
    return d_atomCenteredKpointIndexedSphericalFnQuadValues;
  }
  template <typename ValueType>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues()
  {
    return d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues;
  }

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getSphericalFnTimesVectorFlattenedVectorLocalIds()

  {
    return d_sphericalFnTimesVectorFlattenedVectorLocalIds;
  }

  template <typename ValueType>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getAtomIdToNonTrivialSphericalFnCellStartIndex()
  {
    return d_atomIdToNonTrivialSphericalFnCellStartIndex;
  }

  template <typename ValueType>
  const unsigned int
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getTotalNonTrivialSphericalFnsOverAllCells()
  {
    return d_sumNonTrivialSphericalFnOverAllCells;
  }

  template <typename ValueType>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getCellIdToAtomIdsLocalCompactSupportMap()
  {
    return d_cellIdToAtomIdsLocalCompactSupportMap;
  }

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialSphericalFnsPerCell()
  {
    return d_nonTrivialSphericalFnPerCell;
  }

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialSphericalFnsCellStartIndex()
  {
    return d_nonTrivialSphericalFnsCellStartIndex;
  }

  template <typename ValueType>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getNonTrivialAllCellsSphericalFnAlphaToElemIdMap()
  {
    return d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap;
  }



} // namespace dftfe
