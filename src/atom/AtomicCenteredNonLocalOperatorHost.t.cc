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
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    apply_C_V_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange)
  {}

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst)
  {}


  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    apply_CCT_onX(
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &                                                                 src,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &dst,
      const std::pair<unsigned int, unsigned int> &cellRange)
  {}



  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
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
    d_locallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    d_numberNodesPerElement     = basisOperationsPtr->nDofsPerCell();
    const unsigned int numCells = d_locallyOwnedCells;
    const dftfe::utils::MemoryStorage<double, // ValueType for complex
                                      dftfe::utils::MemorySpace::HOST>
      &shapeValQuads =
        basisOperationsPtr
          ->shapeFunctionBasisData(); // shapeFunctionData() for complex
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

    // std::vector<ValueType> sphericalFunctionBasis(maxkPoints *
    //                                                  numberQuadraturePoints,
    //                                                0.0);
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);

    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
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

                                    // sphericalFunctionBasis[kPoint *
                                    // numberQuadraturePoints +
                                    //  iQuadPoint] +=
                                    // exp(-angle) * sphericalFunctionValue;


                                  } // k-Point Loop
#else
                                ZetalmDeltaVl[tempIndex *
                                                numberQuadraturePoints +
                                              iQuadPoint] +=
                                  sphericalFunctionValue;
                                // sphericalFunctionBasis[iQuadPoint] +=
                                // sphericalFunctionValue;
#endif
                                tempIndex++;
                              } // Angular momentum m loop
                          }

                      } // quad loop

                  } // image atom loop

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
                for (int iPseudoWave = 0;
                     iPseudoWave < NumTotalSphericalFunctions;
                     ++iPseudoWave)
                  for (int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
                    {
                      const unsigned int flattenedIndex =
                        iElemComp * maxkPoints * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        kPoint * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        iPseudoWave * d_numberNodesPerElement + iNode;
                      const double tempReal =
                        projectorMatricesReal[flattenedIndex];
                      const double tempImag =
                        projectorMatricesImag[flattenedIndex];
                      if (isnan(tempReal))
                        std::cout
                          << "Real->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << iPseudoWave << " " << iNode << std::endl;
                      if (isnan(tempImag))
                        std::cout
                          << "Imag->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << iPseudoWave << " " << iNode << std::endl;
                        // const ValueType temp =
                        // projectorMatrices[flattenedIndex];
#ifdef USE_COMPLEX
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * iPseudoWave + iNode]
                          .real(tempReal);
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * iPseudoWave + iNode]
                          .imag(-tempImag);

                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + iPseudoWave]
                          .real(tempReal);
                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + iPseudoWave]
                          .imag(tempImag);
                      // CMatrixEntriesConjugateAtomElem
                      //   [kPoint * d_numberNodesPerElement *
                      //      NumTotalSphericalFunctions +
                      //    d_numberNodesPerElement * iPseudoWave + iNode]
                      //     =temp;


                      // CMatrixEntriesTransposeAtomElem
                      //   [kPoint * d_numberNodesPerElement *
                      //      NumTotalSphericalFunctions +
                      //    NumTotalSphericalFunctions * iNode + iPseudoWave]
                      //      = std::conj(temp);
                      // std::cout
                      //   << "DEBUG: CMatrix Entries: "
                      //   << (d_numberNodesPerElement * iPseudoWave + iNode)
                      //   << " " << tempReal <<" "<<tempImag<< std::endl;


#else
                      CMatrixEntriesConjugateAtomElem[d_numberNodesPerElement *
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
  }
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyCTonX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened,
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
              &d_ShapeFnTimesWavefunction[atomId][0],
              d_numberWaveFunctions);

          } // iAtom

      } // iElem


    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
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
              d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, alpha)];
            d_BLASWrapperPtr->xcopy(
              d_numberWaveFunctions,
              &d_ShapeFnTimesWavefunction[atomId]
                                         [d_numberWaveFunctions * alpha],
              inc,
              sphericalFunctionKetTimesVectorParFlattened.data() +
                sphericalFunctionKetTimesVectorParFlattened.getMPIPatternP2P()
                    ->globalToLocal(id) *
                  d_numberWaveFunctions,
              inc);
          }
      }
  }
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyCTonX(
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST> &X)
  {
    const ValueType    zero(0.0), one(1.0);
    const unsigned int inc                            = 1;
    d_AllReduceCompleted                              = false;
    int                              numberOfElements = d_locallyOwnedCells;
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::map<unsigned int, std::vector<int>> sparsityPattern =
      d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    for (int iElem = 0; iElem < d_locallyOwnedCells; iElem++)
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
              &d_ShapeFnTimesWavefunction[atomId][0],
              d_numberWaveFunctions);

          } // iAtom

      } // iElem


    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
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
              d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, alpha)];
            d_BLASWrapperPtr->xcopy(
              d_numberWaveFunctions,
              &d_ShapeFnTimesWavefunction[atomId]
                                         [d_numberWaveFunctions * alpha],
              inc,
              d_SphericalFunctionKetTimesVectorParFlattened.data() +
                d_SphericalFunctionKetTimesVectorParFlattened
                    .getMPIPatternP2P()
                    ->globalToLocal(id) *
                  d_numberWaveFunctions,
              inc);
          }
      }
    d_SphericalFunctionKetTimesVectorParFlattened.accumulateAddLocallyOwned();
    d_SphericalFunctionKetTimesVectorParFlattened.updateGhostValues();
    d_AllReduceCompleted = true;
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyV_onCTX(
      const CouplingStructure couplingtype,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &couplingMatrix,
      const dftfe::linearAlgebra::MultiVector<ValueType,
                                              dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &shapeFnTimesWavefunctionMatrix)
  {
    // if (!d_AllReduceCompleted)
    //   {
    //     d_SphericalFunctionKetTimesVectorParFlattened
    //       .accumulateAddLocallyOwned();
    //     d_SphericalFunctionKetTimesVectorParFlattened.updateGhostValues();
    //     d_AllReduceCompleted = true;
    //   }
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    if (couplingtype == CouplingStructure::diagonal)
      {
        unsigned int startIndex = 0;
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
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    [std::make_pair(atomId, alpha)];
                std::transform(
                  sphericalFunctionKetTimesVectorParFlattened.begin() +
                    localId * d_numberWaveFunctions,
                  sphericalFunctionKetTimesVectorParFlattened.begin() +
                    localId * d_numberWaveFunctions + d_numberWaveFunctions,
                  shapeFnTimesWavefunctionMatrix[atomId].begin() +
                    d_numberWaveFunctions * alpha,
                  [&nonlocalConstantV](auto &a) {
                    return nonlocalConstantV * a;
                  });
              }
          }
      }
  }
  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &Xout)
  {
    // const ValueType                                zero(0.0), one(1.0);
    // const unsigned int                             inc = 1;
    // const std::map<unsigned int, std::vector<int>> sparsityPattern =
    //   d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    // const std::vector<unsigned int> &atomicNumber =
    //   d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    // for (int iElem = 0; iElem < d_locallyOwnedCells; iElem++)
    //   {
    //     const std::vector<int> atomIdsInElement =
    //       d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(iElem);
    //     int numOfAtomsInElement = atomIdsInElement.size();
    //     for (int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
    //       {
    //         unsigned int       atomId = atomIdsInElement[iAtom];
    //         unsigned int       Zno    = atomicNumber[atomId];
    //         const unsigned int numberSphericalFunctions =
    //           d_atomCenteredSphericalFunctionContainer
    //             ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
    //         const int nonZeroElementMatrixId =
    //           sparsityPattern.find(atomId)->second[iElem];

    //         d_BLASWrapperPtr->xgemm(
    //           'N',
    //           'N',
    //           d_numberWaveFunctions,
    //           d_numberNodesPerElement,
    //           numberSphericalFunctions,
    //           &one,
    //           &d_ShapeFnTimesWavefunction[atomId][0],
    //           d_numberWaveFunctions,
    //           &d_CMatrixEntriesTranspose[atomId][nonZeroElementMatrixId]
    //                                     [d_kPointIndex *
    //                                      d_numberNodesPerElement *
    //                                      numberSphericalFunctions],
    //           numberSphericalFunctions,
    //           &one,
    //           &Xout[iElem * d_numberNodesPerElement *
    //           d_numberNodesPerElement], d_numberWaveFunctions);

    //       } // iAtom

    //   } // iElem
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    applyConVCTX(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &Xout,
       std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &shapeFnTimesWavefunctionMatrix,
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
              d_numberNodesPerElement,
              numberSphericalFunctions,
              &one,
              &shapeFnTimesWavefunctionMatrix[atomId][0],
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
    initialiseOperatorActionOnX(
      unsigned int kPointIndex,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &shapeFnTimesWavefunctionMatrix)
  {
    d_kPointIndex = kPointIndex;
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
        shapeFnTimesWavefunctionMatrix[atomId].setValue(0.0);
      }
    // d_SphericalFunctionKetTimesVectorParFlattened.setValue(0.0);
  }

  template <typename ValueType>
  void
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    initialiseFlattenedDataStructure(
      unsigned int numberWaveFunctions,
      std::map<
        unsigned int,
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
        &shapeFnTimesWavefunctionMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    d_numberWaveFunctions = numberWaveFunctions;
    // dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
    //   d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
    //   numberWaveFunctions,
    //   d_SphericalFunctionKetTimesVectorParFlattened);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
      numberWaveFunctions,
      sphericalFunctionKetTimesVectorParFlattened);
    shapeFnTimesWavefunctionMatrix.clear();
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
        shapeFnTimesWavefunctionMatrix[atomId].setValue(0.0);
      }
  }

  template <typename ValueType>
  const std::map<unsigned int, std::vector<ValueType>> &
  AtomicCenteredNonLocalOperator<ValueType, dftfe::utils::MemorySpace::HOST>::
    getScaledShapeFnTimesWaveFunction()
  {
    return (d_ShapeFnTimesWavefunction);
  }

} // namespace dftfe
