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
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    const unsigned int numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();
    const unsigned int numCells = totalLocallyOwnedCells;
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
