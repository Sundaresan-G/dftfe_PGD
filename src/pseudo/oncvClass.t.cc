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
#include "oncvClass.h"

namespace dftfe
{
  template <typename ValueType>
  oncvClass<
    ValueType>::oncvClass()
  {}

  template <typename ValueType>
  void
  oncvClass<
    ValueType>::createAtomCenteredSphericalFunctionsForDensities()
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



    // for (unsigned int i = 0; i < d_nOMPThreads; i++)
    //   {
    //     // d_atomicCoreDensityVector[i] = (atomicCoreDensityMap);
    //     // d_atomicValenceDensityVector[i] = (atomicValenceDensityMap);
    //     std::map<unsigned int, AtomCenteredSphericalFunctionBase*>
    //     atomValenceDensityMap; for (std::set<unsigned int>::iterator it =
    //     d_atomTypes.begin();
    //          it != d_atomTypes.end();
    //          ++it)
    //       {
    //         std::cout<<"Debug: Line 80"<<std::endl;
    //         *(atomValenceDensityMap[*it]) =
    //         (*(atomicValenceDensityMap[*it]));
    //       }
    //       std::cout<<"Debug: Line 83"<<std::endl;
    //       d_atomicValenceDensityVector[i] = atomValenceDensityMap;
    //   }
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
    std::vector<unsigned int> imageLoc(int(atomLocations.size() / 3), 0.0);
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
  double
  oncvClass<ValueType>::getRadialValenceDensity(
    unsigned int Zno,
    double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value =
      d_atomicValenceDensityVector[threadId][Zno]->getRadialValue(rad);

    return (Value);
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::getRadialValenceDensity(
    unsigned int         Zno,
    double               rad,
    std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicValenceDensityVector[threadId][Zno]->getDerivativeValue(rad);
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRmaxValenceDensity(
    unsigned int Zno)
  {
    return (d_atomicValenceDensityVector[0][Zno]->getRadialCutOff());
  }
  template <typename ValueType>
  double
  oncvClass<ValueType>::getRmaxCoreDensity(
    unsigned int Zno)
  {
    return (d_atomicCoreDensityMap[Zno]->getRadialCutOff());
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRadialCoreDensity(
    unsigned int Zno,
    double       rad)
  {
    unsigned int threadId = omp_get_thread_num();
    double       Value    = d_atomicCoreDensityMap[Zno]->getRadialValue(rad);
    return (Value);
  }
  template <typename ValueType>
  void
  oncvClass<ValueType>::getRadialCoreDensity(
    unsigned int         Zno,
    double               rad,
    std::vector<double> &Val)
  {
    unsigned int threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicCoreDensityMap[Zno]->getDerivativeValue(rad);
  }

  template <typename ValueType>
  double
  oncvClass<ValueType>::getRadialLocalPseudo(
    unsigned int Zno,
    double       rad)
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
  oncvClass<ValueType>::coreNuclearDensityPresent(
    unsigned int Zno)
  {
    return (d_atomTypeCoreFlagMap[Zno]);
  }
} // namespace dftfe
