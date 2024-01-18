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
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionBase.h"
#include <memory>
#include <MemorySpaceType.h>
#include "FEBasisOperations.h"
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>


namespace dftfe
{
  class AtomCenteredSphericalFunctionContainer
  {
  public:
    // the init functions.
    // The listOfSphericalFunctions is a map from std::pair to
    // AtomCenteredSphericalFunctionBase the key to this map is a pair of atomic
    // number and a composite index. The composite index maps uniquely to an
    // orbital.
    void
    init(const std::vector<unsigned int> &atomicNumbers,
         const std::map<std::pair<unsigned int, unsigned int>,
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>>
           &listOfSphericalFunctions);

    void
    initaliseCoordinates(const std::vector<double> &             atomCoords,
                         const std::vector<std::vector<double>> &periodicCoords,
                         const std::vector<int> &                imageIds);
    unsigned int
    getNumAtomCentersSize();

    const std::vector<double> &
    getAtomCoordinates() const;

    const std::map<unsigned int, std::vector<double>> &
    getPeriodicImageCoordinatesList() const;

    // This functions returns the number of spherical functions associated with
    // an atomic number.
    // If the atomic number does not exist, it returns a zero.
    unsigned int
    getTotalNumberOfSphericalFunctionsPerAtom(unsigned int atomicNumber);

    // This functions returns the number of spherical functions associated with
    // an atomic number.
    // If the atomic number does not exist, it returns a zero.
    unsigned int
    getTotalNumberOfRadialSphericalFunctionsPerAtom(unsigned int atomicNumber);

    unsigned int
    getTotalNumberOfSphericalFunctions();

    unsigned int
    getTotalNumberOfSphericalFunctionsInCurrentProcessor();

    unsigned int
    getMaximumNumberOfSphericalFunctions();

    void
    getTotalAtomsAndNonLocalElementsInCurrentProcessor(
      unsigned int &             totalAtomsInCurrentProcessor,
      unsigned int &             totalNonLocalElements,
      std::vector<unsigned int> &numberCellsAccumNonLocalAtoms);

    unsigned int
    getTotalNumberOfRadialSphericalFunctions();

    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>> &
    getSphericalFunctions() const;

    const std::vector<unsigned int> &
    getAtomicNumbers() const;

    const std::vector<unsigned int> &
    getAtomIdsInCurrentProcess() const;

    const unsigned int
    getTotalSphericalFunctionIndexStart(unsigned int Zno, unsigned int alpha);
    // COmputes the sparsity Pattern for the compact support Fn
    // cutOffVal the max/min value to consider to be part of copact support
    // cutOffType = 0 based on Fn Value, cutOffType = 1 based on Distance from
    // atom
    template <typename NumberType>
    void
    computeSparseStructure(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<NumberType,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureIndex,
      const double       cutOffVal  = 1.0E-8,
      const unsigned int cutOffType = 0);


    std::vector<std::vector<unsigned int>> d_elementIndexesInAtomCompactSupport;
    void
    setImageCoordinates(const std::vector<int> &                imageIds,
                        const std::vector<std::vector<double>> &periodicCoords);



    const std::vector<int> &
    getAtomIdsInElement(unsigned int iElem);

    const std::map<unsigned int, std::vector<int>> &
    getSparsityPattern();

    bool
    atomSupportInElement(unsigned int iElem);

  private:
    // A flattened vector that stores the coordinates of the atoms of interest
    // in the unit cell
    // Coord of atom I is stored at 3*I +0 ( x-coord),3*I+1 ( y-coord),3*I+2 (
    // z-coord)
    std::vector<double> d_atomCoords;

    // A vector of size = number of atoms of interest
    // the Ith atom in d_atomicNumbers has its coordinates
    // in d_atomCoords[3*I+0], d_atomCoords[3*I+1], d_atomCoords[3*I+2]
    std::vector<unsigned int> d_atomicNumbers;

    // This maps the atom I in the unit cell to all its image atoms.
    // number of image atoms of Ith atom = d_periodicImageCoord[I].size()/ dim
    // with dim=3 The coordinates are stored as a flattened vector
    std::map<unsigned int, std::vector<double>> d_periodicImageCoord;


    // This maps, from std::pair<atomic number, \alpha> to S_{z,\alpha},
    // where \alpha is the index for unique radial function
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_sphericalFunctionsContainer;
    // Stores the number of  distinct Radial Functions for a particular AtomType
    std::map<unsigned int, unsigned int> d_numRadialSphericalFunctions;
    // Stores the number of  distinct  Functions include m for a particular
    // AtomType
    std::map<unsigned int, unsigned int> d_numSphericalFunctions;
    // This maps is between atomId in unit cell and the sparsity pattern of the
    // atom and its images in the unitcell domain.
    std::map<unsigned int, std::vector<int>> d_sparsityPattern;
    //
    std::vector<std::vector<dealii::CellId>> d_elementIdsInAtomCompactSupport;
    // std::vector<std::vector<unsigned int>>
    // d_elementIndexesInAtomCompactSupport;
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
                                  d_elementOneFieldIteratorsInAtomCompactSupport;
    std::vector<unsigned int>     d_AtomIdsInCurrentProcess;
    std::vector<std::vector<int>> d_AtomIdsInElement;
    std::map<unsigned int, std::vector<unsigned int>>
      d_totalSphericalFunctionIndexStart;

  }; // end of class AtomCenteredSphericalFunctionContainerBase
} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H
