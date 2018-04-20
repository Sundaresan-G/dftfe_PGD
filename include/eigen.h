// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

#ifndef eigen_H_
#define eigen_H_
#include "headers.h"
#include "constants.h"
#include "constraintMatrixInfo.h"
#include "operator.h"

namespace dftfe{

  typedef dealii::parallel::distributed::Vector<double> vectorType;
  template <unsigned int T> class dftClass;

  //
  //Define eigenClass class
  //
  template <unsigned int FEOrder>
    class eigenClass : public operatorClass
    {
      template <unsigned int T>
	friend class dftClass;

      template <unsigned int T>
	friend class symmetryClass;

    public:
      eigenClass(dftClass<FEOrder>* _dftPtr, const MPI_Comm &mpi_comm_replica);

      void HX(std::vector<vectorType> &src, 
	      std::vector<vectorType> &dst);
      

#ifdef ENABLE_PERIODIC_BC
      void HX(dealii::parallel::distributed::Vector<std::complex<double> > & src,
	      const unsigned int numberComponents,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
	      dealii::parallel::distributed::Vector<std::complex<double> > & dst);

      void XtHX(std::vector<vectorType> &src,
		std::vector<std::complex<double> > & ProjHam); 
#else
      void HX(dealii::parallel::distributed::Vector<double> & src,
	      const unsigned int numberComponents,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
	      dealii::parallel::distributed::Vector<double> & dst);

      void XtHX(std::vector<vectorType> &src,
		std::vector<double> & ProjHam);
#endif
   
      void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				    const vectorType & phi,
				    const vectorType & phiExt,
				    unsigned int j,
				    const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				    std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				    const vectorType & phi,
				    const vectorType & phiExt,
				    unsigned int j,
				    const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void reinitkPointIndex(unsigned int & kPointIndex);

      
      void init ();
	    
      //compute mass vector
      void computeMassVector();

      //precompute shapefunction gradient integral
      void preComputeShapeFunctionGradientIntegrals();

      //compute element Hamiltonian matrix
      void computeHamiltonianMatrix(unsigned int kPointIndex);


    private:
      void computeLocalHamiltonianTimesXMF(const dealii::MatrixFree<3,double>  &data,
					   std::vector<vectorType>  &dst, 
					   const std::vector<vectorType>  &src,
					   const std::pair<unsigned int,unsigned int> &cell_range) const;

      void computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
					    std::vector<vectorType>       &dst);
 
      void computeNonLocalHamiltonianTimesXMemoryOpt(const std::vector<vectorType> &src,
						     std::vector<vectorType>       &dst);  
  
      //pointer to dft class
      dftClass<FEOrder>* dftPtr;

      //FE data structres
      dealii::FE_Q<3>   FE;
 
      //data structures
      vectorType invSqrtMassVector,sqrtMassVector;

      dealii::Table<2, dealii::VectorizedArray<double> > vEff;
      dealii::Table<3, dealii::VectorizedArray<double> > derExcWithSigmaTimesGradRho;

      //precomputed data for the Hamiltonian matrix
      std::vector<std::vector<dealii::VectorizedArray<double> > > d_cellShapeFunctionGradientIntegral;
      std::vector<double> d_shapeFunctionValue;

#ifdef ENABLE_PERIODIC_BC
      std::vector<std::vector<std::complex<double> > > d_cellHamiltonianMatrix;
      
      void computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
					 const int numberWaveFunctions,
					 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					 dealii::parallel::distributed::Vector<std::complex<double> > & dst) const;


#else
      std::vector<std::vector<double> > d_cellHamiltonianMatrix;

      void computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
					 const int numberWaveFunctions,
					 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					 dealii::parallel::distributed::Vector<double> & dst) const;

#endif

      //
      //access to matrix-free cell data
      //
      const int d_numberNodesPerElement,d_numberMacroCells;
      std::vector<unsigned int> d_macroCellSubCellMap;

      //parallel objects
      const MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;
      dealii::ConditionalOStream   pcout;

      //compute-time logger
      dealii::TimerOutput computing_timer;

      //mutex thread for managing multi-thread writing to XHXvalue
      mutable dealii::Threads::Mutex  assembler_lock;

      //d_kpoint index for which Hamiltonian is computed
      unsigned int d_kPointIndex;

    };


}
#endif
