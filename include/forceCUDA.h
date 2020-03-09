// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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

#if defined(DFTFE_WITH_GPU)
#ifndef forceCUDA_H_
#define forceCUDA_H_

#include <headers.h>
#include <operatorCUDA.h>
#include <vectorUtilitiesCUDA.h>

namespace dftfe
{
   namespace forceCUDA
   {
     void computeELocWfcEshelbyTensorNonPeriodicH(operatorDFTCUDAClass & operatorMatrix,
                                                 const double * psiQuadValuesH,
                                                 const double * gradPsiQuadValuesXH,
                                                 const double * gradPsiQuadValuesYH,
                                                 const double * gradPsiQuadValuesZH,
                                                 const double * eigenValuesH,
                                                 const double * partialOccupanciesH,
                                                 const unsigned int numCells,
                                                 const unsigned int numQuads,
                                                 const unsigned int numPsi,
                                                 double * eshelbyTensorQuadValuesH00,
                                                 double * eshelbyTensorQuadValuesH10,
                                                 double * eshelbyTensorQuadValuesH11,
                                                 double * eshelbyTensorQuadValuesH20,
                                                 double * eshelbyTensorQuadValuesH21,
                                                 double * eshelbyTensorQuadValuesH22);

     void computeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
                                                  const thrust::device_vector<double> & psiQuadValuesD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesXD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesYD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesZD,
                                                  const thrust::device_vector<double> & eigenValuesD,
                                                  const thrust::device_vector<double> & partialOccupanciesD,
                                                  const unsigned int numCells,
                                                  const unsigned int numQuads,
                                                  const unsigned int numPsi,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD22);

     void computeNonLocalProjectorKetTimesPsiTimesVH(operatorDFTCUDAClass & operatorMatrix,
                                                     const double * X,
                                                     const unsigned int startingVecId,
                                                     const unsigned int BVec,
                                                     const unsigned int N,
                                                     double * projectorKetTimesPsiTimesVH);

     void interpolatePsiH(operatorDFTCUDAClass & operatorMatrix,
                          const double * X,
                          const unsigned int startingVecId,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuads,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          double * psiQuadsFlatH,
                          double * psiQuadsNLPFlatH,
                          double * gradPsiQuadsXFlatH,
                          double * gradPsiQuadsYFlatH,
                          double * gradPsiQuadsZFlatH,
                          const bool interpolateForNLPQuad=false);


     void interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
						  cudaVectorType & Xb,
						  const unsigned int BVec,
						  const unsigned int numCells,
						  const unsigned int numQuads,
                                                  const unsigned int numQuadsNLP,
						  const unsigned int numNodesPerElement,
                                                  const thrust::device_vector<double> & eigenValuesD,
                                                  const thrust::device_vector<double> & partialOccupanciesD,
                                                  const unsigned int innerBlockSizeEloc,
                                                  thrust::device_vector<double> & psiQuadsFlatD,
                                                  thrust::device_vector<double> & psiQuadsNLPFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsXFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsYFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsZFlatD,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD00,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD10,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD11,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD20,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD21,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD22,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD22,
                                                  const bool interpolateForNLPQuad=false);

     void interpolatePsiD(operatorDFTCUDAClass & operatorMatrix,
                          cudaVectorType & Xb,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuads,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          thrust::device_vector<double> & psiQuadsFlatD,
                          thrust::device_vector<double> & psiQuadsNLPFlatD,
                          thrust::device_vector<double> & gradPsiQuadsXFlatD,
                          thrust::device_vector<double> & gradPsiQuadsYFlatD,
                          thrust::device_vector<double> & gradPsiQuadsZFlatD,
                          const bool interpolateForNLPQuad=false);

     void interpolatePsiNLPD(operatorDFTCUDAClass & operatorMatrix,
                          cudaVectorType & Xb,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          thrust::device_vector<double> & psiQuadsNLPFlatD);

     void nlpPsiContractionD(operatorDFTCUDAClass & operatorMatrix,
			    const thrust::device_vector<double> & psiQuadValuesNLPD,
                            const thrust::device_vector<double> & partialOccupanciesD,
                            const double * projectorKetTimesVectorParFlattenedD,
                            const thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
                            const thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD,
                            const unsigned int numCells,
			    const unsigned int numQuadsNLP,
			    const unsigned int numPsi,
                            const unsigned int totalNonTrivialPseudoWfcs,
                            const unsigned int innerBlockSizeEnlp,
                            thrust::device_vector<double> & nlpContractionContributionD,
                            thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH);

      void gpuPortedForceKernelsAllD(operatorDFTCUDAClass & operatorMatrix,
                        cudaVectorType & cudaFlattenedArrayBlock,
                        cudaVectorType & projectorKetTimesVectorD,
                        const double * X,
			const thrust::device_vector<double> & eigenValuesD,
			const thrust::device_vector<double> & partialOccupanciesD,
                        const thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
                        const thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD, 
                        const unsigned int startingVecId,
                        const unsigned int N,
                        const unsigned int numPsi,
			const unsigned int numCells,
			const unsigned int numQuads,
                        const unsigned int numQuadsNLP,
                        const unsigned int numNodesPerElement,
                        const unsigned int totalNonTrivialPseudoWfcs,
			thrust::device_vector<double> & psiQuadsFlatD,
                        thrust::device_vector<double> & psiQuadsNLPFlatD,
			thrust::device_vector<double> & gradPsiQuadsXFlatD,
			thrust::device_vector<double> & gradPsiQuadsYFlatD,
			thrust::device_vector<double> & gradPsiQuadsZFlatD,
			thrust::device_vector<double> & eshelbyTensorContributionsD00,
			thrust::device_vector<double> & eshelbyTensorContributionsD10,
			thrust::device_vector<double> & eshelbyTensorContributionsD11,
			thrust::device_vector<double> & eshelbyTensorContributionsD20,
			thrust::device_vector<double> & eshelbyTensorContributionsD21,
			thrust::device_vector<double> & eshelbyTensorContributionsD22,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
			thrust::device_vector<double> & eshelbyTensorQuadValuesD22,
                        thrust::device_vector<double> & nlpContractionContributionD,
                        thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD,
                        const unsigned int innerBlockSizeEloc,
                        const unsigned int innerBlockSizeEnlp,
                        const bool isPsp,
                        const bool interpolateForNLPQuad=false);

      void gpuPortedForceKernelsAllH(operatorDFTCUDAClass & operatorMatrix,
                        const double * X,
			const double * eigenValuesH,
                        const double  fermiEnergy,
                        const unsigned int * nonTrivialIdToElemIdMapH,
                        const unsigned int * projecterKetTimesFlattenedVectorLocalIdsH, 
                        const unsigned int N,
			const unsigned int numCells,
			const unsigned int numQuads,
                        const unsigned int numQuadsNLP,
                        const unsigned int numNodesPerElement,
                        const unsigned int totalNonTrivialPseudoWfcs,
			double * eshelbyTensorQuadValuesH00,
			double * eshelbyTensorQuadValuesH10,
			double * eshelbyTensorQuadValuesH11,
			double * eshelbyTensorQuadValuesH20,
			double * eshelbyTensorQuadValuesH21,
			double * eshelbyTensorQuadValuesH22,
                        double * projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
                        const MPI_Comm & interBandGroupComm,
                        const bool isPsp,
                        const bool interpolateForNLPQuad=false);
   }
}
#endif
#endif
