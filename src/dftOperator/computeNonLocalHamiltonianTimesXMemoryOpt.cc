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
// @author Phani Motamarri, Sambit Das
//


#ifdef USE_COMPLEX
template <unsigned int              FEOrder,
          unsigned int              FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro, memorySpace>::
  computeNonLocalHamiltonianTimesX(
    const distributedCPUMultiVec<std::complex<double>> &src,
    const unsigned int                                  numberWaveFunctions,
    distributedCPUMultiVec<std::complex<double>> &      dst,
    const double                                        scalar) const
{

}
#else
template <unsigned int              FEOrder,
          unsigned int              FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro, memorySpace>::
  computeNonLocalHamiltonianTimesX(const distributedCPUMultiVec<double> &src,
                                   const unsigned int numberWaveFunctions,
                                   distributedCPUMultiVec<double> &dst,
                                   const double                    scalar) const
{

}

#endif
