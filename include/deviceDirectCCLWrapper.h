// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das, David M. Rogers

#if defined(DFTFE_WITH_DEVICE)
#  ifndef deviceDirectCCLWrapper_h
#    define deviceDirectCCLWrapper_h
#    include <complex>
#    include <mpi.h>
#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>
#    if defined(DFTFE_WITH_CUDA_NCCL)
#      include <nccl.h>
#    elif defined(DFTFE_WITH_HIP_RCCL)
#      include <rccl.h>
#    elif defined(DFTFE_WITH_SYCL_ONECCL)
#      include <oneapi/ccl.hpp>
#    endif

namespace dftfe
{
  namespace utils
  {
#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
#      define NCCLCHECK(cmd)                              \
        do                                                \
          {                                               \
            ncclResult_t r = cmd;                         \
            if (r != ncclSuccess)                         \
              {                                           \
                printf("Failed, NCCL error %s:%d '%s'\n", \
                       __FILE__,                          \
                       __LINE__,                          \
                       ncclGetErrorString(r));            \
                exit(EXIT_FAILURE);                       \
              }                                           \
        } while (0)
#    endif

#    if defined(DFTFE_WITH_SYCL_ONECCL)
#      define ONECCLCHECK(cmd)                              \
        do                                                  \
          {                                                 \
            try                                             \
              {                                             \
                cmd;                                        \
              }                                             \
            catch (const ccl::exception &e)                 \
              {                                             \
                printf("Failed, oneCCL error %s:%d '%s'\n", \
                       __FILE__,                            \
                       __LINE__,                            \
                       e.what());                           \
                exit(EXIT_FAILURE);                         \
              }                                             \
        } while (0)
#    endif

    /**
     *  @brief Wrapper class for Device Direct collective communications library.
     *  Adapted from
     * https://code.ornl.gov/99R/olcf-cookbook/-/blob/develop/comms/nccl_allreduce.rst
     *
     *  @author Sambit Das, David M. Rogers
     */
    class DeviceCCLWrapper
    {
    public:
      DeviceCCLWrapper();

      void
      init(const MPI_Comm &mpiComm,
           const bool      useDCCL,
           const bool      setAsDefaultP2PComm = false);

#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL) || \
      defined(DFTFE_WITH_SYCL_ONECCL)
      /**
       * Initialize the job-wide device CCL communicator used as the parent for
       * all application communicator splits. This must be called collectively
       * on mpiCommParent before init() is called on any device CCL wrapper.
       */
      static void
      initRoot(const MPI_Comm &mpiCommParent, const bool useDCCL);
#    endif

      ~DeviceCCLWrapper();

      dftfe::Int
      deviceDirectAllReduceWrapper(const float    *send,
                                   float          *recv,
                                   dftfe::Int      size,
                                   deviceStream_t &stream);


      dftfe::Int
      deviceDirectAllReduceWrapper(const double   *send,
                                   double         *recv,
                                   dftfe::Int      size,
                                   deviceStream_t &stream);


      dftfe::Int
      deviceDirectAllReduceWrapper(const std::complex<double> *send,
                                   std::complex<double>       *recv,
                                   dftfe::Int                  size,
                                   deviceStream_t             &stream);

      dftfe::Int
      deviceDirectAllReduceWrapper(const std::complex<float> *send,
                                   std::complex<float>       *recv,
                                   dftfe::Int                 size,
                                   deviceStream_t            &stream);


      dftfe::Int
      deviceDirectAllReduceMixedPrecGroupWrapper(const double   *send1,
                                                 const float    *send2,
                                                 double         *recv1,
                                                 float          *recv2,
                                                 dftfe::Int      size1,
                                                 dftfe::Int      size2,
                                                 deviceStream_t &stream);

      dftfe::Int
      deviceDirectAllReduceMixedPrecGroupWrapper(
        const std::complex<double> *send1,
        const std::complex<float>  *send2,
        std::complex<double>       *recv1,
        std::complex<float>        *recv2,
        dftfe::Int                  size1,
        dftfe::Int                  size2,
        deviceStream_t             &stream);

      template <typename NumberType>
      int 
      deviceDirectAllToAllWrapper(const NumberType *  send,
                                  dftfe::uInt            sendCount,
                                  NumberType *       recv,
                                  dftfe::uInt            recvCount,
                                  deviceStream_t stream = dftfe::utils::defaultStream,
                                  bool            useDCCL = true);

#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      // Non-owning alias used by the legacy NCCL/RCCL direct P2P path.
      inline static ncclComm_t *dcclCommPtr = nullptr;
#    endif

      inline static bool                         dcclCommInit;
      inline static dftfe::utils::deviceStream_t d_deviceCommStream;
      inline static bool                         commStreamCreated;
      inline static dftfe::Int d_deviceDirectDCCLInstanceCounter;

    private:
#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      inline static ncclUniqueId dcclRootId;
      inline static ncclComm_t   dcclRootComm = nullptr;
      ncclComm_t                 d_ncclComm    = nullptr;
#    elif defined(DFTFE_WITH_SYCL_ONECCL)
      inline static std::shared_ptr<ccl::kvs> dcclRootIdPtr;
      inline static std::shared_ptr<ccl::communicator> dcclRootCommPtr;
      std::shared_ptr<ccl::communicator> d_oneCCLCommPtr;
#    endif
#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL) || \
      defined(DFTFE_WITH_SYCL_ONECCL)
      inline static MPI_Comm dcclMpiCommRoot = MPI_COMM_NULL;
#    endif
      int      myRank;
      int      totalRanks;
      MPI_Comm d_mpiComm;
    };
  } // namespace utils
} // namespace dftfe

#  endif
#endif
