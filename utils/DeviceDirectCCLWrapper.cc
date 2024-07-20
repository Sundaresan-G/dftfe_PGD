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
// @author Sambit Das, David M. Rogers
//

#if defined(DFTFE_WITH_DEVICE)
#  include <iostream>

#  include <deviceDirectCCLWrapper.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceAPICalls.h>
#  include <Exceptions.h>
#  if defined(DFTFE_WITH_CUDA_NCCL)
#    include <nccl.h>
#  elif defined(DFTFE_WITH_HIP_RCCL)
#    include <rccl.h>
#  endif

namespace dftfe
{
  namespace utils
  {
    DeviceCCLWrapper::DeviceCCLWrapper()
      : d_mpiComm(MPI_COMM_NULL)
    {}

    // Ensure that mpiCommDomain calls it first as static variables need to be initialized
    void
    DeviceCCLWrapper::init(const MPI_Comm &mpiComm, const bool useDCCL, int selector /*= 0*/)
    {
      MPICHECK(MPI_Comm_dup(mpiComm, &d_mpiComm));
      MPICHECK(MPI_Comm_size(mpiComm, &totalRanks));
      MPICHECK(MPI_Comm_rank(mpiComm, &myRank));
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (!ncclCommInit && useDCCL)
        {
          ncclIdPtr   = new ncclUniqueId;
          ncclCommPtr = new ncclComm_t;
          if (myRank == 0)
            ncclGetUniqueId(ncclIdPtr);
          MPICHECK(
            MPI_Bcast(ncclIdPtr, sizeof(*ncclIdPtr), MPI_BYTE, 0, d_mpiComm));
          NCCLCHECK(
            ncclCommInitRank(ncclCommPtr, totalRanks, *ncclIdPtr, myRank));

          // Make NCCL calls non-blocking
          // ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
          // config.blocking = 0;
          // NCCLCHECK(
            // ncclCommInitRankConfig(ncclCommPtr, totalRanks, *ncclIdPtr, myRank, &config));

          // ncclCommInitRankConfig(ncclCommPtr, totalRanks, *ncclIdPtr, myRank, &config);
          // ncclResult_t state;
          // do {
          //   ncclCommGetAsyncError(*ncclCommPtr, &state);
          // } while (state == ncclInProgress);

          ncclCommInit = true;
        }

        if (selector != 0 && useDCCL){
          dcclCommSelector = selector;
          ncclIdPvtPtr = new ncclUniqueId;
          ncclCommPvtPtr = new ncclComm_t;
          if (myRank == 0)
            ncclGetUniqueId(ncclIdPvtPtr);
          MPICHECK(
            MPI_Bcast(ncclIdPvtPtr, sizeof(*ncclIdPvtPtr), MPI_BYTE, 0, d_mpiComm));
          NCCLCHECK(
            ncclCommInitRank(ncclCommPvtPtr, totalRanks, *ncclIdPvtPtr, myRank));

        }
#  endif
      if (!commStreamCreated)
        {
          dftfe::utils::deviceStreamCreate(&d_deviceCommStream, true);
          commStreamCreated = true;
        }
    }

    DeviceCCLWrapper::~DeviceCCLWrapper()
    {
      if (d_mpiComm != MPI_COMM_NULL)
        MPI_Comm_free(&d_mpiComm);
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          ncclCommFinalize(*ncclCommPtr);
          ncclCommDestroy(*ncclCommPtr);
          delete ncclCommPtr;
          delete ncclIdPtr;
          ncclCommInit = false;
        }

      if (dcclCommSelector != 0){
        ncclCommFinalize(*ncclCommPvtPtr);
        ncclCommDestroy(*ncclCommPvtPtr);
        delete ncclCommPvtPtr;
      }
#  endif
      if (commStreamCreated){
        dftfe::utils::deviceStreamDestroy(d_deviceCommStream);
        commStreamCreated = false;        
      }
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const float *   send,
                                                   float *         recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }
            
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif
      return 0;
    }

    template <typename NumberType>
    int
    DeviceCCLWrapper::deviceDirectAllToAllWrapper(const NumberType *        send,
                                                  size_t         sendCount,
                                                  NumberType *              recv,
                                                  size_t         recvCount,
                                                  deviceStream_t stream /*= 0*/,
                                                  bool useDCCL /*= true*/)
    {
      unsigned int sendTo = myRank;
      unsigned int recvFrom = myRank;

      size_t sendOffset = (size_t)sendTo * sendCount;
      size_t recvOffset = (size_t)recvFrom * recvCount;

      // use D2D copy for the first one
      dftfe::utils::deviceMemcpyAsyncD2D(
        recv + recvOffset,
        send + sendOffset,
        sendCount * sizeof(NumberType),
        stream);

      // Printing line and file to show no error
      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(MPI_COMM_WORLD);
      // fflush(stdout);
      // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      //   {
      //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
      //           << std::endl;
      //   }
      // fflush(stdout);

#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      // Printing line and file to show no error
      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(MPI_COMM_WORLD);
      // fflush(stdout);
      // // Get nccl rank
      // int ncclRank, mpiRank;
      // ncclCommUserRank(*ncclCommPtr, &ncclRank);
      // MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
      // std::cout << "ncclRank: " << ncclRank << " mpiRank: " << mpiRank 
      //           << " myRank: " << myRank << " totalRanks: " << totalRanks
      //           << std::endl;
      // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      //   {
      //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
      //           << std::endl;
      //     // print parameter values
      //     std::cout << "sendCount: " << sendCount << std::endl;
      //     std::cout << "recvCount: " << recvCount << std::endl;
      //     std::cout << "sendTo: " << sendTo << std::endl;
      //     std::cout << "recvFrom: " << recvFrom << std::endl;

      //     // print NumberType
      //     if (std::is_same<NumberType, float>::value)
      //       std::cout << "NumberType: float" << std::endl;
      //     else if (std::is_same<NumberType, double>::value)
      //       std::cout << "NumberType: double" << std::endl;
      //   }
      // fflush(stdout);

      // select ncclDouble or ncclFloat based on NumberType
      if (ncclCommInit && useDCCL)
        {
          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          ncclDataType_t ncclType = ncclDouble;
          if (std::is_same<NumberType, float>::value)
            ncclType = ncclFloat;
          
          // NCCLCHECK(ncclGroupStart());
          for (unsigned int i = 1; i < totalRanks; i++)
            {
              // Printing line and file to show no error
              // dftfe::utils::deviceSynchronize();
              // MPI_Barrier(MPI_COMM_WORLD);
              // fflush(stdout);
              // ncclCommUserRank(*ncclCommPtr, &ncclRank);
              // MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
              // std::cout << "ncclRank: " << ncclRank << " mpiRank: " << mpiRank << std::endl;
              // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              //   {
              //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
              //           << std::endl;
              //     // print parameter values
              //     std::cout << "sendCount: " << sendCount << std::endl;
              //     std::cout << "recvCount: " << recvCount << std::endl;
              //     std::cout << "sendTo: " << sendTo << std::endl;
              //     std::cout << "recvFrom: " << recvFrom << std::endl;
              //   }
              // fflush(stdout);

              sendTo += i;
              sendTo %= totalRanks;
              recvFrom += (totalRanks - i);
              recvFrom %= totalRanks;

              sendOffset = (size_t)sendTo * sendCount;
              recvOffset = (size_t)recvFrom * recvCount;

              // Printing line and file to show no error
              // dftfe::utils::deviceSynchronize();
              // MPI_Barrier(MPI_COMM_WORLD);
              // fflush(stdout);
              // ncclCommUserRank(*ncclCommPtr, &ncclRank);
              // MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
              // std::cout << "ncclRank: " << ncclRank << " mpiRank: " << mpiRank 
              //           << " sendOffset: " << sendOffset
              //           << " recvOffset: " << recvOffset
              //           << " sendCount: " << sendCount
              //           << " recvCount: " << recvCount << std::endl;
              // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              //   {
              //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
              //           << std::endl;
              //     // print parameter values
                  // std::cout << "sendCount: " << sendCount << std::endl;
                  // std::cout << "recvCount: " << recvCount << std::endl;
                  // std::cout << "sendTo: " << sendTo << std::endl;
                  // std::cout << "recvFrom: " << recvFrom << std::endl;
              //   }
              // fflush(stdout);

              // if (sendOffset + sendCount > totalNumRows * totalNumCols){
              //   sendCount = totalNumRows * totalNumCols - sendOffset;
              // }
              // if (recvOffset + recvCount > totalNumRows * totalNumCols)
              //   recvCount = totalNumRows * totalNumCols - recvOffset;

              NCCLCHECK(ncclGroupStart());
              
                NCCLCHECK(ncclSend((const void *)(send + sendOffset),
                                  sendCount,
                                  ncclType,
                                  sendTo,
                                  comm,
                                  stream));
                NCCLCHECK(ncclRecv((void *)(recv + recvOffset),
                                    recvCount,
                                    ncclType,
                                    recvFrom,
                                    comm,
                                    stream));

              NCCLCHECK(ncclGroupEnd());
            }
          // NCCLCHECK(ncclGroupEnd());
        } else
#endif
        {
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
          // Printing line and file to show no error
          // dftfe::utils::deviceSynchronize();
          // MPI_Barrier(MPI_COMM_WORLD);
          // fflush(stdout);
          // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          //   {
          //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
          //           << std::endl;
          //   }
          // fflush(stdout);

          dftfe::utils::deviceStreamSynchronize(stream);
          for (unsigned int i = 1; i < totalRanks; i++)
            {
              sendTo += i;
              sendTo %= totalRanks;
              recvFrom += (totalRanks - i);
              recvFrom %= totalRanks;

              sendOffset = (size_t)sendTo * sendCount;
              recvOffset = (size_t)recvFrom * recvCount;

              // if (sendOffset + sendCount > totalNumRows * totalNumCols)
              //   sendCount = totalNumRows * totalNumCols - sendOffset;
              // if (recvOffset + recvCount > totalNumRows * totalNumCols)
              //   recvCount = totalNumRows * totalNumCols - recvOffset;

              MPICHECK(MPI_Sendrecv(send + sendOffset,
                                    sendCount,
                                    dataTypes::mpi_type_id(send),
                                    sendTo,
                                    0,
                                    recv + recvOffset,
                                    recvCount,
                                    dataTypes::mpi_type_id(recv),
                                    recvFrom,
                                    0,
                                    d_mpiComm,
                                    MPI_STATUS_IGNORE));
            }
#  endif
        }

      return 0;
    }

  // initialize alltoall templates
  template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const double * send, size_t sendCount, double * recv, size_t recvCount, deviceStream_t stream, bool useDCCL);

  template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const float * send, size_t sendCount, float * recv, size_t recvCount, deviceStream_t stream, bool useDCCL);


    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const double *  send,
                                                   double *        recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif
      return 0;
    }


    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<double> *send,
      std::complex<double> *      recv,
      int                         size,
      double *                    tempReal,
      double *                    tempImag,
      deviceStream_t &            stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size,
                                                               send,
                                                               tempReal,
                                                               tempImag);
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal,
                                  (void *)tempReal,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag,
                                  (void *)tempImag,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclGroupEnd());

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size,
                                                               tempReal,
                                                               tempImag,
                                                               recv);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif


      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<float> *send,
      std::complex<float> *      recv,
      int                        size,
      float *                    tempReal,
      float *                    tempImag,
      deviceStream_t &           stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }
          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size,
                                                               send,
                                                               tempReal,
                                                               tempImag);
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal,
                                  (void *)tempReal,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag,
                                  (void *)tempImag,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size,
                                                               tempReal,
                                                               tempImag,
                                                               recv);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif

      return 0;
    }


    int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const double *  send1,
      const float *   send2,
      double *        recv1,
      float *         recv2,
      int             size1,
      int             size2,
      deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)send1,
                                  (void *)recv1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)send2,
                                  (void *)recv2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send1 == recv1 && send2 == recv2)
            {
              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Allreduce(send1,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(send2,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
        }
#  endif
      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const std::complex<double> *send1,
      const std::complex<float> * send2,
      std::complex<double> *      recv1,
      std::complex<float> *       recv2,
      int                         size1,
      int                         size2,
      double *                    tempReal1,
      float *                     tempReal2,
      double *                    tempImag1,
      float *                     tempImag2,
      deviceStream_t &            stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size1,
                                                               send1,
                                                               tempReal1,
                                                               tempImag1);

          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size2,
                                                               send2,
                                                               tempReal2,
                                                               tempImag2);

          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal1,
                                  (void *)tempReal1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag1,
                                  (void *)tempImag1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempReal2,
                                  (void *)tempReal2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag2,
                                  (void *)tempImag2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclGroupEnd());

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size1,
                                                               tempReal1,
                                                               tempImag1,
                                                               recv1);

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size2,
                                                               tempReal2,
                                                               tempImag2,
                                                               recv2);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send1 == recv1 && send2 == recv2)
            {
              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Allreduce(send1,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(send2,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
        }
#  endif
      return 0;
    }
  } // namespace utils
} // namespace dftfe
#endif
