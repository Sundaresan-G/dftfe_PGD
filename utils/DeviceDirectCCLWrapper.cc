// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
#  include <DeviceKernelLauncherHelpers.h>
#  include <DeviceAPICalls.h>
#  include <Exceptions.h>

namespace dftfe
{
  namespace utils
  {
    DeviceCCLWrapper::DeviceCCLWrapper()
      : d_mpiComm(MPI_COMM_NULL)
    {
      d_deviceDirectDCCLInstanceCounter++;
    }

    // Ensure that mpiCommDomain calls it first as static variables need to be initialized
    void
    DeviceCCLWrapper::init(const MPI_Comm &mpiComm, const bool useDCCL, int selector /*= 0*/)
    {
      MPICHECK(MPI_Comm_dup(mpiComm, &d_mpiComm));
      MPICHECK(MPI_Comm_size(mpiComm, &totalRanks));
      MPICHECK(MPI_Comm_rank(mpiComm, &myRank));
      if (!commStreamCreated)
        {
          dftfe::utils::deviceStreamCreate(d_deviceCommStream, true);
          commStreamCreated = true;
        }

#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (!dcclCommInit && useDCCL)
        {
          ncclIdPtr   = new ncclUniqueId;
          ncclCommPtr = new ncclComm_t;
          if (myRank == 0)
            ncclGetUniqueId(ncclIdPtr);
          MPICHECK(
            MPI_Bcast(ncclIdPtr, sizeof(*ncclIdPtr), MPI_BYTE, 0, d_mpiComm));
          NCCLCHECK(
            ncclCommInitRank(ncclCommPtr, totalRanks, *ncclIdPtr, myRank));
          dcclCommInit = true;
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

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (!dcclCommInit && useDCCL)
        {
          ccl::kvs::address_type onecclIdAddr;
          if (myRank == 0)
            {
              onecclIdPtr  = ccl::create_main_kvs();
              onecclIdAddr = onecclIdPtr->get_address();
              MPICHECK(MPI_Bcast(onecclIdAddr.data(),
                                 onecclIdAddr.size(),
                                 MPI_BYTE,
                                 0,
                                 d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Bcast(onecclIdAddr.data(),
                                 onecclIdAddr.size(),
                                 MPI_BYTE,
                                 0,
                                 d_mpiComm));
              onecclIdPtr = ccl::create_kvs(onecclIdAddr);
            }

          ccl::vector_class<ccl::pair_class<int, ccl::device>> rankDeviceMap;
          rankDeviceMap.push_back(
            {myRank, ccl::create_device(dftfe::utils::syclDevice)});
          auto onecclContext = ccl::create_context(dftfe::utils::syclContext);
          auto comms         = ccl::create_communicators(totalRanks,
                                                 rankDeviceMap,
                                                 onecclContext,
                                                 onecclIdPtr);
          onecclCommPtr =
            std::make_shared<ccl::communicator>(std::move(comms[0]));
          dcclCommInit = true;
        }
#  endif
    }

    DeviceCCLWrapper::~DeviceCCLWrapper()
    {
      if (d_mpiComm != MPI_COMM_NULL)
        MPI_Comm_free(&d_mpiComm);
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          ncclCommFinalize(*ncclCommPtr);
          ncclCommDestroy(*ncclCommPtr);
          delete ncclCommPtr;
          delete ncclIdPtr;
          dcclCommInit = false;
        }

      if (dcclCommSelector != 0){
        ncclCommFinalize(*ncclCommPvtPtr);
        ncclCommDestroy(*ncclCommPvtPtr);
        delete ncclCommPvtPtr;
      }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          onecclCommPtr.reset();
          onecclIdPtr.reset();
        }
#  endif

      d_deviceDirectDCCLInstanceCounter--;
      if (commStreamCreated && d_deviceDirectDCCLInstanceCounter == 0)
        dftfe::utils::deviceStreamDestroy(d_deviceCommStream);
        commStreamCreated = false;        
    }

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const float    *send,
                                                   float          *recv,
                                                   dftfe::Int      size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
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

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size,
                                         ccl::datatype::float32,
                                         ccl::reduction::sum,
                                         *onecclCommPtr,
                                         devStream));
          e.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const double   *send,
                                                   double         *recv,
                                                   dftfe::Int      size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
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

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size,
                                         ccl::datatype::float64,
                                         ccl::reduction::sum,
                                         *onecclCommPtr,
                                         devStream));
          e.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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


    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<double> *send,
      std::complex<double>       *recv,
      dftfe::Int                  size,
      deviceStream_t             &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size * 2,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size * 2,
                                         ccl::datatype::float64,
                                         ccl::reduction::sum,
                                         *onecclCommPtr,
                                         devStream));
          e.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<float> *send,
      std::complex<float>       *recv,
      dftfe::Int                 size,
      deviceStream_t            &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }
          
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size * 2,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size * 2,
                                         ccl::datatype::float32,
                                         ccl::reduction::sum,
                                         *onecclCommPtr,
                                         devStream));
          e.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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


    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const double   *send1,
      const float    *send2,
      double         *recv1,
      float          *recv2,
      dftfe::Int      size1,
      dftfe::Int      size2,
      deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
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
#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));

          ccl::event e1, e2;
          ONECCLCHECK(ccl::group_start());

          ONECCLCHECK(e1 = ccl::allreduce((const void *)send1,
                                          (void *)recv1,
                                          size1,
                                          ccl::datatype::float64,
                                          ccl::reduction::sum,
                                          *onecclCommPtr,
                                          devStream));

          ONECCLCHECK(e2 = ccl::allreduce((const void *)send2,
                                          (void *)recv2,
                                          size2,
                                          ccl::datatype::float32,
                                          ccl::reduction::sum,
                                          *onecclCommPtr,
                                          devStream));

          ONECCLCHECK(ccl::group_end());
          e1.wait();
          e2.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const std::complex<double> *send1,
      const std::complex<float>  *send2,
      std::complex<double>       *recv1,
      std::complex<float>        *recv2,
      dftfe::Int                  size1,
      dftfe::Int                  size2,
      deviceStream_t             &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }

          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)send1,
                                  (void *)recv1,
                                  size1 * 2,
                                  ncclDouble,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)send2,
                                  (void *)recv2,
                                  size2 * 2,
                                  ncclFloat,
                                  ncclSum,
                                  comm,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));

          ccl::event e1, e2;
          ONECCLCHECK(ccl::group_start());

          ONECCLCHECK(e1 = ccl::allreduce((const void *)send1,
                                          (void *)recv1,
                                          size1 * 2,
                                          ccl::datatype::float64,
                                          ccl::reduction::sum,
                                          *onecclCommPtr,
                                          devStream));

          ONECCLCHECK(e2 = ccl::allreduce((const void *)send2,
                                          (void *)recv2,
                                          size2 * 2,
                                          ccl::datatype::float32,
                                          ccl::reduction::sum,
                                          *onecclCommPtr,
                                          devStream));

          ONECCLCHECK(ccl::group_end());
          e1.wait();
          e2.wait();
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    template <typename NumberType>
    int
    DeviceCCLWrapper::deviceDirectAllToAllWrapper(const NumberType *        send,
                                                  dftfe::uInt         sendCount,
                                                  NumberType *              recv,
                                                  dftfe::uInt         recvCount,
                                                  deviceStream_t stream /*= 0*/,
                                                  bool useDCCL /*= true*/)
    {
      unsigned int sendTo = myRank;
      unsigned int recvFrom = myRank;

      dftfe::uInt sendOffset = (dftfe::uInt)sendTo * sendCount;
      dftfe::uInt recvOffset = (dftfe::uInt)recvFrom * recvCount;

      // use D2D copy for the first one
      dftfe::utils::deviceMemcpyAsyncD2D(
        makeDataTypeDeviceCompatible(recv + recvOffset),
        makeDataTypeDeviceCompatible(send + sendOffset),
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
      // {
      //   int size = totalRanks, this_process = myRank;
      //   // MPI_Comm_size(intrapoolcomm, &size);
      //   // MPI_Comm_rank(intrapoolcomm, &this_process);
      //   std::cout << "Out of " << size << " processes, process " << this_process << " reached line " << __LINE__ << " of file " << __FILE__ << std::endl;
        
      // }

      if (dcclCommInit && useDCCL)
        {

          sendCount = sendCount * sizeof(NumberType);
          recvCount = recvCount * sizeof(NumberType);

          // {
          //   int size = totalRanks, this_process = myRank;
          //   // MPI_Comm_size(intrapoolcomm, &size);
          //   // MPI_Comm_rank(intrapoolcomm, &this_process);
          //   std::cout << "Out of " << size << " processes, process " << this_process << " reached line " << __LINE__ << " of file " << __FILE__ << std::endl;
            
          // }

          ncclComm_t comm = *ncclCommPtr;
          if (dcclCommSelector != 0){
            comm = *ncclCommPvtPtr;            
          }
          
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

              // {
              //   int size = totalRanks, this_process = myRank;
              //   // MPI_Comm_size(intrapoolcomm, &size);
              //   // MPI_Comm_rank(intrapoolcomm, &this_process);
              //   std::cout << "Out of " << size << " processes, process " << this_process << " reached line " << __LINE__ << " of file " << __FILE__ << std::endl;
                
              // }

              sendTo += i;
              sendTo %= totalRanks;
              recvFrom += (totalRanks - i);
              recvFrom %= totalRanks;

              sendOffset = (dftfe::uInt)sendTo * sendCount;
              recvOffset = (dftfe::uInt)recvFrom * recvCount;

              // Printing line and file to show no error
              // {
              //   int ncclRank, mpiRank;
              //   dftfe::utils::deviceSynchronize();
              //   MPI_Barrier(MPI_COMM_WORLD);
              //   fflush(stdout);
              //   ncclCommUserRank(*ncclCommPtr, &ncclRank);
              //   MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
              //   std::cout << "ncclRank: " << ncclRank << " mpiRank: " << mpiRank 
              //             << " sendOffset: " << sendOffset
              //             << " recvOffset: " << recvOffset
              //             << " sendCount: " << sendCount
              //             << " recvCount: " << recvCount << std::endl;
              //   // if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              //   //   {
              //   //     std::cout << "No error in file " << __FILE__ << " at line " << __LINE__
              //   //           << std::endl;
              //   //     // print parameter values
              //   //     std::cout << "sendCount: " << sendCount << std::endl;
              //   //     std::cout << "recvCount: " << recvCount << std::endl;
              //   //     std::cout << "sendTo: " << sendTo << std::endl;
              //   //     std::cout << "recvFrom: " << recvFrom << std::endl;
              //   //   }
              //   fflush(stdout);
              // }

              // if (sendOffset + sendCount > totalNumRows * totalNumCols){
              //   sendCount = totalNumRows * totalNumCols - sendOffset;
              // }
              // if (recvOffset + recvCount > totalNumRows * totalNumCols)
              //   recvCount = totalNumRows * totalNumCols - recvOffset;

              NCCLCHECK(ncclGroupStart());
              
                NCCLCHECK(ncclSend(static_cast<const void *>(send) + sendOffset,
                                  sendCount,
                                  ncclChar,
                                  sendTo,
                                  comm,
                                  stream));
                NCCLCHECK(ncclRecv(static_cast<void *>(recv) + recvOffset,
                                    recvCount,
                                    ncclChar,
                                    recvFrom,
                                    comm,
                                    stream));

              NCCLCHECK(ncclGroupEnd());

              // {
              //   int size = totalRanks, this_process = myRank;
              //   // MPI_Comm_size(intrapoolcomm, &size);
              //   // MPI_Comm_rank(intrapoolcomm, &this_process);
              //   std::cout << "Out of " << size << " processes, process " << this_process << " reached line " << __LINE__ << " of file " << __FILE__ << std::endl;
                
              // }
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

              sendOffset = (dftfe::uInt)sendTo * sendCount;
              recvOffset = (dftfe::uInt)recvFrom * recvCount;

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

      // {
      //   int size = totalRanks, this_process = myRank;
      //   // MPI_Comm_size(intrapoolcomm, &size);
      //   // MPI_Comm_rank(intrapoolcomm, &this_process);
      //   std::cout << "Out of " << size << " processes, process " << this_process << " reached line " << __LINE__ << " of file " << __FILE__ << std::endl;
        
      // }

      return 0;
    }

    // initialize alltoall templates
    template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const double * send, dftfe::uInt sendCount, double * recv, dftfe::uInt recvCount, deviceStream_t stream, bool useDCCL);

    template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const float * send, dftfe::uInt sendCount, float * recv, dftfe::uInt recvCount, deviceStream_t stream, bool useDCCL);

    template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const std::complex<double> * send, dftfe::uInt sendCount, std::complex<double> * recv, dftfe::uInt recvCount, deviceStream_t stream, bool useDCCL);

    template int DeviceCCLWrapper::deviceDirectAllToAllWrapper(const std::complex<float> * send, dftfe::uInt sendCount, std::complex<float> * recv, dftfe::uInt recvCount, deviceStream_t stream, bool useDCCL);

  } // namespace utils
} // namespace dftfe
#endif
