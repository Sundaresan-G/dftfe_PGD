// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#include <BLASWrapper.h>
#include <deviceKernelsGeneric.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <cublas_v2.h>
#include "BLASWrapperDeviceKernels.cc"
namespace dftfe
{
  namespace linearAlgebra
  {
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::BLASWrapper()
    {
      dftfe::utils::deviceBlasStatus_t status;
      status = create();
      status = setStream(NULL);
      status = setMathMode(dftfe::utils::DEVICEBLAS_TF32_TENSOR_OP_MATH);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A,
      const unsigned int lda,
      const float *      B,
      const unsigned int ldb,
      const float *      beta,
      float *            C,
      const unsigned int ldc) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasSgemm(d_deviceBlasHandle,
                                                            transa,
                                                            transb,
                                                            int(m),
                                                            int(n),
                                                            int(k),
                                                            alpha,
                                                            A,
                                                            int(lda),
                                                            B,
                                                            int(ldb),
                                                            beta,
                                                            C,
                                                            int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int         lda,
      const std::complex<float> *B,
      const unsigned int         ldb,
      const std::complex<float> *beta,
      std::complex<float> *      C,
      const unsigned int         ldc) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasCgemm(d_deviceBlasHandle,
                    transa,
                    transb,
                    int(m),
                    int(n),
                    int(k),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(B),
                    int(ldb),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(C),
                    int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A,
      const unsigned int lda,
      const double *     B,
      const unsigned int ldb,
      const double *     beta,
      double *           C,
      const unsigned int ldc) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }


      dftfe::utils::deviceBlasStatus_t status = cublasDgemm(d_deviceBlasHandle,
                                                            transa,
                                                            transb,
                                                            int(m),
                                                            int(n),
                                                            int(k),
                                                            alpha,
                                                            A,
                                                            int(lda),
                                                            B,
                                                            int(ldb),
                                                            beta,
                                                            C,
                                                            int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int          lda,
      const std::complex<double> *B,
      const unsigned int          ldb,
      const std::complex<double> *beta,
      std::complex<double> *      C,
      const unsigned int          ldc) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }


      dftfe::utils::deviceBlasStatus_t status =
        cublasZgemm(d_deviceBlasHandle,
                    transa,
                    transb,
                    int(m),
                    int(n),
                    int(k),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(B),
                    int(ldb),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(C),
                    int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasCreate(&d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasDestroy(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t streamId)
    {
      d_streamId = streamId;
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetStream(d_deviceBlasHandle, d_streamId);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setMathMode(
      dftfe::utils::deviceBlasMath_t mathMode)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetMathMode(d_deviceBlasHandle, mathMode);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int n,
      const double *     alpha,
      double *           x,
      const unsigned int incx,
      double *           y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDaxpy(
        d_deviceBlasHandle, int(n), alpha, x, int(incx), y, int(incy));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int          n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZaxpy(d_deviceBlasHandle,
                    int(n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(incx),
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
                    int(incy));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int N,
      const double *     X,
      const unsigned int INCX,
      const double *     Y,
      const unsigned int INCY,
      double *           result) const
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDdot(
        d_deviceBlasHandle, int(N), X, int(INCX), Y, int(INCY), result);
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int          N,
      const std::complex<double> *X,
      const unsigned int          INCX,
      const std::complex<double> *Y,
      const unsigned int          INCY,
      std::complex<double> *      result) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZdotc(d_deviceBlasHandle,
                    int(N),
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    int(INCX),
                    dftfe::utils::makeDataTypeDeviceCompatible(Y),
                    int(INCY),
                    dftfe::utils::makeDataTypeDeviceCompatible(result));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A,
      const unsigned int lda,
      long long int      strideA,
      const double *     B,
      const unsigned int ldb,
      long long int      strideB,
      const double *     beta,
      double *           C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasDgemmStridedBatched(d_deviceBlasHandle,
                                  transa,
                                  transb,
                                  int(m),
                                  int(n),
                                  int(k),
                                  alpha,
                                  A,
                                  int(lda),
                                  strideA,
                                  B,
                                  int(ldb),
                                  strideB,
                                  beta,
                                  C,
                                  int(ldc),
                                  strideC,
                                  int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int          lda,
      long long int               strideA,
      const std::complex<double> *B,
      const unsigned int          ldb,
      long long int               strideB,
      const std::complex<double> *beta,
      std::complex<double> *      C,
      const unsigned int          ldc,
      long long int               strideC,
      const int                   batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasZgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(m),
        int(n),
        int(k),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        int(lda),
        strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        int(ldb),
        strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(C),
        int(ldc),
        strideC,
        int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A,
      const unsigned int lda,
      long long int      strideA,
      const float *      B,
      const unsigned int ldb,
      long long int      strideB,
      const float *      beta,
      float *            C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasSgemmStridedBatched(d_deviceBlasHandle,
                                  transa,
                                  transb,
                                  int(m),
                                  int(n),
                                  int(k),
                                  alpha,
                                  A,
                                  int(lda),
                                  strideA,
                                  B,
                                  int(ldb),
                                  strideB,
                                  beta,
                                  C,
                                  int(ldc),
                                  strideC,
                                  int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int         lda,
      long long int              strideA,
      const std::complex<float> *B,
      const unsigned int         ldb,
      long long int              strideB,
      const std::complex<float> *beta,
      std::complex<float> *      C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasCgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(m),
        int(n),
        int(k),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        int(lda),
        strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        int(ldb),
        strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(C),
        int(ldc),
        strideC,
        int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A[],
      const unsigned int lda,
      const double *     B[],
      const unsigned int ldb,
      const double *     beta,
      double *           C[],
      const unsigned int ldc,
      const int          batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasDgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(m),
                           int(n),
                           int(k),
                           alpha,
                           A,
                           int(lda),
                           B,
                           int(ldb),
                           beta,
                           C,
                           int(ldc),
                           int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A[],
      const unsigned int          lda,
      const std::complex<double> *B[],
      const unsigned int          ldb,
      const std::complex<double> *beta,
      std::complex<double> *      C[],
      const unsigned int          ldc,
      const int                   batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasZgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(m),
                           int(n),
                           int(k),
                           dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                           (const dftfe::utils::deviceDoubleComplex **)A,
                           int(lda),
                           (const dftfe::utils::deviceDoubleComplex **)B,
                           int(ldb),
                           dftfe::utils::makeDataTypeDeviceCompatible(beta),
                           (dftfe::utils::deviceDoubleComplex **)C,
                           int(ldc),
                           int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A[],
      const unsigned int lda,
      const float *      B[],
      const unsigned int ldb,
      const float *      beta,
      float *            C[],
      const unsigned int ldc,
      const int          batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasSgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(m),
                           int(n),
                           int(k),
                           alpha,
                           A,
                           int(lda),
                           B,
                           int(ldb),
                           beta,
                           C,
                           int(ldc),
                           int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A[],
      const unsigned int         lda,
      const std::complex<float> *B[],
      const unsigned int         ldb,
      const std::complex<float> *beta,
      std::complex<float> *      C[],
      const unsigned int         ldc,
      const int                  batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasCgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(m),
                           int(n),
                           int(k),
                           dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                           (const dftfe::utils::deviceFloatComplex **)A,
                           int(lda),
                           (const dftfe::utils::deviceFloatComplex **)B,
                           int(ldb),
                           dftfe::utils::makeDataTypeDeviceCompatible(beta),
                           (dftfe::utils::deviceFloatComplex **)C,
                           int(ldc),
                           int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const MPI_Comm &            mpi_communicator,
      double *                    result) const
    {
      double                           localresult = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        cublasDznrm2(d_deviceBlasHandle,
                     int(n),
                     dftfe::utils::makeDataTypeDeviceCompatible(x),
                     int(incx),
                     &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double                           localresult = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        cublasDnrm2(d_deviceBlasHandle, int(n), x, int(incx), &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n) const
    {
      ascalDeviceKernel<<<n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        n,
        dftfe::utils::makeDataTypeDeviceCompatible(x),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha));
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyComplexArrToRealArrs(
      const dftfe::size_type  size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal *         realArr,
      ValueTypeReal *         imagArr)
    {
      copyComplexArrToRealArrsDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        size,
        dftfe::utils::makeDataTypeDeviceCompatible(complexArr),
        realArr,
        imagArr);
    }



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const ValueTypeReal *  realArr,
      const ValueTypeReal *  imagArr,
      ValueTypeComplex *     complexArr)
    {
      copyRealArrsToComplexArrDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        size,
        realArr,
        imagArr,
        dftfe::utils::makeDataTypeDeviceCompatible(complexArr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr)
    {
      copyValueType1ArrToValueType2ArrDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        d_streamId>>>(size,
                      dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        valueType2Arr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      stridedCopyToBlockDeviceKernel<<<(contiguousBlockSize *
                                        numContiguousBlocks) /
                                           dftfe::utils::DEVICE_BLOCK_SIZE +
                                         1,
                                       dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVecBlock,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      stridedCopyFromBlockDeviceKernel<<<(contiguousBlockSize *
                                          numContiguousBlocks) /
                                             dftfe::utils::DEVICE_BLOCK_SIZE +
                                           1,
                                         dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVecBlock),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec),
        copyFromVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec)
    {
      stridedCopyToBlockConstantStrideDeviceKernel<<<
        (blockSizeTo * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSizeTo,
        blockSizeFrom,
        numBlocks,
        startingId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
      const dftfe::size_type blockSize,
      const dftfe::size_type strideTo,
      const dftfe::size_type strideFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingToId,
      const dftfe::size_type startingFromId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      stridedCopyConstantStrideDeviceKernel<<<
        (blockSize * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize,
        strideTo,
        strideFrom,
        numBlocks,
        startingToId,
        startingFromId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
    {
      stridedCopyFromBlockConstantStrideDeviceKernel<<<
        (blockSizeFrom * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSizeTo,
        blockSizeFrom,
        numBlocks,
        startingId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      ValueType2 *           x)
    {
      stridedBlockScaleDeviceKernel<<<(contiguousBlockSize *
                                       numContiguousBlocks) /
                                          dftfe::utils::DEVICE_BLOCK_SIZE +
                                        1,
                                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(s),
        dftfe::utils::makeDataTypeDeviceCompatible(x));
    }
    // for stridedBlockScale
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<double> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<float> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<float> *       x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<double> *     x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<double> * x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<float> *  x);


    // for xscal
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      double *               x,
      const double           a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      float *                x,
      const float            a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      std::complex<double> *     x,
      const std::complex<double> a,
      const dftfe::size_type     n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      std::complex<float> *     x,
      const std::complex<float> a,
      const dftfe::size_type    n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 copyFromVec,
      double *                       copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   copyFromVec,
      std::complex<double> *         copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

  } // End of namespace linearAlgebra
} // End of namespace dftfe
