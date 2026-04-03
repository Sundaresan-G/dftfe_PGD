#include "linearSolverCGDeviceKernels.h"

namespace dftfe
{
  template <typename Type, dftfe::Int blockSize>
  DFTFE_CREATE_KERNEL_SMEM_S(
    Type,
    2 * blockSize,
    void,
    computeLocalDotRZAndRRKernel,
    DFTFE_KERNEL_ARGUMENT({
      Type       localRZ = 0;
      Type       localRR = 0;
      dftfe::Int idx     = threadId + blockId * (blockSize * 2);

      if (idx < N)
        {
          const Type r = d_rvec[idx];
          localRZ += r * d_zvec[idx];
          localRR += r * r;
        }

      if (idx + blockSize < N)
        {
          const Type r = d_rvec[idx + blockSize];
          localRZ += r * d_zvec[idx + blockSize];
          localRR += r * r;
        }

      // Store paired partial sums in shared memory so both reductions
      // proceed in a single reduction loop.
      Type *smemRZ     = smem;
      Type *smemRR     = smem + blockSize;
      smemRZ[threadId] = localRZ;
      smemRR[threadId] = localRR;
      SYNCTHREADS;

      _Pragma("unroll") for (dftfe::Int size =
                               dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                             size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
                             size /= 2)
      {
        if ((blockSize >= size) && (threadId < size / 2))
          {
            smemRZ[threadId] = localRZ = localRZ + smemRZ[threadId + size / 2];
            smemRR[threadId] = localRR = localRR + smemRR[threadId + size / 2];
          }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(ind.get_group());
#endif
      }

      if (threadId < dftfe::utils::DEVICE_WARP_SIZE)
        {
          if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
            {
              localRZ += smemRZ[threadId + dftfe::utils::DEVICE_WARP_SIZE];
              localRR += smemRR[threadId + dftfe::utils::DEVICE_WARP_SIZE];
            }

          _Pragma("unroll") for (dftfe::Int offset =
                                   dftfe::utils::DEVICE_WARP_SIZE / 2;
                                 offset > 0;
                                 offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localRZ += __shfl_down_sync(mask, localRZ, offset);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
            localRZ +=
              __shfl_down(localRZ, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
            localRZ +=
              sycl::shift_group_left(ind.get_sub_group(), localRZ, offset);
#endif

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            localRR += __shfl_down_sync(mask, localRR, offset);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
            localRR +=
              __shfl_down(localRR, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
            localRR +=
              sycl::shift_group_left(ind.get_sub_group(), localRR, offset);
#endif
          }
        }

      if (threadId == 0)
        {
          dftfe::utils::atomicAddWrapper(&d_localSums[0], localRZ);
          dftfe::utils::atomicAddWrapper(&d_localSums[1], localRR);
        }
    }),
    const Type      *d_rvec,
    const Type      *d_zvec,
    Type            *d_localSums,
    const dftfe::Int N);

  template <typename Type, dftfe::Int blockSize>
  DFTFE_CREATE_KERNEL_SMEM_S(
    Type,
    blockSize,
    void,
    updateXRandComputeLocalRRKernel,
    DFTFE_KERNEL_ARGUMENT({
      Type       localRR = 0;
      dftfe::Int idx     = threadId + blockId * (blockSize * 2);

      if (idx < N)
        {
          d_xvec[idx] += alpha * d_pvec[idx];
          d_rvec[idx] += alpha * d_wvec[idx];
          const Type r = d_rvec[idx];
          localRR += r * r;
        }

      if (idx + blockSize < N)
        {
          d_xvec[idx + blockSize] += alpha * d_pvec[idx + blockSize];
          d_rvec[idx + blockSize] += alpha * d_wvec[idx + blockSize];
          const Type r = d_rvec[idx + blockSize];
          localRR += r * r;
        }

      smem[threadId] = localRR;
      SYNCTHREADS;

      _Pragma("unroll") for (dftfe::Int size =
                               dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                             size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
                             size /= 2)
      {
        if ((blockSize >= size) && (threadId < size / 2))
          smem[threadId] = localRR = localRR + smem[threadId + size / 2];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(ind.get_group());
#endif
      }

      if (threadId < dftfe::utils::DEVICE_WARP_SIZE)
        {
          if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
            localRR += smem[threadId + dftfe::utils::DEVICE_WARP_SIZE];

          _Pragma("unroll") for (dftfe::Int offset =
                                   dftfe::utils::DEVICE_WARP_SIZE / 2;
                                 offset > 0;
                                 offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localRR += __shfl_down_sync(mask, localRR, offset);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
            localRR +=
              __shfl_down(localRR, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
            localRR +=
              sycl::shift_group_left(ind.get_sub_group(), localRR, offset);
#endif
          }
        }

      if (threadId == 0)
        dftfe::utils::atomicAddWrapper(d_localRR, localRR);
    }),
    Type            *d_xvec,
    Type            *d_rvec,
    const Type      *d_pvec,
    const Type      *d_wvec,
    const Type       alpha,
    Type            *d_localRR,
    const dftfe::Int N);

  void
  computeLocalDotRZAndRRDevice(const double    *d_rvec,
                               const double    *d_zvec,
                               double          *d_localSums,
                               const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);

    DFTFE_LAUNCH_KERNEL_SMEM_S(
      DFTFE_KERNEL_ARGUMENT(
        computeLocalDotRZAndRRKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      double,
      2 * dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      d_rvec,
      d_zvec,
      d_localSums,
      N);
  }

  void
  updateXRandComputeLocalRRDevice(double          *d_xvec,
                                  double          *d_rvec,
                                  const double    *d_pvec,
                                  const double    *d_wvec,
                                  const double     alpha,
                                  double          *d_localRR,
                                  const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);

    DFTFE_LAUNCH_KERNEL_SMEM_S(
      DFTFE_KERNEL_ARGUMENT(
        updateXRandComputeLocalRRKernel<double,
                                        dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      double,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      d_xvec,
      d_rvec,
      d_pvec,
      d_wvec,
      alpha,
      d_localRR,
      N);
  }


} // namespace dftfe
