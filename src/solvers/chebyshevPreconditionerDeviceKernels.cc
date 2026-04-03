#include "chebyshevPreconditionerDeviceKernels.h"

namespace dftfe
{
  template <typename Type>
  DFTFE_CREATE_KERNEL(
    void,
    chebyshevPrecondStep0Kernel,
    {
      for (dftfe::uInt idx = globalThreadId; idx < N;
           idx += nThreadsPerBlock * nThreadBlock)
        {
          dst[idx] = scaleFactor * diag[idx] * src[idx];
        }
    },
    Type             *dst,
    const Type       *src,
    const Type       *diag,
    const Type        scaleFactor,
    const dftfe::uInt N);


  template <typename Type>
  DFTFE_CREATE_KERNEL(
    void,
    chebyshevPrecondStepKernel,
    {
      for (dftfe::uInt idx = globalThreadId; idx < N;
           idx += nThreadsPerBlock * nThreadBlock)
        {
          const Type w   = diag[idx] * (src[idx] - Ax[idx]);
          const Type upd = factor1 * update[idx] + factor2 * w;
          update[idx]    = upd;
          dst[idx] += upd;
        }
    },
    Type             *dst,
    Type             *update,
    const Type       *src,
    const Type       *Ax,
    const Type       *diag,
    const Type        factor1,
    const Type        factor2,
    const dftfe::uInt N);


  void
  chebyshevPrecondStep0Device(double           *dst,
                              const double     *src,
                              const double     *diag,
                              const double      scaleFactor,
                              const dftfe::uInt N)
  {
    const dftfe::uInt gridSize =
      (N / dftfe::utils::DEVICE_BLOCK_SIZE) +
      (N % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
    DFTFE_LAUNCH_KERNEL(chebyshevPrecondStep0Kernel,
                        gridSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::defaultStream,
                        dst,
                        src,
                        diag,
                        scaleFactor,
                        N);
  }


  void
  chebyshevPrecondStepDevice(double           *dst,
                             double           *update,
                             const double     *src,
                             const double     *Ax,
                             const double     *diag,
                             const double      factor1,
                             const double      factor2,
                             const dftfe::uInt N)
  {
    const dftfe::uInt gridSize =
      (N / dftfe::utils::DEVICE_BLOCK_SIZE) +
      (N % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
    DFTFE_LAUNCH_KERNEL(chebyshevPrecondStepKernel,
                        gridSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::defaultStream,
                        dst,
                        update,
                        src,
                        Ax,
                        diag,
                        factor1,
                        factor2,
                        N);
  }


} // namespace dftfe
