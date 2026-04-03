#ifndef chebyshevPreconditionerDeviceKernels_H
#define chebyshevPreconditionerDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherHelpers.h>


namespace dftfe
{
  /**
   * @brief Fused Chebyshev step 0: dst[i] = scaleFactor * diag[i] * src[i]
   */
  void
  chebyshevPrecondStep0Device(double           *dst,
                              const double     *src,
                              const double     *diag,
                              const double      scaleFactor,
                              const dftfe::uInt N);

  /**
   * @brief Fused Chebyshev inner step:
   *   w = diag[i] * (src[i] - Ax[i])
   *   update[i] = factor1 * update[i] + factor2 * w
   *   dst[i] += update[i]
   */
  void
  chebyshevPrecondStepDevice(double           *dst,
                             double           *update,
                             const double     *src,
                             const double     *Ax,
                             const double     *diag,
                             const double      factor1,
                             const double      factor2,
                             const dftfe::uInt N);

} // namespace dftfe
#endif
