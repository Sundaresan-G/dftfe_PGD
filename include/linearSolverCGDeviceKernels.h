#ifndef linearSolverCGDeviceKernels_H
#define linearSolverCGDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherHelpers.h>


namespace dftfe
{
  /**
   * @brief Computes local (non-MPI) dot products in one device pass:
   * localSums[0] = r · z, localSums[1] = r · r.
   */
  void
  computeLocalDotRZAndRRDevice(const double    *d_rvec,
                               const double    *d_zvec,
                               double          *d_localSums,
                               const dftfe::Int N);

  /**
   * @brief Fused update and local residual norm accumulation in one pass:
   * x += alpha * p, r += alpha * w, localRR += r · r.
   */
  void
  updateXRandComputeLocalRRDevice(double          *d_xvec,
                                  double          *d_rvec,
                                  const double    *d_pvec,
                                  const double    *d_wvec,
                                  const double     alpha,
                                  double          *d_localRR,
                                  const dftfe::Int N);

} // namespace dftfe
#endif
