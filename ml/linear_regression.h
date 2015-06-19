////////////////////////////////////////////////////////////////////////////////
///
/// @file linear_regression.h
/// @brief Basic linear regression class.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_ML_LINEAR_REGRESSION_H
#define XEN_ML_LINEAR_REGRESSION_H

#include <xen/math/math.h>
#include <xen/ml/ml_data.h>

namespace xen {
namespace ml {

struct LinearRegression {

  typedef MLData::mat_type mat_type;
  typedef MLData::vec_type vec_type;

  static vec_type solveForTheta(
      const mat_type& X, 
      const vec_type& y);

  // No construction/copying.
  LinearRegression() = delete;
  LinearRegression(const LinearRegression&) = delete;
  LinearRegression& operator=(const LinearRegression&) = delete;

};

} // end namespace ml
} // end namespace xen

#endif // XEN_ML_LINEAR_REGRESSION_H

