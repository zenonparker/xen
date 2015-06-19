#include <xen/ml/linear_regression.h>

namespace xen {
namespace ml {

LinearRegression::vec_type LinearRegression::solveForTheta(
    const mat_type& X,
    const vec_type& y)
{
  // Note: m is number of training examples.
  //       n is number of features (including intercept term).
  //       X is (m x n)
  //       y is (m x 1)
  //       theta (return value) will be (n x 1)
  //
  // Computes ((X'X)^-1) * X'y
  //
  vec_type theta;
  mat_type x_prime_x_inv;
  mat_type x_prime_x = prod(trans(X), X);

  if (xen::math::invertMatrix(x_prime_x, x_prime_x_inv)) {
    vec_type temp = prod(trans(X), y);
    theta = prod(x_prime_x_inv, temp);
  }

  return theta;
}

} // end namespace ml
} // end namespace xen

