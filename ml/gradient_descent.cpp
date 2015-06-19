// xen
#include <xen/ml/gradient_descent.h>

// boost
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace xen::math::ublas;

namespace {

} // end unnamed namespacek

namespace xen {
namespace ml {

GradientDescent::GradientDescent(const mat_type& X, 
                                 const vec_type& y,
                                 double alpha)
  : m_X(X), m_y(y), m_alpha(alpha)
{
  // Initialize theta to zeros.
  m_theta = zero_vector<double>(m_X.size2());
}

double GradientDescent::computeCost() const
{
  vector<double> h_theta(m_X.size2());

  noalias(h_theta) = prod(m_X, m_theta);
  h_theta = (h_theta - m_y);
  for (auto& i : h_theta) i *= i;

  return (sum(h_theta) / (2 * double(m_X.size2())));
}


const GradientDescent::vec_type& GradientDescent::solveForTheta(int iterations)
{
  // Note: m is number of training examples.
  //       n is number of features (including intercept term).
  //       X is (m x n)
  //       y is (m x 1)
  //       theta (return value) will be (n x 1)
  //
  // Minimizes J(theta) = (1/2m)*sum_i[ (hx_i - yx_i)^2 ]
  //
  int m = m_X.size1();
  int n = m_X.size2();

  vector<double> h_theta = m_theta;
  vector<double> h_theta_prod = m_theta;

  double alpha_over_m = m_alpha / double(m);
  for (int iter = 0; iter < iterations; ++iter) {
    noalias(h_theta) = prod(m_X, m_theta);
    h_theta = (h_theta - m_y);

    // Update m_theta values.
    m_theta(0) -= alpha_over_m * sum(h_theta);
    for (int j = 1; j < m_theta.size(); ++j) {
      matrix_column<const matrix<double> > mc(m_X, j);
      for (int i = 0; i < h_theta_prod.size(); ++i) {
        h_theta_prod(i) = h_theta(i) * mc(i);
      }
      m_theta(j) -= alpha_over_m * sum(h_theta_prod);
    }
  }
}

void GradientDescent::resetTheta()
{
  m_theta = zero_vector<double>(m_X.size2());
}

} // end namespace ml
} // end namespace xen

