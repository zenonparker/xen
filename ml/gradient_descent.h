////////////////////////////////////////////////////////////////////////////////
///
/// @file gradient_descent.h
/// @brief Basic gradient descent algorithm.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_ML_GRADIENT_DESCENT_H
#define XEN_ML_GRADIENT_DESCENT_H

#include <xen/math/math.h>
#include <xen/ml/ml_data.h>

namespace xen {
namespace ml {

class GradientDescent {
public:
  typedef MLData::mat_type mat_type;
  typedef MLData::vec_type vec_type;

  static constexpr double DEFAULT_ALPHA = 0.0003;

  /** @brief Construct a gradient descent object with the given sample matrix
    *        and observation vector.
    * @param X     Matrix of samples.
    * @param y     Vector of observations for the given samples.
    * @param alpha Learning rate alpha.
    */
  GradientDescent(const mat_type& X, 
                  const vec_type& y,
                  double alpha = DEFAULT_ALPHA);

  /** @brief Incrementally solves for theta for the given number of iterations.
    * @param iterations   Number of iterations of refining to be done on theta.
    */
  const vec_type& solveForTheta(int iterations);

  /** @brief Compute value of cost function for the current theta against the
    *        observation vector y.
    */
  double computeCost() const;

  /** @brief Reset the theta vector back to its initial state (all zeros).
    */
  void resetTheta();

  // Accessors
  const vec_type& getTheta() const { return m_theta; }
  const vec_type& getY() const { return m_y; }
  const mat_type& getX() const { return m_X; }
  double getAlpha() const { return m_alpha; }

  void setAlpha(double alpha) { m_alpha = alpha; }

private:

  vec_type        m_theta;
  const mat_type& m_X;
  const vec_type& m_y;
  double          m_alpha;

  // No default construction.
  GradientDescent() = delete;
  
  // No copy construction/assignment.
  GradientDescent(const GradientDescent&) = delete;
  GradientDescent& operator=(const GradientDescent&) = delete;

};

} // end namespace ml
} // end namespace xen

#endif // XEN_ML_GRADIENT_DESCENT_H

