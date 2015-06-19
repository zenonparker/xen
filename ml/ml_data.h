////////////////////////////////////////////////////////////////////////////////
///
/// @file ml_data.h
/// @brief Simple class to handle data for machine learning algorithms.
///        An instance of this class can be installed in, for example, a
///        NeuralNetwork where it will be used for training.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_ML_DATA_H
#define XEN_ML_DATA_H

#include <xen/math/math.h>

namespace xen {
namespace ml {

class MLData {
public:
  typedef xen::math::ublas::matrix<double, 
                                   xen::math::ublas::row_major,
                                   std::vector<double>> mat_type;
  typedef xen::math::ublas::vector<double, std::vector<double>> vec_type;

  MLData(const mat_type& X, const mat_type& y) : m_X(X), m_y(y) {
  }

  // Basic accessors.
  const mat_type& X() const { return m_X; }
  const mat_type& y() const { return m_y; }

private:

  mat_type m_X;
  mat_type m_y;

};

} // end namespace ml
} // end namespace xen

#endif // XEN_ML_DATA_H

