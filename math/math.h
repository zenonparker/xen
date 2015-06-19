////////////////////////////////////////////////////////////////////////////////
///
/// @file math.h
/// @brief Common math includes and utilities.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_MATH_MATH_H
#define XEN_MATH_MATH_H

#include <math.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace xen {
namespace math {

// Alias the boost namespace for ease of use.
namespace ublas = boost::numeric::ublas;

/** @brief Range-based for loop begin/end functions.
  */
template <typename T, typename A>
typename ublas::matrix<T, ublas::row_major, A>::iterator begin(ublas::matrix<T, ublas::row_major, A>& mat) { return mat.begin1(); }

template <typename T, typename A>
typename ublas::vector<T, A>::iterator begin(ublas::vector<T, A>& vec) { return vec.begin(); }

////////////////////////////////////////////////////////////////////////////////
///                            BASIC MATH UTILS                              ///
////////////////////////////////////////////////////////////////////////////////

const double EPSILON = 0.000001;
enum class Oper { PROD, DIV, POW };

/** @brief Compare two numeric values to within a given epsilon.
  * @return 0 if (x = y)
  *         1 if (x > y)
  *        -1 if (x < y)
  */
template <typename T>
int compare(T x, T y, double eps = EPSILON);

/** @brief Check if two numeric values are equal to within a given epsilon.
  */
template <typename T>
bool areEqual(T x, T y, double eps = EPSILON);

/** @brief The good ol' sigmoid, f(x) = 1/(1+e^(-x))
  */
inline double sigmoid(double x) { return (1.0 / (1.0 + exp(-x))); }

/** @brief Gradient of sigmoid funcion, f(x) = sigmoid(x) * (1-sigmoid(x))
  */
inline double sigmoidGradient(double x) { return (sigmoid(x) * (1.0 - sigmoid(x))); }

////////////////////////////////////////////////////////////////////////////////
///                           CONTAINER RELATED                              ///
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool areEqual(const std::vector<T>& vec1, const std::vector<T>& vec2,
              double eps = EPSILON);

////////////////////////////////////////////////////////////////////////////////
///                             UBLAS RELATED                                ///
////////////////////////////////////////////////////////////////////////////////

/** @brief Modify a vector to contain the sigmoid of all its entries.
  */
template <typename T, typename A>
void sigmoid(ublas::vector<T, A>& vec);

/** @brief Modify a matrix to contain the sigmoid of all its entries.
  */
template <typename T, typename A>
void sigmoid(ublas::matrix<T, ublas::row_major, A>& mat);

/** @brief Modify a vector to contain the sigmoid gradient of all its entries.
  */
template <typename T, typename A>
void sigmoidGradient(ublas::vector<T, A>& vec);

/** @brief Modify a matrix to contain the sigmoid gradient of all its entries.
  */
template <typename T, typename A>
void sigmoidGradient(ublas::matrix<T, ublas::row_major, A>& mat);

/** @brief Modify a vector to contain the natural logarithm of all its entries.
  */
template <typename T, typename A>
void log_vec(ublas::vector<T, A>& vec);

/** @brief Modify a matrix to contain the natural logarithm of all its entries.
  */
template <typename T, typename A>
void log_mat(ublas::matrix<T, ublas::row_major, A>& mat);

template <typename T, typename A>
bool invertMatrix(const ublas::matrix<T, ublas::row_major, A>& input, ublas::matrix<T, ublas::row_major, A>& output);

template <typename T, typename A>
bool areEqual(const ublas::matrix<T, ublas::row_major, A>& mat1, const ublas::matrix<T, ublas::row_major, A>& mat2,
              double eps = EPSILON);

template <typename T, typename A>
bool areEqual(const ublas::vector<T, A>& vec1, const ublas::vector<T, A>& vec2,
              double eps = EPSILON);

template <typename T, typename A>
void resizeAndAssign(ublas::vector<T, A>& result, ublas::vector<T, A>& source);

template <typename T, typename A>
void resizeAndAssign(ublas::matrix<T, ublas::row_major, A>& result, ublas::matrix<T, ublas::row_major, A>& source);

template <typename T, typename A>
void elementWiseOper(Oper op, ublas::vector<T, A>& result, ublas::vector<T, A>& source);

template <typename T, typename A>
void elementWiseOper(Oper op, ublas::matrix<T, ublas::row_major, A>& result, ublas::matrix<T, ublas::row_major, A>& source);

/** @brief Returns the index of the maximal value.
  */
template <typename T, typename A>
size_t maxIndex(const ublas::vector<T, A>& vec);

////////////////////////////////////////////////////////////////////////////////
///                             IMPLEMENTATION                               ///
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool areEqual(const std::vector<T>& vec1, 
              const std::vector<T>& vec2,
              double eps = EPSILON)
{
  // Definitely not equal if sizes are different.
  if (vec1.size() != vec2.size()) {
    return false;
  }

  auto i1 = vec1.begin();
  auto i2 = vec2.begin();
  for (; i1 != vec1.end(); ++i1, ++i2) {
    if (!areEqual(*i1, *i2)) return false;
  }
  return true;
}

template <typename T, typename A>
void sigmoid(ublas::vector<T, A>& vec) {
  for (auto j = vec.begin(); j != vec.end(); ++j) {
    *j = sigmoid(*j);
  }
}

template <typename T, typename A>
void sigmoid(ublas::matrix<T, ublas::row_major, A>& mat) {
  for (auto i = mat.begin1(); i != mat.end1(); ++i) {
    for (auto j = i.begin(); j != i.end(); ++j) {
      *j = sigmoid(*j);
    }
  }
}

template <typename T, typename A>
void sigmoidGradient(ublas::vector<T, A>& vec) {
  for (auto j = vec.begin(); j != vec.end(); ++j) {
    *j = sigmoidGradient(*j);
  }
}

template <typename T, typename A>
void sigmoidGradient(ublas::matrix<T, ublas::row_major, A>& mat) {
  for (auto i = mat.begin1(); i != mat.end1(); ++i) {
    for (auto j = i.begin(); j != i.end(); ++j) {
      *j = sigmoidGradient(*j);
    }
  }
}

template <typename T, typename A>
void log_vec(ublas::vector<T, A>& vec) {
  for (auto j = vec.begin(); j != vec.end(); ++j) {
    *j = log(*j);
  }
}

template <typename T, typename A>
void log_mat(ublas::matrix<T, ublas::row_major, A>& mat) {
  for (auto i = mat.begin1(); i != mat.end1(); ++i) {
    for (auto j = i.begin(); j != i.end(); ++j) {
      *j = log(*j);
    }
  }
}

template <typename T>
int compare(T x, T y, double eps = EPSILON) {
  if (fabs(x - y) < eps) return 0;
  if (x > y) { return 1; } else { return -1; }
}

template <typename T>
bool areEqual(T x, T y, double eps = EPSILON) {
  return (fabs(x - y) < eps);
}

template <typename T, typename A>
bool invertMatrix(const ublas::matrix<T, ublas::row_major, A>& input, 
                  ublas::matrix<T, ublas::row_major, A>& inverse)
{
  ublas::matrix<T, ublas::row_major, A> temp(input);

  // Create a permutation matrix for the LU-factorization.
  ublas::permutation_matrix<std::size_t> pm(temp.size1());

  if(ublas::lu_factorize(temp, pm)) {
    return false;
  }

  // Start with identity and backsubstitute to get the inverse.
  inverse.resize(temp.size1(), temp.size2(), false);
  inverse.assign(ublas::identity_matrix<T>(temp.size1()));
  ublas::lu_substitute(temp, pm, inverse);

  return true;
}

template <typename T, typename A>
bool areEqual(const ublas::matrix<T, ublas::row_major, A>& mat1, 
              const ublas::matrix<T, ublas::row_major, A>& mat2,
              double eps = EPSILON)
{
  // Definitely not equal if sizes are different.
  if (mat1.size1() != mat2.size1() ||
      mat1.size2() != mat2.size2()) {
    return false;
  }

  auto i1 = mat1.begin1();
  auto i2 = mat2.begin1();
  for (; i1 != mat1.end1(); ++i1, ++i2) {
    auto j1 = i1.begin();
    auto j2 = i2.begin();
    for (; j1 != i1.end(); ++j1, ++j2) {
      if (!areEqual(*j1, *j2)) return false;
    }
  }
  return true;
}

template <typename T, typename A>
bool areEqual(const ublas::vector<T, A>& vec1, 
              const ublas::vector<T, A>& vec2,
              double eps = EPSILON)
{
  // Definitely not equal if sizes are different.
  if (vec1.size() != vec2.size()) {
    return false;
  }

  auto i1 = vec1.begin();
  auto i2 = vec2.begin();
  for (; i1 != vec1.end(); ++i1, ++i2) {
    if (!areEqual(*i1, *i2)) return false;
  }
  return true;
}

template <typename T, typename A>
void resizeAndAssign(ublas::vector<T, A>& result, ublas::vector<T, A>& source)
{
  result.resize(source.size(), false);
  result.assign(source);
}

template <typename T, typename A>
void resizeAndAssign(ublas::matrix<T, ublas::row_major, A>& result, ublas::matrix<T, ublas::row_major, A>& source)
{
  result.resize(source.size1(), source.size2(), false);
  result.assign(source);
}

template <typename T, typename A>
void elementWiseOper(Oper op, ublas::vector<T, A>& result, ublas::vector<T, A>& source)
{
  if (result.size() != source.size()) {
    throw std::logic_error("Invalid operand sizes for element wise operation!");
  }

  switch (op) {
    case Oper::PROD: for (int i = 0; i < result.size(); ++i) { result[i] *= source[i]; } break;
    case Oper::DIV:  for (int i = 0; i < result.size(); ++i) { result[i] /= source[i]; } break;
    case Oper::POW:  for (int i = 0; i < result.size(); ++i) { result[i] = pow(result[i], source[i]); } break;
    default: throw std::logic_error("Unknown operation.");
  }
}

template <typename T, typename A>
void elementWiseOper(Oper op, ublas::matrix<T, ublas::row_major, A>& result, ublas::matrix<T, ublas::row_major, A>& source)
{
  if (result.size1() != source.size1() || result.size2() != source.size2()) {
    throw std::logic_error("Invalid operand sizes for element wise operation!");
  }

  switch (op) {
    case Oper::PROD: for (int i = 0; i < result.size1(); ++i) { 
                       for (int j = 0; j < result.size2(); ++j) { result(i, j) *= source(i, j); } } break;
    case Oper::DIV:  for (int i = 0; i < result.size1(); ++i) {
                       for (int j = 0; j < result.size2(); ++j) { result(i, j) /= source(i, j); } } break;
    case Oper::POW:  for (int i = 0; i < result.size1(); ++i) {
                       for (int j = 0; j < result.size2(); ++j) { result(i, j) = pow(result(i, j), source(i, j)); } } break;
    default: throw std::logic_error("Unknown operation.");
  }
}

template <typename T, typename A>
size_t maxIndex(const ublas::vector<T, A>& vec)
{
  size_t maxInd = 0;
  T maxVal = vec[0];
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i] > maxVal) {
      maxInd = i;
      maxVal = vec[i];
    }
  }
  return maxInd;
}

} // end namespace math
} // end namespace xen

#endif // XEN_MATH_MATH_H

