#ifndef ML_TEST_H
#define ML_TEST_H

// std
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

// gtest
#include <gtest/gtest.h>

// xen
#include <xen/math/math.h>
#include <xen/time/scoped_timer.h>

using namespace xen::math::ublas;
using namespace xen::math;
using namespace xen::time;

class MLTest : public ::testing::Test {
protected:

  typedef matrix<double, row_major, std::vector<double>> mat_type;
  typedef vector<double, std::vector<double>> vec_type;

  virtual void SetUp() {
  }

  virtual void TearDown() {
  }

  template <typename T, typename A>
  bool fromVector(matrix<T, row_major, A>& mat, const std::vector<T>& inVec, int rows, int cols) {
    if (inVec.size() != (rows * cols)) return false;
    mat.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        mat(i, j) = inVec[i * cols + j];
      }
    }
    return true;
  }

  template <typename T, typename A>
  bool fromVector(vector<T, A>& vec, const std::vector<T>& inVec) {
    vec.resize(inVec.size());
    for (int i = 0; i < inVec.size(); ++i) vec(i) = inVec[i];
  }

  template <typename T, typename A>
  bool fromFile(matrix<T, row_major, A>& mat, const std::string& fileName) {
    std::string newFileName("./data/" + fileName);
    std::ifstream file(newFileName);
    std::string line, temp;

    if (!getline(file, line)) {
      return false;
    }

    int rows, cols;
    if (line.substr(0, 1) == "#") {
      // Octave file.
      getline(file, line); // line 2
      getline(file, line); // line 3
      getline(file, line); // line 4
      {
        std::stringstream ss(line);
        ss >> temp >> temp >> rows;
      }
      getline(file, line); // line 5
      {
        std::stringstream ss(line);
        ss >> temp >> temp >> cols;
      }
    } else {
      std::stringstream ss(line);
      ss >> rows >> cols;
    }
    mat.resize(rows, cols);

    for (int i = 0; i < rows; ++i) {
      if (!getline(file, line)) {
        return false;
      }
      std::stringstream ss(line);
      for (int j = 0; j < cols; ++j) {
        double val; ss >> val;
        mat(i, j) = val;
      }
    }
  }

  template <typename T, typename A>
  bool fromFile(vector<T, A>& vec, const std::string& fileName) {
    matrix<T, row_major, A> mat;
    fromFile(mat, fileName);
    if (mat.size2() == 1) {
      // Column vector in matrix form.
      vec.resize(mat.size1());
      vec.assign(column(mat, 0));
    } else if (mat.size1() == 1) {
      // Column vector in matrix form.
      vec.resize(mat.size2());
      vec.assign(row(mat, 0));
    } else {
      return false;
    }
    return true;
  }

};

#endif // ML_TEST_H

