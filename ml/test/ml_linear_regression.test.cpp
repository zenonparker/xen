#include <ml.test.h>
#include <xen/ml/linear_regression.h>

using namespace xen::ml;

TEST_F(MLTest, linearRegressionBasic) {

  // Octave comparison:
  // X = [12 9 1; 0 34 1; 33 2 1]
  // y = [4; 5; 6]
  // theta = [0.12925; 0.10204; 1.53061]

  std::vector<double> v1 = { 12,  9,  1,
                              0, 34,  1,
                             33,  2,  1 };

  std::vector<double> v2 = { 4, 5, 6 };

  mat_type X; fromVector(X, v1, 3, 3);
  vec_type y; fromVector(y, v2);

  auto theta = LinearRegression::solveForTheta(X, y);

  //std::cout << "X: " << X << std::endl;
  //std::cout << "y: " << X << std::endl;
  //std::cout << "theta: " << std::setprecision(9) << theta << std::endl;

  vec_type actualTheta;
  fromFile(actualTheta, "theta1.txt");
  //std::cout << "theta1: " << std::setprecision(9) << actualTheta << std::endl;

  EXPECT_TRUE(areEqual(theta, actualTheta));

}

TEST_F(MLTest, linearRegressionLarge) {

  // For 5000x1000 matrix.
  // Timer [solveForTheta] expired after 430475 milliseconds.
  mat_type X(500, 100);
  vec_type y(500);

  int val = 42;
  for (int i = 0; i < X.size1(); ++i) {
    for (int j = 0; j < X.size2(); ++j) {
      val = (773 * val) % 1117;
      X(i, j) = val;
    }
  }

  for (int i = 0; i < y.size(); ++i) {
    val = (773 * val) % 1117;
    y(i) = val;
  }
  
  vec_type theta;
  {
    //ScopedTimer st("solveForTheta");
    theta = LinearRegression::solveForTheta(X, y);
  }

  vec_type actualTheta;
  fromFile(actualTheta, "theta2.txt");
  EXPECT_TRUE(areEqual(theta, actualTheta));

}

