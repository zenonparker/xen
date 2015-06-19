#include <ml.test.h>
#include <xen/ml/gradient_descent.h>

using namespace xen::ml;

TEST_F(MLTest, gradientDescentBasic) {

  std::vector<double> v1 = { 12,  9,  1,
                              0, 34,  1,
                             33,  2,  1 };

  std::vector<double> v2 = { 4, 5, 6 };

  mat_type X; fromVector(X, v1, 3, 3);
  vec_type y; fromVector(y, v2);

  GradientDescent gd(X, y);
  
  // 5000 Iterations.
  gd.solveForTheta(5000);

  vec_type actualTheta;
  fromFile(actualTheta, "theta3.txt");

  EXPECT_TRUE(areEqual(gd.getTheta(), actualTheta));

}


