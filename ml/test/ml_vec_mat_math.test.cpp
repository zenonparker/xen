#include <ml.test.h>

TEST_F(MLTest, vectorMatrixComparison) {

  std::vector<double> v1 = { 10, 5, 4,
                             45, 1, 3 };

  std::vector<double> v2 = { 10, 5, 4,
                             45, 1, 3 };

  matrix<double> mat1, mat2;
  fromVector(mat1, v1, 2, 3);
  fromVector(mat2, v2, 2, 3);

  EXPECT_TRUE(areEqual(mat1, mat2));
  mat1(0,0) = 10.00000001;
  EXPECT_TRUE(areEqual(mat1, mat2));
  mat1(0,0) = 10.001;
  EXPECT_FALSE(areEqual(mat1, mat2));
  mat1(0,0) = 10;
  mat1.resize(2, 2);
  EXPECT_FALSE(areEqual(mat1, mat2));

}

TEST_F(MLTest, vectorMatrixSigmoid) {

  std::vector<double> norm = { -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0 };

  std::vector<double> sig = { 0.0066929, 0.1192029, 0.2689414, 0.3775407, 
                              0.5, 0.6224593, 0.7310586, 0.8807971, 0.9933071 };

  vector<double> x1, x1sig;
  fromVector(x1, norm);
  fromVector(x1sig, sig);

  EXPECT_FALSE(areEqual(x1, x1sig));
  sigmoid(x1);
  EXPECT_TRUE(areEqual(x1, x1sig));

  matrix<double> y1, y1sig;
  fromVector(y1, norm, 3, 3);
  fromVector(y1sig, sig, 3, 3);

  EXPECT_FALSE(areEqual(y1, y1sig));
  sigmoid(y1);
  EXPECT_TRUE(areEqual(y1, y1sig));

}

TEST_F(MLTest, vectorMatrixSigmoidGradient) {

  std::vector<double> norm = { -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0 };

  std::vector<double> sig = { 0.00664805, 0.10499358, 0.19661193, 0.23500371, 
                              0.25000000, 0.23500371, 0.19661193, 0.10499358, 0.00664805 };

  vector<double> x1, x1sig;
  fromVector(x1, norm);
  fromVector(x1sig, sig);

  EXPECT_FALSE(areEqual(x1, x1sig));
  sigmoidGradient(x1);
  EXPECT_TRUE(areEqual(x1, x1sig));

  matrix<double> y1, y1sig;
  fromVector(y1, norm, 3, 3);
  fromVector(y1sig, sig, 3, 3);

  EXPECT_FALSE(areEqual(y1, y1sig));
  sigmoidGradient(y1);
  EXPECT_TRUE(areEqual(y1, y1sig));

}

TEST_F(MLTest, vectorMatrixResizeAndAssign) {

  std::vector<double> v1 = { 10, 5, 4,
                             45, 1, 3 };

  std::vector<double> v2 = { 9, 5, 4,
                             5, 11, 3,
                             1, 2, 3 };

  // Check matrix assign.
  matrix<double> mat1, mat2, mat3;
  fromVector(mat1, v1, 2, 3);
  fromVector(mat2, v2, 3, 3);

  EXPECT_FALSE(areEqual(mat1, mat3));
  resizeAndAssign(mat3, mat1);
  EXPECT_TRUE(areEqual(mat1, mat3));
  EXPECT_FALSE(areEqual(mat2, mat3));
  resizeAndAssign(mat3, mat2);
  EXPECT_FALSE(areEqual(mat1, mat3));
  EXPECT_TRUE(areEqual(mat2, mat3));

  // Check vector assign.
  vector<double> vec1, vec2, vec3;
  fromVector(vec1, v1);
  fromVector(vec2, v2);

  // Check matrix assign.
  EXPECT_FALSE(areEqual(vec1, vec3));
  resizeAndAssign(vec3, vec1);
  EXPECT_TRUE(areEqual(vec1, vec3));
  EXPECT_FALSE(areEqual(vec2, vec3));
  resizeAndAssign(vec3, vec2);
  EXPECT_FALSE(areEqual(vec1, vec3));
  EXPECT_TRUE(areEqual(vec2, vec3));

}

TEST_F(MLTest, vectorMatrixElementWiseOper) {

  std::vector<double> v1 = { 10, 5, 4,
                             45, 1, 3 };

  std::vector<double> v2 = { 2, 1, 2,
                             5, 0.5, 1 };
                             
  std::vector<double> mult = { 20.00000, 5.00000, 8.00000, 
                               225.00000, 0.50000, 3.00000 };
  std::vector<double> div = { 5, 5, 2, 
                              9, 2, 3 };
  std::vector<double> pow = { 100, 5, 16, 
                              184528125, 1, 3 };

  {
    matrix<double> mat1, mat2, mdiv, mmult, mpow, mdiv_ans, mmult_ans, mpow_ans;
    fromVector(mmult, v1, 2, 3);
    fromVector(mdiv, v1, 2, 3);
    fromVector(mpow, v1, 2, 3);
    fromVector(mat2, v2, 2, 3);
    fromVector(mmult_ans, mult, 2, 3);
    fromVector(mdiv_ans, div, 2, 3);
    fromVector(mpow_ans, pow, 2, 3);

    elementWiseOper(Oper::PROD, mmult, mat2);
    elementWiseOper(Oper::DIV, mdiv, mat2);
    elementWiseOper(Oper::POW, mpow, mat2);

    EXPECT_TRUE(areEqual(mmult, mmult_ans));
    EXPECT_TRUE(areEqual(mdiv, mdiv_ans));
    EXPECT_TRUE(areEqual(mpow, mpow_ans));
  }

  {
    vector<double> vec1, vec2, vdiv, vmult, vpow, vdiv_ans, vmult_ans, vpow_ans;
    fromVector(vmult, v1);
    fromVector(vdiv, v1);
    fromVector(vpow, v1);
    fromVector(vec2, v2);
    fromVector(vmult_ans, mult);
    fromVector(vdiv_ans, div);
    fromVector(vpow_ans, pow);

    elementWiseOper(Oper::PROD, vmult, vec2);
    elementWiseOper(Oper::DIV, vdiv, vec2);
    elementWiseOper(Oper::POW, vpow, vec2);

    EXPECT_TRUE(areEqual(vmult, vmult_ans));
    EXPECT_TRUE(areEqual(vdiv, vdiv_ans));
    EXPECT_TRUE(areEqual(vpow, vpow_ans));
  }
}

TEST_F(MLTest, ublasStorageTests) {

  std::vector<double> v1 = { 10, 5, 4,
                             45, 1, 3 };

  std::vector<double> v2 = { 2, 1, 2,
                             5, 0.5, 1 };

  vector<double, std::vector<double>> vec1;
  fromVector(vec1, v1);

  std::vector<double>& vec1Data = vec1.data();

  matrix<double, row_major, std::vector<double>> mat1;
  fromVector(mat1, v1, 2, 3);
  std::vector<double>& mat1Data = mat1.data();

  matrix<double, column_major, std::vector<double>> mat2(mat1);
  std::vector<double>& mat2Data = mat2.data();

  EXPECT_TRUE(areEqual(vec1.data(), mat1.data()));

  // Because column_major will store differently.
  EXPECT_FALSE(areEqual(vec1.data(), mat2.data()));

}


