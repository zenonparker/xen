#include <ml.test.h>
#include <xen/ml/neural_network.h>

using namespace xen::ml;

TEST_F(MLTest, neuralNetworkCostCompute) {

  mat_type theta1, theta2, X, y, theta1trained, theta2trained;

  fromFile(X, "simpleX_nn.txt");
  fromFile(y, "simpleY_nn.txt");
  fromFile(theta1, "simpleTheta1_nn.txt");
  fromFile(theta2, "simpleTheta2_nn.txt");
  fromFile(theta1trained, "trainedTheta1_nn.txt");
  fromFile(theta2trained, "trainedTheta2_nn.txt");

  std::vector<mat_type> theta;
  theta.push_back(theta1);
  theta.push_back(theta2);

  NeuralNetwork nn(theta);

  EXPECT_EQ(nn.layerSizes()[0], 4);
  EXPECT_EQ(nn.layerSizes()[1], 3);
  EXPECT_EQ(nn.layerSizes()[2], 4);

  EXPECT_TRUE(areEqual(nn.theta()[0], theta1));
  EXPECT_TRUE(areEqual(nn.theta()[1], theta2));

  nn.installData(new MLData(X, y));

  EXPECT_TRUE(areEqual(nn.computeGradient(), 6.45674664));

  std::vector<mat_type> thetaTrained;
  thetaTrained.push_back(theta1trained);
  thetaTrained.push_back(theta2trained);

  NeuralNetwork nn2(thetaTrained);
  nn2.installData(new MLData(X, y));

  EXPECT_TRUE(areEqual(nn2.computeGradient(), 2.66307595));
}

TEST_F(MLTest, neuralNetworkRegularizedCost) {

  mat_type X, y, theta1trained, theta2trained;

  fromFile(X, "simpleX_nn.txt");
  fromFile(y, "simpleY_nn.txt");
  fromFile(theta1trained, "trainedTheta1_nn.txt");
  fromFile(theta2trained, "trainedTheta2_nn.txt");

  std::vector<mat_type> thetaTrained;
  thetaTrained.push_back(theta1trained);
  thetaTrained.push_back(theta2trained);

  NeuralNetwork nn(thetaTrained);
  nn.setRegLambda(1.0);
  nn.installData(new MLData(X, y));

  EXPECT_TRUE(areEqual(nn.computeGradient(), 2.66974329));
}

TEST_F(MLTest, neuralNetworkMultiHiddenLayer) {

  mat_type X, y, theta1, theta2, theta3;

  fromFile(X, "simpleX_nn.txt");
  fromFile(y, "simpleY_nn.txt");
  fromFile(theta1, "theta1_multi.txt");
  fromFile(theta2, "theta2_multi.txt");
  fromFile(theta3, "theta3_multi.txt");

  std::vector<mat_type> theta;
  theta.push_back(theta1);
  theta.push_back(theta2);
  theta.push_back(theta3);

  NeuralNetwork nn(theta);
  nn.installData(new MLData(X, y));

  // Test no regularization.
  EXPECT_TRUE(areEqual(nn.computeGradient(), 2.95153805));

  // Test with regularization.
  nn.setRegLambda(1.0);
  EXPECT_TRUE(areEqual(nn.computeGradient(), 2.97595418));
}

TEST_F(MLTest, neuralNetworkReset) {

  mat_type theta1, theta2, theta3, 
           t1zero, t2zero, t3zero,
           rt1, rt2, rt3;

  fromFile(theta1, "theta1_multi.txt");
  fromFile(theta2, "theta2_multi.txt");
  fromFile(theta3, "theta3_multi.txt");

  fromFile(rt1, "resetTheta1.txt");
  fromFile(rt2, "resetTheta2.txt");
  fromFile(rt3, "resetTheta3.txt");

  t1zero.resize(theta1.size1(), theta1.size2()); t1zero.clear();
  t2zero.resize(theta2.size1(), theta2.size2()); t2zero.clear();
  t3zero.resize(theta3.size1(), theta3.size2()); t3zero.clear();

  std::vector<mat_type> theta;
  theta.push_back(theta1);
  theta.push_back(theta2);
  theta.push_back(theta3);

  NeuralNetwork nn(theta);

  EXPECT_TRUE(areEqual(nn.theta()[0], theta1));
  EXPECT_TRUE(areEqual(nn.theta()[1], theta2));
  EXPECT_TRUE(areEqual(nn.theta()[2], theta3));

  // Use default seed.
  nn.reset();
  EXPECT_FALSE(areEqual(nn.theta()[0], theta1));
  EXPECT_FALSE(areEqual(nn.theta()[1], theta2));
  EXPECT_FALSE(areEqual(nn.theta()[2], theta3));
  EXPECT_FALSE(areEqual(nn.theta()[0], t1zero));
  EXPECT_FALSE(areEqual(nn.theta()[1], t2zero));
  EXPECT_FALSE(areEqual(nn.theta()[2], t3zero));

  nn.reset(42);
  EXPECT_TRUE(areEqual(nn.theta()[0], rt1));
  EXPECT_TRUE(areEqual(nn.theta()[1], rt2));
  EXPECT_TRUE(areEqual(nn.theta()[2], rt3));

  nn.resetZero();

  EXPECT_TRUE(areEqual(nn.theta()[0], t1zero));
  EXPECT_TRUE(areEqual(nn.theta()[1], t2zero));
  EXPECT_TRUE(areEqual(nn.theta()[2], t3zero));

}

TEST_F(MLTest, neuralNetworkPredict) {

  {
    mat_type theta1, theta2;
    vec_type input, output;

    fromVector(input, std::vector<double>{ 0.25, 0.36, 0.09, 0.51 });
    fromVector(output, std::vector<double>{ 0.45654283, 0.48344216, 0.47538256, 0.47840154 });

    fromFile(theta1, "trainedTheta1_nn.txt");
    fromFile(theta2, "trainedTheta2_nn.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);

    EXPECT_TRUE(areEqual(nn.predict(input), output));
  }

  {
    mat_type theta1, theta2, theta3;
    vec_type input, output;

    fromVector(input, std::vector<double>{ 0.25, 0.36, 0.09, 0.51 });
    fromVector(output, std::vector<double>{ 0.54022737, 0.53602241, 0.53744085, 0.56156298} );

    fromFile(theta1, "theta1_multi.txt");
    fromFile(theta2, "theta2_multi.txt");
    fromFile(theta3, "theta3_multi.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);
    theta.push_back(theta3);

    NeuralNetwork nn(theta);

    EXPECT_TRUE(areEqual(nn.predict(input), output));
  }
}

TEST_F(MLTest, neuralNetworkGradient) {

  // Do both tests from costCompute, but use computeGradient instead.
  // Check gradients while also checking that cost compute didn't change.
  {
    mat_type theta1, theta2, X, y, theta1trained, theta2trained,
                   grad1, grad2, grad1trained, grad2trained;

    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1, "simpleTheta1_nn.txt");
    fromFile(theta2, "simpleTheta2_nn.txt");
    fromFile(theta1trained, "trainedTheta1_nn.txt");
    fromFile(theta2trained, "trainedTheta2_nn.txt");
    fromFile(grad1, "grad1.txt");
    fromFile(grad2, "grad2.txt");
    fromFile(grad1trained, "grad1trained.txt");
    fromFile(grad2trained, "grad2trained.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);
    nn.installData(new MLData(X, y));

    EXPECT_EQ(nn.layerSizes()[0], 4);
    EXPECT_EQ(nn.layerSizes()[1], 3);
    EXPECT_EQ(nn.layerSizes()[2], 4);

    EXPECT_TRUE(areEqual(nn.theta()[0], theta1));
    EXPECT_TRUE(areEqual(nn.theta()[1], theta2));

    EXPECT_TRUE(areEqual(nn.computeGradient(), 6.45674664));
    EXPECT_TRUE(areEqual(nn.gradient()[0], grad1));
    EXPECT_TRUE(areEqual(nn.gradient()[1], grad2));

    std::vector<mat_type> thetaTrained;
    thetaTrained.push_back(theta1trained);
    thetaTrained.push_back(theta2trained);

    NeuralNetwork nn2(thetaTrained);
    nn2.installData(new MLData(X, y));

    EXPECT_TRUE(areEqual(nn2.computeGradient(), 2.66307595));
    EXPECT_TRUE(areEqual(nn2.gradient()[0], grad1trained));
    EXPECT_TRUE(areEqual(nn2.gradient()[1], grad2trained));
  }

  // Second test from costCompute (with regularization).
  {
    mat_type X, y, theta1trained, theta2trained,
                   grad1trained, grad2trained;

    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1trained, "trainedTheta1_nn.txt");
    fromFile(theta2trained, "trainedTheta2_nn.txt");
    fromFile(grad1trained, "grad1trained_reg.txt");
    fromFile(grad2trained, "grad2trained_reg.txt");

    std::vector<mat_type> thetaTrained;
    thetaTrained.push_back(theta1trained);
    thetaTrained.push_back(theta2trained);

    NeuralNetwork nn(thetaTrained);
    nn.installData(new MLData(X, y));
    nn.setRegLambda(1.0);

    EXPECT_TRUE(areEqual(nn.computeGradient(), 2.66974329));
    EXPECT_TRUE(areEqual(nn.gradient()[0], grad1trained));
    EXPECT_TRUE(areEqual(nn.gradient()[1], grad2trained));
  }

}

TEST_F(MLTest, neuralNetworkRollUnroll) {

  {
    NeuralNetwork nn(2, 3, 2);
    // Layer 1: 3x3
    // Layer 2: 2x4
    // Total nodes: 9 + 8 = 17
    EXPECT_EQ(nn.getUnrolledThetaSize(), 17);
  }

  {
    mat_type theta1trained, theta2trained;
    vec_type unrolledTrained, unrolledThetaOut;

    fromFile(theta1trained, "trainedTheta1_nn.txt");
    fromFile(theta2trained, "trainedTheta2_nn.txt");
    fromFile(unrolledTrained, "unrolled.txt");

    std::vector<double> unrolled(unrolledTrained.size());
    for (int i = 0; i < unrolledTrained.size(); ++i) {
      unrolled[i] = unrolledTrained[i];
    }

    NeuralNetwork nn(4, 3, 4);
    nn.setThetaFromUnrolled(unrolled);

    EXPECT_TRUE(areEqual(nn.theta()[0], theta1trained));
    EXPECT_TRUE(areEqual(nn.theta()[1], theta2trained));

    std::vector<double> unrollThetaTest(nn.getUnrolledThetaSize());
    nn.unrollTheta(unrollThetaTest);
    fromVector(unrolledThetaOut, unrollThetaTest);

    EXPECT_TRUE(areEqual(unrolledThetaOut, unrolledTrained));
  }

  {
    mat_type theta1, theta2, X, y, grad1, grad2;
    vec_type unrolledGrad, unrolledGradOut;

    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1, "simpleTheta1_nn.txt");
    fromFile(theta2, "simpleTheta2_nn.txt");
    fromFile(grad1, "grad1.txt");
    fromFile(grad2, "grad2.txt");
    fromFile(unrolledGrad, "unrolled_grad.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);
    nn.installData(new MLData(X, y));

    EXPECT_TRUE(areEqual(nn.computeGradient(), 6.45674664));
    EXPECT_TRUE(areEqual(nn.gradient()[0], grad1));
    EXPECT_TRUE(areEqual(nn.gradient()[1], grad2));

    std::vector<double> unrolledTest(nn.getUnrolledThetaSize());

    nn.unrollGradient(unrolledTest);
    fromVector(unrolledGradOut, unrolledTest);

    EXPECT_TRUE(areEqual(unrolledGradOut, unrolledGrad));
  }
}

TEST_F(MLTest, neuralNetworkTraining) {

  {
    mat_type theta1, theta2, X, y;

    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1, "simpleTheta1_nn.txt");
    fromFile(theta2, "simpleTheta2_nn.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);
    nn.installData(new MLData(X, y));
    nn.setRegLambda(1.0);

    nn.train(100);

  }
}

TEST_F(MLTest, neuralNetworkTrainingLarge) {

#ifdef LARGE_NN

  // Changed storage to std::vector - (78253 ms total)
  // Changed roll/unroll to ranged based for - (60847 ms total)

  {
    mat_type theta1, theta2, X, y;

    fromFile(X, "numbersX.txt");
    fromFile(y, "numbersY.txt");
    fromFile(theta1, "numbersTheta1.txt");
    fromFile(theta2, "numbersTheta2.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);
    nn.installData(new MLData(X, y));
    nn.setRegLambda(1.0);

    std::cout << "Cost from octave after 50 iterations: " << std::setprecision(9) << nn.computeGradient() << std::endl;
  }

  {
    mat_type X, y;

    fromFile(X, "numbersX.txt");
    fromFile(y, "numbersY.txt");

    NeuralNetwork nn(400, 25, 10);
    nn.installData(new MLData(X, y));

    nn.reset();

    nn.setRegLambda(1.0);

    nn.train(50);
    std::cout << "Cost after 50 iterations: " << std::setprecision(9) << nn.computeGradient() << std::endl;
    std::cout << "Performance on training data: " << std::setprecision(4) << nn.evaluate() << std::endl;
  }
#endif

}

TEST_F(MLTest, neuralNetworkEvaluate) {

  // With installed data.
  {
    mat_type theta1, theta2, X, y, actualPred;
    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1, "trainedTheta1_nn.txt");
    fromFile(theta2, "trainedTheta2_nn.txt");
    fromFile(actualPred, "simplePredict.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    NeuralNetwork nn(theta);
    nn.installData(new MLData(X, y));

    std::shared_ptr<mat_type> pred(new mat_type());

    EXPECT_TRUE(areEqual(nn.evaluate(), 0.40));
    EXPECT_TRUE(areEqual(nn.evaluate(pred), 0.40));
    EXPECT_TRUE(areEqual(*pred, actualPred));
  }

  // Without installed data.
  {
    mat_type theta1, theta2, X, y, actualPred;
    fromFile(X, "simpleX_nn.txt");
    fromFile(y, "simpleY_nn.txt");
    fromFile(theta1, "trainedTheta1_nn.txt");
    fromFile(theta2, "trainedTheta2_nn.txt");
    fromFile(actualPred, "simplePredict.txt");

    std::vector<mat_type> theta;
    theta.push_back(theta1);
    theta.push_back(theta2);

    MLData myData(X, y);

    NeuralNetwork nn(theta);
    std::shared_ptr<mat_type> pred(new mat_type());

    EXPECT_TRUE(areEqual(nn.evaluate(myData, pred), 0.40));
    EXPECT_TRUE(areEqual(*pred, actualPred));
  }
}

