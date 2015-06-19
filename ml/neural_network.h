////////////////////////////////////////////////////////////////////////////////
///
/// @file neural_network.h
/// @brief Basic neural network classifier.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_ML_NEURAL_NETWORK_H
#define XEN_ML_NEURAL_NETWORK_H

#include <xen/ml/ml_data.h>
#include <random>
#include <memory>
#include <nlopt.hpp>

namespace xen {
namespace ml {

// TODO: ...
// - Online learning (one example at a time)

class NeuralNetwork {
public:
  typedef MLData::mat_type mat_type;
  typedef MLData::vec_type vec_type;

  /** @brief Creates a neural network with the provided layer sizes. Assuming
    *        layerSizes has 'n' entries, there will be (n - 2) hidden layers
    *        with layerSizes[0] nodes in the input layer, layerSizes[1] nodes
    *        in the 2nd layer, up to layerSizes[n-2] in the final hidden layer.
    *        There will be layerSizes[n-1] nodes in the output layer. The
    *        layer sizes for all layers should NOT include the bias units that
    *        will be added automatically.
    * @param layerSizes   Vector of number of nodes in each layer as described
    *                     above. ASSERT(layerSizes.size() >= 3)
    */
  explicit NeuralNetwork(const std::vector<int>& layerSizes);

  /** @brief Convenience constructor for networks with only 1 hidden layer.
    */
  NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize);

  /** @brief Construct a network with a predefined set of starting weights.
    */
  explicit NeuralNetwork(const std::vector<mat_type>& theta);

  /** @brief Installs a provided MLData which will be used for training of
    *        this neural network. Takes ownership of the data.
    */
  void installData(std::unique_ptr<MLData> data);
  void installData(MLData* data);

  /** @brief Uses the given matrices of examples to train the neural network.
    * @param iter   Number of iterations of optimization to perform.
    * @param algo   NLOPT algorithm to use.
    * @param toler  Tolerance at which point to stop iterating.
    * @return Final value of the cost function on last iteration.
    */
  double train(int iter, nlopt::algorithm algo = nlopt::LD_LBFGS, double toler = 1e-6);

  /** @brief Returns a prediction vector (values in the output layer) for
    *        a given input vector.
    */
  vec_type predict(const vec_type& input) const;

  /** @brief Test the performance of the neural network against the already installed
    *        set of data. Returns the fraction of observations for which the network
    *        correctly predicts the output vector.
    * @param pred   If provided, the prediction vector for each input will be assigned
    *               to the corresponding row in this matrix. The matrix will be resized
    *               as needed.
    */
  double evaluate(std::shared_ptr<mat_type> pred = std::shared_ptr<mat_type>()) const;

  /** @brief Test the performance of the neural network against the given set of data.
    *        Returns the fraction of observations for which the network correctly predicts 
    *        the output vector.
    * @param data   Data set to test the network against.
    * @param pred   If provided, the prediction vector for each input will be assigned
    *               to the corresponding row in this matrix. The matrix will be resized
    *               as needed.
    */
  double evaluate(const MLData& data, std::shared_ptr<mat_type> pred = std::shared_ptr<mat_type>()) const;

  /** @brief Compute gradient (and as a result, cost) for the currently installed
    *        MLData. Updated gradient will be in m_gradient.
    * @return Value of the cost function for supplied observations and current theta.
    */
  double computeGradient();

  /** @brief Reset all layer weights to random values in the range [-e_init, e_init]
    *        Where:
    *           e_init = sqrt(6) / sqrt(L_in + L_out)
    *           L_in = number of units on the input side of a given theta
    *           L_out = number of units on the output side of a given theta
    */
  void reset();

  /** @brief Resets the layer weights with a specific seed for the random number
    *        generator. Used primarily for testing purposes.
    */
  void reset(std::mt19937::result_type seed);

  /** @brief Reset all layer weights to zero.
    */
  void resetZero();

  /** @brief Used for NLOPT library optimization functions.
    */
  double nloptOptimize(const std::vector<double> &x, std::vector<double> &grad);

  ////////////////////////////////////////////////////////////////////////////////
  //                                ACCESSORS                                   //
  ////////////////////////////////////////////////////////////////////////////////

  const std::vector<int>& layerSizes() const { return m_layerSizes; }
  const std::vector<mat_type>& gradient() const { return m_gradient; }
  const std::vector<mat_type>& theta() const { return m_theta; }

  void setRegLambda(const double lambda) { m_regLambda = lambda; }
  double getRegLambda() const { return m_regLambda; }

  // Roll/unroll helper functions.
  void setThetaFromUnrolled(const std::vector<double> &theta);
  void unrollTheta(std::vector<double>& theta) const;
  void unrollGradient(std::vector<double>& grad) const;
  int getUnrolledThetaSize() const;

private:

  /// Resets the gradient to all zeros.
  void resetGradient();

  // Utilities for random initialization
  std::random_device m_randDevice;
  std::mt19937 m_randGen;

  /// Regularization lambda, used in both theta and gradient calcs
  double m_regLambda = 0.0;

  /// Track the number of iterations done during training
  int m_optIters = 0;

  /// The machine learning data to use for training.
  std::unique_ptr<MLData> m_data;

  /// Input, hidden and output layers
  std::vector<int> m_layerSizes;

  /// Learned parameters
  std::vector<mat_type> m_theta;

  /// Gradient for use in training
  std::vector<mat_type> m_gradient;

  // No default construction.
  NeuralNetwork() = delete;
  
  // No copy construction/assignment.
  NeuralNetwork(const NeuralNetwork&) = delete;
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

};

} // end namespace ml
} // end namespace xen

#endif // XEN_ML_NEURAL_NETWORK_H

