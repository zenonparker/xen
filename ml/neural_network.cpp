// xen
#include <xen/ml/neural_network.h>
#include <xen/log/basic.h>

// boost
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/assert.hpp>

using namespace xen::math::ublas;
using namespace xen::math;

namespace {

const char* NO_INSTALLED_DATA = "No MLData installed in NeuralNetwork.";

double nloptCallback(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
  return reinterpret_cast<xen::ml::NeuralNetwork*>(my_func_data)->nloptOptimize(x, grad);
}

} // end unnamed namespacek

namespace xen {
namespace ml {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes)
  : m_layerSizes(layerSizes),
    m_theta(layerSizes.size() - 1),
    m_gradient(layerSizes.size() - 1)
{
  BOOST_ASSERT(m_layerSizes.size() >= 3);
  for (int i = 0; i < (layerSizes.size() - 1); ++i) {
    // Size of theta between layer A and B is: (row X col)
    //   (num nodes in B) X (num nodes in A + 1 for bias unit)
    m_theta[i].resize(layerSizes[i + 1], layerSizes[i] + 1);
    m_gradient[i].resize(layerSizes[i + 1], layerSizes[i] + 1);
  }
}

NeuralNetwork::NeuralNetwork(int inputLayerSize, 
                             int hiddenLayerSize,
                             int outputLayerSize) :
  NeuralNetwork(std::vector<int>{inputLayerSize,
                                 hiddenLayerSize,
                                 outputLayerSize})
{
}

NeuralNetwork::NeuralNetwork(const std::vector<mat_type>& theta)
  : m_theta(theta)
{
  m_layerSizes.resize(theta.size() + 1);
  BOOST_ASSERT(m_layerSizes.size() >= 3);
  m_layerSizes[0] = theta[0].size2() - 1;
  m_gradient.resize(m_theta.size());
  for (int i = 1; i <= theta.size(); ++i) {
    m_layerSizes[i] = theta[i - 1].size1();
    m_gradient[i - 1].resize(theta[i - 1].size1(), theta[i - 1].size2());
  }
}

void NeuralNetwork::installData(std::unique_ptr<MLData> data)
{
  m_data = std::move(data);
}

void NeuralNetwork::installData(MLData* data)
{
  m_data.reset(data);
}

double NeuralNetwork::nloptOptimize(const std::vector<double>& x, 
                                    std::vector<double>& grad)
{
  // Set up theta based on incoming x.
  setThetaFromUnrolled(x);

  double thisCost = computeGradient();
  DEBUG_LOG("iter[" << ++m_optIters << "] cost: " << thisCost);

  // Set unrolled gradient.
  unrollGradient(grad);

  return thisCost;
}

double NeuralNetwork::train(int iter, 
                            nlopt::algorithm algo /* = nlopt::LD_LBFGS */, 
                            double toler /* = 1e-6 */)
{
  nlopt::opt opt(algo, getUnrolledThetaSize());

  opt.set_min_objective(nloptCallback, this);

  opt.set_ftol_rel(toler);
  opt.set_maxeval(iter);

  m_optIters = 0;

  std::vector<double> x(getUnrolledThetaSize());
  unrollTheta(x);

  double minf;
  nlopt::result result = opt.optimize(x, minf);

  DEBUG_LOG("x[0]: " << x[0]);
  DEBUG_LOG("minf: " << minf);
  DEBUG_LOG(result);
}

NeuralNetwork::vec_type NeuralNetwork::predict(const vec_type& input) const
{
  vec_type lastA(input.size() + 1);
  lastA[0] = 1.0;
  subrange(lastA, 1, lastA.size()) = input;

  vec_type lastZ(prod(m_theta[0], lastA));

  for (int thisTheta = 1; thisTheta < m_theta.size(); ++thisTheta) {
    sigmoid(lastZ);
    vec_type thisA(lastZ.size() + 1);
    thisA[0] = 1.0;
    subrange(thisA, 1, thisA.size()) = lastZ;

    vec_type thisZ(prod(m_theta[thisTheta], thisA));

    // Prepare for next iteration.
    lastA.swap(thisA);
    lastZ.swap(thisZ);
  }

  sigmoid(lastZ);
  // Hypothesis = h_theta = lastZ
  return lastZ;
}

double NeuralNetwork::evaluate(std::shared_ptr<mat_type> pred /* = std::shared_ptr<mat_type>() */) const
{
  if (!m_data) throw std::runtime_error(NO_INSTALLED_DATA);
  return evaluate(*m_data, pred);
}

double NeuralNetwork::evaluate(const MLData& data, 
                               std::shared_ptr<mat_type> pred /* = std::shared_ptr<mat_type>() */) const
{
  const mat_type& X = data.X();
  const mat_type& y = data.y();

  int numCorrect = 0;
  if (pred.get()) pred->resize(y.size1(), y.size2(), false);

  // For each example...
  for (int i = 0; i < X.size1(); ++i) {
    vec_type thisY(row(y, i));

    // First layer is calculated from inputs, so we treat
    // it a bit differently outside of the layer loop below.
    vec_type lastA(X.size2() + 1);
    lastA[0] = 1.0;
    subrange(lastA, 1, lastA.size()) = row(X, i);
  
    vec_type lastZ(prod(m_theta[0], lastA));

    for (int thisTheta = 1; thisTheta < m_theta.size(); ++thisTheta) {
      sigmoid(lastZ);
      vec_type thisA(lastZ.size() + 1);
      thisA[0] = 1.0;
      subrange(thisA, 1, thisA.size()) = lastZ;

      vec_type thisZ(prod(m_theta[thisTheta], thisA));

      // Prepare for next iteration.
      lastA.swap(thisA);
      lastZ.swap(thisZ);
    }

    sigmoid(lastZ);
    if (pred.get()) {
      row(*pred, i) = lastZ;
    }

    if (maxIndex(lastZ) == maxIndex(thisY)) {
      numCorrect++;
    }
  }
  
  return (double(numCorrect) / double(X.size1()));
}

double NeuralNetwork::computeGradient()
{
  if (!m_data) throw std::runtime_error(NO_INSTALLED_DATA);
  const mat_type& X = m_data->X();
  const mat_type& y = m_data->y();

  double cost = 0.0;
  double m = X.size1();

  resetGradient();

  std::vector<vec_type> Z(m_theta.size());
  std::vector<vec_type> A(m_theta.size());

  // For each example...
  for (int i = 0; i < X.size1(); ++i) {

    // First layer is calculated from inputs, so we treat
    // it a bit differently outside of the layer loop below.
    vec_type lastA(X.size2() + 1);
    lastA[0] = 1.0;
    subrange(lastA, 1, lastA.size()) = row(X, i);
  
    vec_type lastZ(prod(m_theta[0], lastA));

    resizeAndAssign(A[0], lastA); // a1 in octave code.
    resizeAndAssign(Z[0], lastZ); // z2 in octave code.

    for (int thisTheta = 1; thisTheta < m_theta.size(); ++thisTheta) {
      sigmoid(lastZ);
      vec_type thisA(lastZ.size() + 1);
      thisA[0] = 1.0;
      subrange(thisA, 1, thisA.size()) = lastZ;

      vec_type thisZ(prod(m_theta[thisTheta], thisA));

      // Prepare for next iteration.
      lastA.swap(thisA);
      lastZ.swap(thisZ);

      resizeAndAssign(A[thisTheta], lastA); // a{thisTheta+1} in octave code.
      resizeAndAssign(Z[thisTheta], lastZ); // z{thisTheta+2} in octave code.
    }

    sigmoid(lastZ);
    vec_type& h_theta = lastZ;

    vec_type lastD(h_theta - row(y, i));

    m_gradient[m_gradient.size() - 1] += outer_prod(lastD, A[m_theta.size() - 1]);

    for (int j = m_theta.size() - 2; j >= 0; --j) {
      vec_type thisD = prod(lastD, m_theta[j+1]);
      vec_type thisGrad(Z[j].size() + 1);
      thisGrad[0] = 1.0;
      subrange(thisGrad, 1, thisGrad.size()) = Z[j];
      sigmoidGradient(thisGrad);

      elementWiseOper(Oper::PROD, thisD, thisGrad);

      m_gradient[j] += outer_prod(subrange(thisD, 1, thisD.size()), A[j]);
      
      lastD.resize(thisD.size() - 1, false);
      lastD.assign(subrange(thisD, 1, thisD.size()));
    }

    vec_type log_ht(h_theta);
    log_vec(log_ht);

    vec_type log_1m_ht(scalar_vector<double>(h_theta.size(), 1));
    log_1m_ht -= h_theta;
    log_vec(log_1m_ht);

    double tempSum = 0.0;
    for (int j = 0; j < log_ht.size(); ++j) {
      tempSum += (log_ht[j] * -y(i, j)) - (log_1m_ht[j] * (1 - y(i, j)));
    }

    cost += tempSum / m;
  }

  // Gradient calculation, m_gradient is 'capd' from octave code at this point.
  for (int i = 0; i < m_gradient.size(); ++i) {
    m_gradient[i] /= m;
  }

  // Only perform regularization if we really need to.
  if (!areEqual(m_regLambda, 0.0)) {
    double tempRegSum = 0.0;
    double gradientScalar = (m_regLambda / m);
    // Skip first column (don't want bias unit).
    for (int thisTheta = 0; thisTheta < m_theta.size(); ++thisTheta) {
      for (int i = 0; i < m_theta[thisTheta].size1(); ++i) {
        for (int j = 1; j < m_theta[thisTheta].size2(); ++j) {
          // Gradient regularization.
          m_gradient[thisTheta](i,j) += gradientScalar * m_theta[thisTheta](i,j);
          // Cost regularization.
          tempRegSum += m_theta[thisTheta](i,j) * m_theta[thisTheta](i,j);
        }
      }
    }
    cost += (m_regLambda / (2.0 * m)) * tempRegSum;
  }

  return cost;
}

void NeuralNetwork::reset(std::mt19937::result_type seed)
{
  m_randGen.seed(seed);
  for (int t = 0; t < m_theta.size(); ++t) {
    // Determine the 'e_init' value for the random range in this layer.
    double e_init = sqrt(6.0) / 
                    sqrt(double(m_theta[t].size2() - 1.0 + 
                                m_theta[t].size1()));
    std::uniform_real_distribution<double> dist(-e_init, e_init);
    for (int i = 0; i < m_theta[t].size1(); ++i) {
      for (int j = 0; j < m_theta[t].size2(); ++j) {
        m_theta[t](i,j) = dist(m_randGen);
      }
    }
  }
}

void NeuralNetwork::reset()
{
  reset(m_randDevice());
}

void NeuralNetwork::resetZero()
{
  for (mat_type& m : m_theta) m.clear();
}

void NeuralNetwork::setThetaFromUnrolled(const std::vector<double>& theta)
{
  int tInd = 0;
  for (mat_type& m : m_theta) {
    for (double& d : m.data()) d = theta[tInd++];
  }
}

void NeuralNetwork::unrollTheta(std::vector<double>& theta) const
{
  int tInd = 0;
  for (const mat_type& m : m_theta) {
    for (const double& d : m.data()) theta[tInd++] = d;
  }
}

void NeuralNetwork::unrollGradient(std::vector<double>& grad) const
{
  int gInd = 0;
  for (const mat_type& m : m_gradient) {
    for (const double& d : m.data()) grad[gInd++] = d;
  }
}

int NeuralNetwork::getUnrolledThetaSize() const
{
  int totalTheta = 0;
  for (const mat_type& m : m_theta) totalTheta += m.size1() * m.size2();
  return totalTheta;
}

// PRIVATE

void NeuralNetwork::resetGradient() {
  for (mat_type& m : m_gradient) m.clear();
}

} // end namespace ml
} // end namespace xen

