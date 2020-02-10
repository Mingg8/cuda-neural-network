#pragma once
#include "nn_layer.hh"

// for unit testing purposes only
namespace {
	class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
}

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	matrix::Matrix Z;
	matrix::Matrix Z_n;
	matrix::Matrix N;
	matrix::Matrix A;
	matrix::Matrix dA;

	void computeAndStoreLayerOutput(matrix::Matrix& A);
	void computeAndStoreLayerOutput_normal(matrix::Matrix& A);
	void updateWeights(matrix::Matrix& dZ, float learning_rate);
	void updateBias(matrix::Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	~LinearLayer();

	matrix::Matrix W;
	matrix::Matrix b;

	matrix::Matrix& forward(matrix::Matrix& A);
	matrix::Matrix& normal(matrix::Matrix& N);
	matrix::Matrix& normal_relu(matrix::Matrix& Z_n, matrix::Matrix& dh){return Z;};

	int getXDim() const;
	int getYDim() const;

	matrix::Matrix getWeightsMatrix() const;
	matrix::Matrix getBiasVector() const;

	// for unit testing purposes only
	friend class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	friend class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
};
