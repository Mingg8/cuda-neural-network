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

	Matrix Z;
	Matrix Z_n;
	Matrix N;
	Matrix A;
	Matrix dA;

	void computeAndStoreLayerOutput(Matrix& A);
	void computeAndStoreLayerOutput_normal(Matrix& A);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	~LinearLayer();

	Matrix W;
	Matrix b;

	Matrix& forward(Matrix& A);
	Matrix& normal(Matrix& N);
	Matrix& normal_relu(Matrix& Z_n, Matrix& dh){return Z;};

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;

	// for unit testing purposes only
	friend class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	friend class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
};
