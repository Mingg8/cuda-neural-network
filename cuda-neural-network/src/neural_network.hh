#pragma once

#include <vector>
#include "layers/nn_layer.hh"
#include "nn_utils/bce_cost.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);
	void initializeWeights(std::vector<Matrix> weights, std::vector<Matrix> biases);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
