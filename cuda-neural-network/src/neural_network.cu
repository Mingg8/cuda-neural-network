#include "neural_network.hh"
#include "nn_utils/nn_exception.hh"

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::initializeWeights(std::vector<Matrix> weights, std::vector<Matrix> biases) {
	layers[0]->initializeWeight(weights[0]);
	layers[0]->initializeBias(biases[0]);
	layers[2]->initializeWeight(weights[1]);
	layers[2]->initializeBias(biases[1]);
	layers[4]->initializeWeight(weights[2]);
	layers[4]->initializeBias(biases[2]);
	layers[6]->initializeWeight(weights[3]);
	layers[6]->initializeBias(biases[3]);
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix error = bce_cost.dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
