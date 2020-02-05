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

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

void NeuralNetwork::forward(Matrix X, Matrix& output, Matrix& normal) {
	Matrix Z = X;

	Matrix a1 = layers[0]->forward(Z);
	Matrix a2 = layers[1]->forward(a1);
	a2 = layers[2]->forward(a2);
	Matrix a3 = layers[3]->forward(a2);
	a3 = layers[4]->forward(a3);
	Matrix a4 = layers[5]->forward(a3);
	a4 = layers[6]->forward(a4);
	output = layers[7]->forward(a4);

	Matrix dh4 = layers[7]->normal(a4); // N x 1
	dh4 = layers[6]->normal(dh4); // (64 x 1) x (N x 1)'
	Matrix dh3 = layers[5]->normal(a5, dh4);
	dh3 = layers[4]->normal(dh3);
	normal = dh4;
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
