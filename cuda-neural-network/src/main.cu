#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "file_io.hh"

#include "coordinates_dataset.hh"


using namespace std;


int main() {
	srand( time(NULL) );

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(3, 64)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(64, 64)));
	nn.addLayer(new ReLUActivation("relu_2"));
	nn.addLayer(new LinearLayer("linear_3", Shape(64, 64)));
	nn.addLayer(new ReLUActivation("relu_3"));
	nn.addLayer(new LinearLayer("linear_4", Shape(64, 1)));
	nn.addLayer(new SigmoidActivation("tanh_output"));


	// TODO: set values in weights (linear_layer -> updateWeights)
	std::vector<Matrix> weights, biases;
	loadWeight(weights, biases);
	
	nn.initializeWeights(weights, biases);

	// // compute accuracy
	int rows = 4000, cols = 3;
	Matrix pnts(rows, cols);
	pnts.allocateMemory();
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			pnts[cols * i + j] = 1.0;
		}
	}

	Matrix Y;
	Y = nn.forward(pnts);
	Y.copyDeviceToHost();
	cout << Y[0] << ", " << Y[1] << ", " << Y[2] << endl;

	return 0;
}
