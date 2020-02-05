	#include <iostream>
#include <time.h>
#include <chrono>

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
	int rows = 1, cols = 3;
	Matrix pnts(rows, cols);
	pnts.allocateMemory();
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			pnts[cols * i + j] = 1.0;
		}
	}
	pnts.copyHostToDevice();

	Matrix Y;

	auto start = chrono::steady_clock::now();
	Y = nn.forward(pnts);
	auto end = chrono::steady_clock::now();
	Y.copyDeviceToHost();
	// cout << Y[0] << ", " << Y[1] << ", " << Y[2] << endl;
    cout << "Elapsed time in microseconds: "
	<< chrono::duration_cast<chrono::microseconds> (end - start).count()
	<< " us" << endl;

	return 0;
}
