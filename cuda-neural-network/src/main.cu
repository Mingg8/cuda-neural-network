#include <iostream>
#include <time.h>
#include <vector>
#include <chrono>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/tanh_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "file_io.hh"
#include "../../cuda-neural-network-test/test/test_utils.hh"

#include "coordinates_dataset.hh"

int main() {

	srand( time(NULL) );
	int rows = 4900;

	NeuralNetwork nn;
	LinearLayer* linear_layer_1 = new LinearLayer("linear_layer_1", Shape(3, 64));
	ReLUActivation* relu_layer_1 = new ReLUActivation("relu_layer_1");
	LinearLayer* linear_layer_2 = new LinearLayer("linear_layer_2", Shape(64, 64));
	ReLUActivation* relu_layer_2 = new ReLUActivation("relu_layer_2");
	LinearLayer* linear_layer_3 = new LinearLayer("linear_layer_3", Shape(64, 64));
	ReLUActivation* relu_layer_3 = new ReLUActivation("relu_layer_3");
	LinearLayer* linear_layer_4 = new LinearLayer("linear_layer_4", Shape(64, 1));
	TanhActivation* tanh_layer = new TanhActivation("tanh_layer");

	std::vector<Matrix> weights, biases;
	Matrix input_coeff, output_coeff;
	loadWeight(weights, biases);
	loadNormalizationCoeff(input_coeff, output_coeff);
	input_coeff.copyHostToDevice();
	output_coeff.copyHostToDevice();
	nn.setCoeffs(input_coeff, output_coeff);

	Matrix a = weights[0];
	testutils::initializeTensorWithMatrix(linear_layer_1->W, weights[0]);
	testutils::initializeTensorWithMatrix(linear_layer_2->W, weights[1]);
	testutils::initializeTensorWithMatrix(linear_layer_3->W, weights[2]);
	testutils::initializeTensorWithMatrix(linear_layer_4->W, weights[3]);

	testutils::initializeTensorWithMatrix(linear_layer_1->b, biases[0]);
	testutils::initializeTensorWithMatrix(linear_layer_2->b, biases[1]);
	testutils::initializeTensorWithMatrix(linear_layer_3->b, biases[2]);
	testutils::initializeTensorWithMatrix(linear_layer_4->b, biases[3]);

	linear_layer_1->W.copyHostToDevice();
	linear_layer_2->W.copyHostToDevice();
	linear_layer_3->W.copyHostToDevice();
	linear_layer_4->W.copyHostToDevice();

	linear_layer_1->b.copyHostToDevice();
	linear_layer_2->b.copyHostToDevice();
	linear_layer_3->b.copyHostToDevice();
	linear_layer_4->b.copyHostToDevice();

	nn.addLayer(linear_layer_1);
	nn.addLayer(relu_layer_1);
	nn.addLayer(linear_layer_2);
	nn.addLayer(relu_layer_2);
	nn.addLayer(linear_layer_3);
	nn.addLayer(relu_layer_3);
	nn.addLayer(linear_layer_4);
	nn.addLayer(tanh_layer);

	Matrix X(rows, 3);
	X.allocateMemory();
	Matrix Y, N;

	testutils::initializeTensorWithValue(X, 0.5f);
	
	X[0] = 0.0165;
	X[1] = 0.0165;
	X[2] = 0.0371; // 60 us

	auto start = chrono::steady_clock::now();
	
	nn.forward(X, Y, N);
	Y.copyDeviceToHost();
	N.copyDeviceToHost();

	auto end = chrono::steady_clock::now();

	cout << "Elapsed time (microseconds) : "
		<< "Total: "
		<< chrono::duration_cast<chrono::microseconds>(end - start).count()
		<< " (us) " << endl;

	cout << Y[0] << ", " << Y[1] << ", " << Y[2] << ", " << Y[3] << endl;
	cout << N[0] << ", " <<  N[4900] << ", " << N[9800] << endl;

	return 0;
}

Matrix broadPhase(Matrix pnts) {
	
}