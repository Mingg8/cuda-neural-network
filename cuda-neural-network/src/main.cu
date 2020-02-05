#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"

#include "coordinates_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	srand( time(NULL) );

	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(3, 64)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(64, 64)));
	nn.addLayer(new ReLUActivation("sigmoid_output"));
	nn.addLayer(new LinearLayer("linear_3", Shape(64, 64)));
	nn.addLayer(new SigmoidActivation("tanh_output"));

	// network training
	Matrix Y;
	// for (int epoch = 0; epoch < 1001; epoch++) {
	// 	float cost = 0.0;

	// 	for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
	// 		Y = nn.forward(dataset.getBatches().at(batch));
	// 		nn.backprop(Y, dataset.getTargets().at(batch));
	// 		cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
	// 	}

	// 	if (epoch % 100 == 0) {
	// 		std::cout 	<< "Epoch: " << epoch
	// 					<< ", Cost: " << cost / dataset.getNumOfBatches()
	// 					<< std::endl;
	// 	}
	// }

	// TODO: set values in weights (linear_layer -> updateWeights)
	std::vector<Matrix> weights, biases;
	
	nn.initializeWeights(weights, biases);

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	// float accuracy = computeAccuracy(
	// 		Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	// std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}



// void initializeWeightsRandomly(Matrix &W) {
// 	std::default_random_engine generator;
// 	std::normal_distribution<float> normal_distribution(0.0, 1.0);

// 	for (int x = 0; x < W.shape.x; x++) {
// 		for (int y = 0; y < W.shape.y; y++) {
// 			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
// 		}
// 	}

// 	W.copyHostToDevice();
// }


// void LinearLayer::initializeBiasWithZeros() {
// 	for (int x = 0; x < b.shape.x; x++) {
// 		b[x] = 0;
// 	}

// 	b.copyHostToDevice();
// }