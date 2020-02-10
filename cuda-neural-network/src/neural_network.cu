#include "neural_network.hh"
#include "nn_utils/nn_exception.hh"
#include <math.h>

#include <chrono>

using namespace std;

__global__ void normalization(float* pnts, float* coeff, float* n_pnts, int x, int y) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < x && col < y) {
		n_pnts[row * y + col] = pnts[row * y + col] * coeff[col] + coeff[col + 3];
	}
}

__global__ void unnormalization(float* pnts, float* coeff, float* n_pnts, int x, int y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < x * y) {
		n_pnts[index] = (pnts[index] - coeff[4])  / coeff[0];
	}
}

__global__ void normal_unnormalization(float* pnts, float* coeff, float* n_pnts, int x, int y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < x) {
		float sum = 0.0f;
		float a[3];
		for (size_t i = 0; i < 3; i++) {
			a[i] = pnts[index + i * y] * coeff[i];
			sum += a[i] * a[i];
		}
		for (size_t i = 0; i < 3; i++) {
			n_pnts[index + i * y] = a[i] / sqrt(sum);
		}
		
	}
}

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

void NeuralNetwork::forward(Matrix Z, Matrix& output, Matrix& normal) {
	Z.copyHostToDevice();
	Z = this->normalize(Z); // 18 us

	Matrix a1 = layers[0]->forward(Z);
	Matrix a2 = layers[1]->forward(a1); // 326
	a2 = layers[2]->forward(a2); 
	Matrix a3 = layers[3]->forward(a2); // 321
	a3 = layers[4]->forward(a3);
	Matrix a4 = layers[5]->forward(a3); // 319
	a4 = layers[6]->forward(a4);
	output = layers[7]->forward(a4); // 18
	output = this->unnormalize(output);

	normal = layers[7]->normal(a4); // N x 1
	normal = layers[6]->normal(normal); // (64 x 1) x (N x 1)'
	normal = layers[5]->normal_relu(a3, normal);
	normal = layers[4]->normal(normal);
	normal = layers[3]->normal_relu(a2, normal);
	normal = layers[2]->normal(normal);
	normal = layers[1]->normal_relu(a1, normal);
	normal = layers[0]->normal(normal);
	normal = unnormalize_normal(normal);
}
std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}


void NeuralNetwork::setCoeffs(Matrix& input, Matrix& output) {
	input_coeff = input;
	output_coeff = output;
}

Matrix NeuralNetwork::normalize(Matrix &pnts) {
	Matrix normalized_pnts;
	normalized_pnts.allocateMemoryIfNotAllocated(pnts.shape);

	dim3 block_size(8, 8);
	dim3 num_of_blocks( (pnts.shape.x + block_size.x - 1) / block_size.x,
						(pnts.shape.y + block_size.y - 1) / block_size.y);
	normalization<<<num_of_blocks, block_size>>>(pnts.data_device.get(),
												input_coeff.data_device.get(),
												normalized_pnts.data_device.get(),
												pnts.shape.x, pnts.shape.y);
	return normalized_pnts;
}

Matrix NeuralNetwork::unnormalize(Matrix &pnts) {
	Matrix normalized_pnts;
	normalized_pnts.allocateMemoryIfNotAllocated(pnts.shape);
	dim3 block_size(256);
	dim3 num_of_blocks((pnts.shape.y * pnts.shape.x + block_size.x - 1) / block_size.x);
	unnormalization<<<num_of_blocks, block_size>>>(pnts.data_device.get(),
												output_coeff.data_device.get(),
												normalized_pnts.data_device.get(),
												pnts.shape.x, pnts.shape.y);
	return normalized_pnts;
}

Matrix NeuralNetwork::unnormalize_normal(Matrix &pnts) {
	Matrix normalized_pnts;
	normalized_pnts.allocateMemoryIfNotAllocated(pnts.shape);
	dim3 block_size(256);
	dim3 num_of_blocks((pnts.shape.y + block_size.x - 1) / block_size.x);
	normal_unnormalization<<<num_of_blocks, block_size>>>(pnts.data_device.get(),
												input_coeff.data_device.get(),
												normalized_pnts.data_device.get(),
												pnts.shape.x, pnts.shape.y);
	return normalized_pnts;
}