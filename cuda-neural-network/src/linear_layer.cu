#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_layer.hh"
#include "nn_exception.hh"
#include "nn_utils.hh"

__global__ void weightedSum(float* W, float* A, float* Z,
							int W_x_dim, int W_y_dim,
							int A_x_dim, int A_y_dim) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A_x_dim;
	int Z_y_dim = W_y_dim;

	float Z_value = 0;

	if (row < Z_y_dim && col < Z_x_dim) {
		for (int i = 0; i < W_x_dim; i++) {
			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
		}
		Z[row * Z_x_dim + col] = Z_value;
	}
}

__global__ void addBias(float* Z, float* b, int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		int row = index / Z_x_dim;
		int col = index % Z_x_dim;
		Z[index] += b[row];
	}
}

__global__ void linearLayerBackprop(float* W, float* dZ, float *dA,
									int W_x_dim, int W_y_dim,
									int dZ_x_dim, int dZ_y_dim) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// W is treated as transposed
	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;

	if (row < dA_y_dim && col < dA_x_dim) {
		for (int i = 0; i < W_y_dim; i++) {
			dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
}

__global__ void weightsGDC(float* dZ, float* A, float* W,
						   int dZ_x_dim, int dZ_y_dim,
						   int A_x_dim, int A_y_dim,
						   float learning_rate) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// A is treated as transposed
	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;

	float dW_value = 0.0f;

	if (row < W_y_dim && col < W_y_dim) {
		for (int i = 0; i < dZ_x_dim; i++) {
			dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
		}
		W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
	}
}

__global__ void biasGDC(float* dZ, float* b,
						int dZ_x_dim, int dZ_y_dim,
						int b_x_dim,
						float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
	}
}

LinearLayer::LinearLayer(std::string name, nn_utils::Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateCudaMemory();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot initialize layer bias.");
	W.allocateCudaMemory();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot initialize layer weights.");

	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

void LinearLayer::initializeWeightsRandomly() {
	W.allocateHostMemory();

	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
	W.freeHostMemory();
}

void LinearLayer::initializeBiasWithZeros() {
	b.allocateHostMemory();

	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copyHostToDevice();
	b.freeHostMemory();
}

LinearLayer::~LinearLayer() {
	W.freeCudaAndHostMemory();
	b.freeCudaAndHostMemory();
	Z.freeCudaAndHostMemory();
	dA.freeCudaAndHostMemory();
}

nn_utils::Tensor3D LinearLayer::forward(nn_utils::Tensor3D A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Z.allocateIfNotAllocated(nn_utils::Shape(A.shape.x, W.shape.y));

	dim3 block_size(4, 4);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);
	weightedSum<<<num_of_blocks, block_size>>>(W.data_device, A.data_device, Z.data_device,
											   W.shape.x, W.shape.y,
											   A.shape.x, A.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	block_size.x = 256; block_size.y = 1;
	num_of_blocks.x = (Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x;
	num_of_blocks.y = 1;
	addBias<<<num_of_blocks, block_size>>>(Z.data_device, b.data_device, Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	return Z;
}

nn_utils::Tensor3D LinearLayer::backprop(nn_utils::Tensor3D dZ, float learning_rate) {

	dA.allocateIfNotAllocated(A.shape);

	// compute dA
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	linearLayerBackprop<<<num_of_blocks, block_size>>>(W.data_device, dZ.data_device, dA.data_device,
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
	cudaDeviceSynchronize(); // TODO: probably some syncs can be removed
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	// compute db and do GDC
	block_size.x = 256; block_size.y = 1;
	num_of_blocks.x = (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x;
	num_of_blocks.y = 1;
	biasGDC<<<num_of_blocks, block_size>>>(dZ.data_device, b.data_device,
										   dZ.shape.x, dZ.shape.y,
										   b.shape.x, learning_rate);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform bias GDC.");

	// compute dW and do GDC
	block_size = dim3(16, 16);
	num_of_blocks = dim3((W.shape.x + block_size.x - 1) / block_size.x,
						 (W.shape.y + block_size.y - 1) / block_size.y);
	weightsGDC<<<num_of_blocks, block_size>>>(dZ.data_device, A.data_device, W.data_device,
											  dZ.shape.x, dZ.shape.y,
											  A.shape.x, A.shape.y,
											  learning_rate);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform weights GDC.");

	return dA;
}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

nn_utils::Tensor3D LinearLayer::getWeightsMatrix() const {
	return W;
}

nn_utils::Tensor3D LinearLayer::getBiasVector() const {
	return b;
}
