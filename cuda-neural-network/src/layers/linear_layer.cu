#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_layer.hh"
#include "../nn_utils/nn_exception.hh"

using namespace std;

__global__ void linearLayerForward( float* W, float* A, float* Z, float* b,
									int W_x_dim, int W_y_dim,
									int A_x_dim, int A_y_dim) {

	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A_x_dim;
	int Z_y_dim = W_y_dim;

	float Z_value = 0;

	if (row < Z_x_dim && col < Z_y_dim) {
		for (int i = 0; i < W_x_dim; i++) {
			Z_value += A[row * A_y_dim + i] * W[i * W_y_dim + col];
		}
		Z[row * Z_y_dim + col] = Z_value + b[col];
	}
}

__global__ void linearLayerNormal( float* W, float* A, float* Z, float* b,
	int W_x_dim, int W_y_dim,
	int A_x_dim, int A_y_dim) {

	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = W_x_dim;
	int Z_y_dim = A_y_dim;

	float Z_value = 0;

	if (row < Z_x_dim && col < Z_y_dim) {

		for (int i = 0; i < W_y_dim; i++) {
			Z_value += W[row * W_y_dim + i] * A[i * A_y_dim + col];
		}
		Z[row * Z_y_dim + col] = Z_value;
	}
}

LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
}

LinearLayer::~LinearLayer()
{ }

matrix::Matrix& LinearLayer::forward(matrix::Matrix& A) {
	assert(W.shape.x == A.shape.y);
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput(A);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");

	return Z;
}

matrix::Matrix& LinearLayer::normal(matrix::Matrix& N) {
	assert(W.shape.y == N.shape.x);
	Shape Z_shape(W.shape.x, N.shape.y);
	Z_n.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput_normal(N);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");

	return Z_n;
}

void LinearLayer::computeAndStoreLayerOutput_normal(matrix::Matrix& N) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z_n.shape.x + block_size.x - 1) / block_size.x,
						(Z_n.shape.y + block_size.y - 1) / block_size.y);

	linearLayerNormal<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   N.data_device.get(),
													   Z_n.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y,
													   N.shape.x, N.shape.y);
}

void LinearLayer::computeAndStoreLayerOutput(matrix::Matrix& A) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);

	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   A.data_device.get(),
													   Z.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y,
													   A.shape.x, A.shape.y);
}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

matrix::Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

matrix::Matrix LinearLayer::getBiasVector() const {
	return b;
}
