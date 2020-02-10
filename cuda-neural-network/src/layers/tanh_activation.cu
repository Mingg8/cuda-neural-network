#include "tanh_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__device__ float tanh_diff(float x) {
	float coshx = (exp(x) + exp(-x)) / 2;
	return 1.0f / (coshx * coshx);
}

__global__ void tanhActivationForward(float* Z, float* A,
										 int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = tanh(Z[index]);
	}
}


__global__ void tanhActivationNormal(float* Z, float* A,
	int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = tanh_diff(Z[index]);
	}
}

TanhActivation::TanhActivation(std::string name) {
	this->name = name;
}

TanhActivation::~TanhActivation()
{ }

matrix::Matrix& TanhActivation::forward(matrix::Matrix& Z) {
	// this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	tanhActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh forward propagation.");

	return A;
}

matrix::Matrix& TanhActivation::normal(matrix::Matrix& Z_n) {
	// this->Z_n = Z_n;
	A_n.allocateMemoryIfNotAllocated(Shape(Z_n.shape.y, Z_n.shape.x));

	dim3 block_size(256);
	dim3 num_of_blocks((Z_n.shape.y * Z_n.shape.x + block_size.x - 1) / block_size.x);

	tanhActivationNormal<<<num_of_blocks, block_size>>>(Z_n.data_device.get(), A_n.data_device.get(),
														   	Z_n.shape.x, Z_n.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh forward propagation.");

	return A_n;
}