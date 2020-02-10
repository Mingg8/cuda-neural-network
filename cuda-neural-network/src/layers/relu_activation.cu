#include "relu_activation.hh"
#include "../nn_utils/nn_exception.hh"

__global__ void reluActivationForward(float* Z, float* A,
									  int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
	}
}

__global__ void reluActivationNormal(float* Z, float *dh, float* A,
	int A_x_dim, int A_y_dim) {
	// A = relu_diff(Z_n)' .* dh

	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < A_x_dim && col < A_y_dim) {
		float relu = Z[col * A_x_dim + row] > 0 ? 1 : 0;
		A[row * A_y_dim + col] = relu * dh[row * A_y_dim + col];
	}
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

matrix::Matrix& ReLUActivation::forward(matrix::Matrix& Z) {
	// this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");

	return A;
}

matrix::Matrix& ReLUActivation::normal_relu(matrix::Matrix& Z_n, matrix::Matrix& dh) {
	// relu_diff(Z_n)' .* dh
	// this->Z_n = Z_n;
	// this->dh = dh;
	A_n.allocateMemoryIfNotAllocated(dh.shape);

	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(dh.shape.x + block_size.x - 1) / block_size.x,
						(dh.shape.y + block_size.y - 1) / block_size.y);

	reluActivationNormal<<<num_of_blocks, block_size>>>(Z_n.data_device.get(),
														dh.data_device.get(),
														A_n.data_device.get(),
														A_n.shape.x, A_n.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");

	return A_n;
}