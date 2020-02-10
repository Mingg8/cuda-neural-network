#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	matrix::Matrix A, A_n;

	matrix::Matrix Z, Z_n;
	matrix::Matrix dZ;
	matrix::Matrix dh;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	matrix::Matrix& forward(matrix::Matrix& Z);
	matrix::Matrix& normal(matrix::Matrix& Z_n){return A;};
	matrix::Matrix& normal_relu(matrix::Matrix& Z_n, matrix::Matrix& dh);
};
