#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	Matrix A, A_n;

	Matrix Z, Z_n;
	Matrix dZ;
	Matrix dh;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z);
	Matrix& normal(Matrix& Z_n){return A;};
	Matrix& normal_relu(Matrix& Z_n, Matrix& dh);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
