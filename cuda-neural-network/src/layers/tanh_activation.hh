#pragma once

#include "nn_layer.hh"

class TanhActivation : public NNLayer {
private:
	Matrix A, A_n;

	Matrix Z, Z_n;
	Matrix dZ;

public:
	TanhActivation(std::string name);
	~TanhActivation();

	Matrix& forward(Matrix& Z);
	Matrix& normal(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
