#pragma once

#include "nn_layer.hh"

class TanhActivation : public NNLayer {
private:
	matrix::Matrix A, A_n;

	matrix::Matrix Z, Z_n;
	matrix::Matrix dZ;

public:
	TanhActivation(std::string name);
	~TanhActivation();

	matrix::Matrix& forward(matrix::Matrix& Z);
	matrix::Matrix& normal(matrix::Matrix& Z);
	matrix::Matrix& normal_relu(matrix::Matrix& Z_n, matrix::Matrix& dh){return A;};

};
