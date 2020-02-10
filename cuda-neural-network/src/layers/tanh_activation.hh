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
	Matrix& normal_relu(Matrix& Z_n, Matrix& dh){return A;};

};
