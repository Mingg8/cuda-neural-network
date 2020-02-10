#pragma once

#include <iostream>

#include "../nn_utils/matrix.hh"

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual matrix::Matrix& forward(matrix::Matrix& A) = 0;
	virtual matrix::Matrix& normal(matrix::Matrix& Z) = 0;

	virtual matrix::Matrix& normal_relu(matrix::Matrix& Z_n, matrix::Matrix& dh) = 0;

	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
