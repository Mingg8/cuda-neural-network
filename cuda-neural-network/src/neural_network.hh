#pragma once

#include <vector>
#include "layers/nn_layer.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;

	Matrix Y;
	Matrix dY;

	float learning_rate;
	Matrix input_coeff, output_coeff;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	void forward(Matrix X, Matrix& output, Matrix& normal);

	void addLayer(NNLayer *layer);
	void setCoeffs(Matrix& input, Matrix& output);
	std::vector<NNLayer*> getLayers() const;
	Matrix normalize(Matrix& pnts);
	Matrix unnormalize(Matrix &pnts);
	Matrix unnormalize_normal(Matrix &pnts);

};
