#pragma once

#include <vector>
#include "layers/nn_layer.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;

	matrix::Matrix Y;
	matrix::Matrix dY;

	float learning_rate;
	matrix::Matrix input_coeff, output_coeff;

	matrix::Matrix normalize(matrix::Matrix& pnts);
	matrix::Matrix unnormalize(matrix::Matrix &pnts);
	matrix::Matrix unnormalize_normal(matrix::Matrix &pnts);
	matrix::Matrix transform(matrix::Matrix &pnts, matrix::Matrix &rot, 
		matrix::Matrix &trans);

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	void forward(matrix::Matrix X, matrix::Matrix& output, matrix::Matrix& normal,
		matrix::Matrix rot, matrix::Matrix trans);

	void addLayer(NNLayer *layer);
	void setCoeffs(matrix::Matrix& input, matrix::Matrix& output);
	std::vector<NNLayer*> getLayers() const;

};
