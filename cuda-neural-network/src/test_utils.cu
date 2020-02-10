#include <iostream>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "test_utils.hh"

namespace testutils {

	void initializeTensorWithValue(matrix::Matrix M, float value) {
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = value;
			}
		}
	}

	void initializeTensorWithMatrix(matrix::Matrix M, matrix::Matrix M_input) {
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = M_input[y * M.shape.x + x];
			}
		}
	}

	void initializeTensorRandomlyInRange(matrix::Matrix M, float min, float max) {
		srand( time(NULL) );
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = (static_cast<float>(rand()) / RAND_MAX) * (max - min) + min;
			}
		}
	}

	void initializeEachTensorRowWithValue(matrix::Matrix M, std::vector<float> values) {
		assert(M.shape.y == values.size());

		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = values[y];
			}
		}
	}

	void initializeEachTensorColWithValue(matrix::Matrix M, std::vector<float> values) {
		assert(M.shape.x == values.size());

		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = values[x];
			}
		}
	}

	float sigmoid(float x) {
		return exp(x) / (1 + exp(x));
	}

	void setData(matrix::Matrix X, Eigen::MatrixXd the_pnt_nut) {
		for (int x = 0; x < X.shape.x; x++) {
			for (int y = 0; y < X.shape.y; y++) {
				X[x * X.shape.y + y] = (float)the_pnt_nut(x, y);
			}
		}
	}


	Eigen::MatrixXd Matrix2Eigen(matrix::Matrix mat) {
		Eigen::MatrixXd eig(mat.shape.x, mat.shape.y);
		for (size_t i = 0; i < mat.shape.x; i++) {
			for (size_t j = 0; j < mat.shape.y; j++) {
				eig(i, j) = mat[i * mat.shape.y + j];
			}
		}
		return eig;
	}

}
