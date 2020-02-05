#include "file_io.hh"
#include <iostream>

using namespace std;

void loadWeight(std::vector<Matrix> &weight, std::vector<Matrix> &bias) {
	std::string dir = "/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64/weight_csv";
	int lnum = 2;
	int hdim = 64;

    int num = 0;

    std::string file = dir + std::string("/weight") + std::to_string(num) + std::string(".csv");
	num++;
    Matrix W = readCsv(file, 3, 64);

    file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
    num++;
    Matrix b = readCsv_vec(file, hdim);

    weight.push_back(W);
    bias.push_back(b);
    for (int i = 0; i < lnum; i++) {
        file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
        num++;
        W = readCsv(file, hdim, hdim);

        file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
        num++;
        b = readCsv_vec(file, hdim);

        weight.push_back(W);
        bias.push_back(b);
    }

    file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
    num++;
    W = readCsv_last(file, hdim);

    file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
    num++;
	b = readCsv_vec_last(file);

    weight.push_back(W);
    bias.push_back(b);
}

void loadNormalizationCoeff(Matrix& input_coeff, Matrix& output_coeff) {
	std::string dir = "/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64/weight_csv";
    string file_input = dir + string("/input_coeff.csv");
    string file_output = dir + string("/output_coeff.csv");
    input_coeff = readCsv_vec(file_input, 6);
    output_coeff = readCsv_vec(file_output, 8);
}


Matrix readCsv(std::string file, int rows, int cols) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;
    
	Matrix res(rows, cols);
	res.allocateMemory();
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == ',') {
					res[row * res.shape.y + col] = (float)atof(start);
					col++;
                    start = ptr + 1 + i;
                }
            }
			res[row * res.shape.y + col] = (float)atof(start);
			col++;
            row++;
        }
        in.close();
	}
    return res;
}

Matrix readCsv_last(std::string file, int rows) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;
    
	Matrix res(rows, 1);
	res.allocateMemory();
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == ',') {
                    if (col == 0) {
                        res[row * res.shape.y + col] = (float)atof(start);
                    }
					col++;
                    start = ptr + 1 + i;
                }
            }
			// res[row * res.shape.y + col] = (float)atof(start);
			col++;
            row++;
        }
        in.close();
	}
    return res;
}

Matrix readCsv_vec(std::string file, int rows) {
    ifstream in(file);
    std::string line;

    int row = 0;

	Matrix res(rows, 1);
	res.allocateMemory();

    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            res[row] = (float)atof(ptr);
            row++;
        }
        in.close();
    }
    return res;
}

Matrix readCsv_vec_last(std::string file) {
    ifstream in(file);
    std::string line;

    int row = 0;

	Matrix res(1, 1);
	res.allocateMemory();

    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
			res[row] = (float)atof(ptr);
			break;
        }
        in.close();
    }
    return res;
}

void normalize(Matrix &pnts, Matrix input_coeff) {
    for (size_t i = 0; i < pnts.shape.x; i++) {
        for (size_t j = 0; j < pnts.shape.y; j++) {
            pnts[i * pnts.shape.y + j] = pnts[i * pnts.shape.y + j]
                * input_coeff[j] + input_coeff[j + 3];
        }
    }
}

void unnormalize(Matrix &output, Matrix output_coeff) {
    for (size_t i = 0; i < output.shape.x; i++) {
        output[i] = (output[i] - output_coeff[4]) / output_coeff[0];
    }
}

void unnormalize_normal(Matrix &output, Matrix input_coeff) {
    for (size_t i = 0; i < output.shape.y; i++) {
        float sum = 0.0f;
        for (size_t j =0 ; j < output.shape.x; j++) {
            float a = output[j * output.shape.y + i] * input_coeff[j];
            output[j * output.shape.x + i] = a;
            sum += a * a;
        }
        for (size_t j =0 ; j < output.shape.x; j++) {
            output[j * output.shape.x + i] /= sqrt(sum);
        }
    }
}