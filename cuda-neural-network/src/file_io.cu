#include "file_io.hh"
#include <iostream>

using namespace std;

void loadWeight(std::vector<matrix::Matrix> &weight, std::vector<matrix::Matrix> &bias) {
	std::string dir = "/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64/weight_csv";
	int lnum = 2;
	int hdim = 64;

    int num = 0;

    std::string file = dir + std::string("/weight") + std::to_string(num) + std::string(".csv");
	num++;
    matrix::Matrix W = readCsv(file, 3, 64);

    file = dir + std::string("/weight") + to_string(num) + std::string(".csv");
    num++;
    matrix::Matrix b = readCsv_vec(file, hdim);

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

void loadNormalizationCoeff(matrix::Matrix& input_coeff, matrix::Matrix& output_coeff) {
	std::string dir = "/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64/weight_csv";
    string file_input = dir + string("/input_coeff.csv");
    string file_output = dir + string("/output_coeff.csv");
    input_coeff = readCsv_vec(file_input, 6);
    output_coeff = readCsv_vec(file_output, 8);
}


matrix::Matrix readCsv(std::string file, int rows, int cols) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;
    
	matrix::Matrix res(rows, cols);
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

matrix::Matrix readCsv_last(std::string file, int rows) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;
    
	matrix::Matrix res(rows, 1);
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

matrix::Matrix readCsv_vec(std::string file, int rows) {
    ifstream in(file);
    std::string line;

    int row = 0;

	matrix::Matrix res(rows, 1);
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

matrix::Matrix readCsv_vec_last(std::string file) {
    ifstream in(file);
    std::string line;

    int row = 0;

	matrix::Matrix res(1, 1);
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