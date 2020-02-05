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
            res[row * res.shape.x + col] = (float)atof(start);
            row++;
            col++;
        }
        in.close();
	}
	
	// W.copyHostToDevice();
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