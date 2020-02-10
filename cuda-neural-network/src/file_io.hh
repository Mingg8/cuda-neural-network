#include "nn_utils/matrix.hh"
#include <string>
#include <fstream>
#include <vector>
#include <math.h>

using namespace std;

void loadWeight(std::vector<matrix::Matrix> &weight, std::vector<matrix::Matrix> &bias);
void loadNormalizationCoeff(matrix::Matrix& input_coeff, matrix::Matrix& output_coeff);
matrix::Matrix readCsv(std::string file, int rows, int cols);
matrix::Matrix readCsv_vec(std::string file, int rows);
matrix::Matrix readCsv_last(std::string file, int rows);
matrix::Matrix readCsv_vec_last(std::string file);
void normalize(matrix::Matrix &pnts, matrix::Matrix input_coeff);
void unnormalize(matrix::Matrix &output, matrix::Matrix output_coeff);
void unnormalize_normal(matrix::Matrix &output, matrix::Matrix input_coeff);