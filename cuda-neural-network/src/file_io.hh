#include "nn_utils/matrix.hh"
#include <string>
#include <fstream>
#include <vector>

using namespace std;

void loadWeight(std::vector<Matrix> &weight, std::vector<Matrix> &bias);
void loadNormalizationCoeff(Matrix& input_coeff, Matrix& output_coeff);
Matrix readCsv(std::string file, int rows, int cols);
Matrix readCsv_vec(std::string file, int rows);
Matrix readCsv_last(std::string file, int rows);
Matrix readCsv_vec_last(std::string file);
void normalize(Matrix &pnts, Matrix input_coeff);
void unnormalize(Matrix &output, Matrix output_coeff);