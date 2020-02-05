#include "nn_utils/matrix.hh"
#include <string>
#include <fstream>
#include <vector>

using namespace std;

void loadWeight(std::vector<Matrix> &weight, std::vector<Matrix> &bias);
Matrix readCsv(std::string file, int rows, int cols);
Matrix readCsv_vec(std::string file, int rows);
Matrix readCsv_last(std::string file, int rows);
Matrix readCsv_vec_last(std::string file);
