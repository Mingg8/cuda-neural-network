#include "Eigen/Dense"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/tanh_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "file_io.hh"
#include "test_utils.hh"

using namespace Eigen;

class CollisionDetection {
  public:
    CollisionDetection(MatrixXd the_pnt_nut);

    void detectCollision(Matrix3d rot_nut, Vector3d trans_nut,
        MatrixXd *ptr_data, MatrixXd *ptr_normal, VectorXd *ptr_penet,
        MatrixXd *ptr_idx, int* ptr_num);

  private:
	NeuralNetwork nn;
    matrix::Matrix X;
    void setData(MatrixXd pnt);
};