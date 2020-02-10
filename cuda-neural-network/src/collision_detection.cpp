#include "collision_detection.hpp"

CollisionDetection::CollisionDetection(MatrixXd the_pnt_nut) {
	srand( time(NULL) );
	LinearLayer* linear_layer_1 = new LinearLayer("linear_layer_1", Shape(3, 64));
	ReLUActivation* relu_layer_1 = new ReLUActivation("relu_layer_1");
	LinearLayer* linear_layer_2 = new LinearLayer("linear_layer_2", Shape(64, 64));
	ReLUActivation* relu_layer_2 = new ReLUActivation("relu_layer_2");
	LinearLayer* linear_layer_3 = new LinearLayer("linear_layer_3", Shape(64, 64));
	ReLUActivation* relu_layer_3 = new ReLUActivation("relu_layer_3");
	LinearLayer* linear_layer_4 = new LinearLayer("linear_layer_4", Shape(64, 1));
	TanhActivation* tanh_layer = new TanhActivation("tanh_layer");

	std::vector<matrix::Matrix> weights, biases;
	matrix::Matrix input_coeff, output_coeff;
	loadWeight(weights, biases);
	loadNormalizationCoeff(input_coeff, output_coeff);
	input_coeff.copyHostToDevice();
	output_coeff.copyHostToDevice();
	nn.setCoeffs(input_coeff, output_coeff);

	testutils::initializeTensorWithMatrix(linear_layer_1->W, weights[0]);
	testutils::initializeTensorWithMatrix(linear_layer_2->W, weights[1]);
	testutils::initializeTensorWithMatrix(linear_layer_3->W, weights[2]);
	testutils::initializeTensorWithMatrix(linear_layer_4->W, weights[3]);

	testutils::initializeTensorWithMatrix(linear_layer_1->b, biases[0]);
	testutils::initializeTensorWithMatrix(linear_layer_2->b, biases[1]);
	testutils::initializeTensorWithMatrix(linear_layer_3->b, biases[2]);
	testutils::initializeTensorWithMatrix(linear_layer_4->b, biases[3]);

	linear_layer_1->W.copyHostToDevice();
	linear_layer_2->W.copyHostToDevice();
	linear_layer_3->W.copyHostToDevice();
	linear_layer_4->W.copyHostToDevice();

	linear_layer_1->b.copyHostToDevice();
	linear_layer_2->b.copyHostToDevice();
	linear_layer_3->b.copyHostToDevice();
	linear_layer_4->b.copyHostToDevice();

	nn.addLayer(linear_layer_1);
	nn.addLayer(relu_layer_1);
	nn.addLayer(linear_layer_2);
	nn.addLayer(relu_layer_2);
	nn.addLayer(linear_layer_3);
	nn.addLayer(relu_layer_3);
	nn.addLayer(linear_layer_4);
	nn.addLayer(tanh_layer);

    X.shape.x = the_pnt_nut.rows();
    X.shape.y = the_pnt_nut.cols();
	X.allocateMemory(); 
    testutils::setData(X, the_pnt_nut);

    X.copyHostToDevice();
}

void CollisionDetection::detectCollision(Matrix3d rot_nut, Vector3d trans_nut,
    MatrixXd *ptr_data, MatrixXd *ptr_normal, VectorXd *ptr_penet,
    MatrixXd *ptr_idx, int* ptr_num) {
    matrix::Matrix trans(3, 1);
    matrix::Matrix rot(3, 3);
    trans.allocateMemory();
    rot.allocateMemory();
    
    for (size_t i = 0; i < 3; i++) {
        trans[i] = (float)trans_nut(i);
        for (size_t j = 0; j < 3; j++) {
            rot[3 * i + j] = (float)rot_nut(i, j);
        }
    }

    matrix::Matrix Y, N;
	nn.forward(X, Y, N, rot, trans);
	Y.copyDeviceToHost();
	N.copyDeviceToHost();

    *ptr_data = testutils::Matrix2Eigen(Y);
    *ptr_normal = testutils::Matrix2Eigen(N);
}