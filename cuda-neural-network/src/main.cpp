#include <iostream>
#include <time.h>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

#include "collision_detection.hpp"

using namespace Eigen;

int main() {
	int buffer = 2000;
	MatrixXd pnt_nut(4900, 3);
	pnt_nut.setOnes();
	pnt_nut(0, 0) = 0.0165;
	pnt_nut(0, 1) = 0.0165;
	pnt_nut(0, 2) = 0.0371;

	CollisionDetection *cd = new CollisionDetection(pnt_nut);

	Vector3d trans_nut(0, 0, 0);
	Matrix3d rot_nut;
	rot_nut.setIdentity();

	auto start = chrono::steady_clock::now();
	MatrixXd ptr_data, ptr_normal, ptr_idx;
	VectorXd ptr_penet;
	int ptr_num;
	cd->detectCollision(rot_nut, trans_nut, &ptr_data, &ptr_normal, &ptr_penet,
		&ptr_idx, &ptr_num);
	auto end = chrono::steady_clock::now();

	cout << "Elapsed time (microseconds) : "
		<< "Total: "
		<< chrono::duration_cast<chrono::microseconds>(end - start).count()
		<< " (us) " << endl;

	return 0;
}