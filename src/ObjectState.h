#ifndef OBJECT_STATE_H_
#define OBJECT_STATE_H_

#include "Eigen/Dense"

struct objectState {
    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

    // state covariance matrix
    Eigen::MatrixXd P_;
};

#endif // OBJECT_STATE_H_