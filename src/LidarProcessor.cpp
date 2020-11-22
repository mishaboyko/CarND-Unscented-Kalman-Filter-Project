#include "LidarProcessor.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;

LidarProcessor::LidarProcessor() {}

LidarProcessor::~LidarProcessor() {}

void LidarProcessor::Init(int n_x_in, int n_aug_in, VectorXd weights_in) {
    n_x_ = n_x_in;

    n_aug_ = n_aug_in;

    weights_ = VectorXd(2*n_aug_+1);

    weights_ = weights_in;

    // mean predicted measurement
    z_pred = VectorXd(n_z_);

    // matrix for sigma points in measurement space
    Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

    // measurement covariance matrix S
    S = MatrixXd(n_z_,n_z_);
}

void LidarProcessor::PredictMeasurement(MatrixXd& Xsig_pred_){

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v_magnitude  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i); //orientation

    double v_x = cos(yaw)*v_magnitude;
    double v_y = sin(yaw)*v_magnitude;

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  S = S + R;

}

void LidarProcessor::Update(VectorXd& measurement, objectState* object_state, MatrixXd& Xsig_pred_) {
  // create matrix for cross correlation Tc
    // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

   // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - object_state->x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  K = Tc * S.inverse();

  // residual (a.k.a. z_diff)
  VectorXd residual = measurement - z_pred;

  // update state mean and covariance matrix
  object_state->x_ = object_state->x_ + K * residual;
  object_state->P_ = object_state->P_ - K*S*K.transpose();
  }
