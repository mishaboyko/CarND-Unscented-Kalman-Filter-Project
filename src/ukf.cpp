#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // set state dimension
  n_x_ = 5;

  // set augmented dimension
  n_aug_ = 7;

  // define spreading parameter for augmented sigma points
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);
  // TODO:  Once the first sensor measurement arrives, you can initialize p_x and p_y.
  // TODO: For the other variables in the state vector x, you can try different initialization values to see what works best.
  /***
   * TODO: In the CTRV model, the velocity is from the object's perspective, which in this case is the bicycle;
   * TODO: the CTRV velocity is tangential to the circle along which the bicycle travels.
   * TODO: Therefore, you cannot directly use the radar velocity measurement to initialize the state vector.
   */
  //x_ <<  0, 0, 0, 0, 0;
  /*
  x_ <<  5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;
*/
  // initial covariance matrix
  // The identity matrix is a good place to start, since the non-diagonal values represent the covariances between variables, the P matrix is symmetrical.
  // One can also see that P starts to converge to small values relatively quickly.
  // TODO: Instead of setting each of the diagonal values to 1, you can try setting the diagonal values by how much difference you expect
  // TODO: between the true state and the initialized x state vector.
  // TODO: For example, in the project, we assume the standard deviation of the lidar x and y measurements is 0.15.
  // TODO: If we initialized p_x with a lidar measurement, the initial variance or uncertainty in p_x would probably be less than 1.
  P_ = MatrixXd(n_x_, n_x_);
  /*
  P_ <<    0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  */

  P_ <<   1,   0,   0,   0,   0,
          0,   1,   0,   0,   0,
          0,   0,   1,   0,   0,
          0,   0,   0,   1,   0,
          0,   0,   0,   0,   1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // declare augmented sigma-point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // set measurement dimension, radar can measure r, phi, and r_dot
  n_z_ = 3;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // matrix for sigma points in measurement space
  Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  S = MatrixXd(n_z_,n_z_);

  weights_ = VectorXd(2*n_aug_+1);

  /**
   * End DO NOT MODIFY section for measurement noise values
   */
  /*
  GenerateAugmentedSigmaPoints();
  PredictSigmaPoints();
  PredictMeanAndCovariance();
  PredictRadarMeasurement();

  // create example measurement package for incoming radar measurement
  MeasurementPackage mp;
  mp.sensor_type_ = MeasurementPackage::RADAR;
  mp.raw_measurements_ = VectorXd(3);
  mp.raw_measurements_ <<
     5.9214,   // rho in m
     0.2187,   // phi in rad
     2.0062;   // rho_dot in m/s
  UpdateRadar(mp);
  */
}

UKF::~UKF() {}

void UKF::SetWeights(){
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {
    weights_(i) = weight;
  }
}

void UKF::GenerateAugmentedSigmaPoints(){

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug_.col(i+1)  = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i+1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  std::cout << "Xsig_aug_ = " << std::endl << Xsig_aug_ << std::endl;

}

void UKF::PredictSigmaPoints(double delta_t) {

  // TODO: Calculate delta_t from comparing 2 measurements
  //double delta_t = 0.1; // time diff in sec

   for (int i = 0; i< 2 * n_aug_+1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance() {

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::PredictRadarMeasurement(){

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
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

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;

  // print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
   // Initialization

  if (!is_initialized_) {
    SetWeights();

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float x, y, v_x, v_y;
      
      x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);

      x_ << x, y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

  previous_timestamp_ = meas_package.timestamp_;

  // Prediction
  Prediction(delta_t);

  // Update

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    PredictRadarMeasurement();
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  GenerateAugmentedSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  /***
  * TODO: Copy from EKF. Change it to UKF.
  * z and y in cartesian coordinate system
  Eigen::VectorXd y = z - H_laser * x_; // where H_ * x_ == predicted z

  UpdateCommon(y, H_laser, R_laser);
  void KalmanFilter::UpdateCommon(Eigen::VectorXd y, Eigen::MatrixXd H, Eigen::MatrixXd R){

  Eigen::MatrixXd S = H * P_ * H.transpose() + R;
  // Calculate Kalman gain
  Eigen::MatrixXd K = (P_ * H.transpose()) * S.inverse();
  // new estimate
  x_ = x_ + (K * y);
  MatrixXd ident_matrix = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (ident_matrix - K * H) * P_;
  }
  */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

   // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // print result
  std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}