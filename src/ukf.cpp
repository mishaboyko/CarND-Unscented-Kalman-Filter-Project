#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  std::cout << "UKF Constuctor called" << std::endl;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // set state dimension
  n_x_ = 5;

  // set augmented dimension
  n_aug_ = 7;

  // define spreading parameter
  lambda_ = 3 - n_x_;

  // initial state vector
  x_ = VectorXd(n_x_);
  // TODO:  Once the first sensor measurement arrives, you can initialize p_x and p_y.
  // TODO: For the other variables in the state vector x, you can try different initialization values to see what works best.
  /***
   * TODO: In the CTRV model, the velocity is from the object's perspective, which in this case is the bicycle;
   * TODO: the CTRV velocity is tangential to the circle along which the bicycle travels.
   * TODO: Therefore, you cannot directly use the radar velocity measurement to initialize the state vector.
   */
  /*
  x_ <<  0, 0, 0, 0, 0;
  */
  x_ <<  5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  // initial covariance matrix
  // The identity matrix is a good place to start, since the non-diagonal values represent the covariances between variables, the P matrix is symmetrical.
  // One can also see that P starts to converge to small values relatively quickly.
  // TODO: Instead of setting each of the diagonal values to 1, you can try setting the diagonal values by how much difference you expect
  // TODO: between the true state and the initialized x state vector.
  // TODO: For example, in the project, we assume the standard deviation of the lidar x and y measurements is 0.15.
  // TODO: If we initialized p_x with a lidar measurement, the initial variance or uncertainty in p_x would probably be less than 1.
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<    0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  /*
  P_ <<   1,   0,   0,   0,   0,
          0,   1,   0,   0,   0,
          0,   0,   1,   0,   0,
          0,   0,   0,   1,   0,
          0,   0,   0,   0,   1;
  */
  // declare sigma point matrix
  Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // declare augmented sigma-point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

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

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  GenerateSigmaPoints();
  AugmentedSigmaPoints();
  PredictSigmaPoints();
  PredictMeanAndCovariance();
}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints() {

  // calculate square root of P
  Eigen::MatrixXd A = P_.llt().matrixL();

  // set first column of sigma point matrix
  Xsig_.col(0) = x_;

  // set remaining sigma points
  for (int i = 0; i < n_x_; ++i) {
    Xsig_.col(i+1)     = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig_.col(i+1+n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }
  // print result
  std::cout << "Xsig_ = " << std::endl << Xsig_ << std::endl;

}

void UKF::AugmentedSigmaPoints(){
  // define spreading parameter
  double lambda = 3 - n_aug_;

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // TODO: Optimize it by re-using GenerateSigmaPoints()
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug_.col(i+1)       = x_aug + sqrt(lambda+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda+n_aug_) * L.col(i);
  }

  std::cout << "Xsig_aug_ = " << std::endl << Xsig_aug_ << std::endl;

}

void UKF::PredictSigmaPoints() {
  double delta_t = 0.1; // time diff in sec

   for (int i = 0; i< 2*n_aug_+1; ++i) {
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
  // define spreading parameter
  double lambda = 3 - n_aug_;

  // create vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);

  // set weights
  double weight_0 = lambda/(lambda+n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; ++i) {  // 2n+1 weights
    double weight = 0.5/(n_aug_+lambda);
    weights(i) = weight;
  }

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_ = x_ + weights(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights(i) * x_diff * x_diff.transpose();
  }

  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}