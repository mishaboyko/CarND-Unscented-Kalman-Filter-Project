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
  object_state.x_ = VectorXd(n_x_);

  // initial covariance matrix
  // The identity matrix is a good place to start, since the non-diagonal values represent the covariances between variables, the P matrix is symmetrical.
  // One can also see that P starts to converge to small values relatively quickly.
  object_state.P_ = MatrixXd(n_x_, n_x_);

  object_state.P_ <<    1,   0,   0,   0,   0,
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

  weights_ = VectorXd(2*n_aug_+1);

  SetWeights();
  radar_processor.Init(n_x_, n_aug_, weights_);
  lidar_processor.Init(n_x_, n_aug_, weights_);

  //lidar_processor(n_x_, n_aug_, weights_);
}

UKF::~UKF() {}

void UKF::SetWeights(){
  weights_(0) = lambda_/(lambda_+n_aug_);;

  for (int i=1; i<2*n_aug_+1; ++i) {
    weights_(i) = 0.5/(lambda_+n_aug_);
  }
}

void UKF::GenerateAugmentedSigmaPoints(){

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create augmented mean state
  x_aug.head(5) = object_state.x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = object_state.P_;
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
}

void UKF::PredictSigmaPoints(double delta_t) {

   for (int i = 0; i< 2 * n_aug_+1; ++i) {
    // extract diagonal values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    // predict state values (deterministic part)
    double px_p, py_p;

    // handle division by zero when the yaw rate is 0.
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*cos(yaw)*delta_t;
        py_p = p_y + v*sin(yaw)*delta_t;
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add process noise vector (stochastic part)
    px_p = px_p + 0.5*delta_t*delta_t * cos(yaw)*nu_a;  // adding the x acceleration offset if the car were driving perfectly straight
    py_p = py_p + 0.5*delta_t*delta_t * sin(yaw)*nu_a;  // adding the y acceleration offset if the car were driving perfectly straight
    v_p = v_p + delta_t*nu_a;                           // inflence of acceleration on the velocity

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;       // inflence of the change rate of the yaw rate on the yaw angle
    yawd_p = yawd_p + nu_yawdd*delta_t;                 // influence yaw acceleration on yaw rate

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  // predicted state mean
  object_state.x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    object_state.x_ = object_state.x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  object_state.P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - object_state.x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    object_state.P_ = object_state.P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
   // Initialization

  if (!is_initialized_) {

  // TODO:  Once the first sensor measurement arrives, you can initialize p_x and p_y.
  // TODO: For the other variables in the state vector x, you can try different initialization values to see what works best.
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float x, y;

      /***
       * In the CTRV model, the velocity is from the object's perspective, which in this case is the bicycle;
       * the CTRV velocity is tangential to the circle along which the bicycle travels.
       * Therefore, convert radar velocity measurement to initialize the state vector (polar to cartesian coord spaces).
       */
      x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);

      object_state.x_ << x, y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      object_state.x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
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
    radar_processor.PredictMeasurement(Xsig_pred_);
    // Radar updates
    radar_processor.Update(meas_package.raw_measurements_, &object_state, Xsig_pred_);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    lidar_processor.PredictMeasurement(Xsig_pred_);
    // Laser updates
    lidar_processor.Update(meas_package.raw_measurements_, &object_state, Xsig_pred_);
  }

}

void UKF::Prediction(double delta_t) {
  GenerateAugmentedSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}
