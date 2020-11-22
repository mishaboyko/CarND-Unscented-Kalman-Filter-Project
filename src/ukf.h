#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include "ObjectState.h"
#include "RadarProcessor.h"
#include "LidarProcessor.h"

class UKF {
  public:
    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    objectState object_state;
  private:
    void GenerateAugmentedSigmaPoints();

    void SetWeights();

    void PredictSigmaPoints(double delta_t);

    void PredictMeanAndCovariance();

    RadarProcessor radar_processor;
    LidarProcessor lidar_processor;

    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // predicted sigma points matrix
    Eigen::MatrixXd Xsig_pred_;

    // augmented sigma-point matrix
    Eigen::MatrixXd Xsig_aug_;

    // Weights of sigma points
    Eigen::VectorXd weights_;

    // time when the state is true, in us
    long long time_us_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    // Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    // Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    // Radar measurement dimention

    // State dimension
    int n_x_;

    // Augmented state dimension
    int n_aug_;

    // Sigma point spreading parameter
    double lambda_;

    long long previous_timestamp_;
};

#endif  // UKF_H