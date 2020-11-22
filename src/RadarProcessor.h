#ifndef Radar_Processor_H
#define Radar_Processor_H

#include "Eigen/Dense"
#include "ObjectState.h"

class RadarProcessor {
    public:
        /**
         * Constructor
         */
        RadarProcessor();

        /**
         * Destructor
         */
        virtual ~RadarProcessor();

        void Init(int n_x_in, int n_aug_in, Eigen::VectorXd weights_in);

        void PredictMeasurement(Eigen::MatrixXd& Xsig_pred_);
            /**
            * Updates the state and the state covariance matrix using a radar measurement
            * @param meas_package The measurement at k+1
            */
        void Update(Eigen::VectorXd& measurement, objectState* object_state, Eigen::MatrixXd& Xsig_pred_);

    private:

        /**
         * DO NOT MODIFY measurement noise values below.
         * These are provided by the sensor manufacturer.
         */

        // Radar measurement noise standard deviation radius in m
        const double std_radr_ = 0.3;

        // Radar measurement noise standard deviation angle in rad
        const double std_radphi_ = 0.03;

        // Radar measurement noise standard deviation radius change in m/s
        const double std_radrd_ = 0.3;
        /**
         * End DO NOT MODIFY section for measurement noise values
         */

        const int n_z_ = 3;
        int n_x_;
        int n_aug_;

        // create matrix for sigma points in measurement space
        Eigen::MatrixXd Zsig;

        // mean predicted measurement
        Eigen::VectorXd z_pred;

        // Weights of sigma points
        Eigen::VectorXd weights_;

        // measurement covariance matrix S
        Eigen::MatrixXd S;

        // Kalman gain K;
        Eigen::MatrixXd K;
};
#endif  // Radar_Processor_H