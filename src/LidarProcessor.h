#ifndef lidar_Processor_H
#define lidar_Processor_H

#include "Eigen/Dense"
#include "ObjectState.h"

class LidarProcessor {
    public:
        /**
         * Constructor
         */
        LidarProcessor();

        /**
         * Destructor
         */
        virtual ~LidarProcessor();

        void Init(int n_x_in, int n_aug_in, Eigen::VectorXd weights_in);

        void PredictMeasurement(Eigen::MatrixXd& Xsig_pred_);

            /**
            * Updates the state and the state covariance matrix using a laser measurement
            * @param meas_package The measurement at k+1
            */
        void Update(Eigen::VectorXd& measurement, objectState* object_state, Eigen::MatrixXd& Xsig_pred_);


    private:

        /**
         * DO NOT MODIFY measurement noise values below.
         * These are provided by the sensor manufacturer.
         */

        // Laser measurement noise standard deviation position1 in m
        const double std_laspx_ = 0.15;

        // Laser measurement noise standard deviation position2 in m
        const double std_laspy_ = 0.15;

        /**
         * End DO NOT MODIFY section for measurement noise values
         */
        // set measurement dimension, lidar can measure p_x, p_y
        const int n_z_ = 2;
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
#endif  // lidar_Processor_H