#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
   VectorXd rmse(4);
   rmse << 0,0,0,0;

   // Validity checking of the input:
   //  * the estimation vector size should not be zero
   //  * the estimation vector size should equal ground truth vector size

   if (estimations.size() != ground_truth.size() || estimations.size() == 0 || ground_truth.size() == 0) {
      std::cout << "Invalid matrices of estimation or the ground truth data" << std::endl;
      exit(1);
   }

   // sum up all square residuals

   for(unsigned int i = 0; i < estimations.size(); ++i){
      VectorXd residual = estimations[i] - ground_truth[i];

      // coefficient-wise multiplication
      residual = residual.array()*residual.array();

      rmse += residual;
   }

   // calculate the mean
   rmse = rmse / estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   return rmse;

}

VectorXd Tools::CartesianToPolar(const VectorXd& x_state) {
   // Range (rho). Radial distance from origin
   float ro;
   // Bearing. Angle between rho and x-axis
   float theta;
   // radial velocity (moving towards or away from the sensor). Change of rho (range rate)
   float ro_dot;

   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   ro = sqrt(px*px+py*py);
   theta = atan2(py, px);
   ro_dot = (px*vx + py*vy)/ro;

   Eigen::VectorXd polar_space = VectorXd(3);
   polar_space << ro, theta, ro_dot;

   return polar_space;
}