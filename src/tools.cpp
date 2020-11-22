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
