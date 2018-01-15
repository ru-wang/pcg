#pragma once

#include <Eigen/Eigen>

class linear_solver {
 public:
  virtual unsigned solve(Eigen::VectorXd& x) = 0;
};
