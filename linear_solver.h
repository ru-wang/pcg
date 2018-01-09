#pragma once

#include <Eigen/Eigen>

class linear_solver {
 public:
  virtual void solve(Eigen::VectorXd& x) = 0;
};
