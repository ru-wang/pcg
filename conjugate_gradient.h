#pragma once

#include "linear_solver.h"

#include <Eigen/Eigen>

class conjugate_gradient : linear_solver {
 public:
  conjugate_gradient(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b,
                     const double epsilon = 1e-12)
      : A_(A), b_(b), dim_(b.rows()), epsilon_(epsilon) {}

  virtual void solve(Eigen::VectorXd& x) override {
    Eigen::VectorXd r_k = b_ - A_ * x;
    Eigen::VectorXd p_k = r_k;
    for (size_t k = 0; k < dim_; ++k) {
      Eigen::VectorXd A_p_k = A_ * p_k;
      double p_k_T_A_p_k = p_k.transpose() * A_p_k;
      double r_k_T_r_k = r_k.transpose() * r_k;
      double alpha_k = r_k_T_r_k / p_k_T_A_p_k;
      x += alpha_k * p_k;
      r_k -= alpha_k * A_p_k;
      size_t i = 0;
      for (; i < dim_ && r_k[i] < epsilon_; ++i);
      if (i == dim_)
        break;
      double r_k_T_A_p_k = r_k.transpose() * A_p_k;
      double beta_k = r_k_T_A_p_k / p_k_T_A_p_k;
      p_k = r_k - beta_k * p_k;
    }
  }

 private:
  const Eigen::MatrixXd A_;
  const Eigen::VectorXd b_;
  size_t dim_;
  double epsilon_;
};
