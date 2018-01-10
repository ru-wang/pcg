#pragma once

#include "linear_solver.h"

#include <iostream>

#include <Eigen/Eigen>

class conjugate_gradient : linear_solver {
 public:
  conjugate_gradient(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b)
      : A_(A), b_(b), dim_(b.rows()), epsilon_(1e-24) {}

  virtual void solve(Eigen::VectorXd& x) override {
    Eigen::VectorXd r_k = A_ * x - b_;
    Eigen::VectorXd p_k = -r_k;
    double r_k_T_r_k = r_k.squaredNorm();

    size_t k;
    for (k = 0; k < dim_; ++k) {
      if (r_k_T_r_k < epsilon_)
        break;

      Eigen::VectorXd A_p_k = A_ * p_k;
      double p_k_T_A_p_k = p_k.transpose() * A_p_k;
      double alpha_k = r_k_T_r_k / p_k_T_A_p_k;

      x += alpha_k * p_k;
      r_k += alpha_k * A_p_k;

      double r_k_1_T_r_k_1 = r_k.squaredNorm();
      double beta_k = r_k_1_T_r_k_1 / r_k_T_r_k;
      p_k = -r_k + beta_k * p_k;

      r_k_T_r_k = r_k_1_T_r_k_1;
    }
    std::cout << "[iters=" << k << "]";
  }

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  size_t dim_;
  double epsilon_;
};
