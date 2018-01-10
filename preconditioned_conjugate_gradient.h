#pragma once

#include "linear_solver.h"

#include <iostream>

#include <Eigen/Eigen>

class preconditioned_conjugate_gradient : linear_solver {
 public:
  preconditioned_conjugate_gradient(
      const Eigen::MatrixXd& A,
      const Eigen::VectorXd& b)
      : A_(A), b_(b), dim_(b.rows()), epsilon_(1e-30) {}

  virtual void solve(Eigen::VectorXd& x) override {
    compute_jacobi_preconditioner();

    Eigen::VectorXd r_k = A_ * x - b_;
    Eigen::VectorXd y_k = M_inv_ * r_k;
    Eigen::VectorXd p_k = -y_k;
    double r_k_T_y_k = r_k.dot(y_k);

    size_t k;
    for (k = 0; k < dim_; ++k) {
      if (r_k_T_y_k < epsilon_)
        break;

      Eigen::VectorXd A_p_k = A_ * p_k;
      double p_k_T_A_p_k = p_k.transpose() * A_p_k;
      double alpha_k = r_k_T_y_k / p_k_T_A_p_k;

      x += alpha_k * p_k;
      r_k += alpha_k * A_p_k;

      y_k = M_inv_ * r_k;

      double r_k_1_T_y_k_1 = r_k.dot(y_k);
      double beta_k = r_k_1_T_y_k_1 / r_k_T_y_k;
      p_k = -y_k + beta_k * p_k;

      r_k_T_y_k = r_k_1_T_y_k_1;
    }
    std::cout << "[iters=" << k << "]";
  }

 private:
  void compute_jacobi_preconditioner() {
    Eigen::VectorXd P = A_.diagonal();
    for (size_t i = 0; i < dim_; ++i)
      P[i] = P[i] * P[i];
    M_inv_ = P.asDiagonal().inverse();
  }

  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> M_inv_;
  size_t dim_;
  double epsilon_;
};
