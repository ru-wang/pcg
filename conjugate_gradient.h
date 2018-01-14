#pragma once

#include "linear_solver.h"

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include <Eigen/Eigen>

class conjugate_gradient : linear_solver {
 public:
  conjugate_gradient(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b)
      : A_(A), b_(b), dim_(b.rows()), epsilon_(1e-24) {}

  virtual void solve(Eigen::VectorXd& x) override {
    Eigen::VectorXd r_k = A_ * x - b_;
    Eigen::VectorXd p_k = -r_k;
    double r_0_T_r_0 = r_k.squaredNorm();
    double r_k_T_r_k = r_0_T_r_0;

#ifndef NDEBUG
    std::cout << std::scientific << std::setprecision(4);
#endif

    size_t k;
    for (k = 0; k < dim_; ++k) {
      Eigen::VectorXd A_p_k = A_ * p_k;
      double p_k_T_A_p_k = p_k.transpose() * A_p_k;
      double alpha_k = r_k_T_r_k / p_k_T_A_p_k;

#ifndef NDEBUG
      std::cout << "[PCG] " << k << ":"
                << "\np_k: {" << p_k.transpose() << " }";
#endif

      x += alpha_k * p_k;
      r_k += alpha_k * A_p_k;

      double r_k_1_T_r_k_1 = r_k.squaredNorm();
      double beta_k = r_k_1_T_r_k_1 / r_k_T_r_k;

      if (r_k_1_T_r_k_1 / r_0_T_r_0 < 1e-7)
        break;

      p_k = -r_k + beta_k * p_k;

#ifndef NDEBUG
      std::cout << "\nA_p_k: {" << A_p_k.transpose() << " }"
                << "\nr_k_T_r_k    : " << r_k_T_r_k
                << "\nr_k_1_T_r_k_1: " << r_k_1_T_r_k_1
                << "\nalpha_k      : " << alpha_k
                << "\nbeta_k       : " << beta_k << "\n {"
                << x.transpose() << "}\n\n";
#endif

      r_k_T_r_k = r_k_1_T_r_k_1;
    }
  }

 private:
  const Eigen::MatrixXd A_;
  const Eigen::VectorXd b_;
  size_t dim_;
  double epsilon_;
};
