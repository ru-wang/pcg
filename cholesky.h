#pragma once

#include "linear_solver.h"

#include <Eigen/Eigen>

class cholesky_ldlt : linear_solver {
 public:
  cholesky_ldlt(const Eigen::MatrixXd& A,
                const Eigen::VectorXd& b)
      : A_(A), b_(b), dim_(b.rows()) {}

  virtual void solve(Eigen::VectorXd& x) override {
    for (size_t i = 0; i < dim_ - 1; ++i) {
      A_.block(i + 1, i, dim_ - i - 1, 1) /= A_(i, i);
      Eigen::MatrixXd l_21_d_l_21_T = A_.block(i + 1, i, dim_ - i - 1, 1) *
                                      A_.block(i, i + 1, 1, dim_ - i - 1);
      A_.block(i + 1, i + 1, dim_ - i - 1, dim_ - i - 1) -= l_21_d_l_21_T;
    }
    for (size_t i = 1; i < dim_; ++i)
      b_[i] -= (A_.block(i, 0, 1, i) * b_.block(0, 0, i, 1))[0];
    for (size_t i = 0; i < dim_; ++i)
      b_[i] /= A_(i, i);
    for (size_t i = dim_ - 2; i < dim_; --i)
      b_[i] -= (A_.block(i + 1, i, dim_ - i - 1, 1).transpose() *
                b_.block(i + 1, 0, dim_ - i - 1, 1))[0];
    x = b_;
  }

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  size_t dim_;
};
