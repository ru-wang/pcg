#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/Eigen>

#include "conjugate_gradient.h"

using namespace Eigen;

namespace {

const double sqrt5 = sqrt(5);
const double sqrt10 = sqrt(10);

constexpr unsigned K = 1;

constexpr unsigned max_lm_iteratioins = 50;
constexpr double function_tolerance = 1e-12;
constexpr double function_change_tolerance = 1e-6;
constexpr double variable_change_tolerance = 1e-8;
constexpr double init_lambda = 1e-4;
constexpr double min_lambda = 1e-16;
constexpr double max_lambda = 1e32;

void residual(unsigned k, double x1, double x2, double x3, double x4, VectorXd& f) {
  f.block<4, 1>(k * 4, 0) <<             x1 + 10*x2,
                                    sqrt5*(x3 - x4),
                                  pow(x2 - 2*x3, 2),
                             sqrt10*pow(x1 - x4, 2);
};

void jacobian(unsigned k, double x1, double x2, double x3, double x4, MatrixXd& J) {
  J.block<4, 4>(k * 4, k * 4)
      <<                  1,            10,              0,                   0,
                          0,             0,          sqrt5,              -sqrt5,
                          0, 2*(x2 - 2*x3), -4*(x2 - 2*x3),                   0,
         2*sqrt10*(x1 - x4),             0,              0, -2*sqrt10*(x1 - x4);

}

}

int main() {
  VectorXd x(4 * K), f(4 * K), g(4 * K);
  MatrixXd J(4 * K, 4 * K), H(4 * K, 4 * K);
  for (unsigned k = 0; k < K; ++k)
    x.block<4, 1>(k * 4, 0) << 3, -1, 0, 1;

  auto init_x = x;

#ifndef NDEBUG_LM
  std::cout << std::scientific << std::setprecision(2)
            << std::setw(4) << "it"
            << std::setw(12) << "cost"
            << std::setw(12) << "cost_change"
            << std::setw(12) << "|gradient|"
            << std::setw(12) << "|step|"
            << std::setw(12) << "lambda"
            << std::setw(8) << "taken"
            << std::endl;
  std::cout << std::setw(72) << std::setfill('-') << "-" << std::setfill(' ') << std::endl;
#endif

  auto t_begin = std::chrono::steady_clock::now();

  for (unsigned k = 0; k < K; ++k) {
    residual(k, x[k*4+0], x[k*4+1], x[k*4+2], x[k*4+3], f);
    jacobian(k, x[k*4+0], x[k*4+1], x[k*4+2], x[k*4+3], J);
  }
  double lambda = init_lambda;
  double f_old = f.squaredNorm() / 2;
  auto init_f = f_old;

#ifndef NDEBUG_LM
  std::cout << std::setw(4) << 0
            << std::setw(12) << f_old
            << std::setw(12) << " "
            << std::setw(12) << " "
            << std::setw(12) << " "
            << std::setw(8) << " "
            << std::endl;
#endif

  for (unsigned it = 1; it <= max_lm_iteratioins; ++it) {
#ifndef NDEBUG_LM
    std::cout << std::setw(4) << it;
#endif
    g = J.transpose() * f;
    H = J.transpose() * J;

    VectorXd diag = H.diagonal();
    H += (lambda * diag).asDiagonal();

    VectorXd dx = VectorXd::Zero(4 * K);
    conjugate_gradient cg_solver(H, -g);
    unsigned pcg_steps = cg_solver.solve(dx);

    if (dx.norm() <= (x.norm() + variable_change_tolerance) * variable_change_tolerance)
      break;

    x += dx;
    for (unsigned k = 0; k < K; ++k) {
      residual(k, x[k*4+0], x[k*4+1], x[k*4+2], x[k*4+3], f);
      jacobian(k, x[k*4+0], x[k*4+1], x[k*4+2], x[k*4+3], J);
    }
    double f_new = f.squaredNorm() / 2;

    if (f_new <= function_tolerance) {
#ifndef NDEBUG_LM
      std::cout << std::endl;
#endif
      break;
    }

    std::cout << std::scientific << std::setprecision(2)
              << "\t[LM " << it << "]\t"
              << "<PCG=" << pcg_steps << "/" << dx.rows() << ">\t"
              << "f=" << f_old << "-->" << f_new
              << " f_change=" << f_new - f_old
              << " lambda=" << lambda
              << " determinant=" << H.determinant() << "\n";

    double f_change_ratio = std::abs(f_new - f_old) / f_old;

#ifndef NDEBUG_LM
    std::cout << std::setw(12) << f_new
              << std::setw(12) << f_change_ratio
              << std::setw(12) << g.norm() / 2
              << std::setw(12) << dx.norm()
              << std::setw(12) << lambda;
#endif

    if (f_change_ratio <= function_change_tolerance) {
#ifndef NDEBUG_LM
      std::cout << std::endl;
#endif
      break;
    }

    if (f_new < f_old) {
#ifndef NDEBUG_LM
      std::cout << std::setw(8) << "yes" << std::endl;
#endif
      lambda *= 0.1;
      if (lambda < min_lambda)
        lambda = min_lambda;
      f_old = f_new;
    } else {
#ifndef NDEBUG_LM
      std::cout << std::setw(8) << "no" << std::endl;
#endif
      lambda *= 10;
      if (lambda > max_lambda)
        lambda = max_lambda;
      x -= dx;
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(t_end - t_begin);

  std::cout << "time: " << duration.count() << "ms\n"
            << "   x: {" << init_x.transpose() << " }==>{" << x.transpose() << " }\n"
            << "cost: " << init_f << " ==> " << f_old << std::endl;

  return 0;
}
