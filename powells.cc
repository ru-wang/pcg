#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

#include <Eigen/Eigen>

#include "conjugate_gradient.h"

using namespace Eigen;

int main() {
  constexpr unsigned K = 100;

  const double sqrt5 = sqrt(5);
  const double sqrt10 = sqrt(10);

  VectorXd x, f, g;
  MatrixXd J, H;
  x.resize(4 * K, 1);
  f.resize(4 * K, 1);
  g.resize(4 * K, 1);
  J.resize(4 * K, 4 * K);
  H.resize(4 * K, 4 * K);
  for (unsigned k = 0; k < K; ++k)
    x.block<4, 1>(k * 4, 0) = Vector4d(3, -1, 0, 1);

  auto init_x = x;

  std::function<void(unsigned, double, double, double, double, VectorXd&)>
  residual = [sqrt5, sqrt10](unsigned k,
                             double x1,
                             double x2,
                             double x3,
                             double x4,
                             VectorXd& f) {
    f.block<4, 1>(k * 4, 0) <<             x1 + 10*x2,
                                      sqrt5*(x3 - x4),
                                    pow(x2 - 2*x3, 2),
                               sqrt10*pow(x1 - x4, 2);
  };

  std::function<void(unsigned, double, double, double, double, MatrixXd&)>
  jacobian = [sqrt5, sqrt10](unsigned k,
                             double x1,
                             double x2,
                             double x3,
                             double x4,
                             MatrixXd& J) {
  J.block<4, 4>(k * 4, k * 4)
      <<                  1,            10,              0,                   0,
                          0,             0,          sqrt5,              -sqrt5,
                          0, 2*(x2 - 2*x3), -4*(x2 - 2*x3),                   0,
         2*sqrt10*(x1 - x4),             0,              0, -2*sqrt10*(x1 - x4);

//    std::cout
//      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
//      << Vector4d(x1, x2, x3, x4).transpose() << "\n\n"
//      << J.transpose() << "\n"
//      << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

  };

#ifndef NDEBUG
  std::cout << std::scientific << std::setprecision(2)
            << std::setw(4) << "it"
            << std::setw(12) << "cost"
            << std::setw(12) << "|gradient|"
            << std::setw(12) << "|step|"
            << std::setw(12) << "lambda"
            << std::setw(8) << "taken"
            << std::endl;
  std::cout << std::setw(60) << std::setfill('-') << "-" << std::setfill(' ') << std::endl;
#endif

  auto t_begin = std::chrono::steady_clock::now();

  for (unsigned k = 0; k < K; ++k) {
    residual(k, x[k * 4 + 0],
                x[k * 4 + 1],
                x[k * 4 + 2],
                x[k * 4 + 3], f);
    jacobian(k, x[k * 4 + 0],
                x[k * 4 + 1],
                x[k * 4 + 2],
                x[k * 4 + 3], J);
  }
  double lambda = 1e-4;
  double f_old = f.squaredNorm() / 2;
  auto init_f = f_old;

#ifndef NDEBUG
  std::cout << std::setw(4) << 0
            << std::setw(12) << f_old
            << std::setw(12) << " "
            << std::setw(12) << " "
            << std::setw(12) << " "
            << std::setw(8) << " "
            << std::endl;
#endif

  for (unsigned it = 1; it <= 15; ++it) {
#ifndef NDEBUG
    std::cout << std::setw(4) << it;
#endif

    g = J.transpose() * f;
    H = J.transpose() * J;

    auto diag = H.diagonal().asDiagonal();
    H += lambda * diag;

    auto A = H;
    auto b = -g;

#ifndef NDEBUG
    std::cout << std::setprecision(4)
              << "\n________________________________________\n"
              << A << "\n\n"
              << b.transpose()
              << "\n________________________________________\n";
#endif

    VectorXd dx = VectorXd::Zero(4 * K);
    conjugate_gradient cg_solver(A, b);
    cg_solver.solve(dx);
//    dx = A.ldlt().solve(b);
    x += dx;

#ifndef NDEBUG
    std::cout << dx.transpose() << "\n"
              << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
#endif

    for (unsigned k = 0; k < K; ++k) {
      residual(k, x[k * 4 + 0],
                  x[k * 4 + 1],
                  x[k * 4 + 2],
                  x[k * 4 + 3], f);
      jacobian(k, x[k * 4 + 0],
                  x[k * 4 + 1],
                  x[k * 4 + 2],
                  x[k * 4 + 3], J);
    }
    double f_new = f.squaredNorm() / 2;

#ifndef NDEBUG
    std::cout << std::setw(12) << f_new
              << std::setw(12) << g.norm() / 2
              << std::setw(12) << dx.norm()
              << std::setw(12) << lambda;
#endif

    if (f_new < f_old) {
#ifndef NDEBUG
      std::cout << std::setw(8) << "yes" << std::endl;
#endif
      lambda /= 10;
      f_old = f_new;
    } else {
#ifndef NDEBUG
      std::cout << std::setw(8) << "no" << std::endl;
#endif
      lambda *= 10;
      x -= dx;
    }
  }
  auto t_end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(t_end - t_begin);

  std::cout << "time: " << duration.count() << "ms\n"
            << "cost: " << init_f << " ==> " << f_old << std::endl;

  return 0;
}
