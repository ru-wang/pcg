#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

#include <Eigen/Eigen>

#include "conjugate_gradient.h"

using namespace Eigen;

int main() {
  const double sqrt5 = sqrt(5);
  const double sqrt10 = sqrt(10);

  Vector4d x = Vector4d(3, -1, 0, 1);
  auto& x1 = x[0];
  auto& x2 = x[1];
  auto& x3 = x[2];
  auto& x4 = x[3];

  auto init_X = x;

  Vector4d f, g;
  Matrix4d J, H;

  std::function<void()> residual = [=, &f, &x1, &x2, &x3, &x4]() {
    f <<             x1 + 10*x2,
                sqrt5*(x3 - x4),
              pow(x2 - 2*x3, 2),
         sqrt10*pow(x1 - x4, 2);
  };

  std::function<void()> jacobian = [=, &J, &x1, &x2, &x3, &x4] {
    J <<                  1,            10,              0,                   0,
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

  residual();
  jacobian();
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

  for (size_t it = 1; it <= 15; ++it) {
#ifndef NDEBUG
    std::cout << std::setw(4) << it;
#endif

    g = J.transpose() * f;
    H = J.transpose() * J;

    auto diag = H.diagonal().asDiagonal();
    H += lambda * diag;

    Matrix4d A = H;
    Vector4d b = -g;

#ifndef NDEBUG
    std::cout << std::setprecision(4)
              << "\n________________________________________\n"
              << A << "\n\n"
              << b.transpose()
              << "\n________________________________________\n";
#endif

    VectorXd dx = VectorXd::Zero(4);
    conjugate_gradient cg_solver(A, b);
    cg_solver.solve(dx);
    x += dx;

#ifndef NDEBUG
    std::cout << dx.transpose() << "\n"
              << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
#endif

    residual();
    jacobian();
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
            << "X: [ " << init_X.transpose() << " ] ==> [ " << x.transpose() << " ]\n"
            << "cost: " << init_f << " ==> " << f_old << std::endl;

  return 0;
}
