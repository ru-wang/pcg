#include <assert.h>

#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include <Eigen/Eigen>

#include "cholesky.h"
#include "conjugate_gradient.h"
#include "preconditioned_conjugate_gradient.h"
#include "random_problem.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
//  assert(argc > 1);
//  size_t dim = std::stoi(argv[1]);
  size_t dim = 4;

  MatrixXd A;
  VectorXd x_true, x, b;

  A.resize(4, 4);
  A << 
   1.61e+02,  1.00e+01,  0.00e+00, -1.60e+02,
   1.00e+01,  1.04e+02, -8.00e+00,  0.00e+00,
   0.00e+00, -8.00e+00,  2.10e+01, -5.00e+00,
  -1.60e+02,  0.00e+00, -5.00e+00,  1.65e+02;

  b.resize(4, 1);
  b << -1.53e+02, 7.20e+01, 1.00e+00, 1.55e+02;


//  std::cout << "generating random problem...";
//  auto t_begin = std::chrono::steady_clock::now();
//  std::tie(A, x_true, b) = random_spd(dim);
//  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
//            << "ms\n";

  x = VectorXd::Zero(dim);
  std::cout << "CG solver...";
  auto t_begin = std::chrono::steady_clock::now();
  conjugate_gradient cg_solver(A, b);
  cg_solver.solve(x);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " // << "error = " << (x - x_true).norm() << "\n";
            << "x = " << x.transpose() << "\n";

  return 0;

  x = VectorXd::Zero(dim);
  std::cout << "PCG solver...";
  t_begin = std::chrono::steady_clock::now();
  preconditioned_conjugate_gradient pcg_solver(A, b);
  pcg_solver.solve(x);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "LDLT solver...";
  t_begin = std::chrono::steady_clock::now();
  cholesky_ldlt ldlt(A, b);
  ldlt.solve(x);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "partialPivLu solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.partialPivLu().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "fullPivLu solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.fullPivLu().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "householderQr solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.householderQr().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "colPivHouseholderQr solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.colPivHouseholderQr().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "fullPivHouseholderQr solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.fullPivHouseholderQr().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "llt solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.llt().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  x = VectorXd::Zero(dim);
  std::cout << "ldlt solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.ldlt().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count()
            << "ms | " << "error = " << (x - x_true).norm() << "\n";

  return 0;
}
