#include <assert.h>

#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include <Eigen/Eigen>

#include "conjugate_gradient.h"
#include "random_problem.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  assert(argc > 1);
  size_t dim = std::stoi(argv[1]);

  MatrixXd A;
  VectorXd x_true, x, b;

  std::cout << "generating random problem...";
  auto t_begin = std::chrono::steady_clock::now();
  std::tie(A, x_true, b) = random_spd(dim);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count() << "ms\n";

  x = x_true;
  x[0] += 1;
  x[1] += 1;
  x[2] += 1;
  std::cout << "computing using CG solver...";
  t_begin = std::chrono::steady_clock::now();
  conjugate_gradient cg_solver(A, b);
  cg_solver.solve(x);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count() << "ms | "
            << "error = " << (x - x_true).norm() << "\n";

  std::cout << "computing using column pivot Householder QR solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.colPivHouseholderQr().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count() << "ms | "
            << "error = " << (x - x_true).norm() << "\n";

  std::cout << "computing using full pivot Householder QR solver...";
  t_begin = std::chrono::steady_clock::now();
  x = A.colPivHouseholderQr().solve(b);
  std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_begin).count() << "ms | "
            << "error = " << (x - x_true).norm() << "\n";

  return 0;
}
