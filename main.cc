#include <assert.h>

#include <iostream>
#include <string>
#include <tuple>

#include <Eigen/Eigen>

#include "random_problem.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  assert(argc > 1);
  size_t dim = std::stoi(argv[1]);

  MatrixXd A;
  VectorXd x, b;
  std::tie(A, x, b) = random_spd(dim);
  std::cout << "x = " << x.transpose() << "\n"
            << "r = " << (A * x - b).transpose() << "\n";

  return 0;
}
