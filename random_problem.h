#include <tuple>

#include <Eigen/Eigen>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
random_spd(size_t dim) {
  using namespace Eigen;
  MatrixXd A = MatrixXd::Random(dim, dim);
  A = A * A.transpose() + MatrixXd::Identity(dim, dim);
  VectorXd x = VectorXd::Random(dim);
  return std::make_tuple(A, x, A * x);
}
