#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <experimental/array>

#include <Eigen/Eigen>

#include "conjugate_gradient.h"

using namespace Eigen;

namespace {

const auto data = std::experimental::make_array(
  0.000000e+00, 1.133898e+00,
  7.500000e-02, 1.334902e+00,
  1.500000e-01, 1.213546e+00,
  2.250000e-01, 1.252016e+00,
  3.000000e-01, 1.392265e+00,
  3.750000e-01, 1.314458e+00,
  4.500000e-01, 1.472541e+00,
  5.250000e-01, 1.536218e+00,
  6.000000e-01, 1.355679e+00,
  6.750000e-01, 1.463566e+00,
  7.500000e-01, 1.490201e+00,
  8.250000e-01, 1.658699e+00,
  9.000000e-01, 1.067574e+00,
  9.750000e-01, 1.464629e+00,
  1.050000e+00, 1.402653e+00,
  1.125000e+00, 1.713141e+00,
  1.200000e+00, 1.527021e+00,
  1.275000e+00, 1.702632e+00,
  1.350000e+00, 1.423899e+00,
  1.425000e+00, 1.543078e+00,
  1.500000e+00, 1.664015e+00,
  1.575000e+00, 1.732484e+00,
  1.650000e+00, 1.543296e+00,
  1.725000e+00, 1.959523e+00,
  1.800000e+00, 1.685132e+00,
  1.875000e+00, 1.951791e+00,
  1.950000e+00, 2.095346e+00,
  2.025000e+00, 2.361460e+00,
  2.100000e+00, 2.169119e+00,
  2.175000e+00, 2.061745e+00,
  2.250000e+00, 2.178641e+00,
  2.325000e+00, 2.104346e+00,
  2.400000e+00, 2.584470e+00,
  2.475000e+00, 1.914158e+00,
  2.550000e+00, 2.368375e+00,
  2.625000e+00, 2.686125e+00,
  2.700000e+00, 2.712395e+00,
  2.775000e+00, 2.499511e+00,
  2.850000e+00, 2.558897e+00,
  2.925000e+00, 2.309154e+00,
  3.000000e+00, 2.869503e+00,
  3.075000e+00, 3.116645e+00,
  3.150000e+00, 3.094907e+00,
  3.225000e+00, 2.471759e+00,
  3.300000e+00, 3.017131e+00,
  3.375000e+00, 3.232381e+00,
  3.450000e+00, 2.944596e+00,
  3.525000e+00, 3.385343e+00,
  3.600000e+00, 3.199826e+00,
  3.675000e+00, 3.423039e+00,
  3.750000e+00, 3.621552e+00,
  3.825000e+00, 3.559255e+00,
  3.900000e+00, 3.530713e+00,
  3.975000e+00, 3.561766e+00,
  4.050000e+00, 3.544574e+00,
  4.125000e+00, 3.867945e+00,
  4.200000e+00, 4.049776e+00,
  4.275000e+00, 3.885601e+00,
  4.350000e+00, 4.110505e+00,
  4.425000e+00, 4.345320e+00,
  4.500000e+00, 4.161241e+00,
  4.575000e+00, 4.363407e+00,
  4.650000e+00, 4.161576e+00,
  4.725000e+00, 4.619728e+00,
  4.800000e+00, 4.737410e+00,
  4.875000e+00, 4.727863e+00,
  4.950000e+00, 4.669206e+00
);

constexpr unsigned num_observations = 67;

constexpr unsigned max_lm_iteratioins = 50;
constexpr double function_tolerance = 1e-12;
constexpr double function_change_tolerance = 1e-6;
constexpr double variable_change_tolerance = 1e-8;
constexpr double init_lambda = 1e-4;
constexpr double min_lambda = 1e-16;
constexpr double max_lambda = 1e32;

/*
 *          m      c
 * y = exp(0.3x + 0.1)
 *          ^      ^
 * |m|                              |m|
 * | | = argmin 1/2[ y - exp(|x  1|*| |) ]^2
 * |c|     x                        |c|
 */
std::tuple<MatrixXd, MatrixXd> exponential(
    const double x, const double y,
    const double m, const double c) {
  MatrixXd residual(1, 1);
  MatrixXd jacobian(1, 2);
  double exp = std::exp(m * x + c);
  residual << y - exp;
  jacobian << -exp * x, -exp;
  return std::make_tuple(residual, jacobian);
}

void compute(const double m, const double c,
             MatrixXd& residual, MatrixXd& jacobian) {
  for (unsigned i = 0; i < num_observations; ++i) {
    double x = data[i * 2], y = data[i * 2 + 1];
    MatrixXd local_f, local_J;
    std::tie(local_f, local_J) = exponential(x, y, m, c);
    residual.block<1, 1>(i, 0) = local_f;
    jacobian.block<1, 2>(i, 0) = local_J;
  }
}

}

int main() {
  double m = 0;
  double c = 0;
  MatrixXd J(num_observations, 2), f(num_observations, 1), g, H;

  auto t_begin = std::chrono::steady_clock::now();
  compute(m, c, f, J);
  double lambda = init_lambda;
  double f_old = f.squaredNorm() / 2;
  auto init_f = f_old;

  for (unsigned it = 1; it <= max_lm_iteratioins; ++it) {
    g = J.transpose() * f;
    H = J.transpose() * J;

    VectorXd diag = H.diagonal();
    H += (lambda * diag).asDiagonal();

    VectorXd dx = VectorXd::Zero(2);
    conjugate_gradient cg_solver(H, -g);
    unsigned pcg_steps = cg_solver.solve(dx);

    if (dx.norm() <= (std::sqrt(m * m + c * c) + variable_change_tolerance) * variable_change_tolerance)
      break;

    m += dx[0];
    c += dx[1];
    compute(m, c, f, J);
    double f_new = f.squaredNorm() / 2;

    if (f_new <= function_tolerance)
      break;

    std::cout << std::scientific << std::setprecision(6)
              << "\t[LM " << it << "]\t"
              << "<PCG=" << pcg_steps << "/" << dx.rows() << ">\t"
              << "f=" << f_old << "-->" << f_new
              << " f_change=" << f_new - f_old
              << " lambda=" << lambda
              << " determinant=" << H.determinant() << "\n";

    double f_change_ratio = std::abs(f_new - f_old) / f_old;
    if (f_change_ratio <= function_change_tolerance)
      break;

    if (f_new < f_old) {
      lambda *= 0.1;
      if (lambda < min_lambda)
        lambda = min_lambda;
      f_old = f_new;
    } else {
      lambda *= 10;
      if (lambda > max_lambda)
        lambda = max_lambda;
      m -= dx[0];
      c -= dx[1];
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(t_end - t_begin);

  std::cout << "time: " << duration.count() << "ms\n"
            << "m, c: { 0  0 }==>{ " << m << " " << c << " }\n"
            << "cost: " << init_f << " ==> " << f_old << std::endl;
}
