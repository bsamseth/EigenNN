#pragma once

#include <Eigen/Dense>

using Real = double;

using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Array  = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
