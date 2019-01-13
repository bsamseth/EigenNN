#pragma once

#include <Eigen/Dense>

using Real = double;

using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Array  = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Real, 1, Eigen::Dynamic>;

using MatrixRef = Eigen::Ref<Matrix>;
using ArrayRef = Eigen::Ref<Array>;
using VectorRef = Eigen::Ref<Vector>;
using RowVectorRef = Eigen::Ref<RowVector>;
