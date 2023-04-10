//
// Created by jemin on 3/1/22.
//

#ifndef RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
#define RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_

#include <Eigen/Core>
#include <map>
using Dtype = float;
// Observations type
using Obtype = std::map<std::string, Eigen::VectorXd>;

using EigenVec = Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec = Eigen::Matrix<bool, -1, 1>;
using EigenMapVec = Eigen::Matrix<Obtype, -1, 1, Eigen::RowMajor>;
using EigenRowMajorMat = Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;

extern int threadCount;

#endif // RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
