//
// Created by alter on 17/10/2019.
//

#ifndef UNITY_LANS_GZ__ERROR_TERM_H_
#define UNITY_LANS_GZ__ERROR_TERM_H_

#include <sophus/se3.hpp>
#include <ceres/ceres.h>

class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const *T_raw, double const *delta_raw,
                    double *T_plus_delta_raw) const {
    Eigen::Map<Sophus::SE3d const> const T(T_raw);
    Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * Sophus::SE3d::exp(delta);
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const *T_raw,
                               double *jacobian_raw) const {
    Eigen::Map<Sophus::SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
        jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

  virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

struct OdometryFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OdometryFunctor(Sophus::SE3d t_ij) : t_ij(t_ij) {}

  template<typename T>
  bool operator()(const T *t_i_ptr, const T *t_j_ptr, T *residual_ptr) const {
    Eigen::Map<const Sophus::SE3<T>> t_i(t_i_ptr);
    Eigen::Map<const Sophus::SE3<T>> t_j(t_j_ptr);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);

    residual = (t_i * t_ij.template cast<T>() * t_j.inverse()).log();
    return true;
  }

  static ceres::CostFunction *Creat(const Sophus::SE3d t_ij) {
    return (new ceres::AutoDiffCostFunction<OdometryFunctor, 6, 7, 7>(
        new OdometryFunctor(t_ij)));
  }

  Sophus::SE3d t_ij;
};

struct LoopClosureFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LoopClosureFunctor(Sophus::SE3d t_mn) : t_mn(t_mn) {}

  template<typename T>
  bool operator()(const T *t_m_ptr,
                  const T *t_n_ptr,
                  const T *s_ptr,
                  T *residual_ptr) const {
    Eigen::Map<const Sophus::SE3<T>> t_m(t_m_ptr);
    Eigen::Map<const Sophus::SE3<T>> t_n(t_n_ptr);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);

    residual =
        *s_ptr * (t_m * t_mn.template cast<T>() * t_n.inverse()).log();
    return true;
  }

  static ceres::CostFunction *Creat(const Sophus::SE3d t_mn) {
    return (new ceres::AutoDiffCostFunction<LoopClosureFunctor, 6, 7, 7, 1>(
        new LoopClosureFunctor(t_mn)));
  }

  Sophus::SE3d t_mn;
};

struct PriorFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PriorFunctor(double gamma) : gamma(gamma) {}

  template<typename T>
  bool operator()(const T *s_ptr, T *residual_ptr) const {
    residual_ptr[0] = T(gamma) - *s_ptr;
    return true;
  }

  static ceres::CostFunction *Create(const double gamma) {
    return (new ceres::AutoDiffCostFunction<PriorFunctor,
                                            1,
                                            1>(new PriorFunctor(gamma)));
  }

  double gamma;
};

#endif //UNITY_LANS_GZ__ERROR_TERM_H_
