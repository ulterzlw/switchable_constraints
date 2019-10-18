#include <vector>
#include <fstream>
#include <Eigen/Core>
#include "error_term.h"

Sophus::SE3d ReadVertex(std::ifstream *fin) {
  double x, y, z, qx, qy, qz, qw;
  *fin >> x >> y >> z >> qx >> qy >> qz >> qw;
  Sophus::SE3d
      pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z));
  return pose;
}

Sophus::SE3d ReadEdge(std::ifstream *fin) {
  double x, y, z, qx, qy, qz, qw;
  *fin >> x >> y >> z >> qx >> qy >> qz >> qw;
  Sophus::SE3d
      pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z));

  double information;
  for (int i = 0; i < 21; i++)
    *fin >> information;
  return pose;
}

int main(int argc, char **argv) {
  typedef Eigen::aligned_allocator<Sophus::SE3d> sophus_allocator;
  std::vector<Sophus::SE3d, sophus_allocator> vertices;
  std::vector<std::pair<std::pair<int, int>, Sophus::SE3d>, sophus_allocator>
      edges;

  std::ifstream fin("../test.g2o");
  std::string data_type;
  while (fin.good()) {
    fin >> data_type;
    if (data_type == "VERTEX_SE3:QUAT") {
      int id;
      fin >> id;
      vertices.emplace_back(ReadVertex(&fin));
    } else if (data_type == "EDGE_SE3:QUAT") {
      int i, j;
      fin >> i >> j;
      edges.emplace_back(std::pair<int, int>(i, j), ReadEdge(&fin));
    }

    fin >> std::ws;
  }

  std::vector<double> v_s;
  std::vector<double> v_gamma;

  ceres::Problem problem;

  for (auto e: edges) {
    auto ij = e.first;
    auto i = ij.first;
    auto j = ij.second;
    auto &pose_i = vertices.at(i);
    auto &pose_j = vertices.at(j);

    auto edge = e.second;
    if (i + 1 == j) {
      //odometry
      ceres::CostFunction *cost_function = OdometryFunctor::Creat(edge);
      problem.AddResidualBlock(cost_function,
                               NULL, pose_i.data(), pose_j.data());
    } else {
      //loop closure
      double s = 1;
      v_s.emplace_back(s);
      ceres::CostFunction *cost_function = LoopClosureFunctor::Creat(edge);
      problem.AddResidualBlock(cost_function,
                               new ceres::HuberLoss(0.1), pose_i.data(), pose_j.data(), &v_s.back());

      double gamma = 1;
      v_gamma.emplace_back(gamma);
      ceres::CostFunction
          *cost_function1 = PriorFunctor::Create(v_gamma.back());
      problem.AddResidualBlock(cost_function1,NULL,&v_s.back());

      //no robust
      problem.SetParameterBlockConstant(&v_s.back());

      problem.SetParameterLowerBound(&v_s.back(),0,0);
      problem.SetParameterUpperBound(&v_s.back(),0,1);
    }
  }
  for (auto &i: vertices) {
    problem.SetParameterization(i.data(), new LocalParameterizationSE3);
  }

  problem.SetParameterBlockConstant(vertices.at(0).data());

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  for (auto i: vertices) {
    std::cout << i.matrix() << "\n" << std::endl;
  }

  for (auto s: v_s) {
    std::cout << s << std::endl;
  }

  return 0;
}
