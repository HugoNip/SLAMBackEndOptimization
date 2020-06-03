#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <sophus/se3.hpp>
#include "common.h"

// camera pose and intrinsics
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    // set from given address
    explicit PoseAndIntrinsics(double *data_addr) {
        // R = exp(phi^)
        rotation = Sophus::SO3d::exp(
                Eigen::Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // save estimated data
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

// set vertex of camera pose and intrinsics
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    // update
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = Sophus::SO3d::exp(
                Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    // reprojection
    Eigen::Vector2d project(const Eigen::Vector3d &point) {
        // p_c = Rp + t
        Eigen::Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2]; // normalize, [X/Z, Y/Z, 1]
        // undistort
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Eigen::Vector2d(_estimate.focal * distortion * pc[0], // undistorted u
                               _estimate.focal * distortion * pc[1]); // undistorted v
    }

    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d(0, 0, 0);
    }

    // update
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}
};

class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}
};

void SolveBA(BALProblem &bal_problem);

std::string filename = "../data/problem-16-22106-pre.txt";

int main(int argc, char** argv) {
    BALProblem bal_problem(filename);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("../results/initial_g2o.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("../results/final_g2o.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // construct graph optimizatioin
    // set g2o
    // 9d virables, and 3d error
    // pose is 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    // gradient descent, use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph model
    optimizer.setAlgorithm(solver); // set solver
    optimizer.setVerbose(true); // open debug

    // build g2o problems
    const double *observations = bal_problem.observations();
    // vertex
    std::vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    std::vector<VertexPoint *> vertex_points;
    // many cameras, so use for() {}
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Eigen::Vector3d(point[0], point[1], point[2]));
        // set Margin manually
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Eigen::Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }

    for (int j = 0; j < bal_problem.num_points(); ++j) {
        double *point = points + point_block_size * j;
        auto vertex = vertex_points[j];
        for (int k = 0; k < 3; ++k)
            point[k] = vertex->estimate()[k];
    }
}