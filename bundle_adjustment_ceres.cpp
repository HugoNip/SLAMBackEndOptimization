#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

std::string file_name = "../data/problem-16-22106-pre.txt";

void SolveBA(BALProblem &bal_problem);

int main (int argc, char** argv) {

    BALProblem bal_problem(file_name);
    std::cout << "done 1" << std::endl;
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("../results/initial_ceres.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("../results/final_ceres.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    /**
     * Observations is 2 * num_observations long array observations
     * [u_1, u_2, ..., u_n], where each u_i is two dimensional, the x
     * and y position of the observation.
     */
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.number_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // step 1: define parameter blocks (P137)
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        /**
         * step 2: define cost function (residual block computing method, P137)
         *
         * Each Residual block (Cost function) takes a point and a camera as input
         * and outputs a 2-dimensional Residual
         *
         * AutoDiffCostFunction<>()
         */
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0],
                                                         observations[2 * i + 1]);

        /**
         * step 3: define loss function (kernel function, P137 -> details in P251)
         * If enabled use Huber's loss function
         */
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        /**
         * step 4: add residual block to the problems (P137)
         *
         * Each observation corresponds to a pair of a camera and a point
         * which are identidied by camera_index()[i] and point_index[i] respectively
         */
        problem.AddResidualBlock(cost_function,
                                 loss_function,
                                 camera, point // estimated parameters
                                 );
    }

    // show some information here
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " camera and "
              << bal_problem.number_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.number_observations() << " observations. " << std::endl;

    // configure solver
    std::cout << "Solving ceres BA ... " << std::endl;
    ceres::Solver::Options options; // many options
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // how to solve H * dx = g
    options.minimizer_progress_to_stdout = true; // output to cout
    ceres::Solver::Summary summary; // optimization information
    ceres::Solve(options, &problem, &summary); // start optimization
    std::cout << summary.FullReport() << "\n"; // output result
}